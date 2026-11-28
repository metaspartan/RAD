#include "rad_batch_common.cuh"


extern "C" __global__ void gemm_f32_k(const float* __restrict__ W,
                                      const float* __restrict__ X,
                                      float* __restrict__ Y,
                                      int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= n) return;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if ((k & 3) == 0) {
        int k4 = k >> 2;
        const float4* w4 = reinterpret_cast<const float4*>(W + (size_t)row * k);
        for (int j = lane; j < k4; j += 32) {
            float4 wv = w4[j];
            for (int m = 0; m < n_t; ++m) {
                float4 xv = reinterpret_cast<const float4*>(X + (size_t)(t0 + m) * k)[j];
                acc[m] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[m]))));
            }
        }
    } else {
        const float* wrow = W + (size_t)row * k;
        for (int j = lane; j < k; j += 32) {
            float wv = wrow[j];
            for (int m = 0; m < n_t; ++m) {
                acc[m] = fmaf(wv, X[(size_t)(t0 + m) * k + j], acc[m]);
            }
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


extern "C" __global__ void mix_batch_k(const float* __restrict__ blocks,
                                       int n_blocks, int max_B,
                                       const float* __restrict__ cur,
                                       const float* __restrict__ query,
                                       const float* __restrict__ w0,
                                       const float* __restrict__ w1,
                                       const float* __restrict__ w2,
                                       float* __restrict__ out,
                                       int n_embd, int n_queries, float eps) {
    int q = blockIdx.x;
    int t = blockIdx.y;
    const float* qv = query + (size_t)q * n_embd;
    const float* cur_t = cur + (size_t)t * n_embd;
    float* ov = out + ((size_t)q * max_B + t) * n_embd;
    const float* nw = (q == 0) ? w0 : ((q == 1) ? w1 : w2);
    int n_states = n_blocks + 1;
    __shared__ float red_ss[8];
    __shared__ float red_dot[8];
    __shared__ float invs[16];
    __shared__ float scores[16];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int nwarps = blockDim.x >> 5;
    float inv_sqrt_d = rsqrtf((float)n_embd);

    if ((n_embd & 3) == 0 && n_embd <= 3072 && blockDim.x == 256) {
        int e4 = n_embd >> 2;
        const float4* q4 = reinterpret_cast<const float4*>(qv);
        float4 qreg[3];
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            qreg[p] = (j < e4) ? q4[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        for (int s = 0; s < n_states; ++s) {
            const float4* sv4 = reinterpret_cast<const float4*>((s < n_blocks) ? (blocks + ((size_t)s * max_B + t) * n_embd) : cur_t);
            float ss = 0.0f;
            float dot = 0.0f;
#pragma unroll
            for (int p = 0; p < 3; ++p) {
                int j = tid + p * 256;
                if (j < e4) {
                    float4 v = sv4[j];
                    float4 qr = qreg[p];
                    ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
                    dot = fmaf(v.x, qr.x, fmaf(v.y, qr.y, fmaf(v.z, qr.z, fmaf(v.w, qr.w, dot))));
                }
            }
            ss = b_warp_sum32(ss);
            dot = b_warp_sum32(dot);
            if (lane == 0) {
                red_ss[wid] = ss;
                red_dot[wid] = dot;
            }
            __syncthreads();
            if (tid == 0) {
                float tss = 0.0f, tdot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) { tss += red_ss[i]; tdot += red_dot[i]; }
                float inv = rsqrtf(tss / (float)n_embd + 1e-6f);
                invs[s] = inv;
                scores[s] = tdot * inv * inv_sqrt_d;
            }
            __syncthreads();
        }
        if (tid == 0) {
            float mx = scores[0];
            for (int s = 1; s < n_states; ++s) mx = fmaxf(mx, scores[s]);
            float sum = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                scores[s] = expf(scores[s] - mx);
                sum += scores[s];
            }
            float inv_sum = 1.0f / sum;
            for (int s = 0; s < n_states; ++s) {
                scores[s] *= inv_sum * invs[s];
            }
        }
        __syncthreads();

        float4 acc[3];
        float out_ss = 0.0f;
        const float4* cur4 = reinterpret_cast<const float4*>(cur_t);
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            if (j < e4) {
                float sc_c = scores[n_blocks];
                float4 cv = cur4[j];
                float ax = sc_c * cv.x, ay = sc_c * cv.y, az = sc_c * cv.z, aw = sc_c * cv.w;
                for (int s = 0; s < n_blocks; ++s) {
                    float sc_s = scores[s];
                    float4 bv = reinterpret_cast<const float4*>(blocks + ((size_t)s * max_B + t) * n_embd)[j];
                    ax = fmaf(sc_s, bv.x, ax);
                    ay = fmaf(sc_s, bv.y, ay);
                    az = fmaf(sc_s, bv.z, az);
                    aw = fmaf(sc_s, bv.w, aw);
                }
                acc[p] = make_float4(ax, ay, az, aw);
                out_ss = fmaf(ax, ax, fmaf(ay, ay, fmaf(az, az, fmaf(aw, aw, out_ss))));
            }
        }
        float4* ov4 = reinterpret_cast<float4*>(ov);
        if (nw) {
            out_ss = b_warp_sum32(out_ss);
            if (lane == 0) red_ss[wid] = out_ss;
            __syncthreads();
            __shared__ float inv_out;
            if (tid == 0) {
                float tot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) tot += red_ss[i];
                inv_out = rsqrtf(tot / (float)n_embd + eps);
            }
            __syncthreads();
            float io = inv_out;
            const float4* nw4 = reinterpret_cast<const float4*>(nw);
#pragma unroll
            for (int p = 0; p < 3; ++p) {
                int j = tid + p * 256;
                if (j < e4) {
                    float4 a = acc[p];
                    float4 w = nw4[j];
                    ov4[j] = make_float4(a.x * io * w.x, a.y * io * w.y, a.z * io * w.z, a.w * io * w.w);
                }
            }
        } else {
#pragma unroll
            for (int p = 0; p < 3; ++p) {
                int j = tid + p * 256;
                if (j < e4) ov4[j] = acc[p];
            }
        }
        return;
    }

    for (int s = 0; s < n_states; ++s) {
        const float* sv = (s < n_blocks) ? (blocks + ((size_t)s * max_B + t) * n_embd) : cur_t;
        float ss = 0.0f;
        float dot = 0.0f;
        for (int j = tid; j < n_embd; j += blockDim.x) {
            float v = sv[j];
            ss += v * v;
            dot += v * qv[j];
        }
        ss = b_warp_sum32(ss);
        dot = b_warp_sum32(dot);
        if (lane == 0) { red_ss[wid] = ss; red_dot[wid] = dot; }
        __syncthreads();
        if (tid == 0) {
            float tss = 0.0f, tdot = 0.0f;
            for (int i = 0; i < nwarps; ++i) { tss += red_ss[i]; tdot += red_dot[i]; }
            invs[s] = rsqrtf(tss / (float)n_embd + 1e-6f);
            scores[s] = tdot * invs[s] * inv_sqrt_d;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = scores[0];
        for (int s = 1; s < n_states; ++s) mx = fmaxf(mx, scores[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            scores[s] = expf(scores[s] - mx);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        for (int s = 0; s < n_states; ++s) {
            scores[s] *= inv_sum * invs[s];
        }
    }
    __syncthreads();

    float out_ss = 0.0f;
    for (int j = tid; j < n_embd; j += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_blocks; ++s) {
            acc += scores[s] * blocks[((size_t)s * max_B + t) * n_embd + j];
        }
        acc += scores[n_blocks] * cur_t[j];
        ov[j] = acc;
        out_ss += acc * acc;
    }
    if (nw) {
        out_ss = b_warp_sum32(out_ss);
        if (lane == 0) red_ss[wid] = out_ss;
        __syncthreads();
        __shared__ float inv_out;
        if (tid == 0) {
            float tot = 0.0f;
            for (int i = 0; i < nwarps; ++i) tot += red_ss[i];
            inv_out = rsqrtf(tot / (float)n_embd + eps);
        }
        __syncthreads();
        float io = inv_out;
        for (int j = tid; j < n_embd; j += blockDim.x) {
            ov[j] = ov[j] * io * nw[j];
        }
    }
}


extern "C" __global__ void loop_ctrl_down_b_k(const float* __restrict__ cur0,
                                              const float* __restrict__ cur1,
                                              const float* __restrict__ cur2,
                                              const float* __restrict__ msgs,
                                              const float* __restrict__ wd0,
                                              const float* __restrict__ wd1,
                                              const float* __restrict__ wd2,
                                              float* __restrict__ h_out,
                                              int n_embd, int rank, int max_B) {
    int r = blockIdx.x;
    int c = blockIdx.y;
    int t = blockIdx.z;
    const float* cur = ((c == 0) ? cur0 : ((c == 1) ? cur1 : cur2)) + (size_t)t * n_embd;
    const float* msg = msgs + ((size_t)c * max_B + t) * n_embd;
    const float* wd = (c == 0) ? wd0 : ((c == 1) ? wd1 : wd2);
    __shared__ float red_c[8];
    __shared__ float red_m[8];
    __shared__ float rc_s, rm_s;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int nw = blockDim.x >> 5;

    if ((n_embd & 3) == 0 && n_embd <= 3072 && blockDim.x == 256) {
        int e4 = n_embd >> 2;
        const float4* cur4 = reinterpret_cast<const float4*>(cur);
        const float4* msg4 = reinterpret_cast<const float4*>(msg);
        float4 cr[3], mr[3];
        float sc = 0.0f, sm = 0.0f;
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            if (j < e4) {
                float4 cv = cur4[j];
                float4 mv = msg4[j];
                cr[p] = cv;
                mr[p] = mv;
                sc = fmaf(cv.x, cv.x, fmaf(cv.y, cv.y, fmaf(cv.z, cv.z, fmaf(cv.w, cv.w, sc))));
                sm = fmaf(mv.x, mv.x, fmaf(mv.y, mv.y, fmaf(mv.z, mv.z, fmaf(mv.w, mv.w, sm))));
            }
        }
        sc = b_warp_sum32(sc);
        sm = b_warp_sum32(sm);
        if (lane == 0) {
            red_c[wid] = sc;
            red_m[wid] = sm;
        }
        __syncthreads();
        if (tid == 0) {
            float tc = 0.0f, tm = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                tc += red_c[i];
                tm += red_m[i];
            }
            rc_s = rsqrtf(tc / (float)n_embd + 1e-6f);
            rm_s = rsqrtf(tm / (float)n_embd + 1e-6f);
        }
        __syncthreads();

        const float4* wr4 = reinterpret_cast<const float4*>(wd + (size_t)r * (2 * n_embd));
        float rc = rc_s;
        float rm = rm_s;
        float acc = 0.0f;
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            if (j < e4) {
                float4 wc = wr4[j];
                float4 wm = wr4[e4 + j];
                float4 cv = cr[p];
                float4 mv = mr[p];
                acc = fmaf(wc.x, cv.x * rc, fmaf(wc.y, cv.y * rc, fmaf(wc.z, cv.z * rc, fmaf(wc.w, cv.w * rc, acc))));
                acc = fmaf(wm.x, mv.x * rm, fmaf(wm.y, mv.y * rm, fmaf(wm.z, mv.z * rm, fmaf(wm.w, mv.w * rm, acc))));
            }
        }
        acc = b_warp_sum32(acc);
        if (lane == 0) red_c[wid] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) tot += red_c[i];
            h_out[((size_t)c * max_B + t) * rank + r] = tot;
        }
        return;
    }

    float sc = 0.0f, sm = 0.0f;
    for (int j = tid; j < n_embd; j += blockDim.x) {
        float vc = cur[j];
        float vm = msg[j];
        sc += vc * vc;
        sm += vm * vm;
    }
    sc = b_warp_sum32(sc);
    sm = b_warp_sum32(sm);
    if (lane == 0) {
        red_c[wid] = sc;
        red_m[wid] = sm;
    }
    __syncthreads();
    if (tid == 0) {
        float tc = 0.0f, tm = 0.0f;
        for (int i = 0; i < nw; ++i) {
            tc += red_c[i];
            tm += red_m[i];
        }
        rc_s = rsqrtf(tc / (float)n_embd + 1e-6f);
        rm_s = rsqrtf(tm / (float)n_embd + 1e-6f);
    }
    __syncthreads();

    const float* wr = wd + (size_t)r * (2 * n_embd);
    float rc = rc_s;
    float rm = rm_s;
    float acc = 0.0f;
    for (int j = tid; j < n_embd; j += blockDim.x) {
        acc += wr[j] * (cur[j] * rc) + wr[n_embd + j] * (msg[j] * rm);
    }
    acc = b_warp_sum32(acc);
    if (lane == 0) red_c[wid] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nw; ++i) tot += red_c[i];
        h_out[((size_t)c * max_B + t) * rank + r] = tot;
    }
}


__device__ float g_b_softcap_max_v[128 * 32];

__device__ int g_b_softcap_max_i[128 * 32];


extern "C" __global__ void softcap_argmax_batch_p1_k(float* __restrict__ logits,
                                                     float cap, int vocab_padded, int n_unpadded) {
    (void)cap;
    int sp = blockIdx.x;
    int bt = blockIdx.y;
    int tid = threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const float* lg = logits + (size_t)bt * vocab_padded;
    const float4* lg4 = reinterpret_cast<const float4*>(lg);
    int n4 = n_unpadded >> 2;
    float bv = -1e30f;
    int bi = 0;
    for (int i4 = sp * blockDim.x + tid; i4 < n4; i4 += stride) {
        float4 v = lg4[i4];
        int base = i4 << 2;
        if (v.x > bv) { bv = v.x; bi = base; }
        if (v.y > bv) { bv = v.y; bi = base + 1; }
        if (v.z > bv) { bv = v.z; bi = base + 2; }
        if (v.w > bv) { bv = v.w; bi = base + 3; }
    }
    if (sp == 0) {
        for (int i = (n4 << 2) + tid; i < n_unpadded; i += blockDim.x) {
            float v = lg[i];
            if (v > bv) { bv = v; bi = i; }
        }
    }
    __shared__ float rv[8];
    __shared__ int ri[8];
    int lane = tid & 31;
    int wid = tid >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(bv, off, 32);
        int oi = __shfl_down(bi, off, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { rv[wid] = bv; ri[wid] = bi; }
    __syncthreads();
    if (tid == 0) {
        float m = rv[0];
        int mi = ri[0];
        for (int i = 1; i < 8; ++i) {
            if (rv[i] > m || (rv[i] == m && ri[i] < mi)) { m = rv[i]; mi = ri[i]; }
        }
        g_b_softcap_max_v[bt * 32 + sp] = m;
        g_b_softcap_max_i[bt * 32 + sp] = mi;
    }
}


extern "C" __global__ void softcap_argmax_batch_p2_k(int* __restrict__ out_toks) {
    int bt = blockIdx.x;
    int lane = threadIdx.x;
    float v = g_b_softcap_max_v[bt * 32 + lane];
    int idx = g_b_softcap_max_i[bt * 32 + lane];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(v, off, 32);
        int oi = __shfl_down(idx, off, 32);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (lane == 0) out_toks[bt] = idx;
}


extern "C" __global__ void softcap_argmax_batch_k(float* __restrict__ logits,
                                                  float cap, int vocab_padded, int n_unpadded,
                                                  int* __restrict__ out_toks) {
    (void)cap;
    int bt = blockIdx.x;
    const float* lg = logits + (size_t)bt * vocab_padded;
    const float4* lg4 = reinterpret_cast<const float4*>(lg);
    int n4 = n_unpadded >> 2;
    float bv = -1e30f;
    int bi = 0;
    for (int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
        float4 v = lg4[i4];
        int base = i4 << 2;
        if (v.x > bv) { bv = v.x; bi = base; }
        if (v.y > bv) { bv = v.y; bi = base + 1; }
        if (v.z > bv) { bv = v.z; bi = base + 2; }
        if (v.w > bv) { bv = v.w; bi = base + 3; }
    }
    for (int i = (n4 << 2) + threadIdx.x; i < n_unpadded; i += blockDim.x) {
        float v = lg[i];
        if (v > bv) { bv = v; bi = i; }
    }
    __shared__ float rv[8];
    __shared__ int ri[8];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(bv, off, 32);
        int oi = __shfl_down(bi, off, 32);
        if (ov > bv) { bv = ov; bi = oi; }
    }
    if (lane == 0) { rv[wid] = bv; ri[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = rv[0];
        int mi = ri[0];
        int nw = blockDim.x >> 5;
        for (int i = 1; i < nw; ++i) {
            if (rv[i] > m) { m = rv[i]; mi = ri[i]; }
        }
        out_toks[bt] = mi;
    }
}

template<int NT>
__global__ void gemm_f32_f4_nt_k(const float* __restrict__ W,
                                 const float* __restrict__ X,
                                 float* __restrict__ Y,
                                 int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 2 + warp;
    int t0 = blockIdx.y * NT;
    if (row >= n || t0 >= B) return;
    int k4 = k >> 2;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t)row * k);
    float acc[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
    for (int j = lane; j < k4; j += 32) {
        float4 wv = wr[j];
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            const float4* xr = reinterpret_cast<const float4*>(X + (size_t)(t0 + m) * k);
            float4 xv = xr[j];
            acc[m] = fmaf(wv.x, xv.x, acc[m]);
            acc[m] = fmaf(wv.y, xv.y, acc[m]);
            acc[m] = fmaf(wv.z, xv.z, acc[m]);
            acc[m] = fmaf(wv.w, xv.w, acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


__global__ void gemm_f32_f4_rem_k(const float* __restrict__ W,
                                  const float* __restrict__ X,
                                  float* __restrict__ Y,
                                  int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 2 + warp;
    int t0 = blockIdx.y * 8;
    int nt = min(8, B - t0);
    if (row >= n || nt <= 0) return;
    int k4 = k >> 2;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t)row * k);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = lane; j < k4; j += 32) {
        float4 wv = wr[j];
        for (int m = 0; m < nt; ++m) {
            const float4* xr = reinterpret_cast<const float4*>(X + (size_t)(t0 + m) * k);
            float4 xv = xr[j];
            acc[m] = fmaf(wv.x, xv.x, acc[m]);
            acc[m] = fmaf(wv.y, xv.y, acc[m]);
            acc[m] = fmaf(wv.z, xv.z, acc[m]);
            acc[m] = fmaf(wv.w, xv.w, acc[m]);
        }
    }
    for (int m = 0; m < nt; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


extern "C" int ink_gemm_f32_wmma(hipStream_t s, const float* W, const float* X, float* Y, int n, int k, int B);

extern "C" int ink_gemm_f32(hipStream_t s, const float* W, const float* X, float* Y, int n, int k, int B) {
    if (B >= 32) return ink_gemm_f32_wmma(s, W, X, Y, n, k, B);
    if ((k & 3) == 0) {
        if (B == 8 || B == 16 || B == 24) {
            gemm_f32_f4_nt_k<8><<<dim3((n + 1) / 2, B / 8), dim3(64), 0, s>>>(W, X, Y, n, k, B);
            return 0;
        }
        if (B == 6) {
            gemm_f32_f4_nt_k<6><<<dim3((n + 1) / 2, 1), dim3(64), 0, s>>>(W, X, Y, n, k, B);
            return 0;
        }
        if (B == 4) {
            gemm_f32_f4_nt_k<4><<<dim3((n + 1) / 2, 1), dim3(64), 0, s>>>(W, X, Y, n, k, B);
            return 0;
        }
        if (B == 3) {
            gemm_f32_f4_nt_k<3><<<dim3((n + 1) / 2, 1), dim3(64), 0, s>>>(W, X, Y, n, k, B);
            return 0;
        }
        if (B == 2) {
            gemm_f32_f4_nt_k<2><<<dim3((n + 1) / 2, 1), dim3(64), 0, s>>>(W, X, Y, n, k, B);
            return 0;
        }
        gemm_f32_f4_rem_k<<<dim3((n + 1) / 2, (B + 7) / 8), dim3(64), 0, s>>>(W, X, Y, n, k, B);
        return 0;
    }
    gemm_f32_k<<<dim3((n + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(W, X, Y, n, k, B);
    return 0;
}

template<int NQ>
__global__ void mix_batch_f4_k(const float* __restrict__ blocks,
                               int n_blocks, int max_B,
                               const float* __restrict__ cur,
                               const float* __restrict__ query,
                               const float* __restrict__ w0,
                               const float* __restrict__ w1,
                               const float* __restrict__ w2,
                               float* __restrict__ out,
                               int n_embd, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int e4 = n_embd >> 2;
    int n_states = n_blocks + 1;
    const float4* cur4 = reinterpret_cast<const float4*>(cur + (size_t)t * n_embd);

    __shared__ float red_ss[16][8];
    __shared__ float red_dot[NQ][16][8];
    __shared__ float scores[NQ][16];
    __shared__ float red_out[NQ][8];
    __shared__ float inv_out[NQ];

    for (int s = 0; s < n_states; ++s) {
        const float4* sv4 = (s < n_blocks)
            ? reinterpret_cast<const float4*>(blocks + ((size_t)s * max_B + t) * n_embd)
            : cur4;
        float ss = 0.0f;
        float dot[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) dot[q] = 0.0f;
        for (int j = tid; j < e4; j += 256) {
            float4 v = sv4[j];
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float4 qv = reinterpret_cast<const float4*>(query + (size_t)q * n_embd)[j];
                dot[q] = fmaf(v.x, qv.x, fmaf(v.y, qv.y, fmaf(v.z, qv.z, fmaf(v.w, qv.w, dot[q]))));
            }
        }
        ss = b_warp_sum32(ss);
        if (lane == 0) red_ss[s][wid] = ss;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float dq = b_warp_sum32(dot[q]);
            if (lane == 0) red_dot[q][s][wid] = dq;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float inv_sqrt_d = rsqrtf((float)n_embd);
        float invs[16];
        for (int s = 0; s < n_states; ++s) {
            float tot_ss = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot_ss += red_ss[s][w];
            invs[s] = rsqrtf(tot_ss / (float)n_embd + 1e-6f);
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float mx = -1e30f;
            for (int s = 0; s < n_states; ++s) {
                float tot_d = 0.0f;
#pragma unroll
                for (int w = 0; w < 8; ++w) tot_d += red_dot[q][s][w];
                float sc = tot_d * invs[s] * inv_sqrt_d;
                scores[q][s] = sc;
                mx = fmaxf(mx, sc);
            }
            float sum = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                float e = expf(scores[q][s] - mx);
                scores[q][s] = e;
                sum += e;
            }
            float inv_sum = 1.0f / sum;
            for (int s = 0; s < n_states; ++s) {
                scores[q][s] *= inv_sum * invs[s];
            }
        }
    }
    __syncthreads();

    float4 acc[NQ][3];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
#pragma unroll
        for (int p = 0; p < 3; ++p) acc[q][p] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int s = 0; s < n_states; ++s) {
        const float4* sv4 = (s < n_blocks)
            ? reinterpret_cast<const float4*>(blocks + ((size_t)s * max_B + t) * n_embd)
            : cur4;
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            if (j < e4) {
                float4 v = sv4[j];
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float sc = scores[q][s];
                    acc[q][p].x = fmaf(sc, v.x, acc[q][p].x);
                    acc[q][p].y = fmaf(sc, v.y, acc[q][p].y);
                    acc[q][p].z = fmaf(sc, v.z, acc[q][p].z);
                    acc[q][p].w = fmaf(sc, v.w, acc[q][p].w);
                }
            }
        }
    }

#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float oss = 0.0f;
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            if (tid + p * 256 < e4) {
                float4 a = acc[q][p];
                oss = fmaf(a.x, a.x, fmaf(a.y, a.y, fmaf(a.z, a.z, fmaf(a.w, a.w, oss))));
            }
        }
        oss = b_warp_sum32(oss);
        if (lane == 0) red_out[q][wid] = oss;
    }
    __syncthreads();

    if (tid == 0) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot += red_out[q][w];
            inv_out[q] = rsqrtf(tot / (float)n_embd + eps);
        }
    }
    __syncthreads();

#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        const float* nw = (q == 0) ? w0 : ((q == 1) ? w1 : w2);
        float4* ov4 = reinterpret_cast<float4*>(out + ((size_t)q * max_B + t) * n_embd);
        if (nw) {
            const float4* nw4 = reinterpret_cast<const float4*>(nw);
            float io = inv_out[q];
#pragma unroll
            for (int p = 0; p < 3; ++p) {
                int j = tid + p * 256;
                if (j < e4) {
                    float4 a = acc[q][p];
                    float4 w = nw4[j];
                    ov4[j] = make_float4(a.x * io * w.x, a.y * io * w.y, a.z * io * w.z, a.w * io * w.w);
                }
            }
        } else {
#pragma unroll
            for (int p = 0; p < 3; ++p) {
                int j = tid + p * 256;
                if (j < e4) ov4[j] = acc[q][p];
            }
        }
    }
}


extern "C" int ink_mix_batch(hipStream_t s, const float* blocks, int n_blocks, int max_B, const float* cur, const float* query, const float* w0, const float* w1, const float* w2, float* out, int n_embd, int n_queries, int B, float eps) {
    if ((n_embd & 3) == 0 && n_embd <= 3072 && n_blocks < 16) {
        if (n_queries == 3) {
            mix_batch_f4_k<3><<<dim3(B), dim3(256), 0, s>>>(blocks, n_blocks, max_B, cur, query, w0, w1, w2, out, n_embd, eps);
            return 0;
        }
        if (n_queries == 1) {
            mix_batch_f4_k<1><<<dim3(B), dim3(256), 0, s>>>(blocks, n_blocks, max_B, cur, query, w0, w1, w2, out, n_embd, eps);
            return 0;
        }
    }
    mix_batch_k<<<dim3(n_queries, B), dim3(256), 0, s>>>(blocks, n_blocks, max_B, cur, query, w0, w1, w2, out, n_embd, n_queries, eps);
    return 0;
}

extern "C" __global__ void loop_ctrl_up_fast_b_k(const float* __restrict__ cur0,
                                                 const float* __restrict__ cur1,
                                                 const float* __restrict__ cur2,
                                                 const float* __restrict__ msgs,
                                                 const float* __restrict__ h_in,
                                                 const float* __restrict__ wg0,
                                                 const float* __restrict__ wg1,
                                                 const float* __restrict__ wg2,
                                                 const float* __restrict__ wf0,
                                                 const float* __restrict__ wf1,
                                                 const float* __restrict__ wf2,
                                                 float* __restrict__ out0,
                                                 float* __restrict__ out1,
                                                 float* __restrict__ out2,
                                                 int n_embd, int rank, int max_B, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int i = blockIdx.x * 8 + warp;
    int c = blockIdx.y;
    int t0 = blockIdx.z * 8;
    int nt = min(8, B - t0);
    if (i >= n_embd || nt <= 0) return;
    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* h_base = h_in + (size_t)c * max_B * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    float* out_base = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const float* wgi = wg + (size_t)i * rank;
    const float* wfi = wf + (size_t)i * rank;
    float g[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int r = lane; r < rank; r += 32) {
        float gv = wgi[r];
        float fv = wfi[r];
        for (int m = 0; m < nt; ++m) {
            float hr = h_base[(size_t)(t0 + m) * rank + r];
            g[m] = fmaf(gv, hr, g[m]);
            f[m] = fmaf(fv, hr, f[m]);
        }
    }
    for (int m = 0; m < nt; ++m) {
        float gm = b_warp_sum32(g[m]);
        float fm = b_warp_sum32(f[m]);
        if (lane == 0) {
            size_t idx = (size_t)(t0 + m) * n_embd + i;
            float cv = cur_base[idx];
            out_base[idx] = cv - tanhf(fm) * cv + tanhf(gm) * msg_base[idx];
        }
    }
}


extern "C" __global__ void rmsnorm_multi_inplace_b_k(float* __restrict__ out0,
                                                     float* __restrict__ out1,
                                                     float* __restrict__ out2,
                                                     const float* __restrict__ nw0,
                                                     const float* __restrict__ nw1,
                                                     const float* __restrict__ nw2,
                                                     int n_embd, float eps) {
    int c = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    float* out = ((c == 0) ? out0 : ((c == 1) ? out1 : out2)) + (size_t)t * n_embd;
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    __shared__ float red[8];
    __shared__ float inv;
    if ((n_embd & 3) == 0 && n_embd <= 3072) {
        int e4 = n_embd >> 2;
        float4* o4 = reinterpret_cast<float4*>(out);
        const float4* w4 = reinterpret_cast<const float4*>(nw);
        float4 reg[3];
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            if (j < e4) {
                float4 v = o4[j];
                reg[p] = v;
                ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
            }
        }
        ss = b_warp_sum32(ss);
        if (lane == 0) red[wid] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) tot += red[i];
            inv = rsqrtf(tot / (float)n_embd + eps);
        }
        __syncthreads();
        float io = inv;
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            if (j < e4) {
                float4 v = reg[p];
                float4 wv = w4[j];
                o4[j] = make_float4(v.x * io * wv.x, v.y * io * wv.y, v.z * io * wv.z, v.w * io * wv.w);
            }
        }
        return;
    }
    float ss = 0.0f;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float v = out[i];
        ss += v * v;
    }
    ss = b_warp_sum32(ss);
    int nwarps = blockDim.x >> 5;
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nwarps; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        out[i] = out[i] * io * nw[i];
    }
}


extern "C" int ink_gemm_f32_wmma(hipStream_t s, const float* W, const float* X, float* Y, int n, int k, int B);

extern "C" int ink_loop_ctrl_down_wmma(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* wd0, const float* wd1, const float* wd2, float* h_buf, int n_embd, int rank, int max_B, int B, int n_ctrl);

extern "C" int ink_loop_ctrl_up_wmma(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* h_buf, const float* wg0, const float* wg1, const float* wg2, const float* wf0, const float* wf1, const float* wf2, float* out0, float* out1, float* out2, int n_embd, int rank, int max_B, int B, int n_ctrl);


template<int NT>
__global__ void loop_ctrl_down_nt_k(const float* __restrict__ cur0,
                                    const float* __restrict__ cur1,
                                    const float* __restrict__ cur2,
                                    const float* __restrict__ msgs,
                                    const float* __restrict__ wd0,
                                    const float* __restrict__ wd1,
                                    const float* __restrict__ wd2,
                                    float* __restrict__ h_out,
                                    int n_embd, int rank, int max_B) {
    int r = blockIdx.x;
    int c = blockIdx.y;
    int t0 = blockIdx.z * NT;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* wd = (c == 0) ? wd0 : ((c == 1) ? wd1 : wd2);

    __shared__ float rc_s[NT];
    __shared__ float rm_s[NT];
    for (int m = wid; m < NT; m += 8) {
        const float4* cp = reinterpret_cast<const float4*>(cur_base + (size_t)(t0 + m) * n_embd);
        const float4* mp = reinterpret_cast<const float4*>(msg_base + (size_t)(t0 + m) * n_embd);
        float sc = 0.0f, sm = 0.0f;
        int e4 = n_embd >> 2;
        for (int j = lane; j < e4; j += 32) {
            float4 vc = cp[j], vm = mp[j];
            sc = fmaf(vc.x, vc.x, fmaf(vc.y, vc.y, fmaf(vc.z, vc.z, fmaf(vc.w, vc.w, sc))));
            sm = fmaf(vm.x, vm.x, fmaf(vm.y, vm.y, fmaf(vm.z, vm.z, fmaf(vm.w, vm.w, sm))));
        }
        sc = b_warp_sum32(sc);
        sm = b_warp_sum32(sm);
        if (lane == 0) {
            rc_s[m] = rsqrtf(sc / (float)n_embd + 1e-6f);
            rm_s[m] = rsqrtf(sm / (float)n_embd + 1e-6f);
        }
    }
    __syncthreads();

    const float4* w0 = reinterpret_cast<const float4*>(wd + (size_t)r * (2 * n_embd));
    const float4* w1 = reinterpret_cast<const float4*>(wd + (size_t)r * (2 * n_embd) + n_embd);
    int e4 = n_embd >> 2;
    float acc[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
    for (int j = tid; j < e4; j += 256) {
        float4 wc = w0[j];
        float4 wm = w1[j];
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            const float4* cp = reinterpret_cast<const float4*>(cur_base + (size_t)(t0 + m) * n_embd);
            const float4* mp = reinterpret_cast<const float4*>(msg_base + (size_t)(t0 + m) * n_embd);
            float4 vc = cp[j], vm = mp[j];
            float rc = rc_s[m], rm = rm_s[m];
            float dc = wc.x * vc.x + wc.y * vc.y + wc.z * vc.z + wc.w * vc.w;
            float dm = wm.x * vm.x + wm.y * vm.y + wm.z * vm.z + wm.w * vm.w;
            acc[m] = fmaf(dc, rc, fmaf(dm, rm, acc[m]));
        }
    }
    __shared__ float red[NT][8];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) red[m][wid] = v;
    }
    __syncthreads();
    if (tid < NT) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) tot += red[tid][w];
        h_out[((size_t)c * max_B + (t0 + tid)) * rank + r] = tot;
    }
}


template<int NT>
__global__ void loop_ctrl_up_fast_nt_k(const float* __restrict__ cur0,
                                       const float* __restrict__ cur1,
                                       const float* __restrict__ cur2,
                                       const float* __restrict__ msgs,
                                       const float* __restrict__ h_in,
                                       const float* __restrict__ wg0,
                                       const float* __restrict__ wg1,
                                       const float* __restrict__ wg2,
                                       const float* __restrict__ wf0,
                                       const float* __restrict__ wf1,
                                       const float* __restrict__ wf2,
                                       float* __restrict__ out0,
                                       float* __restrict__ out1,
                                       float* __restrict__ out2,
                                       int n_embd, int rank, int max_B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int i = blockIdx.x * 8 + warp;
    int c = blockIdx.y;
    int t0 = blockIdx.z * NT;
    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* h_base = h_in + (size_t)c * max_B * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    float* out_base = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const float* wgi = wg + (size_t)min(i, n_embd - 1) * rank;
    const float* wfi = wf + (size_t)min(i, n_embd - 1) * rank;
    float g[NT], f[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) { g[m] = 0.0f; f[m] = 0.0f; }
    if (rank <= 128) {
        __shared__ float s_h[NT][128];
        int total_h = NT * rank;
        for (int idx = threadIdx.x; idx < total_h; idx += 256) {
            int m = idx / rank;
            int r = idx - m * rank;
            s_h[m][r] = h_base[(size_t)(t0 + m) * rank + r];
        }
        __syncthreads();
        if (i >= n_embd) return;
        for (int r = lane; r < rank; r += 32) {
            float gv = wgi[r];
            float fv = wfi[r];
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                float hr = s_h[m][r];
                g[m] = fmaf(gv, hr, g[m]);
                f[m] = fmaf(fv, hr, f[m]);
            }
        }
    } else {
        if (i >= n_embd) return;
        for (int r = lane; r < rank; r += 32) {
            float gv = wgi[r];
            float fv = wfi[r];
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                float hr = h_base[(size_t)(t0 + m) * rank + r];
                g[m] = fmaf(gv, hr, g[m]);
                f[m] = fmaf(fv, hr, f[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        float gm = b_warp_sum32(g[m]);
        float fm = b_warp_sum32(f[m]);
        if (lane == 0) {
            size_t idx = (size_t)(t0 + m) * n_embd + i;
            float cv = cur_base[idx];
            out_base[idx] = cv - tanhf(fm) * cv + tanhf(gm) * msg_base[idx];
        }
    }
}


extern "C" int ink_loop_ctrl_3_norm_batch(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* wd0, const float* wd1, const float* wd2, const float* wg0, const float* wg1, const float* wg2, const float* wf0, const float* wf1, const float* wf2, const float* nw0, const float* nw1, const float* nw2, float* h_buf, float* out0, float* out1, float* out2, int n_embd, int rank, int max_B, int B, float eps) {
    if (B >= 12) {
        if (B <= 64 && (B & 7) == 0 && (n_embd & 3) == 0) {
            loop_ctrl_down_nt_k<8><<<dim3(rank, 3, B / 8), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B);
        } else {
            ink_loop_ctrl_down_wmma(s, cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B, B, 3);
        }
        ink_loop_ctrl_up_wmma(s, cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank, max_B, B, 3);
    } else if (B == 8 && (n_embd & 3) == 0) {
        loop_ctrl_down_nt_k<8><<<dim3(rank, 3, 1), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B);
        loop_ctrl_up_fast_nt_k<8><<<dim3((n_embd + 7) / 8, 3, 1), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank, max_B);
    } else if (B == 4 && (n_embd & 3) == 0) {
        loop_ctrl_down_nt_k<4><<<dim3(rank, 3, 1), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B);
        loop_ctrl_up_fast_nt_k<4><<<dim3((n_embd + 7) / 8, 3, 1), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank, max_B);
    } else {
        loop_ctrl_down_b_k<<<dim3(rank, 3, B), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B);
        loop_ctrl_up_fast_b_k<<<dim3((n_embd + 7) / 8, 3, (B + 7) / 8), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank, max_B, B);
    }
    rmsnorm_multi_inplace_b_k<<<dim3(3, B), dim3(256), 0, s>>>(out0, out1, out2, nw0, nw1, nw2, n_embd, eps);
    return 0;
}

extern "C" int ink_loop_ctrl_1_norm_batch(hipStream_t s, const float* cur, const float* msg, const float* wd, const float* wg, const float* wf, const float* nw, float* h_buf, float* out, int n_embd, int rank, int max_B, int B, float eps) {
    if (B >= 12) {
        if (B <= 64 && (B & 7) == 0 && (n_embd & 3) == 0) {
            loop_ctrl_down_nt_k<8><<<dim3(rank, 1, B / 8), dim3(256), 0, s>>>(cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank, max_B);
        } else {
            ink_loop_ctrl_down_wmma(s, cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank, max_B, B, 1);
        }
        ink_loop_ctrl_up_wmma(s, cur, cur, cur, msg, h_buf, wg, wg, wg, wf, wf, wf, out, out, out, n_embd, rank, max_B, B, 1);
    } else if (B == 8 && (n_embd & 3) == 0) {
        loop_ctrl_down_nt_k<8><<<dim3(rank, 1, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank, max_B);
        loop_ctrl_up_fast_nt_k<8><<<dim3((n_embd + 7) / 8, 1, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, h_buf, wg, wg, wg, wf, wf, wf, out, out, out, n_embd, rank, max_B);
    } else if (B == 4 && (n_embd & 3) == 0) {
        loop_ctrl_down_nt_k<4><<<dim3(rank, 1, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank, max_B);
        loop_ctrl_up_fast_nt_k<4><<<dim3((n_embd + 7) / 8, 1, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, h_buf, wg, wg, wg, wf, wf, wf, out, out, out, n_embd, rank, max_B);
    } else {
        loop_ctrl_down_b_k<<<dim3(rank, 1, B), dim3(256), 0, s>>>(cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank, max_B);
        loop_ctrl_up_fast_b_k<<<dim3((n_embd + 7) / 8, 1, (B + 7) / 8), dim3(256), 0, s>>>(cur, cur, cur, msg, h_buf, wg, wg, wg, wf, wf, wf, out, out, out, n_embd, rank, max_B, B);
    }
    rmsnorm_multi_inplace_b_k<<<dim3(1, B), dim3(256), 0, s>>>(out, out, out, nw, nw, nw, n_embd, eps);
    return 0;
}

extern "C" __global__ void moe_gu_nvfp4_grouped_k(const uint8_t* __restrict__ W,
                                                  const float* __restrict__ scales,
                                                  const float* __restrict__ X,
                                                  const int* __restrict__ itop,
                                                  float* __restrict__ act_out,
                                                  int n_ff, int k, int n_used, int B,
                                                  float beta, float lb) {
    int exp_id = blockIdx.y;
    __shared__ int tok_list[64];
    __shared__ int slot_list[64];
    __shared__ int count;
    if (threadIdx.x == 0) {
        int c = 0;
        int total = B * n_used;
        for (int i = 0; i < total; ++i) {
            if (itop[i] == exp_id) {
                tok_list[c] = i / n_used;
                slot_list[c] = i % n_used;
                ++c;
            }
        }
        count = c;
    }
    __syncthreads();
    int cnt = count;
    if (cnt == 0) return;

    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_ff) return;

    int nblk = k >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + row) * row_bytes;
    float sc = scales ? scales[exp_id] : 1.0f;

    for (int t0 = 0; t0 < cnt; t0 += 8) {
        int nt = min(8, cnt - t0);
        float ag[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int b = lane8; b < nblk; b += 8) {
            const uint8_t* bg = wg + (size_t)b * 36;
            const uint8_t* bu = wu + (size_t)b * 36;
            uint32_t dsg, dsu;
            memcpy(&dsg, bg, 4);
            memcpy(&dsu, bu, 4);
            const uint32_t* qg = reinterpret_cast<const uint32_t*>(bg + 4);
            const uint32_t* qu = reinterpret_cast<const uint32_t*>(bu + 4);
            int xoff = b << 6;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                float dg = b_ue4m3_to_fp32((uint8_t)((dsg >> (sb * 8)) & 0xFF));
                float du = b_ue4m3_to_fp32((uint8_t)((dsu >> (sb * 8)) & 0xFF));
                uint32_t qg0 = qg[sb * 2], qg1 = qg[sb * 2 + 1];
                uint32_t qu0 = qu[sb * 2], qu1 = qu[sb * 2 + 1];
                int xo = xoff + sb * 16;
                for (int m = 0; m < nt; ++m) {
                    const float4* x4 = reinterpret_cast<const float4*>(X + (size_t)tok_list[t0 + m] * k + xo);
                    float4 x0 = x4[0], x1 = x4[1], x2 = x4[2], x3 = x4[3];
                    float sg = 0.f, su = 0.f;
                    sg = fmaf(b_fp4_nibble(qg0), x0.x, sg); sg = fmaf(b_fp4_nibble(qg0 >> 4), x2.x, sg);
                    sg = fmaf(b_fp4_nibble(qg0 >> 8), x0.y, sg); sg = fmaf(b_fp4_nibble(qg0 >> 12), x2.y, sg);
                    sg = fmaf(b_fp4_nibble(qg0 >> 16), x0.z, sg); sg = fmaf(b_fp4_nibble(qg0 >> 20), x2.z, sg);
                    sg = fmaf(b_fp4_nibble(qg0 >> 24), x0.w, sg); sg = fmaf(b_fp4_nibble(qg0 >> 28), x2.w, sg);
                    sg = fmaf(b_fp4_nibble(qg1), x1.x, sg); sg = fmaf(b_fp4_nibble(qg1 >> 4), x3.x, sg);
                    sg = fmaf(b_fp4_nibble(qg1 >> 8), x1.y, sg); sg = fmaf(b_fp4_nibble(qg1 >> 12), x3.y, sg);
                    sg = fmaf(b_fp4_nibble(qg1 >> 16), x1.z, sg); sg = fmaf(b_fp4_nibble(qg1 >> 20), x3.z, sg);
                    sg = fmaf(b_fp4_nibble(qg1 >> 24), x1.w, sg); sg = fmaf(b_fp4_nibble(qg1 >> 28), x3.w, sg);
                    ag[m] = fmaf(dg, sg, ag[m]);

                    su = fmaf(b_fp4_nibble(qu0), x0.x, su); su = fmaf(b_fp4_nibble(qu0 >> 4), x2.x, su);
                    su = fmaf(b_fp4_nibble(qu0 >> 8), x0.y, su); su = fmaf(b_fp4_nibble(qu0 >> 12), x2.y, su);
                    su = fmaf(b_fp4_nibble(qu0 >> 16), x0.z, su); su = fmaf(b_fp4_nibble(qu0 >> 20), x2.z, su);
                    su = fmaf(b_fp4_nibble(qu0 >> 24), x0.w, su); su = fmaf(b_fp4_nibble(qu0 >> 28), x2.w, su);
                    su = fmaf(b_fp4_nibble(qu1), x1.x, su); su = fmaf(b_fp4_nibble(qu1 >> 4), x3.x, su);
                    su = fmaf(b_fp4_nibble(qu1 >> 8), x1.y, su); su = fmaf(b_fp4_nibble(qu1 >> 12), x3.y, su);
                    su = fmaf(b_fp4_nibble(qu1 >> 16), x1.z, su); su = fmaf(b_fp4_nibble(qu1 >> 20), x3.z, su);
                    su = fmaf(b_fp4_nibble(qu1 >> 24), x1.w, su); su = fmaf(b_fp4_nibble(qu1 >> 28), x3.w, su);
                    au[m] = fmaf(du, su, au[m]);
                }
            }
        }
        for (int m = 0; m < nt; ++m) {
            float g = b_warp_sum8(ag[m]) * sc;
            float u = b_warp_sum8(au[m]) * sc;
            if (lane8 == 0) {
                float sg = 1.0f / (1.0f + expf(-g));
                float act = beta * tanhf(g / beta) * sg;
                float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                act_out[((size_t)tok_list[t0 + m] * n_used + slot_list[t0 + m]) * n_ff + row] = act * up;
            }
        }
    }
}


extern "C" __global__ void moe_dn_nvfp4_grouped_k(const uint8_t* __restrict__ W,
                                                  const float* __restrict__ scales,
                                                  const float* __restrict__ act_in,
                                                  const int* __restrict__ itop,
                                                  const float* __restrict__ wtop,
                                                  float* __restrict__ y_out,
                                                  int n_embd, int n_ff, int n_used, int B) {
    int exp_id = blockIdx.y;
    __shared__ int tok_list[64];
    __shared__ int slot_list[64];
    __shared__ float wt_list[64];
    __shared__ int count;
    if (threadIdx.x == 0) {
        int c = 0;
        int total = B * n_used;
        float sc = scales ? scales[exp_id] : 1.0f;
        for (int i = 0; i < total; ++i) {
            if (itop[i] == exp_id) {
                tok_list[c] = i / n_used;
                slot_list[c] = i % n_used;
                wt_list[c] = wtop[i] * sc;
                ++c;
            }
        }
        count = c;
    }
    __syncthreads();
    int cnt = count;
    if (cnt == 0) return;

    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;

    int nblk = n_ff >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;

    for (int t0 = 0; t0 < cnt; t0 += 8) {
        int nt = min(8, cnt - t0);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int b = lane8; b < nblk; b += 8) {
            const uint8_t* bp = wrow + (size_t)b * 36;
            uint32_t ds;
            memcpy(&ds, bp, 4);
            const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bp + 4);
            int xoff = b << 6;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                float d = b_ue4m3_to_fp32((uint8_t)((ds >> (sb * 8)) & 0xFF));
                uint32_t q0 = q32[sb * 2], q1 = q32[sb * 2 + 1];
                int xo = xoff + sb * 16;
                for (int m = 0; m < nt; ++m) {
                    const float4* x4 = reinterpret_cast<const float4*>(act_in + ((size_t)tok_list[t0 + m] * n_used + slot_list[t0 + m]) * n_ff + xo);
                    float4 x0 = x4[0], x1 = x4[1], x2 = x4[2], x3 = x4[3];
                    float s = 0.f;
                    s = fmaf(b_fp4_nibble(q0), x0.x, s); s = fmaf(b_fp4_nibble(q0 >> 4), x2.x, s);
                    s = fmaf(b_fp4_nibble(q0 >> 8), x0.y, s); s = fmaf(b_fp4_nibble(q0 >> 12), x2.y, s);
                    s = fmaf(b_fp4_nibble(q0 >> 16), x0.z, s); s = fmaf(b_fp4_nibble(q0 >> 20), x2.z, s);
                    s = fmaf(b_fp4_nibble(q0 >> 24), x0.w, s); s = fmaf(b_fp4_nibble(q0 >> 28), x2.w, s);
                    s = fmaf(b_fp4_nibble(q1), x1.x, s); s = fmaf(b_fp4_nibble(q1 >> 4), x3.x, s);
                    s = fmaf(b_fp4_nibble(q1 >> 8), x1.y, s); s = fmaf(b_fp4_nibble(q1 >> 12), x3.y, s);
                    s = fmaf(b_fp4_nibble(q1 >> 16), x1.z, s); s = fmaf(b_fp4_nibble(q1 >> 20), x3.z, s);
                    s = fmaf(b_fp4_nibble(q1 >> 24), x1.w, s); s = fmaf(b_fp4_nibble(q1 >> 28), x3.w, s);
                    acc[m] = fmaf(d, s, acc[m]);
                }
            }
        }
        for (int m = 0; m < nt; ++m) {
            float v = b_warp_sum8(acc[m]) * wt_list[t0 + m];
            if (lane8 == 0) atomicAdd(&y_out[(size_t)tok_list[t0 + m] * n_embd + row], v);
        }
    }
}


extern "C" __global__ void zero_f32_k(float* __restrict__ p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}


extern "C" int ink_moe_gu_nvfp4_batch(hipStream_t s, const uint8_t* W, const float* scales, const float* X, const int* itop, float* act_out, int n_ff, int k, int n_used, float beta, float lb, int B) {
    moe_gu_nvfp4_grouped_k<<<dim3((n_ff + 15) / 16, 128), dim3(128), 0, s>>>(W, scales, X, itop, act_out, n_ff, k, n_used, B, beta, lb);
    return 0;
}

extern "C" int ink_moe_dn_nvfp4_batch(hipStream_t s, const uint8_t* W, const float* scales, const float* act_in, const int* itop, const float* wtop, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    int tot = B * n_embd;
    zero_f32_k<<<dim3((tot + 255) / 256), dim3(256), 0, s>>>(y_out, tot);
    moe_dn_nvfp4_grouped_k<<<dim3((n_embd + 15) / 16, 128), dim3(128), 0, s>>>(W, scales, act_in, itop, wtop, y_out, n_embd, n_ff, n_used, B);
    return 0;
}

extern "C" __global__ void logits_topk64_batch_k(const float* __restrict__ logits,
                                                 int vocab_padded, int n_unpadded,
                                                 float* __restrict__ out_vals,
                                                 int* __restrict__ out_ids) {
    int bt = blockIdx.x;
    int tid = threadIdx.x;
    const float* lg = logits + (size_t)bt * vocab_padded;
    float tv[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    int ti[4] = {0, 0, 0, 0};
    const float4* lg4 = reinterpret_cast<const float4*>(lg);
    int n4 = n_unpadded >> 2;
    for (int i4 = tid; i4 < n4; i4 += 256) {
        float4 v4 = lg4[i4];
        int base = i4 << 2;
        float vals[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float v = vals[k];
            if (v > tv[3]) {
                tv[3] = v;
                ti[3] = base + k;
                if (tv[3] > tv[2]) { float fv = tv[2]; tv[2] = tv[3]; tv[3] = fv; int fi = ti[2]; ti[2] = ti[3]; ti[3] = fi; }
                if (tv[2] > tv[1]) { float fv = tv[1]; tv[1] = tv[2]; tv[2] = fv; int fi = ti[1]; ti[1] = ti[2]; ti[2] = fi; }
                if (tv[1] > tv[0]) { float fv = tv[0]; tv[0] = tv[1]; tv[1] = fv; int fi = ti[0]; ti[0] = ti[1]; ti[1] = fi; }
            }
        }
    }
    for (int i = (n4 << 2) + tid; i < n_unpadded; i += 256) {
        float v = lg[i];
        if (v > tv[3]) {
            tv[3] = v;
            ti[3] = i;
            if (tv[3] > tv[2]) { float fv = tv[2]; tv[2] = tv[3]; tv[3] = fv; int fi = ti[2]; ti[2] = ti[3]; ti[3] = fi; }
            if (tv[2] > tv[1]) { float fv = tv[1]; tv[1] = tv[2]; tv[2] = fv; int fi = ti[1]; ti[1] = ti[2]; ti[2] = fi; }
            if (tv[1] > tv[0]) { float fv = tv[0]; tv[0] = tv[1]; tv[1] = fv; int fi = ti[0]; ti[0] = ti[1]; ti[1] = fi; }
        }
    }
    __shared__ float sv[1024];
    __shared__ int si[1024];
    sv[tid * 4 + 0] = tv[0]; si[tid * 4 + 0] = ti[0];
    sv[tid * 4 + 1] = tv[1]; si[tid * 4 + 1] = ti[1];
    sv[tid * 4 + 2] = tv[2]; si[tid * 4 + 2] = ti[2];
    sv[tid * 4 + 3] = tv[3]; si[tid * 4 + 3] = ti[3];
    __syncthreads();
    if (tid < 32) {
        float* ov = out_vals + (size_t)bt * 64;
        int* oi = out_ids + (size_t)bt * 64;
        for (int r = 0; r < 64; ++r) {
            float bv = -1e30f;
            int bp = -1;
            for (int j = tid; j < 1024; j += 32) {
                float v = sv[j];
                if (v > bv) { bv = v; bp = j; }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float other_v = __shfl_down(bv, off, 32);
                int other_p = __shfl_down(bp, off, 32);
                if (other_v > bv || (other_v == bv && other_p >= 0 && (bp < 0 || other_p < bp))) {
                    bv = other_v;
                    bp = other_p;
                }
            }
            if (tid == 0) {
                if (bp >= 0) {
                    ov[r] = bv;
                    oi[r] = si[bp];
                    sv[bp] = -1e30f;
                } else {
                    ov[r] = -1e30f;
                    oi[r] = 0;
                }
            }
            __threadfence_block();
        }
    }
}


extern "C" int ink_softcap_argmax_batch(hipStream_t s, float* logits, float cap, int vocab_padded, int n_unpadded, int* out_toks, int B) {
    if (B <= 128) {
        softcap_argmax_batch_p1_k<<<dim3(32, B), dim3(256), 0, s>>>(logits, cap, vocab_padded, n_unpadded);
        softcap_argmax_batch_p2_k<<<dim3(B), dim3(32), 0, s>>>(out_toks);
        return 0;
    }
    softcap_argmax_batch_k<<<dim3(B), dim3(256), 0, s>>>(logits, cap, vocab_padded, n_unpadded, out_toks);
    return 0;
}


extern "C" int ink_logits_topk64_batch(hipStream_t s, const float* logits, int vocab_padded, int n_unpadded, float* out_vals, int* out_ids, int B) {
    logits_topk64_batch_k<<<dim3(B), dim3(256), 0, s>>>(logits, vocab_padded, n_unpadded, out_vals, out_ids);
    return 0;
}
