#include "rad_ops_common.cuh"

extern "C" __global__ void gemv_q8_0_k(const uint8_t* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int n, int k) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n) return;
    int nblk = k >> 5;
    const uint8_t* wrow = W + (size_t)row * ((size_t)nblk * 34);
    float v = dot_q8_0_row32(wrow, x, nblk, lane);
    if (lane == 0) y[row] = v;
}

extern "C" __global__ void gemv_nvfp4_k(const uint8_t* __restrict__ W,
                                        const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int n, int k, float scale) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n) return;
    int nblk = k >> 6;
    const uint8_t* wrow = W + (size_t)row * ((size_t)nblk * 36);
    float v = dot_nvfp4_row32(wrow, x, nblk, lane);
    if (lane == 0) y[row] = v * scale;
}

extern "C" __global__ void gemv_bf16_k(const uint16_t* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int n, int k) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= n) return;
    const uint16_t* wrow = W + (size_t)row * k;
    float acc = 0.0f;
    for (int j = lane; j < k; j += 32) {
        acc = fmaf(bf16_to_fp32(wrow[j]), x[j], acc);
    }
    acc = warp_sum32(acc);
    if (lane == 0) y[row] = acc;
}

extern "C" __global__ void gemv_f32_k(const float* __restrict__ W,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int n, int k) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int row = blockIdx.x * 8 + warp;
    int k4 = k >> 2;
    __shared__ float4 sx[576];
    const float4* x4 = reinterpret_cast<const float4*>(x);
    bool use_sm = (k4 <= 576);
    if (use_sm) {
        for (int j = tid; j < k4; j += 256) sx[j] = x4[j];
        __syncthreads();
    }
    if (row >= n) return;
    const float4* w4 = reinterpret_cast<const float4*>(W + (size_t)row * k);
    const float4* xsrc = use_sm ? sx : x4;
    float acc = 0.0f;
    for (int j = lane; j < k4; j += 32) {
        float4 wv = w4[j];
        float4 xv = xsrc[j];
        acc = fmaf(wv.x, xv.x, acc);
        acc = fmaf(wv.y, xv.y, acc);
        acc = fmaf(wv.z, xv.z, acc);
        acc = fmaf(wv.w, xv.w, acc);
    }
    acc = warp_sum32(acc);
    if (lane == 0) y[row] = acc;
}

extern "C" __global__ void embed_q8_0_norm_k(const uint8_t* __restrict__ tok_embd,
                                             const float* __restrict__ tok_norm,
                                             const int* __restrict__ d_tok,
                                             float* __restrict__ out,
                                             int n_embd) {
    int tok = *d_tok;
    int nblk = n_embd >> 5;
    const uint8_t* row = tok_embd + (size_t)tok * ((size_t)nblk * 34);
    float ss = 0.0f;
    for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
        int b = idx >> 5;
        int j = idx & 31;
        const uint8_t* bp = row + (size_t)b * 34;
        uint16_t dh_bits = (uint16_t)bp[0] | ((uint16_t)bp[1] << 8);
        float d = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
        float v = d * (float)(reinterpret_cast<const int8_t*>(bp + 2)[j]);
        out[idx] = v;
        ss += v * v;
    }
    ss = warp_sum32(ss);
    __shared__ float red[8];
    int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float t = 0.0f;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) t += red[i];
        inv = rsqrtf(t / (float)n_embd + 1e-6f);
    }
    __syncthreads();
    if (tok_norm) {
        for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
            out[idx] = out[idx] * inv * tok_norm[idx];
        }
    } else {
        for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
            out[idx] = out[idx] * inv;
        }
    }
}

extern "C" __global__ void rmsnorm_f32_k(const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         float* __restrict__ y,
                                         int n, float eps) {
    float ss = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = x[j];
        ss += v * v;
    }
    ss = warp_sum32(ss);
    __shared__ float red[8];
    int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float t = 0.0f;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) t += red[i];
        inv = rsqrtf(t / (float)n + eps);
    }
    __syncthreads();
    if (w) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            y[j] = x[j] * inv * w[j];
        }
    } else {
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            y[j] = x[j] * inv;
        }
    }
}

extern "C" __global__ void attn_res_mix_k(const float* __restrict__ blocks,
                                          int n_blocks,
                                          const float* __restrict__ cur,
                                          const float* __restrict__ query,
                                          float* __restrict__ out,
                                          int n_embd) {
    int q = blockIdx.x;
    const float* qv = query + (size_t)q * n_embd;
    float* ov = out + (size_t)q * n_embd;
    int n_states = n_blocks + 1;
    __shared__ float red[8];
    __shared__ float invs[16];
    __shared__ float scores[16];
    int wid = threadIdx.x >> 5;
    int nw = blockDim.x >> 5;
    float inv_sqrt_d = rsqrtf((float)n_embd);

    for (int s = 0; s < n_states; ++s) {
        const float* sv = (s < n_blocks) ? (blocks + (size_t)s * n_embd) : cur;
        float ss = 0.0f;
        float dot = 0.0f;
        for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
            float v = sv[j];
            ss += v * v;
            dot += v * qv[j];
        }
        ss = warp_sum32(ss);
        if ((threadIdx.x & 31) == 0) red[wid] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int i = 0; i < nw; ++i) t += red[i];
            invs[s] = rsqrtf(t / (float)n_embd + 1e-6f);
        }
        __syncthreads();
        dot = warp_sum32(dot);
        if ((threadIdx.x & 31) == 0) red[wid] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int i = 0; i < nw; ++i) t += red[i];
            scores[s] = t * invs[s] * inv_sqrt_d;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
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

    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_blocks; ++s) {
            acc += scores[s] * blocks[(size_t)s * n_embd + j];
        }
        acc += scores[n_blocks] * cur[j];
        ov[j] = acc;
    }
}

extern "C" __global__ void loop_ctrl_down_k(const float* __restrict__ cur,
                                            const float* __restrict__ msg,
                                            const float* __restrict__ w_down,
                                            float* __restrict__ h_out,
                                            int n_embd, int rank) {
    int r = blockIdx.x;
    if (r >= rank) return;
    __shared__ float red[8];
    __shared__ float rc_s, rm_s;
    int wid = threadIdx.x >> 5;
    int nw = blockDim.x >> 5;

    float sc = 0.0f, sm = 0.0f;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        float vc = cur[j];
        float vm = msg[j];
        sc += vc * vc;
        sm += vm * vm;
    }
    sc = warp_sum32(sc);
    if ((threadIdx.x & 31) == 0) red[wid] = sc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += red[i];
        rc_s = rsqrtf(t / (float)n_embd + 1e-6f);
    }
    __syncthreads();

    sm = warp_sum32(sm);
    if ((threadIdx.x & 31) == 0) red[wid] = sm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += red[i];
        rm_s = rsqrtf(t / (float)n_embd + 1e-6f);
    }
    __syncthreads();

    const float* wr = w_down + (size_t)r * (2 * n_embd);
    float rc = rc_s;
    float rm = rm_s;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        acc += wr[j] * (cur[j] * rc) + wr[n_embd + j] * (msg[j] * rm);
    }
    acc = warp_sum32(acc);
    if ((threadIdx.x & 31) == 0) red[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += red[i];
        h_out[r] = t;
    }
}

extern "C" __global__ void loop_ctrl_up_k(const float* __restrict__ cur,
                                          const float* __restrict__ msg,
                                          const float* __restrict__ h,
                                          const float* __restrict__ w_gain,
                                          const float* __restrict__ w_forget,
                                          float* __restrict__ out,
                                          int n_embd, int rank) {
    __shared__ float hs[128];
    if (threadIdx.x < rank) hs[threadIdx.x] = h[threadIdx.x];
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float* wg = w_gain + (size_t)i * rank;
    const float* wf = w_forget + (size_t)i * rank;
    float g = 0.0f, f = 0.0f;
    for (int r = 0; r < rank; ++r) {
        float hr = hs[r];
        g = fmaf(wg[r], hr, g);
        f = fmaf(wf[r], hr, f);
    }
    g = tanhf(g);
    f = tanhf(f);
    float c = cur[i];
    out[i] = c - f * c + g * msg[i];
}

extern "C" __global__ void shortconv_inkling_k(const float* __restrict__ cur,
                                               const float* __restrict__ weight,
                                               const float* __restrict__ dynamic,
                                               float* __restrict__ hist,
                                               float* __restrict__ out,
                                               int n_channels, int n_heads_dyn) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    float* hc = hist + (size_t)c * 3;
    float h0 = hc[0];
    float h1 = hc[1];
    float h2 = hc[2];
    float x = cur[c];
    const float* wc = weight + (size_t)c * 4;
    float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * x;
    float scale;
    if (dynamic) {
        int head_dim = n_channels / n_heads_dyn;
        int head = c / head_dim;
        const float* dc = dynamic + (size_t)head * 4;
        acc += dc[0] * x + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        scale = rsqrtf(9.0f);
    } else {
        scale = rsqrtf(5.0f);
    }
    hc[0] = h1;
    hc[1] = h2;
    hc[2] = x;
    out[c] = (x + acc) * scale;
}

extern "C" __global__ void qkvr_post_k(float* __restrict__ Q,
                                       float* __restrict__ K,
                                       float* __restrict__ V,
                                       float* __restrict__ R,
                                       const float* __restrict__ q_clip,
                                       const float* __restrict__ rel_clip,
                                       const int* __restrict__ d_pos,
                                       int is_swa, int n_head, int n_head_kv,
                                       int dh, int d_rel,
                                       float qk_thresh, float rel_thresh,
                                       float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    float tau = 1.0f;
    if (!is_swa) {
        float pos = (float)(*d_pos);
        float ratio = (pos + 1.0f) / log_floor;
        if (ratio < 1.0f) ratio = 1.0f;
        tau = 1.0f + log_alpha * logf(ratio);
    }

    if (h < n_head) {
        float* qh = Q + (size_t)h * dh;
        const float* qch = q_clip + (size_t)h * dh;
        float ss = 0.0f;
        for (int d = lane; d < dh; d += 32) {
            float v = qh[d] * qch[d];
            qh[d] = v;
            ss += v * v;
        }
        ss = warp_sum32(ss);
        float norm = sqrtf(__shfl(ss, 0, 32)) + 1e-6f;
        float factor = fminf(1.0f, qk_thresh / norm) * tau;
        for (int d = lane; d < dh; d += 32) {
            qh[d] *= factor;
        }

        float* rh = R + (size_t)h * d_rel;
        const float* rch = rel_clip + (size_t)h * d_rel;
        float rss = 0.0f;
        if (lane < d_rel) {
            float v = rh[lane] * rch[lane];
            rh[lane] = v;
            rss = v * v;
        }
        rss = warp_sum32(rss);
        float rnorm = sqrtf(__shfl(rss, 0, 32)) + 1e-6f;
        float rfactor = fminf(1.0f, rel_thresh / rnorm) * tau;
        if (lane < d_rel) {
            rh[lane] *= rfactor;
        }
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float* kh = K + (size_t)kh_idx * dh;
        float kss = 0.0f;
        for (int d = lane; d < dh; d += 32) {
            float v = kh[d];
            kss += v * v;
        }
        kss = warp_sum32(kss);
        float knorm = sqrtf(__shfl(kss, 0, 32)) + 1e-6f;
        float kfactor = fminf(1.0f, qk_thresh / knorm);
        for (int d = lane; d < dh; d += 32) {
            kh[d] *= kfactor;
        }

        float* vh = V + (size_t)kh_idx * dh;
        float vss = 0.0f;
        for (int d = lane; d < dh; d += 32) {
            float v = vh[d];
            vss += v * v;
        }
        vss = warp_sum32(vss);
        float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
        for (int d = lane; d < dh; d += 32) {
            vh[d] *= vinv;
        }
    }
}

extern "C" __global__ void kv_store_f16_k(const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          __half* __restrict__ k_cache,
                                          __half* __restrict__ v_cache,
                                          const int* __restrict__ d_pos,
                                          int max_ctx, int n_head_kv, int dh, int is_swa) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_head_kv * dh;
    if (idx >= total) return;
    int h = idx / dh;
    int d = idx % dh;
    int pos = *d_pos;
    int slot = is_swa ? (pos % max_ctx) : pos;
    size_t off = ((size_t)h * max_ctx + slot) * dh + d;
    k_cache[off] = __float2half(K[idx]);
    v_cache[off] = __float2half(V[idx]);
}

extern "C" __global__ void inkling_attn_decode_k(const float* __restrict__ Q,
                                                 const float* __restrict__ R,
                                                 const float* __restrict__ sinks,
                                                 const float* __restrict__ rel_bias,
                                                 const __half* __restrict__ k_cache,
                                                 const __half* __restrict__ v_cache,
                                                 const int* __restrict__ d_pos,
                                                 float* __restrict__ out,
                                                 int n_head, int n_head_kv,
                                                 int dh, int d_rel,
                                                 int rel_extent, int swa_window,
                                                 int is_swa, int max_ctx) {
    int h = blockIdx.x;
    if (h >= n_head) return;
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;

    __shared__ float qs[128];
    __shared__ float rs[16];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[(size_t)h * dh + d] * q_scale;
    }
    if (threadIdx.x < d_rel) {
        rs[threadIdx.x] = R[(size_t)h * d_rel + threadIdx.x];
    }
    __syncthreads();

    int pos = *d_pos;
    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[h];

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int d0 = lane;
    int d1 = lane + 32;
    int d2 = lane + 64;
    int d3 = lane + 96;

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half* kt = kc_h + (size_t)kv_slot * dh;
        float dot = qs[d0] * __half2float(kt[d0])
                  + qs[d1] * __half2float(kt[d1])
                  + qs[d2] * __half2float(kt[d2])
                  + qs[d3] * __half2float(kt[d3]);
        dot = warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = warp_sum32(rp);
            rbias = __shfl(rp, 0, 32);
        }
        score += rbias - sink_h;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        const __half* vt = vc_h + (size_t)kv_slot * dh;
        o0 = o0 * alpha + p * __half2float(vt[d0]);
        o1 = o1 * alpha + p * __half2float(vt[d1]);
        o2 = o2 * alpha + p * __half2float(vt[d2]);
        o3 = o3 * alpha + p * __half2float(vt[d3]);
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    if (lane == 0) {
        sm_m[wid] = m;
        sm_l[wid] = l;
    }
    sm_o[wid][d0] = o0;
    sm_o[wid][d1] = o1;
    sm_o[wid][d2] = o2;
    sm_o[wid][d3] = o3;
    __syncthreads();

    if (wid == 0) {
        float gm = sm_m[0];
        for (int w = 1; w < nw; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
        for (int w = 0; w < nw; ++w) {
            float sc = expf(sm_m[w] - gm);
            gl += sm_l[w] * sc;
            go0 += sm_o[w][d0] * sc;
            go1 += sm_o[w][d1] * sc;
            go2 += sm_o[w][d2] * sc;
            go3 += sm_o[w][d3] * sc;
        }
        float inv_l = 1.0f / gl;
        float* oh = out + (size_t)h * dh;
        oh[d0] = go0 * inv_l;
        oh[d1] = go1 * inv_l;
        oh[d2] = go2 * inv_l;
        oh[d3] = go3 * inv_l;
    }
}

extern "C" __global__ void attn_gate_mul_k(const float* __restrict__ attn_out,
                                           const float* __restrict__ gate,
                                           float* __restrict__ out,
                                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        out[i] = attn_out[i] * g;
    }
}

extern "C" __global__ void moe_router_sigmoid_k(const float* __restrict__ logits,
                                                const float* __restrict__ bias,
                                                float* __restrict__ wtop,
                                                int* __restrict__ itop,
                                                int n_expert, int k,
                                                float route_scale) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    __shared__ float sp[128];
    __shared__ float ss[128];
    __shared__ float wv[4];
    __shared__ int wi[4];
    __shared__ float chosen_p[16];
    if (tid < n_expert) {
        float p = 1.0f / (1.0f + expf(-logits[tid]));
        sp[tid] = p;
        ss[tid] = bias ? (p + bias[tid]) : p;
    } else {
        sp[tid] = 0.0f;
        ss[tid] = -1e30f;
    }
    __syncthreads();

    for (int t = 0; t < k; ++t) {
        float v = ss[tid];
        int idx = tid;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down(v, off, 32);
            int oi = __shfl_down(idx, off, 32);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0) {
            wv[wid] = v;
            wi[wid] = idx;
        }
        __syncthreads();
        if (tid == 0) {
            float bv = wv[0];
            int bi = wi[0];
            for (int w = 1; w < 4; ++w) {
                if (wv[w] > bv || (wv[w] == bv && wi[w] < bi)) {
                    bv = wv[w];
                    bi = wi[w];
                }
            }
            ss[bi] = -1e30f;
            itop[t] = bi;
            chosen_p[t] = sp[bi];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float wsum = 0.0f;
        for (int t = 0; t < k; ++t) wsum += chosen_p[t];
        float inv = route_scale / (wsum + 1e-20f);
        for (int t = 0; t < k; ++t) wtop[t] = chosen_p[t] * inv;
    }
}

extern "C" __global__ void moe_gate_up_situ_nvfp4_k(const uint8_t* __restrict__ W,
                                                    const float* __restrict__ scales,
                                                    const float* __restrict__ x,
                                                    const int* __restrict__ itop,
                                                    float* __restrict__ act_out,
                                                    int n_ff, int k,
                                                    float beta, float lb) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    int slot = blockIdx.y;
    if (row >= n_ff) return;
    int exp_id = itop[slot];
    int nblk = k >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* base = W + (size_t)exp_id * exp_bytes;
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = dot_nvfp4_row8(base + (size_t)row * row_bytes, x, nblk, lane8) * sc;
    float u = dot_nvfp4_row8(base + (size_t)(n_ff + row) * row_bytes, x, nblk, lane8) * sc;
    if (lane8 == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[(size_t)slot * n_ff + row] = act * up;
    }
}

extern "C" __global__ void moe_down_nvfp4_k(const uint8_t* __restrict__ W,
                                            const float* __restrict__ scales,
                                            const float* __restrict__ act_in,
                                            const int* __restrict__ itop,
                                            const float* __restrict__ wtop,
                                            float* __restrict__ y_out,
                                            int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;
    int nblk = n_ff >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[slot];
        float w = wtop[slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = dot_nvfp4_row8(wrow, act_in + (size_t)slot * n_ff, nblk, lane8);
        total = fmaf(v, w, total);
    }
    if (lane8 == 0) y_out[row] = total;
}

extern "C" __global__ void moe_gate_up_situ_q8_0_k(const uint8_t* __restrict__ W,
                                                   const float* __restrict__ scales,
                                                   const float* __restrict__ x,
                                                   const int* __restrict__ itop,
                                                   float* __restrict__ act_out,
                                                   int n_ff, int k,
                                                   float beta, float lb) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int slot = blockIdx.y;
    if (row >= n_ff) return;
    int exp_id = itop[slot];
    int nblk = k >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* base = W + (size_t)exp_id * exp_bytes;
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = dot_q8_0_row32(base + (size_t)row * row_bytes, x, nblk, lane) * sc;
    float u = dot_q8_0_row32(base + (size_t)(n_ff + row) * row_bytes, x, nblk, lane) * sc;
    if (lane == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[(size_t)slot * n_ff + row] = act * up;
    }
}

extern "C" __global__ void moe_down_q8_0_k(const uint8_t* __restrict__ W,
                                           const float* __restrict__ scales,
                                           const float* __restrict__ act_in,
                                           const int* __restrict__ itop,
                                           const float* __restrict__ wtop,
                                           float* __restrict__ y_out,
                                           int n_embd, int n_ff, int n_used) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n_embd) return;
    int nblk = n_ff >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[slot];
        float w = wtop[slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = dot_q8_0_row32(wrow, act_in + (size_t)slot * n_ff, nblk, lane);
        total = fmaf(v, w, total);
    }
    if (lane == 0) y_out[row] = total;
}

extern "C" __global__ void vec_add_k(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ void copy_f32_k(const float* __restrict__ src,
                                      float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__device__ float g_softcap_max_v[64];
__device__ int g_softcap_max_i[64];

extern "C" __global__ void softcap_argmax_p1_k(float* __restrict__ logits,
                                               float cap, int n_unpadded) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = gridDim.x * blockDim.x;
    float bv = -1e30f;
    int bi = 0;
    for (int i = bid * blockDim.x + tid; i < n_unpadded; i += stride) {
        float v = logits[i];
        if (cap > 0.0f) {
            v = cap * tanhf(v / cap);
            logits[i] = v;
        }
        if (v > bv) {
            bv = v;
            bi = i;
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
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        rv[wid] = bv;
        ri[wid] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        float m = rv[0];
        int mi = ri[0];
        for (int i = 1; i < 8; ++i) {
            if (rv[i] > m || (rv[i] == m && ri[i] < mi)) {
                m = rv[i];
                mi = ri[i];
            }
        }
        g_softcap_max_v[bid] = m;
        g_softcap_max_i[bid] = mi;
    }
}

extern "C" __global__ void softcap_argmax_p2_k(int* __restrict__ out_tok) {
    int tid = threadIdx.x;
    float v = g_softcap_max_v[tid];
    int idx = g_softcap_max_i[tid];
    float v2 = g_softcap_max_v[tid + 32];
    int idx2 = g_softcap_max_i[tid + 32];
    if (v2 > v || (v2 == v && idx2 < idx)) {
        v = v2;
        idx = idx2;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(v, off, 32);
        int oi = __shfl_down(idx, off, 32);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    if (tid == 0) *out_tok = idx;
}

extern "C" __global__ void softcap_argmax_k(float* __restrict__ logits,
                                            float cap, int n_unpadded,
                                            int* __restrict__ out_tok) {
    float bv = -1e30f;
    int bi = 0;
    for (int i = threadIdx.x; i < n_unpadded; i += blockDim.x) {
        float v = logits[i];
        if (cap > 0.0f) {
            v = cap * tanhf(v / cap);
            logits[i] = v;
        }
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    __shared__ float rv[8];
    __shared__ int ri[8];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(bv, off, 32);
        int oi = __shfl_down(bi, off, 32);
        if (ov > bv) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        rv[wid] = bv;
        ri[wid] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = rv[0];
        int mi = ri[0];
        int nw = blockDim.x >> 5;
        for (int i = 1; i < nw; ++i) {
            if (rv[i] > m) {
                m = rv[i];
                mi = ri[i];
            }
        }
        *out_tok = mi;
    }
}

extern "C" int ink_gemv_q8_0(hipStream_t s, const uint8_t* W, const float* x, float* y, int n, int k) {
    gemv_q8_0_k<<<dim3((n + 7) / 8), dim3(256), 0, s>>>(W, x, y, n, k);
    return 0;
}

extern "C" int ink_gemv_nvfp4(hipStream_t s, const uint8_t* W, const float* x, float* y, int n, int k, float scale) {
    gemv_nvfp4_k<<<dim3((n + 7) / 8), dim3(256), 0, s>>>(W, x, y, n, k, scale);
    return 0;
}

extern "C" int ink_gemv_bf16(hipStream_t s, const uint16_t* W, const float* x, float* y, int n, int k) {
    gemv_bf16_k<<<dim3((n + 3) / 4), dim3(128), 0, s>>>(W, x, y, n, k);
    return 0;
}

extern "C" int ink_gemv_f32(hipStream_t s, const float* W, const float* x, float* y, int n, int k) {
    gemv_f32_k<<<dim3((n + 7) / 8), dim3(256), 0, s>>>(W, x, y, n, k);
    return 0;
}

extern "C" int ink_embed_q8_0_norm(hipStream_t s, const uint8_t* tok_embd, const float* tok_norm, const int* d_tok, float* out, int n_embd) {
    embed_q8_0_norm_k<<<dim3(1), dim3(256), 0, s>>>(tok_embd, tok_norm, d_tok, out, n_embd);
    return 0;
}

extern "C" int ink_rmsnorm_batch(hipStream_t s, const float* X, const float* w, float* Y, int n, int B, float eps);

extern "C" int ink_rmsnorm(hipStream_t s, const float* x, const float* w, float* y, int n, float eps) {
    return ink_rmsnorm_batch(s, x, w, y, n, 1, eps);
}

extern "C" int ink_mix_batch(hipStream_t s, const float* blocks, int n_blocks, int max_B, const float* cur, const float* query, const float* w0, const float* w1, const float* w2, float* out, int n_embd, int n_queries, int B, float eps);

extern "C" int ink_attn_res_mix(hipStream_t s, const float* blocks, int n_blocks, const float* cur, const float* query, float* out, int n_embd, int n_queries) {
    return ink_mix_batch(s, blocks, n_blocks, 1, cur, query, nullptr, nullptr, nullptr, out, n_embd, n_queries, 1, 1e-6f);
}

extern "C" int ink_loop_ctrl(hipStream_t s, const float* cur, const float* msg, const float* w_down, const float* w_gain, const float* w_forget, float* h_buf, float* out, int n_embd, int rank) {
    loop_ctrl_down_k<<<dim3(rank), dim3(256), 0, s>>>(cur, msg, w_down, h_buf, n_embd, rank);
    loop_ctrl_up_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(cur, msg, h_buf, w_gain, w_forget, out, n_embd, rank);
    return 0;
}

extern "C" int ink_shortconv(hipStream_t s, const float* cur, const float* weight, const float* dynamic, float* hist, float* out, int n_channels, int n_heads_dyn) {
    shortconv_inkling_k<<<dim3((n_channels + 255) / 256), dim3(256), 0, s>>>(cur, weight, dynamic, hist, out, n_channels, n_heads_dyn);
    return 0;
}

extern "C" int ink_qkvr_post(hipStream_t s, float* Q, float* K, float* V, float* R, const float* q_clip, const float* rel_clip, const int* d_pos, int is_swa, int n_head, int n_head_kv, int dh, int d_rel, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    qkvr_post_k<<<dim3(n_head + n_head_kv), dim3(32), 0, s>>>(Q, K, V, R, q_clip, rel_clip, d_pos, is_swa, n_head, n_head_kv, dh, d_rel, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}

extern "C" int ink_kv_store_f16(hipStream_t s, const float* K, const float* V, __half* k_cache, __half* v_cache, const int* d_pos, int max_ctx, int n_head_kv, int dh, int is_swa) {
    int total = n_head_kv * dh;
    kv_store_f16_k<<<dim3((total + 255) / 256), dim3(256), 0, s>>>(K, V, k_cache, v_cache, d_pos, max_ctx, n_head_kv, dh, is_swa);
    return 0;
}

extern "C" int ink_attn_decode(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* k_cache, const __half* v_cache, const int* d_pos, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx) {
    inkling_attn_decode_k<<<dim3(n_head), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, k_cache, v_cache, d_pos, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" int ink_attn_gate_mul(hipStream_t s, const float* attn_out, const float* gate, float* out, int n) {
    attn_gate_mul_k<<<dim3((n + 255) / 256), dim3(256), 0, s>>>(attn_out, gate, out, n);
    return 0;
}

extern "C" int ink_moe_router(hipStream_t s, const float* logits, const float* bias, float* wtop, int* itop, int n_expert, int k, float route_scale) {
    moe_router_sigmoid_k<<<dim3(1), dim3(128), 0, s>>>(logits, bias, wtop, itop, n_expert, k, route_scale);
    return 0;
}

extern "C" int ink_moe_gate_up_situ_nvfp4(hipStream_t s, const uint8_t* W, const float* scales, const float* x, const int* itop, float* act_out, int n_ff, int k, float beta, float lb) {
    moe_gate_up_situ_nvfp4_k<<<dim3((n_ff + 15) / 16, 8), dim3(128), 0, s>>>(W, scales, x, itop, act_out, n_ff, k, beta, lb);
    return 0;
}

extern "C" int ink_moe_down_nvfp4(hipStream_t s, const uint8_t* W, const float* scales, const float* act_in, const int* itop, const float* wtop, float* y_out, int n_embd, int n_ff, int n_used) {
    moe_down_nvfp4_k<<<dim3((n_embd + 15) / 16), dim3(128), 0, s>>>(W, scales, act_in, itop, wtop, y_out, n_embd, n_ff, n_used);
    return 0;
}

extern "C" int ink_moe_gate_up_situ_q8_0(hipStream_t s, const uint8_t* W, const float* scales, const float* x, const int* itop, float* act_out, int n_ff, int k, float beta, float lb) {
    moe_gate_up_situ_q8_0_k<<<dim3((n_ff + 7) / 8, 8), dim3(256), 0, s>>>(W, scales, x, itop, act_out, n_ff, k, beta, lb);
    return 0;
}

extern "C" int ink_moe_down_q8_0(hipStream_t s, const uint8_t* W, const float* scales, const float* act_in, const int* itop, const float* wtop, float* y_out, int n_embd, int n_ff, int n_used) {
    moe_down_q8_0_k<<<dim3((n_embd + 7) / 8), dim3(256), 0, s>>>(W, scales, act_in, itop, wtop, y_out, n_embd, n_ff, n_used);
    return 0;
}

extern "C" int ink_vec_add(hipStream_t s, const float* a, const float* b, float* y, int n) {
    vec_add_k<<<dim3((n + 255) / 256), dim3(256), 0, s>>>(a, b, y, n);
    return 0;
}

extern "C" int ink_copy_f32(hipStream_t s, const float* src, float* dst, int n) {
    copy_f32_k<<<dim3((n + 255) / 256), dim3(256), 0, s>>>(src, dst, n);
    return 0;
}

extern "C" int ink_softcap_argmax(hipStream_t s, float* logits, float cap, int n_unpadded, int* out_tok) {
    softcap_argmax_p1_k<<<dim3(64), dim3(256), 0, s>>>(logits, cap, n_unpadded);
    softcap_argmax_p2_k<<<dim3(1), dim3(32), 0, s>>>(out_tok);
    return 0;
}
