#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float b_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t man = x & 0x7;
    if (exp == 0) {
        return (float)man * 0.0009765625f;
    }
    uint32_t bits = ((exp + 119u) << 23) | (man << 20);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float b_bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ float b_fp4_nibble(uint32_t q) {
    uint32_t n = q & 7u;
    uint32_t e = n >> 1;
    uint32_t m = n & 1u;
    int v = (e == 0u) ? (int)m : (int)((2u + m) << (e - 1u));
    return (q & 8u) ? -(float)v : (float)v;
}

__device__ __forceinline__ float b_warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 32);
    }
    return v;
}

__device__ __forceinline__ float b_warp_sum8(float v) {
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 8);
    }
    return v;
}

__device__ __forceinline__ void gemm_q8_0_row_tile8(const uint8_t* __restrict__ wrow,
                                                    const float* __restrict__ X,
                                                    float* __restrict__ Y,
                                                    int row, int n, int k,
                                                    int t0, int n_t, int lane) {
    int nblk = k >> 5;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = lane; b < nblk; b += 32) {
        const uint8_t* bp = wrow + (size_t)b * 34;
        uint16_t dh_bits;
        memcpy(&dh_bits, bp, 2);
        float d = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
        int xoff = b << 5;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t q;
            memcpy(&q, bp + 2 + j * 4, 4);
            float w0 = (float)(int8_t)(q & 0xFF) * d;
            float w1 = (float)(int8_t)((q >> 8) & 0xFF) * d;
            float w2 = (float)(int8_t)((q >> 16) & 0xFF) * d;
            float w3 = (float)(int8_t)(q >> 24) * d;
            int xo = xoff + j * 4;
            for (int m = 0; m < n_t; ++m) {
                const float4 xv = *reinterpret_cast<const float4*>(X + (size_t)(t0 + m) * k + xo);
                acc[m] = fmaf(w0, xv.x, fmaf(w1, xv.y, fmaf(w2, xv.z, fmaf(w3, xv.w, acc[m]))));
            }
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}

extern "C" __global__ void gemm_q8_0_k(const uint8_t* __restrict__ W,
                                       const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n) return;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    int nblk = k >> 5;
    const uint8_t* wrow = W + (size_t)row * ((size_t)nblk * 34);
    gemm_q8_0_row_tile8(wrow, X, Y, row, n, k, t0, n_t, lane);
}

extern "C" __global__ void gemm_bf16_k(const uint16_t* __restrict__ W,
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
    const uint16_t* wrow = W + (size_t)row * k;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = lane; j < k; j += 32) {
        float wv = b_bf16_to_fp32(wrow[j]);
        for (int m = 0; m < n_t; ++m) {
            acc[m] = fmaf(wv, X[(size_t)(t0 + m) * k + j], acc[m]);
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}

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
    const float* wrow = W + (size_t)row * k;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = lane; j < k; j += 32) {
        float wv = wrow[j];
        for (int m = 0; m < n_t; ++m) {
            acc[m] = fmaf(wv, X[(size_t)(t0 + m) * k + j], acc[m]);
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}

extern "C" __global__ void qkvrg_gemm_k(const uint8_t* __restrict__ wq,
                                        const uint8_t* __restrict__ wk,
                                        const uint8_t* __restrict__ wv,
                                        const uint8_t* __restrict__ wrel,
                                        const uint8_t* __restrict__ wgate,
                                        const uint16_t* __restrict__ wsink,
                                        const uint16_t* __restrict__ wkdyn,
                                        const uint16_t* __restrict__ wvdyn,
                                        const float* __restrict__ inp0,
                                        const float* __restrict__ inp1,
                                        const float* __restrict__ inp2,
                                        float* __restrict__ q_out,
                                        float* __restrict__ k_out,
                                        float* __restrict__ v_out,
                                        float* __restrict__ r_out,
                                        float* __restrict__ gate_out,
                                        float* __restrict__ sink_out,
                                        float* __restrict__ kdyn_out,
                                        float* __restrict__ vdyn_out,
                                        int k_dim, int q_dim, int kv_dim,
                                        int r_dim, int n_head, int dyn_dim, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g_row = blockIdx.x * 8 + warp;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    int nblk = k_dim >> 5;
    size_t rb = (size_t)nblk * 34;

    if (g_row < q_dim) {
        gemm_q8_0_row_tile8(wq + (size_t)g_row * rb, inp0, q_out, g_row, q_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= q_dim;
    if (g_row < kv_dim) {
        gemm_q8_0_row_tile8(wk + (size_t)g_row * rb, inp1, k_out, g_row, kv_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= kv_dim;
    if (g_row < kv_dim) {
        gemm_q8_0_row_tile8(wv + (size_t)g_row * rb, inp2, v_out, g_row, kv_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= kv_dim;
    if (g_row < r_dim) {
        gemm_q8_0_row_tile8(wrel + (size_t)g_row * rb, inp0, r_out, g_row, r_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= r_dim;
    if (g_row < q_dim) {
        gemm_q8_0_row_tile8(wgate + (size_t)g_row * rb, inp0, gate_out, g_row, q_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= q_dim;
    if (g_row < n_head) {
        const uint16_t* wr = wsink + (size_t)g_row * k_dim;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int j = lane; j < k_dim; j += 32) {
            float wval = b_bf16_to_fp32(wr[j]);
            for (int m = 0; m < n_t; ++m) acc[m] = fmaf(wval, inp0[(size_t)(t0 + m) * k_dim + j], acc[m]);
        }
        for (int m = 0; m < n_t; ++m) {
            float v = b_warp_sum32(acc[m]);
            if (lane == 0) sink_out[(size_t)(t0 + m) * n_head + g_row] = v;
        }
        return;
    }
    g_row -= n_head;
    if (dyn_dim > 0) {
        if (g_row < dyn_dim) {
            const uint16_t* wr = wkdyn + (size_t)g_row * k_dim;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = lane; j < k_dim; j += 32) {
                float wval = b_bf16_to_fp32(wr[j]);
                for (int m = 0; m < n_t; ++m) acc[m] = fmaf(wval, inp1[(size_t)(t0 + m) * k_dim + j], acc[m]);
            }
            for (int m = 0; m < n_t; ++m) {
                float v = b_warp_sum32(acc[m]);
                if (lane == 0) kdyn_out[(size_t)(t0 + m) * dyn_dim + g_row] = v;
            }
            return;
        }
        g_row -= dyn_dim;
        if (g_row < dyn_dim) {
            const uint16_t* wr = wvdyn + (size_t)g_row * k_dim;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = lane; j < k_dim; j += 32) {
                float wval = b_bf16_to_fp32(wr[j]);
                for (int m = 0; m < n_t; ++m) acc[m] = fmaf(wval, inp2[(size_t)(t0 + m) * k_dim + j], acc[m]);
            }
            for (int m = 0; m < n_t; ++m) {
                float v = b_warp_sum32(acc[m]);
                if (lane == 0) vdyn_out[(size_t)(t0 + m) * dyn_dim + g_row] = v;
            }
            return;
        }
    }
}

extern "C" __global__ void embed_q8_0_norm_batch_k(const uint8_t* __restrict__ tok_embd,
                                                   const float* __restrict__ tok_norm,
                                                   const int* __restrict__ d_toks,
                                                   float* __restrict__ out,
                                                   int n_embd) {
    int t = blockIdx.x;
    int tok = d_toks[t];
    int nblk = n_embd >> 5;
    const uint8_t* row = tok_embd + (size_t)tok * ((size_t)nblk * 34);
    float* ov = out + (size_t)t * n_embd;
    float ss = 0.0f;
    for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
        int b = idx >> 5;
        int j = idx & 31;
        const uint8_t* bp = row + (size_t)b * 34;
        uint16_t dh_bits = (uint16_t)bp[0] | ((uint16_t)bp[1] << 8);
        float d = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
        float v = d * (float)(reinterpret_cast<const int8_t*>(bp + 2)[j]);
        ov[idx] = v;
        ss += v * v;
    }
    ss = b_warp_sum32(ss);
    __shared__ float red[8];
    int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n_embd + 1e-6f);
    }
    __syncthreads();
    float io = inv;
    if (tok_norm) {
        for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) ov[idx] = ov[idx] * io * tok_norm[idx];
    } else {
        for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) ov[idx] = ov[idx] * io;
    }
}

extern "C" __global__ void rmsnorm_batch_k(const float* __restrict__ X,
                                           const float* __restrict__ w,
                                           float* __restrict__ Y,
                                           int n, float eps) {
    int t = blockIdx.x;
    const float* xv = X + (size_t)t * n;
    float* yv = Y + (size_t)t * n;
    float ss = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = xv[j];
        ss += v * v;
    }
    ss = b_warp_sum32(ss);
    __shared__ float red[8];
    int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n + eps);
    }
    __syncthreads();
    float io = inv;
    if (w) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) yv[j] = xv[j] * io * w[j];
    } else {
        for (int j = threadIdx.x; j < n; j += blockDim.x) yv[j] = xv[j] * io;
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
    __shared__ float red[8];
    __shared__ float invs[16];
    __shared__ float scores[16];
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    float inv_sqrt_d = rsqrtf((float)n_embd);

    for (int s = 0; s < n_states; ++s) {
        const float* sv = (s < n_blocks) ? (blocks + ((size_t)s * max_B + t) * n_embd) : cur_t;
        float ss = 0.0f;
        float dot = 0.0f;
        for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
            float v = sv[j];
            ss += v * v;
            dot += v * qv[j];
        }
        ss = b_warp_sum32(ss);
        if ((threadIdx.x & 31) == 0) red[wid] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.0f;
            for (int i = 0; i < nwarps; ++i) tot += red[i];
            invs[s] = rsqrtf(tot / (float)n_embd + 1e-6f);
        }
        __syncthreads();
        dot = b_warp_sum32(dot);
        if ((threadIdx.x & 31) == 0) red[wid] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.0f;
            for (int i = 0; i < nwarps; ++i) tot += red[i];
            scores[s] = tot * invs[s] * inv_sqrt_d;
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

    float out_ss = 0.0f;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
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
        if ((threadIdx.x & 31) == 0) red[wid] = out_ss;
        __syncthreads();
        __shared__ float inv_out;
        if (threadIdx.x == 0) {
            float tot = 0.0f;
            for (int i = 0; i < nwarps; ++i) tot += red[i];
            inv_out = rsqrtf(tot / (float)n_embd + eps);
        }
        __syncthreads();
        float io = inv_out;
        for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
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
    sc = b_warp_sum32(sc);
    if ((threadIdx.x & 31) == 0) red[wid] = sc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nw; ++i) tot += red[i];
        rc_s = rsqrtf(tot / (float)n_embd + 1e-6f);
    }
    __syncthreads();

    sm = b_warp_sum32(sm);
    if ((threadIdx.x & 31) == 0) red[wid] = sm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nw; ++i) tot += red[i];
        rm_s = rsqrtf(tot / (float)n_embd + 1e-6f);
    }
    __syncthreads();

    const float* wr = wd + (size_t)r * (2 * n_embd);
    float rc = rc_s;
    float rm = rm_s;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        acc += wr[j] * (cur[j] * rc) + wr[n_embd + j] * (msg[j] * rm);
    }
    acc = b_warp_sum32(acc);
    if ((threadIdx.x & 31) == 0) red[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nw; ++i) tot += red[i];
        h_out[((size_t)c * max_B + t) * rank + r] = tot;
    }
}

extern "C" __global__ void loop_ctrl_up_norm_b_k(const float* __restrict__ cur0,
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
                                                 const float* __restrict__ nw0,
                                                 const float* __restrict__ nw1,
                                                 const float* __restrict__ nw2,
                                                 float* __restrict__ out0,
                                                 float* __restrict__ out1,
                                                 float* __restrict__ out2,
                                                 int n_embd, int rank, int max_B, float eps) {
    int c = blockIdx.x;
    int t = blockIdx.y;
    const float* cur = ((c == 0) ? cur0 : ((c == 1) ? cur1 : cur2)) + (size_t)t * n_embd;
    const float* msg = msgs + ((size_t)c * max_B + t) * n_embd;
    const float* h = h_in + ((size_t)c * max_B + t) * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    float* out = ((c == 0) ? out0 : ((c == 1) ? out1 : out2)) + (size_t)t * n_embd;

    __shared__ float hs[128];
    __shared__ float red[8];
    if (threadIdx.x < rank) hs[threadIdx.x] = h[threadIdx.x];
    __syncthreads();

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float* wgi = wg + (size_t)i * rank;
        const float* wfi = wf + (size_t)i * rank;
        float g = 0.0f, f = 0.0f;
        for (int r = 0; r < rank; ++r) {
            float hr = hs[r];
            g = fmaf(wgi[r], hr, g);
            f = fmaf(wfi[r], hr, f);
        }
        float cv = cur[i];
        float v = cv - tanhf(f) * cv + tanhf(g) * msg[i];
        out[i] = v;
        ss += v * v;
    }
    ss = b_warp_sum32(ss);
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nwarps; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        out[i] = out[i] * io * nw[i];
    }
}

extern "C" __global__ void kv_conv_qkvr_store_batch_k(float* __restrict__ Q,
                                                      const float* __restrict__ K_raw,
                                                      const float* __restrict__ V_raw,
                                                      float* __restrict__ R,
                                                      const float* __restrict__ k_conv_w,
                                                      const float* __restrict__ v_conv_w,
                                                      const float* __restrict__ k_dyn,
                                                      const float* __restrict__ v_dyn,
                                                      float* __restrict__ k_hist,
                                                      float* __restrict__ v_hist,
                                                      const float* __restrict__ q_clip,
                                                      const float* __restrict__ rel_clip,
                                                      __half* __restrict__ k_cache,
                                                      __half* __restrict__ v_cache,
                                                      const int* __restrict__ d_pos0,
                                                      int B, int is_swa, int n_head, int n_head_kv,
                                                      int dh, int d_rel, int max_ctx,
                                                      float qk_thresh, float rel_thresh,
                                                      float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (h < n_head) {
        const float* qch = q_clip + (size_t)h * dh;
        const float* rch = rel_clip + (size_t)h * d_rel;
        for (int t = 0; t < B; ++t) {
            int pos = pos0 + t;
            float tau = 1.0f;
            if (!is_swa) {
                float ratio = ((float)pos + 1.0f) / log_floor;
                if (ratio < 1.0f) ratio = 1.0f;
                tau = 1.0f + log_alpha * logf(ratio);
            }
            float* qh = Q + (size_t)t * q_dim + (size_t)h * dh;
            float q0 = qh[lane] * qch[lane];
            float q1 = qh[lane + 32] * qch[lane + 32];
            float q2 = qh[lane + 64] * qch[lane + 64];
            float q3 = qh[lane + 96] * qch[lane + 96];
            float ss = b_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
            float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
            qh[lane] = q0 * factor;
            qh[lane + 32] = q1 * factor;
            qh[lane + 64] = q2 * factor;
            qh[lane + 96] = q3 * factor;

            float* rh = R + (size_t)t * r_dim + (size_t)h * d_rel;
            float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
            float rss = b_warp_sum32(rv * rv);
            float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
            if (lane < d_rel) rh[lane] = rv * rfactor;
        }
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float kh0[4], kh1[4], kh2[4], vh0[4], vh1[4], vh2[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            kh0[p] = k_hist[c * 3 + 0]; kh1[p] = k_hist[c * 3 + 1]; kh2[p] = k_hist[c * 3 + 2];
            vh0[p] = v_hist[c * 3 + 0]; vh1[p] = v_hist[c * 3 + 1]; vh2[p] = v_hist[c * 3 + 2];
        }
        for (int t = 0; t < B; ++t) {
            int pos = pos0 + t;
            int slot = is_swa ? (pos % max_ctx) : pos;
            float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
            float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
            if (k_dyn) {
                const float* dk = k_dyn + (size_t)t * dyn_dim + (size_t)kh_idx * 4;
                const float* dv = v_dyn + (size_t)t * dyn_dim + (size_t)kh_idx * 4;
                kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
                vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
            }
            float kv_v[4], vv_v[4];
            float kss = 0.0f, vss = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int c = kh_idx * dh + lane + p * 32;
                float kx = K_raw[(size_t)t * kv_dim + c];
                const float* kwc = k_conv_w + (size_t)c * 4;
                float kacc = kwc[0] * kh0[p] + kwc[1] * kh1[p] + kwc[2] * kh2[p] + kwc[3] * kx;
                if (k_dyn) kacc += kd0 * kx + kd1 * kh2[p] + kd2 * kh1[p] + kd3 * kh0[p];
                kh0[p] = kh1[p]; kh1[p] = kh2[p]; kh2[p] = kx;
                float kout = (kx + kacc) * sc;
                kv_v[p] = kout;
                kss += kout * kout;

                float vx = V_raw[(size_t)t * kv_dim + c];
                const float* vwc = v_conv_w + (size_t)c * 4;
                float vacc = vwc[0] * vh0[p] + vwc[1] * vh1[p] + vwc[2] * vh2[p] + vwc[3] * vx;
                if (v_dyn) vacc += vd0 * vx + vd1 * vh2[p] + vd2 * vh1[p] + vd3 * vh0[p];
                vh0[p] = vh1[p]; vh1[p] = vh2[p]; vh2[p] = vx;
                float vout = (vx + vacc) * sc;
                vv_v[p] = vout;
                vss += vout * vout;
            }
            kss = b_warp_sum32(kss);
            vss = b_warp_sum32(vss);
            float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
            float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
            __half* kc_slot = k_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
            __half* vc_slot = v_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int d = lane + p * 32;
                kc_slot[d] = __float2half(kv_v[p] * kfac);
                vc_slot[d] = __float2half(vv_v[p] * vinv);
            }
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            k_hist[c * 3 + 0] = kh0[p]; k_hist[c * 3 + 1] = kh1[p]; k_hist[c * 3 + 2] = kh2[p];
            v_hist[c * 3 + 0] = vh0[p]; v_hist[c * 3 + 1] = vh1[p]; v_hist[c * 3 + 2] = vh2[p];
        }
    }
}

extern "C" __global__ void inkling_attn_batch_gated_k(const float* __restrict__ Q,
                                                      const float* __restrict__ R,
                                                      const float* __restrict__ sinks,
                                                      const float* __restrict__ rel_bias,
                                                      const __half* __restrict__ k_cache,
                                                      const __half* __restrict__ v_cache,
                                                      const float* __restrict__ gate,
                                                      const int* __restrict__ d_pos0,
                                                      float* __restrict__ out,
                                                      int n_head, int n_head_kv,
                                                      int dh, int d_rel,
                                                      int rel_extent, int swa_window,
                                                      int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    __shared__ float qs[128];
    __shared__ float rs[16];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[(size_t)bt * q_dim + (size_t)h * dh + d] * q_scale;
    }
    if (threadIdx.x < d_rel) {
        rs[threadIdx.x] = R[(size_t)bt * r_dim + (size_t)h * d_rel + threadIdx.x];
    }
    __syncthreads();

    int pos = (*d_pos0) + bt;
    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[(size_t)bt * n_head + h];

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half* kt = kc_h + (size_t)kv_slot * dh;
        float dot = qs[d0] * __half2float(kt[d0])
                  + qs[d1] * __half2float(kt[d1])
                  + qs[d2] * __half2float(kt[d2])
                  + qs[d3] * __half2float(kt[d3]);
        dot = b_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = b_warp_sum32(rp);
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
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    sm_o[wid][d0] = o0; sm_o[wid][d1] = o1; sm_o[wid][d2] = o2; sm_o[wid][d3] = o3;
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
        float* oh = out + (size_t)bt * q_dim + (size_t)h * dh;
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        oh[d0] = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        oh[d1] = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        oh[d2] = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        oh[d3] = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
    }
}

extern "C" __global__ void shortconv_res_batch_k(const float* __restrict__ raw,
                                                 const float* __restrict__ conv_w,
                                                 float* __restrict__ hist,
                                                 const float* __restrict__ add_src,
                                                 float* __restrict__ block_dst,
                                                 float* __restrict__ out,
                                                 int n_embd, int B) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_embd) return;
    float* hc = hist + (size_t)c * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = conv_w + (size_t)c * 4;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float sc = rsqrtf(5.0f);
    for (int t = 0; t < B; ++t) {
        size_t idx = (size_t)t * n_embd + c;
        float x = raw[idx];
        float conv_out = (x + w0 * h0 + w1 * h1 + w2 * h2 + w3 * x) * sc;
        h0 = h1; h1 = h2; h2 = x;
        if (block_dst) {
            block_dst[idx] = add_src[idx];
            out[idx] = conv_out;
        } else {
            out[idx] = add_src[idx] + conv_out;
        }
    }
    hc[0] = h0; hc[1] = h1; hc[2] = h2;
}

extern "C" __global__ void moe_router_batch_k(const float* __restrict__ logits,
                                              const float* __restrict__ bias,
                                              float* __restrict__ wtop,
                                              int* __restrict__ itop,
                                              int n_expert, int k, float route_scale) {
    int bt = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    const float* lg = logits + (size_t)bt * n_expert;
    float* wt = wtop + (size_t)bt * k;
    int* it = itop + (size_t)bt * k;
    __shared__ float sp[128];
    __shared__ float ss[128];
    __shared__ float wv[4];
    __shared__ int wi[4];
    __shared__ float chosen_p[16];
    if (tid < n_expert) {
        float p = 1.0f / (1.0f + expf(-lg[tid]));
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
            it[t] = bi;
            chosen_p[t] = sp[bi];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float wsum = 0.0f;
        for (int t = 0; t < k; ++t) wsum += chosen_p[t];
        float inv = route_scale / (wsum + 1e-20f);
        for (int t = 0; t < k; ++t) wt[t] = chosen_p[t] * inv;
    }
}

__device__ __forceinline__ float b_dot_nvfp4_row8(const uint8_t* __restrict__ wrow,
                                                  const float* __restrict__ x,
                                                  int nblk, int lane8) {
    float acc = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        const uint8_t* bp = wrow + (size_t)b * 36;
        uint32_t ds;
        memcpy(&ds, bp, 4);
        const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bp + 4);
        const float4* x4 = reinterpret_cast<const float4*>(x + ((size_t)b << 6));
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            float d = b_ue4m3_to_fp32((uint8_t)((ds >> (sb * 8)) & 0xFF));
            uint32_t q0 = q32[sb * 2];
            uint32_t q1 = q32[sb * 2 + 1];
            float4 x0 = x4[sb * 4 + 0];
            float4 x1 = x4[sb * 4 + 1];
            float4 x2 = x4[sb * 4 + 2];
            float4 x3 = x4[sb * 4 + 3];
            float s = 0.0f;
            s = fmaf(b_fp4_nibble(q0), x0.x, s);
            s = fmaf(b_fp4_nibble(q0 >> 4), x2.x, s);
            s = fmaf(b_fp4_nibble(q0 >> 8), x0.y, s);
            s = fmaf(b_fp4_nibble(q0 >> 12), x2.y, s);
            s = fmaf(b_fp4_nibble(q0 >> 16), x0.z, s);
            s = fmaf(b_fp4_nibble(q0 >> 20), x2.z, s);
            s = fmaf(b_fp4_nibble(q0 >> 24), x0.w, s);
            s = fmaf(b_fp4_nibble(q0 >> 28), x2.w, s);
            s = fmaf(b_fp4_nibble(q1), x1.x, s);
            s = fmaf(b_fp4_nibble(q1 >> 4), x3.x, s);
            s = fmaf(b_fp4_nibble(q1 >> 8), x1.y, s);
            s = fmaf(b_fp4_nibble(q1 >> 12), x3.y, s);
            s = fmaf(b_fp4_nibble(q1 >> 16), x1.z, s);
            s = fmaf(b_fp4_nibble(q1 >> 20), x3.z, s);
            s = fmaf(b_fp4_nibble(q1 >> 24), x1.w, s);
            s = fmaf(b_fp4_nibble(q1 >> 28), x3.w, s);
            acc = fmaf(d, s, acc);
        }
    }
    return b_warp_sum8(acc);
}

__device__ __forceinline__ float b_dot_q8_0_row32(const uint8_t* __restrict__ wrow,
                                                  const float* __restrict__ x,
                                                  int nblk, int lane) {
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const uint8_t* bp = wrow + (size_t)b * 34;
        uint16_t dh_bits;
        memcpy(&dh_bits, bp, 2);
        float d = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
        uint32_t qv[8];
        memcpy(qv, bp + 2, 32);
        const float4* x4 = reinterpret_cast<const float4*>(x + ((size_t)b << 5));
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t q = qv[j];
            float4 xv = x4[j];
            s = fmaf((float)(int8_t)(q & 0xFF), xv.x, s);
            s = fmaf((float)(int8_t)((q >> 8) & 0xFF), xv.y, s);
            s = fmaf((float)(int8_t)((q >> 16) & 0xFF), xv.z, s);
            s = fmaf((float)(int8_t)(q >> 24), xv.w, s);
        }
        acc += d * s;
    }
    return b_warp_sum32(acc);
}

extern "C" __global__ void moe_gu_nvfp4_batch_k(const uint8_t* __restrict__ W,
                                                const float* __restrict__ scales,
                                                const float* __restrict__ X,
                                                const int* __restrict__ itop,
                                                float* __restrict__ act_out,
                                                int n_ff, int k, int n_used,
                                                float beta, float lb) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    int slot = blockIdx.y;
    int bt = blockIdx.z;
    if (row >= n_ff) return;
    int exp_id = itop[(size_t)bt * n_used + slot];
    int nblk = k >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* base = W + (size_t)exp_id * exp_bytes;
    const float* xt = X + (size_t)bt * k;
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = b_dot_nvfp4_row8(base + (size_t)row * row_bytes, xt, nblk, lane8) * sc;
    float u = b_dot_nvfp4_row8(base + (size_t)(n_ff + row) * row_bytes, xt, nblk, lane8) * sc;
    if (lane8 == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[((size_t)bt * n_used + slot) * n_ff + row] = act * up;
    }
}

extern "C" __global__ void moe_dn_nvfp4_batch_k(const uint8_t* __restrict__ W,
                                                const float* __restrict__ scales,
                                                const float* __restrict__ act_in,
                                                const int* __restrict__ itop,
                                                const float* __restrict__ wtop,
                                                float* __restrict__ y_out,
                                                int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    int bt = blockIdx.y;
    if (row >= n_embd) return;
    int nblk = n_ff >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[(size_t)bt * n_used + slot];
        float w = wtop[(size_t)bt * n_used + slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = b_dot_nvfp4_row8(wrow, act_in + ((size_t)bt * n_used + slot) * n_ff, nblk, lane8);
        total = fmaf(v, w, total);
    }
    if (lane8 == 0) y_out[(size_t)bt * n_embd + row] = total;
}

extern "C" __global__ void moe_gu_q8_0_batch_k(const uint8_t* __restrict__ W,
                                               const float* __restrict__ scales,
                                               const float* __restrict__ X,
                                               const int* __restrict__ itop,
                                               float* __restrict__ act_out,
                                               int n_ff, int k, int n_used,
                                               float beta, float lb) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int slot = blockIdx.y;
    int bt = blockIdx.z;
    if (row >= n_ff) return;
    int exp_id = itop[(size_t)bt * n_used + slot];
    int nblk = k >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* base = W + (size_t)exp_id * exp_bytes;
    const float* xt = X + (size_t)bt * k;
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = b_dot_q8_0_row32(base + (size_t)row * row_bytes, xt, nblk, lane) * sc;
    float u = b_dot_q8_0_row32(base + (size_t)(n_ff + row) * row_bytes, xt, nblk, lane) * sc;
    if (lane == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[((size_t)bt * n_used + slot) * n_ff + row] = act * up;
    }
}

extern "C" __global__ void moe_dn_q8_0_batch_k(const uint8_t* __restrict__ W,
                                               const float* __restrict__ scales,
                                               const float* __restrict__ act_in,
                                               const int* __restrict__ itop,
                                               const float* __restrict__ wtop,
                                               float* __restrict__ y_out,
                                               int n_embd, int n_ff, int n_used) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int bt = blockIdx.y;
    if (row >= n_embd) return;
    int nblk = n_ff >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[(size_t)bt * n_used + slot];
        float w = wtop[(size_t)bt * n_used + slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = b_dot_q8_0_row32(wrow, act_in + ((size_t)bt * n_used + slot) * n_ff, nblk, lane);
        total = fmaf(v, w, total);
    }
    if (lane == 0) y_out[(size_t)bt * n_embd + row] = total;
}

extern "C" __global__ void softcap_argmax_batch_k(float* __restrict__ logits,
                                                  float cap, int vocab_padded, int n_unpadded,
                                                  int* __restrict__ out_toks) {
    int bt = blockIdx.x;
    float* lg = logits + (size_t)bt * vocab_padded;
    float bv = -1e30f;
    int bi = 0;
    for (int i = threadIdx.x; i < n_unpadded; i += blockDim.x) {
        float v = lg[i];
        if (cap > 0.0f) {
            v = cap * tanhf(v / cap);
            lg[i] = v;
        }
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

extern "C" int ink_gemm_q8_0(hipStream_t s, const uint8_t* W, const float* X, float* Y, int n, int k, int B) {
    gemm_q8_0_k<<<dim3((n + 7) / 8, (B + 7) / 8), dim3(256), 0, s>>>(W, X, Y, n, k, B);
    return 0;
}
extern "C" int ink_gemm_bf16(hipStream_t s, const uint16_t* W, const float* X, float* Y, int n, int k, int B) {
    gemm_bf16_k<<<dim3((n + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(W, X, Y, n, k, B);
    return 0;
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
        if (B == 4) {
            gemm_f32_f4_nt_k<4><<<dim3((n + 1) / 2, 1), dim3(64), 0, s>>>(W, X, Y, n, k, B);
            return 0;
        }
        gemm_f32_f4_rem_k<<<dim3((n + 1) / 2, (B + 7) / 8), dim3(64), 0, s>>>(W, X, Y, n, k, B);
        return 0;
    }
    gemm_f32_k<<<dim3((n + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(W, X, Y, n, k, B);
    return 0;
}
extern "C" int ink_qkvrg_gemm(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wrel, const uint8_t* wgate, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const float* inp0, const float* inp1, const float* inp2, float* q_out, float* k_out, float* v_out, float* r_out, float* gate_out, float* sink_out, float* kdyn_out, float* vdyn_out, int k_dim, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim, int B) {
    int total = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    qkvrg_gemm_k<<<dim3((total + 7) / 8, (B + 7) / 8), dim3(256), 0, s>>>(wq, wk, wv, wrel, wgate, wsink, wkdyn, wvdyn, inp0, inp1, inp2, q_out, k_out, v_out, r_out, gate_out, sink_out, kdyn_out, vdyn_out, k_dim, q_dim, kv_dim, r_dim, n_head, dyn_dim, B);
    return 0;
}
extern "C" int ink_embed_q8_0_norm_batch(hipStream_t s, const uint8_t* tok_embd, const float* tok_norm, const int* d_toks, float* out, int n_embd, int B) {
    embed_q8_0_norm_batch_k<<<dim3(B), dim3(256), 0, s>>>(tok_embd, tok_norm, d_toks, out, n_embd);
    return 0;
}
extern "C" int ink_rmsnorm_batch(hipStream_t s, const float* X, const float* w, float* Y, int n, int B, float eps) {
    rmsnorm_batch_k<<<dim3(B), dim3(256), 0, s>>>(X, w, Y, n, eps);
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
    float* out = ((c == 0) ? out0 : ((c == 1) ? out1 : out2)) + (size_t)t * n_embd;
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    __shared__ float red[8];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float v = out[i];
        ss += v * v;
    }
    ss = b_warp_sum32(ss);
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nwarps; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
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
    if (i >= n_embd) return;
    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* h_base = h_in + (size_t)c * max_B * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    float* out_base = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const float* wgi = wg + (size_t)i * rank;
    const float* wfi = wf + (size_t)i * rank;
    float g[NT], f[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) { g[m] = 0.0f; f[m] = 0.0f; }
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
        ink_loop_ctrl_down_wmma(s, cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B, B, 3);
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
        ink_loop_ctrl_down_wmma(s, cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank, max_B, B, 1);
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
extern "C" int ink_kv_conv_qkvr_store_batch(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist, float* v_hist, const float* q_clip, const float* rel_clip, __half* k_cache, __half* v_cache, const int* d_pos0, int B, int is_swa, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_batch_k<<<dim3(n_head + n_head_kv), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, q_clip, rel_clip, k_cache, v_cache, d_pos0, B, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}
extern "C" int ink_attn_batch_gated(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* k_cache, const __half* v_cache, const float* gate, const int* d_pos0, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    inkling_attn_batch_gated_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, k_cache, v_cache, gate, d_pos0, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}
extern "C" int ink_shortconv_res_batch(hipStream_t s, const float* raw, const float* conv_w, float* hist, const float* add_src, float* block_dst, float* out, int n_embd, int B) {
    shortconv_res_batch_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(raw, conv_w, hist, add_src, block_dst, out, n_embd, B);
    return 0;
}
extern "C" int ink_moe_router_batch(hipStream_t s, const float* logits, const float* bias, float* wtop, int* itop, int n_expert, int k, float route_scale, int B) {
    moe_router_batch_k<<<dim3(B), dim3(128), 0, s>>>(logits, bias, wtop, itop, n_expert, k, route_scale);
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
extern "C" int ink_moe_gu_q8_0_batch(hipStream_t s, const uint8_t* W, const float* scales, const float* X, const int* itop, float* act_out, int n_ff, int k, int n_used, float beta, float lb, int B) {
    moe_gu_q8_0_batch_k<<<dim3((n_ff + 7) / 8, n_used, B), dim3(256), 0, s>>>(W, scales, X, itop, act_out, n_ff, k, n_used, beta, lb);
    return 0;
}
extern "C" int ink_moe_dn_q8_0_batch(hipStream_t s, const uint8_t* W, const float* scales, const float* act_in, const int* itop, const float* wtop, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    moe_dn_q8_0_batch_k<<<dim3((n_embd + 7) / 8, B), dim3(256), 0, s>>>(W, scales, act_in, itop, wtop, y_out, n_embd, n_ff, n_used);
    return 0;
}
extern "C" int ink_softcap_argmax_batch(hipStream_t s, float* logits, float cap, int vocab_padded, int n_unpadded, int* out_toks, int B) {
    softcap_argmax_batch_k<<<dim3(B), dim3(256), 0, s>>>(logits, cap, vocab_padded, n_unpadded, out_toks);
    return 0;
}
