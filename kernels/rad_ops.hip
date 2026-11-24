#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __constant__ float KVAL_FP4[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

__device__ __forceinline__ float ue4m3_to_fp32_fast(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t man = x & 0x7;
    if (exp == 0) {
        return (float)man * 0.0009765625f;
    }
    uint32_t bits = ((exp + 119u) << 23) | (man << 20);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 32);
    }
    return v;
}

__device__ __forceinline__ float warp_sum16(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 16);
    }
    return v;
}

__device__ __forceinline__ float warp_sum8(float v) {
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 8);
    }
    return v;
}

__device__ __forceinline__ float dot_q8_0_blk(const uint8_t* __restrict__ bp,
                                              const float* __restrict__ xb) {
    uint16_t dh_bits;
    memcpy(&dh_bits, bp, 2);
    float d = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
    uint32_t qv[8];
    memcpy(qv, bp + 2, 32);
    const float4* x4 = reinterpret_cast<const float4*>(xb);
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
    return d * s;
}

__device__ __forceinline__ float fp4_nibble(uint32_t q) {
    uint32_t n = q & 7u;
    uint32_t e = n >> 1;
    uint32_t m = n & 1u;
    int v = (e == 0u) ? (int)m : (int)((2u + m) << (e - 1u));
    return (q & 8u) ? -(float)v : (float)v;
}

__device__ __forceinline__ float dot_nvfp4_blk(const uint8_t* __restrict__ bp,
                                               const float* __restrict__ xb) {
    uint32_t ds;
    memcpy(&ds, bp, 4);
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bp + 4);
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    float acc = 0.0f;
#pragma unroll
    for (int sb = 0; sb < 4; ++sb) {
        float d = ue4m3_to_fp32_fast((uint8_t)((ds >> (sb * 8)) & 0xFF));
        uint32_t q0 = q32[sb * 2];
        uint32_t q1 = q32[sb * 2 + 1];
        float4 x0 = x4[sb * 4 + 0];
        float4 x1 = x4[sb * 4 + 1];
        float4 x2 = x4[sb * 4 + 2];
        float4 x3 = x4[sb * 4 + 3];
        float s = 0.0f;
        s = fmaf(fp4_nibble(q0), x0.x, s);
        s = fmaf(fp4_nibble(q0 >> 4), x2.x, s);
        s = fmaf(fp4_nibble(q0 >> 8), x0.y, s);
        s = fmaf(fp4_nibble(q0 >> 12), x2.y, s);
        s = fmaf(fp4_nibble(q0 >> 16), x0.z, s);
        s = fmaf(fp4_nibble(q0 >> 20), x2.z, s);
        s = fmaf(fp4_nibble(q0 >> 24), x0.w, s);
        s = fmaf(fp4_nibble(q0 >> 28), x2.w, s);
        s = fmaf(fp4_nibble(q1), x1.x, s);
        s = fmaf(fp4_nibble(q1 >> 4), x3.x, s);
        s = fmaf(fp4_nibble(q1 >> 8), x1.y, s);
        s = fmaf(fp4_nibble(q1 >> 12), x3.y, s);
        s = fmaf(fp4_nibble(q1 >> 16), x1.z, s);
        s = fmaf(fp4_nibble(q1 >> 20), x3.z, s);
        s = fmaf(fp4_nibble(q1 >> 24), x1.w, s);
        s = fmaf(fp4_nibble(q1 >> 28), x3.w, s);
        acc = fmaf(d, s, acc);
    }
    return acc;
}

__device__ __forceinline__ float dot_q8_0_row32(const uint8_t* __restrict__ wrow,
                                                const float* __restrict__ x,
                                                int nblk, int lane) {
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        acc += dot_q8_0_blk(wrow + (size_t)b * 34, x + ((size_t)b << 5));
    }
    return warp_sum32(acc);
}

__device__ __forceinline__ float dot_nvfp4_row32(const uint8_t* __restrict__ wrow,
                                                 const float* __restrict__ x,
                                                 int nblk, int lane) {
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        acc += dot_nvfp4_blk(wrow + (size_t)b * 36, x + ((size_t)b << 6));
    }
    return warp_sum32(acc);
}

__device__ __forceinline__ float dot_nvfp4_row16(const uint8_t* __restrict__ wrow,
                                                 const float* __restrict__ x,
                                                 int nblk, int lane16) {
    float acc = 0.0f;
    for (int b = lane16; b < nblk; b += 16) {
        acc += dot_nvfp4_blk(wrow + (size_t)b * 36, x + ((size_t)b << 6));
    }
    return warp_sum16(acc);
}

__device__ __forceinline__ float dot_nvfp4_row8(const uint8_t* __restrict__ wrow,
                                                const float* __restrict__ x,
                                                int nblk, int lane8) {
    float acc = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        acc += dot_nvfp4_blk(wrow + (size_t)b * 36, x + ((size_t)b << 6));
    }
    return warp_sum8(acc);
}

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

extern "C" __global__ void qkvrg_gemv_k(const uint8_t* __restrict__ wq,
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
                                        int r_dim, int n_head, int dyn_dim) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g_row = blockIdx.x * 8 + warp;
    int nblk = k_dim >> 5;
    size_t rb = (size_t)nblk * 34;

    if (g_row < q_dim) {
        float v = dot_q8_0_row32(wq + (size_t)g_row * rb, inp0, nblk, lane);
        if (lane == 0) q_out[g_row] = v;
        return;
    }
    g_row -= q_dim;
    if (g_row < kv_dim) {
        float v = dot_q8_0_row32(wk + (size_t)g_row * rb, inp1, nblk, lane);
        if (lane == 0) k_out[g_row] = v;
        return;
    }
    g_row -= kv_dim;
    if (g_row < kv_dim) {
        float v = dot_q8_0_row32(wv + (size_t)g_row * rb, inp2, nblk, lane);
        if (lane == 0) v_out[g_row] = v;
        return;
    }
    g_row -= kv_dim;
    if (g_row < r_dim) {
        float v = dot_q8_0_row32(wrel + (size_t)g_row * rb, inp0, nblk, lane);
        if (lane == 0) r_out[g_row] = v;
        return;
    }
    g_row -= r_dim;
    if (g_row < q_dim) {
        float v = dot_q8_0_row32(wgate + (size_t)g_row * rb, inp0, nblk, lane);
        if (lane == 0) gate_out[g_row] = v;
        return;
    }
    g_row -= q_dim;
    if (g_row < n_head) {
        const uint16_t* wr = wsink + (size_t)g_row * k_dim;
        float acc = 0.0f;
        for (int j = lane; j < k_dim; j += 32) acc = fmaf(bf16_to_fp32(wr[j]), inp0[j], acc);
        acc = warp_sum32(acc);
        if (lane == 0) sink_out[g_row] = acc;
        return;
    }
    g_row -= n_head;
    if (dyn_dim > 0) {
        if (g_row < dyn_dim) {
            const uint16_t* wr = wkdyn + (size_t)g_row * k_dim;
            float acc = 0.0f;
            for (int j = lane; j < k_dim; j += 32) acc = fmaf(bf16_to_fp32(wr[j]), inp1[j], acc);
            acc = warp_sum32(acc);
            if (lane == 0) kdyn_out[g_row] = acc;
            return;
        }
        g_row -= dyn_dim;
        if (g_row < dyn_dim) {
            const uint16_t* wr = wvdyn + (size_t)g_row * k_dim;
            float acc = 0.0f;
            for (int j = lane; j < k_dim; j += 32) acc = fmaf(bf16_to_fp32(wr[j]), inp2[j], acc);
            acc = warp_sum32(acc);
            if (lane == 0) vdyn_out[g_row] = acc;
            return;
        }
    }
}

extern "C" __global__ void kv_conv_qkvr_store_k(float* __restrict__ Q,
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
                                                const int* __restrict__ d_pos,
                                                int is_swa, int n_head, int n_head_kv,
                                                int dh, int d_rel, int max_ctx,
                                                float qk_thresh, float rel_thresh,
                                                float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos = *d_pos;
    float tau = 1.0f;
    if (!is_swa) {
        float ratio = ((float)pos + 1.0f) / log_floor;
        if (ratio < 1.0f) ratio = 1.0f;
        tau = 1.0f + log_alpha * logf(ratio);
    }

    if (h < n_head) {
        float* qh = Q + (size_t)h * dh;
        const float* qch = q_clip + (size_t)h * dh;
        float ss = 0.0f;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        ss = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
        ss = warp_sum32(ss);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;

        float* rh = R + (size_t)h * d_rel;
        const float* rch = rel_clip + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = warp_sum32(rv * rv);
        float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
        if (lane < d_rel) rh[lane] = rv * rfactor;
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        int slot = is_swa ? (pos % max_ctx) : pos;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float kd0 = 0.0f, kd1 = 0.0f, kd2 = 0.0f, kd3 = 0.0f;
        float vd0 = 0.0f, vd1 = 0.0f, vd2 = 0.0f, vd3 = 0.0f;
        if (k_dyn) {
            const float* dk = k_dyn + (size_t)kh_idx * 4;
            const float* dv = v_dyn + (size_t)kh_idx * 4;
            kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
            vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
        }
        float kv_v[4];
        float vv_v[4];
        float kss = 0.0f, vss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            int c = kh_idx * dh + d;
            float* khc = k_hist + (size_t)c * 3;
            float kh0 = khc[0], kh1 = khc[1], kh2 = khc[2];
            float kx = K_raw[c];
            const float* kwc = k_conv_w + (size_t)c * 4;
            float kacc = kwc[0] * kh0 + kwc[1] * kh1 + kwc[2] * kh2 + kwc[3] * kx;
            if (k_dyn) kacc += kd0 * kx + kd1 * kh2 + kd2 * kh1 + kd3 * kh0;
            khc[0] = kh1; khc[1] = kh2; khc[2] = kx;
            float kout = (kx + kacc) * sc;
            kv_v[p] = kout;
            kss += kout * kout;

            float* vhc = v_hist + (size_t)c * 3;
            float vh0 = vhc[0], vh1 = vhc[1], vh2 = vhc[2];
            float vx = V_raw[c];
            const float* vwc = v_conv_w + (size_t)c * 4;
            float vacc = vwc[0] * vh0 + vwc[1] * vh1 + vwc[2] * vh2 + vwc[3] * vx;
            if (v_dyn) vacc += vd0 * vx + vd1 * vh2 + vd2 * vh1 + vd3 * vh0;
            vhc[0] = vh1; vhc[1] = vh2; vhc[2] = vx;
            float vout = (vx + vacc) * sc;
            vv_v[p] = vout;
            vss += vout * vout;
        }
        kss = warp_sum32(kss);
        vss = warp_sum32(vss);
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
}

extern "C" __global__ void inkling_attn_decode_gated_k(const float* __restrict__ Q,
                                                       const float* __restrict__ R,
                                                       const float* __restrict__ sinks,
                                                       const float* __restrict__ rel_bias,
                                                       const __half* __restrict__ k_cache,
                                                       const __half* __restrict__ v_cache,
                                                       const float* __restrict__ gate,
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
        const float* gh = gate + (size_t)h * dh;
        oh[d0] = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        oh[d1] = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        oh[d2] = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        oh[d3] = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
    }
}

extern "C" __global__ void wo_conv_res_k(const uint8_t* __restrict__ wo,
                                         const float* __restrict__ x,
                                         const float* __restrict__ conv_w,
                                         float* __restrict__ hist,
                                         const float* __restrict__ add_src,
                                         float* __restrict__ block_dst,
                                         float* __restrict__ out,
                                         int n_embd, int q_dim) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n_embd) return;
    int nblk = q_dim >> 5;
    float o = dot_q8_0_row32(wo + (size_t)row * ((size_t)nblk * 34), x, nblk, lane);
    if (lane == 0) {
        float* hc = hist + (size_t)row * 3;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float* wc = conv_w + (size_t)row * 4;
        float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * o;
        hc[0] = h1; hc[1] = h2; hc[2] = o;
        float conv_out = (o + acc) * rsqrtf(5.0f);
        if (block_dst) {
            block_dst[row] = add_src[row];
            out[row] = conv_out;
        } else {
            out[row] = add_src[row] + conv_out;
        }
    }
}

extern "C" __global__ void moe_router_gemv_k(const float* __restrict__ w_router,
                                             const float* __restrict__ x,
                                             const float* __restrict__ bias,
                                             float* __restrict__ wtop,
                                             int* __restrict__ itop,
                                             int n_expert, int k_dim, int n_used,
                                             float route_scale) {
    __shared__ float probs[256];
    __shared__ float sel[256];
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    for (int e = wid; e < n_expert; e += nw) {
        const float* wr = w_router + (size_t)e * k_dim;
        float acc = 0.0f;
        for (int j = lane; j < k_dim; j += 32) {
            acc = fmaf(wr[j], x[j], acc);
        }
        acc = warp_sum32(acc);
        if (lane == 0) {
            float p = 1.0f / (1.0f + expf(-acc));
            probs[e] = p;
            sel[e] = bias ? (p + bias[e]) : p;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float wsum = 0.0f;
        for (int t = 0; t < n_used; ++t) {
            float best = -1e30f;
            int bi = 0;
            for (int i = 0; i < n_expert; ++i) {
                if (sel[i] > best) {
                    best = sel[i];
                    bi = i;
                }
            }
            sel[bi] = -1e30f;
            itop[t] = bi;
            wtop[t] = probs[bi];
            wsum += probs[bi];
        }
        float inv = route_scale / (wsum + 1e-20f);
        for (int t = 0; t < n_used; ++t) {
            wtop[t] *= inv;
        }
    }
}

extern "C" __global__ void moe_down_conv_add_nvfp4_k(const uint8_t* __restrict__ W,
                                                     const float* __restrict__ scales,
                                                     const float* __restrict__ act_in,
                                                     const int* __restrict__ itop,
                                                     const float* __restrict__ wtop,
                                                     const float* __restrict__ conv_w,
                                                     float* __restrict__ hist,
                                                     const float* __restrict__ ffn_inp,
                                                     float* __restrict__ cur_out,
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
    if (lane8 == 0) {
        float* hc = hist + (size_t)row * 3;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float* wc = conv_w + (size_t)row * 4;
        float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * total;
        hc[0] = h1; hc[1] = h2; hc[2] = total;
        cur_out[row] = ffn_inp[row] + (total + acc) * rsqrtf(5.0f);
    }
}

extern "C" __global__ void moe_down_conv_add_q8_0_k(const uint8_t* __restrict__ W,
                                                    const float* __restrict__ scales,
                                                    const float* __restrict__ act_in,
                                                    const int* __restrict__ itop,
                                                    const float* __restrict__ wtop,
                                                    const float* __restrict__ conv_w,
                                                    float* __restrict__ hist,
                                                    const float* __restrict__ ffn_inp,
                                                    float* __restrict__ cur_out,
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
    if (lane == 0) {
        float* hc = hist + (size_t)row * 3;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float* wc = conv_w + (size_t)row * 4;
        float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * total;
        hc[0] = h1; hc[1] = h2; hc[2] = total;
        cur_out[row] = ffn_inp[row] + (total + acc) * rsqrtf(5.0f);
    }
}

extern "C" int ink_qkvrg_gemv(hipStream_t s,
                              const uint8_t* wq, const uint8_t* wk, const uint8_t* wv,
                              const uint8_t* wrel, const uint8_t* wgate, const uint16_t* wsink,
                              const uint16_t* wkdyn, const uint16_t* wvdyn,
                              const float* inp0, const float* inp1, const float* inp2,
                              float* q_out, float* k_out, float* v_out,
                              float* r_out, float* gate_out, float* sink_out,
                              float* kdyn_out, float* vdyn_out,
                              int k_dim, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim) {
    int total = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    qkvrg_gemv_k<<<dim3((total + 7) / 8), dim3(256), 0, s>>>(
        wq, wk, wv, wrel, wgate, wsink, wkdyn, wvdyn,
        inp0, inp1, inp2, q_out, k_out, v_out, r_out, gate_out, sink_out, kdyn_out, vdyn_out,
        k_dim, q_dim, kv_dim, r_dim, n_head, dyn_dim);
    return 0;
}

extern "C" int ink_kv_conv_qkvr_store(hipStream_t s,
                                      float* Q, const float* K_raw, const float* V_raw, float* R,
                                      const float* k_conv_w, const float* v_conv_w,
                                      const float* k_dyn, const float* v_dyn,
                                      float* k_hist, float* v_hist,
                                      const float* q_clip, const float* rel_clip,
                                      __half* k_cache, __half* v_cache,
                                      const int* d_pos, int is_swa, int n_head, int n_head_kv,
                                      int dh, int d_rel, int max_ctx,
                                      float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_k<<<dim3(n_head + n_head_kv), dim3(32), 0, s>>>(
        Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist,
        q_clip, rel_clip, k_cache, v_cache, d_pos, is_swa, n_head, n_head_kv,
        dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}

extern "C" int ink_attn_decode_gated(hipStream_t s,
                                     const float* Q, const float* R, const float* sinks, const float* rel_bias,
                                     const __half* k_cache, const __half* v_cache, const float* gate,
                                     const int* d_pos, float* out,
                                     int n_head, int n_head_kv, int dh, int d_rel,
                                     int rel_extent, int swa_window, int is_swa, int max_ctx) {
    inkling_attn_decode_gated_k<<<dim3(n_head), dim3(256), 0, s>>>(
        Q, R, sinks, rel_bias, k_cache, v_cache, gate, d_pos, out,
        n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" int ink_wo_conv_res(hipStream_t s,
                               const uint8_t* wo, const float* x, const float* conv_w,
                               float* hist, const float* add_src, float* block_dst, float* out,
                               int n_embd, int q_dim) {
    wo_conv_res_k<<<dim3((n_embd + 7) / 8), dim3(256), 0, s>>>(
        wo, x, conv_w, hist, add_src, block_dst, out, n_embd, q_dim);
    return 0;
}

extern "C" int ink_moe_router_fused(hipStream_t s,
                                    const float* w_router, const float* x, const float* bias,
                                    float* wtop, int* itop,
                                    int n_expert, int k_dim, int n_used, float route_scale) {
    moe_router_gemv_k<<<dim3(1), dim3(256), 0, s>>>(
        w_router, x, bias, wtop, itop, n_expert, k_dim, n_used, route_scale);
    return 0;
}

extern "C" int ink_moe_down_conv_add_nvfp4(hipStream_t s,
                                           const uint8_t* W, const float* scales, const float* act_in,
                                           const int* itop, const float* wtop,
                                           const float* conv_w, float* hist,
                                           const float* ffn_inp, float* cur_out,
                                           int n_embd, int n_ff, int n_used) {
    moe_down_conv_add_nvfp4_k<<<dim3((n_embd + 15) / 16), dim3(128), 0, s>>>(
        W, scales, act_in, itop, wtop, conv_w, hist, ffn_inp, cur_out, n_embd, n_ff, n_used);
    return 0;
}

extern "C" __global__ void mix_rmsnorm_k(const float* __restrict__ blocks,
                                         int n_blocks,
                                         const float* __restrict__ cur,
                                         const float* __restrict__ query,
                                         const float* __restrict__ w0,
                                         const float* __restrict__ w1,
                                         const float* __restrict__ w2,
                                         float* __restrict__ out,
                                         int n_embd, float eps) {
    int q = blockIdx.x;
    const float* qv = query + (size_t)q * n_embd;
    float* ov = out + (size_t)q * n_embd;
    const float* nw = (q == 0) ? w0 : ((q == 1) ? w1 : w2);
    int n_states = n_blocks + 1;
    __shared__ float red[8];
    __shared__ float invs[16];
    __shared__ float scores[16];
    int wid = threadIdx.x >> 5;
    int nw_w = blockDim.x >> 5;
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
            for (int i = 0; i < nw_w; ++i) t += red[i];
            invs[s] = rsqrtf(t / (float)n_embd + 1e-6f);
        }
        __syncthreads();
        dot = warp_sum32(dot);
        if ((threadIdx.x & 31) == 0) red[wid] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int i = 0; i < nw_w; ++i) t += red[i];
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

    float out_ss = 0.0f;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_blocks; ++s) {
            acc += scores[s] * blocks[(size_t)s * n_embd + j];
        }
        acc += scores[n_blocks] * cur[j];
        ov[j] = acc;
        out_ss += acc * acc;
    }
    out_ss = warp_sum32(out_ss);
    if ((threadIdx.x & 31) == 0) red[wid] = out_ss;
    __syncthreads();
    __shared__ float inv_out;
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw_w; ++i) t += red[i];
        inv_out = rsqrtf(t / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv_out;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        ov[j] = ov[j] * io * nw[j];
    }
}

extern "C" __global__ void loop_ctrl_down_batch_k(const float* __restrict__ cur0,
                                                  const float* __restrict__ cur1,
                                                  const float* __restrict__ cur2,
                                                  const float* __restrict__ msgs,
                                                  const float* __restrict__ wd0,
                                                  const float* __restrict__ wd1,
                                                  const float* __restrict__ wd2,
                                                  float* __restrict__ h_out,
                                                  int n_embd, int rank) {
    int r = blockIdx.x;
    int c = blockIdx.y;
    if (r >= rank) return;
    const float* cur = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * n_embd;
    const float* wd = (c == 0) ? wd0 : ((c == 1) ? wd1 : wd2);
    const float* wr = wd + (size_t)r * (2 * n_embd);
    __shared__ float red_sc[8];
    __shared__ float red_sm[8];
    __shared__ float red_ac[8];
    __shared__ float red_am[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int e4 = n_embd >> 2;
    const float4* c4 = reinterpret_cast<const float4*>(cur);
    const float4* m4 = reinterpret_cast<const float4*>(msg);
    const float4* wc4 = reinterpret_cast<const float4*>(wr);
    const float4* wm4 = reinterpret_cast<const float4*>(wr + n_embd);

    float sc = 0.0f, sm = 0.0f, ac = 0.0f, am = 0.0f;
    for (int j = tid; j < e4; j += 256) {
        float4 vc = c4[j];
        float4 vm = m4[j];
        float4 wcv = wc4[j];
        float4 wmv = wm4[j];
        sc = fmaf(vc.x, vc.x, fmaf(vc.y, vc.y, fmaf(vc.z, vc.z, fmaf(vc.w, vc.w, sc))));
        sm = fmaf(vm.x, vm.x, fmaf(vm.y, vm.y, fmaf(vm.z, vm.z, fmaf(vm.w, vm.w, sm))));
        ac = fmaf(wcv.x, vc.x, fmaf(wcv.y, vc.y, fmaf(wcv.z, vc.z, fmaf(wcv.w, vc.w, ac))));
        am = fmaf(wmv.x, vm.x, fmaf(wmv.y, vm.y, fmaf(wmv.z, vm.z, fmaf(wmv.w, vm.w, am))));
    }
    sc = warp_sum32(sc);
    sm = warp_sum32(sm);
    ac = warp_sum32(ac);
    am = warp_sum32(am);
    if (lane == 0) {
        red_sc[wid] = sc;
        red_sm[wid] = sm;
        red_ac[wid] = ac;
        red_am[wid] = am;
    }
    __syncthreads();
    if (tid == 0) {
        float tsc = 0.0f, tsm = 0.0f, tac = 0.0f, tam = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            tsc += red_sc[i];
            tsm += red_sm[i];
            tac += red_ac[i];
            tam += red_am[i];
        }
        float rc = rsqrtf(tsc / (float)n_embd + 1e-6f);
        float rm = rsqrtf(tsm / (float)n_embd + 1e-6f);
        h_out[(size_t)c * rank + r] = tac * rc + tam * rm;
    }
}

extern "C" __global__ void loop_ctrl_up_norm_batch_k(const float* __restrict__ cur0,
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
                                                     int n_embd, int rank, float eps) {
    int c = blockIdx.x;
    const float* cur = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * n_embd;
    const float* h = h_in + (size_t)c * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    float* out = (c == 0) ? out0 : ((c == 1) ? out1 : out2);

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
    ss = warp_sum32(ss);
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nwarps; ++i) t += red[i];
        inv = rsqrtf(t / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        out[i] = out[i] * io * nw[i];
    }
}

extern "C" __global__ void loop_ctrl_up_fast_k(const float* __restrict__ cur0,
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
                                               int n_embd, int rank) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int i = blockIdx.x * 8 + warp;
    int c = blockIdx.y;
    if (i >= n_embd) return;
    const float* cur = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * n_embd;
    const float* h = h_in + (size_t)c * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    float* out = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const float2* wgi = reinterpret_cast<const float2*>(wg + (size_t)i * rank);
    const float2* wfi = reinterpret_cast<const float2*>(wf + (size_t)i * rank);
    const float2* h2 = reinterpret_cast<const float2*>(h);
    int r2 = rank >> 1;
    float g = 0.0f, f = 0.0f;
    for (int r = lane; r < r2; r += 32) {
        float2 hr = h2[r];
        float2 gv = wgi[r];
        float2 fv = wfi[r];
        g = fmaf(gv.x, hr.x, fmaf(gv.y, hr.y, g));
        f = fmaf(fv.x, hr.x, fmaf(fv.y, hr.y, f));
    }
    g = warp_sum32(g);
    f = warp_sum32(f);
    if (lane == 0) {
        float cv = cur[i];
        out[i] = cv - tanhf(f) * cv + tanhf(g) * msg[i];
    }
}

extern "C" __global__ void rmsnorm_multi_inplace_k(float* __restrict__ out0,
                                                   float* __restrict__ out1,
                                                   float* __restrict__ out2,
                                                   const float* __restrict__ nw0,
                                                   const float* __restrict__ nw1,
                                                   const float* __restrict__ nw2,
                                                   int n_embd, float eps) {
    int c = blockIdx.x;
    float* out = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    __shared__ float red[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
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
    ss = warp_sum32(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += red[i];
        inv = rsqrtf(t / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        int j = tid + p * 256;
        if (j < e4) {
            float4 v = reg[p];
            float4 w = w4[j];
            o4[j] = make_float4(v.x * io * w.x, v.y * io * w.y, v.z * io * w.z, v.w * io * w.w);
        }
    }
}

extern "C" int ink_mix_rmsnorm(hipStream_t s, const float* blocks, int n_blocks, const float* cur, const float* query, const float* w0, const float* w1, const float* w2, float* out, int n_embd, int n_queries, float eps) {
    return ink_mix_batch(s, blocks, n_blocks, 1, cur, query, w0, w1, w2, out, n_embd, n_queries, 1, eps);
}

extern "C" int ink_loop_ctrl_3_norm(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* wd0, const float* wd1, const float* wd2, const float* wg0, const float* wg1, const float* wg2, const float* wf0, const float* wf1, const float* wf2, const float* nw0, const float* nw1, const float* nw2, float* h_buf, float* out0, float* out1, float* out2, int n_embd, int rank, float eps) {
    loop_ctrl_down_batch_k<<<dim3(rank, 3), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank);
    loop_ctrl_up_fast_k<<<dim3((n_embd + 7) / 8, 3), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank);
    rmsnorm_multi_inplace_k<<<dim3(3), dim3(256), 0, s>>>(out0, out1, out2, nw0, nw1, nw2, n_embd, eps);
    return 0;
}

extern "C" int ink_loop_ctrl_1_norm(hipStream_t s, const float* cur, const float* msg, const float* wd, const float* wg, const float* wf, const float* nw, float* h_buf, float* out, int n_embd, int rank, float eps) {
    loop_ctrl_down_batch_k<<<dim3(rank, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank);
    loop_ctrl_up_fast_k<<<dim3((n_embd + 7) / 8, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, h_buf, wg, wg, wg, wf, wf, wf, out, out, out, n_embd, rank);
    rmsnorm_multi_inplace_k<<<dim3(1), dim3(256), 0, s>>>(out, out, out, nw, nw, nw, n_embd, eps);
    return 0;
}

extern "C" int ink_moe_down_conv_add_q8_0(hipStream_t s,
                                          const uint8_t* W, const float* scales, const float* act_in,
                                          const int* itop, const float* wtop,
                                          const float* conv_w, float* hist,
                                          const float* ffn_inp, float* cur_out,
                                          int n_embd, int n_ff, int n_used) {
    moe_down_conv_add_q8_0_k<<<dim3((n_embd + 7) / 8), dim3(256), 0, s>>>(
        W, scales, act_in, itop, wtop, conv_w, hist, ffn_inp, cur_out, n_embd, n_ff, n_used);
    return 0;
}
