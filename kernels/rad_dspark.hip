#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float ds_bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ float ds_warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 32);
    }
    return v;
}

extern "C" __global__ void dspark_fuse_k(const float* __restrict__ feats,
                                         const float* __restrict__ fuse_w,
                                         float* __restrict__ out,
                                         int n_embd, int use_softmax) {
    int row = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    const float* fw = fuse_w + row * 4;
    float w0, w1, w2, w3;
    if (use_softmax) {
        float mx = fmaxf(fmaxf(fw[0], fw[1]), fmaxf(fw[2], fw[3]));
        float e0 = expf(fw[0] - mx), e1 = expf(fw[1] - mx), e2 = expf(fw[2] - mx), e3 = expf(fw[3] - mx);
        float inv = 1.0f / (e0 + e1 + e2 + e3);
        w0 = e0 * inv; w1 = e1 * inv; w2 = e2 * inv; w3 = e3 * inv;
    } else {
        w0 = fw[0]; w1 = fw[1]; w2 = fw[2]; w3 = fw[3];
    }
    out[(size_t)row * n_embd + j] = w0 * feats[j] + w1 * feats[n_embd + j] + w2 * feats[2 * n_embd + j] + w3 * feats[3 * n_embd + j];
}

extern "C" __global__ void dspark_rope_norm_ctx_k(const float* __restrict__ K_in,
                                                  const float* __restrict__ V_in,
                                                  const float* __restrict__ k_norm_w,
                                                  __half* __restrict__ k_cache,
                                                  __half* __restrict__ v_cache,
                                                  const int* __restrict__ d_pos,
                                                  int max_ctx, int n_head_kv, int dh,
                                                  float freq_base, int is_swa) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos = *d_pos;
    int slot = is_swa ? (pos % max_ctx) : pos;
    const float* kh = K_in + (size_t)h * dh;
    const float* vh = V_in + (size_t)h * dh;
    float kv[4];
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int d = lane + p * 32;
        float v = kh[d];
        kv[p] = v;
        ss += v * v;
    }
    ss = ds_warp_sum32(ss);
    float inv = rsqrtf(__shfl(ss, 0, 32) / (float)dh + 1e-6f);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int d = lane + p * 32;
        kv[p] = kv[p] * inv * k_norm_w[d];
    }
    __half* kc = k_cache + ((size_t)h * max_ctx + slot) * dh;
    __half* vc = v_cache + ((size_t)h * max_ctx + slot) * dh;
    int half_dh = dh >> 1;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        int d = lane + p * 32;
        float theta = (float)pos * powf(freq_base, -2.0f * (float)d / (float)dh);
        float c = cosf(theta);
        float s = sinf(theta);
        float x0 = kv[p];
        float x1 = kv[p + 2];
        kc[d] = __float2half(x0 * c - x1 * s);
        kc[d + half_dh] = __float2half(x0 * s + x1 * c);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int d = lane + p * 32;
        vc[d] = __float2half(vh[d]);
    }
}

extern "C" __global__ void dspark_rope_norm_qk_k(float* __restrict__ Q,
                                                 float* __restrict__ K,
                                                 const float* __restrict__ q_norm_w,
                                                 const float* __restrict__ k_norm_w,
                                                 const int* __restrict__ d_pos0,
                                                 int K_len, int n_head, int n_head_kv,
                                                 int dh, float freq_base) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int lane = threadIdx.x;
    int pos = (*d_pos0) + bt;
    int half_dh = dh >> 1;
    if (h < n_head) {
        float* qh = Q + ((size_t)bt * n_head + h) * dh;
        float qv[4];
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            qv[p] = qh[d];
            ss += qv[p] * qv[p];
        }
        ss = ds_warp_sum32(ss);
        float inv = rsqrtf(__shfl(ss, 0, 32) / (float)dh + 1e-6f);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            qv[p] = qv[p] * inv * q_norm_w[d];
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int d = lane + p * 32;
            float theta = (float)pos * powf(freq_base, -2.0f * (float)d / (float)dh);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = qv[p];
            float x1 = qv[p + 2];
            qh[d] = x0 * c - x1 * s;
            qh[d + half_dh] = x0 * s + x1 * c;
        }
    } else {
        int kh_idx = h - n_head;
        float* kh = K + ((size_t)bt * n_head_kv + kh_idx) * dh;
        float kv[4];
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            kv[p] = kh[d];
            ss += kv[p] * kv[p];
        }
        ss = ds_warp_sum32(ss);
        float inv = rsqrtf(__shfl(ss, 0, 32) / (float)dh + 1e-6f);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            kv[p] = kv[p] * inv * k_norm_w[d];
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int d = lane + p * 32;
            float theta = (float)pos * powf(freq_base, -2.0f * (float)d / (float)dh);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = kv[p];
            float x1 = kv[p + 2];
            kh[d] = x0 * c - x1 * s;
            kh[d + half_dh] = x0 * s + x1 * c;
        }
    }
}

extern "C" __global__ void dspark_attn_k(const float* __restrict__ Q,
                                         const float* __restrict__ K_blk,
                                         const float* __restrict__ V_blk,
                                         const __half* __restrict__ k_cache,
                                         const __half* __restrict__ v_cache,
                                         const int* __restrict__ d_ctx_len,
                                         float* __restrict__ out,
                                         int K_len, int n_head, int n_head_kv,
                                         int dh, int swa_window, int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;

    __shared__ float qs[128];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[((size_t)bt * n_head + h) * dh + d] * q_scale;
    }
    __syncthreads();

    int ctx_len = *d_ctx_len;
    int t_start = is_swa ? max(0, ctx_len - swa_window) : 0;
    int total_kv = (ctx_len - t_start) + K_len;

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    float m = -1e30f;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;

    for (int idx = wid; idx < total_kv; idx += nw) {
        float dot, v0, v1, v2, v3;
        if (idx < (ctx_len - t_start)) {
            int t = t_start + idx;
            int slot = is_swa ? (t % max_ctx) : t;
            const __half* kt = kc_h + (size_t)slot * dh;
            dot = qs[d0] * __half2float(kt[d0]) + qs[d1] * __half2float(kt[d1])
                + qs[d2] * __half2float(kt[d2]) + qs[d3] * __half2float(kt[d3]);
            const __half* vt = vc_h + (size_t)slot * dh;
            v0 = __half2float(vt[d0]); v1 = __half2float(vt[d1]);
            v2 = __half2float(vt[d2]); v3 = __half2float(vt[d3]);
        } else {
            int bi = idx - (ctx_len - t_start);
            const float* kt = K_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
            dot = qs[d0] * kt[d0] + qs[d1] * kt[d1] + qs[d2] * kt[d2] + qs[d3] * kt[d3];
            const float* vt = V_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
            v0 = vt[d0]; v1 = vt[d1]; v2 = vt[d2]; v3 = vt[d3];
        }
        dot = ds_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;
        o0 = o0 * alpha + p * v0;
        o1 = o1 * alpha + p * v1;
        o2 = o2 * alpha + p * v2;
        o3 = o3 * alpha + p * v3;
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
        float* oh = out + ((size_t)bt * n_head + h) * dh;
        oh[d0] = go0 * inv_l;
        oh[d1] = go1 * inv_l;
        oh[d2] = go2 * inv_l;
        oh[d3] = go3 * inv_l;
    }
}

extern "C" __global__ void dspark_swiglu_k(const float* __restrict__ gate,
                                           const float* __restrict__ up,
                                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

extern "C" __global__ void dspark_markov_inp_k(const float* __restrict__ state,
                                               const uint16_t* __restrict__ w1_bf16,
                                               const int* __restrict__ d_prev_tok,
                                               const float* __restrict__ hidden_i,
                                               const float* __restrict__ ctx_m,
                                               float* __restrict__ out,
                                               int n_embd, int rank, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * rank + 2 * n_embd;
    if (i >= total) return;
    int tok = *d_prev_tok;
    if (mode == 0) {
        if (i < rank) out[i] = state[i];
        else if (i < 2 * rank) out[i] = ds_bf16_to_fp32(w1_bf16[(size_t)tok * rank + (i - rank)]);
        else if (i < 2 * rank + n_embd) out[i] = hidden_i[i - 2 * rank];
        else out[i] = ctx_m[i - 2 * rank - n_embd];
    } else if (mode == 1) {
        if (i < n_embd) out[i] = hidden_i[i];
        else if (i < 2 * n_embd) out[i] = ctx_m[i - n_embd];
        else if (i < 2 * n_embd + rank) out[i] = state[i - 2 * n_embd];
        else out[i] = ds_bf16_to_fp32(w1_bf16[(size_t)tok * rank + (i - 2 * n_embd - rank)]);
    } else {
        if (i < rank) out[i] = state[i];
        else if (i < 2 * rank) out[i] = ds_bf16_to_fp32(w1_bf16[(size_t)tok * rank + (i - rank)]);
        else if (i < 2 * rank + n_embd) out[i] = ctx_m[i - 2 * rank];
        else out[i] = hidden_i[i - 2 * rank - n_embd];
    }
}

extern "C" __global__ void dspark_markov_update_k(const float* __restrict__ joint,
                                                  const float* __restrict__ bias,
                                                  float* __restrict__ state,
                                                  float* __restrict__ markov_out,
                                                  int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank) return;
    float g = 1.0f / (1.0f + expf(-(joint[i] + bias[i])));
    float c = tanhf(joint[rank + i] + bias[rank + i]);
    float s = state[i];
    state[i] = c + g * (s - c);
    markov_out[i] = tanhf(joint[2 * rank + i] + bias[2 * rank + i]);
}

extern "C" __global__ void dspark_add_bias_k(const float* __restrict__ hidden_i,
                                             const float* __restrict__ delta,
                                             const float* __restrict__ bias,
                                             float* __restrict__ out,
                                             int n_embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = hidden_i[i] + delta[i] + bias[i];
    }
}

extern "C" int ink_dspark_fuse(hipStream_t s, const float* feats, const float* fuse_w, float* out, int n_embd, int use_softmax) {
    dspark_fuse_k<<<dim3((n_embd + 255) / 256, 4), dim3(256), 0, s>>>(feats, fuse_w, out, n_embd, use_softmax);
    return 0;
}
extern "C" int ink_dspark_rope_norm_ctx(hipStream_t s, const float* K_in, const float* V_in, const float* k_norm_w, __half* kc, __half* vc, const int* d_pos, int max_ctx, int n_head_kv, int dh, float freq_base, int is_swa) {
    dspark_rope_norm_ctx_k<<<dim3(n_head_kv), dim3(32), 0, s>>>(K_in, V_in, k_norm_w, kc, vc, d_pos, max_ctx, n_head_kv, dh, freq_base, is_swa);
    return 0;
}
extern "C" int ink_dspark_rope_norm_qk(hipStream_t s, float* Q, float* K, const float* q_norm_w, const float* k_norm_w, const int* d_pos0, int K_len, int n_head, int n_head_kv, int dh, float freq_base) {
    dspark_rope_norm_qk_k<<<dim3(n_head + n_head_kv, K_len), dim3(32), 0, s>>>(Q, K, q_norm_w, k_norm_w, d_pos0, K_len, n_head, n_head_kv, dh, freq_base);
    return 0;
}
extern "C" int ink_dspark_attn(hipStream_t s, const float* Q, const float* K_blk, const float* V_blk, const __half* kc, const __half* vc, const int* d_ctx_len, float* out, int K_len, int n_head, int n_head_kv, int dh, int swa_window, int is_swa, int max_ctx) {
    dspark_attn_k<<<dim3(n_head, K_len), dim3(256), 0, s>>>(Q, K_blk, V_blk, kc, vc, d_ctx_len, out, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx);
    return 0;
}
extern "C" int ink_dspark_swiglu(hipStream_t s, const float* gate, const float* up, float* out, int n) {
    dspark_swiglu_k<<<dim3((n + 255) / 256), dim3(256), 0, s>>>(gate, up, out, n);
    return 0;
}
extern "C" int ink_dspark_markov_inp(hipStream_t s, const float* state, const uint16_t* w1_bf16, const int* d_prev_tok, const float* hidden_i, const float* ctx_m, float* out, int n_embd, int rank, int mode) {
    int tot = 2 * rank + 2 * n_embd;
    dspark_markov_inp_k<<<dim3((tot + 255) / 256), dim3(256), 0, s>>>(state, w1_bf16, d_prev_tok, hidden_i, ctx_m, out, n_embd, rank, mode);
    return 0;
}
extern "C" int ink_dspark_markov_update(hipStream_t s, const float* joint, const float* bias, float* state, float* markov_out, int rank) {
    dspark_markov_update_k<<<dim3((rank + 255) / 256), dim3(256), 0, s>>>(joint, bias, state, markov_out, rank);
    return 0;
}
extern "C" int ink_dspark_add_bias(hipStream_t s, const float* hidden_i, const float* delta, const float* bias, float* out, int n_embd) {
    dspark_add_bias_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(hidden_i, delta, bias, out, n_embd);
    return 0;
}

extern "C" __global__ void dspark_init_mask_multi_k(const float* __restrict__ mask_embd,
                                                    float* __restrict__ x,
                                                    int n_embd, int K_len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (j >= n_embd) return;
    int slot = bt % K_len;
    if (slot > 0) {
        x[(size_t)bt * n_embd + j] = mask_embd[j];
    }
}

extern "C" __global__ void dspark_scatter_anchor_k(const float* __restrict__ anchors,
                                                   float* __restrict__ x,
                                                   int n_embd, int K_len) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;
    if (j < n_embd) {
        x[(size_t)(u * K_len) * n_embd + j] = anchors[(size_t)u * n_embd + j];
    }
}

extern "C" __global__ void dspark_rope_norm_qk_multi_k(float* __restrict__ Q,
                                                       float* __restrict__ K,
                                                       const float* __restrict__ q_norm_w,
                                                       const float* __restrict__ k_norm_w,
                                                       const int* __restrict__ d_pos0,
                                                       int K_len, int n_head, int n_head_kv,
                                                       int dh, float freq_base) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int lane = threadIdx.x;
    int u = bt / K_len;
    int slot = bt % K_len;
    int pos = d_pos0[u] + slot;
    int half_dh = dh >> 1;
    if (h < n_head) {
        float* qh = Q + ((size_t)bt * n_head + h) * dh;
        float qv[4];
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            qv[p] = qh[d];
            ss += qv[p] * qv[p];
        }
        ss = ds_warp_sum32(ss);
        float inv = rsqrtf(__shfl(ss, 0, 32) / (float)dh + 1e-6f);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            qv[p] = qv[p] * inv * q_norm_w[d];
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int d = lane + p * 32;
            float theta = (float)pos * powf(freq_base, -2.0f * (float)d / (float)dh);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = qv[p];
            float x1 = qv[p + 2];
            qh[d] = x0 * c - x1 * s;
            qh[d + half_dh] = x0 * s + x1 * c;
        }
    } else {
        int kh_idx = h - n_head;
        float* kh = K + ((size_t)bt * n_head_kv + kh_idx) * dh;
        float kv[4];
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            kv[p] = kh[d];
            ss += kv[p] * kv[p];
        }
        ss = ds_warp_sum32(ss);
        float inv = rsqrtf(__shfl(ss, 0, 32) / (float)dh + 1e-6f);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            kv[p] = kv[p] * inv * k_norm_w[d];
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int d = lane + p * 32;
            float theta = (float)pos * powf(freq_base, -2.0f * (float)d / (float)dh);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = kv[p];
            float x1 = kv[p + 2];
            kh[d] = x0 * c - x1 * s;
            kh[d + half_dh] = x0 * s + x1 * c;
        }
    }
}

extern "C" __global__ void dspark_attn_multi_k(const float* __restrict__ Q,
                                               const float* __restrict__ K_blk,
                                               const float* __restrict__ V_blk,
                                               const __half* __restrict__ k_cache_pool,
                                               const __half* __restrict__ v_cache_pool,
                                               const int* __restrict__ d_seq_ids,
                                               const int* __restrict__ d_ctx_lens,
                                               float* __restrict__ out,
                                               int K_len, int n_head, int n_head_kv,
                                               int dh, int swa_window, int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int u = bt / K_len;
    int seq = d_seq_ids[u];
    int ctx_len = d_ctx_lens[u];
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;

    __shared__ float qs[128];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[((size_t)bt * n_head + h) * dh + d] * q_scale;
    }
    __syncthreads();

    int t_start = is_swa ? max(0, ctx_len - swa_window) : 0;
    int total_kv = (ctx_len - t_start) + K_len;
    size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
    const __half* kc_h = k_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;

    float m = -1e30f;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;

    for (int idx = wid; idx < total_kv; idx += nw) {
        float dot, v0, v1, v2, v3;
        if (idx < (ctx_len - t_start)) {
            int t = t_start + idx;
            int slot = is_swa ? (t % max_ctx) : t;
            const __half* kt = kc_h + (size_t)slot * dh;
            dot = qs[d0] * __half2float(kt[d0]) + qs[d1] * __half2float(kt[d1])
                + qs[d2] * __half2float(kt[d2]) + qs[d3] * __half2float(kt[d3]);
            const __half* vt = vc_h + (size_t)slot * dh;
            v0 = __half2float(vt[d0]); v1 = __half2float(vt[d1]);
            v2 = __half2float(vt[d2]); v3 = __half2float(vt[d3]);
        } else {
            int bi = u * K_len + (idx - (ctx_len - t_start));
            const float* kt = K_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
            dot = qs[d0] * kt[d0] + qs[d1] * kt[d1] + qs[d2] * kt[d2] + qs[d3] * kt[d3];
            const float* vt = V_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
            v0 = vt[d0]; v1 = vt[d1]; v2 = vt[d2]; v3 = vt[d3];
        }
        dot = ds_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;
        o0 = o0 * alpha + p * v0;
        o1 = o1 * alpha + p * v1;
        o2 = o2 * alpha + p * v2;
        o3 = o3 * alpha + p * v3;
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
        float* oh = out + ((size_t)bt * n_head + h) * dh;
        oh[d0] = go0 * inv_l;
        oh[d1] = go1 * inv_l;
        oh[d2] = go2 * inv_l;
        oh[d3] = go3 * inv_l;
    }
}

extern "C" __global__ void dspark_markov_inp_multi_k(const float* __restrict__ states,
                                                     const uint16_t* __restrict__ w1_bf16,
                                                     const int* __restrict__ d_draft_toks,
                                                     const float* __restrict__ hidden,
                                                     const float* __restrict__ ctx_m_pool,
                                                     const int* __restrict__ d_seq_ids,
                                                     float* __restrict__ out,
                                                     int n_embd, int rank, int K_len, int slot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;
    int total = 2 * rank + 2 * n_embd;
    if (i >= total) return;
    int seq = d_seq_ids[u];
    int tok = d_draft_toks[u * K_len + slot - 1];
    const float* state = states + (size_t)u * rank;
    const float* hi = hidden + (size_t)(u * K_len + slot) * n_embd;
    const float* cm = ctx_m_pool + (size_t)seq * n_embd;
    float* dst = out + (size_t)u * total;
    if (i < rank) dst[i] = state[i];
    else if (i < 2 * rank) dst[i] = ds_bf16_to_fp32(w1_bf16[(size_t)tok * rank + (i - rank)]);
    else if (i < 2 * rank + n_embd) dst[i] = hi[i - 2 * rank];
    else dst[i] = cm[i - 2 * rank - n_embd];
}

extern "C" __global__ void dspark_markov_update_multi_k(const float* __restrict__ joint,
                                                        const float* __restrict__ bias,
                                                        float* __restrict__ states,
                                                        float* __restrict__ markov_out,
                                                        int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;
    if (i >= rank) return;
    const float* j = joint + (size_t)u * 3 * rank;
    float* st = states + (size_t)u * rank;
    float* mo = markov_out + (size_t)u * rank;
    float g = 1.0f / (1.0f + expf(-(j[i] + bias[i])));
    float c = tanhf(j[rank + i] + bias[rank + i]);
    float s = st[i];
    st[i] = c + g * (s - c);
    mo[i] = tanhf(j[2 * rank + i] + bias[2 * rank + i]);
}

extern "C" __global__ void dspark_add_bias_multi_k(const float* __restrict__ hidden,
                                                   const float* __restrict__ delta,
                                                   const float* __restrict__ bias,
                                                   float* __restrict__ out,
                                                   int n_embd, int K_len, int slot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;
    if (i < n_embd) {
        const float* hi = hidden + (size_t)(u * K_len + slot) * n_embd;
        out[(size_t)u * n_embd + i] = hi[i] + delta[(size_t)u * n_embd + i] + bias[i];
    }
}

extern "C" __global__ void dspark_scatter_toks_k(const int* __restrict__ src_toks,
                                                 int* __restrict__ dst_toks,
                                                 int n_users, int K_len, int slot) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n_users) {
        dst_toks[u * K_len + slot] = src_toks[u];
    }
}

extern "C" int ink_dspark_init_multi(hipStream_t s, const float* anchors, const float* mask_embd, float* x, int n_embd, int n_users, int K_len) {
    dspark_init_mask_multi_k<<<dim3((n_embd + 255) / 256, n_users * K_len), dim3(256), 0, s>>>(mask_embd, x, n_embd, K_len);
    dspark_scatter_anchor_k<<<dim3((n_embd + 255) / 256, n_users), dim3(256), 0, s>>>(anchors, x, n_embd, K_len);
    return 0;
}
extern "C" int ink_dspark_rope_norm_qk_multi(hipStream_t s, float* Q, float* K, const float* q_norm_w, const float* k_norm_w, const int* d_pos0, int n_users, int K_len, int n_head, int n_head_kv, int dh, float freq_base) {
    dspark_rope_norm_qk_multi_k<<<dim3(n_head + n_head_kv, n_users * K_len), dim3(32), 0, s>>>(Q, K, q_norm_w, k_norm_w, d_pos0, K_len, n_head, n_head_kv, dh, freq_base);
    return 0;
}
extern "C" int ink_dspark_attn_multi(hipStream_t s, const float* Q, const float* K_blk, const float* V_blk, const __half* kc_pool, const __half* vc_pool, const int* d_seq_ids, const int* d_ctx_lens, float* out, int n_users, int K_len, int n_head, int n_head_kv, int dh, int swa_window, int is_swa, int max_ctx) {
    dspark_attn_multi_k<<<dim3(n_head, n_users * K_len), dim3(256), 0, s>>>(Q, K_blk, V_blk, kc_pool, vc_pool, d_seq_ids, d_ctx_lens, out, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx);
    return 0;
}
extern "C" int ink_dspark_markov_inp_multi(hipStream_t s, const float* states, const uint16_t* w1_bf16, const int* d_draft_toks, const float* hidden, const float* ctx_m_pool, const int* d_seq_ids, float* out, int n_embd, int rank, int n_users, int K_len, int slot) {
    int tot = 2 * rank + 2 * n_embd;
    dspark_markov_inp_multi_k<<<dim3((tot + 255) / 256, n_users), dim3(256), 0, s>>>(states, w1_bf16, d_draft_toks, hidden, ctx_m_pool, d_seq_ids, out, n_embd, rank, K_len, slot);
    return 0;
}
extern "C" int ink_dspark_markov_update_multi(hipStream_t s, const float* joint, const float* bias, float* states, float* markov_out, int rank, int n_users) {
    dspark_markov_update_multi_k<<<dim3((rank + 255) / 256, n_users), dim3(256), 0, s>>>(joint, bias, states, markov_out, rank);
    return 0;
}
extern "C" int ink_dspark_add_bias_multi(hipStream_t s, const float* hidden, const float* delta, const float* bias, float* out, int n_embd, int n_users, int K_len, int slot) {
    dspark_add_bias_multi_k<<<dim3((n_embd + 255) / 256, n_users), dim3(256), 0, s>>>(hidden, delta, bias, out, n_embd, K_len, slot);
    return 0;
}
extern "C" int ink_dspark_scatter_toks(hipStream_t s, const int* src_toks, int* dst_toks, int n_users, int K_len, int slot) {
    dspark_scatter_toks_k<<<dim3((n_users + 31) / 32), dim3(32), 0, s>>>(src_toks, dst_toks, n_users, K_len, slot);
    return 0;
}
