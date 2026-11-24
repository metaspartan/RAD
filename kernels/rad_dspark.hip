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
    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q0a = qs[p0], q0b = qs[p0 + 1], q1a = qs[p1], q1b = qs[p1 + 1];

    for (int idx = wid; idx < total_kv; idx += nw) {
        float dot, v0, v1, v2, v3;
        if (idx < (ctx_len - t_start)) {
            int t = t_start + idx;
            int slot = is_swa ? (t % max_ctx) : t;
            const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)slot * dh);
            float2 k0 = __half22float2(kt2[lane]);
            float2 k1 = __half22float2(kt2[lane + 32]);
            dot = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
            const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)slot * dh);
            float2 vf0 = __half22float2(vt2[lane]);
            float2 vf1 = __half22float2(vt2[lane + 32]);
            v0 = vf0.x; v1 = vf0.y; v2 = vf1.x; v3 = vf1.y;
        } else {
            int bi = idx - (ctx_len - t_start);
            const float* kt = K_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
            dot = q0a * kt[p0] + q0b * kt[p0 + 1] + q1a * kt[p1] + q1b * kt[p1 + 1];
            const float* vt = V_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
            v0 = vt[p0]; v1 = vt[p0 + 1]; v2 = vt[p1]; v3 = vt[p1 + 1];
        }
        dot = ds_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;
        o0 = fmaf(p, v0, o0 * alpha);
        o1 = fmaf(p, v1, o1 * alpha);
        o2 = fmaf(p, v2, o2 * alpha);
        o3 = fmaf(p, v3, o3 * alpha);
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    sm_o[wid][p0] = o0; sm_o[wid][p0 + 1] = o1; sm_o[wid][p1] = o2; sm_o[wid][p1 + 1] = o3;
    __syncthreads();

    if (wid == 0) {
        float gm = sm_m[0];
        for (int w = 1; w < nw; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
        int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
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
static __device__ float g_ds_split_buf[8 * 32 * 8 * 130];

extern "C" __global__ void dspark_attn_split8_k(const float* __restrict__ Q,
                                                const float* __restrict__ K_blk,
                                                const float* __restrict__ V_blk,
                                                const __half* __restrict__ kc,
                                                const __half* __restrict__ vc,
                                                const int* __restrict__ d_ctx_len,
                                                float* __restrict__ split_buf,
                                                int K_len, int n_head, int n_head_kv,
                                                int dh, int swa_window, int is_swa,
                                                int max_ctx) {
    int h = blockIdx.x;
    int sp = blockIdx.y;
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int ctx_len = *d_ctx_len;
    int t_start = is_swa ? max(0, ctx_len - swa_window) : 0;
    int ctx_cnt = ctx_len - t_start;
    int total_kv = ctx_cnt + K_len;

    int chunk = (total_kv + 7) >> 3;
    int s_beg = sp * chunk;
    int s_end = min(total_kv, s_beg + chunk);

    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q_scale = rsqrtf((float)dh);
    float q0a = 0.0f, q0b = 0.0f, q1a = 0.0f, q1b = 0.0f;
    if (wid < K_len) {
        const float* qh = Q + ((size_t)wid * n_head + h) * dh;
        q0a = qh[p0] * q_scale;
        q0b = qh[p0 + 1] * q_scale;
        q1a = qh[p1] * q_scale;
        q1b = qh[p1 + 1] * q_scale;
    }

    const __half* kc_h = kc + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = vc + (size_t)kv_h * max_ctx * dh;

    float m = -1e30f;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    __shared__ float2 sm_k0[8][32];
    __shared__ float2 sm_k1[8][32];
    __shared__ float2 sm_v0[8][32];
    __shared__ float2 sm_v1[8][32];

    for (int base = s_beg; base < s_end; base += 8) {
        int idx = base + wid;
        if (idx < s_end) {
            if (idx < ctx_cnt) {
                int t = t_start + idx;
                int slot = is_swa ? (t % max_ctx) : t;
                const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)slot * dh);
                const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)slot * dh);
                sm_k0[wid][lane] = __half22float2(kt2[lane]);
                sm_k1[wid][lane] = __half22float2(kt2[lane + 32]);
                sm_v0[wid][lane] = __half22float2(vt2[lane]);
                sm_v1[wid][lane] = __half22float2(vt2[lane + 32]);
            } else {
                int bi = idx - ctx_cnt;
                const float* kt = K_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
                const float* vt = V_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
                sm_k0[wid][lane] = make_float2(kt[p0], kt[p0 + 1]);
                sm_k1[wid][lane] = make_float2(kt[p1], kt[p1 + 1]);
                sm_v0[wid][lane] = make_float2(vt[p0], vt[p0 + 1]);
                sm_v1[wid][lane] = make_float2(vt[p1], vt[p1 + 1]);
            }
        }
        __syncthreads();
        int n_tile = min(8, s_end - base);
        if (wid < K_len) {
            for (int j = 0; j < n_tile; ++j) {
                float2 k0 = sm_k0[j][lane];
                float2 k1 = sm_k1[j][lane];
                float dot = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
                float score = __shfl(ds_warp_sum32(dot), 0, 32);
                float m_new = fmaxf(m, score);
                float alpha = (m > -1e20f) ? expf(m - m_new) : 0.0f;
                float p = expf(score - m_new);
                l = l * alpha + p;
                m = m_new;
                float2 v0 = sm_v0[j][lane];
                float2 v1 = sm_v1[j][lane];
                o0 = fmaf(p, v0.x, o0 * alpha);
                o1 = fmaf(p, v0.y, o1 * alpha);
                o2 = fmaf(p, v1.x, o2 * alpha);
                o3 = fmaf(p, v1.y, o3 * alpha);
            }
        }
        __syncthreads();
    }

    if (wid < K_len) {
        float* dst = split_buf + (((size_t)h * K_len + wid) * 8 + sp) * 130;
        dst[p0] = o0;
        dst[p0 + 1] = o1;
        dst[p1] = o2;
        dst[p1 + 1] = o3;
        if (lane == 0) {
            dst[128] = m;
            dst[129] = l;
        }
    }
}

extern "C" __global__ void dspark_attn_split_reduce_k(const float* __restrict__ split_buf,
                                                      float* __restrict__ out,
                                                      int K_len, int n_head, int n_splits) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int lane = threadIdx.x;
    const float* base = split_buf + ((size_t)h * K_len + bt) * n_splits * 130;
    float gm = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        gm = fmaxf(gm, base[sp * 130 + 128]);
    }
    float gl = 0.0f;
    float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* sb = base + sp * 130;
        float ms = sb[128];
        float ls = sb[129];
        if (ls > 0.0f) {
            float sc = expf(ms - gm);
            gl += ls * sc;
            go0 = fmaf(sb[d0], sc, go0);
            go1 = fmaf(sb[d1], sc, go1);
            go2 = fmaf(sb[d2], sc, go2);
            go3 = fmaf(sb[d3], sc, go3);
        }
    }
    float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    float* oh = out + ((size_t)bt * n_head + h) * 128;
    oh[d0] = go0 * inv_l;
    oh[d1] = go1 * inv_l;
    oh[d2] = go2 * inv_l;
    oh[d3] = go3 * inv_l;
}

extern "C" __global__ void dspark_attn_gqa2_split8_k(const float* __restrict__ Q,
                                                     const float* __restrict__ K_blk,
                                                     const float* __restrict__ V_blk,
                                                     const __half* __restrict__ kc,
                                                     const __half* __restrict__ vc,
                                                     const int* __restrict__ d_ctx_len,
                                                     float* __restrict__ split_buf,
                                                     int K_len, int n_head, int n_head_kv,
                                                     int dh, int swa_window, int is_swa,
                                                     int max_ctx) {
    int h_pair = blockIdx.x;
    int h0 = h_pair * 2;
    int h1 = h0 + 1;
    int sp = blockIdx.y;
    int kv_h = h0 / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int ctx_len = *d_ctx_len;
    int t_start = is_swa ? max(0, ctx_len - swa_window) : 0;
    int ctx_cnt = ctx_len - t_start;
    int total_kv = ctx_cnt + K_len;

    int chunk = (total_kv + 7) >> 3;
    int s_beg = sp * chunk;
    int s_end = min(total_kv, s_beg + chunk);

    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q_scale = rsqrtf((float)dh);
    float q0a = 0.0f, q0b = 0.0f, q1a = 0.0f, q1b = 0.0f;
    float r0a = 0.0f, r0b = 0.0f, r1a = 0.0f, r1b = 0.0f;
    if (wid < K_len) {
        const float* qh0 = Q + ((size_t)wid * n_head + h0) * dh;
        q0a = qh0[p0] * q_scale; q0b = qh0[p0 + 1] * q_scale;
        q1a = qh0[p1] * q_scale; q1b = qh0[p1 + 1] * q_scale;
        const float* qh1 = Q + ((size_t)wid * n_head + h1) * dh;
        r0a = qh1[p0] * q_scale; r0b = qh1[p0 + 1] * q_scale;
        r1a = qh1[p1] * q_scale; r1b = qh1[p1 + 1] * q_scale;
    }

    const __half* kc_h = kc + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = vc + (size_t)kv_h * max_ctx * dh;

    float m0 = -1e30f, l0 = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float m1 = -1e30f, l1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

    __shared__ float2 sm_k0[8][32];
    __shared__ float2 sm_k1[8][32];
    __shared__ float2 sm_v0[8][32];
    __shared__ float2 sm_v1[8][32];

    for (int base = s_beg; base < s_end; base += 8) {
        int idx = base + wid;
        if (idx < s_end) {
            if (idx < ctx_cnt) {
                int t = t_start + idx;
                int slot = is_swa ? (t % max_ctx) : t;
                const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)slot * dh);
                const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)slot * dh);
                sm_k0[wid][lane] = __half22float2(kt2[lane]);
                sm_k1[wid][lane] = __half22float2(kt2[lane + 32]);
                sm_v0[wid][lane] = __half22float2(vt2[lane]);
                sm_v1[wid][lane] = __half22float2(vt2[lane + 32]);
            } else {
                int bi = idx - ctx_cnt;
                const float* kt = K_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
                const float* vt = V_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
                sm_k0[wid][lane] = make_float2(kt[p0], kt[p0 + 1]);
                sm_k1[wid][lane] = make_float2(kt[p1], kt[p1 + 1]);
                sm_v0[wid][lane] = make_float2(vt[p0], vt[p0 + 1]);
                sm_v1[wid][lane] = make_float2(vt[p1], vt[p1 + 1]);
            }
        }
        __syncthreads();
        int n_tile = min(8, s_end - base);
        if (wid < K_len) {
            for (int j = 0; j < n_tile; ++j) {
                float2 k0 = sm_k0[j][lane];
                float2 k1 = sm_k1[j][lane];
                float2 v0 = sm_v0[j][lane];
                float2 v1 = sm_v1[j][lane];

                float d0 = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
                float sc0 = __shfl(ds_warp_sum32(d0), 0, 32);
                float mn0 = fmaxf(m0, sc0);
                float al0 = (m0 > -1e20f) ? expf(m0 - mn0) : 0.0f;
                float pv0 = expf(sc0 - mn0);
                l0 = l0 * al0 + pv0; m0 = mn0;
                o0 = fmaf(pv0, v0.x, o0 * al0); o1 = fmaf(pv0, v0.y, o1 * al0);
                o2 = fmaf(pv0, v1.x, o2 * al0); o3 = fmaf(pv0, v1.y, o3 * al0);

                float d1 = r0a * k0.x + r0b * k0.y + r1a * k1.x + r1b * k1.y;
                float sc1 = __shfl(ds_warp_sum32(d1), 0, 32);
                float mn1 = fmaxf(m1, sc1);
                float al1 = (m1 > -1e20f) ? expf(m1 - mn1) : 0.0f;
                float pv1 = expf(sc1 - mn1);
                l1 = l1 * al1 + pv1; m1 = mn1;
                u0 = fmaf(pv1, v0.x, u0 * al1); u1 = fmaf(pv1, v0.y, u1 * al1);
                u2 = fmaf(pv1, v1.x, u2 * al1); u3 = fmaf(pv1, v1.y, u3 * al1);
            }
        }
        __syncthreads();
    }

    if (wid < K_len) {
        float* dst0 = split_buf + (((size_t)h0 * K_len + wid) * 8 + sp) * 130;
        dst0[p0] = o0; dst0[p0 + 1] = o1; dst0[p1] = o2; dst0[p1 + 1] = o3;
        if (lane == 0) { dst0[128] = m0; dst0[129] = l0; }
        float* dst1 = split_buf + (((size_t)h1 * K_len + wid) * 8 + sp) * 130;
        dst1[p0] = u0; dst1[p0 + 1] = u1; dst1[p1] = u2; dst1[p1 + 1] = u3;
        if (lane == 0) { dst1[128] = m1; dst1[129] = l1; }
    }
}

struct ds_q8_1_blk {
    float d;
    int32_t s;
    int32_t q[8];
};

__device__ __forceinline__ float ds_warp_max32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
    return v;
}

__device__ __forceinline__ int32_t ds_warp_sum32_i(int32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ int32_t ds_pack4_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int32_t)(uint8_t)a) | (((int32_t)(uint8_t)b) << 8) | (((int32_t)(uint8_t)c) << 16) | (((int32_t)(uint8_t)d) << 24);
}

extern "C" __global__ void dspark_attn_split_reduce_q8_k(const float* __restrict__ split_buf,
                                                         ds_q8_1_blk* __restrict__ out_q8,
                                                         int K_len, int n_head, int n_splits) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = wid * 32 + lane;
    const float* base = split_buf + ((size_t)h * K_len + bt) * n_splits * 130;
    float gm = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        gm = fmaxf(gm, base[sp * 130 + 128]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* sb = base + sp * 130;
        float ms = sb[128];
        float ls = sb[129];
        if (ls > 0.0f) {
            float sc = expf(ms - gm);
            gl += ls * sc;
            go = fmaf(sb[d], sc, go);
        }
    }
    float v = (gl > 0.0f) ? (go / gl) : 0.0f;
    float amax = __shfl(ds_warp_max32(fabsf(v)), 0, 32);
    float dq = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    int32_t qsum = ds_warp_sum32_i(qi);
    int q1 = __shfl_down(qi, 1, 32);
    int q2 = __shfl_down(qi, 2, 32);
    int q3 = __shfl_down(qi, 3, 32);
    ds_q8_1_blk* dst = out_q8 + ((size_t)bt * n_head + h) * 4 + wid;
    if (lane == 0) {
        dst->d = dq;
        dst->s = qsum;
    }
    if ((lane & 3) == 0) {
        dst->q[lane >> 2] = ds_pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
    }
}

extern "C" int ink_dspark_attn_q8(hipStream_t s, const float* Q, const float* K_blk, const float* V_blk, const __half* kc, const __half* vc, const int* d_ctx_len, void* out_q8, int K_len, int n_head, int n_head_kv, int dh, int swa_window, int is_swa, int max_ctx) {
    float* sbuf = nullptr;
    (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_ds_split_buf));
    if ((n_head / n_head_kv) >= 2 && (n_head & 1) == 0) {
        dspark_attn_gqa2_split8_k<<<dim3(n_head / 2, 8), dim3(256), 0, s>>>(Q, K_blk, V_blk, kc, vc, d_ctx_len, sbuf, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx);
    } else {
        dspark_attn_split8_k<<<dim3(n_head, 8), dim3(256), 0, s>>>(Q, K_blk, V_blk, kc, vc, d_ctx_len, sbuf, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx);
    }
    dspark_attn_split_reduce_q8_k<<<dim3(n_head, K_len), dim3(128), 0, s>>>(sbuf, (ds_q8_1_blk*)out_q8, K_len, n_head, 8);
    return 0;
}

extern "C" int ink_dspark_attn(hipStream_t s, const float* Q, const float* K_blk, const float* V_blk, const __half* kc, const __half* vc, const int* d_ctx_len, float* out, int K_len, int n_head, int n_head_kv, int dh, int swa_window, int is_swa, int max_ctx) {
    if (dh == 128 && K_len <= 8 && n_head <= 32) {
        float* sbuf = nullptr;
        (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_ds_split_buf));
        if ((n_head / n_head_kv) >= 2 && (n_head & 1) == 0) {
            dspark_attn_gqa2_split8_k<<<dim3(n_head / 2, 8), dim3(256), 0, s>>>(Q, K_blk, V_blk, kc, vc, d_ctx_len, sbuf, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx);
        } else {
            dspark_attn_split8_k<<<dim3(n_head, 8), dim3(256), 0, s>>>(Q, K_blk, V_blk, kc, vc, d_ctx_len, sbuf, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx);
        }
        dspark_attn_split_reduce_k<<<dim3(n_head, K_len), dim3(32), 0, s>>>(sbuf, out, K_len, n_head, 8);
        return 0;
    }
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
static __device__ float g_ds_multi_split_buf[4 * 8 * 32 * 8 * 130];

extern "C" __global__ void dspark_attn_multi_split4_k(const float* __restrict__ Q,
                                                      const float* __restrict__ K_blk,
                                                      const float* __restrict__ V_blk,
                                                      const __half* __restrict__ k_cache_pool,
                                                      const __half* __restrict__ v_cache_pool,
                                                      const int* __restrict__ d_seq_ids,
                                                      const int* __restrict__ d_ctx_lens,
                                                      float* __restrict__ split_buf,
                                                      int K_len, int n_head, int n_head_kv,
                                                      int dh, int swa_window, int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int u = blockIdx.y;
    int sp = blockIdx.z;
    int seq = d_seq_ids[u];
    int ctx_len = d_ctx_lens[u];
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int t_start = is_swa ? max(0, ctx_len - swa_window) : 0;
    int ctx_cnt = ctx_len - t_start;
    int total_kv = ctx_cnt + K_len;

    int chunk = (total_kv + 3) >> 2;
    int s_beg = sp * chunk;
    int s_end = min(total_kv, s_beg + chunk);

    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q_scale = rsqrtf((float)dh);
    float q0a = 0.0f, q0b = 0.0f, q1a = 0.0f, q1b = 0.0f;
    if (wid < K_len) {
        int bt = u * K_len + wid;
        const float* qh = Q + ((size_t)bt * n_head + h) * dh;
        q0a = qh[p0] * q_scale;
        q0b = qh[p0 + 1] * q_scale;
        q1a = qh[p1] * q_scale;
        q1b = qh[p1 + 1] * q_scale;
    }

    size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
    const __half* kc_h = k_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;

    float m = -1e30f;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    __shared__ float2 sm_k0[8][32];
    __shared__ float2 sm_k1[8][32];
    __shared__ float2 sm_v0[8][32];
    __shared__ float2 sm_v1[8][32];

    for (int base = s_beg; base < s_end; base += 8) {
        int idx = base + wid;
        if (idx < s_end) {
            if (idx < ctx_cnt) {
                int t = t_start + idx;
                int slot = is_swa ? (t % max_ctx) : t;
                const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)slot * dh);
                const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)slot * dh);
                sm_k0[wid][lane] = __half22float2(kt2[lane]);
                sm_k1[wid][lane] = __half22float2(kt2[lane + 32]);
                sm_v0[wid][lane] = __half22float2(vt2[lane]);
                sm_v1[wid][lane] = __half22float2(vt2[lane + 32]);
            } else {
                int bi = u * K_len + (idx - ctx_cnt);
                const float* kt = K_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
                const float* vt = V_blk + ((size_t)bi * n_head_kv + kv_h) * dh;
                sm_k0[wid][lane] = make_float2(kt[p0], kt[p0 + 1]);
                sm_k1[wid][lane] = make_float2(kt[p1], kt[p1 + 1]);
                sm_v0[wid][lane] = make_float2(vt[p0], vt[p0 + 1]);
                sm_v1[wid][lane] = make_float2(vt[p1], vt[p1 + 1]);
            }
        }
        __syncthreads();
        int n_tile = min(8, s_end - base);
        if (wid < K_len) {
            for (int j = 0; j < n_tile; ++j) {
                float2 k0 = sm_k0[j][lane];
                float2 k1 = sm_k1[j][lane];
                float dot = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
                float score = __shfl(ds_warp_sum32(dot), 0, 32);
                float m_new = fmaxf(m, score);
                float alpha = (m > -1e20f) ? expf(m - m_new) : 0.0f;
                float p = expf(score - m_new);
                l = l * alpha + p;
                m = m_new;
                float2 v0 = sm_v0[j][lane];
                float2 v1 = sm_v1[j][lane];
                o0 = fmaf(p, v0.x, o0 * alpha);
                o1 = fmaf(p, v0.y, o1 * alpha);
                o2 = fmaf(p, v1.x, o2 * alpha);
                o3 = fmaf(p, v1.y, o3 * alpha);
            }
        }
        __syncthreads();
    }

    if (wid < K_len) {
        int bt = u * K_len + wid;
        float* dst = split_buf + (((size_t)bt * n_head + h) * 4 + sp) * 130;
        dst[p0] = o0;
        dst[p0 + 1] = o1;
        dst[p1] = o2;
        dst[p1 + 1] = o3;
        if (lane == 0) {
            dst[128] = m;
            dst[129] = l;
        }
    }
}

extern "C" __global__ void dspark_attn_multi_split_reduce_k(const float* __restrict__ split_buf,
                                                            float* __restrict__ out,
                                                            int n_head, int n_splits) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int lane = threadIdx.x;
    const float* base = split_buf + ((size_t)bt * n_head + h) * n_splits * 130;
    float gm = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        gm = fmaxf(gm, base[sp * 130 + 128]);
    }
    float gl = 0.0f;
    float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* sb = base + sp * 130;
        float ms = sb[128];
        float ls = sb[129];
        if (ls > 0.0f) {
            float sc = expf(ms - gm);
            gl += ls * sc;
            go0 = fmaf(sb[d0], sc, go0);
            go1 = fmaf(sb[d1], sc, go1);
            go2 = fmaf(sb[d2], sc, go2);
            go3 = fmaf(sb[d3], sc, go3);
        }
    }
    float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    float* oh = out + ((size_t)bt * n_head + h) * 128;
    oh[d0] = go0 * inv_l;
    oh[d1] = go1 * inv_l;
    oh[d2] = go2 * inv_l;
    oh[d3] = go3 * inv_l;
}

extern "C" int ink_dspark_attn_multi(hipStream_t s, const float* Q, const float* K_blk, const float* V_blk, const __half* kc_pool, const __half* vc_pool, const int* d_seq_ids, const int* d_ctx_lens, float* out, int n_users, int K_len, int n_head, int n_head_kv, int dh, int swa_window, int is_swa, int max_ctx) {
    if (dh == 128 && n_users <= 8 && K_len <= 8 && n_head <= 32) {
        float* sbuf = nullptr;
        (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_ds_multi_split_buf));
        dspark_attn_multi_split4_k<<<dim3(n_head, n_users, 4), dim3(256), 0, s>>>(Q, K_blk, V_blk, kc_pool, vc_pool, d_seq_ids, d_ctx_lens, sbuf, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx);
        dspark_attn_multi_split_reduce_k<<<dim3(n_head, n_users * K_len), dim3(32), 0, s>>>(sbuf, out, n_head, 4);
        return 0;
    }
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
