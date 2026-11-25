#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__global__ void rms_norm_bf16w_f16_k(
    const half* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    half* __restrict__ y,
    int D,
    float eps
) {
    int row = blockIdx.x;
    int t = threadIdx.x;
    const half* xr = x + (size_t)row * D;
    half* yr = y + (size_t)row * D;
    float ss = 0.0f;
    for (int i = t; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        ss += v * v;
    }
    __shared__ float sm[32];
    int lane = t & 31;
    int warp = t >> 5;
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, m);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
        if (lane == 0) sm[0] = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();
    float inv = sm[0];
    for (int i = t; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]) * inv * __bfloat162float(w[i]);
        yr[i] = __float2half(v);
    }
}

__global__ void clamped_swiglu_f16_k(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int N,
    float limit
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        if (u < -limit) u = -limit;
        float silu = g / (1.0f + expf(-g));
        out[i] = __float2half(silu * u);
    }
}

__global__ void moe_weighted_sum_f16_k(
    const half* __restrict__ expert_outs,
    const float* __restrict__ weights,
    const half* __restrict__ shared_out,
    half* __restrict__ dst,
    int top_k,
    int D
) {
    int mr = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        float acc = shared_out ? __half2float(shared_out[(size_t)mr * D + i]) : 0.0f;
        const float* wm = weights + (size_t)mr * top_k;
        const half* em = expert_outs + (size_t)mr * top_k * D;
        for (int k = 0; k < top_k; ++k) {
            acc += wm[k] * __half2float(em[(size_t)k * D + i]);
        }
        dst[(size_t)mr * D + i] = __float2half(acc);
    }
}

__global__ void embed_to_mhc_f16_k(
    const __nv_bfloat16* __restrict__ table,
    const int* __restrict__ tok_ids,
    float* __restrict__ streams,
    int D
) {
    int mr = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        int tid = tok_ids[mr];
        float v = __bfloat162float(table[(size_t)tid * D + i]);
        float* sr = streams + (size_t)mr * 4 * D;
        sr[0 * D + i] = v;
        sr[1 * D + i] = v;
        sr[2 * D + i] = v;
        sr[3 * D + i] = v;
    }
}

__global__ void rms_norm_f32_bf16w_f16_k(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    half* __restrict__ y,
    int D,
    float eps
) {
    int row = blockIdx.x;
    int t = threadIdx.x;
    const float* xr = x + (size_t)row * D;
    half* yr = y + (size_t)row * D;
    float ss = 0.0f;
    for (int i = t; i < D; i += blockDim.x) {
        float v = xr[i];
        ss += v * v;
    }
    __shared__ float sm[32];
    int lane = t & 31;
    int warp = t >> 5;
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, m);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
        if (lane == 0) sm[0] = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();
    float inv = sm[0];
    for (int i = t; i < D; i += blockDim.x) {
        yr[i] = __float2half(xr[i] * inv * __bfloat162float(w[i]));
    }
}

__global__ void mhc_collapse_mean_f16_k(
    const float* __restrict__ streams,
    float* __restrict__ out,
    int D
) {
    int mr = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        const float* sr = streams + (size_t)mr * 4 * D;
        out[(size_t)mr * D + i] = 0.25f * (sr[0 * D + i] + sr[1 * D + i] + sr[2 * D + i] + sr[3 * D + i]);
    }
}

__global__ void kv_scatter_f16_k(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int start_pos,
    int max_ctx,
    int head_dim
) {
    int mr = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < head_dim) {
        int p = (start_pos + mr) % max_ctx;
        k_cache[(size_t)p * head_dim + i] = k_in[(size_t)mr * head_dim + i];
        if (v_cache != k_cache) {
            v_cache[(size_t)p * head_dim + i] = v_in[(size_t)mr * head_dim + i];
        }
    }
}

__global__ void rope_yarn_f16_k(
    half* __restrict__ q,
    half* __restrict__ k,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    float theta,
    int orig_len,
    float sign
) {
    int h = blockIdx.x;
    int mr = blockIdx.y;
    int i = threadIdx.x;
    int half_r = rope_dim >> 1;
    if (i >= half_r) return;
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
    if (orig_len > 0) {
        float factor = 16.0f;
        float orig_ctx = (float)orig_len;
        float beta_fast = 32.0f;
        float beta_slow = 1.0f;
        float low = floorf((float)rope_dim * logf(orig_ctx / (beta_fast * 2.0f * 3.14159265f)) / (2.0f * logf(theta)));
        float high = ceilf((float)rope_dim * logf(orig_ctx / (beta_slow * 2.0f * 3.14159265f)) / (2.0f * logf(theta)));
        if (low < 0.0f) low = 0.0f;
        if (high > (float)(rope_dim - 1)) high = (float)(rope_dim - 1);
        float ramp = ((float)i - low) / fmaxf(high - low, 0.001f);
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        freq = (freq / factor) * ramp + freq * (1.0f - ramp);
    }
    float pos_f = (float)(start_pos + mr);
    float ang = pos_f * freq;
    float cs = cosf(ang);
    float sn = sign * sinf(ang);
    int rope_off = head_dim - rope_dim;
    if (h < n_q_heads) {
        half* p = q + ((size_t)mr * n_q_heads + h) * head_dim + rope_off;
        float x0 = __half2float(p[2 * i]);
        float x1 = __half2float(p[2 * i + 1]);
        p[2 * i] = __float2half(x0 * cs - x1 * sn);
        p[2 * i + 1] = __float2half(x0 * sn + x1 * cs);
    } else {
        int kh = h - n_q_heads;
        if (kh < n_kv_heads) {
            half* p = k + ((size_t)mr * n_kv_heads + kh) * head_dim + rope_off;
            float x0 = __half2float(p[2 * i]);
            float x1 = __half2float(p[2 * i + 1]);
            p[2 * i] = __float2half(x0 * cs - x1 * sn);
            p[2 * i + 1] = __float2half(x0 * sn + x1 * cs);
        }
    }
}

__global__ void mhc_sinkhorn_mix_f16_k(
    float* __restrict__ streams,
    const half* __restrict__ branch_out,
    const half* __restrict__ peer_branch_out,
    const float* __restrict__ h_coeff,
    int D,
    int sinkhorn_iters,
    float eps
) {
    int mr = blockIdx.y;
    __shared__ float s_post[4];
    __shared__ float s_res[16];
    int t = threadIdx.x;
    if (t == 0) {
        const float* hc = h_coeff + (size_t)mr * 24;
        for (int i = 0; i < 4; ++i) {
            s_post[i] = 2.0f / (1.0f + expf(-hc[4 + i]));
        }
        float m[4][4];
        for (int r = 0; r < 4; ++r) {
            float rmax = hc[8 + r * 4 + 0];
            for (int c = 1; c < 4; ++c) {
                float v = hc[8 + r * 4 + c];
                if (v > rmax) rmax = v;
            }
            float rs = 0.0f;
            for (int c = 0; c < 4; ++c) {
                m[r][c] = expf(hc[8 + r * 4 + c] - rmax);
                rs += m[r][c];
            }
            float inv = 1.0f / rs;
            for (int c = 0; c < 4; ++c) m[r][c] = m[r][c] * inv + eps;
        }
        for (int c = 0; c < 4; ++c) {
            float cs = m[0][c] + m[1][c] + m[2][c] + m[3][c] + eps;
            float inv = 1.0f / cs;
            for (int r = 0; r < 4; ++r) m[r][c] *= inv;
        }
        for (int it = 1; it < sinkhorn_iters; ++it) {
            for (int r = 0; r < 4; ++r) {
                float rs = m[r][0] + m[r][1] + m[r][2] + m[r][3] + eps;
                float inv = 1.0f / rs;
                for (int c = 0; c < 4; ++c) m[r][c] *= inv;
            }
            for (int c = 0; c < 4; ++c) {
                float cs = m[0][c] + m[1][c] + m[2][c] + m[3][c] + eps;
                float inv = 1.0f / cs;
                for (int r = 0; r < 4; ++r) m[r][c] *= inv;
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) s_res[r * 4 + c] = m[r][c];
        }
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + t;
    if (i < D) {
        float* sr = streams + (size_t)mr * 4 * D;
        float x0 = sr[0 * D + i];
        float x1 = sr[1 * D + i];
        float x2 = sr[2 * D + i];
        float x3 = sr[3 * D + i];
        float b = __half2float(branch_out[(size_t)mr * D + i]);
        if (peer_branch_out) {
            b += __half2float(peer_branch_out[(size_t)mr * D + i]);
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            sr[r * D + i] = s_res[0 * 4 + r] * x0 + s_res[1 * 4 + r] * x1 +
                            s_res[2 * 4 + r] * x2 + s_res[3 * 4 + r] * x3 +
                            s_post[r] * b;
        }
    }
}

__global__ void mqa_attn_sink_causal_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const half* __restrict__ comp_kv,
    const float* __restrict__ attn_sink,
    half* __restrict__ out,
    int n_heads,
    int start_pos,
    int max_ctx,
    int sliding_window,
    int comp_ratio,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int seq_len = start_pos + mr + 1;
    if (seq_len > max_ctx) seq_len = max_ctx;
    int p_start = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;
    float m_max = attn_sink ? attn_sink[h] : -1e30f;
    float l_sum = attn_sink ? 1.0f : 0.0f;
    __shared__ float wsum[8];

    if (head_dim == 512) {
        const half2* qh2 = reinterpret_cast<const half2*>(q + ((size_t)mr * n_heads + h) * 512);
        float2 qv = __half22float2(__ldg(qh2 + t));
        float a0 = 0.0f;
        float a1 = 0.0f;
        bool same_kv = (k_cache == v_cache);
        for (int p = p_start; p < seq_len; ++p) {
            const half2* kp2 = reinterpret_cast<const half2*>(k_cache + (size_t)p * 512);
            float2 kv = __half22float2(__ldg(kp2 + t));
            float dot = fmaf(qv.x, kv.x, qv.y * kv.y);
            for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
            if (lane == 0) wsum[warp] = dot;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < 8) ? wsum[lane] : 0.0f;
                for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
                if (lane == 0) wsum[0] = v * scale;
            }
            __syncthreads();
            float score = wsum[0];
            float new_max = fmaxf(m_max, score);
            float alpha = expf(m_max - new_max);
            float beta = expf(score - new_max);
            l_sum = l_sum * alpha + beta;
            m_max = new_max;
            float2 vv = same_kv ? kv : __half22float2(__ldg(reinterpret_cast<const half2*>(v_cache + (size_t)p * 512) + t));
            a0 = fmaf(a0, alpha, beta * vv.x);
            a1 = fmaf(a1, alpha, beta * vv.y);
        }
        if (comp_kv && comp_ratio > 0) {
            int n_comp = seq_len / comp_ratio;
            for (int c = 0; c < n_comp; ++c) {
                const half2* ck2 = reinterpret_cast<const half2*>(comp_kv + (size_t)c * 512);
                float2 kv = __half22float2(__ldg(ck2 + t));
                float dot = fmaf(qv.x, kv.x, qv.y * kv.y);
                for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
                if (lane == 0) wsum[warp] = dot;
                __syncthreads();
                if (warp == 0) {
                    float v = (lane < 8) ? wsum[lane] : 0.0f;
                    for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
                    if (lane == 0) wsum[0] = v * scale;
                }
                __syncthreads();
                float score = wsum[0];
                float new_max = fmaxf(m_max, score);
                float alpha = expf(m_max - new_max);
                float beta = expf(score - new_max);
                l_sum = l_sum * alpha + beta;
                m_max = new_max;
                a0 = fmaf(a0, alpha, beta * kv.x);
                a1 = fmaf(a1, alpha, beta * kv.y);
            }
        }
        float inv = 1.0f / fmaxf(l_sum, 1e-20f);
        half2* oh2 = reinterpret_cast<half2*>(out + ((size_t)mr * n_heads + h) * 512);
        oh2[t] = __float22half2_rn(make_float2(a0 * inv, a1 * inv));
        return;
    }

    const half* qh = q + ((size_t)mr * n_heads + h) * head_dim;
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_acc = smem + head_dim;
    for (int d = t; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(qh[d]);
        s_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int p = p_start; p < seq_len; ++p) {
        const half* kp = k_cache + (size_t)p * head_dim;
        float dot = 0.0f;
        for (int d = t; d < head_dim; d += blockDim.x) {
            dot += s_q[d] * __half2float(kp[d]);
        }
        for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
        if (lane == 0) wsum[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < 8) ? wsum[lane] : 0.0f;
            for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
            if (lane == 0) wsum[0] = v * scale;
        }
        __syncthreads();
        float score = wsum[0];
        __syncthreads();
        float new_max = fmaxf(m_max, score);
        float alpha = expf(m_max - new_max);
        float beta = expf(score - new_max);
        l_sum = l_sum * alpha + beta;
        m_max = new_max;
        const half* vp = v_cache + (size_t)p * head_dim;
        for (int d = t; d < head_dim; d += blockDim.x) {
            s_acc[d] = s_acc[d] * alpha + beta * __half2float(vp[d]);
        }
    }
    float inv = 1.0f / fmaxf(l_sum, 1e-20f);
    half* oh = out + ((size_t)mr * n_heads + h) * head_dim;
    for (int d = t; d < head_dim; d += blockDim.x) {
        oh[d] = __float2half(s_acc[d] * inv);
    }
}

__global__ void csa2_compress_pair_f16_k(
    const half* __restrict__ raw_kv,
    const half* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ norm_w,
    half* __restrict__ comp_out,
    int c_start,
    int head_dim,
    int rope_dim,
    float theta,
    int orig_len,
    float eps
) {
    int c = c_start + blockIdx.x;
    int t = threadIdx.x;
    const half* k0 = raw_kv + (size_t)(2 * c) * head_dim;
    const half* k1 = raw_kv + (size_t)(2 * c + 1) * head_dim;
    const half* g0 = raw_gate + (size_t)(2 * c) * head_dim;
    const half* g1 = raw_gate + (size_t)(2 * c + 1) * head_dim;
    __shared__ float s_v[512];
    __shared__ float sm[32];
    float ss = 0.0f;
    for (int i = t; i < head_dim; i += blockDim.x) {
        float sc0 = __half2float(g0[i]);
        float sc1 = __half2float(g1[i]);
        float mx = fmaxf(sc0, sc1);
        float e0 = expf(sc0 - mx);
        float e1 = expf(sc1 - mx);
        float w0 = e0 / (e0 + e1);
        float v = w0 * __half2float(k0[i]) + (1.0f - w0) * __half2float(k1[i]);
        s_v[i] = v;
        ss += v * v;
    }
    int lane = t & 31;
    int warp = t >> 5;
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, m);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
        if (lane == 0) sm[0] = rsqrtf(v / (float)head_dim + eps);
    }
    __syncthreads();
    float inv = sm[0];
    for (int i = t; i < head_dim; i += blockDim.x) {
        s_v[i] = s_v[i] * inv * __bfloat162float(norm_w[i]);
    }
    __syncthreads();
    int rope_off = head_dim - rope_dim;
    int half_r = rope_dim >> 1;
    if (t < half_r) {
        float freq = 1.0f / powf(theta, (float)(2 * t) / (float)rope_dim);
        if (orig_len > 0) {
            float factor = 16.0f;
            float orig_ctx = (float)orig_len;
            float low = floorf((float)rope_dim * logf(orig_ctx / (32.0f * 2.0f * 3.14159265f)) / (2.0f * logf(theta)));
            float high = ceilf((float)rope_dim * logf(orig_ctx / (1.0f * 2.0f * 3.14159265f)) / (2.0f * logf(theta)));
            if (low < 0.0f) low = 0.0f;
            if (high > (float)(rope_dim - 1)) high = (float)(rope_dim - 1);
            float ramp = ((float)t - low) / fmaxf(high - low, 0.001f);
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            freq = (freq / factor) * ramp + freq * (1.0f - ramp);
        }
        float ang = (float)(2 * c) * freq;
        float cs = cosf(ang);
        float sn = sinf(ang);
        float x0 = s_v[rope_off + 2 * t];
        float x1 = s_v[rope_off + 2 * t + 1];
        s_v[rope_off + 2 * t] = x0 * cs - x1 * sn;
        s_v[rope_off + 2 * t + 1] = x0 * sn + x1 * cs;
    }
    __syncthreads();
    half* dst = comp_out + (size_t)c * head_dim;
    for (int i = t; i < head_dim; i += blockDim.x) {
        dst[i] = __float2half(s_v[i]);
    }
}

__global__ void wo_a_gather_group_f16_k(
    const half* __restrict__ attn_out,
    half* __restrict__ grp_in,
    int n_heads_local,
    int heads_per_grp,
    int head_dim,
    int gi
) {
    int mr = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grp_dim = heads_per_grp * head_dim;
    if (idx < grp_dim) {
        int src_off = ((size_t)mr * n_heads_local + gi * heads_per_grp) * head_dim + idx;
        grp_in[(size_t)mr * grp_dim + idx] = attn_out[src_off];
    }
}

__global__ void wo_a_scatter_group_f16_k(
    const half* __restrict__ grp_out,
    half* __restrict__ wo_mid,
    int groups_local,
    int o_lora_rank,
    int gi
) {
    int mr = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < o_lora_rank) {
        wo_mid[((size_t)mr * groups_local + gi) * o_lora_rank + idx] = grp_out[(size_t)mr * o_lora_rank + idx];
    }
}

extern "C" {

void rad_cuda_rms_norm_bf16w_f16(
    const half* x,
    const __nv_bfloat16* w,
    half* y,
    int M,
    int D,
    float eps,
    cudaStream_t stream
) {
    rms_norm_bf16w_f16_k<<<M, 256, 0, stream>>>(x, w, y, D, eps);
}

void rad_cuda_clamped_swiglu_f16(
    const half* gate,
    const half* up,
    half* out,
    int N,
    float limit,
    cudaStream_t stream
) {
    clamped_swiglu_f16_k<<<(N + 255) / 256, 256, 0, stream>>>(gate, up, out, N, limit);
}

void rad_cuda_moe_weighted_sum_f16(
    const half* expert_outs,
    const float* weights,
    const half* shared_out,
    half* dst,
    int M,
    int top_k,
    int D,
    cudaStream_t stream
) {
    dim3 g((D + 255) / 256, M);
    moe_weighted_sum_f16_k<<<g, 256, 0, stream>>>(expert_outs, weights, shared_out, dst, top_k, D);
}

void rad_cuda_embed_to_mhc_f16(
    const __nv_bfloat16* table,
    const int* tok_ids,
    float* streams,
    int M,
    int D,
    cudaStream_t stream
) {
    dim3 grid((D + 255) / 256, M);
    embed_to_mhc_f16_k<<<grid, 256, 0, stream>>>(table, tok_ids, streams, D);
}

void rad_cuda_rms_norm_f32_bf16w_f16(
    const float* x,
    const __nv_bfloat16* w,
    half* y,
    int M,
    int D,
    float eps,
    cudaStream_t stream
) {
    rms_norm_f32_bf16w_f16_k<<<M, 256, 0, stream>>>(x, w, y, D, eps);
}

void rad_cuda_mhc_collapse_mean_f16(
    const float* streams,
    float* out,
    int M,
    int D,
    cudaStream_t stream
) {
    dim3 grid((D + 255) / 256, M);
    mhc_collapse_mean_f16_k<<<grid, 256, 0, stream>>>(streams, out, D);
}

void rad_cuda_kv_scatter_f16(
    const half* k_in,
    const half* v_in,
    half* k_cache,
    half* v_cache,
    int M,
    int start_pos,
    int max_ctx,
    int head_dim,
    cudaStream_t stream
) {
    dim3 grid((head_dim + 255) / 256, M);
    kv_scatter_f16_k<<<grid, 256, 0, stream>>>(k_in, v_in, k_cache, v_cache, start_pos, max_ctx, head_dim);
}

void rad_cuda_rope_yarn_f16(
    half* q,
    half* k,
    int M,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    float theta,
    int orig_len,
    float sign,
    cudaStream_t stream
) {
    dim3 grid(n_q_heads + n_kv_heads, M);
    rope_yarn_f16_k<<<grid, rope_dim / 2, 0, stream>>>(
        q, k, n_q_heads, n_kv_heads, head_dim, rope_dim, start_pos, theta, orig_len, sign
    );
}

void rad_cuda_mhc_sinkhorn_mix_f16(
    float* streams,
    const half* branch_out,
    const half* peer_branch_out,
    const float* h_coeff,
    int M,
    int D,
    int sinkhorn_iters,
    float eps,
    cudaStream_t stream
) {
    dim3 grid((D + 255) / 256, M);
    mhc_sinkhorn_mix_f16_k<<<grid, 256, 0, stream>>>(
        streams, branch_out, peer_branch_out, h_coeff, D, sinkhorn_iters, eps
    );
}

__global__ void mtp_add_proj_to_mhc_f32_k(
    const __nv_bfloat16* __restrict__ embed,
    const int* __restrict__ tok_ids,
    const half* __restrict__ proj,
    float* __restrict__ streams,
    int D
) {
    int mr = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        int tid = tok_ids[mr];
        float v = __bfloat162float(embed[(size_t)tid * D + i]) + __half2float(proj[i]);
        float* sr = streams + (size_t)mr * 4 * D;
        sr[0 * D + i] = v;
        sr[1 * D + i] = v;
        sr[2 * D + i] = v;
        sr[3 * D + i] = v;
    }
}

void rad_cuda_mqa_attn_sink_causal_f16(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    const half* comp_kv,
    const float* attn_sink,
    half* out,
    int M,
    int n_heads,
    int start_pos,
    int max_ctx,
    int sliding_window,
    int comp_ratio,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    size_t smem = (head_dim == 512) ? 0 : (size_t)head_dim * 2 * sizeof(float);
    dim3 grid(n_heads, M);
    mqa_attn_sink_causal_f16_k<<<grid, 256, smem, stream>>>(
        q, k_cache, v_cache, comp_kv, attn_sink, out, n_heads, start_pos, max_ctx, sliding_window, comp_ratio, head_dim, scale
    );
}

void rad_cuda_csa2_compress_pair_f16(
    const half* raw_kv,
    const half* raw_gate,
    const __nv_bfloat16* norm_w,
    half* comp_out,
    int c_start,
    int n_pairs,
    int head_dim,
    int rope_dim,
    float theta,
    int orig_len,
    float eps,
    cudaStream_t stream
) {
    if (n_pairs <= 0) return;
    csa2_compress_pair_f16_k<<<n_pairs, 256, 0, stream>>>(
        raw_kv, raw_gate, norm_w, comp_out, c_start, head_dim, rope_dim, theta, orig_len, eps
    );
}

__global__ void engram_gate_f32_k(
    float* __restrict__ streams,
    const half* __restrict__ kv,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    int D,
    float eps
) {
    int r = blockIdx.x;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    float* sr = streams + ((size_t)mr * 4 + r) * D;
    const half* kr = kv + ((size_t)mr * 5 + r) * D;
    const half* vr = kv + ((size_t)mr * 5 + 4) * D;
    const float* qw = q_weight + (size_t)r * D;
    const float* kw = k_weight + (size_t)r * D;

    float qss = 0.0f;
    float kss = 0.0f;
    for (int i = t; i < D; i += blockDim.x) {
        float qv = sr[i];
        float kv_v = __half2float(kr[i]);
        qss = fmaf(qv, qv, qss);
        kss = fmaf(kv_v, kv_v, kss);
    }
    int lane = t & 31;
    int warp = t >> 5;
    __shared__ float sm_q[32];
    __shared__ float sm_k[32];
    for (int m = 16; m > 0; m >>= 1) {
        qss += __shfl_xor_sync(0xFFFFFFFFu, qss, m);
        kss += __shfl_xor_sync(0xFFFFFFFFu, kss, m);
    }
    if (lane == 0) {
        sm_q[warp] = qss;
        sm_k[warp] = kss;
    }
    __syncthreads();
    if (warp == 0) {
        float vq = (lane < (blockDim.x >> 5)) ? sm_q[lane] : 0.0f;
        float vk = (lane < (blockDim.x >> 5)) ? sm_k[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) {
            vq += __shfl_xor_sync(0xFFFFFFFFu, vq, m);
            vk += __shfl_xor_sync(0xFFFFFFFFu, vk, m);
        }
        if (lane == 0) {
            sm_q[0] = rsqrtf(vq / (float)D + eps);
            sm_k[0] = rsqrtf(vk / (float)D + eps);
        }
    }
    __syncthreads();
    float qinv = sm_q[0];
    float kinv = sm_k[0];

    float dot = 0.0f;
    for (int i = t; i < D; i += blockDim.x) {
        float qn = sr[i] * qinv * qw[i];
        float kn = __half2float(kr[i]) * kinv * kw[i];
        dot = fmaf(qn, kn, dot);
    }
    for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
    if (lane == 0) sm_q[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        float vd = (lane < (blockDim.x >> 5)) ? sm_q[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) vd += __shfl_xor_sync(0xFFFFFFFFu, vd, m);
        if (lane == 0) {
            float sc = vd * rsqrtf((float)D);
            float sc_s = copysignf(sqrtf(fmaxf(fabsf(sc), 1e-6f)), sc);
            sm_q[0] = 1.0f / (1.0f + expf(-sc_s));
        }
    }
    __syncthreads();
    float gate = sm_q[0];
    for (int i = t; i < D; i += blockDim.x) {
        sr[i] += gate * __half2float(vr[i]);
    }
}

void rad_cuda_engram_gate_f32(
    float* streams,
    const half* kv,
    const float* q_weight,
    const float* k_weight,
    int M,
    int D,
    float eps,
    cudaStream_t stream
) {
    dim3 grid(4, M);
    engram_gate_f32_k<<<grid, 256, 0, stream>>>(streams, kv, q_weight, k_weight, D, eps);
}

void rad_cuda_mtp_add_proj_to_mhc_f32(
    const __nv_bfloat16* embed,
    const int* tok_ids,
    const half* proj,
    float* streams,
    int M,
    int D,
    cudaStream_t stream
) {
    dim3 grid((D + 255) / 256, M);
    mtp_add_proj_to_mhc_f32_k<<<grid, 256, 0, stream>>>(embed, tok_ids, proj, streams, D);
}

void rad_cuda_wo_a_gather_group_f16(
    const half* attn_out,
    half* grp_in,
    int M,
    int n_heads_local,
    int heads_per_grp,
    int head_dim,
    int gi,
    cudaStream_t stream
) {
    int grp_dim = heads_per_grp * head_dim;
    dim3 grid((grp_dim + 255) / 256, M);
    wo_a_gather_group_f16_k<<<grid, 256, 0, stream>>>(attn_out, grp_in, n_heads_local, heads_per_grp, head_dim, gi);
}

__global__ void mhc_collapse_mean_to_f16_k(
    const float* __restrict__ streams,
    half* __restrict__ out,
    int D,
    int out_stride,
    int out_col_off
) {
    int mr = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        const float* sr = streams + (size_t)mr * 4 * D;
        float v = (0.25f / 16.0f) * (sr[0 * D + i] + sr[1 * D + i] + sr[2 * D + i] + sr[3 * D + i]);
        v = fmaxf(fminf(v, 65000.0f), -65000.0f);
        out[(size_t)mr * out_stride + out_col_off + i] = __float2half(v);
    }
}

__global__ void mqa_attn_sink_dspark_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_ring,
    const half* __restrict__ v_ring,
    const half* __restrict__ kv_blk,
    const float* __restrict__ attn_sink,
    half* __restrict__ out,
    int n_heads,
    int q0,
    int max_ctx,
    int sliding_window,
    int blk_len,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int p_start = (sliding_window > 0 && q0 > sliding_window) ? (q0 - sliding_window) : 0;
    float m_max = attn_sink ? attn_sink[h] : -1e30f;
    float l_sum = attn_sink ? 1.0f : 0.0f;
    __shared__ float wsum[8];

    if (head_dim == 512) {
        const half2* qh2 = reinterpret_cast<const half2*>(q + ((size_t)mr * n_heads + h) * 512);
        float2 qv = __half22float2(__ldg(qh2 + t));
        float a0 = 0.0f;
        float a1 = 0.0f;
        bool same_kv = (k_ring == v_ring);
        for (int p = p_start; p < q0; ++p) {
            int slot = p % max_ctx;
            const half2* kp2 = reinterpret_cast<const half2*>(k_ring + (size_t)slot * 512);
            float2 kv = __half22float2(__ldg(kp2 + t));
            float dot = fmaf(qv.x, kv.x, qv.y * kv.y);
            for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
            if (lane == 0) wsum[warp] = dot;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < 8) ? wsum[lane] : 0.0f;
                for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
                if (lane == 0) wsum[0] = v * scale;
            }
            __syncthreads();
            float score = wsum[0];
            float new_max = fmaxf(m_max, score);
            float alpha = expf(m_max - new_max);
            float beta = expf(score - new_max);
            l_sum = l_sum * alpha + beta;
            m_max = new_max;
            float2 vv = same_kv ? kv : __half22float2(__ldg(reinterpret_cast<const half2*>(v_ring + (size_t)slot * 512) + t));
            a0 = fmaf(a0, alpha, beta * vv.x);
            a1 = fmaf(a1, alpha, beta * vv.y);
        }
        for (int b = 0; b < blk_len; ++b) {
            const half2* kb2 = reinterpret_cast<const half2*>(kv_blk + (size_t)b * 512);
            float2 kv = __half22float2(__ldg(kb2 + t));
            float dot = fmaf(qv.x, kv.x, qv.y * kv.y);
            for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
            if (lane == 0) wsum[warp] = dot;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < 8) ? wsum[lane] : 0.0f;
                for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
                if (lane == 0) wsum[0] = v * scale;
            }
            __syncthreads();
            float score = wsum[0];
            float new_max = fmaxf(m_max, score);
            float alpha = expf(m_max - new_max);
            float beta = expf(score - new_max);
            l_sum = l_sum * alpha + beta;
            m_max = new_max;
            a0 = fmaf(a0, alpha, beta * kv.x);
            a1 = fmaf(a1, alpha, beta * kv.y);
        }
        float inv = 1.0f / fmaxf(l_sum, 1e-20f);
        half2* oh2 = reinterpret_cast<half2*>(out + ((size_t)mr * n_heads + h) * 512);
        oh2[t] = __float22half2_rn(make_float2(a0 * inv, a1 * inv));
        return;
    }

    const half* qh = q + ((size_t)mr * n_heads + h) * head_dim;
    extern __shared__ float dspark_smem[];
    float* s_q = dspark_smem;
    float* s_acc = dspark_smem + head_dim;
    for (int d = t; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(qh[d]);
        s_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int p = p_start; p < q0; ++p) {
        int slot = p % max_ctx;
        const half* kp = k_ring + (size_t)slot * head_dim;
        float dot = 0.0f;
        for (int d = t; d < head_dim; d += blockDim.x) {
            dot += s_q[d] * __half2float(kp[d]);
        }
        for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
        if (lane == 0) wsum[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < 8) ? wsum[lane] : 0.0f;
            for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
            if (lane == 0) wsum[0] = v * scale;
        }
        __syncthreads();
        float score = wsum[0];
        __syncthreads();
        float new_max = fmaxf(m_max, score);
        float alpha = expf(m_max - new_max);
        float beta = expf(score - new_max);
        l_sum = l_sum * alpha + beta;
        m_max = new_max;
        const half* vp = v_ring + (size_t)slot * head_dim;
        for (int d = t; d < head_dim; d += blockDim.x) {
            s_acc[d] = s_acc[d] * alpha + beta * __half2float(vp[d]);
        }
    }
    for (int b = 0; b < blk_len; ++b) {
        const half* kb = kv_blk + (size_t)b * head_dim;
        float dot = 0.0f;
        for (int d = t; d < head_dim; d += blockDim.x) {
            dot += s_q[d] * __half2float(kb[d]);
        }
        for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
        if (lane == 0) wsum[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < 8) ? wsum[lane] : 0.0f;
            for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
            if (lane == 0) wsum[0] = v * scale;
        }
        __syncthreads();
        float score = wsum[0];
        __syncthreads();
        float new_max = fmaxf(m_max, score);
        float alpha = expf(m_max - new_max);
        float beta = expf(score - new_max);
        l_sum = l_sum * alpha + beta;
        m_max = new_max;
        for (int d = t; d < head_dim; d += blockDim.x) {
            s_acc[d] = s_acc[d] * alpha + beta * __half2float(kb[d]);
        }
    }
    float inv = 1.0f / fmaxf(l_sum, 1e-20f);
    half* oh = out + ((size_t)mr * n_heads + h) * head_dim;
    for (int d = t; d < head_dim; d += blockDim.x) {
        oh[d] = __float2half(s_acc[d] * inv);
    }
}

__global__ void markov_bias_add_f16_k(
    const half* __restrict__ logits_row,
    const half* __restrict__ markov_head,
    const __nv_bfloat16* __restrict__ markov_embed,
    int prev_tok,
    half* __restrict__ out_row,
    int vocab_local,
    int rank
) {
    __shared__ float s_emb[256];
    for (int i = threadIdx.x; i < rank; i += blockDim.x) {
        s_emb[i] = __bfloat162float(markov_embed[(size_t)prev_tok * rank + i]);
    }
    __syncthreads();
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < vocab_local) {
        const half2* mh2 = reinterpret_cast<const half2*>(markov_head + (size_t)v * rank);
        int r2 = rank >> 1;
        float dot = 0.0f;
        for (int i = 0; i < r2; ++i) {
            float2 wv = __half22float2(mh2[i]);
            dot = fmaf(wv.x, s_emb[2 * i], dot);
            dot = fmaf(wv.y, s_emb[2 * i + 1], dot);
        }
        out_row[v] = __float2half(__half2float(logits_row[v]) + dot);
    }
}

void rad_cuda_mhc_collapse_mean_to_f16(
    const float* streams,
    half* out,
    int M,
    int D,
    int out_stride,
    int out_col_off,
    cudaStream_t stream
) {
    dim3 grid((D + 255) / 256, M);
    mhc_collapse_mean_to_f16_k<<<grid, 256, 0, stream>>>(streams, out, D, out_stride, out_col_off);
}

void rad_cuda_mqa_attn_sink_dspark_f16(
    const half* q,
    const half* k_ring,
    const half* v_ring,
    const half* kv_blk,
    const float* attn_sink,
    half* out,
    int M,
    int n_heads,
    int q0,
    int max_ctx,
    int sliding_window,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    dim3 grid(n_heads, M);
    size_t smem = (head_dim == 512) ? 0 : (size_t)head_dim * 2 * sizeof(float);
    mqa_attn_sink_dspark_f16_k<<<grid, 256, smem, stream>>>(
        q, k_ring, v_ring, kv_blk, attn_sink, out, n_heads, q0, max_ctx, sliding_window, M, head_dim, scale
    );
}

void rad_cuda_markov_bias_add_f16(
    const half* logits_row,
    const half* markov_head,
    const __nv_bfloat16* markov_embed,
    int prev_tok,
    half* out_row,
    int vocab_local,
    int rank,
    cudaStream_t stream
) {
    markov_bias_add_f16_k<<<(vocab_local + 255) / 256, 256, 0, stream>>>(
        logits_row, markov_head, markov_embed, prev_tok, out_row, vocab_local, rank
    );
}

void rad_cuda_wo_a_scatter_group_f16(
    const half* grp_out,
    half* wo_mid,
    int M,
    int groups_local,
    int o_lora_rank,
    int gi,
    cudaStream_t stream
) {
    dim3 grid((o_lora_rank + 255) / 256, M);
    wo_a_scatter_group_f16_k<<<grid, 256, 0, stream>>>(grp_out, wo_mid, groups_local, o_lora_rank, gi);
}

}
