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
    int d4 = D >> 2;
    const uint2* xr4 = reinterpret_cast<const uint2*>(xr);
    for (int i = t; i < d4; i += blockDim.x) {
        uint2 xq = xr4[i];
        float2 x01 = __half22float2(*reinterpret_cast<const half2*>(&xq.x));
        float2 x23 = __half22float2(*reinterpret_cast<const half2*>(&xq.y));
        ss = fmaf(x01.x, x01.x, fmaf(x01.y, x01.y, fmaf(x23.x, x23.x, fmaf(x23.y, x23.y, ss))));
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
    const uint2* w4 = reinterpret_cast<const uint2*>(w);
    uint2* yr4 = reinterpret_cast<uint2*>(yr);
    for (int i = t; i < d4; i += blockDim.x) {
        uint2 xq = xr4[i];
        uint2 wq = __ldg(w4 + i);
        float2 x01 = __half22float2(*reinterpret_cast<const half2*>(&xq.x));
        float2 x23 = __half22float2(*reinterpret_cast<const half2*>(&xq.y));
        float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.x));
        float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.y));
        half2 h01 = __float22half2_rn(make_float2(x01.x * inv * w01.x, x01.y * inv * w01.y));
        half2 h23 = __float22half2_rn(make_float2(x23.x * inv * w23.x, x23.y * inv * w23.y));
        uint2 oq;
        oq.x = *reinterpret_cast<const uint32_t*>(&h01);
        oq.y = *reinterpret_cast<const uint32_t*>(&h23);
        yr4[i] = oq;
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
    const int* __restrict__ expert_ids,
    const half* __restrict__ shared_out,
    half* __restrict__ dst,
    int top_k,
    int D,
    int expert_start,
    int n_local_experts
) {
    int mr = blockIdx.y;
    int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    int d4 = D >> 2;
    if (i4 < d4) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if (shared_out) {
            uint2 sq = reinterpret_cast<const uint2*>(shared_out + (size_t)mr * D)[i4];
            float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
            float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
            a0 = s01.x; a1 = s01.y; a2 = s23.x; a3 = s23.y;
        }
        const float* wm = weights + (size_t)mr * top_k;
        const int* em_id = expert_ids ? (expert_ids + (size_t)mr * top_k) : nullptr;
        const half* em = expert_outs + (size_t)mr * top_k * D;
        for (int k = 0; k < top_k; ++k) {
            if (em_id) {
                int le = em_id[k] - expert_start;
                if (le < 0 || le >= n_local_experts) continue;
            }
            float w = wm[k];
            uint2 eq = reinterpret_cast<const uint2*>(em + (size_t)k * D)[i4];
            float2 e01 = __half22float2(*reinterpret_cast<const half2*>(&eq.x));
            float2 e23 = __half22float2(*reinterpret_cast<const half2*>(&eq.y));
            a0 = fmaf(w, e01.x, a0);
            a1 = fmaf(w, e01.y, a1);
            a2 = fmaf(w, e23.x, a2);
            a3 = fmaf(w, e23.y, a3);
        }
        half2 h01 = __float22half2_rn(make_float2(a0, a1));
        half2 h23 = __float22half2_rn(make_float2(a2, a3));
        uint2 oq;
        oq.x = *reinterpret_cast<const uint32_t*>(&h01);
        oq.y = *reinterpret_cast<const uint32_t*>(&h23);
        reinterpret_cast<uint2*>(dst + (size_t)mr * D)[i4] = oq;
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
    int i8 = blockIdx.x * blockDim.x + threadIdx.x;
    int d8 = head_dim >> 3;
    if (i8 < d8) {
        int p = (start_pos + mr) % max_ctx;
        reinterpret_cast<uint4*>(k_cache + (size_t)p * head_dim)[i8] =
            reinterpret_cast<const uint4*>(k_in + (size_t)mr * head_dim)[i8];
        if (v_cache != k_cache) {
            reinterpret_cast<uint4*>(v_cache + (size_t)p * head_dim)[i8] =
                reinterpret_cast<const uint4*>(v_in + (size_t)mr * head_dim)[i8];
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
    int i4 = blockIdx.x * blockDim.x + t;
    int d4 = D >> 2;
    if (i4 < d4) {
        float4* sr4 = reinterpret_cast<float4*>(streams + (size_t)mr * 4 * D);
        float4 x0 = sr4[0 * d4 + i4];
        float4 x1 = sr4[1 * d4 + i4];
        float4 x2 = sr4[2 * d4 + i4];
        float4 x3 = sr4[3 * d4 + i4];
        const half2* bo2 = reinterpret_cast<const half2*>(branch_out + (size_t)mr * D);
        float2 b0 = __half22float2(bo2[i4 * 2]);
        float2 b1 = __half22float2(bo2[i4 * 2 + 1]);
        if (peer_branch_out) {
            const half2* po2 = reinterpret_cast<const half2*>(peer_branch_out + (size_t)mr * D);
            float2 p0 = __half22float2(po2[i4 * 2]);
            float2 p1 = __half22float2(po2[i4 * 2 + 1]);
            b0.x += p0.x; b0.y += p0.y;
            b1.x += p1.x; b1.y += p1.y;
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float m0 = s_res[0 * 4 + r], m1 = s_res[1 * 4 + r], m2 = s_res[2 * 4 + r], m3 = s_res[3 * 4 + r];
            float sp = s_post[r];
            float4 outv;
            outv.x = m0 * x0.x + m1 * x1.x + m2 * x2.x + m3 * x3.x + sp * b0.x;
            outv.y = m0 * x0.y + m1 * x1.y + m2 * x2.y + m3 * x3.y + sp * b0.y;
            outv.z = m0 * x0.z + m1 * x1.z + m2 * x2.z + m3 * x3.z + sp * b1.x;
            outv.w = m0 * x0.w + m1 * x1.w + m2 * x2.w + m3 * x3.w + sp * b1.y;
            sr4[r * d4 + i4] = outv;
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
        __shared__ float wsum4[4][8];
        const half2* qh2 = reinterpret_cast<const half2*>(q + ((size_t)mr * n_heads + h) * 512);
        float2 qv = __half22float2(__ldg(qh2 + t));
        float a0 = 0.0f;
        float a1 = 0.0f;
        bool same_kv = (k_cache == v_cache);
        int p = p_start;
        for (; p + 3 < seq_len; p += 4) {
            float2 kv0 = __half22float2(__ldg(reinterpret_cast<const half2*>(k_cache + (size_t)(p + 0) * 512) + t));
            float2 kv1 = __half22float2(__ldg(reinterpret_cast<const half2*>(k_cache + (size_t)(p + 1) * 512) + t));
            float2 kv2 = __half22float2(__ldg(reinterpret_cast<const half2*>(k_cache + (size_t)(p + 2) * 512) + t));
            float2 kv3 = __half22float2(__ldg(reinterpret_cast<const half2*>(k_cache + (size_t)(p + 3) * 512) + t));
            float d0 = fmaf(qv.x, kv0.x, qv.y * kv0.y);
            float d1 = fmaf(qv.x, kv1.x, qv.y * kv1.y);
            float d2 = fmaf(qv.x, kv2.x, qv.y * kv2.y);
            float d3 = fmaf(qv.x, kv3.x, qv.y * kv3.y);
            for (int m = 16; m > 0; m >>= 1) {
                d0 += __shfl_xor_sync(0xFFFFFFFFu, d0, m);
                d1 += __shfl_xor_sync(0xFFFFFFFFu, d1, m);
                d2 += __shfl_xor_sync(0xFFFFFFFFu, d2, m);
                d3 += __shfl_xor_sync(0xFFFFFFFFu, d3, m);
            }
            if (lane == 0) {
                wsum4[0][warp] = d0;
                wsum4[1][warp] = d1;
                wsum4[2][warp] = d2;
                wsum4[3][warp] = d3;
            }
            __syncthreads();
            if (warp < 4) {
                float v = (lane < 8) ? wsum4[warp][lane] : 0.0f;
                for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
                if (lane == 0) wsum4[warp][0] = v * scale;
            }
            __syncthreads();
            float s0 = wsum4[0][0], s1 = wsum4[1][0], s2 = wsum4[2][0], s3 = wsum4[3][0];
            float new_max = fmaxf(m_max, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
            float alpha = expf(m_max - new_max);
            float b0 = expf(s0 - new_max);
            float b1 = expf(s1 - new_max);
            float b2 = expf(s2 - new_max);
            float b3 = expf(s3 - new_max);
            l_sum = l_sum * alpha + b0 + b1 + b2 + b3;
            m_max = new_max;
            float2 v0 = same_kv ? kv0 : __half22float2(__ldg(reinterpret_cast<const half2*>(v_cache + (size_t)(p + 0) * 512) + t));
            float2 v1 = same_kv ? kv1 : __half22float2(__ldg(reinterpret_cast<const half2*>(v_cache + (size_t)(p + 1) * 512) + t));
            float2 v2 = same_kv ? kv2 : __half22float2(__ldg(reinterpret_cast<const half2*>(v_cache + (size_t)(p + 2) * 512) + t));
            float2 v3 = same_kv ? kv3 : __half22float2(__ldg(reinterpret_cast<const half2*>(v_cache + (size_t)(p + 3) * 512) + t));
            a0 = a0 * alpha + b0 * v0.x + b1 * v1.x + b2 * v2.x + b3 * v3.x;
            a1 = a1 * alpha + b0 * v0.y + b1 * v1.y + b2 * v2.y + b3 * v3.y;
        }
        for (; p < seq_len; ++p) {
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
            int c = 0;
            for (; c + 3 < n_comp; c += 4) {
                float2 kv0 = __half22float2(__ldg(reinterpret_cast<const half2*>(comp_kv + (size_t)(c + 0) * 512) + t));
                float2 kv1 = __half22float2(__ldg(reinterpret_cast<const half2*>(comp_kv + (size_t)(c + 1) * 512) + t));
                float2 kv2 = __half22float2(__ldg(reinterpret_cast<const half2*>(comp_kv + (size_t)(c + 2) * 512) + t));
                float2 kv3 = __half22float2(__ldg(reinterpret_cast<const half2*>(comp_kv + (size_t)(c + 3) * 512) + t));
                float d0 = fmaf(qv.x, kv0.x, qv.y * kv0.y);
                float d1 = fmaf(qv.x, kv1.x, qv.y * kv1.y);
                float d2 = fmaf(qv.x, kv2.x, qv.y * kv2.y);
                float d3 = fmaf(qv.x, kv3.x, qv.y * kv3.y);
                for (int m = 16; m > 0; m >>= 1) {
                    d0 += __shfl_xor_sync(0xFFFFFFFFu, d0, m);
                    d1 += __shfl_xor_sync(0xFFFFFFFFu, d1, m);
                    d2 += __shfl_xor_sync(0xFFFFFFFFu, d2, m);
                    d3 += __shfl_xor_sync(0xFFFFFFFFu, d3, m);
                }
                if (lane == 0) {
                    wsum4[0][warp] = d0;
                    wsum4[1][warp] = d1;
                    wsum4[2][warp] = d2;
                    wsum4[3][warp] = d3;
                }
                __syncthreads();
                if (warp < 4) {
                    float v = (lane < 8) ? wsum4[warp][lane] : 0.0f;
                    for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
                    if (lane == 0) wsum4[warp][0] = v * scale;
                }
                __syncthreads();
                float s0 = wsum4[0][0], s1 = wsum4[1][0], s2 = wsum4[2][0], s3 = wsum4[3][0];
                float new_max = fmaxf(m_max, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
                float alpha = expf(m_max - new_max);
                float b0 = expf(s0 - new_max);
                float b1 = expf(s1 - new_max);
                float b2 = expf(s2 - new_max);
                float b3 = expf(s3 - new_max);
                l_sum = l_sum * alpha + b0 + b1 + b2 + b3;
                m_max = new_max;
                a0 = a0 * alpha + b0 * kv0.x + b1 * kv1.x + b2 * kv2.x + b3 * kv3.x;
                a1 = a1 * alpha + b0 * kv0.y + b1 * kv1.y + b2 * kv2.y + b3 * kv3.y;
            }
            for (; c < n_comp; ++c) {
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

template <int HB>
__global__ void mqa_attn_sink_causal_mh_f16_k(
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
    float scale
) {
    int h0 = blockIdx.x * HB;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int seq_len = start_pos + mr + 1;
    if (seq_len > max_ctx) seq_len = max_ctx;
    int p_start = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;

    float qx[HB], qy[HB], ax[HB], ay[HB], m_max[HB], l_sum[HB];
    #pragma unroll
    for (int j = 0; j < HB; ++j) {
        const half2* qh2 = reinterpret_cast<const half2*>(q + ((size_t)mr * n_heads + h0 + j) * 512);
        float2 qv = __half22float2(__ldg(qh2 + t));
        qx[j] = qv.x;
        qy[j] = qv.y;
        ax[j] = 0.0f;
        ay[j] = 0.0f;
        m_max[j] = attn_sink ? attn_sink[h0 + j] : -1e30f;
        l_sum[j] = attn_sink ? 1.0f : 0.0f;
    }

    __shared__ float wsum[HB][8];
    __shared__ float s_score[HB];
    bool same_kv = (k_cache == v_cache);

    for (int p = p_start; p < seq_len; ++p) {
        float2 kv = __half22float2(__ldg(reinterpret_cast<const half2*>(k_cache + (size_t)p * 512) + t));
        #pragma unroll
        for (int j = 0; j < HB; ++j) {
            float d = fmaf(qx[j], kv.x, qy[j] * kv.y);
            for (int m = 16; m > 0; m >>= 1) d += __shfl_xor_sync(0xFFFFFFFFu, d, m);
            if (lane == 0) wsum[j][warp] = d;
        }
        __syncthreads();
        if (warp < HB) {
            float v = (lane < 8) ? wsum[warp][lane] : 0.0f;
            for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
            if (lane == 0) s_score[warp] = v * scale;
        }
        __syncthreads();
        float2 vv = same_kv ? kv : __half22float2(__ldg(reinterpret_cast<const half2*>(v_cache + (size_t)p * 512) + t));
        #pragma unroll
        for (int j = 0; j < HB; ++j) {
            float score = s_score[j];
            float nm = fmaxf(m_max[j], score);
            float alpha = expf(m_max[j] - nm);
            float beta = expf(score - nm);
            l_sum[j] = l_sum[j] * alpha + beta;
            m_max[j] = nm;
            ax[j] = fmaf(ax[j], alpha, beta * vv.x);
            ay[j] = fmaf(ay[j], alpha, beta * vv.y);
        }
    }

    if (comp_kv && comp_ratio > 0) {
        int n_comp = seq_len / comp_ratio;
        for (int c = 0; c < n_comp; ++c) {
            float2 kv = __half22float2(__ldg(reinterpret_cast<const half2*>(comp_kv + (size_t)c * 512) + t));
            #pragma unroll
            for (int j = 0; j < HB; ++j) {
                float d = fmaf(qx[j], kv.x, qy[j] * kv.y);
                for (int m = 16; m > 0; m >>= 1) d += __shfl_xor_sync(0xFFFFFFFFu, d, m);
                if (lane == 0) wsum[j][warp] = d;
            }
            __syncthreads();
            if (warp < HB) {
                float v = (lane < 8) ? wsum[warp][lane] : 0.0f;
                for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
                if (lane == 0) s_score[warp] = v * scale;
            }
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < HB; ++j) {
                float score = s_score[j];
                float nm = fmaxf(m_max[j], score);
                float alpha = expf(m_max[j] - nm);
                float beta = expf(score - nm);
                l_sum[j] = l_sum[j] * alpha + beta;
                m_max[j] = nm;
                ax[j] = fmaf(ax[j], alpha, beta * kv.x);
                ay[j] = fmaf(ay[j], alpha, beta * kv.y);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < HB; ++j) {
        float inv = 1.0f / fmaxf(l_sum[j], 1e-20f);
        half2* oh2 = reinterpret_cast<half2*>(out + ((size_t)mr * n_heads + h0 + j) * 512);
        oh2[t] = __float22half2_rn(make_float2(ax[j] * inv, ay[j] * inv));
    }
}

__device__ __forceinline__ void mqa_mma_m16n8k16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__global__ __launch_bounds__(256)
void mqa_flash_mma16_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ comp_kv,
    const float* __restrict__ attn_sink,
    half* __restrict__ out,
    int n_heads,
    int start_pos,
    int max_ctx,
    int sliding_window,
    int comp_ratio,
    float scale
) {
    int h0 = blockIdx.x * 16;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    int warp = t >> 5;
    int lane = t & 31;
    int group_id = lane >> 2;
    int tid = lane & 3;

    int seq_len = start_pos + mr + 1;
    if (seq_len > max_ctx) seq_len = max_ctx;
    int p_start = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;

    __shared__ half s_q[16][512];
    __shared__ half s_kv[16][520];
    __shared__ float s_sp[8][16][16];
    __shared__ half s_p[16][16];
    __shared__ float s_m[16];
    __shared__ float s_l[16];
    __shared__ float s_alpha[16];

    const uint4* q_u4 = reinterpret_cast<const uint4*>(q + ((size_t)mr * n_heads + h0) * 512);
    uint4* sq_u4 = reinterpret_cast<uint4*>(&s_q[0][0]);
    for (int i = t; i < 1024; i += 256) {
        sq_u4[i] = __ldg(q_u4 + i);
    }
    if (t < 16) {
        s_m[t] = attn_sink ? attn_sink[h0 + t] : -1e30f;
        s_l[t] = attn_sink ? 1.0f : 0.0f;
    }
    __syncthreads();

    float o0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float o1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float o2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float o3[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto process_tile = [&](int tile_len) {
        float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
        float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
        int k_base = warp * 64;
        #pragma unroll
        for (int ks = 0; ks < 64; ks += 16) {
            int ko0 = k_base + ks + tid * 2;
            int ko1 = ko0 + 8;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][ko0]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][ko0]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][ko1]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][ko1]);
            uint32_t b00 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id][ko0]);
            uint32_t b01 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id][ko1]);
            uint32_t b10 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id + 8][ko0]);
            uint32_t b11 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id + 8][ko1]);
            mqa_mma_m16n8k16_f32(s00, s01, s02, s03, a0, a1, a2, a3, b00, b01);
            mqa_mma_m16n8k16_f32(s10, s11, s12, s13, a0, a1, a2, a3, b10, b11);
        }
        s_sp[warp][group_id][tid * 2] = s00;
        s_sp[warp][group_id][tid * 2 + 1] = s01;
        s_sp[warp][group_id + 8][tid * 2] = s02;
        s_sp[warp][group_id + 8][tid * 2 + 1] = s03;
        s_sp[warp][group_id][8 + tid * 2] = s10;
        s_sp[warp][group_id][8 + tid * 2 + 1] = s11;
        s_sp[warp][group_id + 8][8 + tid * 2] = s12;
        s_sp[warp][group_id + 8][8 + tid * 2 + 1] = s13;
        __syncthreads();

        int r = t >> 4;
        int c = t & 15;
        float sc = -1e30f;
        if (c < tile_len) {
            float sum = s_sp[0][r][c] + s_sp[1][r][c] + s_sp[2][r][c] + s_sp[3][r][c]
                      + s_sp[4][r][c] + s_sp[5][r][c] + s_sp[6][r][c] + s_sp[7][r][c];
            sc = sum * scale;
        }
        float rmax = sc;
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 8));
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 4));
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 2));
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 1));
        float old_m = s_m[r];
        float new_m = fmaxf(old_m, rmax);
        float alpha = exp2f((old_m - new_m) * 1.44269504f);
        float p = (c < tile_len) ? exp2f((sc - new_m) * 1.44269504f) : 0.0f;
        s_p[r][c] = __float2half(p);
        float rsum = p;
        rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 8);
        rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 4);
        rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 2);
        rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 1);
        if (c == 0) {
            s_l[r] = s_l[r] * alpha + rsum;
            s_m[r] = new_m;
            s_alpha[r] = alpha;
        }
        __syncthreads();

        float a0_sc = s_alpha[group_id];
        float a1_sc = s_alpha[group_id + 8];
        uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&s_p[group_id][tid * 2]);
        uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&s_p[group_id + 8][tid * 2]);
        uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&s_p[group_id][tid * 2 + 8]);
        uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&s_p[group_id + 8][tid * 2 + 8]);
        int col_w = warp * 64;
        #pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
            o0[nb] *= a0_sc;
            o1[nb] *= a0_sc;
            o2[nb] *= a1_sc;
            o3[nb] *= a1_sc;
            int col = col_w + nb * 8 + group_id;
            half2 vb0 = __halves2half2(s_kv[tid * 2][col], s_kv[tid * 2 + 1][col]);
            half2 vb1 = __halves2half2(s_kv[tid * 2 + 8][col], s_kv[tid * 2 + 9][col]);
            uint32_t b0 = *reinterpret_cast<uint32_t*>(&vb0);
            uint32_t b1 = *reinterpret_cast<uint32_t*>(&vb1);
            mqa_mma_m16n8k16_f32(o0[nb], o1[nb], o2[nb], o3[nb], pa0, pa1, pa2, pa3, b0, b1);
        }
        __syncthreads();
    };

    if (p_start < seq_len) {
        int tile_len0 = seq_len - p_start;
        if (tile_len0 > 16) tile_len0 = 16;
        int n_u4_0 = tile_len0 * 64;
        const uint4* src0 = reinterpret_cast<const uint4*>(k_cache + (size_t)p_start * 512);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int i = t + k * 256;
            reinterpret_cast<uint4*>(s_kv[i >> 6])[i & 63] = (i < n_u4_0) ? __ldg(src0 + i) : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        for (int p0 = p_start; p0 < seq_len; p0 += 16) {
            int tile_len = seq_len - p0;
            if (tile_len > 16) tile_len = 16;
            uint4 nxt[4];
            bool has_next = (p0 + 16 < seq_len);
            if (has_next) {
                int nlen = seq_len - (p0 + 16);
                if (nlen > 16) nlen = 16;
                int nu4 = nlen * 64;
                const uint4* nsrc = reinterpret_cast<const uint4*>(k_cache + (size_t)(p0 + 16) * 512);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int i = t + k * 256;
                    nxt[k] = (i < nu4) ? __ldg(nsrc + i) : make_uint4(0u, 0u, 0u, 0u);
                }
            }
            process_tile(tile_len);
            if (has_next) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int i = t + k * 256;
                    reinterpret_cast<uint4*>(s_kv[i >> 6])[i & 63] = nxt[k];
                }
                __syncthreads();
            }
        }
    }

    if (comp_kv && comp_ratio > 0 && p_start > 0) {
        int c_end = p_start / comp_ratio;
        if (c_end > 0) {
            int tile_len0 = (c_end > 16) ? 16 : c_end;
            int n_u4_0 = tile_len0 * 64;
            const uint4* src0 = reinterpret_cast<const uint4*>(comp_kv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int i = t + k * 256;
                reinterpret_cast<uint4*>(s_kv[i >> 6])[i & 63] = (i < n_u4_0) ? __ldg(src0 + i) : make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();
            for (int c0 = 0; c0 < c_end; c0 += 16) {
                int tile_len = c_end - c0;
                if (tile_len > 16) tile_len = 16;
                uint4 nxt[4];
                bool has_next = (c0 + 16 < c_end);
                if (has_next) {
                    int nlen = c_end - (c0 + 16);
                    if (nlen > 16) nlen = 16;
                    int nu4 = nlen * 64;
                    const uint4* nsrc = reinterpret_cast<const uint4*>(comp_kv + (size_t)(c0 + 16) * 512);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int i = t + k * 256;
                        nxt[k] = (i < nu4) ? __ldg(nsrc + i) : make_uint4(0u, 0u, 0u, 0u);
                    }
                }
                process_tile(tile_len);
                if (has_next) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int i = t + k * 256;
                        reinterpret_cast<uint4*>(s_kv[i >> 6])[i & 63] = nxt[k];
                    }
                    __syncthreads();
                }
            }
        }
    }

    int col_w = warp * 64;
    float inv0 = 1.0f / fmaxf(s_l[group_id], 1e-20f);
    float inv1 = 1.0f / fmaxf(s_l[group_id + 8], 1e-20f);
    half* oh0 = out + ((size_t)mr * n_heads + h0 + group_id) * 512 + col_w;
    half* oh1 = out + ((size_t)mr * n_heads + h0 + group_id + 8) * 512 + col_w;
    #pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
        int c0 = nb * 8 + tid * 2;
        oh0[c0] = __float2half(o0[nb] * inv0);
        oh0[c0 + 1] = __float2half(o1[nb] * inv0);
        oh1[c0] = __float2half(o2[nb] * inv1);
        oh1[c0 + 1] = __float2half(o3[nb] * inv1);
    }
}

__global__ __launch_bounds__(256)
void mqa_flash_mma16_splitkv_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ comp_kv,
    const float* __restrict__ attn_sink,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int n_heads,
    int start_pos,
    int max_ctx,
    int sliding_window,
    int comp_ratio,
    float scale,
    int n_splits
) {
    int h0 = blockIdx.x * 16;
    int mr = blockIdx.y;
    int sp = blockIdx.z;
    int t = threadIdx.x;
    int warp = t >> 5;
    int lane = t & 31;
    int group_id = lane >> 2;
    int tid = lane & 3;

    int seq_len = start_pos + mr + 1;
    if (seq_len > max_ctx) seq_len = max_ctx;
    int p_start = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;
    int raw_len = seq_len - p_start;
    int c_end = (comp_kv && comp_ratio > 0 && p_start > 0) ? (p_start / comp_ratio) : 0;

    int raw_tiles = (raw_len + 15) >> 4;
    int comp_tiles = (c_end + 15) >> 4;
    int tot_tiles = raw_tiles + comp_tiles;
    int t_beg = (tot_tiles * sp) / n_splits;
    int t_end = (tot_tiles * (sp + 1)) / n_splits;

    __shared__ half s_q[16][512];
    __shared__ half s_kv[16][520];
    __shared__ float s_sp[8][16][16];
    __shared__ half s_p[16][16];
    __shared__ float s_m[16];
    __shared__ float s_l[16];
    __shared__ float s_alpha[16];

    const uint4* q_u4 = reinterpret_cast<const uint4*>(q + ((size_t)mr * n_heads + h0) * 512);
    uint4* sq_u4 = reinterpret_cast<uint4*>(&s_q[0][0]);
    for (int i = t; i < 1024; i += 256) {
        sq_u4[i] = __ldg(q_u4 + i);
    }
    if (t < 16) {
        s_m[t] = (sp == 0 && attn_sink) ? attn_sink[h0 + t] : -1e30f;
        s_l[t] = (sp == 0 && attn_sink) ? 1.0f : 0.0f;
    }
    __syncthreads();

    float o0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float o1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float o2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float o3[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto tile_info = [&](int ti, int& tile_len, const uint4*& src_u4) {
        if (ti < raw_tiles) {
            int p0 = p_start + ti * 16;
            tile_len = seq_len - p0;
            if (tile_len > 16) tile_len = 16;
            src_u4 = reinterpret_cast<const uint4*>(k_cache + (size_t)p0 * 512);
        } else {
            int c0 = (ti - raw_tiles) * 16;
            tile_len = c_end - c0;
            if (tile_len > 16) tile_len = 16;
            src_u4 = reinterpret_cast<const uint4*>(comp_kv + (size_t)c0 * 512);
        }
    };

    if (t_beg < t_end) {
        int tile_len0;
        const uint4* src0;
        tile_info(t_beg, tile_len0, src0);
        int nu4_0 = tile_len0 * 64;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int i = t + k * 256;
            reinterpret_cast<uint4*>(s_kv[i >> 6])[i & 63] = (i < nu4_0) ? __ldg(src0 + i) : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        for (int ti = t_beg; ti < t_end; ++ti) {
            int tile_len;
            const uint4* src_u4;
            tile_info(ti, tile_len, src_u4);
            uint4 nxt[4];
            bool has_next = (ti + 1 < t_end);
            if (has_next) {
                int nlen;
                const uint4* nsrc;
                tile_info(ti + 1, nlen, nsrc);
                int nu4 = nlen * 64;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int i = t + k * 256;
                    nxt[k] = (i < nu4) ? __ldg(nsrc + i) : make_uint4(0u, 0u, 0u, 0u);
                }
            }

            float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
            float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
            int k_base = warp * 64;
            #pragma unroll
            for (int ks = 0; ks < 64; ks += 16) {
                int ko0 = k_base + ks + tid * 2;
                int ko1 = ko0 + 8;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][ko0]);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][ko0]);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][ko1]);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][ko1]);
                uint32_t b00 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id][ko0]);
                uint32_t b01 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id][ko1]);
                uint32_t b10 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id + 8][ko0]);
                uint32_t b11 = *reinterpret_cast<const uint32_t*>(&s_kv[group_id + 8][ko1]);
                mqa_mma_m16n8k16_f32(s00, s01, s02, s03, a0, a1, a2, a3, b00, b01);
                mqa_mma_m16n8k16_f32(s10, s11, s12, s13, a0, a1, a2, a3, b10, b11);
            }
            s_sp[warp][group_id][tid * 2] = s00;
            s_sp[warp][group_id][tid * 2 + 1] = s01;
            s_sp[warp][group_id + 8][tid * 2] = s02;
            s_sp[warp][group_id + 8][tid * 2 + 1] = s03;
            s_sp[warp][group_id][8 + tid * 2] = s10;
            s_sp[warp][group_id][8 + tid * 2 + 1] = s11;
            s_sp[warp][group_id + 8][8 + tid * 2] = s12;
            s_sp[warp][group_id + 8][8 + tid * 2 + 1] = s13;
            __syncthreads();

            int r = t >> 4;
            int c = t & 15;
            float sc = -1e30f;
            if (c < tile_len) {
                float sum = s_sp[0][r][c] + s_sp[1][r][c] + s_sp[2][r][c] + s_sp[3][r][c]
                          + s_sp[4][r][c] + s_sp[5][r][c] + s_sp[6][r][c] + s_sp[7][r][c];
                sc = sum * scale;
            }
            float rmax = sc;
            rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 8));
            rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 4));
            rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 2));
            rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFu, rmax, 1));
            float old_m = s_m[r];
            float new_m = fmaxf(old_m, rmax);
            float alpha = exp2f((old_m - new_m) * 1.44269504f);
            float p = (c < tile_len) ? exp2f((sc - new_m) * 1.44269504f) : 0.0f;
            s_p[r][c] = __float2half(p);
            float rsum = p;
            rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 8);
            rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 4);
            rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 2);
            rsum += __shfl_xor_sync(0xFFFFFFFFu, rsum, 1);
            if (c == 0) {
                s_l[r] = s_l[r] * alpha + rsum;
                s_m[r] = new_m;
                s_alpha[r] = alpha;
            }
            __syncthreads();

            float a0_sc = s_alpha[group_id];
            float a1_sc = s_alpha[group_id + 8];
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&s_p[group_id][tid * 2]);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&s_p[group_id + 8][tid * 2]);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&s_p[group_id][tid * 2 + 8]);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&s_p[group_id + 8][tid * 2 + 8]);
            int col_w = warp * 64;
            #pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                o0[nb] *= a0_sc;
                o1[nb] *= a0_sc;
                o2[nb] *= a1_sc;
                o3[nb] *= a1_sc;
                int col = col_w + nb * 8 + group_id;
                half2 vb0 = __halves2half2(s_kv[tid * 2][col], s_kv[tid * 2 + 1][col]);
                half2 vb1 = __halves2half2(s_kv[tid * 2 + 8][col], s_kv[tid * 2 + 9][col]);
                uint32_t b0 = *reinterpret_cast<uint32_t*>(&vb0);
                uint32_t b1 = *reinterpret_cast<uint32_t*>(&vb1);
                mqa_mma_m16n8k16_f32(o0[nb], o1[nb], o2[nb], o3[nb], pa0, pa1, pa2, pa3, b0, b1);
            }
            __syncthreads();
            if (has_next) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int i = t + k * 256;
                    reinterpret_cast<uint4*>(s_kv[i >> 6])[i & 63] = nxt[k];
                }
                __syncthreads();
            }
        }
    }

    int col_w = warp * 64;
    size_t base0 = (((size_t)mr * n_heads + h0 + group_id) * n_splits + sp) * 512 + col_w;
    size_t base1 = (((size_t)mr * n_heads + h0 + group_id + 8) * n_splits + sp) * 512 + col_w;
    float* po0 = partial_o + base0;
    float* po1 = partial_o + base1;
    #pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
        int c0 = nb * 8 + tid * 2;
        po0[c0] = o0[nb];
        po0[c0 + 1] = o1[nb];
        po1[c0] = o2[nb];
        po1[c0 + 1] = o3[nb];
    }
    if (t < 16) {
        size_t ml_idx = (((size_t)mr * n_heads + h0 + t) * n_splits + sp) * 2;
        partial_ml[ml_idx] = s_m[t];
        partial_ml[ml_idx + 1] = s_l[t];
    }
}

__global__ void mqa_splitkv_reduce_f16_k(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_ml,
    half* __restrict__ out,
    int n_heads,
    int n_splits
) {
    int h = blockIdx.x;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    size_t ml_base = ((size_t)mr * n_heads + h) * n_splits * 2;
    float g_m = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        float m = partial_ml[ml_base + sp * 2];
        if (m > g_m) g_m = m;
    }
    float g_l = 0.0f;
    float a0 = 0.0f, a1 = 0.0f;
    size_t o_base = ((size_t)mr * n_heads + h) * n_splits * 512;
    for (int sp = 0; sp < n_splits; ++sp) {
        float m = partial_ml[ml_base + sp * 2];
        float l = partial_ml[ml_base + sp * 2 + 1];
        float w = (l > 0.0f) ? exp2f((m - g_m) * 1.44269504f) : 0.0f;
        g_l += l * w;
        const float* po = partial_o + o_base + (size_t)sp * 512;
        a0 += po[t * 2] * w;
        a1 += po[t * 2 + 1] * w;
    }
    float inv = 1.0f / fmaxf(g_l, 1e-20f);
    half2* oh2 = reinterpret_cast<half2*>(out + ((size_t)mr * n_heads + h) * 512);
    oh2[t] = __float22half2_rn(make_float2(a0 * inv, a1 * inv));
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
    const int* expert_ids,
    const half* shared_out,
    half* dst,
    int M,
    int top_k,
    int D,
    int expert_start,
    int n_local_experts,
    cudaStream_t stream
) {
    dim3 g(((D >> 2) + 255) / 256, M);
    moe_weighted_sum_f16_k<<<g, 256, 0, stream>>>(expert_outs, weights, expert_ids, shared_out, dst, top_k, D, expert_start, n_local_experts);
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
    int d8 = head_dim >> 3;
    int threads = (d8 < 64) ? 32 : ((d8 < 128) ? 64 : 128);
    dim3 grid((d8 + threads - 1) / threads, M);
    kv_scatter_f16_k<<<grid, threads, 0, stream>>>(k_in, v_in, k_cache, v_cache, start_pos, max_ctx, head_dim);
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
    dim3 grid(((D >> 2) + 255) / 256, M);
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
    float* split_scratch,
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
    if (head_dim == 512 && k_cache == v_cache && (n_heads % 16) == 0) {
        int eff_len = start_pos + M;
        if (eff_len > max_ctx) eff_len = max_ctx;
        if (sliding_window > 0 && eff_len > sliding_window) {
            eff_len = sliding_window + (comp_kv && comp_ratio > 0 ? (eff_len - sliding_window) / comp_ratio : 0);
        }
        int base_ctas = (n_heads / 16) * M;
        if (split_scratch && eff_len >= 1024 && M <= 16 && base_ctas < 48) {
            int n_splits = (48 + base_ctas - 1) / base_ctas;
            if (n_splits > 8) n_splits = 8;
            if (n_splits > 1) {
                size_t o_elems = (size_t)M * n_heads * n_splits * 512;
                float* partial_ml = split_scratch + o_elems;
                dim3 g1(n_heads / 16, M, n_splits);
                mqa_flash_mma16_splitkv_f16_k<<<g1, 256, 0, stream>>>(
                    q, k_cache, comp_kv, attn_sink, split_scratch, partial_ml,
                    n_heads, start_pos, max_ctx, sliding_window, comp_ratio, scale, n_splits
                );
                dim3 g2(n_heads, M);
                mqa_splitkv_reduce_f16_k<<<g2, 256, 0, stream>>>(
                    split_scratch, partial_ml, out, n_heads, n_splits
                );
                return;
            }
        }
        if (M >= 16 || (M >= 4 && start_pos + M >= 256)) {
            dim3 grid(n_heads / 16, M);
            mqa_flash_mma16_f16_k<<<grid, 256, 0, stream>>>(
                q, k_cache, comp_kv, attn_sink, out, n_heads, start_pos, max_ctx, sliding_window, comp_ratio, scale
            );
            return;
        }
    }
    if (head_dim == 512 && M >= 16 && (n_heads % 8) == 0) {
        dim3 grid(n_heads / 8, M);
        mqa_attn_sink_causal_mh_f16_k<8><<<grid, 256, 0, stream>>>(
            q, k_cache, v_cache, comp_kv, attn_sink, out, n_heads, start_pos, max_ctx, sliding_window, comp_ratio, scale
        );
        return;
    }
    if (head_dim == 512 && M >= 2 && (n_heads % 4) == 0) {
        dim3 grid(n_heads / 4, M);
        mqa_attn_sink_causal_mh_f16_k<4><<<grid, 256, 0, stream>>>(
            q, k_cache, v_cache, comp_kv, attn_sink, out, n_heads, start_pos, max_ctx, sliding_window, comp_ratio, scale
        );
        return;
    }
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

__global__ void markov_argmax_stage1_f16_k(
    const half* __restrict__ logits_row,
    const half* __restrict__ markov_head,
    const __nv_bfloat16* __restrict__ markov_embed,
    int prev_tok,
    float* __restrict__ blk_vals,
    int* __restrict__ blk_ids,
    int vocab_local,
    int rank
) {
    __shared__ float s_emb[256];
    for (int i = threadIdx.x; i < rank; i += blockDim.x) {
        s_emb[i] = __bfloat162float(markov_embed[(size_t)prev_tok * rank + i]);
    }
    __syncthreads();
    int tid = threadIdx.x;
    int base = blockIdx.x * 1024 + tid;
    int r2 = rank >> 1;
    float best_v = -1e30f;
    int best_i = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int v = base + k * 256;
        if (v < vocab_local) {
            const half2* mh2 = reinterpret_cast<const half2*>(markov_head + (size_t)v * rank);
            float dot = 0.0f;
            for (int i = 0; i < r2; ++i) {
                float2 wv = __half22float2(__ldg(mh2 + i));
                dot = fmaf(wv.x, s_emb[2 * i], dot);
                dot = fmaf(wv.y, s_emb[2 * i + 1], dot);
            }
            float val = __half2float(__ldg(logits_row + v)) + dot;
            if (val > best_v) {
                best_v = val;
                best_i = v;
            }
        }
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int m = 16; m > 0; m >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, m);
        int oi = __shfl_xor_sync(0xFFFFFFFFu, best_i, m);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    if (lane == 0) {
        sv[warp] = best_v;
        si[warp] = best_i;
    }
    __syncthreads();
    if (warp == 0) {
        float bv = (lane < 8) ? sv[lane] : -1e30f;
        int bi = (lane < 8) ? si[lane] : 0;
        for (int m = 4; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, m);
            int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, m);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            blk_vals[blockIdx.x] = bv;
            blk_ids[blockIdx.x] = bi;
        }
    }
}

__global__ void markov_argmax_stage2_k(
    const float* __restrict__ blk_vals,
    const int* __restrict__ blk_ids,
    float* __restrict__ out_val,
    int* __restrict__ out_id,
    int n_blks
) {
    float bv = -1e30f;
    int bi = 0;
    for (int i = 0; i < n_blks; ++i) {
        float v = blk_vals[i];
        int id = blk_ids[i];
        if (v > bv || (v == bv && id < bi)) {
            bv = v;
            bi = id;
        }
    }
    out_val[0] = bv;
    out_id[0] = bi;
}

void rad_cuda_markov_argmax_f16(
    const half* logits_row,
    const half* markov_head,
    const __nv_bfloat16* markov_embed,
    int prev_tok,
    float* blk_vals,
    int* blk_ids,
    float* out_val,
    int* out_id,
    int vocab_local,
    int rank,
    cudaStream_t stream
) {
    int n_blks = (vocab_local + 1023) / 1024;
    markov_argmax_stage1_f16_k<<<n_blks, 256, 0, stream>>>(
        logits_row, markov_head, markov_embed, prev_tok, blk_vals, blk_ids, vocab_local, rank
    );
    markov_argmax_stage2_k<<<1, 1, 0, stream>>>(blk_vals, blk_ids, out_val, out_id, n_blks);
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
