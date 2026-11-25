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
        v_cache[(size_t)p * head_dim + i] = v_in[(size_t)mr * head_dim + i];
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
    float sign
) {
    int h = blockIdx.x;
    int mr = blockIdx.y;
    int i = threadIdx.x;
    int half_r = rope_dim >> 1;
    if (i >= half_r) return;
    float base_freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
    float factor = 16.0f;
    float orig_ctx = 65536.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    float low = floorf((float)rope_dim * logf(orig_ctx / (beta_fast * 2.0f * 3.14159265f)) / (2.0f * logf(theta)));
    float high = ceilf((float)rope_dim * logf(orig_ctx / (beta_slow * 2.0f * 3.14159265f)) / (2.0f * logf(theta)));
    if (low < 0.0f) low = 0.0f;
    if (high > (float)(rope_dim - 1)) high = (float)(rope_dim - 1);
    float ramp = ((float)i - low) / fmaxf(high - low, 0.001f);
    if (ramp < 0.0f) ramp = 0.0f;
    if (ramp > 1.0f) ramp = 1.0f;
    float freq = (base_freq / factor) * ramp + base_freq * (1.0f - ramp);
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
            for (int c = 0; c < 4; ++c) {
                m[r][c] = expf(hc[8 + r * 4 + c] - rmax);
            }
        }
        for (int it = 0; it < sinkhorn_iters; ++it) {
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
            sr[r * D + i] = s_res[r * 4 + 0] * x0 + s_res[r * 4 + 1] * x1 +
                            s_res[r * 4 + 2] * x2 + s_res[r * 4 + 3] * x3 +
                            s_post[r] * b;
        }
    }
}

__global__ void mqa_attn_sink_causal_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const float* __restrict__ attn_sink,
    half* __restrict__ out,
    int n_heads,
    int start_pos,
    int max_ctx,
    int sliding_window,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    const half* qh = q + ((size_t)mr * n_heads + h) * head_dim;
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_acc = smem + head_dim;
    __shared__ float wsum[32];
    for (int d = t; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(qh[d]);
        s_acc[d] = 0.0f;
    }
    __syncthreads();
    int seq_len = start_pos + mr + 1;
    if (seq_len > max_ctx) seq_len = max_ctx;
    int p_start = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;
    float m_max = attn_sink ? attn_sink[h] : -1e30f;
    float l_sum = attn_sink ? 1.0f : 0.0f;
    int lane = t & 31;
    int warp = t >> 5;
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
            float v = (lane < (blockDim.x >> 5)) ? wsum[lane] : 0.0f;
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
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
    float sign,
    cudaStream_t stream
) {
    dim3 grid(n_q_heads + n_kv_heads, M);
    rope_yarn_f16_k<<<grid, rope_dim / 2, 0, stream>>>(
        q, k, n_q_heads, n_kv_heads, head_dim, rope_dim, start_pos, theta, sign
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
    const float* attn_sink,
    half* out,
    int M,
    int n_heads,
    int start_pos,
    int max_ctx,
    int sliding_window,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    size_t smem = (size_t)head_dim * 2 * sizeof(float);
    dim3 grid(n_heads, M);
    mqa_attn_sink_causal_f16_k<<<grid, 256, smem, stream>>>(
        q, k_cache, v_cache, attn_sink, out, n_heads, start_pos, max_ctx, sliding_window, head_dim, scale
    );
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
