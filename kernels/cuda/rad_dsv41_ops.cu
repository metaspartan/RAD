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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        float acc = shared_out ? __half2float(shared_out[i]) : 0.0f;
        for (int k = 0; k < top_k; ++k) {
            acc += weights[k] * __half2float(expert_outs[(size_t)k * D + i]);
        }
        dst[i] = __float2half(acc);
    }
}

__global__ void embed_bf16_to_f16_k(
    const __nv_bfloat16* __restrict__ table,
    const int* __restrict__ tok_ids,
    half* __restrict__ out,
    int D
) {
    int row = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        int tid = tok_ids[row];
        out[(size_t)row * D + i] = __float2half(__bfloat162float(table[(size_t)tid * D + i]));
    }
}

__global__ void rope_yarn_f16_k(
    half* __restrict__ q,
    half* __restrict__ k,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rope_dim,
    int pos,
    float theta
) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    int half_r = rope_dim >> 1;
    if (i >= half_r) return;
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
    float ang = (float)pos * freq;
    float cs = cosf(ang);
    float sn = sinf(ang);
    int rope_off = head_dim - rope_dim;
    if (h < n_q_heads) {
        half* p = q + (size_t)h * head_dim + rope_off;
        float x0 = __half2float(p[2 * i]);
        float x1 = __half2float(p[2 * i + 1]);
        p[2 * i] = __float2half(x0 * cs - x1 * sn);
        p[2 * i + 1] = __float2half(x0 * sn + x1 * cs);
    } else {
        int kh = h - n_q_heads;
        if (kh < n_kv_heads) {
            half* p = k + (size_t)kh * head_dim + rope_off;
            float x0 = __half2float(p[2 * i]);
            float x1 = __half2float(p[2 * i + 1]);
            p[2 * i] = __float2half(x0 * cs - x1 * sn);
            p[2 * i + 1] = __float2half(x0 * sn + x1 * cs);
        }
    }
}

__global__ void mhc_sinkhorn_mix_f16_k(
    half* __restrict__ streams,
    const half* __restrict__ branch_out,
    const float* __restrict__ h_coeff,
    int D,
    int sinkhorn_iters,
    float eps
) {
    __shared__ float s_pre[4];
    __shared__ float s_post[4];
    __shared__ float s_res[16];
    int t = threadIdx.x;
    if (t == 0) {
        for (int i = 0; i < 4; ++i) {
            s_pre[i] = 1.0f / (1.0f + expf(-h_coeff[i]));
            s_post[i] = 2.0f / (1.0f + expf(-h_coeff[4 + i]));
        }
        float m[4][4];
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                m[r][c] = expf(h_coeff[8 + r * 4 + c]);
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
        float x0 = __half2float(streams[0 * D + i]);
        float x1 = __half2float(streams[1 * D + i]);
        float x2 = __half2float(streams[2 * D + i]);
        float x3 = __half2float(streams[3 * D + i]);
        float b = __half2float(branch_out[i]);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float nr = s_res[r * 4 + 0] * x0 + s_res[r * 4 + 1] * x1 +
                       s_res[r * 4 + 2] * x2 + s_res[r * 4 + 3] * x3 +
                       s_post[r] * b;
            streams[r * D + i] = __float2half(nr);
        }
    }
}

__global__ void mqa_attn_sink_decode_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const float* __restrict__ attn_sink,
    half* __restrict__ out,
    int seq_len,
    int head_dim,
    float scale
) {
    int h = blockIdx.x;
    int t = threadIdx.x;
    const half* qh = q + (size_t)h * head_dim;
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_acc = smem + head_dim;
    for (int d = t; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(qh[d]);
        s_acc[d] = 0.0f;
    }
    __syncthreads();
    float m_max = attn_sink ? attn_sink[h] : -1e30f;
    float l_sum = attn_sink ? 1.0f : 0.0f;
    for (int p = 0; p < seq_len; ++p) {
        const half* kp = k_cache + (size_t)p * head_dim;
        float dot = 0.0f;
        for (int d = t; d < head_dim; d += blockDim.x) {
            dot += s_q[d] * __half2float(kp[d]);
        }
        int lane = t & 31;
        int warp = t >> 5;
        __shared__ float wsum[32];
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
    half* oh = out + (size_t)h * head_dim;
    for (int d = t; d < head_dim; d += blockDim.x) {
        oh[d] = __float2half(s_acc[d] * inv);
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
    int top_k,
    int D,
    cudaStream_t stream
) {
    moe_weighted_sum_f16_k<<<(D + 255) / 256, 256, 0, stream>>>(expert_outs, weights, shared_out, dst, top_k, D);
}

void rad_cuda_embed_bf16_to_f16(
    const __nv_bfloat16* table,
    const int* tok_ids,
    half* out,
    int M,
    int D,
    cudaStream_t stream
) {
    dim3 grid((D + 255) / 256, M);
    embed_bf16_to_f16_k<<<grid, 256, 0, stream>>>(table, tok_ids, out, D);
}

void rad_cuda_rope_yarn_f16(
    half* q,
    half* k,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rope_dim,
    int pos,
    float theta,
    cudaStream_t stream
) {
    rope_yarn_f16_k<<<n_q_heads + n_kv_heads, rope_dim / 2, 0, stream>>>(
        q, k, n_q_heads, n_kv_heads, head_dim, rope_dim, pos, theta
    );
}

void rad_cuda_mhc_sinkhorn_mix_f16(
    half* streams,
    const half* branch_out,
    const float* h_coeff,
    int D,
    int sinkhorn_iters,
    float eps,
    cudaStream_t stream
) {
    mhc_sinkhorn_mix_f16_k<<<(D + 255) / 256, 256, 0, stream>>>(
        streams, branch_out, h_coeff, D, sinkhorn_iters, eps
    );
}

void rad_cuda_mqa_attn_sink_decode_f16(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    const float* attn_sink,
    half* out,
    int n_heads,
    int seq_len,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    size_t smem = (size_t)head_dim * 2 * sizeof(float);
    mqa_attn_sink_decode_f16_k<<<n_heads, 256, smem, stream>>>(
        q, k_cache, v_cache, attn_sink, out, seq_len, head_dim, scale
    );
}

}
