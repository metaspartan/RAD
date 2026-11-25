#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

__global__ void moe_gate_logits_f16_k(
    const half* __restrict__ x,
    const half* __restrict__ gate_w,
    const float* __restrict__ gate_bias,
    float* __restrict__ raw_scores,
    float* __restrict__ sel_scores,
    int n_experts,
    int D
) {
    int e = blockIdx.x;
    if (e >= n_experts) return;
    int t = threadIdx.x;
    const half* we = gate_w + (size_t)e * D;
    float dot = 0.0f;
    for (int i = t; i < D; i += blockDim.x) {
        dot = fmaf(__half2float(x[i]), __half2float(we[i]), dot);
    }
    int lane = t & 31;
    int warp = t >> 5;
    __shared__ float sm[32];
    for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
    if (lane == 0) sm[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
        if (lane == 0) {
            float sp = (v > 20.0f) ? v : log1pf(expf(v));
            float sc = sqrtf(fmaxf(sp, 0.0f));
            raw_scores[e] = sc;
            sel_scores[e] = sc + (gate_bias ? gate_bias[e] : 0.0f);
        }
    }
}

__global__ void moe_topk_select_k(
    const float* __restrict__ raw_scores,
    const float* __restrict__ sel_scores,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int n_experts,
    int top_k,
    float scaling_factor
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int chosen[8];
    float weights[8];
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int best_e = -1;
        float best_v = -1e30f;
        for (int e = 0; e < n_experts; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j) {
                if (chosen[j] == e) {
                    used = true;
                    break;
                }
            }
            if (!used) {
                float v = sel_scores[e];
                if (v > best_v) {
                    best_v = v;
                    best_e = e;
                }
            }
        }
        chosen[k] = best_e;
        float w = raw_scores[best_e];
        weights[k] = w;
        wsum += w;
    }
    float inv = scaling_factor / fmaxf(wsum, 1e-20f);
    for (int k = 0; k < top_k; ++k) {
        topk_ids[k] = chosen[k];
        topk_weights[k] = weights[k] * inv;
    }
}

__global__ void mhc_coeff_proj_k(
    const half* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_base,
    const float* __restrict__ hc_scale,
    float* __restrict__ h_coeff,
    int D4
) {
    int c = blockIdx.x;
    if (c >= 24) return;
    int t = threadIdx.x;
    const float* row = hc_fn + (size_t)c * D4;
    float dot = 0.0f;
    for (int i = t; i < D4; i += blockDim.x) {
        dot = fmaf(__half2float(streams[i]), row[i], dot);
    }
    int lane = t & 31;
    int warp = t >> 5;
    __shared__ float sm[32];
    for (int m = 16; m > 0; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
    if (lane == 0) sm[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
        if (lane == 0) {
            int sc_idx = (c < 4) ? 0 : ((c < 8) ? 1 : 2);
            h_coeff[c] = v * hc_scale[sc_idx] + hc_base[c];
        }
    }
}

__global__ void mhc_pre_mix_f16_k(
    const half* __restrict__ streams,
    const float* __restrict__ h_coeff,
    half* __restrict__ branch_in,
    int D
) {
    __shared__ float s_pre[4];
    if (threadIdx.x < 4) {
        s_pre[threadIdx.x] = 1.0f / (1.0f + expf(-h_coeff[threadIdx.x]));
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        float acc = s_pre[0] * __half2float(streams[0 * D + i]) +
                    s_pre[1] * __half2float(streams[1 * D + i]) +
                    s_pre[2] * __half2float(streams[2 * D + i]) +
                    s_pre[3] * __half2float(streams[3 * D + i]);
        branch_in[i] = __float2half(acc);
    }
}

__global__ void argmax_f16_stage1_k(
    const half* __restrict__ logits,
    float* __restrict__ blk_vals,
    int* __restrict__ blk_ids,
    int N
) {
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 4;
    float bv = -1e30f;
    int bi = 0;
    for (int j = 0; j < 4; ++j) {
        int idx = base + j * blockDim.x + t;
        if (idx < N) {
            float v = __half2float(logits[idx]);
            if (v > bv) {
                bv = v;
                bi = idx;
            }
        }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[t] = bv;
    si[t] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s && sv[t + s] > sv[t]) {
            sv[t] = sv[t + s];
            si[t] = si[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        blk_vals[blockIdx.x] = sv[0];
        blk_ids[blockIdx.x] = si[0];
    }
}

__global__ void argmax_f16_stage2_k(
    const float* __restrict__ blk_vals,
    const int* __restrict__ blk_ids,
    float* __restrict__ out_val,
    int* __restrict__ out_id,
    int n_blks
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float bv = -1e30f;
    int bi = 0;
    for (int i = 0; i < n_blks; ++i) {
        if (blk_vals[i] > bv) {
            bv = blk_vals[i];
            bi = blk_ids[i];
        }
    }
    *out_val = bv;
    *out_id = bi;
}

__global__ void add_f16_k(half* __restrict__ dst, const half* __restrict__ src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
    }
}

extern "C" {

void rad_cuda_moe_route_sqrtsoftplus(
    const half* x,
    const half* gate_w,
    const float* gate_bias,
    float* raw_scores,
    float* sel_scores,
    int* topk_ids,
    float* topk_weights,
    int n_experts,
    int D,
    int top_k,
    float scaling_factor,
    cudaStream_t stream
) {
    moe_gate_logits_f16_k<<<n_experts, 256, 0, stream>>>(x, gate_w, gate_bias, raw_scores, sel_scores, n_experts, D);
    moe_topk_select_k<<<1, 1, 0, stream>>>(raw_scores, sel_scores, topk_ids, topk_weights, n_experts, top_k, scaling_factor);
}

void rad_cuda_mhc_pre_and_coeff(
    const half* streams,
    const float* hc_fn,
    const float* hc_base,
    const float* hc_scale,
    float* h_coeff,
    half* branch_in,
    int D,
    cudaStream_t stream
) {
    mhc_coeff_proj_k<<<24, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, D * 4);
    mhc_pre_mix_f16_k<<<(D + 255) / 256, 256, 0, stream>>>(streams, h_coeff, branch_in, D);
}

void rad_cuda_argmax_f16(
    const half* logits,
    float* blk_vals,
    int* blk_ids,
    float* out_val,
    int* out_id,
    int N,
    cudaStream_t stream
) {
    int n_blks = (N + 1023) / 1024;
    argmax_f16_stage1_k<<<n_blks, 256, 0, stream>>>(logits, blk_vals, blk_ids, N);
    argmax_f16_stage2_k<<<1, 1, 0, stream>>>(blk_vals, blk_ids, out_val, out_id, n_blks);
}

void rad_cuda_add_f16(half* dst, const half* src, int N, cudaStream_t stream) {
    add_f16_k<<<(N + 255) / 256, 256, 0, stream>>>(dst, src, N);
}

}
