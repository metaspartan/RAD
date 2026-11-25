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
    int mr = blockIdx.y;
    if (e >= n_experts) return;
    int t = threadIdx.x;
    const half2* xr2 = reinterpret_cast<const half2*>(x + (size_t)mr * D);
    const half2* we2 = reinterpret_cast<const half2*>(gate_w + (size_t)e * D);
    int d2 = D >> 1;
    float dot = 0.0f;
    for (int i = t; i < d2; i += blockDim.x) {
        float2 xv = __half22float2(xr2[i]);
        float2 wv = __half22float2(we2[i]);
        dot = fmaf(xv.x, wv.x, dot);
        dot = fmaf(xv.y, wv.y, dot);
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
            size_t off = (size_t)mr * n_experts + e;
            raw_scores[off] = sc;
            sel_scores[off] = sc + (gate_bias ? gate_bias[e] : 0.0f);
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
    int mr = blockIdx.x;
    int t = threadIdx.x;
    const float* rs = raw_scores + (size_t)mr * n_experts;
    const float* ss = sel_scores + (size_t)mr * n_experts;
    int* ids_out = topk_ids + (size_t)mr * top_k;
    float* w_out = topk_weights + (size_t)mr * top_k;

    __shared__ float s_sel[512];
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    __shared__ int chosen[8];
    __shared__ float weights[8];

    for (int e = t; e < n_experts; e += blockDim.x) {
        s_sel[e] = ss[e];
    }
    __syncthreads();

    for (int k = 0; k < top_k; ++k) {
        float bv = -1e30f;
        int bi = 0;
        for (int e = t; e < n_experts; e += blockDim.x) {
            float v = s_sel[e];
            if (v > bv) {
                bv = v;
                bi = e;
            }
        }
        s_val[t] = bv;
        s_idx[t] = bi;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (t < stride && s_val[t + stride] > s_val[t]) {
                s_val[t] = s_val[t + stride];
                s_idx[t] = s_idx[t + stride];
            }
            __syncthreads();
        }
        if (t == 0) {
            int best_e = s_idx[0];
            chosen[k] = best_e;
            weights[k] = rs[best_e];
            s_sel[best_e] = -1e30f;
        }
        __syncthreads();
    }
    if (t == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < top_k; ++k) wsum += weights[k];
        float inv = scaling_factor / fmaxf(wsum, 1e-20f);
        for (int k = 0; k < top_k; ++k) {
            ids_out[k] = chosen[k];
            w_out[k] = weights[k] * inv;
        }
    }
}

__global__ void mhc_coeff_proj_k(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_base,
    const float* __restrict__ hc_scale,
    float* __restrict__ h_coeff,
    int D4,
    float eps
) {
    int c = blockIdx.x;
    int mr = blockIdx.y;
    if (c >= 24) return;
    int t = threadIdx.x;
    const float* sr = streams + (size_t)mr * D4;
    int D = D4 >> 2;
    const float4* row4 = reinterpret_cast<const float4*>(hc_fn + (size_t)c * D4);
    float dot = 0.0f;
    float ss = 0.0f;
    for (int i = t; i < D; i += blockDim.x) {
        float x0 = sr[0 * D + i];
        float x1 = sr[1 * D + i];
        float x2 = sr[2 * D + i];
        float x3 = sr[3 * D + i];
        float4 w = row4[i];
        ss = fmaf(x0, x0, fmaf(x1, x1, fmaf(x2, x2, fmaf(x3, x3, ss))));
        dot = fmaf(x0, w.x, fmaf(x1, w.y, fmaf(x2, w.z, fmaf(x3, w.w, dot))));
    }
    int lane = t & 31;
    int warp = t >> 5;
    __shared__ float sm_dot[32];
    __shared__ float sm_ss[32];
    for (int m = 16; m > 0; m >>= 1) {
        dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, m);
    }
    if (lane == 0) {
        sm_dot[warp] = dot;
        sm_ss[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        float vd = (lane < (blockDim.x >> 5)) ? sm_dot[lane] : 0.0f;
        float vs = (lane < (blockDim.x >> 5)) ? sm_ss[lane] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) {
            vd += __shfl_xor_sync(0xFFFFFFFFu, vd, m);
            vs += __shfl_xor_sync(0xFFFFFFFFu, vs, m);
        }
        if (lane == 0) {
            float rsqrt_v = rsqrtf(vs / (float)D4 + eps);
            int sc_idx = (c < 4) ? 0 : ((c < 8) ? 1 : 2);
            h_coeff[(size_t)mr * 24 + c] = (vd * rsqrt_v) * hc_scale[sc_idx] + hc_base[c];
        }
    }
}

__global__ void mhc_pre_mix_f16_k(
    const float* __restrict__ streams,
    const float* __restrict__ h_coeff,
    float* __restrict__ branch_in,
    int D,
    float eps
) {
    int mr = blockIdx.y;
    __shared__ float s_pre[4];
    if (threadIdx.x < 4) {
        s_pre[threadIdx.x] = 1.0f / (1.0f + expf(-h_coeff[(size_t)mr * 24 + threadIdx.x])) + eps;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        const float* sr = streams + (size_t)mr * 4 * D;
        branch_in[(size_t)mr * D + i] = s_pre[0] * sr[0 * D + i] +
                                        s_pre[1] * sr[1 * D + i] +
                                        s_pre[2] * sr[2 * D + i] +
                                        s_pre[3] * sr[3 * D + i];
    }
}

__global__ void argmax_f16_stage1_k(
    const half* __restrict__ logits,
    float* __restrict__ blk_vals,
    int* __restrict__ blk_ids,
    int N,
    int n_blks
) {
    int mr = blockIdx.y;
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 4;
    const half* lr = logits + (size_t)mr * N;
    float bv = -1e30f;
    int bi = 0;
    for (int j = 0; j < 4; ++j) {
        int idx = base + j * blockDim.x + t;
        if (idx < N) {
            float v = __half2float(lr[idx]);
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
        blk_vals[(size_t)mr * n_blks + blockIdx.x] = sv[0];
        blk_ids[(size_t)mr * n_blks + blockIdx.x] = si[0];
    }
}

__global__ void argmax_f16_stage2_k(
    const float* __restrict__ blk_vals,
    const int* __restrict__ blk_ids,
    float* __restrict__ out_val,
    int* __restrict__ out_id,
    int n_blks
) {
    int mr = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float* bv_row = blk_vals + (size_t)mr * n_blks;
    const int* bi_row = blk_ids + (size_t)mr * n_blks;
    float bv = -1e30f;
    int bi = 0;
    for (int i = 0; i < n_blks; ++i) {
        if (bv_row[i] > bv) {
            bv = bv_row[i];
            bi = bi_row[i];
        }
    }
    out_val[mr] = bv;
    out_id[mr] = bi;
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
    int M,
    int n_experts,
    int D,
    int top_k,
    float scaling_factor,
    cudaStream_t stream
) {
    dim3 g1(n_experts, M);
    moe_gate_logits_f16_k<<<g1, 256, 0, stream>>>(x, gate_w, gate_bias, raw_scores, sel_scores, n_experts, D);
    moe_topk_select_k<<<M, 256, 0, stream>>>(raw_scores, sel_scores, topk_ids, topk_weights, n_experts, top_k, scaling_factor);
}

void rad_cuda_mhc_pre_and_coeff(
    const float* streams,
    const float* hc_fn,
    const float* hc_base,
    const float* hc_scale,
    float* h_coeff,
    float* branch_in,
    int M,
    int D,
    float eps,
    cudaStream_t stream
) {
    dim3 g1(24, M);
    mhc_coeff_proj_k<<<g1, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, D * 4, eps);
    dim3 g2((D + 255) / 256, M);
    mhc_pre_mix_f16_k<<<g2, 256, 0, stream>>>(streams, h_coeff, branch_in, D, eps);
}

void rad_cuda_argmax_f16(
    const half* logits,
    float* blk_vals,
    int* blk_ids,
    float* out_val,
    int* out_id,
    int M,
    int N,
    cudaStream_t stream
) {
    int n_blks = (N + 1023) / 1024;
    dim3 g1(n_blks, M);
    argmax_f16_stage1_k<<<g1, 256, 0, stream>>>(logits, blk_vals, blk_ids, N, n_blks);
    argmax_f16_stage2_k<<<M, 1, 0, stream>>>(blk_vals, blk_ids, out_val, out_id, n_blks);
}

void rad_cuda_add_f16(half* dst, const half* src, int N, cudaStream_t stream) {
    add_f16_k<<<(N + 255) / 256, 256, 0, stream>>>(dst, src, N);
}

}
