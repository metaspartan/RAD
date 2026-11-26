#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ void router_mma_m16n8k16_f32(
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
void moe_gate_logits_mma_f16_k(
    const half* __restrict__ x,
    const half* __restrict__ gate_w,
    const float* __restrict__ gate_bias,
    float* __restrict__ raw_scores,
    float* __restrict__ sel_scores,
    int M,
    int n_experts,
    int D
) {
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int e0 = blockIdx.x * 64 + warp * 8;
    int m_base = blockIdx.y * 16;
    if (e0 >= n_experts || m_base >= M) return;

    int group_id = lane >> 2;
    int tid = lane & 3;
    int m0 = m_base + group_id;
    int m1 = m0 + 8;
    int k_off0 = tid * 2;
    int k_off1 = k_off0 + 8;
    int e_row = e0 + group_id;

    const half* x0 = (m0 < M) ? (x + (size_t)m0 * D) : nullptr;
    const half* x1 = (m1 < M) ? (x + (size_t)m1 * D) : nullptr;
    const half* wr = (e_row < n_experts) ? (gate_w + (size_t)e_row * D) : nullptr;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int k = 0; k < D; k += 16) {
        uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
        uint32_t b0 = 0u, b1 = 0u;
        if (x0) {
            a0 = __ldg(reinterpret_cast<const uint32_t*>(x0 + k + k_off0));
            a2 = __ldg(reinterpret_cast<const uint32_t*>(x0 + k + k_off1));
        }
        if (x1) {
            a1 = __ldg(reinterpret_cast<const uint32_t*>(x1 + k + k_off0));
            a3 = __ldg(reinterpret_cast<const uint32_t*>(x1 + k + k_off1));
        }
        if (wr) {
            b0 = __ldg(reinterpret_cast<const uint32_t*>(wr + k + k_off0));
            b1 = __ldg(reinterpret_cast<const uint32_t*>(wr + k + k_off1));
        }
        router_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }

    auto write_sc = [&](int mr, int ec, float v) {
        if (mr < M && ec < n_experts) {
            float sp = (v > 20.0f) ? v : log1pf(expf(v));
            float sc = sqrtf(fmaxf(sp, 0.0f));
            size_t off = (size_t)mr * n_experts + ec;
            raw_scores[off] = sc;
            sel_scores[off] = sc + (gate_bias ? gate_bias[ec] : 0.0f);
        }
    };
    int c0 = e0 + tid * 2;
    write_sc(m0, c0, d0);
    write_sc(m0, c0 + 1, d1);
    write_sc(m1, c0, d2);
    write_sc(m1, c0 + 1, d3);
}

__global__ __launch_bounds__(256)
void mhc_coeff_proj_24_k(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_base,
    const float* __restrict__ hc_scale,
    float* __restrict__ h_coeff,
    float* __restrict__ next_pre,
    int D4,
    float rms_eps,
    float hc_eps
) {
    int mr = blockIdx.x;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int d4_vec = D4 >> 2;
    const float4* sr4 = reinterpret_cast<const float4*>(streams + (size_t)mr * D4);
    const float4* fn4 = reinterpret_cast<const float4*>(hc_fn);

    float ss = 0.0f;
    float dot[24];
    #pragma unroll
    for (int c = 0; c < 24; ++c) dot[c] = 0.0f;

    for (int i = t; i < d4_vec; i += blockDim.x) {
        float4 x = sr4[i];
        ss = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss))));
        #pragma unroll
        for (int c = 0; c < 24; ++c) {
            float4 w = __ldg(fn4 + (size_t)c * d4_vec + i);
            dot[c] = fmaf(x.x, w.x, fmaf(x.y, w.y, fmaf(x.z, w.z, fmaf(x.w, w.w, dot[c]))));
        }
    }

    __shared__ float sm[25][8];
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, m);
    if (lane == 0) sm[24][warp] = ss;
    #pragma unroll
    for (int c = 0; c < 24; ++c) {
        float d = dot[c];
        for (int m = 16; m > 0; m >>= 1) d += __shfl_xor_sync(0xFFFFFFFFu, d, m);
        if (lane == 0) sm[c][warp] = d;
    }
    __syncthreads();

    if (t < 24) {
        float vs = sm[24][0] + sm[24][1] + sm[24][2] + sm[24][3]
                 + sm[24][4] + sm[24][5] + sm[24][6] + sm[24][7];
        float vd = sm[t][0] + sm[t][1] + sm[t][2] + sm[t][3]
                 + sm[t][4] + sm[t][5] + sm[t][6] + sm[t][7];
        float rsqrt_v = rsqrtf(vs / (float)D4 + rms_eps);
        int sc_idx = (t < 4) ? 0 : ((t < 8) ? 1 : 2);
        float val = (vd * rsqrt_v) * hc_scale[sc_idx] + hc_base[t];
        h_coeff[(size_t)mr * 24 + t] = val;
        if (t < 4 && next_pre) {
            next_pre[(size_t)mr * 4 + t] = 1.0f / (1.0f + expf(-val)) + hc_eps;
        }
    }
}

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
    float* __restrict__ next_pre,
    int D4,
    float rms_eps,
    float hc_eps
) {
    int c = blockIdx.x;
    int mr = blockIdx.y;
    if (c >= 24) return;
    int t = threadIdx.x;
    const float4* sr4 = reinterpret_cast<const float4*>(streams + (size_t)mr * D4);
    const float4* row4 = reinterpret_cast<const float4*>(hc_fn + (size_t)c * D4);
    int d4_vec = D4 >> 2;
    float dot = 0.0f;
    float ss = 0.0f;
    for (int i = t; i < d4_vec; i += blockDim.x) {
        float4 x = sr4[i];
        float4 w = row4[i];
        ss = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss))));
        dot = fmaf(x.x, w.x, fmaf(x.y, w.y, fmaf(x.z, w.z, fmaf(x.w, w.w, dot))));
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
            float rsqrt_v = rsqrtf(vs / (float)D4 + rms_eps);
            int sc_idx = (c < 4) ? 0 : ((c < 8) ? 1 : 2);
            float val = (vd * rsqrt_v) * hc_scale[sc_idx] + hc_base[c];
            h_coeff[(size_t)mr * 24 + c] = val;
            if (c < 4 && next_pre) {
                next_pre[(size_t)mr * 4 + c] = 1.0f / (1.0f + expf(-val)) + hc_eps;
            }
        }
    }
}

__global__ void mhc_pre_mix_f16_k(
    const float* __restrict__ streams,
    const float* __restrict__ prev_pre,
    float* __restrict__ branch_in,
    int D
) {
    int mr = blockIdx.y;
    __shared__ float s_pre[4];
    if (threadIdx.x < 4) {
        if (prev_pre) {
            s_pre[threadIdx.x] = prev_pre[(size_t)mr * 4 + threadIdx.x];
        } else {
            s_pre[threadIdx.x] = (threadIdx.x == 0) ? 1.0f : 0.0f;
        }
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
    if (M >= 16 && (D % 16) == 0) {
        dim3 g1((n_experts + 63) / 64, (M + 15) / 16);
        moe_gate_logits_mma_f16_k<<<g1, 256, 0, stream>>>(x, gate_w, gate_bias, raw_scores, sel_scores, M, n_experts, D);
    } else {
        dim3 g1(n_experts, M);
        moe_gate_logits_f16_k<<<g1, 256, 0, stream>>>(x, gate_w, gate_bias, raw_scores, sel_scores, n_experts, D);
    }
    moe_topk_select_k<<<M, 256, 0, stream>>>(raw_scores, sel_scores, topk_ids, topk_weights, n_experts, top_k, scaling_factor);
}

void rad_cuda_mhc_pre_and_coeff(
    const float* streams,
    const float* hc_fn,
    const float* hc_base,
    const float* hc_scale,
    const float* prev_pre,
    float* h_coeff,
    float* next_pre,
    float* branch_in,
    int M,
    int D,
    float rms_eps,
    float hc_eps,
    cudaStream_t stream
) {
    if (M >= 16) {
        mhc_coeff_proj_24_k<<<M, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    } else {
        dim3 g1(24, M);
        mhc_coeff_proj_k<<<g1, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    }
    dim3 g2((D + 255) / 256, M);
    mhc_pre_mix_f16_k<<<g2, 256, 0, stream>>>(streams, prev_pre, branch_in, D);
}

void rad_cuda_mhc_pre_mix_only(
    const float* streams,
    const float* prev_pre,
    float* branch_in,
    int M,
    int D,
    cudaStream_t stream
) {
    dim3 g2((D + 255) / 256, M);
    mhc_pre_mix_f16_k<<<g2, 256, 0, stream>>>(streams, prev_pre, branch_in, D);
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
