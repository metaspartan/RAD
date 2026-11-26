#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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
    if (m_base >= M) return;

    int group_id = lane >> 2;
    int tid = lane & 3;
    int m0 = m_base + group_id;
    int m1 = m0 + 8;
    int k_off0 = tid * 2;
    int k_off1 = k_off0 + 8;
    int e_row = e0 + group_id;
    const half* wr = (e_row < n_experts) ? (gate_w + (size_t)e_row * D) : nullptr;

    __shared__ alignas(16) half s_x[16][16];
    int ld_row = t >> 2;
    int ld_col = (t & 3) * 4;
    int ld_m = m_base + ld_row;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int k = 0; k < D; k += 16) {
        if (ld_row < 16) {
            if (ld_m < M) {
                *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = *reinterpret_cast<const uint2*>(x + (size_t)ld_m * D + k + ld_col);
            } else {
                *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = make_uint2(0u, 0u);
            }
        }
        __syncthreads();
        if (e0 < n_experts) {
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_x[group_id][k_off0]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_x[group_id][k_off1]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_x[group_id + 8][k_off0]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_x[group_id + 8][k_off1]);
            uint32_t b0 = 0u, b1 = 0u;
            if (wr) {
                b0 = __ldg(reinterpret_cast<const uint32_t*>(wr + k + k_off0));
                b1 = __ldg(reinterpret_cast<const uint32_t*>(wr + k + k_off1));
            }
            router_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        }
        __syncthreads();
    }
    if (e0 >= n_experts) return;

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
void mhc_coeff_proj_24_4tok_k(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_base,
    const float* __restrict__ hc_scale,
    float* __restrict__ h_coeff,
    float* __restrict__ next_pre,
    int M,
    int D4,
    float rms_eps,
    float hc_eps
) {
    int m0 = blockIdx.x * 4;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int c0 = warp * 3;
    int d4_vec = D4 >> 2;
    const float4* fn4 = reinterpret_cast<const float4*>(hc_fn);
    const float4* sr0 = (m0 + 0 < M) ? reinterpret_cast<const float4*>(streams + (size_t)(m0 + 0) * D4) : nullptr;
    const float4* sr1 = (m0 + 1 < M) ? reinterpret_cast<const float4*>(streams + (size_t)(m0 + 1) * D4) : nullptr;
    const float4* sr2 = (m0 + 2 < M) ? reinterpret_cast<const float4*>(streams + (size_t)(m0 + 2) * D4) : nullptr;
    const float4* sr3 = (m0 + 3 < M) ? reinterpret_cast<const float4*>(streams + (size_t)(m0 + 3) * D4) : nullptr;

    float dot[4][3];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        dot[r][0] = 0.0f; dot[r][1] = 0.0f; dot[r][2] = 0.0f;
    }
    float ss0 = 0.0f, ss1 = 0.0f, ss2 = 0.0f, ss3 = 0.0f;

    for (int i = lane; i < d4_vec; i += 32) {
        float4 w0 = __ldg(fn4 + (size_t)(c0 + 0) * d4_vec + i);
        float4 w1 = __ldg(fn4 + (size_t)(c0 + 1) * d4_vec + i);
        float4 w2 = __ldg(fn4 + (size_t)(c0 + 2) * d4_vec + i);
        if (sr0) {
            float4 x = sr0[i];
            if (warp == 0) ss0 = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss0))));
            dot[0][0] = fmaf(x.x, w0.x, fmaf(x.y, w0.y, fmaf(x.z, w0.z, fmaf(x.w, w0.w, dot[0][0]))));
            dot[0][1] = fmaf(x.x, w1.x, fmaf(x.y, w1.y, fmaf(x.z, w1.z, fmaf(x.w, w1.w, dot[0][1]))));
            dot[0][2] = fmaf(x.x, w2.x, fmaf(x.y, w2.y, fmaf(x.z, w2.z, fmaf(x.w, w2.w, dot[0][2]))));
        }
        if (sr1) {
            float4 x = sr1[i];
            if (warp == 0) ss1 = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss1))));
            dot[1][0] = fmaf(x.x, w0.x, fmaf(x.y, w0.y, fmaf(x.z, w0.z, fmaf(x.w, w0.w, dot[1][0]))));
            dot[1][1] = fmaf(x.x, w1.x, fmaf(x.y, w1.y, fmaf(x.z, w1.z, fmaf(x.w, w1.w, dot[1][1]))));
            dot[1][2] = fmaf(x.x, w2.x, fmaf(x.y, w2.y, fmaf(x.z, w2.z, fmaf(x.w, w2.w, dot[1][2]))));
        }
        if (sr2) {
            float4 x = sr2[i];
            if (warp == 0) ss2 = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss2))));
            dot[2][0] = fmaf(x.x, w0.x, fmaf(x.y, w0.y, fmaf(x.z, w0.z, fmaf(x.w, w0.w, dot[2][0]))));
            dot[2][1] = fmaf(x.x, w1.x, fmaf(x.y, w1.y, fmaf(x.z, w1.z, fmaf(x.w, w1.w, dot[2][1]))));
            dot[2][2] = fmaf(x.x, w2.x, fmaf(x.y, w2.y, fmaf(x.z, w2.z, fmaf(x.w, w2.w, dot[2][2]))));
        }
        if (sr3) {
            float4 x = sr3[i];
            if (warp == 0) ss3 = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, ss3))));
            dot[3][0] = fmaf(x.x, w0.x, fmaf(x.y, w0.y, fmaf(x.z, w0.z, fmaf(x.w, w0.w, dot[3][0]))));
            dot[3][1] = fmaf(x.x, w1.x, fmaf(x.y, w1.y, fmaf(x.z, w1.z, fmaf(x.w, w1.w, dot[3][1]))));
            dot[3][2] = fmaf(x.x, w2.x, fmaf(x.y, w2.y, fmaf(x.z, w2.z, fmaf(x.w, w2.w, dot[3][2]))));
        }
    }

    __shared__ float s_rsqrt[4];
    if (warp == 0) {
        for (int m = 16; m > 0; m >>= 1) {
            ss0 += __shfl_xor_sync(0xFFFFFFFFu, ss0, m);
            ss1 += __shfl_xor_sync(0xFFFFFFFFu, ss1, m);
            ss2 += __shfl_xor_sync(0xFFFFFFFFu, ss2, m);
            ss3 += __shfl_xor_sync(0xFFFFFFFFu, ss3, m);
        }
        if (lane == 0) {
            s_rsqrt[0] = rsqrtf(ss0 / (float)D4 + rms_eps);
            s_rsqrt[1] = rsqrtf(ss1 / (float)D4 + rms_eps);
            s_rsqrt[2] = rsqrtf(ss2 / (float)D4 + rms_eps);
            s_rsqrt[3] = rsqrtf(ss3 / (float)D4 + rms_eps);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (m0 + r >= M) break;
        float rsq = s_rsqrt[r];
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            float d = dot[r][j];
            for (int m = 16; m > 0; m >>= 1) d += __shfl_xor_sync(0xFFFFFFFFu, d, m);
            if (lane == 0) {
                int c = c0 + j;
                int sc_idx = (c < 4) ? 0 : ((c < 8) ? 1 : 2);
                float val = (d * rsq) * hc_scale[sc_idx] + hc_base[c];
                h_coeff[(size_t)(m0 + r) * 24 + c] = val;
                if (c < 4 && next_pre) {
                    next_pre[(size_t)(m0 + r) * 4 + c] = 1.0f / (1.0f + expf(-val)) + hc_eps;
                }
            }
        }
    }
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

__global__ void mhc_pre_mix_rmsnorm_f16_k(
    const float* __restrict__ streams,
    const float* __restrict__ prev_pre,
    const __nv_bfloat16* __restrict__ norm_w,
    half* __restrict__ y_out,
    int D,
    float eps
) {
    int mr = blockIdx.x;
    int t = threadIdx.x;
    __shared__ float s_pre[4];
    __shared__ float sm[8];
    if (t < 4) {
        if (prev_pre) {
            s_pre[t] = prev_pre[(size_t)mr * 4 + t];
        } else {
            s_pre[t] = (t == 0) ? 1.0f : 0.0f;
        }
    }
    __syncthreads();
    float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    int d4 = D >> 2;
    const float4* sr4 = reinterpret_cast<const float4*>(streams + (size_t)mr * 4 * D);
    float4 vals[6];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i4 = t + k * 256;
        if (i4 < d4) {
            float4 a = sr4[0 * d4 + i4];
            float4 b = sr4[1 * d4 + i4];
            float4 c = sr4[2 * d4 + i4];
            float4 d = sr4[3 * d4 + i4];
            float4 v;
            v.x = p0 * a.x + p1 * b.x + p2 * c.x + p3 * d.x;
            v.y = p0 * a.y + p1 * b.y + p2 * c.y + p3 * d.y;
            v.z = p0 * a.z + p1 * b.z + p2 * c.z + p3 * d.z;
            v.w = p0 * a.w + p1 * b.w + p2 * c.w + p3 * d.w;
            vals[k] = v;
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
    }
    for (int i4 = t + 6 * 256; i4 < d4; i4 += 256) {
        float4 a = sr4[0 * d4 + i4];
        float4 b = sr4[1 * d4 + i4];
        float4 c = sr4[2 * d4 + i4];
        float4 d = sr4[3 * d4 + i4];
        float vx = p0 * a.x + p1 * b.x + p2 * c.x + p3 * d.x;
        float vy = p0 * a.y + p1 * b.y + p2 * c.y + p3 * d.y;
        float vz = p0 * a.z + p1 * b.z + p2 * c.z + p3 * d.z;
        float vw = p0 * a.w + p1 * b.w + p2 * c.w + p3 * d.w;
        ss = fmaf(vx, vx, fmaf(vy, vy, fmaf(vz, vz, fmaf(vw, vw, ss))));
    }
    int lane = t & 31;
    int warp = t >> 5;
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, m);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? sm[lane] : 0.0f;
        for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
        if (lane == 0) sm[0] = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();
    float inv = sm[0];
    const uint2* nw2 = reinterpret_cast<const uint2*>(norm_w);
    uint2* y2 = reinterpret_cast<uint2*>(y_out + (size_t)mr * D);
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i4 = t + k * 256;
        if (i4 < d4) {
            uint2 wq = __ldg(nw2 + i4);
            float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.x));
            float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.y));
            float4 v = vals[k];
            half2 h01 = __float22half2_rn(make_float2(v.x * inv * w01.x, v.y * inv * w01.y));
            half2 h23 = __float22half2_rn(make_float2(v.z * inv * w23.x, v.w * inv * w23.y));
            uint2 oq;
            oq.x = *reinterpret_cast<const uint32_t*>(&h01);
            oq.y = *reinterpret_cast<const uint32_t*>(&h23);
            y2[i4] = oq;
        }
    }
    for (int i4 = t + 6 * 256; i4 < d4; i4 += 256) {
        float4 a = sr4[0 * d4 + i4];
        float4 b = sr4[1 * d4 + i4];
        float4 c = sr4[2 * d4 + i4];
        float4 d = sr4[3 * d4 + i4];
        float vx = p0 * a.x + p1 * b.x + p2 * c.x + p3 * d.x;
        float vy = p0 * a.y + p1 * b.y + p2 * c.y + p3 * d.y;
        float vz = p0 * a.z + p1 * b.z + p2 * c.z + p3 * d.z;
        float vw = p0 * a.w + p1 * b.w + p2 * c.w + p3 * d.w;
        uint2 wq = __ldg(nw2 + i4);
        float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.x));
        float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.y));
        half2 h01 = __float22half2_rn(make_float2(vx * inv * w01.x, vy * inv * w01.y));
        half2 h23 = __float22half2_rn(make_float2(vz * inv * w23.x, vw * inv * w23.y));
        uint2 oq;
        oq.x = *reinterpret_cast<const uint32_t*>(&h01);
        oq.y = *reinterpret_cast<const uint32_t*>(&h23);
        y2[i4] = oq;
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
    int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    int d4 = D >> 2;
    if (i4 < d4) {
        const float4* sr4 = reinterpret_cast<const float4*>(streams + (size_t)mr * 4 * D);
        float4 a = sr4[0 * d4 + i4];
        float4 b = sr4[1 * d4 + i4];
        float4 c = sr4[2 * d4 + i4];
        float4 d = sr4[3 * d4 + i4];
        float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
        float4 outv;
        outv.x = p0 * a.x + p1 * b.x + p2 * c.x + p3 * d.x;
        outv.y = p0 * a.y + p1 * b.y + p2 * c.y + p3 * d.y;
        outv.z = p0 * a.z + p1 * b.z + p2 * c.z + p3 * d.z;
        outv.w = p0 * a.w + p1 * b.w + p2 * c.w + p3 * d.w;
        reinterpret_cast<float4*>(branch_in + (size_t)mr * D)[i4] = outv;
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

__global__ void mhc_sinkhorn_mix_pre_rmsnorm_f16_k(
    float* __restrict__ streams,
    const half* __restrict__ branch_out,
    const half* __restrict__ peer_branch_out,
    const float* __restrict__ h_coeff,
    const float* __restrict__ prev_pre,
    const __nv_bfloat16* __restrict__ norm_w,
    half* __restrict__ y_out,
    int D,
    int sinkhorn_iters,
    float rms_eps,
    float hc_eps
) {
    int mr = blockIdx.x;
    int t = threadIdx.x;
    __shared__ float s_post[4];
    __shared__ float s_res[16];
    __shared__ float s_pre[4];
    __shared__ float sm[8];
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
            for (int c = 0; c < 4; ++c) m[r][c] = m[r][c] * inv + hc_eps;
        }
        for (int c = 0; c < 4; ++c) {
            float cs = m[0][c] + m[1][c] + m[2][c] + m[3][c] + hc_eps;
            float inv = 1.0f / cs;
            for (int r = 0; r < 4; ++r) m[r][c] *= inv;
        }
        for (int it = 1; it < sinkhorn_iters; ++it) {
            for (int r = 0; r < 4; ++r) {
                float rs = m[r][0] + m[r][1] + m[r][2] + m[r][3] + hc_eps;
                float inv = 1.0f / rs;
                for (int c = 0; c < 4; ++c) m[r][c] *= inv;
            }
            for (int c = 0; c < 4; ++c) {
                float cs = m[0][c] + m[1][c] + m[2][c] + m[3][c] + hc_eps;
                float inv = 1.0f / cs;
                for (int r = 0; r < 4; ++r) m[r][c] *= inv;
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) s_res[r * 4 + c] = m[r][c];
        }
    }
    if (t < 4) {
        s_pre[t] = prev_pre ? prev_pre[(size_t)mr * 4 + t] : ((t == 0) ? 1.0f : 0.0f);
    }
    __syncthreads();
    float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    int d4 = D >> 2;
    float4* sr4 = reinterpret_cast<float4*>(streams + (size_t)mr * 4 * D);
    const uint2* bo2 = reinterpret_cast<const uint2*>(branch_out + (size_t)mr * D);
    const uint2* po2 = peer_branch_out ? reinterpret_cast<const uint2*>(peer_branch_out + (size_t)mr * D) : nullptr;
    float4 vals[6];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i4 = t + k * 256;
        if (i4 < d4) {
            float4 x0 = sr4[0 * d4 + i4];
            float4 x1 = sr4[1 * d4 + i4];
            float4 x2 = sr4[2 * d4 + i4];
            float4 x3 = sr4[3 * d4 + i4];
            uint2 bq = bo2[i4];
            float2 b0 = __half22float2(*reinterpret_cast<const half2*>(&bq.x));
            float2 b1 = __half22float2(*reinterpret_cast<const half2*>(&bq.y));
            if (po2) {
                uint2 pq = po2[i4];
                float2 q0 = __half22float2(*reinterpret_cast<const half2*>(&pq.x));
                float2 q1 = __half22float2(*reinterpret_cast<const half2*>(&pq.y));
                b0.x += q0.x; b0.y += q0.y;
                b1.x += q1.x; b1.y += q1.y;
            }
            float4 o0, o1, o2, o3;
            {
                float m0 = s_res[0], m1 = s_res[4], m2 = s_res[8], m3 = s_res[12], sp = s_post[0];
                o0.x = m0 * x0.x + m1 * x1.x + m2 * x2.x + m3 * x3.x + sp * b0.x;
                o0.y = m0 * x0.y + m1 * x1.y + m2 * x2.y + m3 * x3.y + sp * b0.y;
                o0.z = m0 * x0.z + m1 * x1.z + m2 * x2.z + m3 * x3.z + sp * b1.x;
                o0.w = m0 * x0.w + m1 * x1.w + m2 * x2.w + m3 * x3.w + sp * b1.y;
                sr4[0 * d4 + i4] = o0;
            }
            {
                float m0 = s_res[1], m1 = s_res[5], m2 = s_res[9], m3 = s_res[13], sp = s_post[1];
                o1.x = m0 * x0.x + m1 * x1.x + m2 * x2.x + m3 * x3.x + sp * b0.x;
                o1.y = m0 * x0.y + m1 * x1.y + m2 * x2.y + m3 * x3.y + sp * b0.y;
                o1.z = m0 * x0.z + m1 * x1.z + m2 * x2.z + m3 * x3.z + sp * b1.x;
                o1.w = m0 * x0.w + m1 * x1.w + m2 * x2.w + m3 * x3.w + sp * b1.y;
                sr4[1 * d4 + i4] = o1;
            }
            {
                float m0 = s_res[2], m1 = s_res[6], m2 = s_res[10], m3 = s_res[14], sp = s_post[2];
                o2.x = m0 * x0.x + m1 * x1.x + m2 * x2.x + m3 * x3.x + sp * b0.x;
                o2.y = m0 * x0.y + m1 * x1.y + m2 * x2.y + m3 * x3.y + sp * b0.y;
                o2.z = m0 * x0.z + m1 * x1.z + m2 * x2.z + m3 * x3.z + sp * b1.x;
                o2.w = m0 * x0.w + m1 * x1.w + m2 * x2.w + m3 * x3.w + sp * b1.y;
                sr4[2 * d4 + i4] = o2;
            }
            {
                float m0 = s_res[3], m1 = s_res[7], m2 = s_res[11], m3 = s_res[15], sp = s_post[3];
                o3.x = m0 * x0.x + m1 * x1.x + m2 * x2.x + m3 * x3.x + sp * b0.x;
                o3.y = m0 * x0.y + m1 * x1.y + m2 * x2.y + m3 * x3.y + sp * b0.y;
                o3.z = m0 * x0.z + m1 * x1.z + m2 * x2.z + m3 * x3.z + sp * b1.x;
                o3.w = m0 * x0.w + m1 * x1.w + m2 * x2.w + m3 * x3.w + sp * b1.y;
                sr4[3 * d4 + i4] = o3;
            }
            float4 v;
            v.x = p0 * o0.x + p1 * o1.x + p2 * o2.x + p3 * o3.x;
            v.y = p0 * o0.y + p1 * o1.y + p2 * o2.y + p3 * o3.y;
            v.z = p0 * o0.z + p1 * o1.z + p2 * o2.z + p3 * o3.z;
            v.w = p0 * o0.w + p1 * o1.w + p2 * o2.w + p3 * o3.w;
            vals[k] = v;
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
    }
    int lane = t & 31;
    int warp = t >> 5;
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, m);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? sm[lane] : 0.0f;
        for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
        if (lane == 0) sm[0] = rsqrtf(v / (float)D + rms_eps);
    }
    __syncthreads();
    float inv = sm[0];
    const uint2* nw2 = reinterpret_cast<const uint2*>(norm_w);
    uint2* y2 = reinterpret_cast<uint2*>(y_out + (size_t)mr * D);
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i4 = t + k * 256;
        if (i4 < d4) {
            uint2 wq = __ldg(nw2 + i4);
            float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.x));
            float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wq.y));
            float4 v = vals[k];
            half2 h01 = __float22half2_rn(make_float2(v.x * inv * w01.x, v.y * inv * w01.y));
            half2 h23 = __float22half2_rn(make_float2(v.z * inv * w23.x, v.w * inv * w23.y));
            uint2 oq;
            oq.x = *reinterpret_cast<const uint32_t*>(&h01);
            oq.y = *reinterpret_cast<const uint32_t*>(&h23);
            y2[i4] = oq;
        }
    }
}

void rad_cuda_mhc_sinkhorn_pre_coeff_rmsnorm(
    float* streams,
    const half* branch_out,
    const half* peer_branch_out,
    const float* hc_fn,
    const float* hc_base,
    const float* hc_scale,
    const float* prev_pre,
    const __nv_bfloat16* norm_w,
    float* h_coeff,
    float* next_pre,
    half* normed_out,
    int M,
    int D,
    int sinkhorn_iters,
    float rms_eps,
    float hc_eps,
    cudaStream_t stream
) {
    mhc_sinkhorn_mix_pre_rmsnorm_f16_k<<<M, 256, 0, stream>>>(
        streams, branch_out, peer_branch_out, h_coeff, prev_pre, norm_w, normed_out, D, sinkhorn_iters, rms_eps, hc_eps
    );
    if (M >= 64) {
        mhc_coeff_proj_24_4tok_k<<<(M + 3) / 4, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, M, D * 4, rms_eps, hc_eps);
    } else if (M >= 16) {
        mhc_coeff_proj_24_k<<<M, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    } else {
        dim3 g1(24, M);
        mhc_coeff_proj_k<<<g1, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    }
}

void rad_cuda_mhc_sinkhorn_mix_pre_rmsnorm_only(
    float* streams,
    const half* branch_out,
    const half* peer_branch_out,
    const float* h_coeff,
    const float* prev_pre,
    const __nv_bfloat16* norm_w,
    half* normed_out,
    int M,
    int D,
    int sinkhorn_iters,
    float rms_eps,
    float hc_eps,
    cudaStream_t stream
) {
    mhc_sinkhorn_mix_pre_rmsnorm_f16_k<<<M, 256, 0, stream>>>(
        streams, branch_out, peer_branch_out, h_coeff, prev_pre, norm_w, normed_out, D, sinkhorn_iters, rms_eps, hc_eps
    );
}

void rad_cuda_mhc_coeff_proj_only(
    const float* streams,
    const float* hc_fn,
    const float* hc_base,
    const float* hc_scale,
    float* h_coeff,
    float* next_pre,
    int M,
    int D,
    float rms_eps,
    float hc_eps,
    cudaStream_t stream
) {
    if (M >= 64) {
        mhc_coeff_proj_24_4tok_k<<<(M + 3) / 4, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, M, D * 4, rms_eps, hc_eps);
    } else if (M >= 16) {
        mhc_coeff_proj_24_k<<<M, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    } else {
        dim3 g1(24, M);
        mhc_coeff_proj_k<<<g1, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    }
}

void rad_cuda_mhc_pre_coeff_rmsnorm(
    const float* streams,
    const float* hc_fn,
    const float* hc_base,
    const float* hc_scale,
    const float* prev_pre,
    const __nv_bfloat16* norm_w,
    float* h_coeff,
    float* next_pre,
    half* normed_out,
    int M,
    int D,
    float rms_eps,
    float hc_eps,
    cudaStream_t stream
) {
    if (M >= 64) {
        mhc_coeff_proj_24_4tok_k<<<(M + 3) / 4, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, M, D * 4, rms_eps, hc_eps);
    } else if (M >= 16) {
        mhc_coeff_proj_24_k<<<M, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    } else {
        dim3 g1(24, M);
        mhc_coeff_proj_k<<<g1, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    }
    mhc_pre_mix_rmsnorm_f16_k<<<M, 256, 0, stream>>>(streams, prev_pre, norm_w, normed_out, D, rms_eps);
}

void rad_cuda_mhc_pre_mix_rmsnorm_only(
    const float* streams,
    const float* prev_pre,
    const __nv_bfloat16* norm_w,
    half* normed_out,
    int M,
    int D,
    float rms_eps,
    cudaStream_t stream
) {
    mhc_pre_mix_rmsnorm_f16_k<<<M, 256, 0, stream>>>(streams, prev_pre, norm_w, normed_out, D, rms_eps);
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
    if (M >= 64) {
        mhc_coeff_proj_24_4tok_k<<<(M + 3) / 4, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, M, D * 4, rms_eps, hc_eps);
    } else if (M >= 16) {
        mhc_coeff_proj_24_k<<<M, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    } else {
        dim3 g1(24, M);
        mhc_coeff_proj_k<<<g1, 256, 0, stream>>>(streams, hc_fn, hc_base, hc_scale, h_coeff, next_pre, D * 4, rms_eps, hc_eps);
    }
    dim3 g2(((D >> 2) + 255) / 256, M);
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
    dim3 g2(((D >> 2) + 255) / 256, M);
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

__device__ __forceinline__ void dspark_mma_m16n8k16_f32(
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
void mqa_flash_mma16_dspark_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_ring,
    const half* __restrict__ kv_blk,
    const float* __restrict__ attn_sink,
    half* __restrict__ out,
    int n_heads,
    int q0,
    int max_ctx,
    int sliding_window,
    int blk_len,
    float scale
) {
    int h0 = blockIdx.x * 16;
    int mr = blockIdx.y;
    int t = threadIdx.x;
    int warp = t >> 5;
    int lane = t & 31;
    int group_id = lane >> 2;
    int tid = lane & 3;

    int p_start = (sliding_window > 0 && q0 > sliding_window) ? (q0 - sliding_window) : 0;

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
            dspark_mma_m16n8k16_f32(s00, s01, s02, s03, a0, a1, a2, a3, b00, b01);
            dspark_mma_m16n8k16_f32(s10, s11, s12, s13, a0, a1, a2, a3, b10, b11);
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
            dspark_mma_m16n8k16_f32(o0[nb], o1[nb], o2[nb], o3[nb], pa0, pa1, pa2, pa3, b0, b1);
        }
        __syncthreads();
    };

    for (int p0 = p_start; p0 < q0; p0 += 16) {
        int tile_len = q0 - p0;
        if (tile_len > 16) tile_len = 16;
        for (int i = t; i < 1024; i += 256) {
            int r = i >> 6;
            int c4 = i & 63;
            if (r < tile_len) {
                int slot = (p0 + r) % max_ctx;
                const uint4* src_u4 = reinterpret_cast<const uint4*>(k_ring + (size_t)slot * 512);
                reinterpret_cast<uint4*>(s_kv[r])[c4] = __ldg(src_u4 + c4);
            } else {
                reinterpret_cast<uint4*>(s_kv[r])[c4] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();
        process_tile(tile_len);
    }

    if (blk_len > 0) {
        int n_u4 = blk_len * 64;
        const uint4* src_u4 = reinterpret_cast<const uint4*>(kv_blk);
        for (int i = t; i < 1024; i += 256) {
            int r = i >> 6;
            int c4 = i & 63;
            reinterpret_cast<uint4*>(s_kv[r])[c4] = (i < n_u4) ? __ldg(src_u4 + i) : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        process_tile(blk_len);
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
void mqa_flash_mma16_dspark_splitkv_f16_k(
    const half* __restrict__ q,
    const half* __restrict__ k_ring,
    const half* __restrict__ kv_blk,
    const float* __restrict__ attn_sink,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int n_heads,
    int q0,
    int max_ctx,
    int sliding_window,
    int blk_len,
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

    int p_start = (sliding_window > 0 && q0 > sliding_window) ? (q0 - sliding_window) : 0;
    int raw_len = q0 - p_start;
    int raw_tiles = (raw_len + 15) >> 4;
    int tot_tiles = raw_tiles + (blk_len > 0 ? 1 : 0);
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

    for (int ti = t_beg; ti < t_end; ++ti) {
        int tile_len;
        if (ti < raw_tiles) {
            int p0 = p_start + ti * 16;
            tile_len = q0 - p0;
            if (tile_len > 16) tile_len = 16;
            for (int i = t; i < 1024; i += 256) {
                int r = i >> 6;
                int c4 = i & 63;
                if (r < tile_len) {
                    int slot = (p0 + r) % max_ctx;
                    const uint4* src_u4 = reinterpret_cast<const uint4*>(k_ring + (size_t)slot * 512);
                    reinterpret_cast<uint4*>(s_kv[r])[c4] = __ldg(src_u4 + c4);
                } else {
                    reinterpret_cast<uint4*>(s_kv[r])[c4] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        } else {
            tile_len = blk_len;
            int n_u4 = blk_len * 64;
            const uint4* src_u4 = reinterpret_cast<const uint4*>(kv_blk);
            for (int i = t; i < 1024; i += 256) {
                int r = i >> 6;
                int c4 = i & 63;
                reinterpret_cast<uint4*>(s_kv[r])[c4] = (i < n_u4) ? __ldg(src_u4 + i) : make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

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
            dspark_mma_m16n8k16_f32(s00, s01, s02, s03, a0, a1, a2, a3, b00, b01);
            dspark_mma_m16n8k16_f32(s10, s11, s12, s13, a0, a1, a2, a3, b10, b11);
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
            dspark_mma_m16n8k16_f32(o0[nb], o1[nb], o2[nb], o3[nb], pa0, pa1, pa2, pa3, b0, b1);
        }
        __syncthreads();
    }

    int col_w = warp * 64;
    float* po0 = partial_o + (((size_t)mr * n_heads + h0 + group_id) * n_splits + sp) * 512 + col_w;
    float* po1 = partial_o + (((size_t)mr * n_heads + h0 + group_id + 8) * n_splits + sp) * 512 + col_w;
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

__global__ void dspark_splitkv_reduce_f16_k(
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
    (void)head_dim;
    float* split_scratch = const_cast<float*>(reinterpret_cast<const float*>(v_ring));
    int eff_len = (sliding_window > 0 && q0 > sliding_window) ? sliding_window : q0;
    if (split_scratch && split_scratch != reinterpret_cast<const float*>(k_ring) && eff_len >= 1024 && M <= 16) {
        int n_splits = 4;
        float* partial_ml = split_scratch + (size_t)M * n_heads * n_splits * 512;
        dim3 g1(n_heads / 16, M, n_splits);
        mqa_flash_mma16_dspark_splitkv_f16_k<<<g1, 256, 0, stream>>>(
            q, k_ring, kv_blk, attn_sink, split_scratch, partial_ml,
            n_heads, q0, max_ctx, sliding_window, M, scale, n_splits
        );
        dim3 g2(n_heads, M);
        dspark_splitkv_reduce_f16_k<<<g2, 256, 0, stream>>>(
            split_scratch, partial_ml, out, n_heads, n_splits
        );
        return;
    }
    dim3 grid(n_heads / 16, M);
    mqa_flash_mma16_dspark_f16_k<<<grid, 256, 0, stream>>>(
        q, k_ring, kv_blk, attn_sink, out, n_heads, q0, max_ctx, sliding_window, M, scale
    );
}

void rad_cuda_add_f16(half* dst, const half* src, int N, cudaStream_t stream) {
    add_f16_k<<<(N + 255) / 256, 256, 0, stream>>>(dst, src, N);
}

}
