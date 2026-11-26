#include "rad_sudot4_common.cuh"

extern "C" int ink_zero_f32(hipStream_t s, float* ptr, int n);


extern "C" __global__ void quantize_q8_1_k(const float* __restrict__ X,
                                           q8_1_blk* __restrict__ Y,
                                           int k) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nblk = k >> 5;
    int b = blockIdx.x * 4 + wid;
    if (b >= nblk) return;
    int t = blockIdx.y;
    float v = X[(size_t)t * k + (b << 5) + lane];
    float amax = s_warp_max32(fabsf(v));
    amax = __shfl(amax, 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    int32_t qsum = s_warp_sum32_i(qi);
    int q1 = __shfl_down(qi, 1, 32);
    int q2 = __shfl_down(qi, 2, 32);
    int q3 = __shfl_down(qi, 3, 32);
    q8_1_blk* dst = Y + (size_t)t * nblk + b;
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if ((lane & 3) == 0) {
        dst->q[lane >> 2] = pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
    }
}


extern "C" __global__ void quantize_q8_1_x3_k(const float* __restrict__ X0,
                                              const float* __restrict__ X1,
                                              const float* __restrict__ X2,
                                              q8_1_blk* __restrict__ Y0,
                                              q8_1_blk* __restrict__ Y1,
                                              q8_1_blk* __restrict__ Y2,
                                              int k) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nblk = k >> 5;
    int b = blockIdx.x * 4 + wid;
    if (b >= nblk) return;
    int t = blockIdx.y;
    int z = blockIdx.z;
    const float* X = (z == 0) ? X0 : ((z == 1) ? X1 : X2);
    q8_1_blk* Y = (z == 0) ? Y0 : ((z == 1) ? Y1 : Y2);
    float v = X[(size_t)t * k + (b << 5) + lane];
    float amax = s_warp_max32(fabsf(v));
    amax = __shfl(amax, 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    int32_t qsum = s_warp_sum32_i(qi);
    int q1 = __shfl_down(qi, 1, 32);
    int q2 = __shfl_down(qi, 2, 32);
    int q3 = __shfl_down(qi, 3, 32);
    q8_1_blk* dst = Y + (size_t)t * nblk + b;
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if ((lane & 3) == 0) {
        dst->q[lane >> 2] = pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
    }
}


extern "C" int ink_quantize_q8_1_x3(hipStream_t s, const float* X0, const float* X1, const float* X2, void* Y0, void* Y1, void* Y2, int k, int B) {
    int nblk = k >> 5;
    quantize_q8_1_x3_k<<<dim3((nblk + 3) / 4, B, 3), dim3(128), 0, s>>>(X0, X1, X2, (q8_1_blk*)Y0, (q8_1_blk*)Y1, (q8_1_blk*)Y2, k);
    return 0;
}


extern "C" __global__ void gemm_q8_0_sudot4_k(const uint8_t* __restrict__ W,
                                              const q8_1_blk* __restrict__ Xq,
                                              float* __restrict__ Y,
                                              int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n) return;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    int nblk = k >> 5;
    const uint8_t* wrow = W + (size_t)row * ((size_t)nblk * 34);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = lane; b < nblk; b += 32) {
        const uint8_t* bp = wrow + (size_t)b * 34;
        uint16_t dh_bits;
        memcpy(&dh_bits, bp, 2);
        float dw = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
        int32_t wq[8];
        memcpy(wq, bp + 2, 32);
        for (int m = 0; m < n_t; ++m) {
            const q8_1_blk xb = Xq[(size_t)(t0 + m) * nblk + b];
            int32_t sumi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi = __builtin_amdgcn_sudot4(true, wq[j], true, xb.q[j], sumi, false);
            }
            acc[m] = fmaf(dw * xb.d, (float)sumi, acc[m]);
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = s_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


template <int NT>
__global__ void gemm_q4_0_sudot4_nt8_k(const uint8_t* __restrict__ W,
                                       const q8_1_blk* __restrict__ Xq,
                                       float* __restrict__ Y,
                                       int n, int k) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;
    if (row >= n) return;
    int t0 = blockIdx.y * NT;
    int nblk = k >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        uint4 q4 = qrow[b];
        float dw = __half2float(drow[b]);
        int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu);
        int32_t w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
        int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu);
        int32_t w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
        int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu);
        int32_t w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
        int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu);
        int32_t w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            const q8_1_blk xb = Xq[(size_t)(t0 + m) * nblk + b];
            int32_t sumi = 0;
            sumi = __builtin_amdgcn_sudot4(false, w0, true, xb.q[0], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w1, true, xb.q[1], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w2, true, xb.q[2], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w3, true, xb.q[3], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w4, true, xb.q[4], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w5, true, xb.q[5], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w6, true, xb.q[6], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w7, true, xb.q[7], sumi, false);
            acc[m] = fmaf(dw * xb.d, (float)(sumi - 8 * xb.s), acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        float v = s_warp_sum8(acc[m]);
        if (lane8 == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


template <int NT>
__global__ void gemm_q4_0_sudot4_nt_k(const uint8_t* __restrict__ W,
                                      const q8_1_blk* __restrict__ Xq,
                                      float* __restrict__ Y,
                                      int n, int k) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n) return;
    int t0 = blockIdx.y * NT;
    int nblk = k >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        uint4 q4 = qrow[b];
        float dw = __half2float(drow[b]);
        int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu);
        int32_t w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
        int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu);
        int32_t w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
        int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu);
        int32_t w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
        int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu);
        int32_t w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            const q8_1_blk xb = Xq[(size_t)(t0 + m) * nblk + b];
            int32_t sumi = 0;
            sumi = __builtin_amdgcn_sudot4(false, w0, true, xb.q[0], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w1, true, xb.q[1], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w2, true, xb.q[2], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w3, true, xb.q[3], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w4, true, xb.q[4], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w5, true, xb.q[5], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w6, true, xb.q[6], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w7, true, xb.q[7], sumi, false);
            acc[m] = fmaf(dw * xb.d, (float)(sumi - 8 * xb.s), acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        float v = s_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


extern "C" __global__ void gemm_q4_0_sudot4_k(const uint8_t* __restrict__ W,
                                              const q8_1_blk* __restrict__ Xq,
                                              float* __restrict__ Y,
                                              int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n) return;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    int nblk = k >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = lane; b < nblk; b += 32) {
        uint4 q4 = qrow[b];
        float dw = __half2float(drow[b]);
        for (int m = 0; m < n_t; ++m) {
            const q8_1_blk xb = Xq[(size_t)(t0 + m) * nblk + b];
            int32_t sumi = 0;
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.x & 0x0F0F0F0Fu), true, xb.q[0], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.y & 0x0F0F0F0Fu), true, xb.q[1], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.z & 0x0F0F0F0Fu), true, xb.q[2], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.w & 0x0F0F0F0Fu), true, xb.q[3], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), true, xb.q[4], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu), true, xb.q[5], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), true, xb.q[6], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu), true, xb.q[7], sumi, false);
            acc[m] = fmaf(dw * xb.d, (float)(sumi - 8 * xb.s), acc[m]);
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = s_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


extern "C" __global__ void moe_build_groups_k(const int* __restrict__ itop,
                                              const float* __restrict__ wtop,
                                              const float* __restrict__ scales,
                                              int* __restrict__ exp_counts,
                                              int* __restrict__ exp_offsets,
                                              int* __restrict__ tok_sorted,
                                              int* __restrict__ slot_sorted,
                                              float* __restrict__ wt_sorted,
                                              int n_used, int B) {
    int tid = threadIdx.x;
    __shared__ int cnt[128];
    __shared__ int off[129];
    __shared__ int act[128];
    cnt[tid] = 0;
    act[tid] = -1;
    __syncthreads();
    int total = B * n_used;
    for (int i = tid; i < total; i += 128) {
        atomicAdd(&cnt[itop[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int s = 0;
        int na = 0;
        for (int e = 0; e < 128; ++e) {
            off[e] = s;
            if (cnt[e] > 0) act[na++] = e;
            s += cnt[e];
            cnt[e] = 0;
        }
        off[128] = s;
    }
    __syncthreads();
    exp_offsets[tid] = off[tid];
    if (tid == 0) exp_offsets[128] = off[128];
    for (int i = tid; i < total; i += 128) {
        int e = itop[i];
        int pos = off[e] + atomicAdd(&cnt[e], 1);
        tok_sorted[pos] = i / n_used;
        slot_sorted[pos] = i % n_used;
        wt_sorted[pos] = wtop[i] * (scales ? scales[e] : 1.0f);
    }
    __syncthreads();
    exp_counts[tid] = cnt[tid];
    exp_counts[128 + tid] = act[tid];
}


extern "C" __global__ void moe_router_and_groups_k(const float* __restrict__ logits,
                                                   const float* __restrict__ bias,
                                                   const float* __restrict__ scales,
                                                   int* __restrict__ exp_counts,
                                                   int* __restrict__ exp_offsets,
                                                   int* __restrict__ tok_sorted,
                                                   int* __restrict__ slot_sorted,
                                                   float* __restrict__ wt_sorted,
                                                   float route_scale, int B) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    __shared__ int s_itop[32 * 8];
    __shared__ float s_wtop[32 * 8];
    __shared__ int cnt[128];
    __shared__ int off[129];
    __shared__ int act[128];

    if (tid < 128) {
        cnt[tid] = 0;
        act[tid] = -1;
    }

    for (int bt = wid; bt < B; bt += 8) {
        const float* lg = logits + (size_t)bt * 128;
        float pv[4];
        float sv[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int e = lane + p * 32;
            float pr = 1.0f / (1.0f + expf(-lg[e]));
            pv[p] = pr;
            sv[p] = bias ? (pr + bias[e]) : pr;
        }
        float chosen_p[8];
        int chosen_i[8];
        for (int t = 0; t < 8; ++t) {
            float bv = sv[0];
            float bp = pv[0];
            int bi = lane;
#pragma unroll
            for (int p = 1; p < 4; ++p) {
                int e = lane + p * 32;
                if (sv[p] > bv || (sv[p] == bv && e < bi)) {
                    bv = sv[p];
                    bp = pv[p];
                    bi = e;
                }
            }
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                float ov = __shfl_down(bv, stride, 32);
                float op = __shfl_down(bp, stride, 32);
                int oi = __shfl_down(bi, stride, 32);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bp = op;
                    bi = oi;
                }
            }
            int win_i = __shfl(bi, 0, 32);
            float win_p = __shfl(bp, 0, 32);
            if ((win_i & 31) == lane) {
                int wp = win_i >> 5;
                if (wp == 0) sv[0] = -1e30f;
                else if (wp == 1) sv[1] = -1e30f;
                else if (wp == 2) sv[2] = -1e30f;
                else sv[3] = -1e30f;
            }
            chosen_i[t] = win_i;
            chosen_p[t] = win_p;
        }
        if (lane == 0) {
            float wsum = 0.0f;
#pragma unroll
            for (int t = 0; t < 8; ++t) wsum += chosen_p[t];
            float inv = route_scale / (wsum + 1e-20f);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                s_itop[bt * 8 + t] = chosen_i[t];
                s_wtop[bt * 8 + t] = chosen_p[t] * inv;
            }
        }
    }
    __syncthreads();

    int total = B * 8;
    for (int i = tid; i < total; i += 256) {
        atomicAdd(&cnt[s_itop[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int s = 0;
        int na = 0;
        for (int e = 0; e < 128; ++e) {
            off[e] = s;
            if (cnt[e] > 0) act[na++] = e;
            s += cnt[e];
            cnt[e] = 0;
        }
        off[128] = s;
    }
    __syncthreads();
    if (tid < 128) exp_offsets[tid] = off[tid];
    if (tid == 0) exp_offsets[128] = off[128];
    for (int i = tid; i < total; i += 256) {
        int e = s_itop[i];
        int pos = off[e] + atomicAdd(&cnt[e], 1);
        tok_sorted[pos] = i >> 3;
        slot_sorted[pos] = i & 7;
        wt_sorted[pos] = s_wtop[i] * (scales ? scales[e] : 1.0f);
    }
    __syncthreads();
    if (tid < 128) {
        exp_counts[tid] = cnt[tid];
        exp_counts[128 + tid] = act[tid];
    }
}


extern "C" int ink_moe_router_and_groups(hipStream_t s, const float* logits, const float* bias, const float* scales, int* exp_counts, int* exp_offsets, int* tok_sorted, int* slot_sorted, float* wt_sorted, float route_scale, int B) {
    moe_router_and_groups_k<<<dim3(1), dim3(256), 0, s>>>(logits, bias, scales, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, route_scale, B);
    return 0;
}


extern "C" __global__ void b1_router_and_q8_k(const float* __restrict__ logits,
                                              const float* __restrict__ bias,
                                              float* __restrict__ wtop,
                                              int* __restrict__ itop,
                                              const float* __restrict__ x_in,
                                              q8_1_blk* __restrict__ xq_out,
                                              float route_scale, int k) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    if (wid == 0) {
        float pv[4];
        float sv[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int e = lane + p * 32;
            float pr = 1.0f / (1.0f + expf(-logits[e]));
            pv[p] = pr;
            sv[p] = bias ? (pr + bias[e]) : pr;
        }
        float chosen_p[8];
        int chosen_i[8];
        for (int t = 0; t < 8; ++t) {
            float bv = sv[0];
            float bp = pv[0];
            int bi = lane;
#pragma unroll
            for (int p = 1; p < 4; ++p) {
                int e = lane + p * 32;
                if (sv[p] > bv || (sv[p] == bv && e < bi)) {
                    bv = sv[p];
                    bp = pv[p];
                    bi = e;
                }
            }
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                float ov = __shfl_down(bv, stride, 32);
                float op = __shfl_down(bp, stride, 32);
                int oi = __shfl_down(bi, stride, 32);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bp = op;
                    bi = oi;
                }
            }
            int win_i = __shfl(bi, 0, 32);
            float win_p = __shfl(bp, 0, 32);
            if ((win_i & 31) == lane) {
                int wp = win_i >> 5;
                if (wp == 0) sv[0] = -1e30f;
                else if (wp == 1) sv[1] = -1e30f;
                else if (wp == 2) sv[2] = -1e30f;
                else sv[3] = -1e30f;
            }
            chosen_i[t] = win_i;
            chosen_p[t] = win_p;
        }
        if (lane == 0) {
            float wsum = 0.0f;
#pragma unroll
            for (int t = 0; t < 8; ++t) wsum += chosen_p[t];
            float inv = route_scale / (wsum + 1e-20f);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                itop[t] = chosen_i[t];
                wtop[t] = chosen_p[t] * inv;
            }
        }
    } else {
        int nblk = k >> 5;
        for (int b = wid - 1; b < nblk; b += 7) {
            float v = x_in[(b << 5) + lane];
            float amax = __shfl(s_warp_max32(fabsf(v)), 0, 32);
            float d = amax / 127.0f;
            float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            int qi = __float2int_rn(v * inv);
            if (qi > 127) qi = 127;
            if (qi < -127) qi = -127;
            int32_t qsum = s_warp_sum32_i(qi);
            int q1 = __shfl_down(qi, 1, 32);
            int q2 = __shfl_down(qi, 2, 32);
            int q3 = __shfl_down(qi, 3, 32);
            q8_1_blk* dst = xq_out + b;
            if (lane == 0) {
                dst->d = d;
                dst->s = qsum;
            }
            if ((lane & 3) == 0) {
                dst->q[lane >> 2] = pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
            }
        }
    }
}


extern "C" int ink_b1_router_and_q8(hipStream_t s, const float* logits, const float* bias, float* wtop, int* itop, const float* x_in, void* xq_out, float route_scale, int k) {
    b1_router_and_q8_k<<<dim3(1), dim3(256), 0, s>>>(logits, bias, wtop, itop, x_in, (q8_1_blk*)xq_out, route_scale, k);
    return 0;
}


extern "C" __global__ void moe_router_groups_and_q8_k(const float* __restrict__ logits,
                                                      const float* __restrict__ bias,
                                                      const float* __restrict__ scales,
                                                      int* __restrict__ exp_counts,
                                                      int* __restrict__ exp_offsets,
                                                      int* __restrict__ tok_sorted,
                                                      int* __restrict__ slot_sorted,
                                                      float* __restrict__ wt_sorted,
                                                      const float* __restrict__ x_in,
                                                      q8_1_blk* __restrict__ xq_out,
                                                      float route_scale, int k, int B) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    if (blockIdx.x > 0) {
        int t = blockIdx.x - 1;
        int nblk = k >> 5;
        const float* xt = x_in + (size_t)t * k;
        q8_1_blk* yt = xq_out + (size_t)t * nblk;
        for (int b = wid; b < nblk; b += 8) {
            float v = xt[(b << 5) + lane];
            float amax = __shfl(s_warp_max32(fabsf(v)), 0, 32);
            float d = amax / 127.0f;
            float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            int qi = __float2int_rn(v * inv);
            if (qi > 127) qi = 127;
            if (qi < -127) qi = -127;
            int32_t qsum = s_warp_sum32_i(qi);
            int q1 = __shfl_down(qi, 1, 32);
            int q2 = __shfl_down(qi, 2, 32);
            int q3 = __shfl_down(qi, 3, 32);
            q8_1_blk* dst = yt + b;
            if (lane == 0) {
                dst->d = d;
                dst->s = qsum;
            }
            if ((lane & 3) == 0) {
                dst->q[lane >> 2] = pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
            }
        }
        return;
    }

    __shared__ int s_itop[64 * 8];
    __shared__ float s_wtop[64 * 8];
    __shared__ int cnt[128];
    __shared__ int off[129];
    __shared__ int act[128];

    if (tid < 128) {
        cnt[tid] = 0;
        act[tid] = -1;
    }

    for (int bt = wid; bt < B; bt += 8) {
        const float* lg = logits + (size_t)bt * 128;
        float pv[4];
        float sv[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int e = lane + p * 32;
            float pr = 1.0f / (1.0f + expf(-lg[e]));
            pv[p] = pr;
            sv[p] = bias ? (pr + bias[e]) : pr;
        }
        float chosen_p[8];
        int chosen_i[8];
        for (int t = 0; t < 8; ++t) {
            float bv = sv[0];
            float bp = pv[0];
            int bi = lane;
#pragma unroll
            for (int p = 1; p < 4; ++p) {
                int e = lane + p * 32;
                if (sv[p] > bv || (sv[p] == bv && e < bi)) {
                    bv = sv[p];
                    bp = pv[p];
                    bi = e;
                }
            }
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                float ov = __shfl_down(bv, stride, 32);
                float op = __shfl_down(bp, stride, 32);
                int oi = __shfl_down(bi, stride, 32);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bp = op;
                    bi = oi;
                }
            }
            int win_i = __shfl(bi, 0, 32);
            float win_p = __shfl(bp, 0, 32);
            if ((win_i & 31) == lane) {
                int wp = win_i >> 5;
                if (wp == 0) sv[0] = -1e30f;
                else if (wp == 1) sv[1] = -1e30f;
                else if (wp == 2) sv[2] = -1e30f;
                else sv[3] = -1e30f;
            }
            chosen_i[t] = win_i;
            chosen_p[t] = win_p;
        }
        if (lane == 0) {
            float wsum = 0.0f;
#pragma unroll
            for (int t = 0; t < 8; ++t) wsum += chosen_p[t];
            float inv = route_scale / (wsum + 1e-20f);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                s_itop[bt * 8 + t] = chosen_i[t];
                s_wtop[bt * 8 + t] = chosen_p[t] * inv;
            }
        }
    }
    __syncthreads();

    int total = B * 8;
    for (int i = tid; i < total; i += 256) {
        atomicAdd(&cnt[s_itop[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int s = 0;
        int na = 0;
        for (int e = 0; e < 128; ++e) {
            off[e] = s;
            if (cnt[e] > 0) act[na++] = e;
            s += cnt[e];
            cnt[e] = 0;
        }
        off[128] = s;
    }
    __syncthreads();
    if (tid < 128) exp_offsets[tid] = off[tid];
    if (tid == 0) exp_offsets[128] = off[128];
    for (int i = tid; i < total; i += 256) {
        int e = s_itop[i];
        int pos = off[e] + atomicAdd(&cnt[e], 1);
        tok_sorted[pos] = i >> 3;
        slot_sorted[pos] = i & 7;
        wt_sorted[pos] = s_wtop[i] * (scales ? scales[e] : 1.0f);
    }
    __syncthreads();
    if (tid < 128) {
        exp_counts[tid] = cnt[tid];
        exp_counts[128 + tid] = act[tid];
    }
}


extern "C" int ink_moe_router_groups_and_q8(hipStream_t s, const float* logits, const float* bias, const float* scales, int* exp_counts, int* exp_offsets, int* tok_sorted, int* slot_sorted, float* wt_sorted, const float* x_in, void* xq_out, float route_scale, int k, int B) {
    moe_router_groups_and_q8_k<<<dim3(1 + B), dim3(256), 0, s>>>(logits, bias, scales, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, x_in, (q8_1_blk*)xq_out, route_scale, k, B);
    return 0;
}


template <int NT>
__global__ void dflash2_qkv_q4_nt_k(const uint8_t* __restrict__ wq,
                                    const uint8_t* __restrict__ wk,
                                    const uint8_t* __restrict__ wv,
                                    const q8_1_blk* __restrict__ Xq,
                                    float* __restrict__ Q,
                                    float* __restrict__ K,
                                    float* __restrict__ V,
                                    int k, int q_dim, int kv_dim) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int gid = blockIdx.x * 32 + grp;
    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    if (gid >= o3) return;
    const uint8_t* W;
    float* Y;
    int row, n_out;
    if (gid < o1) { W = wq; Y = Q; row = gid; n_out = q_dim; }
    else if (gid < o2) { W = wk; Y = K; row = gid - o1; n_out = kv_dim; }
    else { W = wv; Y = V; row = gid - o2; n_out = kv_dim; }
    int nblk = k >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        uint4 q4 = qrow[b];
        float dw = __half2float(drow[b]);
        int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu);
        int32_t w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
        int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu);
        int32_t w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
        int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu);
        int32_t w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
        int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu);
        int32_t w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            const q8_1_blk xb = Xq[(size_t)m * nblk + b];
            int32_t sumi = 0;
            sumi = __builtin_amdgcn_sudot4(false, w0, true, xb.q[0], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w1, true, xb.q[1], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w2, true, xb.q[2], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w3, true, xb.q[3], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w4, true, xb.q[4], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w5, true, xb.q[5], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w6, true, xb.q[6], sumi, false);
            sumi = __builtin_amdgcn_sudot4(false, w7, true, xb.q[7], sumi, false);
            acc[m] = fmaf(dw * xb.d, (float)(sumi - 8 * xb.s), acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        float v = s_warp_sum8(acc[m]);
        if (lane8 == 0) Y[(size_t)m * n_out + row] = v;
    }
}


extern "C" int ink_dflash2_qkv_q4(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const void* Xq, float* Q, float* K, float* V, int k, int q_dim, int kv_dim, int B) {
    int total_rows = q_dim + 2 * kv_dim;
    if (B == 8) {
        dflash2_qkv_q4_nt_k<8><<<dim3((total_rows + 31) / 32), dim3(256), 0, s>>>(wq, wk, wv, (const q8_1_blk*)Xq, Q, K, V, k, q_dim, kv_dim);
        return 0;
    }
    if (B == 6) {
        dflash2_qkv_q4_nt_k<6><<<dim3((total_rows + 31) / 32), dim3(256), 0, s>>>(wq, wk, wv, (const q8_1_blk*)Xq, Q, K, V, k, q_dim, kv_dim);
        return 0;
    }
    if (B == 4) {
        dflash2_qkv_q4_nt_k<4><<<dim3((total_rows + 31) / 32), dim3(256), 0, s>>>(wq, wk, wv, (const q8_1_blk*)Xq, Q, K, V, k, q_dim, kv_dim);
        return 0;
    }
    return 1;
}


extern "C" __global__ void moe_gu_nvfp4_sudot4_k(const uint8_t* __restrict__ W,
                                                 const float* __restrict__ scales,
                                                 const q8_1_blk* __restrict__ Xq,
                                                 const int* __restrict__ exp_counts,
                                                 const int* __restrict__ exp_offsets,
                                                 const int* __restrict__ tok_sorted,
                                                 const int* __restrict__ slot_sorted,
                                                 float* __restrict__ act_out,
                                                 int n_ff, int k, int n_used,
                                                 float beta, float lb) {
    int exp_id = exp_counts[128 + blockIdx.y];
    if (exp_id < 0) return;
    int cnt = exp_counts[exp_id];
    if (cnt == 0) return;
    int off0 = exp_offsets[exp_id];
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_ff) return;

    int nblk = k >> 6;
    int q8_nblk = k >> 5;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + row) * row_bytes;
    float sc = scales ? scales[exp_id] : 1.0f;

    for (int t0 = 0; t0 < cnt; t0 += 8) {
        int nt = min(8, cnt - t0);
        int tids[8];
        for (int m = 0; m < nt; ++m) tids[m] = tok_sorted[off0 + t0 + m];
        float ag[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int b = lane8; b < nblk; b += 8) {
            const uint8_t* bg = wg + (size_t)b * 36;
            const uint8_t* bu = wu + (size_t)b * 36;
            uint32_t dsg, dsu;
            memcpy(&dsg, bg, 4);
            memcpy(&dsu, bu, 4);
            const uint32_t* qg32 = reinterpret_cast<const uint32_t*>(bg + 4);
            const uint32_t* qu32 = reinterpret_cast<const uint32_t*>(bu + 4);
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                float dg = s_ue4m3_to_fp32((uint8_t)((dsg >> (sb * 8)) & 0xFF));
                float du = s_ue4m3_to_fp32((uint8_t)((dsu >> (sb * 8)) & 0xFF));
                uint32_t g0 = qg32[sb * 2], g1 = qg32[sb * 2 + 1];
                uint32_t u0 = qu32[sb * 2], u1 = qu32[sb * 2 + 1];
                int32_t glo0 = pack4_i8(s_fp4_i8(g0), s_fp4_i8(g0 >> 8), s_fp4_i8(g0 >> 16), s_fp4_i8(g0 >> 24));
                int32_t ghi0 = pack4_i8(s_fp4_i8(g0 >> 4), s_fp4_i8(g0 >> 12), s_fp4_i8(g0 >> 20), s_fp4_i8(g0 >> 28));
                int32_t glo1 = pack4_i8(s_fp4_i8(g1), s_fp4_i8(g1 >> 8), s_fp4_i8(g1 >> 16), s_fp4_i8(g1 >> 24));
                int32_t ghi1 = pack4_i8(s_fp4_i8(g1 >> 4), s_fp4_i8(g1 >> 12), s_fp4_i8(g1 >> 20), s_fp4_i8(g1 >> 28));
                int32_t ulo0 = pack4_i8(s_fp4_i8(u0), s_fp4_i8(u0 >> 8), s_fp4_i8(u0 >> 16), s_fp4_i8(u0 >> 24));
                int32_t uhi0 = pack4_i8(s_fp4_i8(u0 >> 4), s_fp4_i8(u0 >> 12), s_fp4_i8(u0 >> 20), s_fp4_i8(u0 >> 28));
                int32_t ulo1 = pack4_i8(s_fp4_i8(u1), s_fp4_i8(u1 >> 8), s_fp4_i8(u1 >> 16), s_fp4_i8(u1 >> 24));
                int32_t uhi1 = pack4_i8(s_fp4_i8(u1 >> 4), s_fp4_i8(u1 >> 12), s_fp4_i8(u1 >> 20), s_fp4_i8(u1 >> 28));
                int xb_idx = (b << 1) + (sb >> 1);
                int qoff = (sb & 1) * 4;
                for (int m = 0; m < nt; ++m) {
                    const q8_1_blk& xb = Xq[(size_t)tids[m] * q8_nblk + xb_idx];
                    int32_t x0 = xb.q[qoff + 0], x1 = xb.q[qoff + 1], x2 = xb.q[qoff + 2], x3 = xb.q[qoff + 3];
                    int32_t sg_i = 0;
                    sg_i = __builtin_amdgcn_sudot4(true, glo0, true, x0, sg_i, false);
                    sg_i = __builtin_amdgcn_sudot4(true, glo1, true, x1, sg_i, false);
                    sg_i = __builtin_amdgcn_sudot4(true, ghi0, true, x2, sg_i, false);
                    sg_i = __builtin_amdgcn_sudot4(true, ghi1, true, x3, sg_i, false);
                    ag[m] = fmaf(dg * xb.d, (float)sg_i, ag[m]);
                    int32_t su_i = 0;
                    su_i = __builtin_amdgcn_sudot4(true, ulo0, true, x0, su_i, false);
                    su_i = __builtin_amdgcn_sudot4(true, ulo1, true, x1, su_i, false);
                    su_i = __builtin_amdgcn_sudot4(true, uhi0, true, x2, su_i, false);
                    su_i = __builtin_amdgcn_sudot4(true, uhi1, true, x3, su_i, false);
                    au[m] = fmaf(du * xb.d, (float)su_i, au[m]);
                }
            }
        }
        for (int m = 0; m < nt; ++m) {
            float g = s_warp_sum8(ag[m]) * sc;
            float u = s_warp_sum8(au[m]) * sc;
            if (lane8 == 0) {
                float sg = 1.0f / (1.0f + expf(-g));
                float act = beta * tanhf(g / beta) * sg;
                float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                int sl = slot_sorted[off0 + t0 + m];
                act_out[((size_t)tids[m] * n_used + sl) * n_ff + row] = act * up;
            }
        }
    }
}


extern "C" __global__ void moe_dn_nvfp4_sudot4_k(const uint8_t* __restrict__ W,
                                                 const q8_1_blk* __restrict__ act_q,
                                                 const int* __restrict__ exp_counts,
                                                 const int* __restrict__ exp_offsets,
                                                 const int* __restrict__ tok_sorted,
                                                 const int* __restrict__ slot_sorted,
                                                 const float* __restrict__ wt_sorted,
                                                 float* __restrict__ y_out,
                                                 int n_embd, int n_ff, int n_used) {
    int exp_id = exp_counts[128 + blockIdx.y];
    if (exp_id < 0) return;
    int cnt = exp_counts[exp_id];
    if (cnt == 0) return;
    int off0 = exp_offsets[exp_id];
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;

    int nblk = n_ff >> 6;
    int q8_nblk = n_ff >> 5;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;

    for (int t0 = 0; t0 < cnt; t0 += 8) {
        int nt = min(8, cnt - t0);
        int tids[8];
        int sids[8];
        for (int m = 0; m < nt; ++m) {
            tids[m] = tok_sorted[off0 + t0 + m];
            sids[m] = slot_sorted[off0 + t0 + m];
        }
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int b = lane8; b < nblk; b += 8) {
            const uint8_t* bp = wrow + (size_t)b * 36;
            uint32_t ds;
            memcpy(&ds, bp, 4);
            const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bp + 4);
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                float dw = s_ue4m3_to_fp32((uint8_t)((ds >> (sb * 8)) & 0xFF));
                uint32_t q0 = q32[sb * 2], q1 = q32[sb * 2 + 1];
                int32_t wlo0 = pack4_i8(s_fp4_i8(q0), s_fp4_i8(q0 >> 8), s_fp4_i8(q0 >> 16), s_fp4_i8(q0 >> 24));
                int32_t whi0 = pack4_i8(s_fp4_i8(q0 >> 4), s_fp4_i8(q0 >> 12), s_fp4_i8(q0 >> 20), s_fp4_i8(q0 >> 28));
                int32_t wlo1 = pack4_i8(s_fp4_i8(q1), s_fp4_i8(q1 >> 8), s_fp4_i8(q1 >> 16), s_fp4_i8(q1 >> 24));
                int32_t whi1 = pack4_i8(s_fp4_i8(q1 >> 4), s_fp4_i8(q1 >> 12), s_fp4_i8(q1 >> 20), s_fp4_i8(q1 >> 28));
                int ab_idx = (b << 1) + (sb >> 1);
                int qoff = (sb & 1) * 4;
                for (int m = 0; m < nt; ++m) {
                    const q8_1_blk& ab = act_q[((size_t)tids[m] * n_used + sids[m]) * q8_nblk + ab_idx];
                    int32_t sumi = 0;
                    sumi = __builtin_amdgcn_sudot4(true, wlo0, true, ab.q[qoff + 0], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(true, wlo1, true, ab.q[qoff + 1], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(true, whi0, true, ab.q[qoff + 2], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(true, whi1, true, ab.q[qoff + 3], sumi, false);
                    acc[m] = fmaf(dw * ab.d, (float)sumi, acc[m]);
                }
            }
        }
        for (int m = 0; m < nt; ++m) {
            float v = s_warp_sum8(acc[m]) * wt_sorted[off0 + t0 + m];
            if (lane8 == 0) atomicAdd(&y_out[(size_t)tids[m] * n_embd + row], v);
        }
    }
}


extern "C" int ink_gemv_q4_b1(hipStream_t s, const uint8_t* W, const void* Xq, float* Y, int n, int k);

extern "C" int ink_qkvrg_q4_b1(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim);

extern "C" __global__ void wo_conv_res_q4_b1_8_k(const uint8_t* __restrict__ W, const q8_1_blk* __restrict__ Xq, const float* __restrict__ conv_w, float* __restrict__ hist, const float* __restrict__ add_src, float* __restrict__ block_dst, float* __restrict__ out, int n_embd, int q_dim);

extern "C" __global__ void moe_dn_conv_add_q4_b1_4_k(const uint8_t* __restrict__ W, const float* __restrict__ scales, const q8_1_blk* __restrict__ act_q, const int* __restrict__ itop, const float* __restrict__ wtop, const float* __restrict__ conv_w, float* __restrict__ hist, const float* __restrict__ ffn_inp, float* __restrict__ cur, int n_embd, int n_ff, int n_used);


extern "C" int ink_quantize_q8_1(hipStream_t s, const float* X, void* Y, int k, int B) {
    int nblk = k >> 5;
    quantize_q8_1_k<<<dim3((nblk + 3) / 4, B), dim3(128), 0, s>>>(X, (q8_1_blk*)Y, k);
    return 0;
}


template <int NT>
__global__ void gemm_q8_0_sudot4_nt8_k(const uint8_t* __restrict__ W,
                                       const q8_1_blk* __restrict__ Xq,
                                       float* __restrict__ Y,
                                       int n, int k) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;
    int nblk = k >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    const uint8_t* wrow = W + (size_t)min(row, n - 1) * row_bytes;
    __shared__ q8_1_blk sm_x[NT][8];
    float acc[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
    for (int b0 = 0; b0 < nblk; b0 += 8) {
        if (threadIdx.x < NT * 8) {
            int tm = threadIdx.x >> 3;
            int tb = threadIdx.x & 7;
            if (b0 + tb < nblk) sm_x[tm][tb] = Xq[(size_t)tm * nblk + b0 + tb];
        }
        __syncthreads();
        int b = b0 + lane8;
        if (row < n && b < nblk) {
            const uint8_t* bp = wrow + (size_t)b * 34;
            uint16_t dh_bits;
            memcpy(&dh_bits, bp, 2);
            float dw = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
            int32_t wq[8];
            memcpy(wq, bp + 2, 32);
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                const q8_1_blk xb = sm_x[m][lane8];
                int32_t sumi = 0;
                sumi = __builtin_amdgcn_sudot4(true, wq[0], true, xb.q[0], sumi, false);
                sumi = __builtin_amdgcn_sudot4(true, wq[1], true, xb.q[1], sumi, false);
                sumi = __builtin_amdgcn_sudot4(true, wq[2], true, xb.q[2], sumi, false);
                sumi = __builtin_amdgcn_sudot4(true, wq[3], true, xb.q[3], sumi, false);
                sumi = __builtin_amdgcn_sudot4(true, wq[4], true, xb.q[4], sumi, false);
                sumi = __builtin_amdgcn_sudot4(true, wq[5], true, xb.q[5], sumi, false);
                sumi = __builtin_amdgcn_sudot4(true, wq[6], true, xb.q[6], sumi, false);
                sumi = __builtin_amdgcn_sudot4(true, wq[7], true, xb.q[7], sumi, false);
                acc[m] = fmaf(dw * xb.d, (float)sumi, acc[m]);
            }
        }
        __syncthreads();
    }
    if (row < n) {
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            float v = s_warp_sum8(acc[m]);
            if (lane8 == 0) Y[(size_t)m * n + row] = v;
        }
    }
}


extern "C" int ink_gemm_q8_0_sudot4(hipStream_t s, const uint8_t* W, const void* Xq, float* Y, int n, int k, int B) {
    bool use8 = (((k >> 5) & 31) != 0) && (((k >> 5) & 7) == 0);
    if (use8) {
        if (B == 8) { gemm_q8_0_sudot4_nt8_k<8><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k); return 0; }
        if (B == 7) { gemm_q8_0_sudot4_nt8_k<7><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k); return 0; }
        if (B == 6) { gemm_q8_0_sudot4_nt8_k<6><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k); return 0; }
        if (B == 4) { gemm_q8_0_sudot4_nt8_k<4><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k); return 0; }
        if (B == 1) { gemm_q8_0_sudot4_nt8_k<1><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k); return 0; }
    }
    gemm_q8_0_sudot4_k<<<dim3((n + 7) / 8, (B + 7) / 8), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k, B);
    return 0;
}


extern "C" int ink_gemm_q4_0_sudot4(hipStream_t s, const uint8_t* W, const void* Xq, float* Y, int n, int k, int B) {
    if (B == 1) return ink_gemv_q4_b1(s, W, Xq, Y, n, k);
    bool use8 = (((k >> 5) & 31) != 0) && (((k >> 5) & 7) == 0);
    if (B == 8) {
        if (use8) gemm_q4_0_sudot4_nt8_k<8><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<8><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        return 0;
    }
    if (B == 7) {
        if (use8) gemm_q4_0_sudot4_nt8_k<7><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<7><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        return 0;
    }
    if (B == 6) {
        if (use8) gemm_q4_0_sudot4_nt8_k<6><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<6><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        return 0;
    }
    if (B == 5) {
        if (use8) gemm_q4_0_sudot4_nt8_k<5><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<5><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        return 0;
    }
    if (B == 4) {
        if (use8) gemm_q4_0_sudot4_nt8_k<4><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<4><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        return 0;
    }
    if (B == 3) {
        if (use8) gemm_q4_0_sudot4_nt8_k<3><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<3><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        return 0;
    }
    if (B == 2) {
        if (use8) gemm_q4_0_sudot4_nt8_k<2><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<2><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        return 0;
    }
    gemm_q4_0_sudot4_k<<<dim3((n + 7) / 8, (B + 7) / 8), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k, B);
    return 0;
}


extern "C" int ink_moe_build_groups(hipStream_t s, const int* itop, const float* wtop, const float* scales, int* exp_counts, int* exp_offsets, int* tok_sorted, int* slot_sorted, float* wt_sorted, int n_used, int B) {
    moe_build_groups_k<<<dim3(1), dim3(128), 0, s>>>(itop, wtop, scales, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, n_used, B);
    return 0;
}


extern "C" int ink_moe_gu_nvfp4_sudot4(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, float* act_out, int n_ff, int k, int n_used, float beta, float lb) {
    moe_gu_nvfp4_sudot4_k<<<dim3((n_ff + 15) / 16, 128), dim3(128), 0, s>>>(W, scales, (const q8_1_blk*)Xq, exp_counts, exp_offsets, tok_sorted, slot_sorted, act_out, n_ff, k, n_used, beta, lb);
    return 0;
}


extern "C" __global__ void kv_conv_qkvr_store_multi_k(float* __restrict__ Q,
                                                      const float* __restrict__ K_raw,
                                                      const float* __restrict__ V_raw,
                                                      float* __restrict__ R,
                                                      const float* __restrict__ k_conv_w,
                                                      const float* __restrict__ v_conv_w,
                                                      const float* __restrict__ k_dyn,
                                                      const float* __restrict__ v_dyn,
                                                      float* __restrict__ k_hist_pool,
                                                      float* __restrict__ v_hist_pool,
                                                      const float* __restrict__ q_clip,
                                                      const float* __restrict__ rel_clip,
                                                      __half* __restrict__ k_cache_pool,
                                                      __half* __restrict__ v_cache_pool,
                                                      const int* __restrict__ d_seq_ids,
                                                      const int* __restrict__ d_positions,
                                                      int is_swa, int n_head, int n_head_kv,
                                                      int dh, int d_rel, int max_ctx,
                                                      float qk_thresh, float rel_thresh,
                                                      float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int lane = threadIdx.x;
    int seq = d_seq_ids[bt];
    int pos = d_positions[bt];
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (h < n_head) {
        float tau = 1.0f;
        if (!is_swa) {
            float ratio = ((float)pos + 1.0f) / log_floor;
            if (ratio < 1.0f) ratio = 1.0f;
            tau = 1.0f + log_alpha * logf(ratio);
        }
        float* qh = Q + (size_t)bt * q_dim + (size_t)h * dh;
        const float* qch = q_clip + (size_t)h * dh;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        float ss = s_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;

        float* rh = R + (size_t)bt * r_dim + (size_t)h * d_rel;
        const float* rch = rel_clip + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = s_warp_sum32(rv * rv);
        float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
        if (lane < d_rel) rh[lane] = rv * rfactor;
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        int slot = is_swa ? (pos % max_ctx) : pos;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
        float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
        if (k_dyn) {
            const float* dk = k_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
            const float* dv = v_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
            kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
            vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
        }
        float* k_hist = k_hist_pool + (size_t)seq * kv_dim * 3;
        float* v_hist = v_hist_pool + (size_t)seq * kv_dim * 3;
        float kv_v[4], vv_v[4];
        float kss = 0.0f, vss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            float* khc = k_hist + (size_t)c * 3;
            float kh0 = khc[0], kh1 = khc[1], kh2 = khc[2];
            float kx = K_raw[(size_t)bt * kv_dim + c];
            const float* kwc = k_conv_w + (size_t)c * 4;
            float kacc = kwc[0] * kh0 + kwc[1] * kh1 + kwc[2] * kh2 + kwc[3] * kx;
            if (k_dyn) kacc += kd0 * kx + kd1 * kh2 + kd2 * kh1 + kd3 * kh0;
            khc[0] = kh1; khc[1] = kh2; khc[2] = kx;
            float kout = (kx + kacc) * sc;
            kv_v[p] = kout;
            kss += kout * kout;

            float* vhc = v_hist + (size_t)c * 3;
            float vh0 = vhc[0], vh1 = vhc[1], vh2 = vhc[2];
            float vx = V_raw[(size_t)bt * kv_dim + c];
            const float* vwc = v_conv_w + (size_t)c * 4;
            float vacc = vwc[0] * vh0 + vwc[1] * vh1 + vwc[2] * vh2 + vwc[3] * vx;
            if (v_dyn) vacc += vd0 * vx + vd1 * vh2 + vd2 * vh1 + vd3 * vh0;
            vhc[0] = vh1; vhc[1] = vh2; vhc[2] = vx;
            float vout = (vx + vacc) * sc;
            vv_v[p] = vout;
            vss += vout * vout;
        }
        kss = s_warp_sum32(kss);
        vss = s_warp_sum32(vss);
        float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
        float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
        size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
        __half* kc_slot = k_cache_pool + (size_t)seq * seq_kv_stride + ((size_t)kh_idx * max_ctx + slot) * dh;
        __half* vc_slot = v_cache_pool + (size_t)seq * seq_kv_stride + ((size_t)kh_idx * max_ctx + slot) * dh;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            kc_slot[d] = __float2half(kv_v[p] * kfac);
            vc_slot[d] = __float2half(vv_v[p] * vinv);
        }
    }
}


extern "C" __global__ void inkling_attn_multi_gated_k(const float* __restrict__ Q,
                                                      const float* __restrict__ R,
                                                      const float* __restrict__ sinks,
                                                      const float* __restrict__ rel_bias,
                                                      const __half* __restrict__ k_cache_pool,
                                                      const __half* __restrict__ v_cache_pool,
                                                      const float* __restrict__ gate,
                                                      const int* __restrict__ d_seq_ids,
                                                      const int* __restrict__ d_positions,
                                                      float* __restrict__ out,
                                                      int n_head, int n_head_kv,
                                                      int dh, int d_rel,
                                                      int rel_extent, int swa_window,
                                                      int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int seq = d_seq_ids[bt];
    int pos = d_positions[bt];
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    __shared__ float qs[128];
    __shared__ float rs[16];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[(size_t)bt * q_dim + (size_t)h * dh + d] * q_scale;
    }
    if (threadIdx.x < d_rel) {
        rs[threadIdx.x] = R[(size_t)bt * r_dim + (size_t)h * d_rel + threadIdx.x];
    }
    __syncthreads();

    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[(size_t)bt * n_head + h];

    size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
    const __half* kc_h = k_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q0a = qs[p0], q0b = qs[p0 + 1], q1a = qs[p1], q1b = qs[p1 + 1];

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        float dot = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
        dot = s_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = s_warp_sum32(rp);
            rbias = __shfl(rp, 0, 32);
        }
        score += rbias - sink_h;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        o0 = fmaf(p, v0.x, o0 * alpha);
        o1 = fmaf(p, v0.y, o1 * alpha);
        o2 = fmaf(p, v1.x, o2 * alpha);
        o3 = fmaf(p, v1.y, o3 * alpha);
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
        float* oh = out + (size_t)bt * q_dim + (size_t)h * dh;
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        oh[d0] = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        oh[d1] = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        oh[d2] = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        oh[d3] = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
    }
}


extern "C" __global__ void shortconv_res_multi_k(const float* __restrict__ raw,
                                                 const float* __restrict__ conv_w,
                                                 float* __restrict__ hist_pool,
                                                 const float* __restrict__ add_src,
                                                 float* __restrict__ block_dst,
                                                 float* __restrict__ out,
                                                 const int* __restrict__ d_seq_ids,
                                                 int n_embd) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (c >= n_embd) return;
    int seq = d_seq_ids[bt];
    float* hc = hist_pool + ((size_t)seq * n_embd + c) * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = conv_w + (size_t)c * 4;
    size_t idx = (size_t)bt * n_embd + c;
    float x = raw[idx];
    float conv_out = (x + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * x) * rsqrtf(5.0f);
    hc[0] = h1; hc[1] = h2; hc[2] = x;
    if (block_dst) {
        block_dst[idx] = add_src[idx];
        out[idx] = conv_out;
    } else {
        out[idx] = add_src[idx] + conv_out;
    }
}


extern "C" int ink_kv_conv_qkvr_store_multi(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist_pool, float* v_hist_pool, const float* q_clip, const float* rel_clip, __half* kc_pool, __half* vc_pool, const int* d_seq_ids, const int* d_positions, int B, int is_swa, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_multi_k<<<dim3(n_head + n_head_kv, B), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist_pool, v_hist_pool, q_clip, rel_clip, kc_pool, vc_pool, d_seq_ids, d_positions, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}


extern "C" int ink_attn_multi_gated(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc_pool, const __half* vc_pool, const float* gate, const int* d_seq_ids, const int* d_positions, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    inkling_attn_multi_gated_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, gate, d_seq_ids, d_positions, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}


extern "C" __global__ void shortconv_res_multi_clear_k(float* __restrict__ raw,
                                                       const float* __restrict__ conv_w,
                                                       float* __restrict__ hist_pool,
                                                       const float* __restrict__ add_src,
                                                       float* __restrict__ block_dst,
                                                       float* __restrict__ out,
                                                       const int* __restrict__ d_seq_ids,
                                                       int n_embd) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (c >= n_embd) return;
    int seq = d_seq_ids[bt];
    float* hc = hist_pool + ((size_t)seq * n_embd + c) * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = conv_w + (size_t)c * 4;
    size_t idx = (size_t)bt * n_embd + c;
    float x = raw[idx];
    raw[idx] = 0.0f;
    float conv_out = (x + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * x) * rsqrtf(5.0f);
    hc[0] = h1; hc[1] = h2; hc[2] = x;
    if (block_dst) {
        block_dst[idx] = add_src[idx];
        out[idx] = conv_out;
    } else {
        out[idx] = add_src[idx] + conv_out;
    }
}


extern "C" int ink_shortconv_res_multi(hipStream_t s, const float* raw, const float* conv_w, float* hist_pool, const float* add_src, float* block_dst, float* out, const int* d_seq_ids, int n_embd, int B) {
    shortconv_res_multi_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(raw, conv_w, hist_pool, add_src, block_dst, out, d_seq_ids, n_embd);
    return 0;
}


extern "C" int ink_shortconv_res_multi_clear(hipStream_t s, float* raw, const float* conv_w, float* hist_pool, const float* add_src, float* block_dst, float* out, const int* d_seq_ids, int n_embd, int B) {
    shortconv_res_multi_clear_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(raw, conv_w, hist_pool, add_src, block_dst, out, d_seq_ids, n_embd);
    return 0;
}


extern "C" int ink_moe_dn_nvfp4_sudot4(hipStream_t s, const uint8_t* W, const void* act_q, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, const float* wt_sorted, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    int tot = B * n_embd;
    ink_zero_f32(s, y_out, tot);
    moe_dn_nvfp4_sudot4_k<<<dim3((n_embd + 15) / 16, 128), dim3(128), 0, s>>>(W, (const q8_1_blk*)act_q, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, y_out, n_embd, n_ff, n_used);
    return 0;
}


extern "C" __global__ void q8_0_to_q4_0_k(const uint8_t* __restrict__ src,
                                          uint8_t* __restrict__ dst,
                                          size_t n_rows, int nblk) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_rows * (size_t)nblk;
    if (idx >= total) return;
    size_t row = idx / (size_t)nblk;
    int b = (int)(idx % (size_t)nblk);
    const uint8_t* s = src + idx * 34;
    size_t rbytes = q4_soa_row_bytes(nblk);
    uint8_t* drow = dst + row * rbytes;
    uint16_t dh_bits;
    memcpy(&dh_bits, s, 2);
    float d_in = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float v = (float)(int8_t)s[2 + j] * d_in;
        vals[j] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    float d_out = amax / 7.5f;
    float inv = (d_out > 0.0f) ? (1.0f / d_out) : 0.0f;
    reinterpret_cast<__half*>(drow + (size_t)nblk * 16)[b] = __float2half(d_out);
    uint8_t* qo = drow + (size_t)b * 16;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int q0 = (int)floorf(vals[j] * inv + 8.5f);
        int q1 = (int)floorf(vals[j + 16] * inv + 8.5f);
        q0 = max(0, min(15, q0));
        q1 = max(0, min(15, q1));
        qo[j] = (uint8_t)(q0 | (q1 << 4));
    }
}


extern "C" __global__ void wo_conv_res_q4_0_sudot4_k(const uint8_t* __restrict__ W,
                                                     const q8_1_blk* __restrict__ Xq,
                                                     const float* __restrict__ conv_w,
                                                     float* __restrict__ hist,
                                                     const float* __restrict__ add_src,
                                                     float* __restrict__ block_dst,
                                                     float* __restrict__ out,
                                                     int n_embd, int q_dim) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n_embd) return;
    int nblk = q_dim >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), Xq[b]);
    }
    acc = s_warp_sum32(acc);
    if (lane == 0) {
        float* hc = hist + (size_t)row * 3;
        const float* wc = conv_w + (size_t)row * 4;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        float conv_out = (acc + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * acc) * rsqrtf(5.0f);
        hc[0] = h1; hc[1] = h2; hc[2] = acc;
        if (block_dst) {
            block_dst[row] = add_src[row];
            out[row] = conv_out;
        } else {
            out[row] = add_src[row] + conv_out;
        }
    }
}


extern "C" int ink_q8_0_to_q4_0(hipStream_t s, const uint8_t* src, uint8_t* dst, size_t n_rows, int nblk) {
    size_t total = n_rows * (size_t)nblk;
    size_t grid = (total + 255) / 256;
    q8_0_to_q4_0_k<<<dim3((unsigned int)grid), dim3(256), 0, s>>>(src, dst, n_rows, nblk);
    return 0;
}


extern "C" int ink_wo_conv_res_q4_0_sudot4(hipStream_t s, const uint8_t* W, const void* Xq, const float* conv_w, float* hist, const float* add_src, float* block_dst, float* out, int n_embd, int q_dim) {
    wo_conv_res_q4_b1_8_k<<<dim3((n_embd + 15) / 16), dim3(128), 0, s>>>(W, (const q8_1_blk*)Xq, conv_w, hist, add_src, block_dst, out, n_embd, q_dim);
    return 0;
}


extern "C" __global__ void dflash2_kv_q4_k(const uint8_t* __restrict__ Wk,
                                           const uint8_t* __restrict__ Wv,
                                           const q8_1_blk* __restrict__ Xq,
                                           float* __restrict__ K_out,
                                           float* __restrict__ V_out,
                                           int kv_dim, int k_dim, int B) {
    int tid = threadIdx.x;
    int grp = tid >> 3;
    int lane8 = tid & 7;
    int row = blockIdx.x * 16 + grp;
    int is_v = blockIdx.y;
    const uint8_t* W = is_v ? Wv : Wk;
    float* Y = is_v ? V_out : K_out;
    int nblk = k_dim >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = (row < kv_dim) ? (W + (size_t)row * rbytes) : W;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    __shared__ q8_1_blk sm_x[8][8];
    float acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int b0 = 0; b0 < nblk; b0 += 8) {
        if (tid < B * 8) {
            int tt = tid >> 3;
            int bb = b0 + (tid & 7);
            if (bb < nblk) sm_x[tt][tid & 7] = Xq[(size_t)tt * nblk + bb];
        }
        __syncthreads();
        int b = b0 + lane8;
        if (row < kv_dim && b < nblk) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu);
            int32_t w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
            int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu);
            int32_t w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
            int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu);
            int32_t w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
            int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu);
            int32_t w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
            for (int t = 0; t < B; ++t) {
                const q8_1_blk xb = sm_x[t][lane8];
                int32_t sumi = 0;
                sumi = __builtin_amdgcn_sudot4(false, w0, true, xb.q[0], sumi, false);
                sumi = __builtin_amdgcn_sudot4(false, w1, true, xb.q[1], sumi, false);
                sumi = __builtin_amdgcn_sudot4(false, w2, true, xb.q[2], sumi, false);
                sumi = __builtin_amdgcn_sudot4(false, w3, true, xb.q[3], sumi, false);
                sumi = __builtin_amdgcn_sudot4(false, w4, true, xb.q[4], sumi, false);
                sumi = __builtin_amdgcn_sudot4(false, w5, true, xb.q[5], sumi, false);
                sumi = __builtin_amdgcn_sudot4(false, w6, true, xb.q[6], sumi, false);
                sumi = __builtin_amdgcn_sudot4(false, w7, true, xb.q[7], sumi, false);
                acc[t] = fmaf(dw * xb.d, (float)(sumi - 8 * xb.s), acc[t]);
            }
        }
        __syncthreads();
    }
    if (row < kv_dim) {
        for (int t = 0; t < B; ++t) {
            float v = s_warp_sum8(acc[t]);
            if (lane8 == 0) Y[(size_t)t * kv_dim + row] = v;
        }
    }
}

extern "C" int ink_dflash2_kv_q4(hipStream_t s, const uint8_t* wk, const uint8_t* wv, const void* xq, float* k_out, float* v_out, int kv_dim, int k_dim, int b) {
    dflash2_kv_q4_k<<<dim3((kv_dim + 15) / 16, 2), dim3(128), 0, s>>>(wk, wv, (const q8_1_blk*)xq, k_out, v_out, kv_dim, k_dim, b);
    return 0;
}

extern "C" int ink_zero_f32(hipStream_t s, float* ptr, int n) {
    ink_zero_f32(s, ptr, n);
    return 0;
}
