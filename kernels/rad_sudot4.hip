#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

struct q8_1_blk {
    float d;
    int32_t s;
    int32_t q[8];
};

__device__ __forceinline__ float s_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t man = x & 0x7;
    if (exp == 0) return (float)man * 0.0009765625f;
    uint32_t bits = ((exp + 119u) << 23) | (man << 20);
    return __uint_as_float(bits);
}

__device__ __forceinline__ int8_t s_fp4_i8(uint32_t q) {
    uint32_t n = q & 7u;
    uint32_t e = n >> 1;
    uint32_t m = n & 1u;
    int v = (e == 0u) ? (int)m : (int)((2u + m) << (e - 1u));
    return (int8_t)((q & 8u) ? -v : v);
}

__device__ __forceinline__ int32_t pack4_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (int32_t)((uint32_t)(uint8_t)a
                   | ((uint32_t)(uint8_t)b << 8)
                   | ((uint32_t)(uint8_t)c << 16)
                   | ((uint32_t)(uint8_t)d << 24));
}

__device__ __forceinline__ float s_warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ int32_t s_warp_sum32_i(int32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ float s_warp_sum8(float v) {
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) v += __shfl_down(v, off, 8);
    return v;
}

__device__ __forceinline__ float s_warp_sum4(float v) {
    v += __shfl_down(v, 2, 4);
    v += __shfl_down(v, 1, 4);
    return v;
}

__device__ __forceinline__ float s_warp_max32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
    return v;
}

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

__device__ __forceinline__ float dot_nvfp4_q8_1_blk(const uint8_t* __restrict__ bp,
                                                    const q8_1_blk* __restrict__ xq2) {
    uint32_t ds;
    memcpy(&ds, bp, 4);
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bp + 4);
    float acc = 0.0f;
#pragma unroll
    for (int sb = 0; sb < 4; ++sb) {
        float d = s_ue4m3_to_fp32((uint8_t)((ds >> (sb * 8)) & 0xFF));
        uint32_t q0 = q32[sb * 2];
        uint32_t q1 = q32[sb * 2 + 1];
        const q8_1_blk& xb = xq2[sb >> 1];
        int qoff = (sb & 1) * 4;
        int32_t wlo0 = pack4_i8(s_fp4_i8(q0), s_fp4_i8(q0 >> 8), s_fp4_i8(q0 >> 16), s_fp4_i8(q0 >> 24));
        int32_t whi0 = pack4_i8(s_fp4_i8(q0 >> 4), s_fp4_i8(q0 >> 12), s_fp4_i8(q0 >> 20), s_fp4_i8(q0 >> 28));
        int32_t wlo1 = pack4_i8(s_fp4_i8(q1), s_fp4_i8(q1 >> 8), s_fp4_i8(q1 >> 16), s_fp4_i8(q1 >> 24));
        int32_t whi1 = pack4_i8(s_fp4_i8(q1 >> 4), s_fp4_i8(q1 >> 12), s_fp4_i8(q1 >> 20), s_fp4_i8(q1 >> 28));
        int32_t sumi = 0;
        sumi = __builtin_amdgcn_sudot4(true, wlo0, true, xb.q[qoff + 0], sumi, false);
        sumi = __builtin_amdgcn_sudot4(true, wlo1, true, xb.q[qoff + 1], sumi, false);
        sumi = __builtin_amdgcn_sudot4(true, whi0, true, xb.q[qoff + 2], sumi, false);
        sumi = __builtin_amdgcn_sudot4(true, whi1, true, xb.q[qoff + 3], sumi, false);
        acc = fmaf(d * xb.d, (float)sumi, acc);
    }
    return acc;
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

__device__ __forceinline__ size_t q4_soa_row_bytes(int nblk) {
    return (size_t)nblk * 16 + (((size_t)nblk * 2 + 15) & ~(size_t)15);
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

extern "C" __global__ void zero_buf_f32_k(float* __restrict__ p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
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

extern "C" int ink_gemm_q8_0_sudot4(hipStream_t s, const uint8_t* W, const void* Xq, float* Y, int n, int k, int B) {
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
    if (B == 4) {
        if (use8) gemm_q4_0_sudot4_nt8_k<4><<<dim3((n + 31) / 32, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
        else gemm_q4_0_sudot4_nt_k<4><<<dim3((n + 7) / 8, 1), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
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

extern "C" int ink_shortconv_res_multi(hipStream_t s, const float* raw, const float* conv_w, float* hist_pool, const float* add_src, float* block_dst, float* out, const int* d_seq_ids, int n_embd, int B) {
    shortconv_res_multi_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(raw, conv_w, hist_pool, add_src, block_dst, out, d_seq_ids, n_embd);
    return 0;
}

extern "C" int ink_moe_dn_nvfp4_sudot4(hipStream_t s, const uint8_t* W, const void* act_q, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, const float* wt_sorted, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    int tot = B * n_embd;
    zero_buf_f32_k<<<dim3((tot + 255) / 256), dim3(256), 0, s>>>(y_out, tot);
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

__device__ __forceinline__ float dot_q4_soa_q8_1_blk(const uint4 q4,
                                                     float dw,
                                                     const q8_1_blk& xb) {
    int32_t sumi = 0;
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.x & 0x0F0F0F0Fu), true, xb.q[0], sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.y & 0x0F0F0F0Fu), true, xb.q[1], sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.z & 0x0F0F0F0Fu), true, xb.q[2], sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)(q4.w & 0x0F0F0F0Fu), true, xb.q[3], sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), true, xb.q[4], sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu), true, xb.q[5], sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), true, xb.q[6], sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu), true, xb.q[7], sumi, false);
    return dw * xb.d * (float)(sumi - 8 * xb.s);
}

__device__ __forceinline__ float dot_q8_0_q8_1_blk(const uint8_t* __restrict__ bp,
                                                   const q8_1_blk& xb) {
    uint16_t dh_bits;
    memcpy(&dh_bits, bp, 2);
    float dw = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
    int32_t wq[8];
    memcpy(wq, bp + 2, 32);
    int32_t sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        sumi = __builtin_amdgcn_sudot4(true, wq[j], true, xb.q[j], sumi, false);
    }
    return dw * xb.d * (float)sumi;
}

__device__ __forceinline__ float s_bf16_f32(uint16_t u) {
    uint32_t bits = (uint32_t)u << 16;
    return __uint_as_float(bits);
}

extern "C" __global__ void qkvrg_gemm_sudot4_k(const uint8_t* __restrict__ wq,
                                               const uint8_t* __restrict__ wk,
                                               const uint8_t* __restrict__ wv,
                                               const uint8_t* __restrict__ wr,
                                               const uint8_t* __restrict__ wg,
                                               const uint16_t* __restrict__ wsink,
                                               const uint16_t* __restrict__ wkdyn,
                                               const uint16_t* __restrict__ wvdyn,
                                               const q8_1_blk* __restrict__ Xq0,
                                               const q8_1_blk* __restrict__ Xq1,
                                               const q8_1_blk* __restrict__ Xq2,
                                               const float* __restrict__ X0,
                                               const float* __restrict__ X1,
                                               const float* __restrict__ X2,
                                               float* __restrict__ Q,
                                               float* __restrict__ K,
                                               float* __restrict__ V,
                                               float* __restrict__ R,
                                               float* __restrict__ G,
                                               float* __restrict__ S,
                                               float* __restrict__ Kdyn,
                                               float* __restrict__ Vdyn,
                                               int k, int q_dim, int kv_dim,
                                               int r_dim, int n_head, int dyn_dim,
                                               int B, int is_q4) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int gid = blockIdx.x * 4 + warp;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;

    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    int o6 = o5 + n_head;
    int o7 = o6 + dyn_dim;
    int o8 = o7 + dyn_dim;
    if (gid >= o8) return;

    if (gid < o5) {
        const uint8_t* W;
        const q8_1_blk* Xq;
        float* Y;
        int row, n_out;
        if (gid < o1) { W = wq; Xq = Xq0; Y = Q; row = gid; n_out = q_dim; }
        else if (gid < o2) { W = wk; Xq = Xq1; Y = K; row = gid - o1; n_out = kv_dim; }
        else if (gid < o3) { W = wv; Xq = Xq2; Y = V; row = gid - o2; n_out = kv_dim; }
        else if (gid < o4) { W = wr; Xq = Xq0; Y = R; row = gid - o3; n_out = r_dim; }
        else { W = wg; Xq = Xq0; Y = G; row = gid - o4; n_out = q_dim; }

        int nblk = k >> 5;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (is_q4) {
            size_t rbytes = q4_soa_row_bytes(nblk);
            const uint8_t* wrow = W + (size_t)row * rbytes;
            const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
            const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
            for (int b = lane; b < nblk; b += 32) {
                uint4 q4 = qrow[b];
                float dw = __half2float(drow[b]);
                for (int m = 0; m < n_t; ++m) {
                    const q8_1_blk xb = Xq[(size_t)(t0 + m) * nblk + b];
                    acc[m] += dot_q4_soa_q8_1_blk(q4, dw, xb);
                }
            }
        } else {
            const uint8_t* wrow = W + (size_t)row * ((size_t)nblk * 34);
            for (int b = lane; b < nblk; b += 32) {
                const uint8_t* bp = wrow + (size_t)b * 34;
                uint16_t dh_bits;
                memcpy(&dh_bits, bp, 2);
                float dw = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
                int32_t wq8[8];
                memcpy(wq8, bp + 2, 32);
                for (int m = 0; m < n_t; ++m) {
                    const q8_1_blk xb = Xq[(size_t)(t0 + m) * nblk + b];
                    int32_t sumi = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sumi = __builtin_amdgcn_sudot4(true, wq8[j], true, xb.q[j], sumi, false);
                    }
                    acc[m] = fmaf(dw * xb.d, (float)sumi, acc[m]);
                }
            }
        }
        for (int m = 0; m < n_t; ++m) {
            float v = s_warp_sum32(acc[m]);
            if (lane == 0) Y[(size_t)(t0 + m) * n_out + row] = v;
        }
    } else {
        const uint16_t* W;
        const float* X;
        float* Y;
        int row, n_out;
        if (gid < o6) { W = wsink; X = X0; Y = S; row = gid - o5; n_out = n_head; }
        else if (gid < o7) { W = wkdyn; X = X1; Y = Kdyn; row = gid - o6; n_out = dyn_dim; }
        else { W = wvdyn; X = X2; Y = Vdyn; row = gid - o7; n_out = dyn_dim; }
        const uint16_t* wrow = W + (size_t)row * k;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int j = lane; j < k; j += 32) {
            float wv_f = s_bf16_f32(wrow[j]);
            for (int m = 0; m < n_t; ++m) {
                acc[m] = fmaf(wv_f, X[(size_t)(t0 + m) * k + j], acc[m]);
            }
        }
        for (int m = 0; m < n_t; ++m) {
            float v = s_warp_sum32(acc[m]);
            if (lane == 0) Y[(size_t)(t0 + m) * n_out + row] = v;
        }
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

extern "C" __global__ void moe_gu_nvfp4_b1_sudot4_k(const uint8_t* __restrict__ W,
                                                    const float* __restrict__ scales,
                                                    const q8_1_blk* __restrict__ Xq,
                                                    const int* __restrict__ itop,
                                                    float* __restrict__ act_out,
                                                    int n_ff, int k, float beta, float lb) {
    int slot = blockIdx.y;
    int exp_id = itop[slot];
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_ff) return;

    int nblk = k >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + row) * row_bytes;

    float ag = 0.0f, au = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        const q8_1_blk* xq2 = Xq + (b << 1);
        ag += dot_nvfp4_q8_1_blk(wg + (size_t)b * 36, xq2);
        au += dot_nvfp4_q8_1_blk(wu + (size_t)b * 36, xq2);
    }
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = s_warp_sum8(ag) * sc;
    float u = s_warp_sum8(au) * sc;
    if (lane8 == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[(size_t)slot * n_ff + row] = act * up;
    }
}

extern "C" __global__ void moe_dn_conv_add_nvfp4_b1_sudot4_k(const uint8_t* __restrict__ W,
                                                             const float* __restrict__ scales,
                                                             const q8_1_blk* __restrict__ act_q,
                                                             const int* __restrict__ itop,
                                                             const float* __restrict__ wtop,
                                                             const float* __restrict__ conv_w,
                                                             float* __restrict__ hist,
                                                             const float* __restrict__ ffn_inp,
                                                             float* __restrict__ cur,
                                                             int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;

    int nblk = n_ff >> 6;
    int q8_nblk = n_ff >> 5;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)n_embd * row_bytes;

    float total = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        int exp_id = itop[s];
        float wt = wtop[s] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        const q8_1_blk* aq = act_q + (size_t)s * q8_nblk;
        float acc = 0.0f;
        for (int b = lane8; b < nblk; b += 8) {
            acc += dot_nvfp4_q8_1_blk(wrow + (size_t)b * 36, aq + (b << 1));
        }
        total += acc * wt;
    }
    float v = s_warp_sum8(total);
    if (lane8 == 0) {
        float* hc = hist + (size_t)row * 3;
        const float* wc = conv_w + (size_t)row * 4;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        float conv_out = (v + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * v) * rsqrtf(5.0f);
        hc[0] = h1; hc[1] = h2; hc[2] = v;
        cur[row] = ffn_inp[row] + conv_out;
    }
}

extern "C" __global__ void moe_gu_q4_0_b1_sudot4_k(const uint8_t* __restrict__ W,
                                                   const float* __restrict__ scales,
                                                   const q8_1_blk* __restrict__ Xq,
                                                   const int* __restrict__ itop,
                                                   float* __restrict__ act_out,
                                                   int n_ff, int k, float beta, float lb) {
    int slot = blockIdx.y;
    int exp_id = itop[slot];
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_ff) return;

    int nblk = k >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + row) * row_bytes;
    const uint4* qg_row = reinterpret_cast<const uint4*>(wg);
    const __half* dg_row = reinterpret_cast<const __half*>(wg + (size_t)nblk * 16);
    const uint4* qu_row = reinterpret_cast<const uint4*>(wu);
    const __half* du_row = reinterpret_cast<const __half*>(wu + (size_t)nblk * 16);

    float ag = 0.0f, au = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        const q8_1_blk xb = Xq[b];
        ag += dot_q4_soa_q8_1_blk(qg_row[b], __half2float(dg_row[b]), xb);
        au += dot_q4_soa_q8_1_blk(qu_row[b], __half2float(du_row[b]), xb);
    }
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = s_warp_sum8(ag) * sc;
    float u = s_warp_sum8(au) * sc;
    if (lane8 == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[(size_t)slot * n_ff + row] = act * up;
    }
}

extern "C" __global__ void moe_dn_conv_add_q4_0_b1_sudot4_k(const uint8_t* __restrict__ W,
                                                            const float* __restrict__ scales,
                                                            const q8_1_blk* __restrict__ act_q,
                                                            const int* __restrict__ itop,
                                                            const float* __restrict__ wtop,
                                                            const float* __restrict__ conv_w,
                                                            float* __restrict__ hist,
                                                            const float* __restrict__ ffn_inp,
                                                            float* __restrict__ cur,
                                                            int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;

    int nblk = n_ff >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * row_bytes;

    float total = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        int exp_id = itop[s];
        float wt = wtop[s] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
        const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
        const q8_1_blk* aq = act_q + (size_t)s * nblk;
        float acc = 0.0f;
        for (int b = lane8; b < nblk; b += 8) {
            acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), aq[b]);
        }
        total += acc * wt;
    }
    float v = s_warp_sum8(total);
    if (lane8 == 0) {
        float* hc = hist + (size_t)row * 3;
        const float* wc = conv_w + (size_t)row * 4;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        float conv_out = (v + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * v) * rsqrtf(5.0f);
        hc[0] = h1; hc[1] = h2; hc[2] = v;
        cur[row] = ffn_inp[row] + conv_out;
    }
}

extern "C" __global__ void moe_gu_q4_0_sudot4_k(const uint8_t* __restrict__ W,
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

    int nblk = k >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + row) * row_bytes;
    const uint4* qg_row = reinterpret_cast<const uint4*>(wg);
    const __half* dg_row = reinterpret_cast<const __half*>(wg + (size_t)nblk * 16);
    const uint4* qu_row = reinterpret_cast<const uint4*>(wu);
    const __half* du_row = reinterpret_cast<const __half*>(wu + (size_t)nblk * 16);
    float sc = scales ? scales[exp_id] : 1.0f;

    for (int t0 = 0; t0 < cnt; t0 += 8) {
        int nt = min(8, cnt - t0);
        int tids[8];
        for (int m = 0; m < nt; ++m) tids[m] = tok_sorted[off0 + t0 + m];
        float ag[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int b = lane8; b < nblk; b += 8) {
            uint4 qg = qg_row[b];
            uint4 qu = qu_row[b];
            float dg = __half2float(dg_row[b]);
            float du = __half2float(du_row[b]);
            for (int m = 0; m < nt; ++m) {
                const q8_1_blk xb = Xq[(size_t)tids[m] * nblk + b];
                ag[m] += dot_q4_soa_q8_1_blk(qg, dg, xb);
                au[m] += dot_q4_soa_q8_1_blk(qu, du, xb);
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

extern "C" __global__ void moe_dn_q4_0_sudot4_k(const uint8_t* __restrict__ W,
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
    int grp = threadIdx.x >> 2;
    int lane4 = threadIdx.x & 3;
    int row = blockIdx.x * 32 + grp;
    if (row >= n_embd) return;

    int nblk = n_ff >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);

    if (cnt == 1) {
        int tid0 = tok_sorted[off0], sid0 = slot_sorted[off0];
        float acc0 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            const q8_1_blk ab = act_q[((size_t)tid0 * n_used + sid0) * nblk + b];
            acc0 += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), ab);
        }
        float v = s_warp_sum4(acc0) * wt_sorted[off0];
        if (lane4 == 0) atomicAdd(&y_out[(size_t)tid0 * n_embd + row], v);
        return;
    }
    if (cnt == 2) {
        int tid0 = tok_sorted[off0], sid0 = slot_sorted[off0];
        int tid1 = tok_sorted[off0 + 1], sid1 = slot_sorted[off0 + 1];
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu), w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
            int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu), w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
            int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
            int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
            const q8_1_blk ab0 = act_q[((size_t)tid0 * n_used + sid0) * nblk + b];
            int32_t s0 = 0;
            s0 = __builtin_amdgcn_sudot4(false, w0, true, ab0.q[0], s0, false);
            s0 = __builtin_amdgcn_sudot4(false, w1, true, ab0.q[1], s0, false);
            s0 = __builtin_amdgcn_sudot4(false, w2, true, ab0.q[2], s0, false);
            s0 = __builtin_amdgcn_sudot4(false, w3, true, ab0.q[3], s0, false);
            s0 = __builtin_amdgcn_sudot4(false, w4, true, ab0.q[4], s0, false);
            s0 = __builtin_amdgcn_sudot4(false, w5, true, ab0.q[5], s0, false);
            s0 = __builtin_amdgcn_sudot4(false, w6, true, ab0.q[6], s0, false);
            s0 = __builtin_amdgcn_sudot4(false, w7, true, ab0.q[7], s0, false);
            acc0 = fmaf(dw * ab0.d, (float)(s0 - 8 * ab0.s), acc0);
            const q8_1_blk ab1 = act_q[((size_t)tid1 * n_used + sid1) * nblk + b];
            int32_t s1 = 0;
            s1 = __builtin_amdgcn_sudot4(false, w0, true, ab1.q[0], s1, false);
            s1 = __builtin_amdgcn_sudot4(false, w1, true, ab1.q[1], s1, false);
            s1 = __builtin_amdgcn_sudot4(false, w2, true, ab1.q[2], s1, false);
            s1 = __builtin_amdgcn_sudot4(false, w3, true, ab1.q[3], s1, false);
            s1 = __builtin_amdgcn_sudot4(false, w4, true, ab1.q[4], s1, false);
            s1 = __builtin_amdgcn_sudot4(false, w5, true, ab1.q[5], s1, false);
            s1 = __builtin_amdgcn_sudot4(false, w6, true, ab1.q[6], s1, false);
            s1 = __builtin_amdgcn_sudot4(false, w7, true, ab1.q[7], s1, false);
            acc1 = fmaf(dw * ab1.d, (float)(s1 - 8 * ab1.s), acc1);
        }
        float v0 = s_warp_sum4(acc0) * wt_sorted[off0];
        float v1 = s_warp_sum4(acc1) * wt_sorted[off0 + 1];
        if (lane4 == 0) {
            atomicAdd(&y_out[(size_t)tid0 * n_embd + row], v0);
            atomicAdd(&y_out[(size_t)tid1 * n_embd + row], v1);
        }
        return;
    }
    if (cnt == 3) {
        int t0i = tok_sorted[off0], s0i = slot_sorted[off0];
        int t1i = tok_sorted[off0 + 1], s1i = slot_sorted[off0 + 1];
        int t2i = tok_sorted[off0 + 2], s2i = slot_sorted[off0 + 2];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            a0 += dot_q4_soa_q8_1_blk(q4, dw, act_q[((size_t)t0i * n_used + s0i) * nblk + b]);
            a1 += dot_q4_soa_q8_1_blk(q4, dw, act_q[((size_t)t1i * n_used + s1i) * nblk + b]);
            a2 += dot_q4_soa_q8_1_blk(q4, dw, act_q[((size_t)t2i * n_used + s2i) * nblk + b]);
        }
        float v0 = s_warp_sum4(a0) * wt_sorted[off0];
        float v1 = s_warp_sum4(a1) * wt_sorted[off0 + 1];
        float v2 = s_warp_sum4(a2) * wt_sorted[off0 + 2];
        if (lane4 == 0) {
            atomicAdd(&y_out[(size_t)t0i * n_embd + row], v0);
            atomicAdd(&y_out[(size_t)t1i * n_embd + row], v1);
            atomicAdd(&y_out[(size_t)t2i * n_embd + row], v2);
        }
        return;
    }
    if (cnt == 4) {
        int t0i = tok_sorted[off0], s0i = slot_sorted[off0];
        int t1i = tok_sorted[off0 + 1], s1i = slot_sorted[off0 + 1];
        int t2i = tok_sorted[off0 + 2], s2i = slot_sorted[off0 + 2];
        int t3i = tok_sorted[off0 + 3], s3i = slot_sorted[off0 + 3];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            a0 += dot_q4_soa_q8_1_blk(q4, dw, act_q[((size_t)t0i * n_used + s0i) * nblk + b]);
            a1 += dot_q4_soa_q8_1_blk(q4, dw, act_q[((size_t)t1i * n_used + s1i) * nblk + b]);
            a2 += dot_q4_soa_q8_1_blk(q4, dw, act_q[((size_t)t2i * n_used + s2i) * nblk + b]);
            a3 += dot_q4_soa_q8_1_blk(q4, dw, act_q[((size_t)t3i * n_used + s3i) * nblk + b]);
        }
        float v0 = s_warp_sum4(a0) * wt_sorted[off0];
        float v1 = s_warp_sum4(a1) * wt_sorted[off0 + 1];
        float v2 = s_warp_sum4(a2) * wt_sorted[off0 + 2];
        float v3 = s_warp_sum4(a3) * wt_sorted[off0 + 3];
        if (lane4 == 0) {
            atomicAdd(&y_out[(size_t)t0i * n_embd + row], v0);
            atomicAdd(&y_out[(size_t)t1i * n_embd + row], v1);
            atomicAdd(&y_out[(size_t)t2i * n_embd + row], v2);
            atomicAdd(&y_out[(size_t)t3i * n_embd + row], v3);
        }
        return;
    }

    for (int t0 = 0; t0 < cnt; t0 += 4) {
        int nt = min(4, cnt - t0);
        int tids[4];
        int sids[4];
        for (int m = 0; m < nt; ++m) {
            tids[m] = tok_sorted[off0 + t0 + m];
            sids[m] = slot_sorted[off0 + t0 + m];
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = lane4; b < nblk; b += 4) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            for (int m = 0; m < nt; ++m) {
                const q8_1_blk ab = act_q[((size_t)tids[m] * n_used + sids[m]) * nblk + b];
                acc[m] += dot_q4_soa_q8_1_blk(q4, dw, ab);
            }
        }
        for (int m = 0; m < nt; ++m) {
            float v = s_warp_sum4(acc[m]) * wt_sorted[off0 + t0 + m];
            if (lane4 == 0) atomicAdd(&y_out[(size_t)tids[m] * n_embd + row], v);
        }
    }
}

extern "C" int ink_q8_0_to_q4_0(hipStream_t s, const uint8_t* src, uint8_t* dst, size_t n_rows, int nblk) {
    size_t total = n_rows * (size_t)nblk;
    size_t grid = (total + 255) / 256;
    q8_0_to_q4_0_k<<<dim3((unsigned int)grid), dim3(256), 0, s>>>(src, dst, n_rows, nblk);
    return 0;
}

template <int NT>
__global__ void qkvrg_q4_nt_k(const uint8_t* __restrict__ wq,
                              const uint8_t* __restrict__ wk,
                              const uint8_t* __restrict__ wv,
                              const uint8_t* __restrict__ wr,
                              const uint8_t* __restrict__ wg,
                              const uint16_t* __restrict__ wsink,
                              const uint16_t* __restrict__ wkdyn,
                              const uint16_t* __restrict__ wvdyn,
                              const q8_1_blk* __restrict__ Xq0,
                              const q8_1_blk* __restrict__ Xq1,
                              const q8_1_blk* __restrict__ Xq2,
                              const float* __restrict__ X0,
                              const float* __restrict__ X1,
                              const float* __restrict__ X2,
                              float* __restrict__ Q,
                              float* __restrict__ K,
                              float* __restrict__ V,
                              float* __restrict__ R,
                              float* __restrict__ G,
                              float* __restrict__ S,
                              float* __restrict__ Kdyn,
                              float* __restrict__ Vdyn,
                              int k, int q_dim, int kv_dim,
                              int r_dim, int n_head, int dyn_dim) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int gid = blockIdx.x * 4 + warp;
    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    int o6 = o5 + n_head;
    int o7 = o6 + dyn_dim;
    int o8 = o7 + dyn_dim;
    if (gid >= o8) return;

    if (gid < o5) {
        const uint8_t* W;
        const q8_1_blk* Xq;
        float* Y;
        int row, n_out;
        if (gid < o1) { W = wq; Xq = Xq0; Y = Q; row = gid; n_out = q_dim; }
        else if (gid < o2) { W = wk; Xq = Xq1; Y = K; row = gid - o1; n_out = kv_dim; }
        else if (gid < o3) { W = wv; Xq = Xq2; Y = V; row = gid - o2; n_out = kv_dim; }
        else if (gid < o4) { W = wr; Xq = Xq0; Y = R; row = gid - o3; n_out = r_dim; }
        else { W = wg; Xq = Xq0; Y = G; row = gid - o4; n_out = q_dim; }

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
            float v = s_warp_sum32(acc[m]);
            if (lane == 0) Y[(size_t)m * n_out + row] = v;
        }
    } else {
        const uint16_t* W;
        const float* X;
        float* Y;
        int row, n_out;
        if (gid < o6) { W = wsink; X = X0; Y = S; row = gid - o5; n_out = n_head; }
        else if (gid < o7) { W = wkdyn; X = X1; Y = Kdyn; row = gid - o6; n_out = dyn_dim; }
        else { W = wvdyn; X = X2; Y = Vdyn; row = gid - o7; n_out = dyn_dim; }
        const uint16_t* wrow = W + (size_t)row * k;
        float acc[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
        for (int j = lane; j < k; j += 32) {
            float wv_f = s_bf16_f32(wrow[j]);
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                acc[m] = fmaf(wv_f, X[(size_t)m * k + j], acc[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            float v = s_warp_sum32(acc[m]);
            if (lane == 0) Y[(size_t)m * n_out + row] = v;
        }
    }
}

extern "C" int ink_qkvrg_gemm_sudot4(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim, int B, int is_q4) {
    if (B == 1 && is_q4) {
        return ink_qkvrg_q4_b1(s, wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, Xq0, Xq1, Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
    }
    int total_rows = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    if (is_q4 && B == 8) {
        qkvrg_q4_nt_k<8><<<dim3((total_rows + 3) / 4), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
        return 0;
    }
    if (is_q4 && B == 6) {
        qkvrg_q4_nt_k<6><<<dim3((total_rows + 3) / 4), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
        return 0;
    }
    if (is_q4 && B == 4) {
        qkvrg_q4_nt_k<4><<<dim3((total_rows + 3) / 4), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
        return 0;
    }
    qkvrg_gemm_sudot4_k<<<dim3((total_rows + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim, B, is_q4);
    return 0;
}

extern "C" int ink_wo_conv_res_q4_0_sudot4(hipStream_t s, const uint8_t* W, const void* Xq, const float* conv_w, float* hist, const float* add_src, float* block_dst, float* out, int n_embd, int q_dim) {
    wo_conv_res_q4_b1_8_k<<<dim3((n_embd + 15) / 16), dim3(128), 0, s>>>(W, (const q8_1_blk*)Xq, conv_w, hist, add_src, block_dst, out, n_embd, q_dim);
    return 0;
}

extern "C" int ink_moe_gu_nvfp4_b1_sudot4(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* itop, float* act_out, int n_ff, int k, int n_used, float beta, float lb) {
    moe_gu_nvfp4_b1_sudot4_k<<<dim3((n_ff + 15) / 16, n_used), dim3(128), 0, s>>>(W, scales, (const q8_1_blk*)Xq, itop, act_out, n_ff, k, beta, lb);
    return 0;
}

extern "C" int ink_moe_dn_conv_add_nvfp4_b1_sudot4(hipStream_t s, const uint8_t* W, const float* scales, const void* act_q, const int* itop, const float* wtop, const float* conv_w, float* hist, const float* ffn_inp, float* cur, int n_embd, int n_ff, int n_used) {
    moe_dn_conv_add_nvfp4_b1_sudot4_k<<<dim3((n_embd + 15) / 16), dim3(128), 0, s>>>(W, scales, (const q8_1_blk*)act_q, itop, wtop, conv_w, hist, ffn_inp, cur, n_embd, n_ff, n_used);
    return 0;
}

extern "C" int ink_moe_gu_q4_0_b1_sudot4(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* itop, float* act_out, int n_ff, int k, int n_used, float beta, float lb) {
    moe_gu_q4_0_b1_sudot4_k<<<dim3((n_ff + 15) / 16, n_used), dim3(128), 0, s>>>(W, scales, (const q8_1_blk*)Xq, itop, act_out, n_ff, k, beta, lb);
    return 0;
}

extern "C" int ink_moe_dn_conv_add_q4_0_b1_sudot4(hipStream_t s, const uint8_t* W, const float* scales, const void* act_q, const int* itop, const float* wtop, const float* conv_w, float* hist, const float* ffn_inp, float* cur, int n_embd, int n_ff, int n_used) {
    moe_dn_conv_add_q4_0_b1_sudot4_k<<<dim3((n_embd + 15) / 16), dim3(128), 0, s>>>(W, scales, (const q8_1_blk*)act_q, itop, wtop, conv_w, hist, ffn_inp, cur, n_embd, n_ff, n_used);
    return 0;
}

extern "C" int ink_moe_gu_q4_0_sudot4(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, float* act_out, int n_ff, int k, int n_used, float beta, float lb, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    moe_gu_q4_0_sudot4_k<<<dim3((n_ff + 15) / 16, gy), dim3(128), 0, s>>>(W, scales, (const q8_1_blk*)Xq, exp_counts, exp_offsets, tok_sorted, slot_sorted, act_out, n_ff, k, n_used, beta, lb);
    return 0;
}

extern "C" int ink_moe_dn_q4_0_sudot4(hipStream_t s, const uint8_t* W, const void* act_q, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, const float* wt_sorted, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    int tot = B * n_embd;
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    zero_buf_f32_k<<<dim3((tot + 255) / 256), dim3(256), 0, s>>>(y_out, tot);
    moe_dn_q4_0_sudot4_k<<<dim3((n_embd + 31) / 32, gy), dim3(128), 0, s>>>(W, (const q8_1_blk*)act_q, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, y_out, n_embd, n_ff, n_used);
    return 0;
}

extern "C" __global__ void gather_dspark_feats_k(const float* __restrict__ bsc_feats,
                                                 float* __restrict__ dst,
                                                 int tok_idx, int max_batch, int n_embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = blockIdx.y;
    if (i < n_embd) {
        dst[(size_t)layer * n_embd + i] = bsc_feats[((size_t)layer * max_batch + tok_idx) * n_embd + i];
    }
}

extern "C" __global__ void kv_conv_qkvr_store_batch_snap_k(float* __restrict__ Q,
                                                           const float* __restrict__ K_raw,
                                                           const float* __restrict__ V_raw,
                                                           float* __restrict__ R,
                                                           const float* __restrict__ k_conv_w,
                                                           const float* __restrict__ v_conv_w,
                                                           const float* __restrict__ k_dyn,
                                                           const float* __restrict__ v_dyn,
                                                           float* __restrict__ k_hist,
                                                           float* __restrict__ v_hist,
                                                           float* __restrict__ k_snap,
                                                           float* __restrict__ v_snap,
                                                           const float* __restrict__ q_clip,
                                                           const float* __restrict__ rel_clip,
                                                           __half* __restrict__ k_cache,
                                                           __half* __restrict__ v_cache,
                                                           const int* __restrict__ d_pos0,
                                                           int B, int is_swa, int n_head, int n_head_kv,
                                                           int dh, int d_rel, int max_ctx,
                                                           float qk_thresh, float rel_thresh,
                                                           float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (h < n_head) {
        const float* qch = q_clip + (size_t)h * dh;
        const float* rch = rel_clip + (size_t)h * d_rel;
        for (int t = 0; t < B; ++t) {
            int pos = pos0 + t;
            float tau = 1.0f;
            if (!is_swa) {
                float ratio = ((float)pos + 1.0f) / log_floor;
                if (ratio < 1.0f) ratio = 1.0f;
                tau = 1.0f + log_alpha * logf(ratio);
            }
            float* qh = Q + (size_t)t * q_dim + (size_t)h * dh;
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

            float* rh = R + (size_t)t * r_dim + (size_t)h * d_rel;
            float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
            float rss = s_warp_sum32(rv * rv);
            float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
            if (lane < d_rel) rh[lane] = rv * rfactor;
        }
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float kh0[4], kh1[4], kh2[4], vh0[4], vh1[4], vh2[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            kh0[p] = k_hist[c * 3 + 0]; kh1[p] = k_hist[c * 3 + 1]; kh2[p] = k_hist[c * 3 + 2];
            vh0[p] = v_hist[c * 3 + 0]; vh1[p] = v_hist[c * 3 + 1]; vh2[p] = v_hist[c * 3 + 2];
        }
        for (int t = 0; t < B; ++t) {
            int pos = pos0 + t;
            int slot = is_swa ? (pos % max_ctx) : pos;
            float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
            float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
            if (k_dyn) {
                const float* dk = k_dyn + (size_t)t * dyn_dim + (size_t)kh_idx * 4;
                const float* dv = v_dyn + (size_t)t * dyn_dim + (size_t)kh_idx * 4;
                kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
                vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
            }
            float kv_v[4], vv_v[4];
            float kss = 0.0f, vss = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int c = kh_idx * dh + lane + p * 32;
                float kx = K_raw[(size_t)t * kv_dim + c];
                const float* kwc = k_conv_w + (size_t)c * 4;
                float kacc = kwc[0] * kh0[p] + kwc[1] * kh1[p] + kwc[2] * kh2[p] + kwc[3] * kx;
                if (k_dyn) kacc += kd0 * kx + kd1 * kh2[p] + kd2 * kh1[p] + kd3 * kh0[p];
                kh0[p] = kh1[p]; kh1[p] = kh2[p]; kh2[p] = kx;
                if (k_snap) {
                    float* ks = k_snap + ((size_t)t * kv_dim + c) * 3;
                    ks[0] = kh0[p]; ks[1] = kh1[p]; ks[2] = kh2[p];
                }
                float kout = (kx + kacc) * sc;
                kv_v[p] = kout;
                kss += kout * kout;

                float vx = V_raw[(size_t)t * kv_dim + c];
                const float* vwc = v_conv_w + (size_t)c * 4;
                float vacc = vwc[0] * vh0[p] + vwc[1] * vh1[p] + vwc[2] * vh2[p] + vwc[3] * vx;
                if (v_dyn) vacc += vd0 * vx + vd1 * vh2[p] + vd2 * vh1[p] + vd3 * vh0[p];
                vh0[p] = vh1[p]; vh1[p] = vh2[p]; vh2[p] = vx;
                if (v_snap) {
                    float* vs = v_snap + ((size_t)t * kv_dim + c) * 3;
                    vs[0] = vh0[p]; vs[1] = vh1[p]; vs[2] = vh2[p];
                }
                float vout = (vx + vacc) * sc;
                vv_v[p] = vout;
                vss += vout * vout;
            }
            kss = s_warp_sum32(kss);
            vss = s_warp_sum32(vss);
            float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
            float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
            __half* kc_slot = k_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
            __half* vc_slot = v_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int d = lane + p * 32;
                kc_slot[d] = __float2half(kv_v[p] * kfac);
                vc_slot[d] = __float2half(vv_v[p] * vinv);
            }
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            k_hist[c * 3 + 0] = kh0[p]; k_hist[c * 3 + 1] = kh1[p]; k_hist[c * 3 + 2] = kh2[p];
            v_hist[c * 3 + 0] = vh0[p]; v_hist[c * 3 + 1] = vh1[p]; v_hist[c * 3 + 2] = vh2[p];
        }
    }
}

extern "C" __global__ void shortconv_res_batch_snap_k(const float* __restrict__ raw,
                                                      const float* __restrict__ conv_w,
                                                      float* __restrict__ hist,
                                                      float* __restrict__ snap,
                                                      const float* __restrict__ add_src,
                                                      float* __restrict__ block_dst,
                                                      float* __restrict__ out,
                                                      int n_embd, int B) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_embd) return;
    float* hc = hist + (size_t)c * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = conv_w + (size_t)c * 4;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    for (int t = 0; t < B; ++t) {
        size_t idx = (size_t)t * n_embd + c;
        float x = raw[idx];
        float conv_out = (x + w0 * h0 + w1 * h1 + w2 * h2 + w3 * x) * rsqrtf(5.0f);
        h0 = h1; h1 = h2; h2 = x;
        if (snap) {
            float* sn = snap + ((size_t)t * n_embd + c) * 3;
            sn[0] = h0; sn[1] = h1; sn[2] = h2;
        }
        if (block_dst) {
            block_dst[idx] = add_src[idx];
            out[idx] = conv_out;
        } else {
            out[idx] = add_src[idx] + conv_out;
        }
    }
    hc[0] = h0; hc[1] = h1; hc[2] = h2;
}

extern "C" int ink_zero_f32(hipStream_t s, float* ptr, int n) {
    zero_buf_f32_k<<<dim3((n + 255) / 256), dim3(256), 0, s>>>(ptr, n);
    return 0;
}

extern "C" int ink_gather_dspark_feats(hipStream_t s, const float* bsc_feats, float* dst, int tok_idx, int max_batch, int n_embd) {
    gather_dspark_feats_k<<<dim3((n_embd + 255) / 256, 4), dim3(256), 0, s>>>(bsc_feats, dst, tok_idx, max_batch, n_embd);
    return 0;
}

extern "C" __global__ void kv_conv_qkvr_store_prefill_k(float* __restrict__ Q,
                                                        const float* __restrict__ K_raw,
                                                        const float* __restrict__ V_raw,
                                                        float* __restrict__ R,
                                                        const float* __restrict__ k_conv_w,
                                                        const float* __restrict__ v_conv_w,
                                                        const float* __restrict__ k_dyn,
                                                        const float* __restrict__ v_dyn,
                                                        const float* __restrict__ k_hist,
                                                        const float* __restrict__ v_hist,
                                                        const float* __restrict__ q_clip,
                                                        const float* __restrict__ rel_clip,
                                                        __half* __restrict__ k_cache,
                                                        __half* __restrict__ v_cache,
                                                        const int* __restrict__ d_pos0,
                                                        int is_swa, int n_head, int n_head_kv,
                                                        int dh, int d_rel, int max_ctx,
                                                        float qk_thresh, float rel_thresh,
                                                        float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int lane = threadIdx.x;
    int pos = (*d_pos0) + t;
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
        const float* qch = q_clip + (size_t)h * dh;
        float* qh = Q + (size_t)t * q_dim + (size_t)h * dh;
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

        const float* rch = rel_clip + (size_t)h * d_rel;
        float* rh = R + (size_t)t * r_dim + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = s_warp_sum32(rv * rv);
        float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
        if (lane < d_rel) rh[lane] = rv * rfactor;
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        int slot = is_swa ? (pos % max_ctx) : pos;
        float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
        float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
        if (k_dyn) {
            const float* dk = k_dyn + (size_t)t * dyn_dim + (size_t)kh_idx * 4;
            const float* dv = v_dyn + (size_t)t * dyn_dim + (size_t)kh_idx * 4;
            kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
            vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
        }
        float kv_v[4], vv_v[4];
        float kss = 0.0f, vss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            float kx = K_raw[(size_t)t * kv_dim + c];
            float kh2 = (t >= 1) ? K_raw[(size_t)(t - 1) * kv_dim + c] : k_hist[c * 3 + 2];
            float kh1 = (t >= 2) ? K_raw[(size_t)(t - 2) * kv_dim + c] : k_hist[c * 3 + 1 + t];
            float kh0 = (t >= 3) ? K_raw[(size_t)(t - 3) * kv_dim + c] : k_hist[c * 3 + t];
            const float* kwc = k_conv_w + (size_t)c * 4;
            float kacc = kwc[0] * kh0 + kwc[1] * kh1 + kwc[2] * kh2 + kwc[3] * kx;
            if (k_dyn) kacc += kd0 * kx + kd1 * kh2 + kd2 * kh1 + kd3 * kh0;
            float kout = (kx + kacc) * sc;
            kv_v[p] = kout;
            kss += kout * kout;

            float vx = V_raw[(size_t)t * kv_dim + c];
            float vh2 = (t >= 1) ? V_raw[(size_t)(t - 1) * kv_dim + c] : v_hist[c * 3 + 2];
            float vh1 = (t >= 2) ? V_raw[(size_t)(t - 2) * kv_dim + c] : v_hist[c * 3 + 1 + t];
            float vh0 = (t >= 3) ? V_raw[(size_t)(t - 3) * kv_dim + c] : v_hist[c * 3 + t];
            const float* vwc = v_conv_w + (size_t)c * 4;
            float vacc = vwc[0] * vh0 + vwc[1] * vh1 + vwc[2] * vh2 + vwc[3] * vx;
            if (v_dyn) vacc += vd0 * vx + vd1 * vh2 + vd2 * vh1 + vd3 * vh0;
            float vout = (vx + vacc) * sc;
            vv_v[p] = vout;
            vss += vout * vout;
        }
        kss = s_warp_sum32(kss);
        vss = s_warp_sum32(vss);
        float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
        float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
        __half* kc_slot = k_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
        __half* vc_slot = v_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            kc_slot[d] = __float2half(kv_v[p] * kfac);
            vc_slot[d] = __float2half(vv_v[p] * vinv);
        }
    }
}

extern "C" __global__ void hist_update_3_k(const float* __restrict__ raw, float* __restrict__ hist, int dim, int B) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    float* hc = hist + (size_t)c * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    hc[0] = (B >= 3) ? raw[(size_t)(B - 3) * dim + c] : ((B == 2) ? h2 : h1);
    hc[1] = (B >= 2) ? raw[(size_t)(B - 2) * dim + c] : h2;
    hc[2] = raw[(size_t)(B - 1) * dim + c];
}

extern "C" __global__ void shortconv_res_prefill_k(const float* __restrict__ raw,
                                                   const float* __restrict__ conv_w,
                                                   const float* __restrict__ hist,
                                                   const float* __restrict__ add_src,
                                                   float* __restrict__ block_dst,
                                                   float* __restrict__ out,
                                                   int n_embd) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= n_embd) return;
    size_t idx = (size_t)t * n_embd + c;
    float x = raw[idx];
    const float* hc = hist + (size_t)c * 3;
    float h2 = (t >= 1) ? raw[(size_t)(t - 1) * n_embd + c] : hc[2];
    float h1 = (t >= 2) ? raw[(size_t)(t - 2) * n_embd + c] : hc[1 + t];
    float h0 = (t >= 3) ? raw[(size_t)(t - 3) * n_embd + c] : hc[t];
    const float* wc = conv_w + (size_t)c * 4;
    float conv_out = (x + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * x) * rsqrtf(5.0f);
    if (block_dst) {
        block_dst[idx] = add_src[idx];
        out[idx] = conv_out;
    } else {
        out[idx] = add_src[idx] + conv_out;
    }
}

extern "C" int ink_kv_conv_qkvr_store_batch_snap(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist, float* v_hist, float* k_snap, float* v_snap, const float* q_clip, const float* rel_clip, __half* kc, __half* vc, const int* d_pos0, int B, int is_swa, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    if (B > 8 && k_snap == nullptr && v_snap == nullptr) {
        kv_conv_qkvr_store_prefill_k<<<dim3(n_head + n_head_kv, B), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, q_clip, rel_clip, kc, vc, d_pos0, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
        int kv_dim = n_head_kv * dh;
        hist_update_3_k<<<dim3((kv_dim + 255) / 256), dim3(256), 0, s>>>(K_raw, k_hist, kv_dim, B);
        hist_update_3_k<<<dim3((kv_dim + 255) / 256), dim3(256), 0, s>>>(V_raw, v_hist, kv_dim, B);
        return 0;
    }
    kv_conv_qkvr_store_batch_snap_k<<<dim3(n_head + n_head_kv), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, k_snap, v_snap, q_clip, rel_clip, kc, vc, d_pos0, B, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}

extern "C" int ink_shortconv_res_batch_snap(hipStream_t s, const float* raw, const float* conv_w, float* hist, float* snap, const float* add_src, float* block_dst, float* out, int n_embd, int B) {
    if (B > 8 && snap == nullptr) {
        shortconv_res_prefill_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(raw, conv_w, hist, add_src, block_dst, out, n_embd);
        hist_update_3_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(raw, hist, n_embd, B);
        return 0;
    }
    shortconv_res_batch_snap_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(raw, conv_w, hist, snap, add_src, block_dst, out, n_embd, B);
    return 0;
}

extern "C" __global__ void nvfp4_to_q4_0_k(const uint8_t* __restrict__ src,
                                           uint8_t* __restrict__ dst,
                                           size_t n_rows, int nblk_nv) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_rows * (size_t)nblk_nv;
    if (idx >= total) return;
    size_t row = idx / (size_t)nblk_nv;
    int b_nv = (int)(idx % (size_t)nblk_nv);
    int nblk_q4 = nblk_nv * 2;
    size_t rbytes = q4_soa_row_bytes(nblk_q4);
    uint8_t* drow = dst + row * rbytes;
    const uint8_t* sp = src + idx * 36;
    uint32_t ds;
    memcpy(&ds, sp, 4);
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(sp + 4);
    for (int half_idx = 0; half_idx < 2; ++half_idx) {
        float vals[32];
        float amax = 0.0f;
        for (int sb_l = 0; sb_l < 2; ++sb_l) {
            int sb = half_idx * 2 + sb_l;
            float d = s_ue4m3_to_fp32((uint8_t)((ds >> (sb * 8)) & 0xFF));
            uint32_t q0 = q32[sb * 2];
            uint32_t q1 = q32[sb * 2 + 1];
            int base = sb_l * 16;
            for (int i = 0; i < 4; ++i) {
                float v0 = (float)s_fp4_i8(q0 >> (i * 8)) * d;
                float v1 = (float)s_fp4_i8(q1 >> (i * 8)) * d;
                float v2 = (float)s_fp4_i8(q0 >> (i * 8 + 4)) * d;
                float v3 = (float)s_fp4_i8(q1 >> (i * 8 + 4)) * d;
                vals[base + i] = v0;
                vals[base + 4 + i] = v1;
                vals[base + 8 + i] = v2;
                vals[base + 12 + i] = v3;
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
            }
        }
        float d_out = amax / 7.5f;
        float inv = (d_out > 0.0f) ? (1.0f / d_out) : 0.0f;
        int b_q4 = b_nv * 2 + half_idx;
        reinterpret_cast<__half*>(drow + (size_t)nblk_q4 * 16)[b_q4] = __float2half(d_out);
        uint8_t* qo = drow + (size_t)b_q4 * 16;
        for (int j = 0; j < 16; ++j) {
            int q0 = max(0, min(15, (int)floorf(vals[j] * inv + 8.5f)));
            int q1 = max(0, min(15, (int)floorf(vals[j + 16] * inv + 8.5f)));
            qo[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

extern "C" int ink_nvfp4_to_q4_0(hipStream_t s, const uint8_t* src, uint8_t* dst, size_t n_rows, int nblk_nv) {
    size_t total = n_rows * (size_t)nblk_nv;
    size_t grid = (total + 255) / 256;
    nvfp4_to_q4_0_k<<<dim3((unsigned int)grid), dim3(256), 0, s>>>(src, dst, n_rows, nblk_nv);
    return 0;
}

__device__ __forceinline__ void store_q8_1_from_warp(float v, int lane, q8_1_blk* dst, int8_t* sm) {
    float amax = s_warp_max32(fabsf(v));
    amax = __shfl(amax, 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    int32_t qsum = s_warp_sum32_i(qi);
    sm[lane] = (int8_t)qi;
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if (lane < 8) {
        dst->q[lane] = pack4_i8(sm[lane * 4], sm[lane * 4 + 1], sm[lane * 4 + 2], sm[lane * 4 + 3]);
    }
}

extern "C" __global__ void inkling_attn_decode_gated_q8_k(const float* __restrict__ Q,
                                                          const float* __restrict__ R,
                                                          const float* __restrict__ sinks,
                                                          const float* __restrict__ rel_bias,
                                                          const __half* __restrict__ k_cache,
                                                          const __half* __restrict__ v_cache,
                                                          const float* __restrict__ gate,
                                                          const int* __restrict__ d_pos,
                                                          q8_1_blk* __restrict__ out_q8,
                                                          int n_head, int n_head_kv,
                                                          int dh, int d_rel,
                                                          int rel_extent, int swa_window,
                                                          int is_swa, int max_ctx) {
    int h = blockIdx.x;
    if (h >= n_head) return;
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;

    __shared__ float qs[128];
    __shared__ float rs[16];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[(size_t)h * dh + d] * q_scale;
    }
    if (threadIdx.x < d_rel) {
        rs[threadIdx.x] = R[(size_t)h * d_rel + threadIdx.x];
    }
    __syncthreads();

    int pos = *d_pos;
    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[h];

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

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
        const float* gh = gate + (size_t)h * dh;
        float v0 = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        float v1 = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        float v2 = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        float v3 = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
        __shared__ int8_t smq[32];
        store_q8_1_from_warp(v0, lane, out_q8 + h * 4 + 0, smq);
        store_q8_1_from_warp(v1, lane, out_q8 + h * 4 + 1, smq);
        store_q8_1_from_warp(v2, lane, out_q8 + h * 4 + 2, smq);
        store_q8_1_from_warp(v3, lane, out_q8 + h * 4 + 3, smq);
    }
}

extern "C" __global__ void attn_decode_gqa_splitkv_k(const float* __restrict__ Q,
                                                     const float* __restrict__ R,
                                                     const float* __restrict__ sinks,
                                                     const float* __restrict__ rel_bias,
                                                     const __half* __restrict__ k_cache,
                                                     const __half* __restrict__ v_cache,
                                                     const int* __restrict__ d_pos,
                                                     float* __restrict__ split_buf,
                                                     int n_head, int n_head_kv,
                                                     int dh, int d_rel,
                                                     int rel_extent, int swa_window,
                                                     int is_swa, int max_ctx, int n_splits) {
    int kv_h = blockIdx.x;
    int sp = blockIdx.y;
    int h0 = kv_h * 3;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int pos = *d_pos;
    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    int total_t = t_end - t_start;
    int chunk = (total_t + n_splits - 1) / n_splits;
    int ts = t_start + sp * chunk;
    int te = min(t_end, ts + chunk);

    if (ts >= te) {
        if (wid == 0 && lane < 3) {
            int h = h0 + lane;
            float* dst = split_buf + ((size_t)h * n_splits + sp) * 130;
            dst[0] = -1e30f;
            dst[1] = 0.0f;
        }
        return;
    }

    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q_scale = rsqrtf((float)dh);
    float q0a[3], q0b[3], q1a[3], q1b[3];
    float rv[3], sk[3];
    float m[3], l[3], o0[3], o1[3], o2[3], o3[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h0 + g;
        const float* qh = Q + (size_t)h * dh;
        q0a[g] = qh[p0] * q_scale;
        q0b[g] = qh[p0 + 1] * q_scale;
        q1a[g] = qh[p1] * q_scale;
        q1b[g] = qh[p1 + 1] * q_scale;
        rv[g] = (lane < d_rel) ? R[(size_t)h * d_rel + lane] : 0.0f;
        sk[g] = sinks[h];
        m[g] = (sp == 0 && wid == 0) ? 0.0f : -1e30f;
        l[g] = (sp == 0 && wid == 0) ? 1.0f : 0.0f;
        o0[g] = 0.0f; o1[g] = 0.0f; o2[g] = 0.0f; o3[g] = 0.0f;
    }

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    for (int t = ts + wid; t < te; t += 8) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        int dist = pos - t;
        float rb_l = (dist < rel_extent && lane < d_rel) ? rel_bias[(size_t)dist * d_rel + lane] : 0.0f;
        bool has_rb = (dist < rel_extent);

#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = q0a[g] * k0.x + q0b[g] * k0.y + q1a[g] * k1.x + q1b[g] * k1.y;
            float score = __shfl(s_warp_sum32(dot), 0, 32);
            if (has_rb) {
                score += __shfl(s_warp_sum32(rb_l * rv[g]), 0, 32);
            }
            score -= sk[g];
            float m_new = fmaxf(m[g], score);
            float alpha = (m[g] > -1e20f) ? expf(m[g] - m_new) : 0.0f;
            float p = expf(score - m_new);
            l[g] = l[g] * alpha + p;
            m[g] = m_new;
            o0[g] = fmaf(p, v0.x, o0[g] * alpha);
            o1[g] = fmaf(p, v0.y, o1[g] * alpha);
            o2[g] = fmaf(p, v1.x, o2[g] * alpha);
            o3[g] = fmaf(p, v1.y, o3[g] * alpha);
        }
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[wid] = m[g]; sm_l[wid] = l[g]; }
        sm_o[wid][p0] = o0[g]; sm_o[wid][p0 + 1] = o1[g];
        sm_o[wid][p1] = o2[g]; sm_o[wid][p1 + 1] = o3[g];
        __syncthreads();
        if (wid == 0) {
            float gm = sm_m[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) gm = fmaxf(gm, sm_m[w]);
            float gl = 0.0f;
            float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                float sc = (sm_m[w] > -1e20f) ? expf(sm_m[w] - gm) : 0.0f;
                gl += sm_l[w] * sc;
                go0 += sm_o[w][d0] * sc;
                go1 += sm_o[w][d1] * sc;
                go2 += sm_o[w][d2] * sc;
                go3 += sm_o[w][d3] * sc;
            }
            int h = h0 + g;
            float* dst = split_buf + ((size_t)h * n_splits + sp) * 130;
            if (lane == 0) { dst[0] = gm; dst[1] = gl; }
            dst[2 + d0] = go0;
            dst[2 + d1] = go1;
            dst[2 + d2] = go2;
            dst[2 + d3] = go3;
        }
        __syncthreads();
    }
}

extern "C" __global__ void attn_decode_splitkv_reduce_q8_k(const float* __restrict__ split_buf,
                                                           const float* __restrict__ gate,
                                                           q8_1_blk* __restrict__ out_q8,
                                                           int n_splits) {
    int h = blockIdx.x;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = wid * 32 + lane;
    const float* base = split_buf + (size_t)h * n_splits * 130;

    float gm = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        gm = fmaxf(gm, base[(size_t)sp * 130]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        float ms = base[(size_t)sp * 130];
        if (ms > -1e20f) {
            float sc = expf(ms - gm);
            gl += base[(size_t)sp * 130 + 1] * sc;
            go += base[(size_t)sp * 130 + 2 + d] * sc;
        }
    }
    float v = (go / gl) / (1.0f + expf(-gate[(size_t)h * 128 + d]));
    __shared__ int8_t smq[4][32];
    store_q8_1_from_warp(v, lane, out_q8 + h * 4 + wid, smq[wid]);
}

__device__ float g_b1_split_buf[32 * 16 * 130];

extern "C" int ink_attn_decode_gated_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx) {
    if (dh == 128 && n_head == n_head_kv * 3 && n_head <= 32) {
        int n_splits = 8;
        float* sbuf = nullptr;
        (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_b1_split_buf));
        attn_decode_gqa_splitkv_k<<<dim3(n_head_kv, n_splits), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, d_pos, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, n_splits);
        attn_decode_splitkv_reduce_q8_k<<<dim3(n_head), dim3(128), 0, s>>>(sbuf, gate, (q8_1_blk*)out_q8, n_splits);
        return 0;
    }
    inkling_attn_decode_gated_q8_k<<<dim3(n_head), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" __global__ void moe_gu_q4_0_b1_to_q8_k(const uint8_t* __restrict__ W,
                                                  const float* __restrict__ scales,
                                                  const q8_1_blk* __restrict__ Xq,
                                                  const int* __restrict__ itop,
                                                  q8_1_blk* __restrict__ act_q,
                                                  int n_ff, int k, float beta, float lb) {
    int slot = blockIdx.y;
    int exp_id = itop[slot];
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;

    __shared__ float sm_v[32];
    if (row < n_ff) {
        int nblk = k >> 5;
        size_t row_bytes = q4_soa_row_bytes(nblk);
        size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
        const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + row) * row_bytes;
        const uint4* qg_row = reinterpret_cast<const uint4*>(wg);
        const __half* dg_row = reinterpret_cast<const __half*>(wg + (size_t)nblk * 16);
        const uint4* qu_row = reinterpret_cast<const uint4*>(wu);
        const __half* du_row = reinterpret_cast<const __half*>(wu + (size_t)nblk * 16);

        float ag = 0.0f, au = 0.0f;
        for (int b = lane8; b < nblk; b += 8) {
            const q8_1_blk xb = Xq[b];
            ag += dot_q4_soa_q8_1_blk(qg_row[b], __half2float(dg_row[b]), xb);
            au += dot_q4_soa_q8_1_blk(qu_row[b], __half2float(du_row[b]), xb);
        }
        float sc = scales ? scales[exp_id] : 1.0f;
        float g = s_warp_sum8(ag) * sc;
        float u = s_warp_sum8(au) * sc;
        if (lane8 == 0) {
            float sg = 1.0f / (1.0f + expf(-g));
            float act = beta * tanhf(g / beta) * sg;
            float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
            sm_v[grp] = act * up;
        }
    } else if (lane8 == 0) {
        sm_v[grp] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        __shared__ int8_t sm_q[32];
        int qblk_per_slot = n_ff >> 5;
        store_q8_1_from_warp(sm_v[threadIdx.x], threadIdx.x, act_q + (size_t)slot * qblk_per_slot + blockIdx.x, sm_q);
    }
}

extern "C" int ink_moe_gu_q4_0_b1_to_q8(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* itop, void* act_q, int n_ff, int k, int n_used, float beta, float lb) {
    moe_gu_q4_0_b1_to_q8_k<<<dim3((n_ff + 31) / 32, n_used), dim3(256), 0, s>>>(W, scales, (const q8_1_blk*)Xq, itop, (q8_1_blk*)act_q, n_ff, k, beta, lb);
    return 0;
}

extern "C" __global__ void qkvrg_q4_b1_k(const uint8_t* __restrict__ wq,
                                         const uint8_t* __restrict__ wk,
                                         const uint8_t* __restrict__ wv,
                                         const uint8_t* __restrict__ wr,
                                         const uint8_t* __restrict__ wg,
                                         const uint16_t* __restrict__ wsink,
                                         const uint16_t* __restrict__ wkdyn,
                                         const uint16_t* __restrict__ wvdyn,
                                         const q8_1_blk* __restrict__ Xq0,
                                         const q8_1_blk* __restrict__ Xq1,
                                         const q8_1_blk* __restrict__ Xq2,
                                         const float* __restrict__ X0,
                                         const float* __restrict__ X1,
                                         const float* __restrict__ X2,
                                         float* __restrict__ Q,
                                         float* __restrict__ K,
                                         float* __restrict__ V,
                                         float* __restrict__ R,
                                         float* __restrict__ G,
                                         float* __restrict__ S,
                                         float* __restrict__ Kdyn,
                                         float* __restrict__ Vdyn,
                                         int k, int q_dim, int kv_dim,
                                         int r_dim, int n_head, int dyn_dim) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int gid = blockIdx.x * 16 + grp;

    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    int o6 = o5 + n_head;
    int o7 = o6 + dyn_dim;
    int o8 = o7 + dyn_dim;
    if (gid >= o8) return;

    if (gid < o5) {
        const uint8_t* W;
        const q8_1_blk* Xq;
        float* Y;
        int row;
        if (gid < o1) { W = wq; Xq = Xq0; Y = Q; row = gid; }
        else if (gid < o2) { W = wk; Xq = Xq1; Y = K; row = gid - o1; }
        else if (gid < o3) { W = wv; Xq = Xq2; Y = V; row = gid - o2; }
        else if (gid < o4) { W = wr; Xq = Xq0; Y = R; row = gid - o3; }
        else { W = wg; Xq = Xq0; Y = G; row = gid - o4; }

        int nblk = k >> 5;
        size_t rbytes = q4_soa_row_bytes(nblk);
        const uint8_t* wrow = W + (size_t)row * rbytes;
        const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
        const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
        float acc = 0.0f;
        for (int b = lane8; b < nblk; b += 8) {
            acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), Xq[b]);
        }
        acc = s_warp_sum8(acc);
        if (lane8 == 0) Y[row] = acc;
    } else {
        const uint16_t* W;
        const float* X;
        float* Y;
        int row;
        if (gid < o6) { W = wsink; X = X0; Y = S; row = gid - o5; }
        else if (gid < o7) { W = wkdyn; X = X1; Y = Kdyn; row = gid - o6; }
        else { W = wvdyn; X = X2; Y = Vdyn; row = gid - o7; }
        const uint16_t* wrow = W + (size_t)row * k;
        float acc = 0.0f;
        for (int j = lane8; j < k; j += 8) {
            acc = fmaf(s_bf16_f32(wrow[j]), X[j], acc);
        }
        acc = s_warp_sum8(acc);
        if (lane8 == 0) Y[row] = acc;
    }
}

extern "C" __global__ void wo_conv_res_q4_b1_8_k(const uint8_t* __restrict__ W,
                                                 const q8_1_blk* __restrict__ Xq,
                                                 const float* __restrict__ conv_w,
                                                 float* __restrict__ hist,
                                                 const float* __restrict__ add_src,
                                                 float* __restrict__ block_dst,
                                                 float* __restrict__ out,
                                                 int n_embd, int q_dim) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;
    int nblk = q_dim >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), Xq[b]);
    }
    acc = s_warp_sum8(acc);
    if (lane8 == 0) {
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

extern "C" __global__ void gemv_q4_b1_8_k(const uint8_t* __restrict__ W,
                                          const q8_1_blk* __restrict__ Xq,
                                          float* __restrict__ Y,
                                          int n, int k) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n) return;
    int nblk = k >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), Xq[b]);
    }
    acc = s_warp_sum8(acc);
    if (lane8 == 0) Y[row] = acc;
}

extern "C" __global__ void moe_dn_conv_add_q4_b1_4_k(const uint8_t* __restrict__ W,
                                                     const float* __restrict__ scales,
                                                     const q8_1_blk* __restrict__ act_q,
                                                     const int* __restrict__ itop,
                                                     const float* __restrict__ wtop,
                                                     const float* __restrict__ conv_w,
                                                     float* __restrict__ hist,
                                                     const float* __restrict__ ffn_inp,
                                                     float* __restrict__ cur,
                                                     int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 2;
    int lane4 = threadIdx.x & 3;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;

    int nblk = n_ff >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * row_bytes;

    float total = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        int exp_id = itop[s];
        float wt = wtop[s] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
        const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
        const q8_1_blk* aq = act_q + (size_t)s * nblk;
        float acc = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), aq[b]);
        }
        total += acc * wt;
    }
    float v = s_warp_sum4(total);
    if (lane4 == 0) {
        float* hc = hist + (size_t)row * 3;
        const float* wc = conv_w + (size_t)row * 4;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        float conv_out = (v + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * v) * rsqrtf(5.0f);
        hc[0] = h1; hc[1] = h2; hc[2] = v;
        cur[row] = ffn_inp[row] + conv_out;
    }
}

extern "C" int ink_qkvrg_q4_b1(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim) {
    int total_rows = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    qkvrg_q4_b1_k<<<dim3((total_rows + 15) / 16), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
    return 0;
}

extern "C" int ink_gemv_q4_b1(hipStream_t s, const uint8_t* W, const void* Xq, float* Y, int n, int k) {
    gemv_q4_b1_8_k<<<dim3((n + 15) / 16), dim3(128), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
    return 0;
}

extern "C" __global__ void inkling_attn_multi_gated_q8_k(const float* __restrict__ Q,
                                                         const float* __restrict__ R,
                                                         const float* __restrict__ sinks,
                                                         const float* __restrict__ rel_bias,
                                                         const __half* __restrict__ k_cache_pool,
                                                         const __half* __restrict__ v_cache_pool,
                                                         const float* __restrict__ gate,
                                                         const int* __restrict__ d_seq_ids,
                                                         const int* __restrict__ d_positions,
                                                         q8_1_blk* __restrict__ out_q8,
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
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        float v0 = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        float v1 = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        float v2 = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        float v3 = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
        __shared__ int8_t smq[32];
        size_t qbase = ((size_t)bt * n_head + h) * 4;
        store_q8_1_from_warp(v0, lane, out_q8 + qbase + 0, smq);
        store_q8_1_from_warp(v1, lane, out_q8 + qbase + 1, smq);
        store_q8_1_from_warp(v2, lane, out_q8 + qbase + 2, smq);
        store_q8_1_from_warp(v3, lane, out_q8 + qbase + 3, smq);
    }
}

extern "C" __global__ void inkling_attn_multi_gqa3_q8_k(const float* __restrict__ Q,
                                                        const float* __restrict__ R,
                                                        const float* __restrict__ sinks,
                                                        const float* __restrict__ rel_bias,
                                                        const __half* __restrict__ k_cache_pool,
                                                        const __half* __restrict__ v_cache_pool,
                                                        const float* __restrict__ gate,
                                                        const int* __restrict__ d_seq_ids,
                                                        const int* __restrict__ d_positions,
                                                        q8_1_blk* __restrict__ out_q8,
                                                        int n_head, int n_head_kv,
                                                        int dh, int d_rel,
                                                        int rel_extent, int swa_window,
                                                        int is_swa, int max_ctx) {
    int kv_h = blockIdx.x;
    int bt = blockIdx.y;
    int seq = d_seq_ids[bt];
    int pos = d_positions[bt];
    int h0 = kv_h * 3;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q_scale = rsqrtf((float)dh);
    float q0a[3], q0b[3], q1a[3], q1b[3];
    float rv[3], sk[3];
    float m[3], l[3], o0[3], o1[3], o2[3], o3[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h0 + g;
        const float* qh = Q + (size_t)bt * q_dim + (size_t)h * dh;
        q0a[g] = qh[p0] * q_scale;
        q0b[g] = qh[p0 + 1] * q_scale;
        q1a[g] = qh[p1] * q_scale;
        q1b[g] = qh[p1 + 1] * q_scale;
        rv[g] = (lane < d_rel) ? R[(size_t)bt * r_dim + (size_t)h * d_rel + lane] : 0.0f;
        sk[g] = sinks[(size_t)bt * n_head + h];
        m[g] = (wid == 0) ? 0.0f : -1e30f;
        l[g] = (wid == 0) ? 1.0f : 0.0f;
        o0[g] = 0.0f; o1[g] = 0.0f; o2[g] = 0.0f; o3[g] = 0.0f;
    }

    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
    const __half* kc_h = k_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;

    for (int t = t_start + wid; t < t_end; t += 8) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        int dist = pos - t;
        bool has_rb = (dist < rel_extent);
        float rb_l = (has_rb && lane < d_rel) ? rel_bias[(size_t)dist * d_rel + lane] : 0.0f;

#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = q0a[g] * k0.x + q0b[g] * k0.y + q1a[g] * k1.x + q1b[g] * k1.y;
            float score = __shfl(s_warp_sum32(dot), 0, 32);
            if (has_rb) score += __shfl(s_warp_sum32(rb_l * rv[g]), 0, 32);
            score -= sk[g];
            float m_new = fmaxf(m[g], score);
            float alpha = (m[g] > -1e20f) ? expf(m[g] - m_new) : 0.0f;
            float p = expf(score - m_new);
            l[g] = l[g] * alpha + p;
            m[g] = m_new;
            o0[g] = fmaf(p, v0.x, o0[g] * alpha);
            o1[g] = fmaf(p, v0.y, o1[g] * alpha);
            o2[g] = fmaf(p, v1.x, o2[g] * alpha);
            o3[g] = fmaf(p, v1.y, o3[g] * alpha);
        }
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    __shared__ int8_t smq[32];
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[wid] = m[g]; sm_l[wid] = l[g]; }
        sm_o[wid][p0] = o0[g]; sm_o[wid][p0 + 1] = o1[g];
        sm_o[wid][p1] = o2[g]; sm_o[wid][p1 + 1] = o3[g];
        __syncthreads();
        if (wid == 0) {
            float gm = sm_m[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) gm = fmaxf(gm, sm_m[w]);
            float gl = 0.0f;
            float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                float sc = (sm_m[w] > -1e20f) ? expf(sm_m[w] - gm) : 0.0f;
                gl += sm_l[w] * sc;
                go0 += sm_o[w][d0] * sc;
                go1 += sm_o[w][d1] * sc;
                go2 += sm_o[w][d2] * sc;
                go3 += sm_o[w][d3] * sc;
            }
            float inv_l = 1.0f / gl;
            int h = h0 + g;
            const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
            float v0 = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
            float v1 = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
            float v2 = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
            float v3 = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
            size_t qbase = ((size_t)bt * n_head + h) * 4;
            store_q8_1_from_warp(v0, lane, out_q8 + qbase + 0, smq);
            store_q8_1_from_warp(v1, lane, out_q8 + qbase + 1, smq);
            store_q8_1_from_warp(v2, lane, out_q8 + qbase + 2, smq);
            store_q8_1_from_warp(v3, lane, out_q8 + qbase + 3, smq);
        }
        __syncthreads();
    }
}

extern "C" int ink_attn_multi_gated_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc_pool, const __half* vc_pool, const float* gate, const int* d_seq_ids, const int* d_positions, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    if (dh == 128 && n_head == n_head_kv * 3 && B >= 8) {
        inkling_attn_multi_gqa3_q8_k<<<dim3(n_head_kv, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, gate, d_seq_ids, d_positions, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
        return 0;
    }
    inkling_attn_multi_gated_q8_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, gate, d_seq_ids, d_positions, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" __global__ void moe_gu_q4_0_sudot4_to_q8_k(const uint8_t* __restrict__ W,
                                                      const float* __restrict__ scales,
                                                      const q8_1_blk* __restrict__ Xq,
                                                      const int* __restrict__ exp_counts,
                                                      const int* __restrict__ exp_offsets,
                                                      const int* __restrict__ tok_sorted,
                                                      const int* __restrict__ slot_sorted,
                                                      q8_1_blk* __restrict__ act_q,
                                                      int n_ff, int k, int n_used,
                                                      float beta, float lb) {
    int exp_id = exp_counts[128 + blockIdx.y];
    if (exp_id < 0) return;
    int cnt = exp_counts[exp_id];
    if (cnt == 0) return;
    int off0 = exp_offsets[exp_id];

    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int nblk = k >> 5;
    int qblk_per_slot = n_ff >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    int r_safe = min(row, n_ff - 1);
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)r_safe * row_bytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + r_safe) * row_bytes;
    const uint4* qg_row = reinterpret_cast<const uint4*>(wg);
    const __half* dg_row = reinterpret_cast<const __half*>(wg + (size_t)nblk * 16);
    const uint4* qu_row = reinterpret_cast<const uint4*>(wu);
    const __half* du_row = reinterpret_cast<const __half*>(wu + (size_t)nblk * 16);
    float sc = scales ? scales[exp_id] : 1.0f;

    __shared__ float sm_v[8][32];
    __shared__ int8_t sm_q[8][32];

    if (cnt == 1) {
        int tid0 = tok_sorted[off0];
        int sid0 = slot_sorted[off0];
        float ag0 = 0.0f, au0 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                const q8_1_blk xb = Xq[(size_t)tid0 * nblk + b];
                ag0 += dot_q4_soa_q8_1_blk(qg_row[b], __half2float(dg_row[b]), xb);
                au0 += dot_q4_soa_q8_1_blk(qu_row[b], __half2float(du_row[b]), xb);
            }
        }
        float g = s_warp_sum8(ag0) * sc;
        float u = s_warp_sum8(au0) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                float sg = 1.0f / (1.0f + expf(-g));
                float act = beta * tanhf(g / beta) * sg;
                float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                sm_v[0][grp] = act * up;
            } else {
                sm_v[0][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid == 0) {
            size_t qidx = ((size_t)tid0 * n_used + sid0) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[0][lane], lane, act_q + qidx, sm_q[0]);
        }
        return;
    }
    if (cnt == 2) {
        int tid0 = tok_sorted[off0], sid0 = slot_sorted[off0];
        int tid1 = tok_sorted[off0 + 1], sid1 = slot_sorted[off0 + 1];
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                uint4 qg = qg_row[b];
                uint4 qu = qu_row[b];
                float dg = __half2float(dg_row[b]);
                float du = __half2float(du_row[b]);
                const q8_1_blk xb0 = Xq[(size_t)tid0 * nblk + b];
                const q8_1_blk xb1 = Xq[(size_t)tid1 * nblk + b];
                ag0 += dot_q4_soa_q8_1_blk(qg, dg, xb0);
                au0 += dot_q4_soa_q8_1_blk(qu, du, xb0);
                ag1 += dot_q4_soa_q8_1_blk(qg, dg, xb1);
                au1 += dot_q4_soa_q8_1_blk(qu, du, xb1);
            }
        }
        float g0 = s_warp_sum8(ag0) * sc, u0 = s_warp_sum8(au0) * sc;
        float g1 = s_warp_sum8(ag1) * sc, u1 = s_warp_sum8(au1) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                float sg0 = 1.0f / (1.0f + expf(-g0));
                float a0 = beta * tanhf(g0 / beta) * sg0;
                float up0 = (lb > 0.0f) ? (lb * tanhf(u0 / lb)) : u0;
                sm_v[0][grp] = a0 * up0;
                float sg1 = 1.0f / (1.0f + expf(-g1));
                float a1 = beta * tanhf(g1 / beta) * sg1;
                float up1 = (lb > 0.0f) ? (lb * tanhf(u1 / lb)) : u1;
                sm_v[1][grp] = a1 * up1;
            } else {
                sm_v[0][grp] = 0.0f;
                sm_v[1][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid == 0) {
            size_t qidx = ((size_t)tid0 * n_used + sid0) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[0][lane], lane, act_q + qidx, sm_q[0]);
        } else if (wid == 1) {
            size_t qidx = ((size_t)tid1 * n_used + sid1) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[1][lane], lane, act_q + qidx, sm_q[1]);
        }
        return;
    }
    if (cnt == 3) {
        int t0i = tok_sorted[off0], s0i = slot_sorted[off0];
        int t1i = tok_sorted[off0 + 1], s1i = slot_sorted[off0 + 1];
        int t2i = tok_sorted[off0 + 2], s2i = slot_sorted[off0 + 2];
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f, ag2 = 0.0f, au2 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                uint4 qg = qg_row[b];
                uint4 qu = qu_row[b];
                float dg = __half2float(dg_row[b]);
                float du = __half2float(du_row[b]);
                const q8_1_blk xb0 = Xq[(size_t)t0i * nblk + b];
                const q8_1_blk xb1 = Xq[(size_t)t1i * nblk + b];
                const q8_1_blk xb2 = Xq[(size_t)t2i * nblk + b];
                ag0 += dot_q4_soa_q8_1_blk(qg, dg, xb0); au0 += dot_q4_soa_q8_1_blk(qu, du, xb0);
                ag1 += dot_q4_soa_q8_1_blk(qg, dg, xb1); au1 += dot_q4_soa_q8_1_blk(qu, du, xb1);
                ag2 += dot_q4_soa_q8_1_blk(qg, dg, xb2); au2 += dot_q4_soa_q8_1_blk(qu, du, xb2);
            }
        }
        float g0 = s_warp_sum8(ag0) * sc, u0 = s_warp_sum8(au0) * sc;
        float g1 = s_warp_sum8(ag1) * sc, u1 = s_warp_sum8(au1) * sc;
        float g2 = s_warp_sum8(ag2) * sc, u2 = s_warp_sum8(au2) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                sm_v[0][grp] = (beta * tanhf(g0 / beta) / (1.0f + expf(-g0))) * ((lb > 0.0f) ? (lb * tanhf(u0 / lb)) : u0);
                sm_v[1][grp] = (beta * tanhf(g1 / beta) / (1.0f + expf(-g1))) * ((lb > 0.0f) ? (lb * tanhf(u1 / lb)) : u1);
                sm_v[2][grp] = (beta * tanhf(g2 / beta) / (1.0f + expf(-g2))) * ((lb > 0.0f) ? (lb * tanhf(u2 / lb)) : u2);
            } else {
                sm_v[0][grp] = 0.0f; sm_v[1][grp] = 0.0f; sm_v[2][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid < 3) {
            int ti = (wid == 0) ? t0i : ((wid == 1) ? t1i : t2i);
            int si = (wid == 0) ? s0i : ((wid == 1) ? s1i : s2i);
            size_t qidx = ((size_t)ti * n_used + si) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[wid][lane], lane, act_q + qidx, sm_q[wid]);
        }
        return;
    }
    if (cnt == 4) {
        int t0i = tok_sorted[off0], s0i = slot_sorted[off0];
        int t1i = tok_sorted[off0 + 1], s1i = slot_sorted[off0 + 1];
        int t2i = tok_sorted[off0 + 2], s2i = slot_sorted[off0 + 2];
        int t3i = tok_sorted[off0 + 3], s3i = slot_sorted[off0 + 3];
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f, ag2 = 0.0f, au2 = 0.0f, ag3 = 0.0f, au3 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                uint4 qg = qg_row[b];
                uint4 qu = qu_row[b];
                float dg = __half2float(dg_row[b]);
                float du = __half2float(du_row[b]);
                const q8_1_blk xb0 = Xq[(size_t)t0i * nblk + b];
                const q8_1_blk xb1 = Xq[(size_t)t1i * nblk + b];
                const q8_1_blk xb2 = Xq[(size_t)t2i * nblk + b];
                const q8_1_blk xb3 = Xq[(size_t)t3i * nblk + b];
                ag0 += dot_q4_soa_q8_1_blk(qg, dg, xb0); au0 += dot_q4_soa_q8_1_blk(qu, du, xb0);
                ag1 += dot_q4_soa_q8_1_blk(qg, dg, xb1); au1 += dot_q4_soa_q8_1_blk(qu, du, xb1);
                ag2 += dot_q4_soa_q8_1_blk(qg, dg, xb2); au2 += dot_q4_soa_q8_1_blk(qu, du, xb2);
                ag3 += dot_q4_soa_q8_1_blk(qg, dg, xb3); au3 += dot_q4_soa_q8_1_blk(qu, du, xb3);
            }
        }
        float g0 = s_warp_sum8(ag0) * sc, u0 = s_warp_sum8(au0) * sc;
        float g1 = s_warp_sum8(ag1) * sc, u1 = s_warp_sum8(au1) * sc;
        float g2 = s_warp_sum8(ag2) * sc, u2 = s_warp_sum8(au2) * sc;
        float g3 = s_warp_sum8(ag3) * sc, u3 = s_warp_sum8(au3) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                sm_v[0][grp] = (beta * tanhf(g0 / beta) / (1.0f + expf(-g0))) * ((lb > 0.0f) ? (lb * tanhf(u0 / lb)) : u0);
                sm_v[1][grp] = (beta * tanhf(g1 / beta) / (1.0f + expf(-g1))) * ((lb > 0.0f) ? (lb * tanhf(u1 / lb)) : u1);
                sm_v[2][grp] = (beta * tanhf(g2 / beta) / (1.0f + expf(-g2))) * ((lb > 0.0f) ? (lb * tanhf(u2 / lb)) : u2);
                sm_v[3][grp] = (beta * tanhf(g3 / beta) / (1.0f + expf(-g3))) * ((lb > 0.0f) ? (lb * tanhf(u3 / lb)) : u3);
            } else {
                sm_v[0][grp] = 0.0f; sm_v[1][grp] = 0.0f; sm_v[2][grp] = 0.0f; sm_v[3][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid < 4) {
            int ti = (wid == 0) ? t0i : ((wid == 1) ? t1i : ((wid == 2) ? t2i : t3i));
            int si = (wid == 0) ? s0i : ((wid == 1) ? s1i : ((wid == 2) ? s2i : s3i));
            size_t qidx = ((size_t)ti * n_used + si) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[wid][lane], lane, act_q + qidx, sm_q[wid]);
        }
        return;
    }

    for (int t0 = 0; t0 < cnt; t0 += 8) {
        int nt = min(8, cnt - t0);
        int tids[8];
        int sids[8];
        for (int m = 0; m < nt; ++m) {
            tids[m] = tok_sorted[off0 + t0 + m];
            sids[m] = slot_sorted[off0 + t0 + m];
        }
        float ag[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                uint4 qg = qg_row[b];
                uint4 qu = qu_row[b];
                float dg = __half2float(dg_row[b]);
                float du = __half2float(du_row[b]);
                for (int m = 0; m < nt; ++m) {
                    const q8_1_blk xb = Xq[(size_t)tids[m] * nblk + b];
                    ag[m] += dot_q4_soa_q8_1_blk(qg, dg, xb);
                    au[m] += dot_q4_soa_q8_1_blk(qu, du, xb);
                }
            }
        }
        for (int m = 0; m < nt; ++m) {
            float g = s_warp_sum8(ag[m]) * sc;
            float u = s_warp_sum8(au[m]) * sc;
            if (lane8 == 0) {
                if (row < n_ff) {
                    float sg = 1.0f / (1.0f + expf(-g));
                    float act = beta * tanhf(g / beta) * sg;
                    float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                    sm_v[m][grp] = act * up;
                } else {
                    sm_v[m][grp] = 0.0f;
                }
            }
        }
        __syncthreads();
        if (wid < nt) {
            size_t qidx = ((size_t)tids[wid] * n_used + sids[wid]) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[wid][lane], lane, act_q + qidx, sm_q[wid]);
        }
        __syncthreads();
    }
}

extern "C" int ink_moe_gu_q4_0_sudot4_to_q8(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, void* act_q, int n_ff, int k, int n_used, float beta, float lb, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    moe_gu_q4_0_sudot4_to_q8_k<<<dim3((n_ff + 31) / 32, gy), dim3(256), 0, s>>>(W, scales, (const q8_1_blk*)Xq, exp_counts, exp_offsets, tok_sorted, slot_sorted, (q8_1_blk*)act_q, n_ff, k, n_used, beta, lb);
    return 0;
}

extern "C" __global__ void kv_conv_qkvr_store_multi_snap_k(float* __restrict__ Q,
                                                           const float* __restrict__ K_raw,
                                                           const float* __restrict__ V_raw,
                                                           float* __restrict__ R,
                                                           const float* __restrict__ k_conv_w,
                                                           const float* __restrict__ v_conv_w,
                                                           const float* __restrict__ k_dyn,
                                                           const float* __restrict__ v_dyn,
                                                           float* __restrict__ k_hist_pool,
                                                           float* __restrict__ v_hist_pool,
                                                           float* __restrict__ k_snap_pool,
                                                           float* __restrict__ v_snap_pool,
                                                           const float* __restrict__ q_clip,
                                                           const float* __restrict__ rel_clip,
                                                           __half* __restrict__ k_cache_pool,
                                                           __half* __restrict__ v_cache_pool,
                                                           const int* __restrict__ d_seq_ids,
                                                           const int* __restrict__ d_positions,
                                                           int K, int is_swa, int n_head, int n_head_kv,
                                                           int dh, int d_rel, int max_ctx,
                                                           float qk_thresh, float rel_thresh,
                                                           float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int u = blockIdx.y;
    int lane = threadIdx.x;
    int seq = d_seq_ids[u * K];
    int pos0 = d_positions[u * K];
    int bt_base = u * K;
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (h < n_head) {
        const float* qch = q_clip + (size_t)h * dh;
        const float* rch = rel_clip + (size_t)h * d_rel;
        for (int t = 0; t < K; ++t) {
            int bt = bt_base + t;
            int pos = pos0 + t;
            float tau = 1.0f;
            if (!is_swa) {
                float ratio = ((float)pos + 1.0f) / log_floor;
                if (ratio < 1.0f) ratio = 1.0f;
                tau = 1.0f + log_alpha * logf(ratio);
            }
            float* qh = Q + (size_t)bt * q_dim + (size_t)h * dh;
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
            float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
            float rss = s_warp_sum32(rv * rv);
            float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
            if (lane < d_rel) rh[lane] = rv * rfactor;
        }
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float* k_hist = k_hist_pool + (size_t)seq * kv_dim * 3;
        float* v_hist = v_hist_pool + (size_t)seq * kv_dim * 3;
        float* k_snap = k_snap_pool ? (k_snap_pool + (size_t)u * 8 * kv_dim * 3) : nullptr;
        float* v_snap = v_snap_pool ? (v_snap_pool + (size_t)u * 8 * kv_dim * 3) : nullptr;
        size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
        __half* kc_base = k_cache_pool + (size_t)seq * seq_kv_stride;
        __half* vc_base = v_cache_pool + (size_t)seq * seq_kv_stride;
        float kh0[4], kh1[4], kh2[4], vh0[4], vh1[4], vh2[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            kh0[p] = k_hist[c * 3 + 0]; kh1[p] = k_hist[c * 3 + 1]; kh2[p] = k_hist[c * 3 + 2];
            vh0[p] = v_hist[c * 3 + 0]; vh1[p] = v_hist[c * 3 + 1]; vh2[p] = v_hist[c * 3 + 2];
        }
        for (int t = 0; t < K; ++t) {
            int bt = bt_base + t;
            int pos = pos0 + t;
            int slot = is_swa ? (pos % max_ctx) : pos;
            float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
            float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
            if (k_dyn) {
                const float* dk = k_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
                const float* dv = v_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
                kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
                vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
            }
            float kv_v[4], vv_v[4];
            float kss = 0.0f, vss = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int c = kh_idx * dh + lane + p * 32;
                float kx = K_raw[(size_t)bt * kv_dim + c];
                const float* kwc = k_conv_w + (size_t)c * 4;
                float kacc = kwc[0] * kh0[p] + kwc[1] * kh1[p] + kwc[2] * kh2[p] + kwc[3] * kx;
                if (k_dyn) kacc += kd0 * kx + kd1 * kh2[p] + kd2 * kh1[p] + kd3 * kh0[p];
                kh0[p] = kh1[p]; kh1[p] = kh2[p]; kh2[p] = kx;
                if (k_snap) {
                    float* ks = k_snap + ((size_t)t * kv_dim + c) * 3;
                    ks[0] = kh0[p]; ks[1] = kh1[p]; ks[2] = kh2[p];
                }
                float kout = (kx + kacc) * sc;
                kv_v[p] = kout;
                kss += kout * kout;

                float vx = V_raw[(size_t)bt * kv_dim + c];
                const float* vwc = v_conv_w + (size_t)c * 4;
                float vacc = vwc[0] * vh0[p] + vwc[1] * vh1[p] + vwc[2] * vh2[p] + vwc[3] * vx;
                if (v_dyn) vacc += vd0 * vx + vd1 * vh2[p] + vd2 * vh1[p] + vd3 * vh0[p];
                vh0[p] = vh1[p]; vh1[p] = vh2[p]; vh2[p] = vx;
                if (v_snap) {
                    float* vs = v_snap + ((size_t)t * kv_dim + c) * 3;
                    vs[0] = vh0[p]; vs[1] = vh1[p]; vs[2] = vh2[p];
                }
                float vout = (vx + vacc) * sc;
                vv_v[p] = vout;
                vss += vout * vout;
            }
            kss = s_warp_sum32(kss);
            vss = s_warp_sum32(vss);
            float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
            float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
            __half* kc_slot = kc_base + ((size_t)kh_idx * max_ctx + slot) * dh;
            __half* vc_slot = vc_base + ((size_t)kh_idx * max_ctx + slot) * dh;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int d = lane + p * 32;
                kc_slot[d] = __float2half(kv_v[p] * kfac);
                vc_slot[d] = __float2half(vv_v[p] * vinv);
            }
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            k_hist[c * 3 + 0] = kh0[p]; k_hist[c * 3 + 1] = kh1[p]; k_hist[c * 3 + 2] = kh2[p];
            v_hist[c * 3 + 0] = vh0[p]; v_hist[c * 3 + 1] = vh1[p]; v_hist[c * 3 + 2] = vh2[p];
        }
    }
}

extern "C" __global__ void shortconv_res_multi_snap_k(const float* __restrict__ raw,
                                                      const float* __restrict__ conv_w,
                                                      float* __restrict__ hist_pool,
                                                      float* __restrict__ snap_pool,
                                                      const float* __restrict__ add_src,
                                                      float* __restrict__ block_dst,
                                                      float* __restrict__ out,
                                                      const int* __restrict__ d_seq_ids,
                                                      int n_embd, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;
    if (c >= n_embd) return;
    int seq = d_seq_ids[u * K];
    int bt_base = u * K;
    float* hc = hist_pool + ((size_t)seq * n_embd + c) * 3;
    float* sn_base = snap_pool ? (snap_pool + (size_t)u * 8 * n_embd * 3) : nullptr;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = conv_w + (size_t)c * 4;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    for (int t = 0; t < K; ++t) {
        size_t idx = (size_t)(bt_base + t) * n_embd + c;
        float x = raw[idx];
        float conv_out = (x + w0 * h0 + w1 * h1 + w2 * h2 + w3 * x) * rsqrtf(5.0f);
        h0 = h1; h1 = h2; h2 = x;
        if (sn_base) {
            float* sn = sn_base + ((size_t)t * n_embd + c) * 3;
            sn[0] = h0; sn[1] = h1; sn[2] = h2;
        }
        if (block_dst) {
            block_dst[idx] = add_src[idx];
            out[idx] = conv_out;
        } else {
            out[idx] = add_src[idx] + conv_out;
        }
    }
    hc[0] = h0; hc[1] = h1; hc[2] = h2;
}

extern "C" int ink_kv_conv_qkvr_store_multi_snap(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist_pool, float* v_hist_pool, float* k_snap_pool, float* v_snap_pool, const float* q_clip, const float* rel_clip, __half* kc_pool, __half* vc_pool, const int* d_seq_ids, const int* d_positions, int n_users, int K, int is_swa, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_multi_snap_k<<<dim3(n_head + n_head_kv, n_users), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist_pool, v_hist_pool, k_snap_pool, v_snap_pool, q_clip, rel_clip, kc_pool, vc_pool, d_seq_ids, d_positions, K, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}

extern "C" int ink_shortconv_res_multi_snap(hipStream_t s, const float* raw, const float* conv_w, float* hist_pool, float* snap_pool, const float* add_src, float* block_dst, float* out, const int* d_seq_ids, int n_embd, int n_users, int K) {
    shortconv_res_multi_snap_k<<<dim3((n_embd + 255) / 256, n_users), dim3(256), 0, s>>>(raw, conv_w, hist_pool, snap_pool, add_src, block_dst, out, d_seq_ids, n_embd, K);
    return 0;
}

extern "C" __global__ void inkling_attn_batch_gated_q8_k(const float* __restrict__ Q,
                                                         const float* __restrict__ R,
                                                         const float* __restrict__ sinks,
                                                         const float* __restrict__ rel_bias,
                                                         const __half* __restrict__ k_cache,
                                                         const __half* __restrict__ v_cache,
                                                         const float* __restrict__ gate,
                                                         const int* __restrict__ d_pos0,
                                                         q8_1_blk* __restrict__ out_q8,
                                                         int n_head, int n_head_kv,
                                                         int dh, int d_rel,
                                                         int rel_extent, int swa_window,
                                                         int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int pos = (*d_pos0) + bt;
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

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

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
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        float v0 = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        float v1 = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        float v2 = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        float v3 = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
        __shared__ int8_t smq[32];
        size_t qbase = ((size_t)bt * n_head + h) * 4;
        store_q8_1_from_warp(v0, lane, out_q8 + qbase + 0, smq);
        store_q8_1_from_warp(v1, lane, out_q8 + qbase + 1, smq);
        store_q8_1_from_warp(v2, lane, out_q8 + qbase + 2, smq);
        store_q8_1_from_warp(v3, lane, out_q8 + qbase + 3, smq);
    }
}

extern "C" int ink_attn_b8_flash_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B);
extern "C" int ink_attn_batch_gated_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    if (B >= 4 && B <= 16 && dh == 128 && d_rel == 16) {
        return ink_attn_b8_flash_q8(s, Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B);
    }
    inkling_attn_batch_gated_q8_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}
