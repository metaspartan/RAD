#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

static constexpr float S_LOG2E = 1.4426950408889634f;

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

__device__ __forceinline__ size_t q4_soa_row_bytes(int nblk) {
    return (size_t)nblk * 16 + (((size_t)nblk * 2 + 15) & ~(size_t)15);
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


__device__ __forceinline__ void store_q8_1_from_warp(float v, int lane, q8_1_blk* dst, int8_t* sm) {
    (void)sm;
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
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if ((lane & 3) == 0) {
        dst->q[lane >> 2] = pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
    }
}
