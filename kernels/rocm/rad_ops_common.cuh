#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

static __device__ __constant__ float KVAL_FP4[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

__device__ __forceinline__ float ue4m3_to_fp32_fast(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t man = x & 0x7;
    if (exp == 0) {
        return (float)man * 0.0009765625f;
    }
    uint32_t bits = ((exp + 119u) << 23) | (man << 20);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 32);
    }
    return v;
}

__device__ __forceinline__ float warp_sum16(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 16);
    }
    return v;
}

__device__ __forceinline__ float warp_sum8(float v) {
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 8);
    }
    return v;
}

__device__ __forceinline__ float dot_q8_0_blk(const uint8_t* __restrict__ bp,
                                              const float* __restrict__ xb) {
    uint16_t dh_bits;
    memcpy(&dh_bits, bp, 2);
    float d = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
    uint32_t qv[8];
    memcpy(qv, bp + 2, 32);
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t q = qv[j];
        float4 xv = x4[j];
        s = fmaf((float)(int8_t)(q & 0xFF), xv.x, s);
        s = fmaf((float)(int8_t)((q >> 8) & 0xFF), xv.y, s);
        s = fmaf((float)(int8_t)((q >> 16) & 0xFF), xv.z, s);
        s = fmaf((float)(int8_t)(q >> 24), xv.w, s);
    }
    return d * s;
}

__device__ __forceinline__ float fp4_nibble(uint32_t q) {
    uint32_t n = q & 7u;
    uint32_t e = n >> 1;
    uint32_t m = n & 1u;
    int v = (e == 0u) ? (int)m : (int)((2u + m) << (e - 1u));
    return (q & 8u) ? -(float)v : (float)v;
}

__device__ __forceinline__ float dot_nvfp4_blk(const uint8_t* __restrict__ bp,
                                               const float* __restrict__ xb) {
    uint32_t ds;
    memcpy(&ds, bp, 4);
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(bp + 4);
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    float acc = 0.0f;
#pragma unroll
    for (int sb = 0; sb < 4; ++sb) {
        float d = ue4m3_to_fp32_fast((uint8_t)((ds >> (sb * 8)) & 0xFF));
        uint32_t q0 = q32[sb * 2];
        uint32_t q1 = q32[sb * 2 + 1];
        float4 x0 = x4[sb * 4 + 0];
        float4 x1 = x4[sb * 4 + 1];
        float4 x2 = x4[sb * 4 + 2];
        float4 x3 = x4[sb * 4 + 3];
        float s = 0.0f;
        s = fmaf(fp4_nibble(q0), x0.x, s);
        s = fmaf(fp4_nibble(q0 >> 4), x2.x, s);
        s = fmaf(fp4_nibble(q0 >> 8), x0.y, s);
        s = fmaf(fp4_nibble(q0 >> 12), x2.y, s);
        s = fmaf(fp4_nibble(q0 >> 16), x0.z, s);
        s = fmaf(fp4_nibble(q0 >> 20), x2.z, s);
        s = fmaf(fp4_nibble(q0 >> 24), x0.w, s);
        s = fmaf(fp4_nibble(q0 >> 28), x2.w, s);
        s = fmaf(fp4_nibble(q1), x1.x, s);
        s = fmaf(fp4_nibble(q1 >> 4), x3.x, s);
        s = fmaf(fp4_nibble(q1 >> 8), x1.y, s);
        s = fmaf(fp4_nibble(q1 >> 12), x3.y, s);
        s = fmaf(fp4_nibble(q1 >> 16), x1.z, s);
        s = fmaf(fp4_nibble(q1 >> 20), x3.z, s);
        s = fmaf(fp4_nibble(q1 >> 24), x1.w, s);
        s = fmaf(fp4_nibble(q1 >> 28), x3.w, s);
        acc = fmaf(d, s, acc);
    }
    return acc;
}

__device__ __forceinline__ float dot_q8_0_row32(const uint8_t* __restrict__ wrow,
                                                const float* __restrict__ x,
                                                int nblk, int lane) {
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        acc += dot_q8_0_blk(wrow + (size_t)b * 34, x + ((size_t)b << 5));
    }
    return warp_sum32(acc);
}

__device__ __forceinline__ float dot_nvfp4_row32(const uint8_t* __restrict__ wrow,
                                                 const float* __restrict__ x,
                                                 int nblk, int lane) {
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        acc += dot_nvfp4_blk(wrow + (size_t)b * 36, x + ((size_t)b << 6));
    }
    return warp_sum32(acc);
}

__device__ __forceinline__ float dot_nvfp4_row16(const uint8_t* __restrict__ wrow,
                                                 const float* __restrict__ x,
                                                 int nblk, int lane16) {
    float acc = 0.0f;
    for (int b = lane16; b < nblk; b += 16) {
        acc += dot_nvfp4_blk(wrow + (size_t)b * 36, x + ((size_t)b << 6));
    }
    return warp_sum16(acc);
}

__device__ __forceinline__ float dot_nvfp4_row8(const uint8_t* __restrict__ wrow,
                                                const float* __restrict__ x,
                                                int nblk, int lane8) {
    float acc = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        acc += dot_nvfp4_blk(wrow + (size_t)b * 36, x + ((size_t)b << 6));
    }
    return warp_sum8(acc);
}
