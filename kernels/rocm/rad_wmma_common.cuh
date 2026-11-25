#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

static constexpr float LOG2E = 1.4426950408889634f;

typedef int32_t wmma_i8_v __attribute__((ext_vector_type(8)));
typedef int32_t wmma_i2_v __attribute__((ext_vector_type(2)));
typedef float wmma_f8_v __attribute__((ext_vector_type(8)));
typedef _Float16 wmma_h8_v __attribute__((ext_vector_type(8)));

struct w_q8_1_blk {
    float d;
    int32_t s;
    int32_t q[8];
};

__device__ __forceinline__ size_t w_q4_soa_row_bytes(int nblk) {
    return (size_t)nblk * 16 + (((size_t)nblk * 2 + 15) & ~(size_t)15);
}

__device__ __forceinline__ float w_bf16_f32(uint16_t u) {
    uint32_t bits = (uint32_t)u << 16;
    return __uint_as_float(bits);
}

__device__ __forceinline__ float w_warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ float w_fmax32_top(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
    return v;
}

__device__ __forceinline__ int32_t w_isum32_top(int32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ int32_t w_p4i8_top(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int32_t)(uint8_t)a) | (((int32_t)(uint8_t)b) << 8) | (((int32_t)(uint8_t)c) << 16) | (((int32_t)(uint8_t)d) << 24);
}
