#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

typedef _Float16 q8kv_h8_v __attribute__((ext_vector_type(8)));
typedef float q8kv_f8_v __attribute__((ext_vector_type(8)));

struct q8kv_q8_1_blk {
    float d;
    int32_t s;
    int32_t q[8];
};

__device__ __forceinline__ float q8kv_warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ float q8kv_warp_max32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
    return v;
}

__device__ __forceinline__ int32_t q8kv_warp_sum32_i(int32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ int32_t q8kv_pack4_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (int32_t)((uint32_t)(uint8_t)a
                   | ((uint32_t)(uint8_t)b << 8)
                   | ((uint32_t)(uint8_t)c << 16)
                   | ((uint32_t)(uint8_t)d << 24));
}

__device__ __forceinline__ void q8kv_store_q8_1(float v, int lane, q8kv_q8_1_blk* dst, int8_t* smq) {
    float amax = __shfl(q8kv_warp_max32(fabsf(v)), 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    smq[lane] = (int8_t)qi;
    int32_t qsum = q8kv_warp_sum32_i(qi);
    __builtin_amdgcn_wave_barrier();
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if (lane < 8) {
        int i = lane * 4;
        dst->q[lane] = q8kv_pack4_i8(smq[i], smq[i + 1], smq[i + 2], smq[i + 3]);
    }
}

__device__ __forceinline__ void q8kv_quantize_store_128(float v0, float v1, float v2, float v3, int lane, int8_t* dst, __half* dst_scale) {
    float lm = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = __shfl(q8kv_warp_max32(lm), 0, 32);
    float sc = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    if (lane == 0) *dst_scale = __float2half(sc);
    int q0 = max(-127, min(127, __float2int_rn(v0 * inv)));
    int q1 = max(-127, min(127, __float2int_rn(v1 * inv)));
    int q2 = max(-127, min(127, __float2int_rn(v2 * inv)));
    int q3 = max(-127, min(127, __float2int_rn(v3 * inv)));
    dst[lane] = (int8_t)q0;
    dst[lane + 32] = (int8_t)q1;
    dst[lane + 64] = (int8_t)q2;
    dst[lane + 96] = (int8_t)q3;
}
