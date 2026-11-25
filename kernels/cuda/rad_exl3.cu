#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ uint32_t exl3_fshift(uint32_t b, uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}

__device__ __forceinline__ half2 decode_mul1_2(uint32_t idx0, uint32_t idx1) {
    uint32_t x0 = idx0 * 0x83DCD12Du;
    uint32_t x1 = idx1 * 0x83DCD12Du;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    half2 k_inv = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias = __half2half2(__ushort_as_half(0xc931));
    half2 h = __halves2half2(__ushort_as_half((uint16_t)sum0), __ushort_as_half((uint16_t)sum1));
    return __hfma2(h, k_inv, k_bias);
}

template <int BITS>
__device__ __forceinline__ void dq4_mul1(const uint32_t* ptr, int t_offset, half2& out0, half2& out1) {
    constexpr int WORDS = BITS * 8;
    int b0 = (t_offset + 257) * BITS - 16;
    int b2 = b0 + 3 * BITS + 16;
    int i0 = b0 >> 5;
    int i2 = (b2 - 1) >> 5;
    int s2 = ((i2 + 1) << 5) - b2;
    uint32_t a = ptr[i0 % WORDS];
    uint32_t b = ptr[i2 % WORDS];
    uint32_t w3 = exl3_fshift(b, a, s2) & 0xffffu;
    uint32_t w2 = exl3_fshift(b, a, s2 + BITS) & 0xffffu;
    uint32_t w1 = exl3_fshift(b, a, s2 + BITS * 2) & 0xffffu;
    uint32_t w0 = exl3_fshift(b, a, s2 + BITS * 3) & 0xffffu;
    out0 = decode_mul1_2(w0, w1);
    out1 = decode_mul1_2(w2, w3);
}

template <int BITS>
__device__ __forceinline__ void dq2x2_mul1(const uint32_t* ptr, int t_offset, half2& out0, half2& out1) {
    constexpr int WORDS = BITS * 8;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int b0 = (t_offset + 2 * i + 257) * BITS - 16;
        int b2 = b0 + BITS + 16;
        int i0 = b0 >> 5;
        int i2 = (b2 - 1) >> 5;
        int s2 = ((i2 + 1) << 5) - b2;
        uint32_t a = ptr[i0 % WORDS];
        uint32_t b = ptr[i2 % WORDS];
        uint32_t w1 = exl3_fshift(b, a, s2) & 0xffffu;
        uint32_t w0 = exl3_fshift(b, a, s2 + BITS) & 0xffffu;
        half2 d = decode_mul1_2(w0, w1);
        if (i == 0) out0 = d; else out1 = d;
    }
}

template <int BITS>
__device__ __forceinline__ void dq8_mul1(
    const uint32_t* ptr,
    int t_offset,
    half2& f0,
    half2& f1,
    half2& f2,
    half2& f3
) {
    if constexpr (BITS <= 5) {
        dq4_mul1<BITS>(ptr, t_offset, f0, f1);
        dq4_mul1<BITS>(ptr, t_offset + 4, f2, f3);
    } else {
        dq2x2_mul1<BITS>(ptr, t_offset, f0, f1);
        dq2x2_mul1<BITS>(ptr, t_offset + 4, f2, f3);
    }
}

__device__ __forceinline__ void warp_wht128_f32(float& v0, float& v1, float& v2, float& v3, int lane, float scale) {
    float a0 = v0 + v1, a1 = v0 - v1;
    float a2 = v2 + v3, a3 = v2 - v3;
    v0 = a0 + a2; v1 = a1 + a3;
    v2 = a0 - a2; v3 = a1 - a3;
    #pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
        float sign = (lane & mask) ? -1.0f : 1.0f;
        float p0 = __shfl_xor_sync(0xFFFFFFFFu, v0, mask);
        float p1 = __shfl_xor_sync(0xFFFFFFFFu, v1, mask);
        float p2 = __shfl_xor_sync(0xFFFFFFFFu, v2, mask);
        float p3 = __shfl_xor_sync(0xFFFFFFFFu, v3, mask);
        v0 = p0 + sign * v0;
        v1 = p1 + sign * v1;
        v2 = p2 + sign * v2;
        v3 = p3 + sign * v3;
    }
    v0 *= scale;
    v1 *= scale;
    v2 *= scale;
    v3 *= scale;
}

__global__ void wht128_pre_mul_f16_k(
    const half* __restrict__ x,
    const half* __restrict__ suh,
    half* __restrict__ y,
    int K,
    float scale
) {
    int blk = blockIdx.x;
    int row = blockIdx.y;
    int lane = threadIdx.x;
    int base = blk * 128 + lane * 4;
    const half* xr = x + (size_t)row * K + base;
    half* yr = y + (size_t)row * K + base;
    float v0 = __half2float(xr[0]) * (suh ? __half2float(suh[base + 0]) : 1.0f);
    float v1 = __half2float(xr[1]) * (suh ? __half2float(suh[base + 1]) : 1.0f);
    float v2 = __half2float(xr[2]) * (suh ? __half2float(suh[base + 2]) : 1.0f);
    float v3 = __half2float(xr[3]) * (suh ? __half2float(suh[base + 3]) : 1.0f);
    warp_wht128_f32(v0, v1, v2, v3, lane, scale);
    yr[0] = __float2half(v0);
    yr[1] = __float2half(v1);
    yr[2] = __float2half(v2);
    yr[3] = __float2half(v3);
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_reconstruct_mul1_k(
    half* __restrict__ out,
    const uint16_t* __restrict__ trellis,
    int K_tiles,
    int N_tiles
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int k_tile = blockIdx.y;
    int n_tile_base = blockIdx.x * 8;
    __shared__ uint32_t s_packed[8][PACKED_U32];
    const uint16_t* g_src = trellis + ((size_t)k_tile * N_tiles + n_tile_base) * PACKED_U16;
    if (t < PACKED_U16) {
        ((int4*)s_packed)[t] = ((const int4*)g_src)[t];
    }
    __syncthreads();
    half2 f0, f1, f2, f3;
    dq8_mul1<BITS>(s_packed[warp], lane * 8, f0, f1, f2, f3);
    __shared__ half s_tile[16][128];
    int r0 = (lane & 3) * 2;
    int r1 = r0 + 1;
    int r2 = r0 + 8;
    int r3 = r0 + 9;
    int c_sub = (lane >> 2) & 1;
    int c0 = warp * 16 + (lane >> 3) * 2 + c_sub;
    int c1 = c0 + 8;
    s_tile[r0][c0] = __low2half(f0);
    s_tile[r1][c0] = __high2half(f0);
    s_tile[r2][c0] = __low2half(f1);
    s_tile[r3][c0] = __high2half(f1);
    s_tile[r0][c1] = __low2half(f2);
    s_tile[r1][c1] = __high2half(f2);
    s_tile[r2][c1] = __low2half(f3);
    s_tile[r3][c1] = __high2half(f3);
    __syncthreads();
    int N = N_tiles * 16;
    int r = t >> 4;
    int c8 = (t & 15) * 8;
    *((int4*)(out + (size_t)(k_tile * 16 + r) * N + n_tile_base * 16 + c8)) =
        *((const int4*)(&s_tile[r][c8]));
}

template <int BITS, int M_MAX>
__global__ __launch_bounds__(256)
void exl3_gemv_mul1_k(
    const half* __restrict__ x_rot,
    const uint16_t* __restrict__ trellis,
    const half* __restrict__ svh,
    half* __restrict__ y,
    int M,
    int K,
    int N,
    float wht_scale,
    int add_to_y
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_block = blockIdx.x;
    int n_tile_base = n_block * 8;
    int K_tiles = K >> 4;
    int N_tiles = N >> 4;

    __shared__ uint32_t s_packed[8][PACKED_U32];
    __shared__ float s_x[M_MAX][16];
    __shared__ float s_out[M_MAX][128];

    float acc0[M_MAX];
    float acc1[M_MAX];
    #pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    int r0 = (lane & 3) * 2;
    int r1 = r0 + 1;
    int r2 = r0 + 8;
    int r3 = r0 + 9;
    int c_sub = (lane >> 2) & 1;
    int c0 = (lane >> 3) * 2 + c_sub;
    int c1 = c0 + 8;

    for (int kt = 0; kt < K_tiles; ++kt) {
        const uint16_t* g_src = trellis + ((size_t)kt * N_tiles + n_tile_base) * PACKED_U16;
        if (t < PACKED_U16) {
            ((int4*)s_packed)[t] = ((const int4*)g_src)[t];
        }
        if (t < M * 16) {
            int mr = t >> 4;
            int kc = t & 15;
            s_x[mr][kc] = __half2float(x_rot[(size_t)mr * K + kt * 16 + kc]);
        }
        __syncthreads();

        half2 f0, f1, f2, f3;
        dq8_mul1<BITS>(s_packed[warp], lane * 8, f0, f1, f2, f3);
        float w00 = __half2float(__low2half(f0));
        float w10 = __half2float(__high2half(f0));
        float w20 = __half2float(__low2half(f1));
        float w30 = __half2float(__high2half(f1));
        float w01 = __half2float(__low2half(f2));
        float w11 = __half2float(__high2half(f2));
        float w21 = __half2float(__low2half(f3));
        float w31 = __half2float(__high2half(f3));

        #pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m < M) {
                float x0 = s_x[m][r0];
                float x1 = s_x[m][r1];
                float x2 = s_x[m][r2];
                float x3 = s_x[m][r3];
                acc0[m] = fmaf(w00, x0, fmaf(w10, x1, fmaf(w20, x2, fmaf(w30, x3, acc0[m]))));
                acc1[m] = fmaf(w01, x0, fmaf(w11, x1, fmaf(w21, x2, fmaf(w31, x3, acc1[m]))));
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
        if (m < M) {
            acc0[m] += __shfl_xor_sync(0xFFFFFFFFu, acc0[m], 1);
            acc0[m] += __shfl_xor_sync(0xFFFFFFFFu, acc0[m], 2);
            acc1[m] += __shfl_xor_sync(0xFFFFFFFFu, acc1[m], 1);
            acc1[m] += __shfl_xor_sync(0xFFFFFFFFu, acc1[m], 2);
            if ((lane & 3) == 0) {
                s_out[m][warp * 16 + c0] = acc0[m];
                s_out[m][warp * 16 + c1] = acc1[m];
            }
        }
    }
    __syncthreads();

    if (warp < M) {
        int m = warp;
        int col = n_block * 128 + lane * 4;
        float v0 = s_out[m][lane * 4 + 0];
        float v1 = s_out[m][lane * 4 + 1];
        float v2 = s_out[m][lane * 4 + 2];
        float v3 = s_out[m][lane * 4 + 3];
        if (wht_scale > 0.0f) {
            warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
        }
        if (svh) {
            v0 *= __half2float(svh[col + 0]);
            v1 *= __half2float(svh[col + 1]);
            v2 *= __half2float(svh[col + 2]);
            v3 *= __half2float(svh[col + 3]);
        }
        half* ym = y + (size_t)m * N + col;
        if (add_to_y) {
            ym[0] = __float2half(__half2float(ym[0]) + v0);
            ym[1] = __float2half(__half2float(ym[1]) + v1);
            ym[2] = __float2half(__half2float(ym[2]) + v2);
            ym[3] = __float2half(__half2float(ym[3]) + v3);
        } else {
            ym[0] = __float2half(v0);
            ym[1] = __float2half(v1);
            ym[2] = __float2half(v2);
            ym[3] = __float2half(v3);
        }
    }
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_moe_gemv_mul1_k(
    const half* __restrict__ x_in,
    const uint16_t* const* __restrict__ trellis_ptrs,
    const half* const* __restrict__ suh_ptrs,
    const half* const* __restrict__ svh_ptrs,
    const int* __restrict__ expert_ids,
    half* __restrict__ y_out,
    int K,
    int N,
    float wht_scale,
    int x_per_expert
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_block = blockIdx.x;
    int slot = blockIdx.y;
    int eid = expert_ids[slot];
    const uint16_t* trellis = trellis_ptrs[eid];
    const half* suh = suh_ptrs ? suh_ptrs[eid] : nullptr;
    const half* svh = svh_ptrs ? svh_ptrs[eid] : nullptr;
    const half* x_src = x_per_expert ? (x_in + (size_t)slot * K) : x_in;

    int K_tiles = K >> 4;
    int N_tiles = N >> 4;
    int n_tile_base = n_block * 8;

    __shared__ uint32_t s_packed[8][PACKED_U32];
    __shared__ float s_x128[128];
    __shared__ float s_out[128];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int r0 = (lane & 3) * 2;
    int r1 = r0 + 1;
    int r2 = r0 + 8;
    int r3 = r0 + 9;
    int c_sub = (lane >> 2) & 1;
    int c0 = (lane >> 3) * 2 + c_sub;
    int c1 = c0 + 8;

    for (int kb = 0; kb < K; kb += 128) {
        if (warp == 0) {
            int base = kb + lane * 4;
            float v0 = __half2float(x_src[base + 0]) * (suh ? __half2float(suh[base + 0]) : 1.0f);
            float v1 = __half2float(x_src[base + 1]) * (suh ? __half2float(suh[base + 1]) : 1.0f);
            float v2 = __half2float(x_src[base + 2]) * (suh ? __half2float(suh[base + 2]) : 1.0f);
            float v3 = __half2float(x_src[base + 3]) * (suh ? __half2float(suh[base + 3]) : 1.0f);
            warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
            s_x128[lane * 4 + 0] = v0;
            s_x128[lane * 4 + 1] = v1;
            s_x128[lane * 4 + 2] = v2;
            s_x128[lane * 4 + 3] = v3;
        }
        __syncthreads();
        int kt0 = kb >> 4;
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            int kt = kt0 + sub;
            const uint16_t* g_src = trellis + ((size_t)kt * N_tiles + n_tile_base) * PACKED_U16;
            if (t < PACKED_U16) {
                ((int4*)s_packed)[t] = ((const int4*)g_src)[t];
            }
            __syncthreads();
            half2 f0, f1, f2, f3;
            dq8_mul1<BITS>(s_packed[warp], lane * 8, f0, f1, f2, f3);
            int off = sub * 16;
            float x0 = s_x128[off + r0];
            float x1 = s_x128[off + r1];
            float x2 = s_x128[off + r2];
            float x3 = s_x128[off + r3];
            acc0 = fmaf(__half2float(__low2half(f0)), x0,
                   fmaf(__half2float(__high2half(f0)), x1,
                   fmaf(__half2float(__low2half(f1)), x2,
                   fmaf(__half2float(__high2half(f1)), x3, acc0))));
            acc1 = fmaf(__half2float(__low2half(f2)), x0,
                   fmaf(__half2float(__high2half(f2)), x1,
                   fmaf(__half2float(__low2half(f3)), x2,
                   fmaf(__half2float(__high2half(f3)), x3, acc1))));
            __syncthreads();
        }
    }

    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1);
    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 2);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 2);
    if ((lane & 3) == 0) {
        s_out[warp * 16 + c0] = acc0;
        s_out[warp * 16 + c1] = acc1;
    }
    __syncthreads();

    if (warp == 0) {
        int col = n_block * 128 + lane * 4;
        float v0 = s_out[lane * 4 + 0];
        float v1 = s_out[lane * 4 + 1];
        float v2 = s_out[lane * 4 + 2];
        float v3 = s_out[lane * 4 + 3];
        if (wht_scale > 0.0f) {
            warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
        }
        if (svh) {
            v0 *= __half2float(svh[col + 0]);
            v1 *= __half2float(svh[col + 1]);
            v2 *= __half2float(svh[col + 2]);
            v3 *= __half2float(svh[col + 3]);
        }
        half* ys = y_out + (size_t)slot * N + col;
        ys[0] = __float2half(v0);
        ys[1] = __float2half(v1);
        ys[2] = __float2half(v2);
        ys[3] = __float2half(v3);
    }
}

extern "C" {

void rad_cuda_wht128_pre_mul_f16(
    const half* x,
    const half* suh,
    half* y,
    int M,
    int K,
    float scale,
    cudaStream_t stream
) {
    dim3 grid(K / 128, M);
    wht128_pre_mul_f16_k<<<grid, 32, 0, stream>>>(x, suh, y, K, scale);
}

void rad_cuda_exl3_reconstruct_f16(
    half* out,
    const uint16_t* trellis,
    int K,
    int N,
    int bits,
    cudaStream_t stream
) {
    int K_tiles = K / 16;
    int N_tiles = N / 16;
    dim3 grid(N_tiles / 8, K_tiles);
    switch (bits) {
        case 2: exl3_reconstruct_mul1_k<2><<<grid, 256, 0, stream>>>(out, trellis, K_tiles, N_tiles); break;
        case 3: exl3_reconstruct_mul1_k<3><<<grid, 256, 0, stream>>>(out, trellis, K_tiles, N_tiles); break;
        case 4: exl3_reconstruct_mul1_k<4><<<grid, 256, 0, stream>>>(out, trellis, K_tiles, N_tiles); break;
        case 5: exl3_reconstruct_mul1_k<5><<<grid, 256, 0, stream>>>(out, trellis, K_tiles, N_tiles); break;
        case 6: exl3_reconstruct_mul1_k<6><<<grid, 256, 0, stream>>>(out, trellis, K_tiles, N_tiles); break;
        case 8: exl3_reconstruct_mul1_k<8><<<grid, 256, 0, stream>>>(out, trellis, K_tiles, N_tiles); break;
    }
}

void rad_cuda_exl3_gemv_f16(
    const half* x_rot,
    const uint16_t* trellis,
    const half* svh,
    half* y,
    int M,
    int K,
    int N,
    int bits,
    float wht_scale,
    int add_to_y,
    cudaStream_t stream
) {
    dim3 grid(N / 128);
    if (M == 1) {
        switch (bits) {
            case 3: exl3_gemv_mul1_k<3, 1><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, 1, K, N, wht_scale, add_to_y); break;
            case 4: exl3_gemv_mul1_k<4, 1><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, 1, K, N, wht_scale, add_to_y); break;
            case 5: exl3_gemv_mul1_k<5, 1><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, 1, K, N, wht_scale, add_to_y); break;
            case 6: exl3_gemv_mul1_k<6, 1><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, 1, K, N, wht_scale, add_to_y); break;
            case 8: exl3_gemv_mul1_k<8, 1><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, 1, K, N, wht_scale, add_to_y); break;
        }
    } else if (M <= 5) {
        switch (bits) {
            case 3: exl3_gemv_mul1_k<3, 5><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 4: exl3_gemv_mul1_k<4, 5><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 5: exl3_gemv_mul1_k<5, 5><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 6: exl3_gemv_mul1_k<6, 5><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 8: exl3_gemv_mul1_k<8, 5><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
        }
    } else {
        switch (bits) {
            case 3: exl3_gemv_mul1_k<3, 8><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 4: exl3_gemv_mul1_k<4, 8><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 5: exl3_gemv_mul1_k<5, 8><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 6: exl3_gemv_mul1_k<6, 8><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
            case 8: exl3_gemv_mul1_k<8, 8><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, y, M, K, N, wht_scale, add_to_y); break;
        }
    }
}

void rad_cuda_exl3_moe_gemv_f16(
    const half* x_in,
    const uint16_t* const* trellis_ptrs,
    const half* const* suh_ptrs,
    const half* const* svh_ptrs,
    const int* expert_ids,
    half* y_out,
    int top_k,
    int K,
    int N,
    int bits,
    float wht_scale,
    int x_per_expert,
    cudaStream_t stream
) {
    dim3 grid(N / 128, top_k);
    switch (bits) {
        case 3: exl3_moe_gemv_mul1_k<3><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, svh_ptrs, expert_ids, y_out, K, N, wht_scale, x_per_expert); break;
        case 4: exl3_moe_gemv_mul1_k<4><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, svh_ptrs, expert_ids, y_out, K, N, wht_scale, x_per_expert); break;
        case 5: exl3_moe_gemv_mul1_k<5><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, svh_ptrs, expert_ids, y_out, K, N, wht_scale, x_per_expert); break;
    }
}

}
