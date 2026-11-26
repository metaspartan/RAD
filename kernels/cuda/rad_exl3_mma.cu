#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ uint32_t mma_fshift(uint32_t b, uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}

__device__ __forceinline__ half2 mma_decode_mul1_2(uint32_t idx0, uint32_t idx1) {
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
__device__ __forceinline__ void mma_dq4_mul1(const uint32_t* ptr, int t_offset, half2& out0, half2& out1) {
    constexpr int WORDS = BITS * 8;
    int b0 = (t_offset + 257) * BITS - 16;
    int b2 = b0 + 3 * BITS + 16;
    int i0 = b0 >> 5;
    int i2 = (b2 - 1) >> 5;
    int s2 = ((i2 + 1) << 5) - b2;
    uint32_t a = ptr[(i0 >= WORDS) ? (i0 - WORDS) : i0];
    uint32_t b = ptr[(i2 >= WORDS) ? (i2 - WORDS) : i2];
    uint32_t w3 = mma_fshift(b, a, s2) & 0xffffu;
    uint32_t w2 = mma_fshift(b, a, s2 + BITS) & 0xffffu;
    uint32_t w1 = mma_fshift(b, a, s2 + BITS * 2) & 0xffffu;
    uint32_t w0 = mma_fshift(b, a, s2 + BITS * 3) & 0xffffu;
    out0 = mma_decode_mul1_2(w0, w1);
    out1 = mma_decode_mul1_2(w2, w3);
}

template <int BITS>
__device__ __forceinline__ void mma_dq2x2_mul1(const uint32_t* ptr, int t_offset, half2& out0, half2& out1) {
    constexpr int WORDS = BITS * 8;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int b0 = (t_offset + 2 * i + 257) * BITS - 16;
        int b2 = b0 + BITS + 16;
        int i0 = b0 >> 5;
        int i2 = (b2 - 1) >> 5;
        int s2 = ((i2 + 1) << 5) - b2;
        uint32_t a = ptr[(i0 >= WORDS) ? (i0 - WORDS) : i0];
        uint32_t b = ptr[(i2 >= WORDS) ? (i2 - WORDS) : i2];
        uint32_t w1 = mma_fshift(b, a, s2) & 0xffffu;
        uint32_t w0 = mma_fshift(b, a, s2 + BITS) & 0xffffu;
        half2 d = mma_decode_mul1_2(w0, w1);
        if (i == 0) out0 = d; else out1 = d;
    }
}

template <int BITS>
__device__ __forceinline__ void mma_dq8_mul1(
    const uint32_t* ptr,
    int t_offset,
    uint32_t& b00,
    uint32_t& b01,
    uint32_t& b10,
    uint32_t& b11
) {
    half2 f0, f1, f2, f3;
    if constexpr (BITS <= 5) {
        mma_dq4_mul1<BITS>(ptr, t_offset, f0, f1);
        mma_dq4_mul1<BITS>(ptr, t_offset + 4, f2, f3);
    } else {
        mma_dq2x2_mul1<BITS>(ptr, t_offset, f0, f1);
        mma_dq2x2_mul1<BITS>(ptr, t_offset + 4, f2, f3);
    }
    b00 = *reinterpret_cast<const uint32_t*>(&f0);
    b01 = *reinterpret_cast<const uint32_t*>(&f1);
    b10 = *reinterpret_cast<const uint32_t*>(&f2);
    b11 = *reinterpret_cast<const uint32_t*>(&f3);
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ void mma_warp_wht128_f32(float& v0, float& v1, float& v2, float& v3, int lane, float scale) {
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

__global__ void mma_wht128_post_f32_to_f16_k(
    const float* __restrict__ z_f32,
    const half* __restrict__ svh,
    const half* __restrict__ gate_in,
    const half* __restrict__ next_suh,
    half* __restrict__ y,
    int N,
    float scale,
    float swiglu_limit,
    int add_to_y
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= N) return;
    int row = blockIdx.y;
    int col = blk * 128 + lane * 4;
    float4 zv = *reinterpret_cast<const float4*>(z_f32 + (size_t)row * N + col);
    float v0 = zv.x, v1 = zv.y, v2 = zv.z, v3 = zv.w;
    if (scale > 0.0f) {
        mma_warp_wht128_f32(v0, v1, v2, v3, lane, scale);
    }
    if (svh) {
        uint2 sq = __ldg(reinterpret_cast<const uint2*>(svh + col));
        float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
        float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
        v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
    }
    if (gate_in) {
        uint2 gq = *reinterpret_cast<const uint2*>(gate_in + (size_t)row * N + col);
        float2 g01 = __half22float2(*reinterpret_cast<const half2*>(&gq.x));
        float2 g23 = __half22float2(*reinterpret_cast<const half2*>(&gq.y));
        auto act = [&](float g, float u) -> float {
            float gc = fminf(g, swiglu_limit);
            float uc = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            return (gc / (1.0f + expf(-gc))) * uc;
        };
        v0 = act(g01.x, v0);
        v1 = act(g01.y, v1);
        v2 = act(g23.x, v2);
        v3 = act(g23.y, v3);
        if (next_suh) {
            uint2 nq = __ldg(reinterpret_cast<const uint2*>(next_suh + col));
            float2 n01 = __half22float2(*reinterpret_cast<const half2*>(&nq.x));
            float2 n23 = __half22float2(*reinterpret_cast<const half2*>(&nq.y));
            v0 *= n01.x; v1 *= n01.y; v2 *= n23.x; v3 *= n23.y;
            mma_warp_wht128_f32(v0, v1, v2, v3, lane, scale);
        }
    } else if (add_to_y) {
        uint2 yq = *reinterpret_cast<const uint2*>(y + (size_t)row * N + col);
        float2 y01 = __half22float2(*reinterpret_cast<const half2*>(&yq.x));
        float2 y23 = __half22float2(*reinterpret_cast<const half2*>(&yq.y));
        v0 += y01.x; v1 += y01.y; v2 += y23.x; v3 += y23.y;
    }
    half2 h01 = __float22half2_rn(make_float2(fmaxf(fminf(v0, 65000.0f), -65000.0f), fmaxf(fminf(v1, 65000.0f), -65000.0f)));
    half2 h23 = __float22half2_rn(make_float2(fmaxf(fminf(v2, 65000.0f), -65000.0f), fmaxf(fminf(v3, 65000.0f), -65000.0f)));
    uint2 out_u2;
    out_u2.x = *reinterpret_cast<const uint32_t*>(&h01);
    out_u2.y = *reinterpret_cast<const uint32_t*>(&h23);
    *reinterpret_cast<uint2*>(y + (size_t)row * N + col) = out_u2;
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_mma_m16_mul1_k(
    const half* __restrict__ x_rot,
    const uint16_t* __restrict__ trellis,
    float* __restrict__ z_f32,
    int M,
    int K,
    int N,
    int split_k
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int m_block = blockIdx.y;
    int sk = blockIdx.z;
    int m_base = m_block * 16;
    int K_tiles = K >> 4;
    int N_tiles = N >> 4;
    if (n_tile >= N_tiles || m_base >= M) return;

    int kt_start = (K_tiles * sk) / split_k;
    int kt_end = (K_tiles * (sk + 1)) / split_k;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];

    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int m0 = m_base + group_id;
    int m1 = m_base + group_id + 8;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;

    constexpr int PACKED_U4 = 2 * BITS;
    uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);
    for (int kt = kt_start; kt < kt_end; ++kt) {
        const uint4* g_u4 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
        if (lane < PACKED_U4) {
            s_u4[lane] = __ldg(g_u4 + lane);
        }
        __syncwarp();

        uint32_t b00, b01, b10, b11;
        mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

        int kb = kt * 16;
        uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
        if (m0 < M) {
            const half* r0 = x_rot + (size_t)m0 * K + kb;
            a0 = *reinterpret_cast<const uint32_t*>(r0 + k_off0);
            a2 = *reinterpret_cast<const uint32_t*>(r0 + k_off1);
        }
        if (m1 < M) {
            const half* r1 = x_rot + (size_t)m1 * K + kb;
            a1 = *reinterpret_cast<const uint32_t*>(r1 + k_off0);
            a3 = *reinterpret_cast<const uint32_t*>(r1 + k_off1);
        }

        mma_m16n8k16_f32(c00, c01, c02, c03, a0, a1, a2, a3, b00, b01);
        mma_m16n8k16_f32(c10, c11, c12, c13, a0, a1, a2, a3, b10, b11);
        __syncwarp();
    }

    int col_base = n_tile * 16 + tid_in_grp * 2;
    if (m0 < M) {
        float* z0 = z_f32 + (size_t)m0 * N + col_base;
        if (split_k == 1) {
            z0[0] = c00; z0[1] = c01;
            z0[8] = c10; z0[9] = c11;
        } else {
            atomicAdd(z0 + 0, c00); atomicAdd(z0 + 1, c01);
            atomicAdd(z0 + 8, c10); atomicAdd(z0 + 9, c11);
        }
    }
    if (m1 < M) {
        float* z1 = z_f32 + (size_t)m1 * N + col_base;
        if (split_k == 1) {
            z1[0] = c02; z1[1] = c03;
            z1[8] = c12; z1[9] = c13;
        } else {
            atomicAdd(z1 + 0, c02); atomicAdd(z1 + 1, c03);
            atomicAdd(z1 + 8, c12); atomicAdd(z1 + 9, c13);
        }
    }
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_mma_m64_mul1_k(
    const half* __restrict__ x_rot,
    const uint16_t* __restrict__ trellis,
    const half* __restrict__ svh,
    const half* __restrict__ gate_in,
    const half* __restrict__ next_suh,
    half* __restrict__ y_out,
    float* __restrict__ z_f32,
    int M,
    int K,
    int N,
    int split_k,
    float wht_scale,
    float swiglu_limit,
    int add_to_y
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int m_block = blockIdx.y;
    int sk = blockIdx.z;
    int m_base = m_block * 64;
    int K_tiles = K >> 4;
    int N_tiles = N >> 4;
    if (m_base >= M) return;

    int kt_start = (K_tiles * sk) / split_k;
    int kt_end = (K_tiles * (sk + 1)) / split_k;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];
    __shared__ alignas(16) half s_x[64][16];
    __shared__ alignas(16) float s_z[16][128];

    float c0[4][4];
    float c1[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            c0[r][j] = 0.0f;
            c1[r][j] = 0.0f;
        }
    }

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;
    int ld_row = t >> 2;
    int ld_col = (t & 3) * 4;
    int ld_m = m_base + ld_row;

    constexpr int PACKED_U4 = 2 * BITS;
    uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);
    if (kt_start < kt_end) {
        int kb0 = kt_start * 16;
        if (ld_m < M) {
            *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = *reinterpret_cast<const uint2*>(x_rot + (size_t)ld_m * K + kb0 + ld_col);
        } else {
            *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = make_uint2(0u, 0u);
        }
        if (n_tile < N_tiles && lane < PACKED_U4) {
            const uint4* g0 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt_start * N_tiles + n_tile) * PACKED_U16);
            s_u4[lane] = __ldg(g0 + lane);
        }
        __syncthreads();
        for (int kt = kt_start; kt < kt_end; ++kt) {
            uint2 next_x = make_uint2(0u, 0u);
            uint4 next_g;
            if (kt + 1 < kt_end) {
                int kbn = (kt + 1) * 16;
                if (ld_m < M) next_x = *reinterpret_cast<const uint2*>(x_rot + (size_t)ld_m * K + kbn + ld_col);
                if (n_tile < N_tiles && lane < PACKED_U4) {
                    const uint4* gn = reinterpret_cast<const uint4*>(trellis + ((size_t)(kt + 1) * N_tiles + n_tile) * PACKED_U16);
                    next_g = __ldg(gn + lane);
                }
            }
            if (n_tile < N_tiles) {
                uint32_t b00, b01, b10, b11;
                mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int r0 = r * 16 + group_id;
                    int r1 = r0 + 8;
                    uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_x[r0][k_off0]);
                    uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_x[r0][k_off1]);
                    uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_x[r1][k_off0]);
                    uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_x[r1][k_off1]);
                    mma_m16n8k16_f32(c0[r][0], c0[r][1], c0[r][2], c0[r][3], a0, a1, a2, a3, b00, b01);
                    mma_m16n8k16_f32(c1[r][0], c1[r][1], c1[r][2], c1[r][3], a0, a1, a2, a3, b10, b11);
                }
            }
            if (kt + 1 < kt_end) {
                __syncthreads();
                *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = next_x;
                if (n_tile < N_tiles && lane < PACKED_U4) s_u4[lane] = next_g;
                __syncthreads();
            }
        }
    }

    if (split_k == 1 && y_out != nullptr) {
        int col = blockIdx.x * 128 + lane * 4;
        float sv0 = 1.0f, sv1 = 1.0f, sv2 = 1.0f, sv3 = 1.0f;
        if (svh) {
            uint2 sq = __ldg(reinterpret_cast<const uint2*>(svh + col));
            float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
            float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
            sv0 = s01.x; sv1 = s01.y; sv2 = s23.x; sv3 = s23.y;
        }
        int w_col = warp * 16 + tid_in_grp * 2;
        for (int r = 0; r < 4; ++r) {
            if (m_base + r * 16 >= M) break;
            s_z[group_id][w_col + 0] = c0[r][0]; s_z[group_id][w_col + 1] = c0[r][1];
            s_z[group_id][w_col + 8] = c1[r][0]; s_z[group_id][w_col + 9] = c1[r][1];
            s_z[group_id + 8][w_col + 0] = c0[r][2]; s_z[group_id + 8][w_col + 1] = c0[r][3];
            s_z[group_id + 8][w_col + 8] = c1[r][2]; s_z[group_id + 8][w_col + 9] = c1[r][3];
            __syncthreads();
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                int lr = warp + sub * 8;
                int mr = m_base + r * 16 + lr;
                if (mr < M) {
                    size_t off = (size_t)mr * N + col;
                    float4 zv = *reinterpret_cast<const float4*>(&s_z[lr][lane * 4]);
                    float v0 = zv.x, v1 = zv.y, v2 = zv.z, v3 = zv.w;
                    if (wht_scale > 0.0f) mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
                    v0 *= sv0; v1 *= sv1; v2 *= sv2; v3 *= sv3;
                    if (gate_in) {
                        uint2 gq = *reinterpret_cast<const uint2*>(gate_in + off);
                        float2 g01 = __half22float2(*reinterpret_cast<const half2*>(&gq.x));
                        float2 g23 = __half22float2(*reinterpret_cast<const half2*>(&gq.y));
                        auto act = [&](float g, float u) -> float {
                            float gc = fminf(g, swiglu_limit);
                            float uc = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
                            return (gc / (1.0f + expf(-gc))) * uc;
                        };
                        v0 = act(g01.x, v0); v1 = act(g01.y, v1); v2 = act(g23.x, v2); v3 = act(g23.y, v3);
                        if (next_suh) {
                            uint2 nq = __ldg(reinterpret_cast<const uint2*>(next_suh + col));
                            float2 n01 = __half22float2(*reinterpret_cast<const half2*>(&nq.x));
                            float2 n23 = __half22float2(*reinterpret_cast<const half2*>(&nq.y));
                            v0 *= n01.x; v1 *= n01.y; v2 *= n23.x; v3 *= n23.y;
                            mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
                        }
                    } else if (add_to_y) {
                        uint2 yq = *reinterpret_cast<const uint2*>(y_out + off);
                        float2 y01 = __half22float2(*reinterpret_cast<const half2*>(&yq.x));
                        float2 y23 = __half22float2(*reinterpret_cast<const half2*>(&yq.y));
                        v0 += y01.x; v1 += y01.y; v2 += y23.x; v3 += y23.y;
                    }
                    half2 h01 = __float22half2_rn(make_float2(fmaxf(fminf(v0, 65000.0f), -65000.0f), fmaxf(fminf(v1, 65000.0f), -65000.0f)));
                    half2 h23 = __float22half2_rn(make_float2(fmaxf(fminf(v2, 65000.0f), -65000.0f), fmaxf(fminf(v3, 65000.0f), -65000.0f)));
                    *reinterpret_cast<uint2*>(y_out + off) = make_uint2(*reinterpret_cast<const uint32_t*>(&h01), *reinterpret_cast<const uint32_t*>(&h23));
                }
            }
            __syncthreads();
        }
        return;
    }

    if (n_tile >= N_tiles) return;

    int col_base = n_tile * 16 + tid_in_grp * 2;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int m0 = m_base + r * 16 + group_id;
        int m1 = m0 + 8;
        if (m0 < M) {
            float* z0 = z_f32 + (size_t)m0 * N + col_base;
            if (split_k == 1) {
                z0[0] = c0[r][0]; z0[1] = c0[r][1];
                z0[8] = c1[r][0]; z0[9] = c1[r][1];
            } else {
                atomicAdd(z0 + 0, c0[r][0]); atomicAdd(z0 + 1, c0[r][1]);
                atomicAdd(z0 + 8, c1[r][0]); atomicAdd(z0 + 9, c1[r][1]);
            }
        }
        if (m1 < M) {
            float* z1 = z_f32 + (size_t)m1 * N + col_base;
            if (split_k == 1) {
                z1[0] = c0[r][2]; z1[1] = c0[r][3];
                z1[8] = c1[r][2]; z1[9] = c1[r][3];
            } else {
                atomicAdd(z1 + 0, c0[r][2]); atomicAdd(z1 + 1, c0[r][3]);
                atomicAdd(z1 + 8, c1[r][2]); atomicAdd(z1 + 9, c1[r][3]);
            }
        }
    }
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_moe_mma_m16_mul1_k(
    const half* __restrict__ x_in,
    const uint16_t* const* __restrict__ trellis_ptrs,
    const half* const* __restrict__ suh_ptrs,
    const int* __restrict__ expert_ids,
    float* __restrict__ z_f32,
    int top_k,
    int K,
    int N,
    int split_k,
    float wht_scale,
    int x_per_expert,
    int expert_start,
    int n_local_experts
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int pair_idx = blockIdx.y;
    int sk = blockIdx.z;
    int eid = expert_ids[pair_idx];
    int local_eid = eid - expert_start;
    if (local_eid < 0 || local_eid >= n_local_experts) return;
    const uint16_t* trellis = trellis_ptrs[local_eid];
    const half* suh = suh_ptrs ? suh_ptrs[local_eid] : nullptr;
    int mr = pair_idx / top_k;
    const half* x_src = x_per_expert ? (x_in + (size_t)pair_idx * K) : (x_in + (size_t)mr * K);

    int K_blocks128 = K >> 7;
    int kb_start = (K_blocks128 * sk) / split_k;
    int kb_end = (K_blocks128 * (sk + 1)) / split_k;
    int N_tiles = N >> 4;
    if (n_tile >= N_tiles) return;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];
    __shared__ half s_x128[128];

    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;

    for (int kb_idx = kb_start; kb_idx < kb_end; ++kb_idx) {
        int kb = kb_idx * 128;
        if (warp == 0) {
            int base = kb + lane * 4;
            float v0 = __half2float(x_src[base + 0]) * (suh ? __half2float(suh[base + 0]) : 1.0f);
            float v1 = __half2float(x_src[base + 1]) * (suh ? __half2float(suh[base + 1]) : 1.0f);
            float v2 = __half2float(x_src[base + 2]) * (suh ? __half2float(suh[base + 2]) : 1.0f);
            float v3 = __half2float(x_src[base + 3]) * (suh ? __half2float(suh[base + 3]) : 1.0f);
            mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
            s_x128[lane * 4 + 0] = __float2half(v0);
            s_x128[lane * 4 + 1] = __float2half(v1);
            s_x128[lane * 4 + 2] = __float2half(v2);
            s_x128[lane * 4 + 3] = __float2half(v3);
        }
        __syncthreads();

        int kt0 = kb >> 4;
        constexpr int PACKED_U4 = 2 * BITS;
        uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            int kt = kt0 + sub;
            const uint4* g_u4 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
            if (lane < PACKED_U4) {
                s_u4[lane] = __ldg(g_u4 + lane);
            }
            __syncwarp();

            uint32_t b00, b01, b10, b11;
            mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

            uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
            if (group_id == 0) {
                const half* r0 = &s_x128[sub * 16];
                a0 = *reinterpret_cast<const uint32_t*>(r0 + k_off0);
                a2 = *reinterpret_cast<const uint32_t*>(r0 + k_off1);
            }
            mma_m16n8k16_f32(c00, c01, c02, c03, a0, a1, a2, a3, b00, b01);
            mma_m16n8k16_f32(c10, c11, c12, c13, a0, a1, a2, a3, b10, b11);
            __syncwarp();
        }
        __syncthreads();
    }

    int col_base = n_tile * 16 + tid_in_grp * 2;
    if (group_id == 0) {
        float* z0 = z_f32 + (size_t)pair_idx * N + col_base;
        if (split_k == 1) {
            z0[0] = c00; z0[1] = c01; z0[8] = c10; z0[9] = c11;
        } else {
            atomicAdd(z0 + 0, c00); atomicAdd(z0 + 1, c01);
            atomicAdd(z0 + 8, c10); atomicAdd(z0 + 9, c11);
        }
    }
}

__global__ void moe_wht128_post_f32_to_f16_k(
    const float* __restrict__ z_f32,
    const half* const* __restrict__ svh_ptrs,
    const half* const* __restrict__ next_suh_ptrs,
    const int* __restrict__ expert_ids,
    const half* __restrict__ gate_in,
    half* __restrict__ y_out,
    int N,
    float scale,
    float swiglu_limit,
    int expert_start,
    int n_local_experts
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int blk = blockIdx.x * 4 + warp;
    if (blk * 128 >= N) return;
    int pair_idx = blockIdx.y;
    int col = blk * 128 + lane * 4;
    size_t off = (size_t)pair_idx * N + col;
    int eid = expert_ids[pair_idx];
    int local_eid = eid - expert_start;
    if (local_eid < 0 || local_eid >= n_local_experts) {
        *reinterpret_cast<uint2*>(y_out + off) = make_uint2(0u, 0u);
        return;
    }
    const half* svh = svh_ptrs ? svh_ptrs[local_eid] : nullptr;
    float4 zv = *reinterpret_cast<const float4*>(z_f32 + off);
    float v0 = zv.x, v1 = zv.y, v2 = zv.z, v3 = zv.w;
    if (scale > 0.0f) {
        mma_warp_wht128_f32(v0, v1, v2, v3, lane, scale);
    }
    if (svh) {
        uint2 sv = *reinterpret_cast<const uint2*>(svh + col);
        float2 s0 = __half22float2(*reinterpret_cast<const half2*>(&sv.x));
        float2 s1 = __half22float2(*reinterpret_cast<const half2*>(&sv.y));
        v0 *= s0.x;
        v1 *= s0.y;
        v2 *= s1.x;
        v3 *= s1.y;
    }
    if (gate_in) {
        uint2 gv = *reinterpret_cast<const uint2*>(gate_in + off);
        float2 g0 = __half22float2(*reinterpret_cast<const half2*>(&gv.x));
        float2 g1 = __half22float2(*reinterpret_cast<const half2*>(&gv.y));
        auto sw = [&](float g, float u) -> float {
            if (g > swiglu_limit) g = swiglu_limit;
            if (u > swiglu_limit) u = swiglu_limit;
            if (u < -swiglu_limit) u = -swiglu_limit;
            return (g / (1.0f + expf(-g))) * u;
        };
        v0 = sw(g0.x, v0);
        v1 = sw(g0.y, v1);
        v2 = sw(g1.x, v2);
        v3 = sw(g1.y, v3);
    }
    if (next_suh_ptrs) {
        const half* nsuh = next_suh_ptrs[local_eid];
        if (nsuh) {
            uint2 nq = __ldg(reinterpret_cast<const uint2*>(nsuh + col));
            float2 n0 = __half22float2(*reinterpret_cast<const half2*>(&nq.x));
            float2 n1 = __half22float2(*reinterpret_cast<const half2*>(&nq.y));
            v0 *= n0.x; v1 *= n0.y; v2 *= n1.x; v3 *= n1.y;
        }
        mma_warp_wht128_f32(v0, v1, v2, v3, lane, scale);
    }
    half2 h0 = __float22half2_rn(make_float2(v0, v1));
    half2 h1 = __float22half2_rn(make_float2(v2, v3));
    *reinterpret_cast<uint2*>(y_out + off) = make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
}

__global__ void moe_build_groups_k(
    const int* __restrict__ expert_ids,
    int pairs,
    int expert_start,
    int n_local_experts,
    int group_cap,
    int* __restrict__ sorted_pairs,
    int* __restrict__ grp_eids,
    int* __restrict__ grp_offs,
    int* __restrict__ grp_lens,
    int* __restrict__ num_groups
) {
    __shared__ int s_counts[256];
    __shared__ int s_heads[256];
    int t = threadIdx.x;
    if (t < n_local_experts) {
        s_counts[t] = 0;
    }
    __syncthreads();
    for (int p = t; p < pairs; p += blockDim.x) {
        int le = expert_ids[p] - expert_start;
        if (le >= 0 && le < n_local_experts) {
            atomicAdd(&s_counts[le], 1);
        }
    }
    __syncthreads();
    if (t == 0) {
        int cur = 0;
        int ng = 0;
        for (int e = 0; e < n_local_experts; ++e) {
            int c = s_counts[e];
            s_heads[e] = cur;
            if (c > 0) {
                int rem = c;
                int off = cur;
                while (rem > 0) {
                    int take = (rem > group_cap) ? group_cap : rem;
                    grp_eids[ng] = e;
                    grp_offs[ng] = off;
                    grp_lens[ng] = take;
                    ng += 1;
                    off += take;
                    rem -= take;
                }
                cur += c;
            }
        }
        num_groups[0] = ng;
        num_groups[1] = cur;
    }
    __syncthreads();
    for (int p = t; p < pairs; p += blockDim.x) {
        int le = expert_ids[p] - expert_start;
        if (le >= 0 && le < n_local_experts) {
            int pos = atomicAdd(&s_heads[le], 1);
            sorted_pairs[pos] = p;
        }
    }
}

__global__ void moe_wht128_pre_mul_sorted_f16_k(
    const half* __restrict__ x_in,
    const half* const* __restrict__ suh_ptrs,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ num_groups,
    half* __restrict__ x_rot_out,
    int top_k,
    int K,
    float wht_scale,
    int x_per_expert,
    int expert_start
) {
    int pos = blockIdx.y;
    if (pos >= num_groups[1]) return;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int kb = (blockIdx.x * 4 + warp) * 128;
    if (kb >= K) return;
    int pid = sorted_pairs[pos];
    int local_eid = expert_ids[pid] - expert_start;
    int mr = pid / top_k;
    const half* x_src = x_per_expert ? (x_in + (size_t)pid * K) : (x_in + (size_t)mr * K);
    const half* suh = suh_ptrs ? suh_ptrs[local_eid] : nullptr;
    int base = kb + lane * 4;
    uint2 xq = *reinterpret_cast<const uint2*>(x_src + base);
    float2 x01 = __half22float2(*reinterpret_cast<const half2*>(&xq.x));
    float2 x23 = __half22float2(*reinterpret_cast<const half2*>(&xq.y));
    float v0 = x01.x, v1 = x01.y, v2 = x23.x, v3 = x23.y;
    if (suh) {
        uint2 sq = __ldg(reinterpret_cast<const uint2*>(suh + base));
        float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
        float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
        v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
    }
    mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
    half2 h01 = __float22half2_rn(make_float2(v0, v1));
    half2 h23 = __float22half2_rn(make_float2(v2, v3));
    uint2 oq;
    oq.x = *reinterpret_cast<const uint32_t*>(&h01);
    oq.y = *reinterpret_cast<const uint32_t*>(&h23);
    *reinterpret_cast<uint2*>(x_rot_out + (size_t)pos * K + base) = oq;
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_moe_mma_grouped_m64_prerot_k(
    const half* __restrict__ x_rot,
    const uint16_t* const* __restrict__ trellis_ptrs,
    const half* const* __restrict__ svh_ptrs,
    const half* const* __restrict__ next_suh_ptrs,
    const half* __restrict__ gate_in,
    half* __restrict__ y_out,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ grp_eids,
    const int* __restrict__ grp_offs,
    const int* __restrict__ grp_lens,
    const int* __restrict__ num_groups,
    float* __restrict__ z_f32,
    int K,
    int N,
    int split_k,
    float wht_scale,
    float swiglu_limit,
    int x_is_sorted,
    int y_is_sorted
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    constexpr int PACKED_U4 = 2 * BITS;
    int g = blockIdx.y;
    if (g >= num_groups[0]) return;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int sk = blockIdx.z;
    int N_tiles = N >> 4;
    int K_tiles = K >> 4;

    int local_eid = grp_eids[g];
    int g_off = grp_offs[g];
    int g_len = grp_lens[g];
    const uint16_t* trellis = trellis_ptrs[local_eid];

    int kt_start = (K_tiles * sk) / split_k;
    int kt_end = (K_tiles * (sk + 1)) / split_k;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];
    __shared__ alignas(16) half s_x[64][16];
    __shared__ alignas(16) float s_z[16][128];
    __shared__ int s_pair_ids[64];
    if (t < g_len) {
        s_pair_ids[t] = sorted_pairs[g_off + t];
    }
    __syncthreads();

    float c0[4][4];
    float c1[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            c0[r][j] = 0.0f;
            c1[r][j] = 0.0f;
        }
    }

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;
    int ld_row = t >> 2;
    int ld_col = (t & 3) * 4;
    int ld_idx = (ld_row < g_len) ? (x_is_sorted ? (g_off + ld_row) : s_pair_ids[ld_row]) : -1;
    uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);

    if (kt_start < kt_end) {
        int kb0 = kt_start * 16;
        if (ld_idx >= 0) {
            *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = *reinterpret_cast<const uint2*>(x_rot + (size_t)ld_idx * K + kb0 + ld_col);
        } else {
            *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = make_uint2(0u, 0u);
        }
        if (n_tile < N_tiles && lane < PACKED_U4) {
            const uint4* g0 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt_start * N_tiles + n_tile) * PACKED_U16);
            s_u4[lane] = __ldg(g0 + lane);
        }
        __syncthreads();
        for (int kt = kt_start; kt < kt_end; ++kt) {
            uint2 next_x = make_uint2(0u, 0u);
            uint4 next_g;
            if (kt + 1 < kt_end) {
                int kbn = (kt + 1) * 16;
                if (ld_idx >= 0) next_x = *reinterpret_cast<const uint2*>(x_rot + (size_t)ld_idx * K + kbn + ld_col);
                if (n_tile < N_tiles && lane < PACKED_U4) {
                    const uint4* gn = reinterpret_cast<const uint4*>(trellis + ((size_t)(kt + 1) * N_tiles + n_tile) * PACKED_U16);
                    next_g = __ldg(gn + lane);
                }
            }
            if (n_tile < N_tiles) {
                uint32_t b00, b01, b10, b11;
                mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int r0 = r * 16 + group_id;
                    int r1 = r0 + 8;
                    uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_x[r0][k_off0]);
                    uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_x[r0][k_off1]);
                    uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_x[r1][k_off0]);
                    uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_x[r1][k_off1]);
                    mma_m16n8k16_f32(c0[r][0], c0[r][1], c0[r][2], c0[r][3], a0, a1, a2, a3, b00, b01);
                    mma_m16n8k16_f32(c1[r][0], c1[r][1], c1[r][2], c1[r][3], a0, a1, a2, a3, b10, b11);
                }
            }
            if (kt + 1 < kt_end) {
                __syncthreads();
                *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = next_x;
                if (n_tile < N_tiles && lane < PACKED_U4) s_u4[lane] = next_g;
                __syncthreads();
            }
        }
    }

    if (split_k == 1 && y_out != nullptr) {
        int col = blockIdx.x * 128 + lane * 4;
        float sv0 = 1.0f, sv1 = 1.0f, sv2 = 1.0f, sv3 = 1.0f;
        if (svh_ptrs && svh_ptrs[local_eid]) {
            uint2 sv = __ldg(reinterpret_cast<const uint2*>(svh_ptrs[local_eid] + col));
            float2 s0 = __half22float2(*reinterpret_cast<const half2*>(&sv.x));
            float2 s1 = __half22float2(*reinterpret_cast<const half2*>(&sv.y));
            sv0 = s0.x; sv1 = s0.y; sv2 = s1.x; sv3 = s1.y;
        }
        float ns0 = 1.0f, ns1 = 1.0f, ns2 = 1.0f, ns3 = 1.0f;
        if (next_suh_ptrs && next_suh_ptrs[local_eid]) {
            uint2 nq = __ldg(reinterpret_cast<const uint2*>(next_suh_ptrs[local_eid] + col));
            float2 n0 = __half22float2(*reinterpret_cast<const half2*>(&nq.x));
            float2 n1 = __half22float2(*reinterpret_cast<const half2*>(&nq.y));
            ns0 = n0.x; ns1 = n0.y; ns2 = n1.x; ns3 = n1.y;
        }
        int w_col = warp * 16 + tid_in_grp * 2;
        for (int r = 0; r < 4; ++r) {
            if (r * 16 >= g_len) break;
            s_z[group_id][w_col + 0] = c0[r][0];
            s_z[group_id][w_col + 1] = c0[r][1];
            s_z[group_id][w_col + 8] = c1[r][0];
            s_z[group_id][w_col + 9] = c1[r][1];
            s_z[group_id + 8][w_col + 0] = c0[r][2];
            s_z[group_id + 8][w_col + 1] = c0[r][3];
            s_z[group_id + 8][w_col + 8] = c1[r][2];
            s_z[group_id + 8][w_col + 9] = c1[r][3];
            __syncthreads();
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                int lr = warp + sub * 8;
                int row = r * 16 + lr;
                if (row < g_len) {
                    int out_idx = y_is_sorted ? (g_off + row) : s_pair_ids[row];
                    size_t off = (size_t)out_idx * N + col;
                    float4 zv = *reinterpret_cast<const float4*>(&s_z[lr][lane * 4]);
                    float v0 = zv.x, v1 = zv.y, v2 = zv.z, v3 = zv.w;
                    if (wht_scale > 0.0f) {
                        mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
                    }
                    v0 *= sv0; v1 *= sv1; v2 *= sv2; v3 *= sv3;
                    if (gate_in) {
                        uint2 gv = *reinterpret_cast<const uint2*>(gate_in + off);
                        float2 g0 = __half22float2(*reinterpret_cast<const half2*>(&gv.x));
                        float2 g1 = __half22float2(*reinterpret_cast<const half2*>(&gv.y));
                        auto sw = [&](float g_v, float u_v) -> float {
                            if (g_v > swiglu_limit) g_v = swiglu_limit;
                            if (u_v > swiglu_limit) u_v = swiglu_limit;
                            if (u_v < -swiglu_limit) u_v = -swiglu_limit;
                            return (g_v / (1.0f + expf(-g_v))) * u_v;
                        };
                        v0 = sw(g0.x, v0);
                        v1 = sw(g0.y, v1);
                        v2 = sw(g1.x, v2);
                        v3 = sw(g1.y, v3);
                    }
                    if (next_suh_ptrs) {
                        v0 *= ns0; v1 *= ns1; v2 *= ns2; v3 *= ns3;
                        mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
                    }
                    half2 h0 = __float22half2_rn(make_float2(v0, v1));
                    half2 h1 = __float22half2_rn(make_float2(v2, v3));
                    *reinterpret_cast<uint2*>(y_out + off) = make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
                }
            }
            __syncthreads();
        }
        return;
    }

    if (n_tile >= N_tiles) return;
    int col_base = n_tile * 16 + tid_in_grp * 2;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int row0 = r * 16 + group_id;
        int row1 = row0 + 8;
        if (row0 < g_len) {
            int pid0 = s_pair_ids[row0];
            float* z0 = z_f32 + (size_t)pid0 * N + col_base;
            if (split_k == 1) {
                z0[0] = c0[r][0]; z0[1] = c0[r][1]; z0[8] = c1[r][0]; z0[9] = c1[r][1];
            } else {
                atomicAdd(z0 + 0, c0[r][0]); atomicAdd(z0 + 1, c0[r][1]);
                atomicAdd(z0 + 8, c1[r][0]); atomicAdd(z0 + 9, c1[r][1]);
            }
        }
        if (row1 < g_len) {
            int pid1 = s_pair_ids[row1];
            float* z1 = z_f32 + (size_t)pid1 * N + col_base;
            if (split_k == 1) {
                z1[0] = c0[r][2]; z1[1] = c0[r][3]; z1[8] = c1[r][2]; z1[9] = c1[r][3];
            } else {
                atomicAdd(z1 + 0, c0[r][2]); atomicAdd(z1 + 1, c0[r][3]);
                atomicAdd(z1 + 8, c1[r][2]); atomicAdd(z1 + 9, c1[r][3]);
            }
        }
    }
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_moe_mma_grouped_m16_mul1_k(
    const half* __restrict__ x_in,
    const uint16_t* const* __restrict__ trellis_ptrs,
    const half* const* __restrict__ suh_ptrs,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ grp_eids,
    const int* __restrict__ grp_offs,
    const int* __restrict__ grp_lens,
    const int* __restrict__ num_groups,
    float* __restrict__ z_f32,
    int top_k,
    int K,
    int N,
    int split_k,
    float wht_scale,
    int x_per_expert
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int g = blockIdx.y;
    if (g >= num_groups[0]) return;
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int sk = blockIdx.z;
    int N_tiles = N >> 4;
    if (n_tile >= N_tiles) return;

    int local_eid = grp_eids[g];
    int g_off = grp_offs[g];
    int g_len = grp_lens[g];
    const uint16_t* trellis = trellis_ptrs[local_eid];
    const half* suh = suh_ptrs ? suh_ptrs[local_eid] : nullptr;

    int K_blocks128 = K >> 7;
    int kb_start = (K_blocks128 * sk) / split_k;
    int kb_end = (K_blocks128 * (sk + 1)) / split_k;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];
    __shared__ alignas(16) half s_x128[16][128];
    __shared__ int s_pair_ids[16];
    if (t < g_len) {
        s_pair_ids[t] = sorted_pairs[g_off + t];
    }
    __syncthreads();

    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;

    for (int kb_idx = kb_start; kb_idx < kb_end; ++kb_idx) {
        int kb = kb_idx * 128;
        for (int r = warp; r < g_len; r += 8) {
            int pid = s_pair_ids[r];
            int mr = pid / top_k;
            const half* x_src = x_per_expert ? (x_in + (size_t)pid * K) : (x_in + (size_t)mr * K);
            int base = kb + lane * 4;
            uint2 xq = *reinterpret_cast<const uint2*>(x_src + base);
            float2 x01 = __half22float2(*reinterpret_cast<const half2*>(&xq.x));
            float2 x23 = __half22float2(*reinterpret_cast<const half2*>(&xq.y));
            float v0 = x01.x, v1 = x01.y, v2 = x23.x, v3 = x23.y;
            if (suh) {
                uint2 sq = __ldg(reinterpret_cast<const uint2*>(suh + base));
                float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
                float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
                v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
            }
            mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
            half2 h01 = __float22half2_rn(make_float2(v0, v1));
            half2 h23 = __float22half2_rn(make_float2(v2, v3));
            uint2 oq;
            oq.x = *reinterpret_cast<const uint32_t*>(&h01);
            oq.y = *reinterpret_cast<const uint32_t*>(&h23);
            *reinterpret_cast<uint2*>(&s_x128[r][lane * 4]) = oq;
        }
        __syncthreads();

        int kt0 = kb >> 4;
        constexpr int PACKED_U4 = 2 * BITS;
        uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            int kt = kt0 + sub;
            const uint4* g_u4 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
            if (lane < PACKED_U4) {
                s_u4[lane] = __ldg(g_u4 + lane);
            }
            __syncwarp();

            uint32_t b00, b01, b10, b11;
            mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

            uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
            if (group_id < g_len) {
                const half* r0 = &s_x128[group_id][sub * 16];
                a0 = *reinterpret_cast<const uint32_t*>(r0 + k_off0);
                a2 = *reinterpret_cast<const uint32_t*>(r0 + k_off1);
            }
            if (group_id + 8 < g_len) {
                const half* r1 = &s_x128[group_id + 8][sub * 16];
                a1 = *reinterpret_cast<const uint32_t*>(r1 + k_off0);
                a3 = *reinterpret_cast<const uint32_t*>(r1 + k_off1);
            }
            mma_m16n8k16_f32(c00, c01, c02, c03, a0, a1, a2, a3, b00, b01);
            mma_m16n8k16_f32(c10, c11, c12, c13, a0, a1, a2, a3, b10, b11);
            __syncwarp();
        }
        __syncthreads();
    }

    int col_base = n_tile * 16 + tid_in_grp * 2;
    if (group_id < g_len) {
        int pid0 = s_pair_ids[group_id];
        float* z0 = z_f32 + (size_t)pid0 * N + col_base;
        if (split_k == 1) {
            z0[0] = c00; z0[1] = c01; z0[8] = c10; z0[9] = c11;
        } else {
            atomicAdd(z0 + 0, c00); atomicAdd(z0 + 1, c01);
            atomicAdd(z0 + 8, c10); atomicAdd(z0 + 9, c11);
        }
    }
    if (group_id + 8 < g_len) {
        int pid1 = s_pair_ids[group_id + 8];
        float* z1 = z_f32 + (size_t)pid1 * N + col_base;
        if (split_k == 1) {
            z1[0] = c02; z1[1] = c03; z1[8] = c12; z1[9] = c13;
        } else {
            atomicAdd(z1 + 0, c02); atomicAdd(z1 + 1, c03);
            atomicAdd(z1 + 8, c12); atomicAdd(z1 + 9, c13);
        }
    }
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_batched_group_mma_m16_mul1_k(
    const half* __restrict__ x_in,
    const uint16_t* const* __restrict__ trellis_ptrs,
    const half* const* __restrict__ suh_ptrs,
    float* __restrict__ z_f32,
    int M,
    int G,
    int K,
    int N,
    int split_k,
    float wht_scale
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int g = blockIdx.y % G;
    int mb = blockIdx.y / G;
    int m0 = mb * 16;
    int m_len = M - m0;
    if (m_len <= 0) return;
    if (m_len > 16) m_len = 16;

    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int sk = blockIdx.z;
    int N_tiles = N >> 4;
    if (n_tile >= N_tiles) return;

    const uint16_t* trellis = trellis_ptrs[g];
    const half* suh = suh_ptrs[g];

    int K_blocks128 = K >> 7;
    int kb_start = (K_blocks128 * sk) / split_k;
    int kb_end = (K_blocks128 * (sk + 1)) / split_k;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];
    __shared__ alignas(16) half s_x128[16][128];

    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;

    for (int kb_idx = kb_start; kb_idx < kb_end; ++kb_idx) {
        int kb = kb_idx * 128;
        for (int r = warp; r < m_len; r += 8) {
            const half* x_src = x_in + (size_t)((m0 + r) * G + g) * K;
            int base = kb + lane * 4;
            uint2 xq = *reinterpret_cast<const uint2*>(x_src + base);
            float2 x01 = __half22float2(*reinterpret_cast<const half2*>(&xq.x));
            float2 x23 = __half22float2(*reinterpret_cast<const half2*>(&xq.y));
            uint2 sq = __ldg(reinterpret_cast<const uint2*>(suh + base));
            float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
            float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
            float v0 = x01.x * s01.x, v1 = x01.y * s01.y, v2 = x23.x * s23.x, v3 = x23.y * s23.y;
            mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
            half2 h01 = __float22half2_rn(make_float2(v0, v1));
            half2 h23 = __float22half2_rn(make_float2(v2, v3));
            uint2 oq;
            oq.x = *reinterpret_cast<const uint32_t*>(&h01);
            oq.y = *reinterpret_cast<const uint32_t*>(&h23);
            *reinterpret_cast<uint2*>(&s_x128[r][lane * 4]) = oq;
        }
        __syncthreads();

        int kt0 = kb >> 4;
        constexpr int PACKED_U4 = 2 * BITS;
        uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            int kt = kt0 + sub;
            const uint4* g_u4 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
            if (lane < PACKED_U4) {
                s_u4[lane] = __ldg(g_u4 + lane);
            }
            __syncwarp();

            uint32_t b00, b01, b10, b11;
            mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

            uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
            if (group_id < m_len) {
                const half* r0 = &s_x128[group_id][sub * 16];
                a0 = *reinterpret_cast<const uint32_t*>(r0 + k_off0);
                a2 = *reinterpret_cast<const uint32_t*>(r0 + k_off1);
            }
            if (group_id + 8 < m_len) {
                const half* r1 = &s_x128[group_id + 8][sub * 16];
                a1 = *reinterpret_cast<const uint32_t*>(r1 + k_off0);
                a3 = *reinterpret_cast<const uint32_t*>(r1 + k_off1);
            }
            mma_m16n8k16_f32(c00, c01, c02, c03, a0, a1, a2, a3, b00, b01);
            mma_m16n8k16_f32(c10, c11, c12, c13, a0, a1, a2, a3, b10, b11);
            __syncwarp();
        }
        __syncthreads();
    }

    int col_base = n_tile * 16 + tid_in_grp * 2;
    if (group_id < m_len) {
        float* z0 = z_f32 + (size_t)((m0 + group_id) * G + g) * N + col_base;
        if (split_k == 1) {
            z0[0] = c00; z0[1] = c01; z0[8] = c10; z0[9] = c11;
        } else {
            atomicAdd(z0 + 0, c00); atomicAdd(z0 + 1, c01);
            atomicAdd(z0 + 8, c10); atomicAdd(z0 + 9, c11);
        }
    }
    if (group_id + 8 < m_len) {
        float* z1 = z_f32 + (size_t)((m0 + group_id + 8) * G + g) * N + col_base;
        if (split_k == 1) {
            z1[0] = c02; z1[1] = c03; z1[8] = c12; z1[9] = c13;
        } else {
            atomicAdd(z1 + 0, c02); atomicAdd(z1 + 1, c03);
            atomicAdd(z1 + 8, c12); atomicAdd(z1 + 9, c13);
        }
    }
}

__global__ void batched_group_wht128_pre_mul_f16_k(
    const half* __restrict__ x_in,
    const half* const* __restrict__ suh_ptrs,
    half* __restrict__ x_rot_out,
    int G,
    int K,
    float wht_scale
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int kb = (blockIdx.x * 4 + warp) * 128;
    if (kb >= K) return;
    int row_g = blockIdx.y;
    int g = row_g % G;
    int base = kb + lane * 4;
    const half* x_src = x_in + (size_t)row_g * K;
    const half* suh = suh_ptrs[g];
    uint2 xq = *reinterpret_cast<const uint2*>(x_src + base);
    float2 x01 = __half22float2(*reinterpret_cast<const half2*>(&xq.x));
    float2 x23 = __half22float2(*reinterpret_cast<const half2*>(&xq.y));
    uint2 sq = __ldg(reinterpret_cast<const uint2*>(suh + base));
    float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
    float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
    float v0 = x01.x * s01.x, v1 = x01.y * s01.y, v2 = x23.x * s23.x, v3 = x23.y * s23.y;
    mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
    half2 h01 = __float22half2_rn(make_float2(v0, v1));
    half2 h23 = __float22half2_rn(make_float2(v2, v3));
    uint2 oq;
    oq.x = *reinterpret_cast<const uint32_t*>(&h01);
    oq.y = *reinterpret_cast<const uint32_t*>(&h23);
    *reinterpret_cast<uint2*>(x_rot_out + (size_t)row_g * K + base) = oq;
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_batched_group_mma_m64_prerot_k(
    const half* __restrict__ x_rot,
    const uint16_t* const* __restrict__ trellis_ptrs,
    const half* const* __restrict__ svh_ptrs,
    half* __restrict__ y_out,
    float* __restrict__ z_f32,
    int M,
    int G,
    int K,
    int N,
    int split_k,
    float wht_scale
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    constexpr int PACKED_U4 = 2 * BITS;
    int g = blockIdx.y % G;
    int mb = blockIdx.y / G;
    int m0 = mb * 64;
    int m_len = M - m0;
    if (m_len <= 0) return;
    if (m_len > 64) m_len = 64;

    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int sk = blockIdx.z;
    int N_tiles = N >> 4;
    int K_tiles = K >> 4;

    const uint16_t* trellis = trellis_ptrs[g];
    int kt_start = (K_tiles * sk) / split_k;
    int kt_end = (K_tiles * (sk + 1)) / split_k;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];
    __shared__ alignas(16) half s_x[64][16];
    __shared__ alignas(16) float s_z[16][128];

    float c0[4][4];
    float c1[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            c0[r][j] = 0.0f;
            c1[r][j] = 0.0f;
        }
    }

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;
    int ld_row = t >> 2;
    int ld_col = (t & 3) * 4;
    uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);

    for (int kt = kt_start; kt < kt_end; ++kt) {
        int kb = kt * 16;
        if (ld_row < m_len) {
            const half* x_src = x_rot + (size_t)((m0 + ld_row) * G + g) * K + kb + ld_col;
            *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = *reinterpret_cast<const uint2*>(x_src);
        } else {
            *reinterpret_cast<uint2*>(&s_x[ld_row][ld_col]) = make_uint2(0u, 0u);
        }
        if (n_tile < N_tiles && lane < PACKED_U4) {
            const uint4* g_u4 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
            s_u4[lane] = __ldg(g_u4 + lane);
        }
        __syncthreads();

        if (n_tile < N_tiles) {
            uint32_t b00, b01, b10, b11;
            mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int r0 = r * 16 + group_id;
                int r1 = r0 + 8;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_x[r0][k_off0]);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_x[r0][k_off1]);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_x[r1][k_off0]);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_x[r1][k_off1]);
                mma_m16n8k16_f32(c0[r][0], c0[r][1], c0[r][2], c0[r][3], a0, a1, a2, a3, b00, b01);
                mma_m16n8k16_f32(c1[r][0], c1[r][1], c1[r][2], c1[r][3], a0, a1, a2, a3, b10, b11);
            }
        }
        __syncthreads();
    }

    if (split_k == 1 && y_out != nullptr) {
        int col = blockIdx.x * 128 + lane * 4;
        uint2 sq = __ldg(reinterpret_cast<const uint2*>(svh_ptrs[g] + col));
        float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
        float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
        int w_col = warp * 16 + tid_in_grp * 2;
        for (int r = 0; r < 4; ++r) {
            if (r * 16 >= m_len) break;
            s_z[group_id][w_col + 0] = c0[r][0]; s_z[group_id][w_col + 1] = c0[r][1];
            s_z[group_id][w_col + 8] = c1[r][0]; s_z[group_id][w_col + 9] = c1[r][1];
            s_z[group_id + 8][w_col + 0] = c0[r][2]; s_z[group_id + 8][w_col + 1] = c0[r][3];
            s_z[group_id + 8][w_col + 8] = c1[r][2]; s_z[group_id + 8][w_col + 9] = c1[r][3];
            __syncthreads();
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                int lr = warp + sub * 8;
                int row = r * 16 + lr;
                if (row < m_len) {
                    float4 zv = *reinterpret_cast<const float4*>(&s_z[lr][lane * 4]);
                    float v0 = zv.x, v1 = zv.y, v2 = zv.z, v3 = zv.w;
                    mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
                    v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
                    half2 h01 = __float22half2_rn(make_float2(fminf(fmaxf(v0, -65504.0f), 65504.0f), fminf(fmaxf(v1, -65504.0f), 65504.0f)));
                    half2 h23 = __float22half2_rn(make_float2(fminf(fmaxf(v2, -65504.0f), 65504.0f), fminf(fmaxf(v3, -65504.0f), 65504.0f)));
                    *reinterpret_cast<uint2*>(y_out + (size_t)((m0 + row) * G + g) * N + col) = make_uint2(*reinterpret_cast<const uint32_t*>(&h01), *reinterpret_cast<const uint32_t*>(&h23));
                }
            }
            __syncthreads();
        }
        return;
    }

    if (n_tile >= N_tiles) return;
    int col_base = n_tile * 16 + tid_in_grp * 2;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int r0 = r * 16 + group_id;
        int r1 = r0 + 8;
        if (r0 < m_len) {
            float* z0 = z_f32 + (size_t)((m0 + r0) * G + g) * N + col_base;
            if (split_k == 1) {
                z0[0] = c0[r][0]; z0[1] = c0[r][1]; z0[8] = c1[r][0]; z0[9] = c1[r][1];
            } else {
                atomicAdd(z0 + 0, c0[r][0]); atomicAdd(z0 + 1, c0[r][1]);
                atomicAdd(z0 + 8, c1[r][0]); atomicAdd(z0 + 9, c1[r][1]);
            }
        }
        if (r1 < m_len) {
            float* z1 = z_f32 + (size_t)((m0 + r1) * G + g) * N + col_base;
            if (split_k == 1) {
                z1[0] = c0[r][2]; z1[1] = c0[r][3]; z1[8] = c1[r][2]; z1[9] = c1[r][3];
            } else {
                atomicAdd(z1 + 0, c0[r][2]); atomicAdd(z1 + 1, c0[r][3]);
                atomicAdd(z1 + 8, c1[r][2]); atomicAdd(z1 + 9, c1[r][3]);
            }
        }
    }
}

template <int BITS>
__global__ __launch_bounds__(256)
void exl3_batched_group_mma_m64_mul1_k(
    const half* __restrict__ x_in,
    const uint16_t* const* __restrict__ trellis_ptrs,
    const half* const* __restrict__ suh_ptrs,
    float* __restrict__ z_f32,
    int M,
    int G,
    int K,
    int N,
    int split_k,
    float wht_scale
) {
    constexpr int PACKED_U16 = 16 * BITS;
    constexpr int PACKED_U32 = 8 * BITS;
    int g = blockIdx.y % G;
    int mb = blockIdx.y / G;
    int m0 = mb * 64;
    int m_len = M - m0;
    if (m_len <= 0) return;
    if (m_len > 64) m_len = 64;

    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int n_tile = blockIdx.x * 8 + warp;
    int sk = blockIdx.z;
    int N_tiles = N >> 4;

    const uint16_t* trellis = trellis_ptrs[g];
    const half* suh = suh_ptrs[g];

    int K_blocks128 = K >> 7;
    int kb_start = (K_blocks128 * sk) / split_k;
    int kb_end = (K_blocks128 * (sk + 1)) / split_k;

    __shared__ alignas(16) uint32_t s_packed[8][PACKED_U32];
    __shared__ alignas(16) half s_x128[64][128];

    float c0[4][4];
    float c1[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            c0[r][j] = 0.0f;
            c1[r][j] = 0.0f;
        }
    }

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;

    for (int kb_idx = kb_start; kb_idx < kb_end; ++kb_idx) {
        int kb = kb_idx * 128;
        int base = kb + lane * 4;
        uint2 sq = __ldg(reinterpret_cast<const uint2*>(suh + base));
        float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
        float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
        for (int r = warp; r < m_len; r += 8) {
            const half* x_src = x_in + (size_t)((m0 + r) * G + g) * K;
            uint2 xq = *reinterpret_cast<const uint2*>(x_src + base);
            float2 x01 = __half22float2(*reinterpret_cast<const half2*>(&xq.x));
            float2 x23 = __half22float2(*reinterpret_cast<const half2*>(&xq.y));
            float v0 = x01.x * s01.x, v1 = x01.y * s01.y, v2 = x23.x * s23.x, v3 = x23.y * s23.y;
            mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
            half2 h01 = __float22half2_rn(make_float2(v0, v1));
            half2 h23 = __float22half2_rn(make_float2(v2, v3));
            uint2 oq;
            oq.x = *reinterpret_cast<const uint32_t*>(&h01);
            oq.y = *reinterpret_cast<const uint32_t*>(&h23);
            *reinterpret_cast<uint2*>(&s_x128[r][lane * 4]) = oq;
        }
        __syncthreads();

        if (n_tile < N_tiles) {
            int kt0 = kb >> 4;
            constexpr int PACKED_U4 = 2 * BITS;
            uint4* s_u4 = reinterpret_cast<uint4*>(s_packed[warp]);
            #pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                int kt = kt0 + sub;
                const uint4* g_u4 = reinterpret_cast<const uint4*>(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
                if (lane < PACKED_U4) {
                    s_u4[lane] = __ldg(g_u4 + lane);
                }
                __syncwarp();

                uint32_t b00, b01, b10, b11;
                mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int r0 = r * 16 + group_id;
                    int r1 = r0 + 8;
                    uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
                    if (r0 < m_len) {
                        const half* p0 = &s_x128[r0][sub * 16];
                        a0 = *reinterpret_cast<const uint32_t*>(p0 + k_off0);
                        a2 = *reinterpret_cast<const uint32_t*>(p0 + k_off1);
                    }
                    if (r1 < m_len) {
                        const half* p1 = &s_x128[r1][sub * 16];
                        a1 = *reinterpret_cast<const uint32_t*>(p1 + k_off0);
                        a3 = *reinterpret_cast<const uint32_t*>(p1 + k_off1);
                    }
                    mma_m16n8k16_f32(c0[r][0], c0[r][1], c0[r][2], c0[r][3], a0, a1, a2, a3, b00, b01);
                    mma_m16n8k16_f32(c1[r][0], c1[r][1], c1[r][2], c1[r][3], a0, a1, a2, a3, b10, b11);
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    if (n_tile >= N_tiles) return;
    int col_base = n_tile * 16 + tid_in_grp * 2;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int r0 = r * 16 + group_id;
        int r1 = r0 + 8;
        if (r0 < m_len) {
            float* z0 = z_f32 + (size_t)((m0 + r0) * G + g) * N + col_base;
            if (split_k == 1) {
                z0[0] = c0[r][0]; z0[1] = c0[r][1]; z0[8] = c1[r][0]; z0[9] = c1[r][1];
            } else {
                atomicAdd(z0 + 0, c0[r][0]); atomicAdd(z0 + 1, c0[r][1]);
                atomicAdd(z0 + 8, c1[r][0]); atomicAdd(z0 + 9, c1[r][1]);
            }
        }
        if (r1 < m_len) {
            float* z1 = z_f32 + (size_t)((m0 + r1) * G + g) * N + col_base;
            if (split_k == 1) {
                z1[0] = c0[r][2]; z1[1] = c0[r][3]; z1[8] = c1[r][2]; z1[9] = c1[r][3];
            } else {
                atomicAdd(z1 + 0, c0[r][2]); atomicAdd(z1 + 1, c0[r][3]);
                atomicAdd(z1 + 8, c1[r][2]); atomicAdd(z1 + 9, c1[r][3]);
            }
        }
    }
}

__global__ void batched_group_wht128_post_f32_to_f16_k(
    const float* __restrict__ z_f32,
    const half* const* __restrict__ svh_ptrs,
    half* __restrict__ y_out,
    int G,
    int N,
    float wht_scale
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int block_idx = blockIdx.x * 4 + warp;
    if (block_idx * 128 >= N) return;
    int row_g = blockIdx.y;
    int g = row_g % G;
    int base = block_idx * 128 + lane * 4;
    float4 zv = *reinterpret_cast<const float4*>(z_f32 + (size_t)row_g * N + base);
    float v0 = zv.x, v1 = zv.y, v2 = zv.z, v3 = zv.w;
    mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
    uint2 sq = __ldg(reinterpret_cast<const uint2*>(svh_ptrs[g] + base));
    float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sq.x));
    float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sq.y));
    v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
    half2 h01 = __float22half2_rn(make_float2(fminf(fmaxf(v0, -65504.0f), 65504.0f), fminf(fmaxf(v1, -65504.0f), 65504.0f)));
    half2 h23 = __float22half2_rn(make_float2(fminf(fmaxf(v2, -65504.0f), 65504.0f), fminf(fmaxf(v3, -65504.0f), 65504.0f)));
    uint2 out_u2;
    out_u2.x = *reinterpret_cast<const uint32_t*>(&h01);
    out_u2.y = *reinterpret_cast<const uint32_t*>(&h23);
    *reinterpret_cast<uint2*>(y_out + (size_t)row_g * N + base) = out_u2;
}

static int* g_moe_meta = nullptr;

extern "C" {

void rad_cuda_moe_init_meta(void) {
    if (!g_moe_meta) {
        cudaMalloc(&g_moe_meta, 131072 * sizeof(int));
    }
}

void rad_cuda_exl3_moe_mma_f16(
    const half* x_in,
    const uint16_t* const* trellis_ptrs,
    const half* const* suh_ptrs,
    const half* const* svh_ptrs,
    const half* const* next_suh_ptrs,
    const int* expert_ids,
    float* z_scratch_f32,
    const half* gate_in,
    half* y_out,
    half* x_rot_scratch,
    int top_k,
    int M,
    int K,
    int N,
    int bits,
    float wht_scale,
    float swiglu_limit,
    int x_per_expert,
    int expert_start,
    int n_local_experts,
    int reuse_groups,
    cudaStream_t stream
) {
    int n_blocks = N / 128;
    int pairs = M * top_k;
    if (M >= 2 && pairs <= 32768 && n_local_experts <= 256) {
        if (!g_moe_meta) {
            cudaMalloc(&g_moe_meta, 131072 * sizeof(int));
        }
        int use_m64 = (M >= 512 && x_rot_scratch != nullptr);
        int gcap = use_m64 ? 64 : 16;
        int* d_sorted = g_moe_meta;
        int* d_eids = g_moe_meta + 32768;
        int* d_offs = g_moe_meta + 65536;
        int* d_lens = g_moe_meta + 98304;
        int* d_ng = g_moe_meta + 131070;
        if (!reuse_groups) {
            moe_build_groups_k<<<1, 256, 0, stream>>>(
                expert_ids, pairs, expert_start, n_local_experts, gcap,
                d_sorted, d_eids, d_offs, d_lens, d_ng
            );
        }
        int max_groups = (pairs < n_local_experts + (pairs / gcap) + 1) ? pairs : (n_local_experts + (pairs / gcap) + 1);
        int est_active = (pairs / 2 < n_local_experts) ? (pairs / 2) : n_local_experts;
        if (est_active < 1) est_active = 1;
        int split_k = (n_blocks * est_active < 64) ? 4 : ((n_blocks * est_active < 256 && M <= 8) ? 2 : ((n_blocks * est_active < 128) ? 2 : 1));
        if (K / 128 < split_k) split_k = K / 128;
        if (split_k < 1) split_k = 1;
        if (split_k > 1) {
            cudaMemsetAsync(z_scratch_f32, 0, (size_t)pairs * N * sizeof(float), stream);
        }
        dim3 grid(n_blocks, max_groups, split_k);
        if (use_m64) {
            int x_is_sorted = 0;
            if (suh_ptrs != nullptr || x_rot_scratch != x_in) {
                dim3 pre_grid((K / 128 + 3) / 4, pairs);
                moe_wht128_pre_mul_sorted_f16_k<<<pre_grid, 128, 0, stream>>>(
                    x_in, suh_ptrs, expert_ids, d_sorted, d_ng, x_rot_scratch, top_k, K, wht_scale, x_per_expert, expert_start
                );
                x_is_sorted = 1;
            } else if (x_per_expert && split_k == 1) {
                x_is_sorted = 1;
            }
            int y_is_sorted = (split_k == 1 && !x_per_expert) ? 1 : 0;
            half* fused_y = (split_k == 1) ? y_out : nullptr;
            switch (bits) {
                case 2: exl3_moe_mma_grouped_m64_prerot_k<2><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, next_suh_ptrs, gate_in, fused_y, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, K, N, split_k, wht_scale, swiglu_limit, x_is_sorted, y_is_sorted); break;
                case 3: exl3_moe_mma_grouped_m64_prerot_k<3><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, next_suh_ptrs, gate_in, fused_y, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, K, N, split_k, wht_scale, swiglu_limit, x_is_sorted, y_is_sorted); break;
                case 4: exl3_moe_mma_grouped_m64_prerot_k<4><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, next_suh_ptrs, gate_in, fused_y, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, K, N, split_k, wht_scale, swiglu_limit, x_is_sorted, y_is_sorted); break;
                case 5: exl3_moe_mma_grouped_m64_prerot_k<5><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, next_suh_ptrs, gate_in, fused_y, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, K, N, split_k, wht_scale, swiglu_limit, x_is_sorted, y_is_sorted); break;
                case 6: exl3_moe_mma_grouped_m64_prerot_k<6><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, next_suh_ptrs, gate_in, fused_y, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, K, N, split_k, wht_scale, swiglu_limit, x_is_sorted, y_is_sorted); break;
                case 8: exl3_moe_mma_grouped_m64_prerot_k<8><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, next_suh_ptrs, gate_in, fused_y, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, K, N, split_k, wht_scale, swiglu_limit, x_is_sorted, y_is_sorted); break;
            }
            if (split_k == 1) {
                return;
            }
        } else {
            switch (bits) {
                case 2: exl3_moe_mma_grouped_m16_mul1_k<2><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
                case 3: exl3_moe_mma_grouped_m16_mul1_k<3><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
                case 4: exl3_moe_mma_grouped_m16_mul1_k<4><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
                case 5: exl3_moe_mma_grouped_m16_mul1_k<5><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
                case 6: exl3_moe_mma_grouped_m16_mul1_k<6><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
                case 8: exl3_moe_mma_grouped_m16_mul1_k<8><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
            }
        }
        dim3 post_grid((n_blocks + 3) / 4, pairs);
        moe_wht128_post_f32_to_f16_k<<<post_grid, 128, 0, stream>>>(z_scratch_f32, svh_ptrs, use_m64 ? next_suh_ptrs : nullptr, expert_ids, gate_in, y_out, N, wht_scale, swiglu_limit, expert_start, n_local_experts);
        return;
    }
    int split_k = (n_blocks * pairs < 128) ? 4 : ((n_blocks * pairs < 256) ? 2 : 1);
    if (K / 128 < split_k) split_k = K / 128;
    if (split_k < 1) split_k = 1;
    if (split_k > 1) {
        cudaMemsetAsync(z_scratch_f32, 0, (size_t)pairs * N * sizeof(float), stream);
    }
    dim3 grid(n_blocks, pairs, split_k);
    switch (bits) {
        case 2: exl3_moe_mma_m16_mul1_k<2><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, expert_ids, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert, expert_start, n_local_experts); break;
        case 3: exl3_moe_mma_m16_mul1_k<3><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, expert_ids, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert, expert_start, n_local_experts); break;
        case 4: exl3_moe_mma_m16_mul1_k<4><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, expert_ids, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert, expert_start, n_local_experts); break;
        case 5: exl3_moe_mma_m16_mul1_k<5><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, expert_ids, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert, expert_start, n_local_experts); break;
        case 6: exl3_moe_mma_m16_mul1_k<6><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, expert_ids, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert, expert_start, n_local_experts); break;
        case 8: exl3_moe_mma_m16_mul1_k<8><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, expert_ids, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert, expert_start, n_local_experts); break;
    }
    dim3 post_grid((n_blocks + 3) / 4, pairs);
    moe_wht128_post_f32_to_f16_k<<<post_grid, 128, 0, stream>>>(z_scratch_f32, svh_ptrs, nullptr, expert_ids, gate_in, y_out, N, wht_scale, swiglu_limit, expert_start, n_local_experts);
}

void rad_cuda_exl3_batched_group_mma_f16(
    const half* x_in,
    const uint16_t* const* trellis_ptrs,
    const half* const* suh_ptrs,
    const half* const* svh_ptrs,
    float* z_scratch_f32,
    half* y_out,
    half* x_rot_scratch,
    int M,
    int G,
    int K,
    int N,
    int bits,
    float wht_scale,
    cudaStream_t stream
) {
    int n_blocks = N / 128;
    int use_m64 = (M >= 32);
    int m_blocks = use_m64 ? ((M + 63) / 64) : ((M + 15) / 16);
    int total_blocks = n_blocks * m_blocks * G;
    int split_k = (total_blocks < 64) ? 4 : ((total_blocks < 128) ? 2 : 1);
    int max_sk = K / 128;
    if (split_k > max_sk) split_k = max_sk;
    if (split_k < 1) split_k = 1;
    if (split_k > 1) {
        cudaMemsetAsync(z_scratch_f32, 0, (size_t)M * G * N * sizeof(float), stream);
    }
    dim3 grid(n_blocks, m_blocks * G, split_k);
    if (use_m64 && M >= 256 && x_rot_scratch) {
        dim3 pre_grid((K / 128 + 3) / 4, M * G);
        batched_group_wht128_pre_mul_f16_k<<<pre_grid, 128, 0, stream>>>(
            x_in, suh_ptrs, x_rot_scratch, G, K, wht_scale
        );
        half* fused_y = (split_k == 1) ? y_out : nullptr;
        switch (bits) {
            case 2: exl3_batched_group_mma_m64_prerot_k<2><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, fused_y, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 3: exl3_batched_group_mma_m64_prerot_k<3><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, fused_y, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 4: exl3_batched_group_mma_m64_prerot_k<4><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, fused_y, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 5: exl3_batched_group_mma_m64_prerot_k<5><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, fused_y, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 6: exl3_batched_group_mma_m64_prerot_k<6><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, fused_y, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 8: exl3_batched_group_mma_m64_prerot_k<8><<<grid, 256, 0, stream>>>(x_rot_scratch, trellis_ptrs, svh_ptrs, fused_y, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
        }
        if (split_k == 1) return;
    } else if (use_m64) {
        switch (bits) {
            case 2: exl3_batched_group_mma_m64_mul1_k<2><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 3: exl3_batched_group_mma_m64_mul1_k<3><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 4: exl3_batched_group_mma_m64_mul1_k<4><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 5: exl3_batched_group_mma_m64_mul1_k<5><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 6: exl3_batched_group_mma_m64_mul1_k<6><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 8: exl3_batched_group_mma_m64_mul1_k<8><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
        }
    } else {
        switch (bits) {
            case 2: exl3_batched_group_mma_m16_mul1_k<2><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 3: exl3_batched_group_mma_m16_mul1_k<3><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 4: exl3_batched_group_mma_m16_mul1_k<4><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 5: exl3_batched_group_mma_m16_mul1_k<5><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 6: exl3_batched_group_mma_m16_mul1_k<6><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
            case 8: exl3_batched_group_mma_m16_mul1_k<8><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, z_scratch_f32, M, G, K, N, split_k, wht_scale); break;
        }
    }
    dim3 post_grid((n_blocks + 3) / 4, M * G);
    batched_group_wht128_post_f32_to_f16_k<<<post_grid, 128, 0, stream>>>(z_scratch_f32, svh_ptrs, y_out, G, N, wht_scale);
}

void rad_cuda_exl3_mma_f16(
    const half* x_rot,
    const uint16_t* trellis,
    const half* svh,
    float* z_scratch_f32,
    half* y,
    int M,
    int K,
    int N,
    int bits,
    float wht_scale,
    int add_to_y,
    cudaStream_t stream
) {
    int n_blocks = N / 128;
    int use_m64 = (M >= 32);
    int m_blocks = use_m64 ? ((M + 63) / 64) : ((M + 15) / 16);
    int total_mn = n_blocks * m_blocks;
    int target_ctas = (M >= 64) ? 48 : 192;
    int split_k = (target_ctas + total_mn - 1) / total_mn;
    int max_sk = K / 128;
    if (split_k > max_sk) split_k = max_sk;
    if (split_k > 16) split_k = 16;
    if (split_k < 1) split_k = 1;
    if (split_k > 1) {
        cudaMemsetAsync(z_scratch_f32, 0, (size_t)M * N * sizeof(float), stream);
    }
    dim3 grid(n_blocks, m_blocks, split_k);
    if (use_m64) {
        half* fused_y = (split_k == 1) ? y : nullptr;
        switch (bits) {
            case 2: exl3_mma_m64_mul1_k<2><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, nullptr, nullptr, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, 0.0f, add_to_y); break;
            case 3: exl3_mma_m64_mul1_k<3><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, nullptr, nullptr, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, 0.0f, add_to_y); break;
            case 4: exl3_mma_m64_mul1_k<4><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, nullptr, nullptr, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, 0.0f, add_to_y); break;
            case 5: exl3_mma_m64_mul1_k<5><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, nullptr, nullptr, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, 0.0f, add_to_y); break;
            case 6: exl3_mma_m64_mul1_k<6><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, nullptr, nullptr, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, 0.0f, add_to_y); break;
            case 8: exl3_mma_m64_mul1_k<8><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, nullptr, nullptr, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, 0.0f, add_to_y); break;
        }
        if (split_k == 1) return;
    } else {
        switch (bits) {
            case 2: exl3_mma_m16_mul1_k<2><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 3: exl3_mma_m16_mul1_k<3><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 4: exl3_mma_m16_mul1_k<4><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 5: exl3_mma_m16_mul1_k<5><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 6: exl3_mma_m16_mul1_k<6><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 8: exl3_mma_m16_mul1_k<8><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
        }
    }
    dim3 post_grid((n_blocks + 3) / 4, M);
    mma_wht128_post_f32_to_f16_k<<<post_grid, 128, 0, stream>>>(z_scratch_f32, svh, nullptr, nullptr, y, N, wht_scale, 0.0f, add_to_y);
}

void rad_cuda_exl3_mma_swiglu_f16(
    const half* x_rot,
    const uint16_t* trellis,
    const half* svh,
    float* z_scratch_f32,
    const half* gate_in,
    const half* next_suh,
    half* y,
    int M,
    int K,
    int N,
    int bits,
    float wht_scale,
    float swiglu_limit,
    cudaStream_t stream
) {
    int n_blocks = N / 128;
    int use_m64 = (M >= 32);
    int m_blocks = use_m64 ? ((M + 63) / 64) : ((M + 15) / 16);
    int total_mn = n_blocks * m_blocks;
    int target_ctas = (M >= 64) ? 48 : 192;
    int split_k = (target_ctas + total_mn - 1) / total_mn;
    int max_sk = K / 128;
    if (split_k > max_sk) split_k = max_sk;
    if (split_k > 16) split_k = 16;
    if (split_k < 1) split_k = 1;
    if (split_k > 1) {
        cudaMemsetAsync(z_scratch_f32, 0, (size_t)M * N * sizeof(float), stream);
    }
    dim3 grid(n_blocks, m_blocks, split_k);
    if (use_m64) {
        half* fused_y = (split_k == 1) ? y : nullptr;
        switch (bits) {
            case 2: exl3_mma_m64_mul1_k<2><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, gate_in, next_suh, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, swiglu_limit, 0); break;
            case 3: exl3_mma_m64_mul1_k<3><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, gate_in, next_suh, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, swiglu_limit, 0); break;
            case 4: exl3_mma_m64_mul1_k<4><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, gate_in, next_suh, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, swiglu_limit, 0); break;
            case 5: exl3_mma_m64_mul1_k<5><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, gate_in, next_suh, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, swiglu_limit, 0); break;
            case 6: exl3_mma_m64_mul1_k<6><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, gate_in, next_suh, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, swiglu_limit, 0); break;
            case 8: exl3_mma_m64_mul1_k<8><<<grid, 256, 0, stream>>>(x_rot, trellis, svh, gate_in, next_suh, fused_y, z_scratch_f32, M, K, N, split_k, wht_scale, swiglu_limit, 0); break;
        }
        if (split_k == 1) return;
    } else {
        switch (bits) {
            case 2: exl3_mma_m16_mul1_k<2><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 3: exl3_mma_m16_mul1_k<3><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 4: exl3_mma_m16_mul1_k<4><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 5: exl3_mma_m16_mul1_k<5><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 6: exl3_mma_m16_mul1_k<6><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 8: exl3_mma_m16_mul1_k<8><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
        }
    }
    dim3 post_grid((n_blocks + 3) / 4, M);
    mma_wht128_post_f32_to_f16_k<<<post_grid, 128, 0, stream>>>(z_scratch_f32, svh, gate_in, next_suh, y, N, wht_scale, swiglu_limit, 0);
}

}
