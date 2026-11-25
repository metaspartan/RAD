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
    uint32_t a = ptr[i0 % WORDS];
    uint32_t b = ptr[i2 % WORDS];
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
        uint32_t a = ptr[i0 % WORDS];
        uint32_t b = ptr[i2 % WORDS];
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
    half* __restrict__ y,
    int N,
    float scale,
    int add_to_y
) {
    int blk = blockIdx.x;
    int row = blockIdx.y;
    int lane = threadIdx.x;
    int col = blk * 128 + lane * 4;
    const float* zr = z_f32 + (size_t)row * N + col;
    float v0 = zr[0], v1 = zr[1], v2 = zr[2], v3 = zr[3];
    if (scale > 0.0f) {
        mma_warp_wht128_f32(v0, v1, v2, v3, lane, scale);
    }
    if (svh) {
        v0 *= __half2float(svh[col + 0]);
        v1 *= __half2float(svh[col + 1]);
        v2 *= __half2float(svh[col + 2]);
        v3 *= __half2float(svh[col + 3]);
    }
    half* yr = y + (size_t)row * N + col;
    if (add_to_y) {
        yr[0] = __float2half(__half2float(yr[0]) + v0);
        yr[1] = __float2half(__half2float(yr[1]) + v1);
        yr[2] = __float2half(__half2float(yr[2]) + v2);
        yr[3] = __float2half(__half2float(yr[3]) + v3);
    } else {
        yr[0] = __float2half(v0);
        yr[1] = __float2half(v1);
        yr[2] = __float2half(v2);
        yr[3] = __float2half(v3);
    }
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

    __shared__ uint32_t s_packed[8][PACKED_U32];

    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;

    int group_id = lane >> 2;
    int tid_in_grp = lane & 3;
    int m0 = m_base + group_id;
    int m1 = m_base + group_id + 8;
    int k_off0 = tid_in_grp * 2;
    int k_off1 = k_off0 + 8;

    for (int kt = kt_start; kt < kt_end; ++kt) {
        const uint32_t* g_tile = (const uint32_t*)(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
        #pragma unroll
        for (int w = lane; w < PACKED_U32; w += 32) {
            s_packed[warp][w] = __ldg(g_tile + w);
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
    int m_base = m_block * 64;
    int K_tiles = K >> 4;
    int N_tiles = N >> 4;
    if (n_tile >= N_tiles || m_base >= M) return;

    int kt_start = (K_tiles * sk) / split_k;
    int kt_end = (K_tiles * (sk + 1)) / split_k;

    __shared__ uint32_t s_packed[8][PACKED_U32];

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

    for (int kt = kt_start; kt < kt_end; ++kt) {
        const uint32_t* g_tile = (const uint32_t*)(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
        #pragma unroll
        for (int w = lane; w < PACKED_U32; w += 32) {
            s_packed[warp][w] = __ldg(g_tile + w);
        }
        __syncwarp();

        uint32_t b00, b01, b10, b11;
        mma_dq8_mul1<BITS>(s_packed[warp], lane * 8, b00, b01, b10, b11);

        int kb = kt * 16;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int m0 = m_base + r * 16 + group_id;
            int m1 = m0 + 8;
            uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
            if (m0 < M) {
                const half* p0 = x_rot + (size_t)m0 * K + kb;
                a0 = *reinterpret_cast<const uint32_t*>(p0 + k_off0);
                a2 = *reinterpret_cast<const uint32_t*>(p0 + k_off1);
            }
            if (m1 < M) {
                const half* p1 = x_rot + (size_t)m1 * K + kb;
                a1 = *reinterpret_cast<const uint32_t*>(p1 + k_off0);
                a3 = *reinterpret_cast<const uint32_t*>(p1 + k_off1);
            }
            mma_m16n8k16_f32(c0[r][0], c0[r][1], c0[r][2], c0[r][3], a0, a1, a2, a3, b00, b01);
            mma_m16n8k16_f32(c1[r][0], c1[r][1], c1[r][2], c1[r][3], a0, a1, a2, a3, b10, b11);
        }
        __syncwarp();
    }

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

    __shared__ uint32_t s_packed[8][PACKED_U32];
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
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            int kt = kt0 + sub;
            const uint32_t* g_tile = (const uint32_t*)(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
            #pragma unroll
            for (int w = lane; w < PACKED_U32; w += 32) {
                s_packed[warp][w] = __ldg(g_tile + w);
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
    const int* __restrict__ expert_ids,
    half* __restrict__ y_out,
    int N,
    float scale,
    int expert_start,
    int n_local_experts
) {
    int blk = blockIdx.x;
    int pair_idx = blockIdx.y;
    int lane = threadIdx.x;
    int col = blk * 128 + lane * 4;
    size_t off = (size_t)pair_idx * N + col;
    int eid = expert_ids[pair_idx];
    int local_eid = eid - expert_start;
    if (local_eid < 0 || local_eid >= n_local_experts) {
        half zero = __float2half(0.0f);
        y_out[off + 0] = zero;
        y_out[off + 1] = zero;
        y_out[off + 2] = zero;
        y_out[off + 3] = zero;
        return;
    }
    const half* svh = svh_ptrs ? svh_ptrs[local_eid] : nullptr;
    float v0 = z_f32[off + 0], v1 = z_f32[off + 1], v2 = z_f32[off + 2], v3 = z_f32[off + 3];
    if (scale > 0.0f) {
        mma_warp_wht128_f32(v0, v1, v2, v3, lane, scale);
    }
    if (svh) {
        v0 *= __half2float(svh[col + 0]);
        v1 *= __half2float(svh[col + 1]);
        v2 *= __half2float(svh[col + 2]);
        v3 *= __half2float(svh[col + 3]);
    }
    y_out[off + 0] = __float2half(v0);
    y_out[off + 1] = __float2half(v1);
    y_out[off + 2] = __float2half(v2);
    y_out[off + 3] = __float2half(v3);
}

__global__ void moe_build_groups_k(
    const int* __restrict__ expert_ids,
    int pairs,
    int expert_start,
    int n_local_experts,
    int* __restrict__ sorted_pairs,
    int* __restrict__ grp_eids,
    int* __restrict__ grp_offs,
    int* __restrict__ grp_lens,
    int* __restrict__ num_groups
) {
    if (threadIdx.x != 0) return;
    int counts[256];
    int heads[256];
    for (int e = 0; e < n_local_experts; ++e) counts[e] = 0;
    for (int p = 0; p < pairs; ++p) {
        int le = expert_ids[p] - expert_start;
        if (le >= 0 && le < n_local_experts) counts[le] += 1;
    }
    int cur = 0;
    int ng = 0;
    for (int e = 0; e < n_local_experts; ++e) {
        int c = counts[e];
        heads[e] = cur;
        if (c > 0) {
            int rem = c;
            int off = cur;
            while (rem > 0) {
                int take = (rem > 16) ? 16 : rem;
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
    for (int p = 0; p < pairs; ++p) {
        int le = expert_ids[p] - expert_start;
        if (le >= 0 && le < n_local_experts) {
            int pos = heads[le]++;
            sorted_pairs[pos] = p;
        }
    }
    num_groups[0] = ng;
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

    __shared__ uint32_t s_packed[8][PACKED_U32];
    __shared__ half s_x128[16][128];
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
            float v0 = __half2float(x_src[base + 0]) * (suh ? __half2float(suh[base + 0]) : 1.0f);
            float v1 = __half2float(x_src[base + 1]) * (suh ? __half2float(suh[base + 1]) : 1.0f);
            float v2 = __half2float(x_src[base + 2]) * (suh ? __half2float(suh[base + 2]) : 1.0f);
            float v3 = __half2float(x_src[base + 3]) * (suh ? __half2float(suh[base + 3]) : 1.0f);
            mma_warp_wht128_f32(v0, v1, v2, v3, lane, wht_scale);
            s_x128[r][lane * 4 + 0] = __float2half(v0);
            s_x128[r][lane * 4 + 1] = __float2half(v1);
            s_x128[r][lane * 4 + 2] = __float2half(v2);
            s_x128[r][lane * 4 + 3] = __float2half(v3);
        }
        __syncthreads();

        int kt0 = kb >> 4;
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            int kt = kt0 + sub;
            const uint32_t* g_tile = (const uint32_t*)(trellis + ((size_t)kt * N_tiles + n_tile) * PACKED_U16);
            #pragma unroll
            for (int w = lane; w < PACKED_U32; w += 32) {
                s_packed[warp][w] = __ldg(g_tile + w);
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

static int* g_moe_meta = nullptr;

extern "C" {

void rad_cuda_exl3_moe_mma_f16(
    const half* x_in,
    const uint16_t* const* trellis_ptrs,
    const half* const* suh_ptrs,
    const half* const* svh_ptrs,
    const int* expert_ids,
    float* z_scratch_f32,
    half* y_out,
    int top_k,
    int M,
    int K,
    int N,
    int bits,
    float wht_scale,
    int x_per_expert,
    int expert_start,
    int n_local_experts,
    cudaStream_t stream
) {
    int n_blocks = N / 128;
    int pairs = M * top_k;
    if (M >= 2 && pairs <= 512 && n_local_experts <= 256) {
        if (!g_moe_meta) {
            cudaMalloc(&g_moe_meta, 2048 * sizeof(int));
        }
        int* d_sorted = g_moe_meta;
        int* d_eids = g_moe_meta + 512;
        int* d_offs = g_moe_meta + 1024;
        int* d_lens = g_moe_meta + 1536;
        int* d_ng = g_moe_meta + 2047;
        moe_build_groups_k<<<1, 1, 0, stream>>>(
            expert_ids, pairs, expert_start, n_local_experts,
            d_sorted, d_eids, d_offs, d_lens, d_ng
        );
        int max_groups = (pairs < n_local_experts + (pairs / 16) + 1) ? pairs : (n_local_experts + (pairs / 16) + 1);
        int split_k = (n_blocks * max_groups < 192) ? 2 : 1;
        if (K / 128 < split_k) split_k = K / 128;
        if (split_k < 1) split_k = 1;
        if (split_k > 1) {
            cudaMemsetAsync(z_scratch_f32, 0, (size_t)pairs * N * sizeof(float), stream);
        }
        dim3 grid(n_blocks, max_groups, split_k);
        switch (bits) {
            case 2: exl3_moe_mma_grouped_m16_mul1_k<2><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
            case 3: exl3_moe_mma_grouped_m16_mul1_k<3><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
            case 4: exl3_moe_mma_grouped_m16_mul1_k<4><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
            case 5: exl3_moe_mma_grouped_m16_mul1_k<5><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
            case 6: exl3_moe_mma_grouped_m16_mul1_k<6><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
            case 8: exl3_moe_mma_grouped_m16_mul1_k<8><<<grid, 256, 0, stream>>>(x_in, trellis_ptrs, suh_ptrs, d_sorted, d_eids, d_offs, d_lens, d_ng, z_scratch_f32, top_k, K, N, split_k, wht_scale, x_per_expert); break;
        }
        dim3 post_grid(n_blocks, pairs);
        moe_wht128_post_f32_to_f16_k<<<post_grid, 32, 0, stream>>>(z_scratch_f32, svh_ptrs, expert_ids, y_out, N, wht_scale, expert_start, n_local_experts);
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
    dim3 post_grid(n_blocks, pairs);
    moe_wht128_post_f32_to_f16_k<<<post_grid, 32, 0, stream>>>(z_scratch_f32, svh_ptrs, expert_ids, y_out, N, wht_scale, expert_start, n_local_experts);
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
    int target_ctas = 192;
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
        switch (bits) {
            case 2: exl3_mma_m64_mul1_k<2><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 3: exl3_mma_m64_mul1_k<3><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 4: exl3_mma_m64_mul1_k<4><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 5: exl3_mma_m64_mul1_k<5><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 6: exl3_mma_m64_mul1_k<6><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
            case 8: exl3_mma_m64_mul1_k<8><<<grid, 256, 0, stream>>>(x_rot, trellis, z_scratch_f32, M, K, N, split_k); break;
        }
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
    dim3 post_grid(n_blocks, M);
    mma_wht128_post_f32_to_f16_k<<<post_grid, 32, 0, stream>>>(z_scratch_f32, svh, y, N, wht_scale, add_to_y);
}

}
