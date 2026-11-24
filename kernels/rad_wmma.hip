#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

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

extern "C" __global__ void gemm_q4_0_wmma64_k(const uint8_t* __restrict__ W,
                                              const w_q8_1_blk* __restrict__ Xq,
                                              float* __restrict__ Y,
                                              int n, int k, int B) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int wm = warp >> 1;
    int wn = warp & 1;
    int blk_row0 = blockIdx.x * 128;
    int blk_t0 = blockIdx.y * 64;
    if (blk_row0 >= n || blk_t0 >= B) return;

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);

    __shared__ uint4 sm_wq[4][128];
    __shared__ float sm_dw[4][128];
    __shared__ int2 sm_xq[4][4][64];
    __shared__ float sm_xd[4][64];
    __shared__ int32_t sm_xs[4][64];

    int r_loc = tid >> 2;
    int kb_load = tid & 3;
    int r_g0 = min(blk_row0 + r_loc, n - 1);
    int r_g1 = min(blk_row0 + r_loc + 64, n - 1);
    const uint8_t* wr0 = W + (size_t)r_g0 * rbytes;
    const uint8_t* wr1 = W + (size_t)r_g1 * rbytes;
    const uint4* qr0 = reinterpret_cast<const uint4*>(wr0);
    const uint4* qr1 = reinterpret_cast<const uint4*>(wr1);
    const __half* dr0 = reinterpret_cast<const __half*>(wr0 + (size_t)nblk * 16);
    const __half* dr1 = reinterpret_cast<const __half*>(wr1 + (size_t)nblk * 16);

    int t_g0 = min(blk_t0 + r_loc, B - 1);
    const w_q8_1_blk* xr0 = Xq + (size_t)t_g0 * nblk;

    int m0 = wm * 32;
    int m1 = m0 + 16;
    int n0 = wn * 32;
    int n1 = n0 + 16;

    float acc00[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc01[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc10[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc11[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int b0 = 0; b0 < nblk; b0 += 4) {
        int b = b0 + kb_load;
        if (b < nblk) {
            sm_wq[kb_load][r_loc] = qr0[b];
            sm_dw[kb_load][r_loc] = __half2float(dr0[b]);
            sm_wq[kb_load][r_loc + 64] = qr1[b];
            sm_dw[kb_load][r_loc + 64] = __half2float(dr1[b]);
            const w_q8_1_blk xb0 = xr0[b];
            sm_xq[kb_load][0][r_loc] = make_int2(xb0.q[0], xb0.q[1]);
            sm_xq[kb_load][1][r_loc] = make_int2(xb0.q[2], xb0.q[3]);
            sm_xq[kb_load][2][r_loc] = make_int2(xb0.q[4], xb0.q[5]);
            sm_xq[kb_load][3][r_loc] = make_int2(xb0.q[6], xb0.q[7]);
            sm_xd[kb_load][r_loc] = xb0.d;
            sm_xs[kb_load][r_loc] = 8 * xb0.s;
        } else {
            sm_wq[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc] = 0.0f;
            sm_wq[kb_load][r_loc + 64] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc + 64] = 0.0f;
            sm_xd[kb_load][r_loc] = 0.0f;
            sm_xs[kb_load][r_loc] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint4 q0 = sm_wq[kb][m0 + col];
            uint32_t w00 = half ? q0.z : q0.x;
            uint32_t w01 = half ? q0.w : q0.y;
            wmma_i2_v a0_lo = {(int32_t)(w00 & 0x0F0F0F0Fu), (int32_t)(w01 & 0x0F0F0F0Fu)};
            wmma_i2_v a0_hi = {(int32_t)((w00 >> 4) & 0x0F0F0F0Fu), (int32_t)((w01 >> 4) & 0x0F0F0F0Fu)};

            uint4 q1 = sm_wq[kb][m1 + col];
            uint32_t w10 = half ? q1.z : q1.x;
            uint32_t w11 = half ? q1.w : q1.y;
            wmma_i2_v a1_lo = {(int32_t)(w10 & 0x0F0F0F0Fu), (int32_t)(w11 & 0x0F0F0F0Fu)};
            wmma_i2_v a1_hi = {(int32_t)((w10 >> 4) & 0x0F0F0F0Fu), (int32_t)((w11 >> 4) & 0x0F0F0F0Fu)};

            int2 x0l = sm_xq[kb][half][n0 + col];
            int2 x0h = sm_xq[kb][2 + half][n0 + col];
            wmma_i2_v b0_lo = {x0l.x, x0l.y};
            wmma_i2_v b0_hi = {x0h.x, x0h.y};

            int2 x1l = sm_xq[kb][half][n1 + col];
            int2 x1h = sm_xq[kb][2 + half][n1 + col];
            wmma_i2_v b1_lo = {x1l.x, x1l.y};
            wmma_i2_v b1_hi = {x1h.x, x1h.y};

            wmma_i8_v c00 = {0, 0, 0, 0, 0, 0, 0, 0};
            wmma_i8_v c01 = {0, 0, 0, 0, 0, 0, 0, 0};
            wmma_i8_v c10 = {0, 0, 0, 0, 0, 0, 0, 0};
            wmma_i8_v c11 = {0, 0, 0, 0, 0, 0, 0, 0};
            c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_lo, true, b0_lo, c00, false);
            c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_hi, true, b0_hi, c00, false);
            c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_lo, true, b1_lo, c01, false);
            c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_hi, true, b1_hi, c01, false);
            c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_lo, true, b0_lo, c10, false);
            c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_hi, true, b0_hi, c10, false);
            c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_lo, true, b1_lo, c11, false);
            c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_hi, true, b1_hi, c11, false);

            float xd0 = sm_xd[kb][n0 + col];
            int32_t xs0 = sm_xs[kb][n0 + col];
            float xd1 = sm_xd[kb][n1 + col];
            int32_t xs1 = sm_xs[kb][n1 + col];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dw0 = sm_dw[kb][m0 + half * 8 + i];
                float dw1 = sm_dw[kb][m1 + half * 8 + i];
                acc00[i] = fmaf(dw0 * xd0, (float)(c00[i] - xs0), acc00[i]);
                acc01[i] = fmaf(dw0 * xd1, (float)(c01[i] - xs1), acc01[i]);
                acc10[i] = fmaf(dw1 * xd0, (float)(c10[i] - xs0), acc10[i]);
                acc11[i] = fmaf(dw1 * xd1, (float)(c11[i] - xs1), acc11[i]);
            }
        }
        __syncthreads();
    }

    int gt0 = blk_t0 + n0 + col;
    if (gt0 < B) {
        float* yt = Y + (size_t)gt0 * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r0 = blk_row0 + m0 + half * 8 + i;
            if (r0 < n) yt[r0] = acc00[i];
            int r1 = blk_row0 + m1 + half * 8 + i;
            if (r1 < n) yt[r1] = acc10[i];
        }
    }
    int gt1 = blk_t0 + n1 + col;
    if (gt1 < B) {
        float* yt = Y + (size_t)gt1 * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r0 = blk_row0 + m0 + half * 8 + i;
            if (r0 < n) yt[r0] = acc01[i];
            int r1 = blk_row0 + m1 + half * 8 + i;
            if (r1 < n) yt[r1] = acc11[i];
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 4) void gemm_q4_0_wmma_k(const uint8_t* __restrict__ W,
                                            const w_q8_1_blk* __restrict__ Xq,
                                            float* __restrict__ Y,
                                            int n, int k, int B) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int blk_row0 = blockIdx.x * 64;
    int blk_t0 = blockIdx.y * 32;
    if (blk_row0 >= n || blk_t0 >= B) return;
    bool has1 = (blk_t0 + 16 < B);

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);

    __shared__ uint4 sm_wq[4][64];
    __shared__ float sm_dw[4][64];
    __shared__ int2 sm_xq[4][4][32];
    __shared__ float sm_xd[4][32];
    __shared__ int32_t sm_xs[4][32];

    int r_loc = tid >> 2;
    int kb_load = tid & 3;
    int r_g0 = min(blk_row0 + r_loc, n - 1);
    int r_g1 = min(blk_row0 + r_loc + 32, n - 1);
    const uint8_t* wr0 = W + (size_t)r_g0 * rbytes;
    const uint8_t* wr1 = W + (size_t)r_g1 * rbytes;
    const uint4* qr0 = reinterpret_cast<const uint4*>(wr0);
    const uint4* qr1 = reinterpret_cast<const uint4*>(wr1);
    const __half* dr0 = reinterpret_cast<const __half*>(wr0 + (size_t)nblk * 16);
    const __half* dr1 = reinterpret_cast<const __half*>(wr1 + (size_t)nblk * 16);

    int t_g = min(blk_t0 + r_loc, B - 1);
    const w_q8_1_blk* xr = Xq + (size_t)t_g * nblk;

    int m0 = warp * 16;
    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int b0 = 0; b0 < nblk; b0 += 4) {
        int b = b0 + kb_load;
        if (b < nblk) {
            sm_wq[kb_load][r_loc] = qr0[b];
            sm_dw[kb_load][r_loc] = __half2float(dr0[b]);
            sm_wq[kb_load][r_loc + 32] = qr1[b];
            sm_dw[kb_load][r_loc + 32] = __half2float(dr1[b]);
            const w_q8_1_blk xb = xr[b];
            sm_xq[kb_load][0][r_loc] = make_int2(xb.q[0], xb.q[1]);
            sm_xq[kb_load][1][r_loc] = make_int2(xb.q[2], xb.q[3]);
            sm_xq[kb_load][2][r_loc] = make_int2(xb.q[4], xb.q[5]);
            sm_xq[kb_load][3][r_loc] = make_int2(xb.q[6], xb.q[7]);
            sm_xd[kb_load][r_loc] = xb.d;
            sm_xs[kb_load][r_loc] = 8 * xb.s;
        } else {
            sm_wq[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc] = 0.0f;
            sm_wq[kb_load][r_loc + 32] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc + 32] = 0.0f;
            sm_xd[kb_load][r_loc] = 0.0f;
            sm_xs[kb_load][r_loc] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint4 q4 = sm_wq[kb][m0 + col];
            uint32_t w0 = half ? q4.z : q4.x;
            uint32_t w1 = half ? q4.w : q4.y;
            wmma_i2_v a_lo = {(int32_t)(w0 & 0x0F0F0F0Fu), (int32_t)(w1 & 0x0F0F0F0Fu)};
            wmma_i2_v a_hi = {(int32_t)((w0 >> 4) & 0x0F0F0F0Fu), (int32_t)((w1 >> 4) & 0x0F0F0F0Fu)};

            int2 x0l = sm_xq[kb][half][col];
            int2 x0h = sm_xq[kb][2 + half][col];
            wmma_i2_v b0_lo = {x0l.x, x0l.y};
            wmma_i2_v b0_hi = {x0h.x, x0h.y};
            wmma_i8_v cv0 = {0, 0, 0, 0, 0, 0, 0, 0};
            cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b0_lo, cv0, false);
            cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b0_hi, cv0, false);

            wmma_i8_v cv1 = {0, 0, 0, 0, 0, 0, 0, 0};
            if (has1) {
                int2 x1l = sm_xq[kb][half][16 + col];
                int2 x1h = sm_xq[kb][2 + half][16 + col];
                wmma_i2_v b1_lo = {x1l.x, x1l.y};
                wmma_i2_v b1_hi = {x1h.x, x1h.y};
                cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b1_lo, cv1, false);
                cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b1_hi, cv1, false);
            }
            float xd0 = sm_xd[kb][col];
            int32_t xs0 = sm_xs[kb][col];
            float xd1 = has1 ? sm_xd[kb][16 + col] : 0.0f;
            int32_t xs1 = has1 ? sm_xs[kb][16 + col] : 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dw_i = sm_dw[kb][m0 + half * 8 + i];
                acc0[i] = fmaf(dw_i * xd0, (float)(cv0[i] - xs0), acc0[i]);
                if (has1) acc1[i] = fmaf(dw_i * xd1, (float)(cv1[i] - xs1), acc1[i]);
            }
        }
        __syncthreads();
    }

    if (blk_t0 + col < B) {
        float* yt = Y + (size_t)(blk_t0 + col) * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = blk_row0 + m0 + half * 8 + i;
            if (r < n) yt[r] = acc0[i];
        }
    }
    if (has1 && blk_t0 + 16 + col < B) {
        float* yt = Y + (size_t)(blk_t0 + 16 + col) * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = blk_row0 + m0 + half * 8 + i;
            if (r < n) yt[r] = acc1[i];
        }
    }
}

extern "C" __global__ void qkvrg_gemm_q4_wmma64_k(const uint8_t* __restrict__ wq,
                                                  const uint8_t* __restrict__ wk,
                                                  const uint8_t* __restrict__ wv,
                                                  const uint8_t* __restrict__ wr,
                                                  const uint8_t* __restrict__ wg,
                                                  const w_q8_1_blk* __restrict__ Xq0,
                                                  const w_q8_1_blk* __restrict__ Xq1,
                                                  const w_q8_1_blk* __restrict__ Xq2,
                                                  float* __restrict__ Q,
                                                  float* __restrict__ K,
                                                  float* __restrict__ V,
                                                  float* __restrict__ R,
                                                  float* __restrict__ G,
                                                  int k, int q_dim, int kv_dim,
                                                  int r_dim, int B) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int wm = warp >> 1;
    int wn = warp & 1;
    int blk_row0 = blockIdx.x * 128;
    int blk_t0 = blockIdx.y * 64;
    if (blk_t0 >= B) return;

    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    if (blk_row0 >= o5) return;

    const uint8_t* W;
    const w_q8_1_blk* Xq;
    float* Y;
    int local_row0, n_out;
    if (blk_row0 < o1) { W = wq; Xq = Xq0; Y = Q; local_row0 = blk_row0; n_out = q_dim; }
    else if (blk_row0 < o2) { W = wk; Xq = Xq1; Y = K; local_row0 = blk_row0 - o1; n_out = kv_dim; }
    else if (blk_row0 < o3) { W = wv; Xq = Xq2; Y = V; local_row0 = blk_row0 - o2; n_out = kv_dim; }
    else if (blk_row0 < o4) { W = wr; Xq = Xq0; Y = R; local_row0 = blk_row0 - o3; n_out = r_dim; }
    else { W = wg; Xq = Xq0; Y = G; local_row0 = blk_row0 - o4; n_out = q_dim; }

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);

    __shared__ uint4 sm_wq[4][128];
    __shared__ float sm_dw[4][128];
    __shared__ int2 sm_xq[4][4][64];
    __shared__ float sm_xd[4][64];
    __shared__ int32_t sm_xs[4][64];

    int r_loc = tid >> 2;
    int kb_load = tid & 3;
    int r_g0 = min(local_row0 + r_loc, n_out - 1);
    int r_g1 = min(local_row0 + r_loc + 64, n_out - 1);
    const uint8_t* wr0 = W + (size_t)r_g0 * rbytes;
    const uint8_t* wr1 = W + (size_t)r_g1 * rbytes;
    const uint4* qr0 = reinterpret_cast<const uint4*>(wr0);
    const uint4* qr1 = reinterpret_cast<const uint4*>(wr1);
    const __half* dr0 = reinterpret_cast<const __half*>(wr0 + (size_t)nblk * 16);
    const __half* dr1 = reinterpret_cast<const __half*>(wr1 + (size_t)nblk * 16);

    int t_g0 = min(blk_t0 + r_loc, B - 1);
    const w_q8_1_blk* xr0 = Xq + (size_t)t_g0 * nblk;

    int m0 = wm * 32;
    int m1 = m0 + 16;
    int n0 = wn * 32;
    int n1 = n0 + 16;

    float acc00[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc01[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc10[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc11[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int b0 = 0; b0 < nblk; b0 += 4) {
        int b = b0 + kb_load;
        if (b < nblk) {
            sm_wq[kb_load][r_loc] = qr0[b];
            sm_dw[kb_load][r_loc] = __half2float(dr0[b]);
            sm_wq[kb_load][r_loc + 64] = qr1[b];
            sm_dw[kb_load][r_loc + 64] = __half2float(dr1[b]);
            const w_q8_1_blk xb0 = xr0[b];
            sm_xq[kb_load][0][r_loc] = make_int2(xb0.q[0], xb0.q[1]);
            sm_xq[kb_load][1][r_loc] = make_int2(xb0.q[2], xb0.q[3]);
            sm_xq[kb_load][2][r_loc] = make_int2(xb0.q[4], xb0.q[5]);
            sm_xq[kb_load][3][r_loc] = make_int2(xb0.q[6], xb0.q[7]);
            sm_xd[kb_load][r_loc] = xb0.d;
            sm_xs[kb_load][r_loc] = 8 * xb0.s;
        } else {
            sm_wq[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc] = 0.0f;
            sm_wq[kb_load][r_loc + 64] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc + 64] = 0.0f;
            sm_xd[kb_load][r_loc] = 0.0f;
            sm_xs[kb_load][r_loc] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint4 q0 = sm_wq[kb][m0 + col];
            uint32_t w00 = half ? q0.z : q0.x;
            uint32_t w01 = half ? q0.w : q0.y;
            wmma_i2_v a0_lo = {(int32_t)(w00 & 0x0F0F0F0Fu), (int32_t)(w01 & 0x0F0F0F0Fu)};
            wmma_i2_v a0_hi = {(int32_t)((w00 >> 4) & 0x0F0F0F0Fu), (int32_t)((w01 >> 4) & 0x0F0F0F0Fu)};

            uint4 q1 = sm_wq[kb][m1 + col];
            uint32_t w10 = half ? q1.z : q1.x;
            uint32_t w11 = half ? q1.w : q1.y;
            wmma_i2_v a1_lo = {(int32_t)(w10 & 0x0F0F0F0Fu), (int32_t)(w11 & 0x0F0F0F0Fu)};
            wmma_i2_v a1_hi = {(int32_t)((w10 >> 4) & 0x0F0F0F0Fu), (int32_t)((w11 >> 4) & 0x0F0F0F0Fu)};

            int2 x0l = sm_xq[kb][half][n0 + col];
            int2 x0h = sm_xq[kb][2 + half][n0 + col];
            wmma_i2_v b0_lo = {x0l.x, x0l.y};
            wmma_i2_v b0_hi = {x0h.x, x0h.y};

            int2 x1l = sm_xq[kb][half][n1 + col];
            int2 x1h = sm_xq[kb][2 + half][n1 + col];
            wmma_i2_v b1_lo = {x1l.x, x1l.y};
            wmma_i2_v b1_hi = {x1h.x, x1h.y};

            wmma_i8_v c00 = {0, 0, 0, 0, 0, 0, 0, 0};
            wmma_i8_v c01 = {0, 0, 0, 0, 0, 0, 0, 0};
            wmma_i8_v c10 = {0, 0, 0, 0, 0, 0, 0, 0};
            wmma_i8_v c11 = {0, 0, 0, 0, 0, 0, 0, 0};
            c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_lo, true, b0_lo, c00, false);
            c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_hi, true, b0_hi, c00, false);
            c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_lo, true, b1_lo, c01, false);
            c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_hi, true, b1_hi, c01, false);
            c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_lo, true, b0_lo, c10, false);
            c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_hi, true, b0_hi, c10, false);
            c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_lo, true, b1_lo, c11, false);
            c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_hi, true, b1_hi, c11, false);

            float xd0 = sm_xd[kb][n0 + col];
            int32_t xs0 = sm_xs[kb][n0 + col];
            float xd1 = sm_xd[kb][n1 + col];
            int32_t xs1 = sm_xs[kb][n1 + col];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dw0 = sm_dw[kb][m0 + half * 8 + i];
                float dw1 = sm_dw[kb][m1 + half * 8 + i];
                acc00[i] = fmaf(dw0 * xd0, (float)(c00[i] - xs0), acc00[i]);
                acc01[i] = fmaf(dw0 * xd1, (float)(c01[i] - xs1), acc01[i]);
                acc10[i] = fmaf(dw1 * xd0, (float)(c10[i] - xs0), acc10[i]);
                acc11[i] = fmaf(dw1 * xd1, (float)(c11[i] - xs1), acc11[i]);
            }
        }
        __syncthreads();
    }

    int gt0 = blk_t0 + n0 + col;
    if (gt0 < B) {
        float* yt = Y + (size_t)gt0 * n_out;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r0 = local_row0 + m0 + half * 8 + i;
            if (r0 < n_out) yt[r0] = acc00[i];
            int r1 = local_row0 + m1 + half * 8 + i;
            if (r1 < n_out) yt[r1] = acc10[i];
        }
    }
    int gt1 = blk_t0 + n1 + col;
    if (gt1 < B) {
        float* yt = Y + (size_t)gt1 * n_out;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r0 = local_row0 + m0 + half * 8 + i;
            if (r0 < n_out) yt[r0] = acc01[i];
            int r1 = local_row0 + m1 + half * 8 + i;
            if (r1 < n_out) yt[r1] = acc11[i];
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 4) void qkvrg_gemm_q4_wmma_k(const uint8_t* __restrict__ wq,
                                                const uint8_t* __restrict__ wk,
                                                const uint8_t* __restrict__ wv,
                                                const uint8_t* __restrict__ wr,
                                                const uint8_t* __restrict__ wg,
                                                const w_q8_1_blk* __restrict__ Xq0,
                                                const w_q8_1_blk* __restrict__ Xq1,
                                                const w_q8_1_blk* __restrict__ Xq2,
                                                float* __restrict__ Q,
                                                float* __restrict__ K,
                                                float* __restrict__ V,
                                                float* __restrict__ R,
                                                float* __restrict__ G,
                                                int k, int q_dim, int kv_dim,
                                                int r_dim, int B) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int blk_row0 = blockIdx.x * 64;
    int blk_t0 = blockIdx.y * 32;
    if (blk_t0 >= B) return;
    bool has1 = (blk_t0 + 16 < B);

    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    if (blk_row0 >= o5) return;

    const uint8_t* W;
    const w_q8_1_blk* Xq;
    float* Y;
    int local_row0, n_out;
    if (blk_row0 < o1) { W = wq; Xq = Xq0; Y = Q; local_row0 = blk_row0; n_out = q_dim; }
    else if (blk_row0 < o2) { W = wk; Xq = Xq1; Y = K; local_row0 = blk_row0 - o1; n_out = kv_dim; }
    else if (blk_row0 < o3) { W = wv; Xq = Xq2; Y = V; local_row0 = blk_row0 - o2; n_out = kv_dim; }
    else if (blk_row0 < o4) { W = wr; Xq = Xq0; Y = R; local_row0 = blk_row0 - o3; n_out = r_dim; }
    else { W = wg; Xq = Xq0; Y = G; local_row0 = blk_row0 - o4; n_out = q_dim; }

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);

    __shared__ uint4 sm_wq[4][64];
    __shared__ float sm_dw[4][64];
    __shared__ int2 sm_xq[4][4][32];
    __shared__ float sm_xd[4][32];
    __shared__ int32_t sm_xs[4][32];

    int r_loc = tid >> 2;
    int kb_load = tid & 3;
    int r_g0 = min(local_row0 + r_loc, n_out - 1);
    int r_g1 = min(local_row0 + r_loc + 32, n_out - 1);
    const uint8_t* wr0 = W + (size_t)r_g0 * rbytes;
    const uint8_t* wr1 = W + (size_t)r_g1 * rbytes;
    const uint4* qr0 = reinterpret_cast<const uint4*>(wr0);
    const uint4* qr1 = reinterpret_cast<const uint4*>(wr1);
    const __half* dr0 = reinterpret_cast<const __half*>(wr0 + (size_t)nblk * 16);
    const __half* dr1 = reinterpret_cast<const __half*>(wr1 + (size_t)nblk * 16);

    int t_g = min(blk_t0 + r_loc, B - 1);
    const w_q8_1_blk* xr = Xq + (size_t)t_g * nblk;

    int m0 = warp * 16;
    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int b0 = 0; b0 < nblk; b0 += 4) {
        int b = b0 + kb_load;
        if (b < nblk) {
            sm_wq[kb_load][r_loc] = qr0[b];
            sm_dw[kb_load][r_loc] = __half2float(dr0[b]);
            sm_wq[kb_load][r_loc + 32] = qr1[b];
            sm_dw[kb_load][r_loc + 32] = __half2float(dr1[b]);
            const w_q8_1_blk xb = xr[b];
            sm_xq[kb_load][0][r_loc] = make_int2(xb.q[0], xb.q[1]);
            sm_xq[kb_load][1][r_loc] = make_int2(xb.q[2], xb.q[3]);
            sm_xq[kb_load][2][r_loc] = make_int2(xb.q[4], xb.q[5]);
            sm_xq[kb_load][3][r_loc] = make_int2(xb.q[6], xb.q[7]);
            sm_xd[kb_load][r_loc] = xb.d;
            sm_xs[kb_load][r_loc] = 8 * xb.s;
        } else {
            sm_wq[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc] = 0.0f;
            sm_wq[kb_load][r_loc + 32] = make_uint4(0, 0, 0, 0);
            sm_dw[kb_load][r_loc + 32] = 0.0f;
            sm_xd[kb_load][r_loc] = 0.0f;
            sm_xs[kb_load][r_loc] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint4 q4 = sm_wq[kb][m0 + col];
            uint32_t w0 = half ? q4.z : q4.x;
            uint32_t w1 = half ? q4.w : q4.y;
            wmma_i2_v a_lo = {(int32_t)(w0 & 0x0F0F0F0Fu), (int32_t)(w1 & 0x0F0F0F0Fu)};
            wmma_i2_v a_hi = {(int32_t)((w0 >> 4) & 0x0F0F0F0Fu), (int32_t)((w1 >> 4) & 0x0F0F0F0Fu)};

            int2 x0l = sm_xq[kb][half][col];
            int2 x0h = sm_xq[kb][2 + half][col];
            wmma_i2_v b0_lo = {x0l.x, x0l.y};
            wmma_i2_v b0_hi = {x0h.x, x0h.y};
            wmma_i8_v cv0 = {0, 0, 0, 0, 0, 0, 0, 0};
            cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b0_lo, cv0, false);
            cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b0_hi, cv0, false);

            wmma_i8_v cv1 = {0, 0, 0, 0, 0, 0, 0, 0};
            if (has1) {
                int2 x1l = sm_xq[kb][half][16 + col];
                int2 x1h = sm_xq[kb][2 + half][16 + col];
                wmma_i2_v b1_lo = {x1l.x, x1l.y};
                wmma_i2_v b1_hi = {x1h.x, x1h.y};
                cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b1_lo, cv1, false);
                cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b1_hi, cv1, false);
            }
            float xd0 = sm_xd[kb][col];
            int32_t xs0 = sm_xs[kb][col];
            float xd1 = has1 ? sm_xd[kb][16 + col] : 0.0f;
            int32_t xs1 = has1 ? sm_xs[kb][16 + col] : 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dw_i = sm_dw[kb][m0 + half * 8 + i];
                acc0[i] = fmaf(dw_i * xd0, (float)(cv0[i] - xs0), acc0[i]);
                if (has1) acc1[i] = fmaf(dw_i * xd1, (float)(cv1[i] - xs1), acc1[i]);
            }
        }
        __syncthreads();
    }

    if (blk_t0 + col < B) {
        float* yt = Y + (size_t)(blk_t0 + col) * n_out;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = local_row0 + m0 + half * 8 + i;
            if (r < n_out) yt[r] = acc0[i];
        }
    }
    if (has1 && blk_t0 + 16 + col < B) {
        float* yt = Y + (size_t)(blk_t0 + 16 + col) * n_out;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = local_row0 + m0 + half * 8 + i;
            if (r < n_out) yt[r] = acc1[i];
        }
    }
}

extern "C" __global__ void sink_dyn_bf16_wmma_k(const uint16_t* __restrict__ wsink,
                                                const uint16_t* __restrict__ wkdyn,
                                                const uint16_t* __restrict__ wvdyn,
                                                const float* __restrict__ X0,
                                                const float* __restrict__ X1,
                                                const float* __restrict__ X2,
                                                float* __restrict__ S,
                                                float* __restrict__ Kdyn,
                                                float* __restrict__ Vdyn,
                                                int k, int n_head, int dyn_dim, int B) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int part = blockIdx.x;
    int t0 = blockIdx.y * 64;
    if (t0 >= B) return;

    const uint16_t* W;
    const float* X;
    float* Y;
    int n_out, row0;
    if (part == 0) { W = wsink; X = X0; Y = S; n_out = n_head; row0 = 0; }
    else if (part == 1) { W = wsink; X = X0; Y = S; n_out = n_head; row0 = 16; }
    else if (part == 2) { W = wkdyn; X = X1; Y = Kdyn; n_out = dyn_dim; row0 = 0; }
    else { W = wvdyn; X = X2; Y = Vdyn; n_out = dyn_dim; row0 = 0; }
    if (row0 >= n_out) return;

    __shared__ _Float16 sm_w[16][32];
    __shared__ _Float16 sm_x[64][32];

    int n_r = min(16, n_out - row0);
    int nt = min(64, B - t0);
    int k4 = k >> 2;

    wmma_f8_v cv = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int tn_base = warp * 16;

    for (int kk = 0; kk < k; kk += 32) {
        if (tid < 128) {
            int r_loc = tid >> 3;
            int k4_loc = (tid & 7) * 4;
            int r_g = row0 + min(r_loc, n_r - 1);
            const uint16_t* wp = W + (size_t)r_g * k + kk + k4_loc;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sm_w[r_loc][k4_loc + j] = (_Float16)w_bf16_f32(wp[j]);
            }
        }
        for (int p = 0; p < 4; ++p) {
            int idx = p * 128 + tid;
            int t_loc = idx >> 3;
            int k4_loc = (idx & 7) * 4;
            int t_g = t0 + min(t_loc, nt - 1);
            float4 xv = *reinterpret_cast<const float4*>(X + (size_t)t_g * k + kk + k4_loc);
            sm_x[t_loc][k4_loc + 0] = (_Float16)xv.x;
            sm_x[t_loc][k4_loc + 1] = (_Float16)xv.y;
            sm_x[t_loc][k4_loc + 2] = (_Float16)xv.z;
            sm_x[t_loc][k4_loc + 3] = (_Float16)xv.w;
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            int kbase = kb * 16 + half * 8;
            wmma_h8_v av, bv;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                av[j] = sm_w[col][kbase + j];
                bv[j] = sm_x[tn_base + col][kbase + j];
            }
            cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, cv);
        }
        __syncthreads();
    }

    if (tn_base + col < nt) {
        float* yt = Y + (size_t)(t0 + tn_base + col) * n_out;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = row0 + half * 8 + i;
            if (r < n_out) yt[r] = cv[i];
        }
    }
}

extern "C" __global__ void sink_dyn_bf16_k(const uint16_t* __restrict__ wsink,
                                           const uint16_t* __restrict__ wkdyn,
                                           const uint16_t* __restrict__ wvdyn,
                                           const float* __restrict__ X0,
                                           const float* __restrict__ X1,
                                           const float* __restrict__ X2,
                                           float* __restrict__ S,
                                           float* __restrict__ Kdyn,
                                           float* __restrict__ Vdyn,
                                           int k, int n_head, int dyn_dim, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int gid = blockIdx.x * 4 + warp;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0 || gid >= n_head + 2 * dyn_dim) return;
    const uint16_t* W;
    const float* X;
    float* Y;
    int row, n_out;
    if (gid < n_head) { W = wsink; X = X0; Y = S; row = gid; n_out = n_head; }
    else if (gid < n_head + dyn_dim) { W = wkdyn; X = X1; Y = Kdyn; row = gid - n_head; n_out = dyn_dim; }
    else { W = wvdyn; X = X2; Y = Vdyn; row = gid - n_head - dyn_dim; n_out = dyn_dim; }
    const uint16_t* wrow = W + (size_t)row * k;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = lane; j < k; j += 32) {
        float wv = w_bf16_f32(wrow[j]);
        for (int m = 0; m < n_t; ++m) {
            acc[m] = fmaf(wv, X[(size_t)(t0 + m) * k + j], acc[m]);
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = w_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n_out + row] = v;
    }
}

extern "C" __global__ void moe_gu_q4_0_wmma_k(const uint8_t* __restrict__ W,
                                              const float* __restrict__ scales,
                                              const w_q8_1_blk* __restrict__ Xq,
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

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int wm = warp >> 1;
    int wn = warp & 1;
    int blk_row0 = blockIdx.x * 64;
    if (blk_row0 >= n_ff) return;

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)(2 * n_ff) * rbytes;

    __shared__ uint4 sm_wg[4][64];
    __shared__ uint4 sm_wu[4][64];
    __shared__ float sm_dg[4][64];
    __shared__ float sm_du[4][64];
    __shared__ int2 sm_xq[4][4][64];
    __shared__ float sm_xd[4][64];
    __shared__ int32_t sm_xs[4][64];

    int r_loc = tid >> 2;
    int kb_load = tid & 3;
    int r_g = min(blk_row0 + r_loc, n_ff - 1);
    const uint8_t* wg_p = W + (size_t)exp_id * exp_bytes + (size_t)r_g * rbytes;
    const uint8_t* wu_p = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + r_g) * rbytes;
    const uint4* qg_r = reinterpret_cast<const uint4*>(wg_p);
    const __half* dg_r = reinterpret_cast<const __half*>(wg_p + (size_t)nblk * 16);
    const uint4* qu_r = reinterpret_cast<const uint4*>(wu_p);
    const __half* du_r = reinterpret_cast<const __half*>(wu_p + (size_t)nblk * 16);
    float sc = scales ? scales[exp_id] : 1.0f;

    int m0 = wm * 16;
    int n0 = wn * 32;
    int n1 = n0 + 16;

    int z_id = blockIdx.z;
    int z_n = gridDim.z;
    for (int t0 = z_id * 64; t0 < cnt; t0 += z_n * 64) {
        int nt = min(64, cnt - t0);
        int idx_l = off0 + t0 + min(r_loc, nt - 1);
        const w_q8_1_blk* xr = Xq + (size_t)tok_sorted[idx_l] * nblk;

        float ag0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float ag1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        for (int b0 = 0; b0 < nblk; b0 += 4) {
            int b = b0 + kb_load;
            if (b < nblk) {
                sm_wg[kb_load][r_loc] = qg_r[b];
                sm_dg[kb_load][r_loc] = __half2float(dg_r[b]);
                sm_wu[kb_load][r_loc] = qu_r[b];
                sm_du[kb_load][r_loc] = __half2float(du_r[b]);
                const w_q8_1_blk xb = xr[b];
                sm_xq[kb_load][0][r_loc] = make_int2(xb.q[0], xb.q[1]);
                sm_xq[kb_load][1][r_loc] = make_int2(xb.q[2], xb.q[3]);
                sm_xq[kb_load][2][r_loc] = make_int2(xb.q[4], xb.q[5]);
                sm_xq[kb_load][3][r_loc] = make_int2(xb.q[6], xb.q[7]);
                sm_xd[kb_load][r_loc] = xb.d;
                sm_xs[kb_load][r_loc] = 8 * xb.s;
            } else {
                sm_wg[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
                sm_dg[kb_load][r_loc] = 0.0f;
                sm_wu[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
                sm_du[kb_load][r_loc] = 0.0f;
                sm_xd[kb_load][r_loc] = 0.0f;
                sm_xs[kb_load][r_loc] = 0;
            }
            __syncthreads();

#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                uint4 qg = sm_wg[kb][m0 + col];
                uint32_t g0 = half ? qg.z : qg.x;
                uint32_t g1 = half ? qg.w : qg.y;
                wmma_i2_v ag_lo = {(int32_t)(g0 & 0x0F0F0F0Fu), (int32_t)(g1 & 0x0F0F0F0Fu)};
                wmma_i2_v ag_hi = {(int32_t)((g0 >> 4) & 0x0F0F0F0Fu), (int32_t)((g1 >> 4) & 0x0F0F0F0Fu)};

                uint4 qu = sm_wu[kb][m0 + col];
                uint32_t u0 = half ? qu.z : qu.x;
                uint32_t u1 = half ? qu.w : qu.y;
                wmma_i2_v au_lo = {(int32_t)(u0 & 0x0F0F0F0Fu), (int32_t)(u1 & 0x0F0F0F0Fu)};
                wmma_i2_v au_hi = {(int32_t)((u0 >> 4) & 0x0F0F0F0Fu), (int32_t)((u1 >> 4) & 0x0F0F0F0Fu)};

                int2 x0l = sm_xq[kb][half][n0 + col];
                int2 x0h = sm_xq[kb][2 + half][n0 + col];
                wmma_i2_v b0_lo = {x0l.x, x0l.y};
                wmma_i2_v b0_hi = {x0h.x, x0h.y};

                int2 x1l = sm_xq[kb][half][n1 + col];
                int2 x1h = sm_xq[kb][2 + half][n1 + col];
                wmma_i2_v b1_lo = {x1l.x, x1l.y};
                wmma_i2_v b1_hi = {x1h.x, x1h.y};

                wmma_i8_v cg0 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v cu0 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v cg1 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v cu1 = {0, 0, 0, 0, 0, 0, 0, 0};
                cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_lo, true, b0_lo, cg0, false);
                cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_hi, true, b0_hi, cg0, false);
                cu0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_lo, true, b0_lo, cu0, false);
                cu0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_hi, true, b0_hi, cu0, false);
                cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_lo, true, b1_lo, cg1, false);
                cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_hi, true, b1_hi, cg1, false);
                cu1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_lo, true, b1_lo, cu1, false);
                cu1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_hi, true, b1_hi, cu1, false);

                float xd0 = sm_xd[kb][n0 + col];
                int32_t xs0 = sm_xs[kb][n0 + col];
                float xd1 = sm_xd[kb][n1 + col];
                int32_t xs1 = sm_xs[kb][n1 + col];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float dgi = sm_dg[kb][m0 + half * 8 + i];
                    float dui = sm_du[kb][m0 + half * 8 + i];
                    ag0[i] = fmaf(dgi * xd0, (float)(cg0[i] - xs0), ag0[i]);
                    au0[i] = fmaf(dui * xd0, (float)(cu0[i] - xs0), au0[i]);
                    ag1[i] = fmaf(dgi * xd1, (float)(cg1[i] - xs1), ag1[i]);
                    au1[i] = fmaf(dui * xd1, (float)(cu1[i] - xs1), au1[i]);
                }
            }
            __syncthreads();
        }
        if (n0 + col < nt) {
            int idx0 = off0 + t0 + n0 + col;
            float* dst = act_out + ((size_t)tok_sorted[idx0] * n_used + slot_sorted[idx0]) * n_ff;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r = blk_row0 + m0 + half * 8 + i;
                if (r < n_ff) {
                    float g = ag0[i] * sc;
                    float u = au0[i] * sc;
                    float sg = 1.0f / (1.0f + expf(-g));
                    float act = beta * tanhf(g / beta) * sg;
                    float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                    dst[r] = act * up;
                }
            }
        }
        if (n1 + col < nt) {
            int idx1 = off0 + t0 + n1 + col;
            float* dst = act_out + ((size_t)tok_sorted[idx1] * n_used + slot_sorted[idx1]) * n_ff;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r = blk_row0 + m0 + half * 8 + i;
                if (r < n_ff) {
                    float g = ag1[i] * sc;
                    float u = au1[i] * sc;
                    float sg = 1.0f / (1.0f + expf(-g));
                    float act = beta * tanhf(g / beta) * sg;
                    float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                    dst[r] = act * up;
                }
            }
        }
    }
}

extern "C" __global__ void moe_gu_q4_0_wmma_q8_k(const uint8_t* __restrict__ W,
                                                 const float* __restrict__ scales,
                                                 const w_q8_1_blk* __restrict__ Xq,
                                                 const int* __restrict__ exp_counts,
                                                 const int* __restrict__ exp_offsets,
                                                 const int* __restrict__ tok_sorted,
                                                 const int* __restrict__ slot_sorted,
                                                 w_q8_1_blk* __restrict__ act_q,
                                                 int n_ff, int k, int n_used,
                                                 float beta, float lb) {
    int exp_id = exp_counts[128 + blockIdx.y];
    if (exp_id < 0) return;
    int cnt = exp_counts[exp_id];
    if (cnt == 0) return;
    int off0 = exp_offsets[exp_id];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int wm = warp >> 1;
    int wn = warp & 1;
    int blk_row0 = blockIdx.x * 64;
    if (blk_row0 >= n_ff) return;

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)(2 * n_ff) * rbytes;

    __shared__ union {
        struct {
            uint4 wg[4][64];
            uint4 wu[4][64];
            float dg[4][64];
            float du[4][64];
            int2 xq[4][4][64];
            float xd[4][64];
            int32_t xs[4][64];
        } t;
        float act[64][65];
    } sm;

    int r_loc = tid >> 2;
    int kb_load = tid & 3;
    int r_g = min(blk_row0 + r_loc, n_ff - 1);
    const uint8_t* wg_p = W + (size_t)exp_id * exp_bytes + (size_t)r_g * rbytes;
    const uint8_t* wu_p = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + r_g) * rbytes;
    const uint4* qg_r = reinterpret_cast<const uint4*>(wg_p);
    const __half* dg_r = reinterpret_cast<const __half*>(wg_p + (size_t)nblk * 16);
    const uint4* qu_r = reinterpret_cast<const uint4*>(wu_p);
    const __half* du_r = reinterpret_cast<const __half*>(wu_p + (size_t)nblk * 16);
    float sc = scales ? scales[exp_id] : 1.0f;

    int m0 = wm * 16;
    int n0 = wn * 32;
    int n1 = n0 + 16;
    int qblk_per_slot = n_ff >> 5;
    int blk_base = blk_row0 >> 5;

    int z_id = blockIdx.z;
    int z_n = gridDim.z;
    for (int t0 = z_id * 64; t0 < cnt; t0 += z_n * 64) {
        int nt = min(64, cnt - t0);
        int idx_l = off0 + t0 + min(r_loc, nt - 1);
        const w_q8_1_blk* xr = Xq + (size_t)tok_sorted[idx_l] * nblk;

        float ag0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float ag1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        for (int b0 = 0; b0 < nblk; b0 += 4) {
            int b = b0 + kb_load;
            if (b < nblk) {
                sm.t.wg[kb_load][r_loc] = qg_r[b];
                sm.t.dg[kb_load][r_loc] = __half2float(dg_r[b]);
                sm.t.wu[kb_load][r_loc] = qu_r[b];
                sm.t.du[kb_load][r_loc] = __half2float(du_r[b]);
                const w_q8_1_blk xb = xr[b];
                sm.t.xq[kb_load][0][r_loc] = make_int2(xb.q[0], xb.q[1]);
                sm.t.xq[kb_load][1][r_loc] = make_int2(xb.q[2], xb.q[3]);
                sm.t.xq[kb_load][2][r_loc] = make_int2(xb.q[4], xb.q[5]);
                sm.t.xq[kb_load][3][r_loc] = make_int2(xb.q[6], xb.q[7]);
                sm.t.xd[kb_load][r_loc] = xb.d;
                sm.t.xs[kb_load][r_loc] = 8 * xb.s;
            } else {
                sm.t.wg[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
                sm.t.dg[kb_load][r_loc] = 0.0f;
                sm.t.wu[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
                sm.t.du[kb_load][r_loc] = 0.0f;
                sm.t.xd[kb_load][r_loc] = 0.0f;
                sm.t.xs[kb_load][r_loc] = 0;
            }
            __syncthreads();

#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                uint4 qg = sm.t.wg[kb][m0 + col];
                uint32_t g0 = half ? qg.z : qg.x;
                uint32_t g1 = half ? qg.w : qg.y;
                wmma_i2_v ag_lo = {(int32_t)(g0 & 0x0F0F0F0Fu), (int32_t)(g1 & 0x0F0F0F0Fu)};
                wmma_i2_v ag_hi = {(int32_t)((g0 >> 4) & 0x0F0F0F0Fu), (int32_t)((g1 >> 4) & 0x0F0F0F0Fu)};

                uint4 qu = sm.t.wu[kb][m0 + col];
                uint32_t u0 = half ? qu.z : qu.x;
                uint32_t u1 = half ? qu.w : qu.y;
                wmma_i2_v au_lo = {(int32_t)(u0 & 0x0F0F0F0Fu), (int32_t)(u1 & 0x0F0F0F0Fu)};
                wmma_i2_v au_hi = {(int32_t)((u0 >> 4) & 0x0F0F0F0Fu), (int32_t)((u1 >> 4) & 0x0F0F0F0Fu)};

                int2 x0l = sm.t.xq[kb][half][n0 + col];
                int2 x0h = sm.t.xq[kb][2 + half][n0 + col];
                wmma_i2_v b0_lo = {x0l.x, x0l.y};
                wmma_i2_v b0_hi = {x0h.x, x0h.y};

                int2 x1l = sm.t.xq[kb][half][n1 + col];
                int2 x1h = sm.t.xq[kb][2 + half][n1 + col];
                wmma_i2_v b1_lo = {x1l.x, x1l.y};
                wmma_i2_v b1_hi = {x1h.x, x1h.y};

                wmma_i8_v cg0 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v cu0 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v cg1 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v cu1 = {0, 0, 0, 0, 0, 0, 0, 0};
                cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_lo, true, b0_lo, cg0, false);
                cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_hi, true, b0_hi, cg0, false);
                cu0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_lo, true, b0_lo, cu0, false);
                cu0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_hi, true, b0_hi, cu0, false);
                cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_lo, true, b1_lo, cg1, false);
                cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_hi, true, b1_hi, cg1, false);
                cu1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_lo, true, b1_lo, cu1, false);
                cu1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_hi, true, b1_hi, cu1, false);

                float xd0 = sm.t.xd[kb][n0 + col];
                int32_t xs0 = sm.t.xs[kb][n0 + col];
                float xd1 = sm.t.xd[kb][n1 + col];
                int32_t xs1 = sm.t.xs[kb][n1 + col];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float dgi = sm.t.dg[kb][m0 + half * 8 + i];
                    float dui = sm.t.du[kb][m0 + half * 8 + i];
                    ag0[i] = fmaf(dgi * xd0, (float)(cg0[i] - xs0), ag0[i]);
                    au0[i] = fmaf(dui * xd0, (float)(cu0[i] - xs0), au0[i]);
                    ag1[i] = fmaf(dgi * xd1, (float)(cg1[i] - xs1), ag1[i]);
                    au1[i] = fmaf(dui * xd1, (float)(cu1[i] - xs1), au1[i]);
                }
            }
            __syncthreads();
        }
        if (n0 + col < nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int lr = m0 + half * 8 + i;
                int r = blk_row0 + lr;
                float v = 0.0f;
                if (r < n_ff) {
                    float g = ag0[i] * sc;
                    float u = au0[i] * sc;
                    float sg = 1.0f / (1.0f + expf(-g));
                    float act = beta * tanhf(g / beta) * sg;
                    float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                    v = act * up;
                }
                sm.act[n0 + col][lr] = v;
            }
        }
        if (n1 + col < nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int lr = m0 + half * 8 + i;
                int r = blk_row0 + lr;
                float v = 0.0f;
                if (r < n_ff) {
                    float g = ag1[i] * sc;
                    float u = au1[i] * sc;
                    float sg = 1.0f / (1.0f + expf(-g));
                    float act = beta * tanhf(g / beta) * sg;
                    float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                    v = act * up;
                }
                sm.act[n1 + col][lr] = v;
            }
        }
        __syncthreads();
        for (int idx = warp; idx < nt * 2; idx += 8) {
            int tk = idx >> 1;
            int blk = idx & 1;
            if (blk_base + blk < qblk_per_slot) {
                float v = sm.act[tk][blk * 32 + lane];
                float amax = __shfl(w_fmax32_top(fabsf(v)), 0, 32);
                float d = amax / 127.0f;
                float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                int qi = __float2int_rn(v * inv);
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                int32_t qsum = w_isum32_top(qi);
                int q1 = __shfl_down(qi, 1, 32);
                int q2 = __shfl_down(qi, 2, 32);
                int q3 = __shfl_down(qi, 3, 32);
                int t_idx = off0 + t0 + tk;
                w_q8_1_blk* dst = act_q + ((size_t)tok_sorted[t_idx] * n_used + slot_sorted[t_idx]) * qblk_per_slot + blk_base + blk;
                if (lane == 0) {
                    dst->d = d;
                    dst->s = qsum;
                }
                if ((lane & 3) == 0) {
                    dst->q[lane >> 2] = w_p4i8_top((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void moe_dn_q4_0_wmma_k(const uint8_t* __restrict__ W,
                                              const w_q8_1_blk* __restrict__ act_q,
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

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int wm = warp >> 1;
    int wn = warp & 1;
    int blk_row0 = blockIdx.x * 128;
    if (blk_row0 >= n_embd) return;

    int nblk = n_ff >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * rbytes;

    __shared__ uint4 sm_wq[4][128];
    __shared__ float sm_dw[4][128];
    __shared__ int2 sm_aq[4][4][64];
    __shared__ float sm_ad[4][64];
    __shared__ int32_t sm_as[4][64];

    int r_loc = tid >> 2;
    int kb_load = tid & 3;
    int r_g0 = min(blk_row0 + r_loc, n_embd - 1);
    int r_g1 = min(blk_row0 + r_loc + 64, n_embd - 1);
    const uint8_t* wr0 = W + (size_t)exp_id * exp_bytes + (size_t)r_g0 * rbytes;
    const uint8_t* wr1 = W + (size_t)exp_id * exp_bytes + (size_t)r_g1 * rbytes;
    const uint4* qr0 = reinterpret_cast<const uint4*>(wr0);
    const uint4* qr1 = reinterpret_cast<const uint4*>(wr1);
    const __half* dr0 = reinterpret_cast<const __half*>(wr0 + (size_t)nblk * 16);
    const __half* dr1 = reinterpret_cast<const __half*>(wr1 + (size_t)nblk * 16);

    int m0 = wm * 32;
    int m1 = m0 + 16;
    int n0 = wn * 32;
    int n1 = n0 + 16;

    int z_id = blockIdx.z;
    int z_n = gridDim.z;
    for (int t0 = z_id * 64; t0 < cnt; t0 += z_n * 64) {
        int nt = min(64, cnt - t0);
        int idx_l0 = off0 + t0 + min(r_loc, nt - 1);
        const w_q8_1_blk* ar0 = act_q + ((size_t)tok_sorted[idx_l0] * n_used + slot_sorted[idx_l0]) * nblk;

        float acc00[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float acc01[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float acc10[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float acc11[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        for (int b0 = 0; b0 < nblk; b0 += 4) {
            int b = b0 + kb_load;
            if (b < nblk) {
                sm_wq[kb_load][r_loc] = qr0[b];
                sm_dw[kb_load][r_loc] = __half2float(dr0[b]);
                sm_wq[kb_load][r_loc + 64] = qr1[b];
                sm_dw[kb_load][r_loc + 64] = __half2float(dr1[b]);
                const w_q8_1_blk ab0 = ar0[b];
                sm_aq[kb_load][0][r_loc] = make_int2(ab0.q[0], ab0.q[1]);
                sm_aq[kb_load][1][r_loc] = make_int2(ab0.q[2], ab0.q[3]);
                sm_aq[kb_load][2][r_loc] = make_int2(ab0.q[4], ab0.q[5]);
                sm_aq[kb_load][3][r_loc] = make_int2(ab0.q[6], ab0.q[7]);
                sm_ad[kb_load][r_loc] = ab0.d;
                sm_as[kb_load][r_loc] = 8 * ab0.s;
            } else {
                sm_wq[kb_load][r_loc] = make_uint4(0, 0, 0, 0);
                sm_dw[kb_load][r_loc] = 0.0f;
                sm_wq[kb_load][r_loc + 64] = make_uint4(0, 0, 0, 0);
                sm_dw[kb_load][r_loc + 64] = 0.0f;
                sm_ad[kb_load][r_loc] = 0.0f;
                sm_as[kb_load][r_loc] = 0;
            }
            __syncthreads();

#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                uint4 q0 = sm_wq[kb][m0 + col];
                uint32_t w00 = half ? q0.z : q0.x;
                uint32_t w01 = half ? q0.w : q0.y;
                wmma_i2_v a0_lo = {(int32_t)(w00 & 0x0F0F0F0Fu), (int32_t)(w01 & 0x0F0F0F0Fu)};
                wmma_i2_v a0_hi = {(int32_t)((w00 >> 4) & 0x0F0F0F0Fu), (int32_t)((w01 >> 4) & 0x0F0F0F0Fu)};

                uint4 q1 = sm_wq[kb][m1 + col];
                uint32_t w10 = half ? q1.z : q1.x;
                uint32_t w11 = half ? q1.w : q1.y;
                wmma_i2_v a1_lo = {(int32_t)(w10 & 0x0F0F0F0Fu), (int32_t)(w11 & 0x0F0F0F0Fu)};
                wmma_i2_v a1_hi = {(int32_t)((w10 >> 4) & 0x0F0F0F0Fu), (int32_t)((w11 >> 4) & 0x0F0F0F0Fu)};

                int2 x0l = sm_aq[kb][half][n0 + col];
                int2 x0h = sm_aq[kb][2 + half][n0 + col];
                wmma_i2_v b0_lo = {x0l.x, x0l.y};
                wmma_i2_v b0_hi = {x0h.x, x0h.y};

                int2 x1l = sm_aq[kb][half][n1 + col];
                int2 x1h = sm_aq[kb][2 + half][n1 + col];
                wmma_i2_v b1_lo = {x1l.x, x1l.y};
                wmma_i2_v b1_hi = {x1h.x, x1h.y};

                wmma_i8_v c00 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v c01 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v c10 = {0, 0, 0, 0, 0, 0, 0, 0};
                wmma_i8_v c11 = {0, 0, 0, 0, 0, 0, 0, 0};
                c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_lo, true, b0_lo, c00, false);
                c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_hi, true, b0_hi, c00, false);
                c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_lo, true, b1_lo, c01, false);
                c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a0_hi, true, b1_hi, c01, false);
                c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_lo, true, b0_lo, c10, false);
                c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_hi, true, b0_hi, c10, false);
                c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_lo, true, b1_lo, c11, false);
                c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a1_hi, true, b1_hi, c11, false);

                float ad0 = sm_ad[kb][n0 + col];
                int32_t as0 = sm_as[kb][n0 + col];
                float ad1 = sm_ad[kb][n1 + col];
                int32_t as1 = sm_as[kb][n1 + col];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float dw0 = sm_dw[kb][m0 + half * 8 + i];
                    float dw1 = sm_dw[kb][m1 + half * 8 + i];
                    acc00[i] = fmaf(dw0 * ad0, (float)(c00[i] - as0), acc00[i]);
                    acc01[i] = fmaf(dw0 * ad1, (float)(c01[i] - as1), acc01[i]);
                    acc10[i] = fmaf(dw1 * ad0, (float)(c10[i] - as0), acc10[i]);
                    acc11[i] = fmaf(dw1 * ad1, (float)(c11[i] - as1), acc11[i]);
                }
            }
            __syncthreads();
        }
        if (n0 + col < nt) {
            int idx0 = off0 + t0 + n0 + col;
            float wt0 = wt_sorted[idx0];
            float* dst = y_out + ((size_t)tok_sorted[idx0] * n_used + slot_sorted[idx0]) * n_embd;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r0 = blk_row0 + m0 + half * 8 + i;
                if (r0 < n_embd) dst[r0] = acc00[i] * wt0;
                int r1 = blk_row0 + m1 + half * 8 + i;
                if (r1 < n_embd) dst[r1] = acc10[i] * wt0;
            }
        }
        if (n1 + col < nt) {
            int idx1 = off0 + t0 + n1 + col;
            float wt1 = wt_sorted[idx1];
            float* dst = y_out + ((size_t)tok_sorted[idx1] * n_used + slot_sorted[idx1]) * n_embd;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r0 = blk_row0 + m0 + half * 8 + i;
                if (r0 < n_embd) dst[r0] = acc01[i] * wt1;
                int r1 = blk_row0 + m1 + half * 8 + i;
                if (r1 < n_embd) dst[r1] = acc11[i] * wt1;
            }
        }
    }
}

extern "C" __global__ void reduce_moe_slots_k(const float* __restrict__ slots,
                                              float* __restrict__ y_out,
                                              int n_embd, int n_used) {
    int r4 = blockIdx.x * blockDim.x + threadIdx.x;
    int e4 = n_embd >> 2;
    int t = blockIdx.y;
    if (r4 >= e4) return;
    const float4* base = reinterpret_cast<const float4*>(slots + (size_t)t * n_used * n_embd) + r4;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < n_used) {
            float4 v = base[(size_t)s * e4];
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }
    reinterpret_cast<float4*>(y_out + (size_t)t * n_embd)[r4] = sum;
}

extern "C" __global__ void gemm_f32_wmma_k(const float* __restrict__ W,
                                           const float* __restrict__ X,
                                           float* __restrict__ Y,
                                           int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int row0 = blockIdx.x * 16;
    int t0 = blockIdx.y * 16;
    if (row0 >= n || t0 >= B) return;

    int r_load = min(row0 + col, n - 1);
    int t_load = min(t0 + col, B - 1);
    const float* wrow = W + (size_t)r_load * k;
    const float* xrow = X + (size_t)t_load * k;

    wmma_f8_v cv = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int kk = warp * 16; kk < k; kk += 64) {
        int kb = kk + half * 8;
        wmma_h8_v av, bv;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            av[j] = (_Float16)wrow[kb + j];
            bv[j] = (_Float16)xrow[kb + j];
        }
        cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, cv);
    }
    __shared__ float sm[4][16][16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm[warp][half * 8 + i][col] = cv[i];
    }
    __syncthreads();
    if (warp == 0 && t0 + col < B) {
        float* yt = Y + (size_t)(t0 + col) * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = row0 + half * 8 + i;
            if (r < n) {
                int rr = half * 8 + i;
                yt[r] = sm[0][rr][col] + sm[1][rr][col] + sm[2][rr][col] + sm[3][rr][col];
            }
        }
    }
}

extern "C" __global__ void loop_ctrl_down_wmma_k(const float* __restrict__ cur0,
                                                 const float* __restrict__ cur1,
                                                 const float* __restrict__ cur2,
                                                 const float* __restrict__ msgs,
                                                 const float* __restrict__ wd0,
                                                 const float* __restrict__ wd1,
                                                 const float* __restrict__ wd2,
                                                 float* __restrict__ h_out,
                                                 int n_embd, int rank, int max_B, int B) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int wm = warp >> 1;
    int wn = warp & 1;
    int blk_row0 = blockIdx.x * 32;
    int c = blockIdx.y;
    int t0 = blockIdx.z * 32;
    if (blk_row0 >= rank || t0 >= B) return;

    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* wd = (c == 0) ? wd0 : ((c == 1) ? wd1 : wd2);

    __shared__ float rc_s[32];
    __shared__ float rm_s[32];
    int nt = min(32, B - t0);
    int e4 = n_embd >> 2;
    for (int m = warp; m < nt; m += 4) {
        const float4* cp = reinterpret_cast<const float4*>(cur_base + (size_t)(t0 + m) * n_embd);
        const float4* mp = reinterpret_cast<const float4*>(msg_base + (size_t)(t0 + m) * n_embd);
        float sc = 0.0f, sm = 0.0f;
        for (int j = lane; j < e4; j += 32) {
            float4 vc = cp[j], vm = mp[j];
            sc = fmaf(vc.x, vc.x, fmaf(vc.y, vc.y, fmaf(vc.z, vc.z, fmaf(vc.w, vc.w, sc))));
            sm = fmaf(vm.x, vm.x, fmaf(vm.y, vm.y, fmaf(vm.z, vm.z, fmaf(vm.w, vm.w, sm))));
        }
        sc = w_warp_sum32(sc);
        sm = w_warp_sum32(sm);
        if (lane == 0) {
            rc_s[m] = rsqrtf(sc / (float)n_embd + 1e-6f);
            rm_s[m] = rsqrtf(sm / (float)n_embd + 1e-6f);
        }
    }
    __syncthreads();

    __shared__ _Float16 sm_w0[32][24];
    __shared__ _Float16 sm_w1[32][24];
    __shared__ _Float16 sm_x0[32][24];
    __shared__ _Float16 sm_x1[32][24];

    int r_loc = tid >> 2;
    int k4_loc = tid & 3;
    int r_g = min(blk_row0 + r_loc, rank - 1);
    int m_g = min(r_loc, nt - 1);
    const float4* wr0 = reinterpret_cast<const float4*>(wd + (size_t)r_g * (2 * n_embd));
    const float4* wr1 = reinterpret_cast<const float4*>(wd + (size_t)r_g * (2 * n_embd) + n_embd);
    const float4* cr = reinterpret_cast<const float4*>(cur_base + (size_t)(t0 + m_g) * n_embd);
    const float4* mr = reinterpret_cast<const float4*>(msg_base + (size_t)(t0 + m_g) * n_embd);
    float rc = rc_s[m_g];
    float rm = rm_s[m_g];

    int m0 = wm * 16;
    int n0 = wn * 16;
    wmma_f8_v cv = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int kk4 = 0; kk4 < e4; kk4 += 4) {
        int j4 = kk4 + k4_loc;
        int base_k = k4_loc * 4;
        float4 vw0 = wr0[j4];
        float4 vw1 = wr1[j4];
        float4 vc = cr[j4];
        float4 vm = mr[j4];
        sm_w0[r_loc][base_k + 0] = (_Float16)vw0.x; sm_w0[r_loc][base_k + 1] = (_Float16)vw0.y;
        sm_w0[r_loc][base_k + 2] = (_Float16)vw0.z; sm_w0[r_loc][base_k + 3] = (_Float16)vw0.w;
        sm_w1[r_loc][base_k + 0] = (_Float16)vw1.x; sm_w1[r_loc][base_k + 1] = (_Float16)vw1.y;
        sm_w1[r_loc][base_k + 2] = (_Float16)vw1.z; sm_w1[r_loc][base_k + 3] = (_Float16)vw1.w;
        sm_x0[r_loc][base_k + 0] = (_Float16)(vc.x * rc); sm_x0[r_loc][base_k + 1] = (_Float16)(vc.y * rc);
        sm_x0[r_loc][base_k + 2] = (_Float16)(vc.z * rc); sm_x0[r_loc][base_k + 3] = (_Float16)(vc.w * rc);
        sm_x1[r_loc][base_k + 0] = (_Float16)(vm.x * rm); sm_x1[r_loc][base_k + 1] = (_Float16)(vm.y * rm);
        sm_x1[r_loc][base_k + 2] = (_Float16)(vm.z * rm); sm_x1[r_loc][base_k + 3] = (_Float16)(vm.w * rm);
        __syncthreads();

        wmma_h8_v aw0, ax0, aw1, ax1;
        int kb = half * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            aw0[j] = sm_w0[m0 + col][kb + j];
            ax0[j] = sm_x0[n0 + col][kb + j];
            aw1[j] = sm_w1[m0 + col][kb + j];
            ax1[j] = sm_x1[n0 + col][kb + j];
        }
        cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aw0, ax0, cv);
        cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aw1, ax1, cv);
        __syncthreads();
    }
    if (n0 + col < nt) {
        float* ht = h_out + ((size_t)c * max_B + (t0 + n0 + col)) * rank;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = blk_row0 + m0 + half * 8 + i;
            if (r < rank) ht[r] = cv[i];
        }
    }
}

extern "C" __global__ void loop_ctrl_up_wmma_k(const float* __restrict__ cur0,
                                               const float* __restrict__ cur1,
                                               const float* __restrict__ cur2,
                                               const float* __restrict__ msgs,
                                               const float* __restrict__ h_in,
                                               const float* __restrict__ wg0,
                                               const float* __restrict__ wg1,
                                               const float* __restrict__ wg2,
                                               const float* __restrict__ wf0,
                                               const float* __restrict__ wf1,
                                               const float* __restrict__ wf2,
                                               float* __restrict__ out0,
                                               float* __restrict__ out1,
                                               float* __restrict__ out2,
                                               int n_embd, int rank, int max_B, int B) {
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int blk_row0 = blockIdx.x * 64;
    int c = blockIdx.y;
    int t0 = blockIdx.z * 32;
    if (blk_row0 >= n_embd || t0 >= B) return;
    bool has1 = (t0 + 16 < B);

    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* h_base = h_in + (size_t)c * max_B * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    float* out_base = (c == 0) ? out0 : ((c == 1) ? out1 : out2);

    __shared__ _Float16 sm_wg[64][24];
    __shared__ _Float16 sm_wf[64][24];
    __shared__ _Float16 sm_h[32][24];

    int r_loc = tid >> 2;
    int k4_loc = tid & 3;
    int r_g0 = min(blk_row0 + r_loc, n_embd - 1);
    int r_g1 = min(blk_row0 + r_loc + 32, n_embd - 1);
    int t_g = min(t0 + r_loc, B - 1);
    const float4* wg_r0 = reinterpret_cast<const float4*>(wg + (size_t)r_g0 * rank);
    const float4* wg_r1 = reinterpret_cast<const float4*>(wg + (size_t)r_g1 * rank);
    const float4* wf_r0 = reinterpret_cast<const float4*>(wf + (size_t)r_g0 * rank);
    const float4* wf_r1 = reinterpret_cast<const float4*>(wf + (size_t)r_g1 * rank);
    const float4* h_r = reinterpret_cast<const float4*>(h_base + (size_t)t_g * rank);

    int m0 = warp * 16;
    wmma_f8_v cg0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    wmma_f8_v cf0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    wmma_f8_v cg1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    wmma_f8_v cf1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int r4 = rank >> 2;

    for (int kk4 = 0; kk4 < r4; kk4 += 4) {
        int j4 = kk4 + k4_loc;
        int bk = k4_loc * 4;
        float4 g0v = wg_r0[j4], g1v = wg_r1[j4];
        float4 f0v = wf_r0[j4], f1v = wf_r1[j4];
        float4 hv = h_r[j4];
        sm_wg[r_loc][bk + 0] = (_Float16)g0v.x; sm_wg[r_loc][bk + 1] = (_Float16)g0v.y;
        sm_wg[r_loc][bk + 2] = (_Float16)g0v.z; sm_wg[r_loc][bk + 3] = (_Float16)g0v.w;
        sm_wg[r_loc + 32][bk + 0] = (_Float16)g1v.x; sm_wg[r_loc + 32][bk + 1] = (_Float16)g1v.y;
        sm_wg[r_loc + 32][bk + 2] = (_Float16)g1v.z; sm_wg[r_loc + 32][bk + 3] = (_Float16)g1v.w;
        sm_wf[r_loc][bk + 0] = (_Float16)f0v.x; sm_wf[r_loc][bk + 1] = (_Float16)f0v.y;
        sm_wf[r_loc][bk + 2] = (_Float16)f0v.z; sm_wf[r_loc][bk + 3] = (_Float16)f0v.w;
        sm_wf[r_loc + 32][bk + 0] = (_Float16)f1v.x; sm_wf[r_loc + 32][bk + 1] = (_Float16)f1v.y;
        sm_wf[r_loc + 32][bk + 2] = (_Float16)f1v.z; sm_wf[r_loc + 32][bk + 3] = (_Float16)f1v.w;
        sm_h[r_loc][bk + 0] = (_Float16)hv.x; sm_h[r_loc][bk + 1] = (_Float16)hv.y;
        sm_h[r_loc][bk + 2] = (_Float16)hv.z; sm_h[r_loc][bk + 3] = (_Float16)hv.w;
        __syncthreads();

        wmma_h8_v ag, af, bh0, bh1;
        int kb = half * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            ag[j] = sm_wg[m0 + col][kb + j];
            af[j] = sm_wf[m0 + col][kb + j];
            bh0[j] = sm_h[col][kb + j];
            bh1[j] = has1 ? sm_h[16 + col][kb + j] : (_Float16)0.0f;
        }
        cg0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ag, bh0, cg0);
        cf0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bh0, cf0);
        if (has1) {
            cg1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ag, bh1, cg1);
            cf1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bh1, cf1);
        }
        __syncthreads();
    }
    if (t0 + col < B) {
        size_t base = (size_t)(t0 + col) * n_embd;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = blk_row0 + m0 + half * 8 + i;
            if (r < n_embd) {
                float cv = cur_base[base + r];
                out_base[base + r] = cv - tanhf(cf0[i]) * cv + tanhf(cg0[i]) * msg_base[base + r];
            }
        }
    }
    if (has1 && t0 + 16 + col < B) {
        size_t base = (size_t)(t0 + 16 + col) * n_embd;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = blk_row0 + m0 + half * 8 + i;
            if (r < n_embd) {
                float cv = cur_base[base + r];
                out_base[base + r] = cv - tanhf(cf1[i]) * cv + tanhf(cg1[i]) * msg_base[base + r];
            }
        }
    }
}

extern "C" __global__ void attn_batch_gated_flash_k(const float* __restrict__ Q,
                                                    const float* __restrict__ R,
                                                    const float* __restrict__ sinks,
                                                    const float* __restrict__ rel_bias,
                                                    const __half* __restrict__ k_cache,
                                                    const __half* __restrict__ v_cache,
                                                    const float* __restrict__ gate,
                                                    const int* __restrict__ d_pos0,
                                                    float* __restrict__ out,
                                                    int n_head, int n_head_kv,
                                                    int dh, int d_rel,
                                                    int rel_extent, int swa_window,
                                                    int is_swa, int max_ctx, int B) {
    int h = blockIdx.x;
    int bt_blk = (gridDim.y - 1 - blockIdx.y) * 64;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int kv_h = h / (n_head / n_head_kv);
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    int bt_w = bt_blk + wid * 16;
    int bt_q = bt_w + col;
    bool vq = (bt_q < B);
    int pos_q = pos0 + bt_q;
    int t_start_q = is_swa ? max(0, pos_q - swa_window + 1) : 0;

    int t_min = is_swa ? max(0, pos0 + bt_blk - swa_window + 1) : 0;
    int t_max = pos0 + min(bt_blk + 64, B);

    float q_scale = rsqrtf((float)dh);
    wmma_h8_v q_reg[8];
    float4 r_reg0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg3 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sink_q = 0.0f;
    if (vq) {
        const float* qh = Q + (size_t)bt_q * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
            }
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)bt_q * r_dim + (size_t)h * d_rel);
        r_reg0 = rh4[0];
        r_reg1 = rh4[1];
        r_reg2 = rh4[2];
        r_reg3 = rh4[3];
        sink_q = sinks[(size_t)bt_q * n_head + h];
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = 0.0f;
    float l_q = 1.0f;
    wmma_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) {
        o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    const _Float16* kc_h = reinterpret_cast<const _Float16*>(k_cache) + (size_t)kv_h * max_ctx * dh;
    const _Float16* vc_h = reinterpret_cast<const _Float16*>(v_cache) + (size_t)kv_h * max_ctx * dh;

    __shared__ _Float16 sm_k[16][136];
    __shared__ _Float16 sm_vt[128][24];

    for (int tile = t_min; tile < t_max; tile += 16) {
        int n_kv = min(16, t_max - tile);
        int tk_l = threadIdx.x >> 3;
        int d8_l = (threadIdx.x & 7) * 16;
        if (tk_l < n_kv) {
            int kv_slot = is_swa ? ((tile + tk_l) % max_ctx) : (tile + tk_l);
            const _Float16* kp = kc_h + (size_t)kv_slot * dh + d8_l;
            const _Float16* vp = vc_h + (size_t)kv_slot * dh + d8_l;
            *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l]) = *reinterpret_cast<const uint4*>(kp);
            *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l + 8]) = *reinterpret_cast<const uint4*>(kp + 8);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                sm_vt[d8_l + j][tk_l] = vp[j];
            }
        } else {
            *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l + 8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                sm_vt[d8_l + j][tk_l] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        wmma_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v kv_v;
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) kv_v[j] = sm_k[col][dbase + j];
            s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
        }

        float local_max = -1e30f;
        float sc_arr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int tk = half * 8 + i;
            int t = tile + tk;
            if (vq && tk < n_kv && t >= t_start_q && t <= pos_q) {
                float sc = s_v[i] - sink_q;
                int dist = pos_q - t;
                if (dist < rel_extent) {
                    const float4* rbp4 = reinterpret_cast<const float4*>(rel_bias + (size_t)dist * d_rel);
                    float4 b0 = rbp4[0], b1 = rbp4[1], b2 = rbp4[2], b3 = rbp4[3];
                    float rp = fmaf(b0.x, r_reg0.x, fmaf(b0.y, r_reg0.y, fmaf(b0.z, r_reg0.z, b0.w * r_reg0.w)));
                    rp = fmaf(b1.x, r_reg1.x, fmaf(b1.y, r_reg1.y, fmaf(b1.z, r_reg1.z, fmaf(b1.w, r_reg1.w, rp))));
                    rp = fmaf(b2.x, r_reg2.x, fmaf(b2.y, r_reg2.y, fmaf(b2.z, r_reg2.z, fmaf(b2.w, r_reg2.w, rp))));
                    rp = fmaf(b3.x, r_reg3.x, fmaf(b3.y, r_reg3.y, fmaf(b3.z, r_reg3.z, fmaf(b3.w, r_reg3.w, rp))));
                    sc += rp;
                }
                sc_arr[i] = sc;
                local_max = fmaxf(local_max, sc);
            } else {
                sc_arr[i] = -1e30f;
            }
        }
        float other_max = __shfl_xor(local_max, 16, 32);
        float tile_max = fmaxf(local_max, other_max);
        float m_new = fmaxf(m_q, tile_max);
        float alpha = expf(m_q - m_new);

        wmma_h8_v p_reg;
        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float p = (sc_arr[i] > -1e20f) ? expf(sc_arr[i] - m_new) : 0.0f;
            p_reg[i] = (_Float16)p;
            local_sum += p;
        }
        float other_sum = __shfl_xor(local_sum, 16, 32);
        l_q = l_q * alpha + local_sum + other_sum;
        m_q = m_new;

#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v vt_v;
            int d_row = db * 16 + col;
#pragma unroll
            for (int j = 0; j < 8; ++j) vt_v[j] = sm_vt[d_row][half * 8 + j];
            o_acc[db] *= alpha;
            o_acc[db] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_v, p_reg, o_acc[db]);
        }
        __syncthreads();
    }

    if (vq) {
        float inv_l = 1.0f / l_q;
        float* oh = out + (size_t)bt_q * q_dim + (size_t)h * dh;
        const float* gh = gate + (size_t)bt_q * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int d = db * 16 + half * 8 + i;
                oh[d] = (o_acc[db][i] * inv_l) / (1.0f + expf(-gh[d]));
            }
        }
    }
}

extern "C" int ink_gemm_q4_0_wmma(hipStream_t s, const uint8_t* W, const void* Xq, float* Y, int n, int k, int B) {
    if (B >= 48) {
        gemm_q4_0_wmma64_k<<<dim3((n + 127) / 128, (B + 63) / 64), dim3(256), 0, s>>>(W, (const w_q8_1_blk*)Xq, Y, n, k, B);
    } else {
        gemm_q4_0_wmma_k<<<dim3((n + 63) / 64, (B + 31) / 32), dim3(128), 0, s>>>(W, (const w_q8_1_blk*)Xq, Y, n, k, B);
    }
    return 0;
}

extern "C" int ink_qkvrg_gemm_q4_wmma(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim, int B) {
    int qkv_rows = q_dim + kv_dim + kv_dim + r_dim + q_dim;
    if (B >= 48 && (q_dim & 127) == 0 && (kv_dim & 127) == 0 && (r_dim & 127) == 0) {
        qkvrg_gemm_q4_wmma64_k<<<dim3((qkv_rows + 127) / 128, (B + 63) / 64), dim3(256), 0, s>>>(wq, wk, wv, wr, wg, (const w_q8_1_blk*)Xq0, (const w_q8_1_blk*)Xq1, (const w_q8_1_blk*)Xq2, Q, K, V, R, G, k, q_dim, kv_dim, r_dim, B);
    } else {
        qkvrg_gemm_q4_wmma_k<<<dim3((qkv_rows + 63) / 64, (B + 31) / 32), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, (const w_q8_1_blk*)Xq0, (const w_q8_1_blk*)Xq1, (const w_q8_1_blk*)Xq2, Q, K, V, R, G, k, q_dim, kv_dim, r_dim, B);
    }
    int sd_rows = n_head + 2 * dyn_dim;
    if (B >= 16 && (k & 31) == 0 && n_head <= 32 && dyn_dim <= 16) {
        int n_parts = (dyn_dim > 0) ? 4 : ((n_head > 16) ? 2 : 1);
        sink_dyn_bf16_wmma_k<<<dim3(n_parts, (B + 63) / 64), dim3(128), 0, s>>>(wsink, wkdyn, wvdyn, X0, X1, X2, S, Kdyn, Vdyn, k, n_head, dyn_dim, B);
    } else {
        sink_dyn_bf16_k<<<dim3((sd_rows + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(wsink, wkdyn, wvdyn, X0, X1, X2, S, Kdyn, Vdyn, k, n_head, dyn_dim, B);
    }
    return 0;
}

extern "C" int ink_moe_gu_q4_0_wmma(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, float* act_out, int n_ff, int k, int n_used, float beta, float lb, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    int gz = (B >= 512) ? 6 : ((B >= 128) ? 2 : 1);
    moe_gu_q4_0_wmma_k<<<dim3((n_ff + 63) / 64, gy, gz), dim3(256), 0, s>>>(W, scales, (const w_q8_1_blk*)Xq, exp_counts, exp_offsets, tok_sorted, slot_sorted, act_out, n_ff, k, n_used, beta, lb);
    return 0;
}

extern "C" int ink_moe_gu_q4_0_wmma_to_q8(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, void* act_q, int n_ff, int k, int n_used, float beta, float lb, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    int gz = (B >= 512) ? 6 : ((B >= 128) ? 2 : 1);
    moe_gu_q4_0_wmma_q8_k<<<dim3((n_ff + 63) / 64, gy, gz), dim3(256), 0, s>>>(W, scales, (const w_q8_1_blk*)Xq, exp_counts, exp_offsets, tok_sorted, slot_sorted, (w_q8_1_blk*)act_q, n_ff, k, n_used, beta, lb);
    return 0;
}

extern "C" int ink_moe_dn_q4_0_wmma(hipStream_t s, const uint8_t* W, const void* act_q, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, const float* wt_sorted, float* moe_slots, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    int gz = (B >= 512) ? 6 : ((B >= 128) ? 2 : 1);
    moe_dn_q4_0_wmma_k<<<dim3((n_embd + 127) / 128, gy, gz), dim3(256), 0, s>>>(W, (const w_q8_1_blk*)act_q, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, moe_slots, n_embd, n_ff, n_used);
    reduce_moe_slots_k<<<dim3(((n_embd >> 2) + 255) / 256, B), dim3(256), 0, s>>>(moe_slots, y_out, n_embd, n_used);
    return 0;
}

__device__ __forceinline__ float w_fmax32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
    return v;
}

__device__ __forceinline__ int32_t w_isum32(int32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ int32_t w_p4i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int32_t)(uint8_t)a) | (((int32_t)(uint8_t)b) << 8) | (((int32_t)(uint8_t)c) << 16) | (((int32_t)(uint8_t)d) << 24);
}

__device__ __forceinline__ void w_q8_store(float v, int lane, w_q8_1_blk* dst, int8_t* sm) {
    float amax = __shfl(w_fmax32(fabsf(v)), 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    int32_t qsum = w_isum32(qi);
    sm[lane] = (int8_t)qi;
    if (lane == 0) { dst->d = d; dst->s = qsum; }
    if (lane < 8) dst->q[lane] = w_p4i8(sm[lane * 4], sm[lane * 4 + 1], sm[lane * 4 + 2], sm[lane * 4 + 3]);
}

extern "C" __global__ void attn_b8_flash_q8_k(const float* __restrict__ Q,
                                              const float* __restrict__ R,
                                              const float* __restrict__ sinks,
                                              const float* __restrict__ rel_bias,
                                              const __half* __restrict__ k_cache,
                                              const __half* __restrict__ v_cache,
                                              const float* __restrict__ gate,
                                              const int* __restrict__ d_pos0,
                                              w_q8_1_blk* __restrict__ out_q8,
                                              int n_head, int n_head_kv,
                                              int dh, int d_rel,
                                              int rel_extent, int swa_window,
                                              int is_swa, int max_ctx, int B) {
    int h = blockIdx.x;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int kv_h = h / (n_head / n_head_kv);
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    bool vq = (col < B);
    int pos_q = pos0 + col;
    int t_start_q = is_swa ? max(0, pos_q - swa_window + 1) : 0;
    int t_min = is_swa ? max(0, pos0 - swa_window + 1) : 0;
    int t_max = pos0 + B;

    float q_scale = rsqrtf((float)dh);
    wmma_h8_v q_reg[8];
    float4 r_reg0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg3 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sink_q = 0.0f;
    if (vq) {
        const float* qh = Q + (size_t)col * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)col * r_dim + (size_t)h * d_rel);
        r_reg0 = rh4[0]; r_reg1 = rh4[1]; r_reg2 = rh4[2]; r_reg3 = rh4[3];
        sink_q = sinks[(size_t)col * n_head + h];
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = (wid == 0) ? 0.0f : -1e30f;
    float l_q = (wid == 0) ? 1.0f : 0.0f;
    wmma_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const _Float16* kc_h = reinterpret_cast<const _Float16*>(k_cache) + (size_t)kv_h * max_ctx * dh;
    const _Float16* vc_h = reinterpret_cast<const _Float16*>(v_cache) + (size_t)kv_h * max_ctx * dh;

    __shared__ union {
        struct {
            _Float16 k[4][16][136];
            _Float16 vt[4][128][24];
        } kv;
        float o[4][16][128];
    } sm_u;

    for (int tile = t_min + wid * 16; tile < t_max; tile += 64) {
        int n_kv = min(16, t_max - tile);
        for (int tk = half; tk < 16; tk += 2) {
            int d8 = col * 8;
            if (tk < n_kv) {
                int kv_slot = is_swa ? ((tile + tk) % max_ctx) : (tile + tk);
                const _Float16* kp = kc_h + (size_t)kv_slot * dh + d8;
                const _Float16* vp = vc_h + (size_t)kv_slot * dh + d8;
                *reinterpret_cast<uint4*>(&sm_u.kv.k[wid][tk][d8]) = *reinterpret_cast<const uint4*>(kp);
#pragma unroll
                for (int j = 0; j < 8; ++j) sm_u.kv.vt[wid][d8 + j][tk] = vp[j];
            } else {
                *reinterpret_cast<uint4*>(&sm_u.kv.k[wid][tk][d8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 8; ++j) sm_u.kv.vt[wid][d8 + j][tk] = (_Float16)0.0f;
            }
        }
        __builtin_amdgcn_wave_barrier();

        wmma_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v kv_v;
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) kv_v[j] = sm_u.kv.k[wid][col][dbase + j];
            s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
        }

        float local_max = -1e30f;
        float sc_arr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int tk = half * 8 + i;
            int t = tile + tk;
            if (vq && tk < n_kv && t >= t_start_q && t <= pos_q) {
                float sc = s_v[i] - sink_q;
                int dist = pos_q - t;
                if (dist < rel_extent) {
                    const float4* rbp4 = reinterpret_cast<const float4*>(rel_bias + (size_t)dist * d_rel);
                    float4 b0 = rbp4[0], b1 = rbp4[1], b2 = rbp4[2], b3 = rbp4[3];
                    float rp = fmaf(b0.x, r_reg0.x, fmaf(b0.y, r_reg0.y, fmaf(b0.z, r_reg0.z, b0.w * r_reg0.w)));
                    rp = fmaf(b1.x, r_reg1.x, fmaf(b1.y, r_reg1.y, fmaf(b1.z, r_reg1.z, fmaf(b1.w, r_reg1.w, rp))));
                    rp = fmaf(b2.x, r_reg2.x, fmaf(b2.y, r_reg2.y, fmaf(b2.z, r_reg2.z, fmaf(b2.w, r_reg2.w, rp))));
                    rp = fmaf(b3.x, r_reg3.x, fmaf(b3.y, r_reg3.y, fmaf(b3.z, r_reg3.z, fmaf(b3.w, r_reg3.w, rp))));
                    sc += rp;
                }
                sc_arr[i] = sc;
                local_max = fmaxf(local_max, sc);
            } else {
                sc_arr[i] = -1e30f;
            }
        }
        float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
        float m_new = fmaxf(m_q, tile_max);
        float alpha = (m_q > -1e20f) ? expf(m_q - m_new) : 0.0f;

        wmma_h8_v p_reg;
        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float p = (sc_arr[i] > -1e20f) ? expf(sc_arr[i] - m_new) : 0.0f;
            p_reg[i] = (_Float16)p;
            local_sum += p;
        }
        l_q = l_q * alpha + local_sum + __shfl_xor(local_sum, 16, 32);
        m_q = m_new;

#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v vt_v;
            int d_row = db * 16 + col;
#pragma unroll
            for (int j = 0; j < 8; ++j) vt_v[j] = sm_u.kv.vt[wid][d_row][half * 8 + j];
            o_acc[db] *= alpha;
            o_acc[db] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_v, p_reg, o_acc[db]);
        }
        __builtin_amdgcn_wave_barrier();
    }
    __syncthreads();

    __shared__ float sm_m[4][16];
    __shared__ float sm_l[4][16];
    if (half == 0 && vq) {
        sm_m[wid][col] = m_q;
        sm_l[wid][col] = l_q;
    }
    if (vq) {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sm_u.o[wid][col][db * 16 + half * 8 + i] = o_acc[db][i];
            }
        }
    }
    __syncthreads();

    __shared__ int8_t smq[4][32];
    for (int bt = wid; bt < B; bt += 4) {
        float gm = fmaxf(fmaxf(sm_m[0][bt], sm_m[1][bt]), fmaxf(sm_m[2][bt], sm_m[3][bt]));
        float sc0 = expf(sm_m[0][bt] - gm);
        float sc1 = (sm_m[1][bt] > -1e20f) ? expf(sm_m[1][bt] - gm) : 0.0f;
        float sc2 = (sm_m[2][bt] > -1e20f) ? expf(sm_m[2][bt] - gm) : 0.0f;
        float sc3 = (sm_m[3][bt] > -1e20f) ? expf(sm_m[3][bt] - gm) : 0.0f;
        float inv_l = 1.0f / (sm_l[0][bt] * sc0 + sm_l[1][bt] * sc1 + sm_l[2][bt] * sc2 + sm_l[3][bt] * sc3);
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        size_t qbase = ((size_t)bt * n_head + h) * 4;
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            int d = blk * 32 + lane;
            float go = sm_u.o[0][bt][d] * sc0 + sm_u.o[1][bt][d] * sc1 + sm_u.o[2][bt][d] * sc2 + sm_u.o[3][bt][d] * sc3;
            float v = (go * inv_l) / (1.0f + expf(-gh[d]));
            w_q8_store(v, lane, out_q8 + qbase + blk, smq[wid]);
        }
    }
}

__device__ float g_b8_split_buf[32 * 4 * 16 * 130];

extern "C" __global__ void attn_b8_flash_split_k(const float* __restrict__ Q,
                                                 const float* __restrict__ R,
                                                 const float* __restrict__ sinks,
                                                 const float* __restrict__ rel_bias,
                                                 const __half* __restrict__ k_cache,
                                                 const __half* __restrict__ v_cache,
                                                 const int* __restrict__ d_pos0,
                                                 float* __restrict__ split_buf,
                                                 int n_head, int n_head_kv,
                                                 int dh, int d_rel,
                                                 int rel_extent, int swa_window,
                                                 int is_swa, int max_ctx, int B, int n_splits) {
    int h = blockIdx.x;
    int sp = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int kv_h = h / (n_head / n_head_kv);
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    bool vq = (col < B);
    int pos_q = pos0 + col;
    int t_start_q = is_swa ? max(0, pos_q - swa_window + 1) : 0;
    int t_min = is_swa ? max(0, pos0 - swa_window + 1) : 0;
    int t_max = pos0 + B;

    int total_tiles = (t_max - t_min + 15) / 16;
    int tiles_per_sp = (total_tiles + n_splits - 1) / n_splits;
    int tile_s = t_min + sp * tiles_per_sp * 16;
    int tile_e = min(t_max, tile_s + tiles_per_sp * 16);

    if (tile_s >= tile_e) {
        if (wid == 0 && half == 0 && vq) {
            float* dst = split_buf + (((size_t)h * n_splits + sp) * 16 + col) * 130;
            dst[0] = -1e30f;
            dst[1] = 0.0f;
        }
        return;
    }

    float q_scale = rsqrtf((float)dh);
    wmma_h8_v q_reg[8];
    float4 r_reg0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg3 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sink_q = 0.0f;
    if (vq) {
        const float* qh = Q + (size_t)col * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)col * r_dim + (size_t)h * d_rel);
        r_reg0 = rh4[0]; r_reg1 = rh4[1]; r_reg2 = rh4[2]; r_reg3 = rh4[3];
        sink_q = sinks[(size_t)col * n_head + h];
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = (sp == 0 && wid == 0) ? 0.0f : -1e30f;
    float l_q = (sp == 0 && wid == 0) ? 1.0f : 0.0f;
    wmma_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const _Float16* kc_h = reinterpret_cast<const _Float16*>(k_cache) + (size_t)kv_h * max_ctx * dh;
    const _Float16* vc_h = reinterpret_cast<const _Float16*>(v_cache) + (size_t)kv_h * max_ctx * dh;

    __shared__ union {
        struct {
            _Float16 k[4][16][136];
            _Float16 vt[4][128][24];
        } kv;
        float o[4][16][128];
    } sm_u;

    for (int tile = tile_s + wid * 16; tile < tile_e; tile += 64) {
        int n_kv = min(16, tile_e - tile);
        for (int tk = half; tk < 16; tk += 2) {
            int d8 = col * 8;
            if (tk < n_kv) {
                int kv_slot = is_swa ? ((tile + tk) % max_ctx) : (tile + tk);
                const _Float16* kp = kc_h + (size_t)kv_slot * dh + d8;
                const _Float16* vp = vc_h + (size_t)kv_slot * dh + d8;
                *reinterpret_cast<uint4*>(&sm_u.kv.k[wid][tk][d8]) = *reinterpret_cast<const uint4*>(kp);
#pragma unroll
                for (int j = 0; j < 8; ++j) sm_u.kv.vt[wid][d8 + j][tk] = vp[j];
            } else {
                *reinterpret_cast<uint4*>(&sm_u.kv.k[wid][tk][d8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 8; ++j) sm_u.kv.vt[wid][d8 + j][tk] = (_Float16)0.0f;
            }
        }
        __builtin_amdgcn_wave_barrier();

        wmma_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v kv_v;
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) kv_v[j] = sm_u.kv.k[wid][col][dbase + j];
            s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
        }

        float local_max = -1e30f;
        float sc_arr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int tk = half * 8 + i;
            int t = tile + tk;
            if (vq && tk < n_kv && t >= t_start_q && t <= pos_q) {
                float sc = s_v[i] - sink_q;
                int dist = pos_q - t;
                if (dist < rel_extent) {
                    const float4* rbp4 = reinterpret_cast<const float4*>(rel_bias + (size_t)dist * d_rel);
                    float4 b0 = rbp4[0], b1 = rbp4[1], b2 = rbp4[2], b3 = rbp4[3];
                    float rp = fmaf(b0.x, r_reg0.x, fmaf(b0.y, r_reg0.y, fmaf(b0.z, r_reg0.z, b0.w * r_reg0.w)));
                    rp = fmaf(b1.x, r_reg1.x, fmaf(b1.y, r_reg1.y, fmaf(b1.z, r_reg1.z, fmaf(b1.w, r_reg1.w, rp))));
                    rp = fmaf(b2.x, r_reg2.x, fmaf(b2.y, r_reg2.y, fmaf(b2.z, r_reg2.z, fmaf(b2.w, r_reg2.w, rp))));
                    rp = fmaf(b3.x, r_reg3.x, fmaf(b3.y, r_reg3.y, fmaf(b3.z, r_reg3.z, fmaf(b3.w, r_reg3.w, rp))));
                    sc += rp;
                }
                sc_arr[i] = sc;
                local_max = fmaxf(local_max, sc);
            } else {
                sc_arr[i] = -1e30f;
            }
        }
        float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
        float m_new = fmaxf(m_q, tile_max);
        float alpha = (m_q > -1e20f) ? expf(m_q - m_new) : 0.0f;

        wmma_h8_v p_reg;
        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float p = (sc_arr[i] > -1e20f) ? expf(sc_arr[i] - m_new) : 0.0f;
            p_reg[i] = (_Float16)p;
            local_sum += p;
        }
        l_q = l_q * alpha + local_sum + __shfl_xor(local_sum, 16, 32);
        m_q = m_new;

#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v vt_v;
            int d_row = db * 16 + col;
#pragma unroll
            for (int j = 0; j < 8; ++j) vt_v[j] = sm_u.kv.vt[wid][d_row][half * 8 + j];
            o_acc[db] *= alpha;
            o_acc[db] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_v, p_reg, o_acc[db]);
        }
        __builtin_amdgcn_wave_barrier();
    }
    __syncthreads();

    __shared__ float sm_m[4][16];
    __shared__ float sm_l[4][16];
    if (half == 0 && vq) {
        sm_m[wid][col] = m_q;
        sm_l[wid][col] = l_q;
    }
    if (vq) {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sm_u.o[wid][col][db * 16 + half * 8 + i] = o_acc[db][i];
            }
        }
    }
    __syncthreads();

    for (int bt = wid; bt < B; bt += 4) {
        float gm = fmaxf(fmaxf(sm_m[0][bt], sm_m[1][bt]), fmaxf(sm_m[2][bt], sm_m[3][bt]));
        float sc0 = (sm_m[0][bt] > -1e20f) ? expf(sm_m[0][bt] - gm) : 0.0f;
        float sc1 = (sm_m[1][bt] > -1e20f) ? expf(sm_m[1][bt] - gm) : 0.0f;
        float sc2 = (sm_m[2][bt] > -1e20f) ? expf(sm_m[2][bt] - gm) : 0.0f;
        float sc3 = (sm_m[3][bt] > -1e20f) ? expf(sm_m[3][bt] - gm) : 0.0f;
        float gl = sm_l[0][bt] * sc0 + sm_l[1][bt] * sc1 + sm_l[2][bt] * sc2 + sm_l[3][bt] * sc3;
        float* dst = split_buf + (((size_t)h * n_splits + sp) * 16 + bt) * 130;
        if (lane == 0) { dst[0] = gm; dst[1] = gl; }
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            int d = blk * 32 + lane;
            dst[2 + d] = sm_u.o[0][bt][d] * sc0 + sm_u.o[1][bt][d] * sc1 + sm_u.o[2][bt][d] * sc2 + sm_u.o[3][bt][d] * sc3;
        }
    }
}

extern "C" __global__ void attn_b8_flash_reduce_q8_k(const float* __restrict__ split_buf,
                                                     const float* __restrict__ gate,
                                                     w_q8_1_blk* __restrict__ out_q8,
                                                     int n_head, int n_splits) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = wid * 32 + lane;
    int q_dim = n_head * 128;

    float gm = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* p = split_buf + (((size_t)h * n_splits + sp) * 16 + bt) * 130;
        gm = fmaxf(gm, p[0]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* p = split_buf + (((size_t)h * n_splits + sp) * 16 + bt) * 130;
        if (p[0] > -1e20f) {
            float sc = expf(p[0] - gm);
            gl += p[1] * sc;
            go += p[2 + d] * sc;
        }
    }
    float v = (go / gl) / (1.0f + expf(-gate[(size_t)bt * q_dim + (size_t)h * 128 + d]));
    __shared__ int8_t smq[4][32];
    size_t qbase = ((size_t)bt * n_head + h) * 4;
    w_q8_store(v, lane, out_q8 + qbase + wid, smq[wid]);
}

extern "C" int ink_attn_b8_flash_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    if (n_head <= 32) {
        int n_splits = 4;
        float* sbuf = nullptr;
        (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_b8_split_buf));
        attn_b8_flash_split_k<<<dim3(n_head, n_splits), dim3(128), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, d_pos0, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B, n_splits);
        attn_b8_flash_reduce_q8_k<<<dim3(n_head, B), dim3(128), 0, s>>>(sbuf, gate, (w_q8_1_blk*)out_q8, n_head, n_splits);
        return 0;
    }
    attn_b8_flash_q8_k<<<dim3(n_head), dim3(128), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, (w_q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B);
    return 0;
}

extern "C" int ink_attn_batch_gated_flash(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    attn_batch_gated_flash_k<<<dim3(n_head, (B + 63) / 64), dim3(128), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B);
    return 0;
}

extern "C" int ink_gemm_f32_wmma(hipStream_t s, const float* W, const float* X, float* Y, int n, int k, int B) {
    gemm_f32_wmma_k<<<dim3((n + 15) / 16, (B + 15) / 16), dim3(128), 0, s>>>(W, X, Y, n, k, B);
    return 0;
}

extern "C" int ink_loop_ctrl_down_wmma(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* wd0, const float* wd1, const float* wd2, float* h_buf, int n_embd, int rank, int max_B, int B, int n_ctrl) {
    loop_ctrl_down_wmma_k<<<dim3((rank + 31) / 32, n_ctrl, (B + 31) / 32), dim3(128), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B, B);
    return 0;
}

extern "C" int ink_loop_ctrl_up_wmma(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* h_buf, const float* wg0, const float* wg1, const float* wg2, const float* wf0, const float* wf1, const float* wf2, float* out0, float* out1, float* out2, int n_embd, int rank, int max_B, int B, int n_ctrl) {
    loop_ctrl_up_wmma_k<<<dim3((n_embd + 63) / 64, n_ctrl, (B + 31) / 32), dim3(128), 0, s>>>(cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank, max_B, B);
    return 0;
}
