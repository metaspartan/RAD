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

extern "C" __global__ void gemm_q4_0_wmma_k(const uint8_t* __restrict__ W,
                                            const w_q8_1_blk* __restrict__ Xq,
                                            float* __restrict__ Y,
                                            int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int row0 = (blockIdx.x * 4 + warp) * 16;
    int t0 = blockIdx.y * 32;
    if (row0 >= n || t0 >= B) return;
    bool has1 = (t0 + 16 < B);

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);
    int r_load = min(row0 + col, n - 1);
    int t_load0 = min(t0 + col, B - 1);
    int t_load1 = min(t0 + 16 + col, B - 1);
    const uint8_t* wrow = W + (size_t)r_load * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    const w_q8_1_blk* xrow0 = Xq + (size_t)t_load0 * nblk;
    const w_q8_1_blk* xrow1 = Xq + (size_t)t_load1 * nblk;

    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < nblk; ++b) {
        uint4 q4 = qrow[b];
        float dw_col = __half2float(drow[b]);
        uint32_t w0 = half ? q4.z : q4.x;
        uint32_t w1 = half ? q4.w : q4.y;
        wmma_i2_v a_lo = {(int32_t)(w0 & 0x0F0F0F0Fu), (int32_t)(w1 & 0x0F0F0F0Fu)};
        wmma_i2_v a_hi = {(int32_t)((w0 >> 4) & 0x0F0F0F0Fu), (int32_t)((w1 >> 4) & 0x0F0F0F0Fu)};

        const w_q8_1_blk xb0 = xrow0[b];
        wmma_i2_v b0_lo = {xb0.q[half * 2 + 0], xb0.q[half * 2 + 1]};
        wmma_i2_v b0_hi = {xb0.q[4 + half * 2 + 0], xb0.q[4 + half * 2 + 1]};
        wmma_i8_v cv0 = {0, 0, 0, 0, 0, 0, 0, 0};
        cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b0_lo, cv0, false);
        cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b0_hi, cv0, false);

        wmma_i8_v cv1 = {0, 0, 0, 0, 0, 0, 0, 0};
        float xd1 = 0.0f;
        int32_t off8_1 = 0;
        if (has1) {
            const w_q8_1_blk xb1 = xrow1[b];
            wmma_i2_v b1_lo = {xb1.q[half * 2 + 0], xb1.q[half * 2 + 1]};
            wmma_i2_v b1_hi = {xb1.q[4 + half * 2 + 0], xb1.q[4 + half * 2 + 1]};
            cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b1_lo, cv1, false);
            cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b1_hi, cv1, false);
            xd1 = xb1.d;
            off8_1 = 8 * xb1.s;
        }
        int32_t off8_0 = 8 * xb0.s;
        float xd0 = xb0.d;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float dw_i = __shfl(dw_col, half * 8 + i, 32);
            acc0[i] = fmaf(dw_i * xd0, (float)(cv0[i] - off8_0), acc0[i]);
            if (has1) acc1[i] = fmaf(dw_i * xd1, (float)(cv1[i] - off8_1), acc1[i]);
        }
    }
    if (t0 + col < B) {
        float* yt = Y + (size_t)(t0 + col) * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = row0 + half * 8 + i;
            if (r < n) yt[r] = acc0[i];
        }
    }
    if (has1 && t0 + 16 + col < B) {
        float* yt = Y + (size_t)(t0 + 16 + col) * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = row0 + half * 8 + i;
            if (r < n) yt[r] = acc1[i];
        }
    }
}

extern "C" __global__ void qkvrg_gemm_q4_wmma_k(const uint8_t* __restrict__ wq,
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
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int row0 = (blockIdx.x * 4 + warp) * 16;
    int t0 = blockIdx.y * 32;
    if (t0 >= B) return;
    bool has1 = (t0 + 16 < B);

    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    if (row0 >= o5) return;

    const uint8_t* W;
    const w_q8_1_blk* Xq;
    float* Y;
    int local_row0, n_out;
    if (row0 < o1) { W = wq; Xq = Xq0; Y = Q; local_row0 = row0; n_out = q_dim; }
    else if (row0 < o2) { W = wk; Xq = Xq1; Y = K; local_row0 = row0 - o1; n_out = kv_dim; }
    else if (row0 < o3) { W = wv; Xq = Xq2; Y = V; local_row0 = row0 - o2; n_out = kv_dim; }
    else if (row0 < o4) { W = wr; Xq = Xq0; Y = R; local_row0 = row0 - o3; n_out = r_dim; }
    else { W = wg; Xq = Xq0; Y = G; local_row0 = row0 - o4; n_out = q_dim; }

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);
    int r_load = min(local_row0 + col, n_out - 1);
    int t_load0 = min(t0 + col, B - 1);
    int t_load1 = min(t0 + 16 + col, B - 1);
    const uint8_t* wrow = W + (size_t)r_load * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    const w_q8_1_blk* xrow0 = Xq + (size_t)t_load0 * nblk;
    const w_q8_1_blk* xrow1 = Xq + (size_t)t_load1 * nblk;

    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < nblk; ++b) {
        uint4 q4 = qrow[b];
        float dw_col = __half2float(drow[b]);
        uint32_t w0 = half ? q4.z : q4.x;
        uint32_t w1 = half ? q4.w : q4.y;
        wmma_i2_v a_lo = {(int32_t)(w0 & 0x0F0F0F0Fu), (int32_t)(w1 & 0x0F0F0F0Fu)};
        wmma_i2_v a_hi = {(int32_t)((w0 >> 4) & 0x0F0F0F0Fu), (int32_t)((w1 >> 4) & 0x0F0F0F0Fu)};

        const w_q8_1_blk xb0 = xrow0[b];
        wmma_i2_v b0_lo = {xb0.q[half * 2 + 0], xb0.q[half * 2 + 1]};
        wmma_i2_v b0_hi = {xb0.q[4 + half * 2 + 0], xb0.q[4 + half * 2 + 1]};
        wmma_i8_v cv0 = {0, 0, 0, 0, 0, 0, 0, 0};
        cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b0_lo, cv0, false);
        cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b0_hi, cv0, false);

        wmma_i8_v cv1 = {0, 0, 0, 0, 0, 0, 0, 0};
        float xd1 = 0.0f;
        int32_t off8_1 = 0;
        if (has1) {
            const w_q8_1_blk xb1 = xrow1[b];
            wmma_i2_v b1_lo = {xb1.q[half * 2 + 0], xb1.q[half * 2 + 1]};
            wmma_i2_v b1_hi = {xb1.q[4 + half * 2 + 0], xb1.q[4 + half * 2 + 1]};
            cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b1_lo, cv1, false);
            cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b1_hi, cv1, false);
            xd1 = xb1.d;
            off8_1 = 8 * xb1.s;
        }
        int32_t off8_0 = 8 * xb0.s;
        float xd0 = xb0.d;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float dw_i = __shfl(dw_col, half * 8 + i, 32);
            acc0[i] = fmaf(dw_i * xd0, (float)(cv0[i] - off8_0), acc0[i]);
            if (has1) acc1[i] = fmaf(dw_i * xd1, (float)(cv1[i] - off8_1), acc1[i]);
        }
    }
    if (t0 + col < B) {
        float* yt = Y + (size_t)(t0 + col) * n_out;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = local_row0 + half * 8 + i;
            if (r < n_out) yt[r] = acc0[i];
        }
    }
    if (has1 && t0 + 16 + col < B) {
        float* yt = Y + (size_t)(t0 + 16 + col) * n_out;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = local_row0 + half * 8 + i;
            if (r < n_out) yt[r] = acc1[i];
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

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int row0 = (blockIdx.x * 4 + warp) * 16;
    if (row0 >= n_ff) return;

    int nblk = k >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)(2 * n_ff) * rbytes;
    int r_load = min(row0 + col, n_ff - 1);
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)r_load * rbytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + r_load) * rbytes;
    const uint4* qg_row = reinterpret_cast<const uint4*>(wg);
    const __half* dg_row = reinterpret_cast<const __half*>(wg + (size_t)nblk * 16);
    const uint4* qu_row = reinterpret_cast<const uint4*>(wu);
    const __half* du_row = reinterpret_cast<const __half*>(wu + (size_t)nblk * 16);
    float sc = scales ? scales[exp_id] : 1.0f;

    for (int t0 = 0; t0 < cnt; t0 += 32) {
        int nt0 = min(16, cnt - t0);
        bool has1 = (t0 + 16 < cnt);
        int nt1 = has1 ? min(16, cnt - t0 - 16) : 0;

        int idx0 = off0 + t0 + min(col, nt0 - 1);
        int tid0 = tok_sorted[idx0];
        int sl0 = slot_sorted[idx0];
        const w_q8_1_blk* xrow0 = Xq + (size_t)tid0 * nblk;

        int idx1 = has1 ? (off0 + t0 + 16 + min(col, nt1 - 1)) : idx0;
        int tid1 = tok_sorted[idx1];
        int sl1 = slot_sorted[idx1];
        const w_q8_1_blk* xrow1 = Xq + (size_t)tid1 * nblk;

        float ag0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float ag1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < nblk; ++b) {
            uint4 qg = qg_row[b];
            uint4 qu = qu_row[b];
            float dg_col = __half2float(dg_row[b]);
            float du_col = __half2float(du_row[b]);
            uint32_t g0 = half ? qg.z : qg.x;
            uint32_t g1 = half ? qg.w : qg.y;
            wmma_i2_v ag_lo = {(int32_t)(g0 & 0x0F0F0F0Fu), (int32_t)(g1 & 0x0F0F0F0Fu)};
            wmma_i2_v ag_hi = {(int32_t)((g0 >> 4) & 0x0F0F0F0Fu), (int32_t)((g1 >> 4) & 0x0F0F0F0Fu)};
            uint32_t u0 = half ? qu.z : qu.x;
            uint32_t u1 = half ? qu.w : qu.y;
            wmma_i2_v au_lo = {(int32_t)(u0 & 0x0F0F0F0Fu), (int32_t)(u1 & 0x0F0F0F0Fu)};
            wmma_i2_v au_hi = {(int32_t)((u0 >> 4) & 0x0F0F0F0Fu), (int32_t)((u1 >> 4) & 0x0F0F0F0Fu)};

            const w_q8_1_blk xb0 = xrow0[b];
            wmma_i2_v b0_lo = {xb0.q[half * 2 + 0], xb0.q[half * 2 + 1]};
            wmma_i2_v b0_hi = {xb0.q[4 + half * 2 + 0], xb0.q[4 + half * 2 + 1]};
            wmma_i8_v cg0 = {0, 0, 0, 0, 0, 0, 0, 0};
            cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_lo, true, b0_lo, cg0, false);
            cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_hi, true, b0_hi, cg0, false);
            wmma_i8_v cu0 = {0, 0, 0, 0, 0, 0, 0, 0};
            cu0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_lo, true, b0_lo, cu0, false);
            cu0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_hi, true, b0_hi, cu0, false);

            wmma_i8_v cg1 = {0, 0, 0, 0, 0, 0, 0, 0};
            wmma_i8_v cu1 = {0, 0, 0, 0, 0, 0, 0, 0};
            float xd1 = 0.0f;
            int32_t off8_1 = 0;
            if (has1) {
                const w_q8_1_blk xb1 = xrow1[b];
                wmma_i2_v b1_lo = {xb1.q[half * 2 + 0], xb1.q[half * 2 + 1]};
                wmma_i2_v b1_hi = {xb1.q[4 + half * 2 + 0], xb1.q[4 + half * 2 + 1]};
                cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_lo, true, b1_lo, cg1, false);
                cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, ag_hi, true, b1_hi, cg1, false);
                cu1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_lo, true, b1_lo, cu1, false);
                cu1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, au_hi, true, b1_hi, cu1, false);
                xd1 = xb1.d;
                off8_1 = 8 * xb1.s;
            }
            int32_t off8_0 = 8 * xb0.s;
            float xd0 = xb0.d;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dg_i = __shfl(dg_col, half * 8 + i, 32);
                float du_i = __shfl(du_col, half * 8 + i, 32);
                ag0[i] = fmaf(dg_i * xd0, (float)(cg0[i] - off8_0), ag0[i]);
                au0[i] = fmaf(du_i * xd0, (float)(cu0[i] - off8_0), au0[i]);
                if (has1) {
                    ag1[i] = fmaf(dg_i * xd1, (float)(cg1[i] - off8_1), ag1[i]);
                    au1[i] = fmaf(du_i * xd1, (float)(cu1[i] - off8_1), au1[i]);
                }
            }
        }
        if (col < nt0) {
            float* dst = act_out + ((size_t)tid0 * n_used + sl0) * n_ff;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r = row0 + half * 8 + i;
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
        if (has1 && col < nt1) {
            float* dst = act_out + ((size_t)tid1 * n_used + sl1) * n_ff;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r = row0 + half * 8 + i;
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

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int row0 = (blockIdx.x * 4 + warp) * 16;
    if (row0 >= n_embd) return;

    int nblk = n_ff >> 5;
    size_t rbytes = w_q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * rbytes;
    int r_load = min(row0 + col, n_embd - 1);
    const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)r_load * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);

    for (int t0 = 0; t0 < cnt; t0 += 32) {
        int nt0 = min(16, cnt - t0);
        bool has1 = (t0 + 16 < cnt);
        int nt1 = has1 ? min(16, cnt - t0 - 16) : 0;

        int idx0 = off0 + t0 + min(col, nt0 - 1);
        int tid0 = tok_sorted[idx0];
        int sl0 = slot_sorted[idx0];
        float wt0 = wt_sorted[idx0];
        const w_q8_1_blk* arow0 = act_q + ((size_t)tid0 * n_used + sl0) * nblk;

        int idx1 = has1 ? (off0 + t0 + 16 + min(col, nt1 - 1)) : idx0;
        int tid1 = tok_sorted[idx1];
        int sl1 = slot_sorted[idx1];
        float wt1 = wt_sorted[idx1];
        const w_q8_1_blk* arow1 = act_q + ((size_t)tid1 * n_used + sl1) * nblk;

        float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < nblk; ++b) {
            uint4 q4 = qrow[b];
            float dw_col = __half2float(drow[b]);
            uint32_t w0 = half ? q4.z : q4.x;
            uint32_t w1 = half ? q4.w : q4.y;
            wmma_i2_v a_lo = {(int32_t)(w0 & 0x0F0F0F0Fu), (int32_t)(w1 & 0x0F0F0F0Fu)};
            wmma_i2_v a_hi = {(int32_t)((w0 >> 4) & 0x0F0F0F0Fu), (int32_t)((w1 >> 4) & 0x0F0F0F0Fu)};

            const w_q8_1_blk ab0 = arow0[b];
            wmma_i2_v b0_lo = {ab0.q[half * 2 + 0], ab0.q[half * 2 + 1]};
            wmma_i2_v b0_hi = {ab0.q[4 + half * 2 + 0], ab0.q[4 + half * 2 + 1]};
            wmma_i8_v cv0 = {0, 0, 0, 0, 0, 0, 0, 0};
            cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b0_lo, cv0, false);
            cv0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b0_hi, cv0, false);

            wmma_i8_v cv1 = {0, 0, 0, 0, 0, 0, 0, 0};
            float ad1 = 0.0f;
            int32_t off8_1 = 0;
            if (has1) {
                const w_q8_1_blk ab1 = arow1[b];
                wmma_i2_v b1_lo = {ab1.q[half * 2 + 0], ab1.q[half * 2 + 1]};
                wmma_i2_v b1_hi = {ab1.q[4 + half * 2 + 0], ab1.q[4 + half * 2 + 1]};
                cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_lo, true, b1_lo, cv1, false);
                cv1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a_hi, true, b1_hi, cv1, false);
                ad1 = ab1.d;
                off8_1 = 8 * ab1.s;
            }
            int32_t off8_0 = 8 * ab0.s;
            float ad0 = ab0.d;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dw_i = __shfl(dw_col, half * 8 + i, 32);
                acc0[i] = fmaf(dw_i * ad0, (float)(cv0[i] - off8_0), acc0[i]);
                if (has1) acc1[i] = fmaf(dw_i * ad1, (float)(cv1[i] - off8_1), acc1[i]);
            }
        }
        if (col < nt0) {
            float* dst = y_out + ((size_t)tid0 * n_used + sl0) * n_embd;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r = row0 + half * 8 + i;
                if (r < n_embd) dst[r] = acc0[i] * wt0;
            }
        }
        if (has1 && col < nt1) {
            float* dst = y_out + ((size_t)tid1 * n_used + sl1) * n_embd;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r = row0 + half * 8 + i;
                if (r < n_embd) dst[r] = acc1[i] * wt1;
            }
        }
    }
}

extern "C" __global__ void reduce_moe_slots_k(const float* __restrict__ slots,
                                              float* __restrict__ y_out,
                                              int n_embd, int n_used) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (r >= n_embd) return;
    const float* base = slots + (size_t)t * n_used * n_embd + r;
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < n_used) sum += base[(size_t)s * n_embd];
    }
    y_out[(size_t)t * n_embd + r] = sum;
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
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int row0 = (blockIdx.x * 4 + warp) * 16;
    int c = blockIdx.y;
    int t0 = blockIdx.z * 16;
    if (t0 >= B) return;

    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* wd = (c == 0) ? wd0 : ((c == 1) ? wd1 : wd2);

    __shared__ float rc_s[16];
    __shared__ float rm_s[16];
    int nt = min(16, B - t0);
    for (int m = warp; m < nt; m += 4) {
        const float* cp = cur_base + (size_t)(t0 + m) * n_embd;
        const float* mp = msg_base + (size_t)(t0 + m) * n_embd;
        float sc = 0.0f, sm = 0.0f;
        for (int j = lane; j < n_embd; j += 32) {
            float vc = cp[j], vm = mp[j];
            sc = fmaf(vc, vc, sc);
            sm = fmaf(vm, vm, sm);
        }
        sc = w_warp_sum32(sc);
        sm = w_warp_sum32(sm);
        if (lane == 0) {
            rc_s[m] = rsqrtf(sc / (float)n_embd + 1e-6f);
            rm_s[m] = rsqrtf(sm / (float)n_embd + 1e-6f);
        }
    }
    __syncthreads();
    if (row0 >= rank) return;

    int r_load = min(row0 + col, rank - 1);
    int m_load = min(col, nt - 1);
    const float* wrow = wd + (size_t)r_load * (2 * n_embd);
    const float* crow = cur_base + (size_t)(t0 + m_load) * n_embd;
    const float* mrow = msg_base + (size_t)(t0 + m_load) * n_embd;
    float rc = rc_s[m_load];
    float rm = rm_s[m_load];

    wmma_f8_v cv = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int kk = 0; kk < n_embd; kk += 16) {
        int kb = kk + half * 8;
        wmma_h8_v aw0, ax0, aw1, ax1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            aw0[j] = (_Float16)wrow[kb + j];
            ax0[j] = (_Float16)(crow[kb + j] * rc);
            aw1[j] = (_Float16)wrow[n_embd + kb + j];
            ax1[j] = (_Float16)(mrow[kb + j] * rm);
        }
        cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aw0, ax0, cv);
        cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aw1, ax1, cv);
    }
    if (col < nt) {
        float* ht = h_out + ((size_t)c * max_B + (t0 + col)) * rank;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = row0 + half * 8 + i;
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
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int row0 = (blockIdx.x * 4 + warp) * 16;
    int c = blockIdx.y;
    int t0 = blockIdx.z * 32;
    if (row0 >= n_embd || t0 >= B) return;
    bool has1 = (t0 + 16 < B);

    const float* cur_base = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg_base = msgs + (size_t)c * max_B * n_embd;
    const float* h_base = h_in + (size_t)c * max_B * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    float* out_base = (c == 0) ? out0 : ((c == 1) ? out1 : out2);

    int r_load = min(row0 + col, n_embd - 1);
    int t_load0 = min(t0 + col, B - 1);
    int t_load1 = min(t0 + 16 + col, B - 1);
    const float* wgr = wg + (size_t)r_load * rank;
    const float* wfr = wf + (size_t)r_load * rank;
    const float* hr0 = h_base + (size_t)t_load0 * rank;
    const float* hr1 = h_base + (size_t)t_load1 * rank;

    wmma_f8_v cg0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    wmma_f8_v cf0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    wmma_f8_v cg1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    wmma_f8_v cf1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int kk = 0; kk < rank; kk += 16) {
        int kb = kk + half * 8;
        wmma_h8_v ag, af, bh0, bh1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            ag[j] = (_Float16)wgr[kb + j];
            af[j] = (_Float16)wfr[kb + j];
            bh0[j] = (_Float16)hr0[kb + j];
            bh1[j] = has1 ? (_Float16)hr1[kb + j] : (_Float16)0.0f;
        }
        cg0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ag, bh0, cg0);
        cf0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bh0, cf0);
        if (has1) {
            cg1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ag, bh1, cg1);
            cf1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bh1, cf1);
        }
    }
    if (t0 + col < B) {
        size_t base = (size_t)(t0 + col) * n_embd;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = row0 + half * 8 + i;
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
            int r = row0 + half * 8 + i;
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
    int bt_blk = blockIdx.y * 64;
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

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    __shared__ _Float16 sm_k[16][128];
    __shared__ _Float16 sm_vt[128][16];

    for (int tile = t_min; tile < t_max; tile += 16) {
        int n_kv = min(16, t_max - tile);
        for (int idx = threadIdx.x; idx < 16 * 128; idx += 128) {
            int tk = idx >> 7;
            int d = idx & 127;
            if (tk < n_kv) {
                int kv_slot = is_swa ? ((tile + tk) % max_ctx) : (tile + tk);
                sm_k[tk][d] = (_Float16)__half2float(kc_h[(size_t)kv_slot * dh + d]);
                sm_vt[d][tk] = (_Float16)__half2float(vc_h[(size_t)kv_slot * dh + d]);
            } else {
                sm_k[tk][d] = (_Float16)0.0f;
                sm_vt[d][tk] = (_Float16)0.0f;
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
                    const float* rbp = rel_bias + (size_t)dist * d_rel;
                    float rp = 0.0f;
                    const float* rh = R + (size_t)bt_q * r_dim + (size_t)h * d_rel;
#pragma unroll
                    for (int rr = 0; rr < 16; ++rr) rp = fmaf(rbp[rr], rh[rr], rp);
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
    gemm_q4_0_wmma_k<<<dim3((n + 63) / 64, (B + 31) / 32), dim3(128), 0, s>>>(W, (const w_q8_1_blk*)Xq, Y, n, k, B);
    return 0;
}

extern "C" int ink_qkvrg_gemm_q4_wmma(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim, int B) {
    int qkv_rows = q_dim + kv_dim + kv_dim + r_dim + q_dim;
    qkvrg_gemm_q4_wmma_k<<<dim3((qkv_rows + 63) / 64, (B + 31) / 32), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, (const w_q8_1_blk*)Xq0, (const w_q8_1_blk*)Xq1, (const w_q8_1_blk*)Xq2, Q, K, V, R, G, k, q_dim, kv_dim, r_dim, B);
    int sd_rows = n_head + 2 * dyn_dim;
    sink_dyn_bf16_k<<<dim3((sd_rows + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(wsink, wkdyn, wvdyn, X0, X1, X2, S, Kdyn, Vdyn, k, n_head, dyn_dim, B);
    return 0;
}

extern "C" int ink_moe_gu_q4_0_wmma(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, float* act_out, int n_ff, int k, int n_used, float beta, float lb, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    moe_gu_q4_0_wmma_k<<<dim3((n_ff + 63) / 64, gy), dim3(128), 0, s>>>(W, scales, (const w_q8_1_blk*)Xq, exp_counts, exp_offsets, tok_sorted, slot_sorted, act_out, n_ff, k, n_used, beta, lb);
    return 0;
}

extern "C" int ink_moe_dn_q4_0_wmma(hipStream_t s, const uint8_t* W, const void* act_q, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, const float* wt_sorted, float* moe_slots, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    moe_dn_q4_0_wmma_k<<<dim3((n_embd + 63) / 64, gy), dim3(128), 0, s>>>(W, (const w_q8_1_blk*)act_q, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, moe_slots, n_embd, n_ff, n_used);
    reduce_moe_slots_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(moe_slots, y_out, n_embd, n_used);
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
    loop_ctrl_down_wmma_k<<<dim3((rank + 63) / 64, n_ctrl, (B + 15) / 16), dim3(128), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank, max_B, B);
    return 0;
}

extern "C" int ink_loop_ctrl_up_wmma(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* h_buf, const float* wg0, const float* wg1, const float* wg2, const float* wf0, const float* wf1, const float* wf2, float* out0, float* out1, float* out2, int n_embd, int rank, int max_B, int B, int n_ctrl) {
    loop_ctrl_up_wmma_k<<<dim3((n_embd + 63) / 64, n_ctrl, (B + 31) / 32), dim3(128), 0, s>>>(cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank, max_B, B);
    return 0;
}
