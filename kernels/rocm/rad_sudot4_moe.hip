#include "rad_sudot4_common.cuh"

extern "C" int ink_qkvrg_q4_b1(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim);


extern "C" __global__ void zero_buf_f32_k(float* __restrict__ p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
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
    int nblk = n_ff >> 5;
    int tot_aq = n_used * nblk;

    __shared__ q8_1_blk sm_aq[288];
    if (tot_aq <= 288) {
        for (int i = threadIdx.x; i < tot_aq; i += 128) sm_aq[i] = act_q[i];
        __syncthreads();
    }
    if (row >= n_embd) return;
    const q8_1_blk* aq_base = (tot_aq <= 288) ? sm_aq : act_q;

    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * row_bytes;

    float total = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        int exp_id = itop[s];
        float wt = wtop[s] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
        const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
        const q8_1_blk* aq = aq_base + (size_t)s * nblk;
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
    int z_id = blockIdx.z;
    int z_n = gridDim.z;
    if (cnt == 0 || z_id * 8 >= cnt) return;
    int off0 = exp_offsets[exp_id];
    int tid = threadIdx.x;
    int grp = tid >> 2;
    int lane4 = tid & 3;
    int row = blockIdx.x * 32 + grp;
    bool valid = (row < n_embd);

    int nblk = n_ff >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    const uint8_t* wrow = valid ? (W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes) : W;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    __shared__ q8_1_blk sm_act[4][32];
    bool use_lds = (nblk <= 32);

    if (cnt == 1) {
        int tid0 = tok_sorted[off0], sid0 = slot_sorted[off0];
        const q8_1_blk* ap0 = act_q + ((size_t)tid0 * n_used + sid0) * nblk;
        if (use_lds) {
            if (tid < nblk) sm_act[0][tid] = ap0[tid];
            __syncthreads();
        }
        if (!valid) return;
        float acc0 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            const q8_1_blk ab = use_lds ? sm_act[0][b] : ap0[b];
            acc0 += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), ab);
        }
        float v = s_warp_sum4(acc0) * wt_sorted[off0];
        if (lane4 == 0) atomicAdd(&y_out[(size_t)tid0 * n_embd + row], v);
        return;
    }
    if (cnt == 2) {
        int tid0 = tok_sorted[off0], sid0 = slot_sorted[off0];
        int tid1 = tok_sorted[off0 + 1], sid1 = slot_sorted[off0 + 1];
        const q8_1_blk* ap0 = act_q + ((size_t)tid0 * n_used + sid0) * nblk;
        const q8_1_blk* ap1 = act_q + ((size_t)tid1 * n_used + sid1) * nblk;
        if (use_lds) {
            if (tid < nblk) sm_act[0][tid] = ap0[tid];
            else if (tid < 2 * nblk) sm_act[1][tid - nblk] = ap1[tid - nblk];
            __syncthreads();
        }
        if (!valid) return;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu), w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
            int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu), w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
            int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
            int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
            const q8_1_blk ab0 = use_lds ? sm_act[0][b] : ap0[b];
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
            const q8_1_blk ab1 = use_lds ? sm_act[1][b] : ap1[b];
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
        const q8_1_blk* ap0 = act_q + ((size_t)t0i * n_used + s0i) * nblk;
        const q8_1_blk* ap1 = act_q + ((size_t)t1i * n_used + s1i) * nblk;
        const q8_1_blk* ap2 = act_q + ((size_t)t2i * n_used + s2i) * nblk;
        if (use_lds) {
            if (tid < nblk) sm_act[0][tid] = ap0[tid];
            else if (tid < 2 * nblk) sm_act[1][tid - nblk] = ap1[tid - nblk];
            else if (tid < 3 * nblk) sm_act[2][tid - 2 * nblk] = ap2[tid - 2 * nblk];
            __syncthreads();
        }
        if (!valid) return;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu), w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
            int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu), w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
            int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
            int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
            const q8_1_blk ab0 = use_lds ? sm_act[0][b] : ap0[b];
            int32_t p0 = 0;
            p0 = __builtin_amdgcn_sudot4(false, w0, true, ab0.q[0], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w1, true, ab0.q[1], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w2, true, ab0.q[2], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w3, true, ab0.q[3], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w4, true, ab0.q[4], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w5, true, ab0.q[5], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w6, true, ab0.q[6], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w7, true, ab0.q[7], p0, false);
            a0 = fmaf(dw * ab0.d, (float)(p0 - 8 * ab0.s), a0);
            const q8_1_blk ab1 = use_lds ? sm_act[1][b] : ap1[b];
            int32_t p1 = 0;
            p1 = __builtin_amdgcn_sudot4(false, w0, true, ab1.q[0], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w1, true, ab1.q[1], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w2, true, ab1.q[2], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w3, true, ab1.q[3], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w4, true, ab1.q[4], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w5, true, ab1.q[5], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w6, true, ab1.q[6], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w7, true, ab1.q[7], p1, false);
            a1 = fmaf(dw * ab1.d, (float)(p1 - 8 * ab1.s), a1);
            const q8_1_blk ab2 = use_lds ? sm_act[2][b] : ap2[b];
            int32_t p2 = 0;
            p2 = __builtin_amdgcn_sudot4(false, w0, true, ab2.q[0], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w1, true, ab2.q[1], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w2, true, ab2.q[2], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w3, true, ab2.q[3], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w4, true, ab2.q[4], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w5, true, ab2.q[5], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w6, true, ab2.q[6], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w7, true, ab2.q[7], p2, false);
            a2 = fmaf(dw * ab2.d, (float)(p2 - 8 * ab2.s), a2);
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
        const q8_1_blk* ap0 = act_q + ((size_t)t0i * n_used + s0i) * nblk;
        const q8_1_blk* ap1 = act_q + ((size_t)t1i * n_used + s1i) * nblk;
        const q8_1_blk* ap2 = act_q + ((size_t)t2i * n_used + s2i) * nblk;
        const q8_1_blk* ap3 = act_q + ((size_t)t3i * n_used + s3i) * nblk;
        if (use_lds) {
            if (tid < nblk) sm_act[0][tid] = ap0[tid];
            else if (tid < 2 * nblk) sm_act[1][tid - nblk] = ap1[tid - nblk];
            else if (tid < 3 * nblk) sm_act[2][tid - 2 * nblk] = ap2[tid - 2 * nblk];
            else if (tid < 4 * nblk) sm_act[3][tid - 3 * nblk] = ap3[tid - 3 * nblk];
            __syncthreads();
        }
        if (!valid) return;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu), w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
            int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu), w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
            int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
            int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
            const q8_1_blk ab0 = use_lds ? sm_act[0][b] : ap0[b];
            int32_t p0 = 0;
            p0 = __builtin_amdgcn_sudot4(false, w0, true, ab0.q[0], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w1, true, ab0.q[1], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w2, true, ab0.q[2], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w3, true, ab0.q[3], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w4, true, ab0.q[4], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w5, true, ab0.q[5], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w6, true, ab0.q[6], p0, false);
            p0 = __builtin_amdgcn_sudot4(false, w7, true, ab0.q[7], p0, false);
            a0 = fmaf(dw * ab0.d, (float)(p0 - 8 * ab0.s), a0);
            const q8_1_blk ab1 = use_lds ? sm_act[1][b] : ap1[b];
            int32_t p1 = 0;
            p1 = __builtin_amdgcn_sudot4(false, w0, true, ab1.q[0], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w1, true, ab1.q[1], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w2, true, ab1.q[2], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w3, true, ab1.q[3], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w4, true, ab1.q[4], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w5, true, ab1.q[5], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w6, true, ab1.q[6], p1, false);
            p1 = __builtin_amdgcn_sudot4(false, w7, true, ab1.q[7], p1, false);
            a1 = fmaf(dw * ab1.d, (float)(p1 - 8 * ab1.s), a1);
            const q8_1_blk ab2 = use_lds ? sm_act[2][b] : ap2[b];
            int32_t p2 = 0;
            p2 = __builtin_amdgcn_sudot4(false, w0, true, ab2.q[0], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w1, true, ab2.q[1], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w2, true, ab2.q[2], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w3, true, ab2.q[3], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w4, true, ab2.q[4], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w5, true, ab2.q[5], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w6, true, ab2.q[6], p2, false);
            p2 = __builtin_amdgcn_sudot4(false, w7, true, ab2.q[7], p2, false);
            a2 = fmaf(dw * ab2.d, (float)(p2 - 8 * ab2.s), a2);
            const q8_1_blk ab3 = use_lds ? sm_act[3][b] : ap3[b];
            int32_t p3 = 0;
            p3 = __builtin_amdgcn_sudot4(false, w0, true, ab3.q[0], p3, false);
            p3 = __builtin_amdgcn_sudot4(false, w1, true, ab3.q[1], p3, false);
            p3 = __builtin_amdgcn_sudot4(false, w2, true, ab3.q[2], p3, false);
            p3 = __builtin_amdgcn_sudot4(false, w3, true, ab3.q[3], p3, false);
            p3 = __builtin_amdgcn_sudot4(false, w4, true, ab3.q[4], p3, false);
            p3 = __builtin_amdgcn_sudot4(false, w5, true, ab3.q[5], p3, false);
            p3 = __builtin_amdgcn_sudot4(false, w6, true, ab3.q[6], p3, false);
            p3 = __builtin_amdgcn_sudot4(false, w7, true, ab3.q[7], p3, false);
            a3 = fmaf(dw * ab3.d, (float)(p3 - 8 * ab3.s), a3);
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
    if (!valid) return;

    for (int t0 = z_id * 8; t0 < cnt; t0 += z_n * 8) {
        int nt = min(8, cnt - t0);
        int tids[8];
        int sids[8];
        for (int m = 0; m < nt; ++m) {
            tids[m] = tok_sorted[off0 + t0 + m];
            sids[m] = slot_sorted[off0 + t0 + m];
        }
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (nt == 8) {
            const q8_1_blk* ap0 = act_q + ((size_t)tids[0] * n_used + sids[0]) * nblk;
            const q8_1_blk* ap1 = act_q + ((size_t)tids[1] * n_used + sids[1]) * nblk;
            const q8_1_blk* ap2 = act_q + ((size_t)tids[2] * n_used + sids[2]) * nblk;
            const q8_1_blk* ap3 = act_q + ((size_t)tids[3] * n_used + sids[3]) * nblk;
            const q8_1_blk* ap4 = act_q + ((size_t)tids[4] * n_used + sids[4]) * nblk;
            const q8_1_blk* ap5 = act_q + ((size_t)tids[5] * n_used + sids[5]) * nblk;
            const q8_1_blk* ap6 = act_q + ((size_t)tids[6] * n_used + sids[6]) * nblk;
            const q8_1_blk* ap7 = act_q + ((size_t)tids[7] * n_used + sids[7]) * nblk;
            const q8_1_blk* aps[8] = { ap0, ap1, ap2, ap3, ap4, ap5, ap6, ap7 };
            for (int b = lane4; b < nblk; b += 4) {
                uint4 q4 = qrow[b];
                float dw = __half2float(drow[b]);
                int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu), w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
                int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu), w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
                int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
                int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const q8_1_blk ab = aps[m][b];
                    int32_t sumi = 0;
                    sumi = __builtin_amdgcn_sudot4(false, w0, true, ab.q[0], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w1, true, ab.q[1], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w2, true, ab.q[2], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w3, true, ab.q[3], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w4, true, ab.q[4], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w5, true, ab.q[5], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w6, true, ab.q[6], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w7, true, ab.q[7], sumi, false);
                    acc[m] = fmaf(dw * ab.d, (float)(sumi - 8 * ab.s), acc[m]);
                }
            }
        } else {
            for (int b = lane4; b < nblk; b += 4) {
                uint4 q4 = qrow[b];
                float dw = __half2float(drow[b]);
                int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu), w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
                int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu), w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
                int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
                int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
                for (int m = 0; m < nt; ++m) {
                    const q8_1_blk ab = act_q[((size_t)tids[m] * n_used + sids[m]) * nblk + b];
                    int32_t sumi = 0;
                    sumi = __builtin_amdgcn_sudot4(false, w0, true, ab.q[0], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w1, true, ab.q[1], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w2, true, ab.q[2], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w3, true, ab.q[3], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w4, true, ab.q[4], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w5, true, ab.q[5], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w6, true, ab.q[6], sumi, false);
                    sumi = __builtin_amdgcn_sudot4(false, w7, true, ab.q[7], sumi, false);
                    acc[m] = fmaf(dw * ab.d, (float)(sumi - 8 * ab.s), acc[m]);
                }
            }
        }
        for (int m = 0; m < nt; ++m) {
            float v = s_warp_sum4(acc[m]) * wt_sorted[off0 + t0 + m];
            if (lane4 == 0) atomicAdd(&y_out[(size_t)tids[m] * n_embd + row], v);
        }
    }
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
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int blk_gid0 = blockIdx.x * 32;
    int gid = blk_gid0 + grp;
    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    int o6 = o5 + n_head;
    int o7 = o6 + dyn_dim;
    int o8 = o7 + dyn_dim;

    const q8_1_blk* blk_xq = nullptr;
    if (blk_gid0 + 31 < o5) {
        if (blk_gid0 + 31 < o1 || (blk_gid0 >= o3 && blk_gid0 + 31 < o5)) blk_xq = Xq0;
        else if (blk_gid0 >= o1 && blk_gid0 + 31 < o2) blk_xq = Xq1;
        else if (blk_gid0 >= o2 && blk_gid0 + 31 < o3) blk_xq = Xq2;
    }
    __shared__ q8_1_blk sm_x[NT][8];

    if (blk_xq) {
        const uint8_t* W;
        float* Y;
        int row, n_out;
        if (gid < o1) { W = wq; Y = Q; row = gid; n_out = q_dim; }
        else if (gid < o2) { W = wk; Y = K; row = gid - o1; n_out = kv_dim; }
        else if (gid < o3) { W = wv; Y = V; row = gid - o2; n_out = kv_dim; }
        else if (gid < o4) { W = wr; Y = R; row = gid - o3; n_out = r_dim; }
        else { W = wg; Y = G; row = gid - o4; n_out = q_dim; }

        int nblk = k >> 5;
        size_t rbytes = q4_soa_row_bytes(nblk);
        const uint8_t* wrow = W + (size_t)row * rbytes;
        const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
        const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
        float acc[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
        for (int b0 = 0; b0 < nblk; b0 += 8) {
            if (threadIdx.x < NT * 8) {
                int tm = threadIdx.x >> 3;
                int tb = threadIdx.x & 7;
                if (b0 + tb < nblk) sm_x[tm][tb] = blk_xq[(size_t)tm * nblk + b0 + tb];
            }
            __syncthreads();
            int b = b0 + lane8;
            if (b < nblk) {
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
                    const q8_1_blk xb = sm_x[m][lane8];
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
            __syncthreads();
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            float v = s_warp_sum8(acc[m]);
            if (lane8 == 0) Y[(size_t)m * n_out + row] = v;
        }
        return;
    }

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
            float v = s_warp_sum8(acc[m]);
            if (lane8 == 0) Y[(size_t)m * n_out + row] = v;
        }
    } else {
        const uint16_t* W;
        const float* X;
        float* Y;
        int row, n_out;
        if (gid < o6) { W = wsink; X = X0; Y = S; row = gid - o5; n_out = n_head; }
        else if (gid < o7) { W = wkdyn; X = X1; Y = Kdyn; row = gid - o6; n_out = dyn_dim; }
        else { W = wvdyn; X = X2; Y = Vdyn; row = gid - o7; n_out = dyn_dim; }
        const uint4* w4p = reinterpret_cast<const uint4*>(W + (size_t)row * k);
        int k8 = k >> 3;
        float acc[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
        for (int j = lane8; j < k8; j += 8) {
            uint4 wv = w4p[j];
            float f0 = s_bf16_f32((uint16_t)(wv.x & 0xFFFFu));
            float f1 = s_bf16_f32((uint16_t)(wv.x >> 16));
            float f2 = s_bf16_f32((uint16_t)(wv.y & 0xFFFFu));
            float f3 = s_bf16_f32((uint16_t)(wv.y >> 16));
            float f4 = s_bf16_f32((uint16_t)(wv.z & 0xFFFFu));
            float f5 = s_bf16_f32((uint16_t)(wv.z >> 16));
            float f6 = s_bf16_f32((uint16_t)(wv.w & 0xFFFFu));
            float f7 = s_bf16_f32((uint16_t)(wv.w >> 16));
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                const float4* x4 = reinterpret_cast<const float4*>(X + (size_t)m * k + (size_t)j * 8);
                float4 xa = x4[0], xb = x4[1];
                float s = fmaf(f0, xa.x, fmaf(f1, xa.y, fmaf(f2, xa.z, f3 * xa.w)));
                s = fmaf(f4, xb.x, fmaf(f5, xb.y, fmaf(f6, xb.z, fmaf(f7, xb.w, s))));
                acc[m] += s;
            }
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            float v = s_warp_sum8(acc[m]);
            if (lane8 == 0) Y[(size_t)m * n_out + row] = v;
        }
    }
}


template <int NT>
__global__ void wo_conv_res_q4_nt_snap_k(const uint8_t* __restrict__ W,
                                         const q8_1_blk* __restrict__ Xq,
                                         const float* __restrict__ conv_w,
                                         float* __restrict__ hist,
                                         float* __restrict__ snap,
                                         const float* __restrict__ add_src,
                                         float* __restrict__ block_dst,
                                         float* __restrict__ out,
                                         int n_embd, int q_dim) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;
    int nblk = q_dim >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    bool valid = (row < n_embd);
    const uint8_t* wrow = valid ? (W + (size_t)row * rbytes) : W;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    __shared__ q8_1_blk sm_x[NT][8];
    float acc[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) acc[m] = 0.0f;
    for (int b0 = 0; b0 < nblk; b0 += 8) {
        if (threadIdx.x < NT * 8) {
            int tm = threadIdx.x >> 3;
            int tb = threadIdx.x & 7;
            if (b0 + tb < nblk) sm_x[tm][tb] = Xq[(size_t)tm * nblk + b0 + tb];
        }
        __syncthreads();
        int b = b0 + lane8;
        if (valid && b < nblk) {
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
                const q8_1_blk xb = sm_x[m][lane8];
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
        __syncthreads();
    }
    if (!valid) return;
    float vals[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        vals[m] = s_warp_sum8(acc[m]);
    }
    if (lane8 == 0) {
        float* hc = hist + (size_t)row * 3;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float* wc = conv_w + (size_t)row * 4;
        float cw0 = wc[0], cw1 = wc[1], cw2 = wc[2], cw3 = wc[3];
        float inv5 = rsqrtf(5.0f);
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            size_t idx = (size_t)m * n_embd + row;
            float x = vals[m];
            float conv_out = (x + cw0 * h0 + cw1 * h1 + cw2 * h2 + cw3 * x) * inv5;
            h0 = h1; h1 = h2; h2 = x;
            if (snap) {
                float* sn = snap + ((size_t)m * n_embd + row) * 3;
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
}


extern "C" int ink_wo_conv_res_q4_batch_snap(hipStream_t s, const uint8_t* W, const void* Xq, const float* conv_w, float* hist, float* snap, const float* add_src, float* block_dst, float* out, int n_embd, int q_dim, int B) {
    if (B == 8) {
        wo_conv_res_q4_nt_snap_k<8><<<dim3((n_embd + 31) / 32), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, conv_w, hist, snap, add_src, block_dst, out, n_embd, q_dim);
        return 0;
    }
    if (B == 6) {
        wo_conv_res_q4_nt_snap_k<6><<<dim3((n_embd + 31) / 32), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, conv_w, hist, snap, add_src, block_dst, out, n_embd, q_dim);
        return 0;
    }
    if (B == 4) {
        wo_conv_res_q4_nt_snap_k<4><<<dim3((n_embd + 31) / 32), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, conv_w, hist, snap, add_src, block_dst, out, n_embd, q_dim);
        return 0;
    }
    return 1;
}


extern "C" int ink_qkvrg_gemm_sudot4(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim, int B, int is_q4) {
    if (B == 1 && is_q4) {
        return ink_qkvrg_q4_b1(s, wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, Xq0, Xq1, Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
    }
    int total_rows = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    if (is_q4 && B == 8) {
        qkvrg_q4_nt_k<8><<<dim3((total_rows + 31) / 32), dim3(256), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
        return 0;
    }
    if (is_q4 && B == 6) {
        qkvrg_q4_nt_k<6><<<dim3((total_rows + 31) / 32), dim3(256), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
        return 0;
    }
    if (is_q4 && B == 4) {
        qkvrg_q4_nt_k<4><<<dim3((total_rows + 31) / 32), dim3(256), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
        return 0;
    }
    qkvrg_gemm_sudot4_k<<<dim3((total_rows + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim, B, is_q4);
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
    int gz = (B >= 40) ? 4 : ((B >= 20) ? 2 : 1);
    zero_buf_f32_k<<<dim3((tot + 255) / 256), dim3(256), 0, s>>>(y_out, tot);
    moe_dn_q4_0_sudot4_k<<<dim3((n_embd + 31) / 32, gy, gz), dim3(128), 0, s>>>(W, (const q8_1_blk*)act_q, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, y_out, n_embd, n_ff, n_used);
    return 0;
}


extern "C" int ink_moe_dn_q4_0_sudot4_nozero(hipStream_t s, const uint8_t* W, const void* act_q, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, const float* wt_sorted, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    int gz = (B >= 40) ? 4 : ((B >= 20) ? 2 : 1);
    moe_dn_q4_0_sudot4_k<<<dim3((n_embd + 31) / 32, gy, gz), dim3(128), 0, s>>>(W, (const q8_1_blk*)act_q, exp_counts, exp_offsets, tok_sorted, slot_sorted, wt_sorted, y_out, n_embd, n_ff, n_used);
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
    int bid = blockIdx.x;
    int lane = threadIdx.x;
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (bid < n_head * B) {
        int h = bid / B;
        int t = bid % B;
        const float* qch = q_clip + (size_t)h * dh;
        const float* rch = rel_clip + (size_t)h * d_rel;
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
    } else if (bid < n_head * B + n_head_kv) {
        int kh_idx = bid - n_head * B;
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
    kv_conv_qkvr_store_batch_snap_k<<<dim3(n_head * B + n_head_kv), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, k_snap, v_snap, q_clip, rel_clip, kc, vc, d_pos0, B, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}


extern "C" __global__ void shortconv_res_batch_snap_clear_k(float* __restrict__ raw,
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
        raw[idx] = 0.0f;
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


extern "C" int ink_shortconv_res_batch_snap_clear(hipStream_t s, float* raw, const float* conv_w, float* hist, float* snap, const float* add_src, float* block_dst, float* out, int n_embd, int B) {
    shortconv_res_batch_snap_clear_k<<<dim3((n_embd + 31) / 32), dim3(32), 0, s>>>(raw, conv_w, hist, snap, add_src, block_dst, out, n_embd, B);
    return 0;
}


extern "C" int ink_shortconv_res_batch_snap(hipStream_t s, const float* raw, const float* conv_w, float* hist, float* snap, const float* add_src, float* block_dst, float* out, int n_embd, int B) {
    if (B > 8 && snap == nullptr) {
        shortconv_res_prefill_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(raw, conv_w, hist, add_src, block_dst, out, n_embd);
        hist_update_3_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(raw, hist, n_embd, B);
        return 0;
    }
    shortconv_res_batch_snap_k<<<dim3((n_embd + 31) / 32), dim3(32), 0, s>>>(raw, conv_w, hist, snap, add_src, block_dst, out, n_embd, B);
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
