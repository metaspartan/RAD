#include "rad_sudot4_common.cuh"

extern "C" __global__ void inkling_attn_decode_gated_q8_k(const float* __restrict__ Q,
                                                          const float* __restrict__ R,
                                                          const float* __restrict__ sinks,
                                                          const float* __restrict__ rel_bias,
                                                          const __half* __restrict__ k_cache,
                                                          const __half* __restrict__ v_cache,
                                                          const float* __restrict__ gate,
                                                          const int* __restrict__ d_pos,
                                                          q8_1_blk* __restrict__ out_q8,
                                                          int n_head, int n_head_kv,
                                                          int dh, int d_rel,
                                                          int rel_extent, int swa_window,
                                                          int is_swa, int max_ctx) {
    int h = blockIdx.x;
    if (h >= n_head) return;
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;

    __shared__ float qs[128];
    __shared__ float rs[16];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[(size_t)h * dh + d] * q_scale;
    }
    if (threadIdx.x < d_rel) {
        rs[threadIdx.x] = R[(size_t)h * d_rel + threadIdx.x];
    }
    __syncthreads();

    int pos = *d_pos;
    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[h];

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q0a = qs[p0], q0b = qs[p0 + 1], q1a = qs[p1], q1b = qs[p1 + 1];

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        float dot = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
        dot = s_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = s_warp_sum32(rp);
            rbias = __shfl(rp, 0, 32);
        }
        score += rbias - sink_h;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        o0 = fmaf(p, v0.x, o0 * alpha);
        o1 = fmaf(p, v0.y, o1 * alpha);
        o2 = fmaf(p, v1.x, o2 * alpha);
        o3 = fmaf(p, v1.y, o3 * alpha);
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    sm_o[wid][p0] = o0; sm_o[wid][p0 + 1] = o1; sm_o[wid][p1] = o2; sm_o[wid][p1 + 1] = o3;
    __syncthreads();

    if (wid == 0) {
        float gm = sm_m[0];
        for (int w = 1; w < nw; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
        int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
        for (int w = 0; w < nw; ++w) {
            float sc = expf(sm_m[w] - gm);
            gl += sm_l[w] * sc;
            go0 += sm_o[w][d0] * sc;
            go1 += sm_o[w][d1] * sc;
            go2 += sm_o[w][d2] * sc;
            go3 += sm_o[w][d3] * sc;
        }
        float inv_l = 1.0f / gl;
        const float* gh = gate + (size_t)h * dh;
        float v0 = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        float v1 = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        float v2 = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        float v3 = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
        __shared__ int8_t smq[32];
        store_q8_1_from_warp(v0, lane, out_q8 + h * 4 + 0, smq);
        store_q8_1_from_warp(v1, lane, out_q8 + h * 4 + 1, smq);
        store_q8_1_from_warp(v2, lane, out_q8 + h * 4 + 2, smq);
        store_q8_1_from_warp(v3, lane, out_q8 + h * 4 + 3, smq);
    }
}

extern "C" __global__ void attn_decode_gqa_splitkv_k(const float* __restrict__ Q,
                                                     const float* __restrict__ R,
                                                     const float* __restrict__ sinks,
                                                     const float* __restrict__ rel_bias,
                                                     const __half* __restrict__ k_cache,
                                                     const __half* __restrict__ v_cache,
                                                     const int* __restrict__ d_pos,
                                                     float* __restrict__ split_buf,
                                                     int n_head, int n_head_kv,
                                                     int dh, int d_rel,
                                                     int rel_extent, int swa_window,
                                                     int is_swa, int max_ctx, int n_splits) {
    int kv_h = blockIdx.x;
    int sp = blockIdx.y;
    int h0 = kv_h * 3;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int pos = *d_pos;
    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    int total_t = t_end - t_start;
    int chunk = (total_t + n_splits - 1) / n_splits;
    int ts = t_start + sp * chunk;
    int te = min(t_end, ts + chunk);

    if (ts >= te) {
        if (wid == 0 && lane < 3) {
            int h = h0 + lane;
            float* dst = split_buf + ((size_t)h * n_splits + sp) * 130;
            dst[0] = -1e30f;
            dst[1] = 0.0f;
        }
        return;
    }

    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q_scale = rsqrtf((float)dh) * S_LOG2E;
    float q0a[3], q0b[3], q1a[3], q1b[3];
    float rv[3], sk[3];
    float m[3], l[3], o0[3], o1[3], o2[3], o3[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h0 + g;
        const float* qh = Q + (size_t)h * dh;
        q0a[g] = qh[p0] * q_scale;
        q0b[g] = qh[p0 + 1] * q_scale;
        q1a[g] = qh[p1] * q_scale;
        q1b[g] = qh[p1 + 1] * q_scale;
        rv[g] = (lane < d_rel) ? (R[(size_t)h * d_rel + lane] * S_LOG2E) : 0.0f;
        sk[g] = sinks[h] * S_LOG2E;
        m[g] = (sp == 0 && wid == 0) ? 0.0f : -1e30f;
        l[g] = (sp == 0 && wid == 0) ? 1.0f : 0.0f;
        o0[g] = 0.0f; o1[g] = 0.0f; o2[g] = 0.0f; o3[g] = 0.0f;
    }

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    for (int t = ts + wid; t < te; t += 8) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        int dist = pos - t;
        float rb_l = (dist < rel_extent && lane < d_rel) ? rel_bias[(size_t)dist * d_rel + lane] : 0.0f;
        bool has_rb = (dist < rel_extent);

#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = q0a[g] * k0.x + q0b[g] * k0.y + q1a[g] * k1.x + q1b[g] * k1.y;
            float score = __shfl(s_warp_sum32(dot), 0, 32);
            if (has_rb) {
                score += __shfl(s_warp_sum32(rb_l * rv[g]), 0, 32);
            }
            score -= sk[g];
            float m_new = fmaxf(m[g], score);
            float alpha = (m[g] > -1e20f) ? exp2f(m[g] - m_new) : 0.0f;
            float p = exp2f(score - m_new);
            l[g] = l[g] * alpha + p;
            m[g] = m_new;
            o0[g] = fmaf(p, v0.x, o0[g] * alpha);
            o1[g] = fmaf(p, v0.y, o1[g] * alpha);
            o2[g] = fmaf(p, v1.x, o2[g] * alpha);
            o3[g] = fmaf(p, v1.y, o3[g] * alpha);
        }
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[wid] = m[g]; sm_l[wid] = l[g]; }
        sm_o[wid][p0] = o0[g]; sm_o[wid][p0 + 1] = o1[g];
        sm_o[wid][p1] = o2[g]; sm_o[wid][p1 + 1] = o3[g];
        __syncthreads();
        if (wid == 0) {
            float gm = sm_m[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) gm = fmaxf(gm, sm_m[w]);
            float gl = 0.0f;
            float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                float sc = (sm_m[w] > -1e20f) ? exp2f(sm_m[w] - gm) : 0.0f;
                gl += sm_l[w] * sc;
                go0 += sm_o[w][d0] * sc;
                go1 += sm_o[w][d1] * sc;
                go2 += sm_o[w][d2] * sc;
                go3 += sm_o[w][d3] * sc;
            }
            int h = h0 + g;
            float* dst = split_buf + ((size_t)h * n_splits + sp) * 130;
            if (lane == 0) { dst[0] = gm; dst[1] = gl; }
            dst[2 + d0] = go0;
            dst[2 + d1] = go1;
            dst[2 + d2] = go2;
            dst[2 + d3] = go3;
        }
        __syncthreads();
    }
}

extern "C" __global__ void attn_decode_splitkv_reduce_q8_k(const float* __restrict__ split_buf,
                                                           const float* __restrict__ gate,
                                                           q8_1_blk* __restrict__ out_q8,
                                                           int n_splits) {
    int h = blockIdx.x;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = wid * 32 + lane;
    const float* base = split_buf + (size_t)h * n_splits * 130;

    float gm = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        gm = fmaxf(gm, base[(size_t)sp * 130]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        float ms = base[(size_t)sp * 130];
        if (ms > -1e20f) {
            float sc = exp2f(ms - gm);
            gl += base[(size_t)sp * 130 + 1] * sc;
            go += base[(size_t)sp * 130 + 2 + d] * sc;
        }
    }
    float v = (go / gl) / (1.0f + exp2f(-gate[(size_t)h * 128 + d] * S_LOG2E));
    __shared__ int8_t smq[4][32];
    store_q8_1_from_warp(v, lane, out_q8 + h * 4 + wid, smq[wid]);
}

__device__ float g_b1_split_buf[32 * 16 * 130];

extern "C" int ink_attn_decode_gated_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx) {
    if (dh == 128 && n_head == n_head_kv * 3 && n_head <= 32) {
        int n_splits = 8;
        float* sbuf = nullptr;
        (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_b1_split_buf));
        attn_decode_gqa_splitkv_k<<<dim3(n_head_kv, n_splits), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, d_pos, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, n_splits);
        attn_decode_splitkv_reduce_q8_k<<<dim3(n_head), dim3(128), 0, s>>>(sbuf, gate, (q8_1_blk*)out_q8, n_splits);
        return 0;
    }
    inkling_attn_decode_gated_q8_k<<<dim3(n_head), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" __global__ void moe_gu_q4_0_b1_to_q8_k(const uint8_t* __restrict__ W,
                                                  const float* __restrict__ scales,
                                                  const q8_1_blk* __restrict__ Xq,
                                                  const int* __restrict__ itop,
                                                  q8_1_blk* __restrict__ act_q,
                                                  int n_ff, int k, float beta, float lb) {
    int slot = blockIdx.y;
    int exp_id = itop[slot];
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;
    int nblk = k >> 5;

    __shared__ q8_1_blk sm_x[96];
    if (nblk <= 96) {
        if (threadIdx.x < nblk) sm_x[threadIdx.x] = Xq[threadIdx.x];
        __syncthreads();
    }
    const q8_1_blk* xsrc = (nblk <= 96) ? sm_x : Xq;

    __shared__ float sm_v[32];
    if (row < n_ff) {
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
            const q8_1_blk xb = xsrc[b];
            ag += dot_q4_soa_q8_1_blk(qg_row[b], __half2float(dg_row[b]), xb);
            au += dot_q4_soa_q8_1_blk(qu_row[b], __half2float(du_row[b]), xb);
        }
        float sc = scales ? scales[exp_id] : 1.0f;
        float g = s_warp_sum8(ag) * sc;
        float u = s_warp_sum8(au) * sc;
        if (lane8 == 0) {
            float sg = 1.0f / (1.0f + exp2f(-g * S_LOG2E));
            float act = beta * tanhf(g / beta) * sg;
            float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
            sm_v[grp] = act * up;
        }
    } else if (lane8 == 0) {
        sm_v[grp] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        __shared__ int8_t sm_q[32];
        int qblk_per_slot = n_ff >> 5;
        store_q8_1_from_warp(sm_v[threadIdx.x], threadIdx.x, act_q + (size_t)slot * qblk_per_slot + blockIdx.x, sm_q);
    }
}

extern "C" int ink_moe_gu_q4_0_b1_to_q8(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* itop, void* act_q, int n_ff, int k, int n_used, float beta, float lb) {
    moe_gu_q4_0_b1_to_q8_k<<<dim3((n_ff + 31) / 32, n_used), dim3(256), 0, s>>>(W, scales, (const q8_1_blk*)Xq, itop, (q8_1_blk*)act_q, n_ff, k, beta, lb);
    return 0;
}

extern "C" __global__ void qkvrg_q4_b1_k(const uint8_t* __restrict__ wq,
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
    int blk_gid0 = blockIdx.x * 16;
    int gid = blk_gid0 + grp;

    int o1 = q_dim;
    int o2 = o1 + kv_dim;
    int o3 = o2 + kv_dim;
    int o4 = o3 + r_dim;
    int o5 = o4 + q_dim;
    int o6 = o5 + n_head;
    int o7 = o6 + dyn_dim;
    int o8 = o7 + dyn_dim;

    int nblk = k >> 5;
    __shared__ q8_1_blk sm_x[96];
    const q8_1_blk* blk_xq = nullptr;
    if (nblk <= 96 && blk_gid0 + 15 < o5) {
        if (blk_gid0 + 15 < o1 || (blk_gid0 >= o3 && blk_gid0 + 15 < o5)) blk_xq = Xq0;
        else if (blk_gid0 >= o1 && blk_gid0 + 15 < o2) blk_xq = Xq1;
        else if (blk_gid0 >= o2 && blk_gid0 + 15 < o3) blk_xq = Xq2;
    }
    if (blk_xq) {
        if (threadIdx.x < nblk) sm_x[threadIdx.x] = blk_xq[threadIdx.x];
        __syncthreads();
    }
    if (gid >= o8) return;

    if (gid < o5) {
        const uint8_t* W;
        const q8_1_blk* Xq;
        float* Y;
        int row;
        if (gid < o1) { W = wq; Xq = Xq0; Y = Q; row = gid; }
        else if (gid < o2) { W = wk; Xq = Xq1; Y = K; row = gid - o1; }
        else if (gid < o3) { W = wv; Xq = Xq2; Y = V; row = gid - o2; }
        else if (gid < o4) { W = wr; Xq = Xq0; Y = R; row = gid - o3; }
        else { W = wg; Xq = Xq0; Y = G; row = gid - o4; }

        const q8_1_blk* xsrc = blk_xq ? sm_x : Xq;
        size_t rbytes = q4_soa_row_bytes(nblk);
        const uint8_t* wrow = W + (size_t)row * rbytes;
        const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
        const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
        float acc = 0.0f;
        for (int b = lane8; b < nblk; b += 8) {
            acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), xsrc[b]);
        }
        acc = s_warp_sum8(acc);
        if (lane8 == 0) Y[row] = acc;
    } else {
        const uint16_t* W;
        const float* X;
        float* Y;
        int row;
        if (gid < o6) { W = wsink; X = X0; Y = S; row = gid - o5; }
        else if (gid < o7) { W = wkdyn; X = X1; Y = Kdyn; row = gid - o6; }
        else { W = wvdyn; X = X2; Y = Vdyn; row = gid - o7; }
        const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)row * k);
        const float4* x4 = reinterpret_cast<const float4*>(X);
        int k8 = k >> 3;
        float acc = 0.0f;
        for (int j = lane8; j < k8; j += 8) {
            uint4 wv = w4[j];
            float4 x0 = x4[j * 2];
            float4 x1 = x4[j * 2 + 1];
            acc = fmaf(s_bf16_f32((uint16_t)(wv.x & 0xFFFFu)), x0.x, acc);
            acc = fmaf(s_bf16_f32((uint16_t)(wv.x >> 16)), x0.y, acc);
            acc = fmaf(s_bf16_f32((uint16_t)(wv.y & 0xFFFFu)), x0.z, acc);
            acc = fmaf(s_bf16_f32((uint16_t)(wv.y >> 16)), x0.w, acc);
            acc = fmaf(s_bf16_f32((uint16_t)(wv.z & 0xFFFFu)), x1.x, acc);
            acc = fmaf(s_bf16_f32((uint16_t)(wv.z >> 16)), x1.y, acc);
            acc = fmaf(s_bf16_f32((uint16_t)(wv.w & 0xFFFFu)), x1.z, acc);
            acc = fmaf(s_bf16_f32((uint16_t)(wv.w >> 16)), x1.w, acc);
        }
        acc = s_warp_sum8(acc);
        if (lane8 == 0) Y[row] = acc;
    }
}

extern "C" __global__ void wo_conv_res_q4_b1_8_k(const uint8_t* __restrict__ W,
                                                 const q8_1_blk* __restrict__ Xq,
                                                 const float* __restrict__ conv_w,
                                                 float* __restrict__ hist,
                                                 const float* __restrict__ add_src,
                                                 float* __restrict__ block_dst,
                                                 float* __restrict__ out,
                                                 int n_embd, int q_dim) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    int nblk = q_dim >> 5;
    __shared__ q8_1_blk sm_x[96];
    if (nblk <= 96) {
        if (threadIdx.x < nblk) sm_x[threadIdx.x] = Xq[threadIdx.x];
        __syncthreads();
    }
    if (row >= n_embd) return;
    const q8_1_blk* xsrc = (nblk <= 96) ? sm_x : Xq;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), xsrc[b]);
    }
    acc = s_warp_sum8(acc);
    if (lane8 == 0) {
        float* hc = hist + (size_t)row * 3;
        const float* wc = conv_w + (size_t)row * 4;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        float conv_out = (acc + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * acc) * rsqrtf(5.0f);
        hc[0] = h1; hc[1] = h2; hc[2] = acc;
        if (block_dst) {
            block_dst[row] = add_src[row];
            out[row] = conv_out;
        } else {
            out[row] = add_src[row] + conv_out;
        }
    }
}

extern "C" __global__ void gemv_q4_b1_8_k(const uint8_t* __restrict__ W,
                                          const q8_1_blk* __restrict__ Xq,
                                          float* __restrict__ Y,
                                          int n, int k) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    int nblk = k >> 5;
    __shared__ q8_1_blk sm_x[96];
    if (nblk <= 96) {
        if (threadIdx.x < nblk) sm_x[threadIdx.x] = Xq[threadIdx.x];
        __syncthreads();
    }
    if (row >= n) return;
    const q8_1_blk* xsrc = (nblk <= 96) ? sm_x : Xq;
    size_t rbytes = q4_soa_row_bytes(nblk);
    const uint8_t* wrow = W + (size_t)row * rbytes;
    const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
    const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
    float acc = 0.0f;
    for (int b = lane8; b < nblk; b += 8) {
        acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), xsrc[b]);
    }
    acc = s_warp_sum8(acc);
    if (lane8 == 0) Y[row] = acc;
}

extern "C" __global__ void moe_dn_conv_add_q4_b1_4_k(const uint8_t* __restrict__ W,
                                                     const float* __restrict__ scales,
                                                     const q8_1_blk* __restrict__ act_q,
                                                     const int* __restrict__ itop,
                                                     const float* __restrict__ wtop,
                                                     const float* __restrict__ conv_w,
                                                     float* __restrict__ hist,
                                                     const float* __restrict__ ffn_inp,
                                                     float* __restrict__ cur,
                                                     int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 2;
    int lane4 = threadIdx.x & 3;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;

    int nblk = n_ff >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)n_embd * row_bytes;

    float total = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        int exp_id = itop[s];
        float wt = wtop[s] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        const uint4* qrow = reinterpret_cast<const uint4*>(wrow);
        const __half* drow = reinterpret_cast<const __half*>(wrow + (size_t)nblk * 16);
        const q8_1_blk* aq = act_q + (size_t)s * nblk;
        float acc = 0.0f;
        for (int b = lane4; b < nblk; b += 4) {
            acc += dot_q4_soa_q8_1_blk(qrow[b], __half2float(drow[b]), aq[b]);
        }
        total += acc * wt;
    }
    float v = s_warp_sum4(total);
    if (lane4 == 0) {
        float* hc = hist + (size_t)row * 3;
        const float* wc = conv_w + (size_t)row * 4;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        float conv_out = (v + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * v) * rsqrtf(5.0f);
        hc[0] = h1; hc[1] = h2; hc[2] = v;
        cur[row] = ffn_inp[row] + conv_out;
    }
}

extern "C" int ink_qkvrg_q4_b1(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wr, const uint8_t* wg, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const void* Xq0, const void* Xq1, const void* Xq2, const float* X0, const float* X1, const float* X2, float* Q, float* K, float* V, float* R, float* G, float* S, float* Kdyn, float* Vdyn, int k, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim) {
    int total_rows = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    qkvrg_q4_b1_k<<<dim3((total_rows + 15) / 16), dim3(128), 0, s>>>(wq, wk, wv, wr, wg, wsink, wkdyn, wvdyn, (const q8_1_blk*)Xq0, (const q8_1_blk*)Xq1, (const q8_1_blk*)Xq2, X0, X1, X2, Q, K, V, R, G, S, Kdyn, Vdyn, k, q_dim, kv_dim, r_dim, n_head, dyn_dim);
    return 0;
}

extern "C" int ink_gemv_q4_b1(hipStream_t s, const uint8_t* W, const void* Xq, float* Y, int n, int k) {
    gemv_q4_b1_8_k<<<dim3((n + 15) / 16), dim3(128), 0, s>>>(W, (const q8_1_blk*)Xq, Y, n, k);
    return 0;
}

extern "C" __global__ void inkling_attn_multi_gated_q8_k(const float* __restrict__ Q,
                                                         const float* __restrict__ R,
                                                         const float* __restrict__ sinks,
                                                         const float* __restrict__ rel_bias,
                                                         const __half* __restrict__ k_cache_pool,
                                                         const __half* __restrict__ v_cache_pool,
                                                         const float* __restrict__ gate,
                                                         const int* __restrict__ d_seq_ids,
                                                         const int* __restrict__ d_positions,
                                                         q8_1_blk* __restrict__ out_q8,
                                                         int n_head, int n_head_kv,
                                                         int dh, int d_rel,
                                                         int rel_extent, int swa_window,
                                                         int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int seq = d_seq_ids[bt];
    int pos = d_positions[bt];
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    __shared__ float qs[128];
    __shared__ float rs[16];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[(size_t)bt * q_dim + (size_t)h * dh + d] * q_scale;
    }
    if (threadIdx.x < d_rel) {
        rs[threadIdx.x] = R[(size_t)bt * r_dim + (size_t)h * d_rel + threadIdx.x];
    }
    __syncthreads();

    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[(size_t)bt * n_head + h];

    size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
    const __half* kc_h = k_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q0a = qs[p0], q0b = qs[p0 + 1], q1a = qs[p1], q1b = qs[p1 + 1];

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        float dot = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
        dot = s_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = s_warp_sum32(rp);
            rbias = __shfl(rp, 0, 32);
        }
        score += rbias - sink_h;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        o0 = fmaf(p, v0.x, o0 * alpha);
        o1 = fmaf(p, v0.y, o1 * alpha);
        o2 = fmaf(p, v1.x, o2 * alpha);
        o3 = fmaf(p, v1.y, o3 * alpha);
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    sm_o[wid][p0] = o0; sm_o[wid][p0 + 1] = o1; sm_o[wid][p1] = o2; sm_o[wid][p1 + 1] = o3;
    __syncthreads();

    if (wid == 0) {
        float gm = sm_m[0];
        for (int w = 1; w < nw; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
        int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
        for (int w = 0; w < nw; ++w) {
            float sc = expf(sm_m[w] - gm);
            gl += sm_l[w] * sc;
            go0 += sm_o[w][d0] * sc;
            go1 += sm_o[w][d1] * sc;
            go2 += sm_o[w][d2] * sc;
            go3 += sm_o[w][d3] * sc;
        }
        float inv_l = 1.0f / gl;
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        float v0 = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        float v1 = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        float v2 = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        float v3 = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
        __shared__ int8_t smq[32];
        size_t qbase = ((size_t)bt * n_head + h) * 4;
        store_q8_1_from_warp(v0, lane, out_q8 + qbase + 0, smq);
        store_q8_1_from_warp(v1, lane, out_q8 + qbase + 1, smq);
        store_q8_1_from_warp(v2, lane, out_q8 + qbase + 2, smq);
        store_q8_1_from_warp(v3, lane, out_q8 + qbase + 3, smq);
    }
}

extern "C" __global__ void inkling_attn_multi_gqa3_q8_k(const float* __restrict__ Q,
                                                        const float* __restrict__ R,
                                                        const float* __restrict__ sinks,
                                                        const float* __restrict__ rel_bias,
                                                        const __half* __restrict__ k_cache_pool,
                                                        const __half* __restrict__ v_cache_pool,
                                                        const float* __restrict__ gate,
                                                        const int* __restrict__ d_seq_ids,
                                                        const int* __restrict__ d_positions,
                                                        q8_1_blk* __restrict__ out_q8,
                                                        int n_head, int n_head_kv,
                                                        int dh, int d_rel,
                                                        int rel_extent, int swa_window,
                                                        int is_swa, int max_ctx) {
    int kv_h = blockIdx.x;
    int bt = blockIdx.y;
    int seq = d_seq_ids[bt];
    int pos = d_positions[bt];
    int h0 = kv_h * 3;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q_scale = rsqrtf((float)dh) * S_LOG2E;
    float q0a[3], q0b[3], q1a[3], q1b[3];
    float rv[3], sk[3];
    float m[3], l[3], o0[3], o1[3], o2[3], o3[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h0 + g;
        const float* qh = Q + (size_t)bt * q_dim + (size_t)h * dh;
        q0a[g] = qh[p0] * q_scale;
        q0b[g] = qh[p0 + 1] * q_scale;
        q1a[g] = qh[p1] * q_scale;
        q1b[g] = qh[p1 + 1] * q_scale;
        rv[g] = (lane < d_rel) ? (R[(size_t)bt * r_dim + (size_t)h * d_rel + lane] * S_LOG2E) : 0.0f;
        sk[g] = sinks[(size_t)bt * n_head + h] * S_LOG2E;
        m[g] = (wid == 0) ? 0.0f : -1e30f;
        l[g] = (wid == 0) ? 1.0f : 0.0f;
        o0[g] = 0.0f; o1[g] = 0.0f; o2[g] = 0.0f; o3[g] = 0.0f;
    }

    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
    const __half* kc_h = k_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache_pool + (size_t)seq * seq_kv_stride + (size_t)kv_h * max_ctx * dh;

    for (int t = t_start + wid; t < t_end; t += 8) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        int dist = pos - t;
        bool has_rb = (dist < rel_extent);
        float rb_l = (has_rb && lane < d_rel) ? rel_bias[(size_t)dist * d_rel + lane] : 0.0f;

#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = q0a[g] * k0.x + q0b[g] * k0.y + q1a[g] * k1.x + q1b[g] * k1.y;
            float score = __shfl(s_warp_sum32(dot), 0, 32);
            if (has_rb) score += __shfl(s_warp_sum32(rb_l * rv[g]), 0, 32);
            score -= sk[g];
            float m_new = fmaxf(m[g], score);
            float alpha = (m[g] > -1e20f) ? exp2f(m[g] - m_new) : 0.0f;
            float p = exp2f(score - m_new);
            l[g] = l[g] * alpha + p;
            m[g] = m_new;
            o0[g] = fmaf(p, v0.x, o0[g] * alpha);
            o1[g] = fmaf(p, v0.y, o1[g] * alpha);
            o2[g] = fmaf(p, v1.x, o2[g] * alpha);
            o3[g] = fmaf(p, v1.y, o3[g] * alpha);
        }
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    __shared__ int8_t smq[32];
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[wid] = m[g]; sm_l[wid] = l[g]; }
        sm_o[wid][p0] = o0[g]; sm_o[wid][p0 + 1] = o1[g];
        sm_o[wid][p1] = o2[g]; sm_o[wid][p1 + 1] = o3[g];
        __syncthreads();
        if (wid == 0) {
            float gm = sm_m[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) gm = fmaxf(gm, sm_m[w]);
            float gl = 0.0f;
            float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                float sc = (sm_m[w] > -1e20f) ? exp2f(sm_m[w] - gm) : 0.0f;
                gl += sm_l[w] * sc;
                go0 += sm_o[w][d0] * sc;
                go1 += sm_o[w][d1] * sc;
                go2 += sm_o[w][d2] * sc;
                go3 += sm_o[w][d3] * sc;
            }
            float inv_l = 1.0f / gl;
            int h = h0 + g;
            const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
            float v0 = (go0 * inv_l) / (1.0f + exp2f(-gh[d0] * S_LOG2E));
            float v1 = (go1 * inv_l) / (1.0f + exp2f(-gh[d1] * S_LOG2E));
            float v2 = (go2 * inv_l) / (1.0f + exp2f(-gh[d2] * S_LOG2E));
            float v3 = (go3 * inv_l) / (1.0f + exp2f(-gh[d3] * S_LOG2E));
            size_t qbase = ((size_t)bt * n_head + h) * 4;
            store_q8_1_from_warp(v0, lane, out_q8 + qbase + 0, smq);
            store_q8_1_from_warp(v1, lane, out_q8 + qbase + 1, smq);
            store_q8_1_from_warp(v2, lane, out_q8 + qbase + 2, smq);
            store_q8_1_from_warp(v3, lane, out_q8 + qbase + 3, smq);
        }
        __syncthreads();
    }
}

extern "C" int ink_attn_multi_gated_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc_pool, const __half* vc_pool, const float* gate, const int* d_seq_ids, const int* d_positions, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    if (dh == 128 && n_head == n_head_kv * 3 && B >= 8) {
        inkling_attn_multi_gqa3_q8_k<<<dim3(n_head_kv, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, gate, d_seq_ids, d_positions, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
        return 0;
    }
    inkling_attn_multi_gated_q8_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, gate, d_seq_ids, d_positions, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" __global__ void moe_gu_q4_0_sudot4_to_q8_k(const uint8_t* __restrict__ W,
                                                      const float* __restrict__ scales,
                                                      const q8_1_blk* __restrict__ Xq,
                                                      const int* __restrict__ exp_counts,
                                                      const int* __restrict__ exp_offsets,
                                                      const int* __restrict__ tok_sorted,
                                                      const int* __restrict__ slot_sorted,
                                                      q8_1_blk* __restrict__ act_q,
                                                      int n_ff, int k, int n_used,
                                                      float beta, float lb) {
    int exp_id = exp_counts[128 + blockIdx.y];
    if (exp_id < 0) return;
    int cnt = exp_counts[exp_id];
    int z_id = blockIdx.z;
    int z_n = gridDim.z;
    if (cnt == 0 || z_id * 8 >= cnt) return;
    int off0 = exp_offsets[exp_id];

    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int nblk = k >> 5;
    int qblk_per_slot = n_ff >> 5;
    size_t row_bytes = q4_soa_row_bytes(nblk);
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    int r_safe = min(row, n_ff - 1);
    const uint8_t* wg = W + (size_t)exp_id * exp_bytes + (size_t)r_safe * row_bytes;
    const uint8_t* wu = W + (size_t)exp_id * exp_bytes + (size_t)(n_ff + r_safe) * row_bytes;
    const uint4* qg_row = reinterpret_cast<const uint4*>(wg);
    const __half* dg_row = reinterpret_cast<const __half*>(wg + (size_t)nblk * 16);
    const uint4* qu_row = reinterpret_cast<const uint4*>(wu);
    const __half* du_row = reinterpret_cast<const __half*>(wu + (size_t)nblk * 16);
    float sc = scales ? scales[exp_id] : 1.0f;

    __shared__ float sm_v[8][32];
    __shared__ int8_t sm_q[8][32];
    __shared__ q8_1_blk sm_x0[96];
    __shared__ q8_1_blk sm_x1[96];

    if (cnt == 1) {
        int tid0 = tok_sorted[off0];
        int sid0 = slot_sorted[off0];
        const q8_1_blk* xsrc0 = Xq + (size_t)tid0 * nblk;
        if (nblk <= 96) {
            if (threadIdx.x < nblk) sm_x0[threadIdx.x] = xsrc0[threadIdx.x];
            __syncthreads();
            xsrc0 = sm_x0;
        }
        float ag0 = 0.0f, au0 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                const q8_1_blk xb = xsrc0[b];
                ag0 += dot_q4_soa_q8_1_blk(qg_row[b], __half2float(dg_row[b]), xb);
                au0 += dot_q4_soa_q8_1_blk(qu_row[b], __half2float(du_row[b]), xb);
            }
        }
        float g = s_warp_sum8(ag0) * sc;
        float u = s_warp_sum8(au0) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                float sg = 1.0f / (1.0f + expf(-g));
                float act = beta * tanhf(g / beta) * sg;
                float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                sm_v[0][grp] = act * up;
            } else {
                sm_v[0][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid == 0) {
            size_t qidx = ((size_t)tid0 * n_used + sid0) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[0][lane], lane, act_q + qidx, sm_q[0]);
        }
        return;
    }
    if (cnt == 2) {
        int tid0 = tok_sorted[off0], sid0 = slot_sorted[off0];
        int tid1 = tok_sorted[off0 + 1], sid1 = slot_sorted[off0 + 1];
        const q8_1_blk* xsrc0 = Xq + (size_t)tid0 * nblk;
        const q8_1_blk* xsrc1 = Xq + (size_t)tid1 * nblk;
        if (nblk <= 96) {
            if (threadIdx.x < nblk) sm_x0[threadIdx.x] = xsrc0[threadIdx.x];
            else if (threadIdx.x >= 128 && (threadIdx.x - 128) < nblk) sm_x1[threadIdx.x - 128] = xsrc1[threadIdx.x - 128];
            __syncthreads();
            xsrc0 = sm_x0;
            xsrc1 = sm_x1;
        }
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                uint4 qg = qg_row[b];
                uint4 qu = qu_row[b];
                float dg = __half2float(dg_row[b]);
                float du = __half2float(du_row[b]);
                int32_t g0 = (int32_t)(qg.x & 0x0F0F0F0Fu), g1 = (int32_t)(qg.y & 0x0F0F0F0Fu);
                int32_t g2 = (int32_t)(qg.z & 0x0F0F0F0Fu), g3 = (int32_t)(qg.w & 0x0F0F0F0Fu);
                int32_t g4 = (int32_t)((qg.x >> 4) & 0x0F0F0F0Fu), g5 = (int32_t)((qg.y >> 4) & 0x0F0F0F0Fu);
                int32_t g6 = (int32_t)((qg.z >> 4) & 0x0F0F0F0Fu), g7 = (int32_t)((qg.w >> 4) & 0x0F0F0F0Fu);
                int32_t u0 = (int32_t)(qu.x & 0x0F0F0F0Fu), u1 = (int32_t)(qu.y & 0x0F0F0F0Fu);
                int32_t u2 = (int32_t)(qu.z & 0x0F0F0F0Fu), u3 = (int32_t)(qu.w & 0x0F0F0F0Fu);
                int32_t u4 = (int32_t)((qu.x >> 4) & 0x0F0F0F0Fu), u5 = (int32_t)((qu.y >> 4) & 0x0F0F0F0Fu);
                int32_t u6 = (int32_t)((qu.z >> 4) & 0x0F0F0F0Fu), u7 = (int32_t)((qu.w >> 4) & 0x0F0F0F0Fu);
                const q8_1_blk xb0 = xsrc0[b];
                const q8_1_blk xb1 = xsrc1[b];
                int32_t sg0 = 0, su0 = 0, sg1 = 0, su1 = 0;
                sg0 = __builtin_amdgcn_sudot4(false, g0, true, xb0.q[0], sg0, false);
                sg0 = __builtin_amdgcn_sudot4(false, g1, true, xb0.q[1], sg0, false);
                sg0 = __builtin_amdgcn_sudot4(false, g2, true, xb0.q[2], sg0, false);
                sg0 = __builtin_amdgcn_sudot4(false, g3, true, xb0.q[3], sg0, false);
                sg0 = __builtin_amdgcn_sudot4(false, g4, true, xb0.q[4], sg0, false);
                sg0 = __builtin_amdgcn_sudot4(false, g5, true, xb0.q[5], sg0, false);
                sg0 = __builtin_amdgcn_sudot4(false, g6, true, xb0.q[6], sg0, false);
                sg0 = __builtin_amdgcn_sudot4(false, g7, true, xb0.q[7], sg0, false);
                ag0 = fmaf(dg * xb0.d, (float)(sg0 - 8 * xb0.s), ag0);
                su0 = __builtin_amdgcn_sudot4(false, u0, true, xb0.q[0], su0, false);
                su0 = __builtin_amdgcn_sudot4(false, u1, true, xb0.q[1], su0, false);
                su0 = __builtin_amdgcn_sudot4(false, u2, true, xb0.q[2], su0, false);
                su0 = __builtin_amdgcn_sudot4(false, u3, true, xb0.q[3], su0, false);
                su0 = __builtin_amdgcn_sudot4(false, u4, true, xb0.q[4], su0, false);
                su0 = __builtin_amdgcn_sudot4(false, u5, true, xb0.q[5], su0, false);
                su0 = __builtin_amdgcn_sudot4(false, u6, true, xb0.q[6], su0, false);
                su0 = __builtin_amdgcn_sudot4(false, u7, true, xb0.q[7], su0, false);
                au0 = fmaf(du * xb0.d, (float)(su0 - 8 * xb0.s), au0);
                sg1 = __builtin_amdgcn_sudot4(false, g0, true, xb1.q[0], sg1, false);
                sg1 = __builtin_amdgcn_sudot4(false, g1, true, xb1.q[1], sg1, false);
                sg1 = __builtin_amdgcn_sudot4(false, g2, true, xb1.q[2], sg1, false);
                sg1 = __builtin_amdgcn_sudot4(false, g3, true, xb1.q[3], sg1, false);
                sg1 = __builtin_amdgcn_sudot4(false, g4, true, xb1.q[4], sg1, false);
                sg1 = __builtin_amdgcn_sudot4(false, g5, true, xb1.q[5], sg1, false);
                sg1 = __builtin_amdgcn_sudot4(false, g6, true, xb1.q[6], sg1, false);
                sg1 = __builtin_amdgcn_sudot4(false, g7, true, xb1.q[7], sg1, false);
                ag1 = fmaf(dg * xb1.d, (float)(sg1 - 8 * xb1.s), ag1);
                su1 = __builtin_amdgcn_sudot4(false, u0, true, xb1.q[0], su1, false);
                su1 = __builtin_amdgcn_sudot4(false, u1, true, xb1.q[1], su1, false);
                su1 = __builtin_amdgcn_sudot4(false, u2, true, xb1.q[2], su1, false);
                su1 = __builtin_amdgcn_sudot4(false, u3, true, xb1.q[3], su1, false);
                su1 = __builtin_amdgcn_sudot4(false, u4, true, xb1.q[4], su1, false);
                su1 = __builtin_amdgcn_sudot4(false, u5, true, xb1.q[5], su1, false);
                su1 = __builtin_amdgcn_sudot4(false, u6, true, xb1.q[6], su1, false);
                su1 = __builtin_amdgcn_sudot4(false, u7, true, xb1.q[7], su1, false);
                au1 = fmaf(du * xb1.d, (float)(su1 - 8 * xb1.s), au1);
            }
        }
        float g0 = s_warp_sum8(ag0) * sc, u0 = s_warp_sum8(au0) * sc;
        float g1 = s_warp_sum8(ag1) * sc, u1 = s_warp_sum8(au1) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                float sg0 = 1.0f / (1.0f + expf(-g0));
                float a0 = beta * tanhf(g0 / beta) * sg0;
                float up0 = (lb > 0.0f) ? (lb * tanhf(u0 / lb)) : u0;
                sm_v[0][grp] = a0 * up0;
                float sg1 = 1.0f / (1.0f + expf(-g1));
                float a1 = beta * tanhf(g1 / beta) * sg1;
                float up1 = (lb > 0.0f) ? (lb * tanhf(u1 / lb)) : u1;
                sm_v[1][grp] = a1 * up1;
            } else {
                sm_v[0][grp] = 0.0f;
                sm_v[1][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid == 0) {
            size_t qidx = ((size_t)tid0 * n_used + sid0) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[0][lane], lane, act_q + qidx, sm_q[0]);
        } else if (wid == 1) {
            size_t qidx = ((size_t)tid1 * n_used + sid1) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[1][lane], lane, act_q + qidx, sm_q[1]);
        }
        return;
    }
    if (cnt == 3) {
        int t0i = tok_sorted[off0], s0i = slot_sorted[off0];
        int t1i = tok_sorted[off0 + 1], s1i = slot_sorted[off0 + 1];
        int t2i = tok_sorted[off0 + 2], s2i = slot_sorted[off0 + 2];
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f, ag2 = 0.0f, au2 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                uint4 qg = qg_row[b];
                uint4 qu = qu_row[b];
                float dg = __half2float(dg_row[b]);
                float du = __half2float(du_row[b]);
                int32_t g0 = (int32_t)(qg.x & 0x0F0F0F0Fu), g1 = (int32_t)(qg.y & 0x0F0F0F0Fu);
                int32_t g2 = (int32_t)(qg.z & 0x0F0F0F0Fu), g3 = (int32_t)(qg.w & 0x0F0F0F0Fu);
                int32_t g4 = (int32_t)((qg.x >> 4) & 0x0F0F0F0Fu), g5 = (int32_t)((qg.y >> 4) & 0x0F0F0F0Fu);
                int32_t g6 = (int32_t)((qg.z >> 4) & 0x0F0F0F0Fu), g7 = (int32_t)((qg.w >> 4) & 0x0F0F0F0Fu);
                int32_t u0 = (int32_t)(qu.x & 0x0F0F0F0Fu), u1 = (int32_t)(qu.y & 0x0F0F0F0Fu);
                int32_t u2 = (int32_t)(qu.z & 0x0F0F0F0Fu), u3 = (int32_t)(qu.w & 0x0F0F0F0Fu);
                int32_t u4 = (int32_t)((qu.x >> 4) & 0x0F0F0F0Fu), u5 = (int32_t)((qu.y >> 4) & 0x0F0F0F0Fu);
                int32_t u6 = (int32_t)((qu.z >> 4) & 0x0F0F0F0Fu), u7 = (int32_t)((qu.w >> 4) & 0x0F0F0F0Fu);
                const q8_1_blk xbs[3] = { Xq[(size_t)t0i * nblk + b], Xq[(size_t)t1i * nblk + b], Xq[(size_t)t2i * nblk + b] };
                float* agp[3] = { &ag0, &ag1, &ag2 };
                float* aup[3] = { &au0, &au1, &au2 };
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    int32_t sg = 0, su = 0;
                    sg = __builtin_amdgcn_sudot4(false, g0, true, xbs[m].q[0], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g1, true, xbs[m].q[1], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g2, true, xbs[m].q[2], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g3, true, xbs[m].q[3], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g4, true, xbs[m].q[4], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g5, true, xbs[m].q[5], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g6, true, xbs[m].q[6], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g7, true, xbs[m].q[7], sg, false);
                    *agp[m] = fmaf(dg * xbs[m].d, (float)(sg - 8 * xbs[m].s), *agp[m]);
                    su = __builtin_amdgcn_sudot4(false, u0, true, xbs[m].q[0], su, false);
                    su = __builtin_amdgcn_sudot4(false, u1, true, xbs[m].q[1], su, false);
                    su = __builtin_amdgcn_sudot4(false, u2, true, xbs[m].q[2], su, false);
                    su = __builtin_amdgcn_sudot4(false, u3, true, xbs[m].q[3], su, false);
                    su = __builtin_amdgcn_sudot4(false, u4, true, xbs[m].q[4], su, false);
                    su = __builtin_amdgcn_sudot4(false, u5, true, xbs[m].q[5], su, false);
                    su = __builtin_amdgcn_sudot4(false, u6, true, xbs[m].q[6], su, false);
                    su = __builtin_amdgcn_sudot4(false, u7, true, xbs[m].q[7], su, false);
                    *aup[m] = fmaf(du * xbs[m].d, (float)(su - 8 * xbs[m].s), *aup[m]);
                }
            }
        }
        float g0 = s_warp_sum8(ag0) * sc, u0 = s_warp_sum8(au0) * sc;
        float g1 = s_warp_sum8(ag1) * sc, u1 = s_warp_sum8(au1) * sc;
        float g2 = s_warp_sum8(ag2) * sc, u2 = s_warp_sum8(au2) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                sm_v[0][grp] = (beta * tanhf(g0 / beta) / (1.0f + expf(-g0))) * ((lb > 0.0f) ? (lb * tanhf(u0 / lb)) : u0);
                sm_v[1][grp] = (beta * tanhf(g1 / beta) / (1.0f + expf(-g1))) * ((lb > 0.0f) ? (lb * tanhf(u1 / lb)) : u1);
                sm_v[2][grp] = (beta * tanhf(g2 / beta) / (1.0f + expf(-g2))) * ((lb > 0.0f) ? (lb * tanhf(u2 / lb)) : u2);
            } else {
                sm_v[0][grp] = 0.0f; sm_v[1][grp] = 0.0f; sm_v[2][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid < 3) {
            int ti = (wid == 0) ? t0i : ((wid == 1) ? t1i : t2i);
            int si = (wid == 0) ? s0i : ((wid == 1) ? s1i : s2i);
            size_t qidx = ((size_t)ti * n_used + si) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[wid][lane], lane, act_q + qidx, sm_q[wid]);
        }
        return;
    }
    if (cnt == 4) {
        int t0i = tok_sorted[off0], s0i = slot_sorted[off0];
        int t1i = tok_sorted[off0 + 1], s1i = slot_sorted[off0 + 1];
        int t2i = tok_sorted[off0 + 2], s2i = slot_sorted[off0 + 2];
        int t3i = tok_sorted[off0 + 3], s3i = slot_sorted[off0 + 3];
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f, ag2 = 0.0f, au2 = 0.0f, ag3 = 0.0f, au3 = 0.0f;
        if (row < n_ff) {
            for (int b = lane8; b < nblk; b += 8) {
                uint4 qg = qg_row[b];
                uint4 qu = qu_row[b];
                float dg = __half2float(dg_row[b]);
                float du = __half2float(du_row[b]);
                int32_t g0 = (int32_t)(qg.x & 0x0F0F0F0Fu), g1 = (int32_t)(qg.y & 0x0F0F0F0Fu);
                int32_t g2 = (int32_t)(qg.z & 0x0F0F0F0Fu), g3 = (int32_t)(qg.w & 0x0F0F0F0Fu);
                int32_t g4 = (int32_t)((qg.x >> 4) & 0x0F0F0F0Fu), g5 = (int32_t)((qg.y >> 4) & 0x0F0F0F0Fu);
                int32_t g6 = (int32_t)((qg.z >> 4) & 0x0F0F0F0Fu), g7 = (int32_t)((qg.w >> 4) & 0x0F0F0F0Fu);
                int32_t u0 = (int32_t)(qu.x & 0x0F0F0F0Fu), u1 = (int32_t)(qu.y & 0x0F0F0F0Fu);
                int32_t u2 = (int32_t)(qu.z & 0x0F0F0F0Fu), u3 = (int32_t)(qu.w & 0x0F0F0F0Fu);
                int32_t u4 = (int32_t)((qu.x >> 4) & 0x0F0F0F0Fu), u5 = (int32_t)((qu.y >> 4) & 0x0F0F0F0Fu);
                int32_t u6 = (int32_t)((qu.z >> 4) & 0x0F0F0F0Fu), u7 = (int32_t)((qu.w >> 4) & 0x0F0F0F0Fu);
                const q8_1_blk xbs[4] = {
                    Xq[(size_t)t0i * nblk + b], Xq[(size_t)t1i * nblk + b],
                    Xq[(size_t)t2i * nblk + b], Xq[(size_t)t3i * nblk + b]
                };
                float* agp[4] = { &ag0, &ag1, &ag2, &ag3 };
                float* aup[4] = { &au0, &au1, &au2, &au3 };
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    int32_t sg = 0, su = 0;
                    sg = __builtin_amdgcn_sudot4(false, g0, true, xbs[m].q[0], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g1, true, xbs[m].q[1], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g2, true, xbs[m].q[2], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g3, true, xbs[m].q[3], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g4, true, xbs[m].q[4], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g5, true, xbs[m].q[5], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g6, true, xbs[m].q[6], sg, false);
                    sg = __builtin_amdgcn_sudot4(false, g7, true, xbs[m].q[7], sg, false);
                    *agp[m] = fmaf(dg * xbs[m].d, (float)(sg - 8 * xbs[m].s), *agp[m]);
                    su = __builtin_amdgcn_sudot4(false, u0, true, xbs[m].q[0], su, false);
                    su = __builtin_amdgcn_sudot4(false, u1, true, xbs[m].q[1], su, false);
                    su = __builtin_amdgcn_sudot4(false, u2, true, xbs[m].q[2], su, false);
                    su = __builtin_amdgcn_sudot4(false, u3, true, xbs[m].q[3], su, false);
                    su = __builtin_amdgcn_sudot4(false, u4, true, xbs[m].q[4], su, false);
                    su = __builtin_amdgcn_sudot4(false, u5, true, xbs[m].q[5], su, false);
                    su = __builtin_amdgcn_sudot4(false, u6, true, xbs[m].q[6], su, false);
                    su = __builtin_amdgcn_sudot4(false, u7, true, xbs[m].q[7], su, false);
                    *aup[m] = fmaf(du * xbs[m].d, (float)(su - 8 * xbs[m].s), *aup[m]);
                }
            }
        }
        float g0 = s_warp_sum8(ag0) * sc, u0 = s_warp_sum8(au0) * sc;
        float g1 = s_warp_sum8(ag1) * sc, u1 = s_warp_sum8(au1) * sc;
        float g2 = s_warp_sum8(ag2) * sc, u2 = s_warp_sum8(au2) * sc;
        float g3 = s_warp_sum8(ag3) * sc, u3 = s_warp_sum8(au3) * sc;
        if (lane8 == 0) {
            if (row < n_ff) {
                sm_v[0][grp] = (beta * tanhf(g0 / beta) / (1.0f + expf(-g0))) * ((lb > 0.0f) ? (lb * tanhf(u0 / lb)) : u0);
                sm_v[1][grp] = (beta * tanhf(g1 / beta) / (1.0f + expf(-g1))) * ((lb > 0.0f) ? (lb * tanhf(u1 / lb)) : u1);
                sm_v[2][grp] = (beta * tanhf(g2 / beta) / (1.0f + expf(-g2))) * ((lb > 0.0f) ? (lb * tanhf(u2 / lb)) : u2);
                sm_v[3][grp] = (beta * tanhf(g3 / beta) / (1.0f + expf(-g3))) * ((lb > 0.0f) ? (lb * tanhf(u3 / lb)) : u3);
            } else {
                sm_v[0][grp] = 0.0f; sm_v[1][grp] = 0.0f; sm_v[2][grp] = 0.0f; sm_v[3][grp] = 0.0f;
            }
        }
        __syncthreads();
        if (wid < 4) {
            int ti = (wid == 0) ? t0i : ((wid == 1) ? t1i : ((wid == 2) ? t2i : t3i));
            int si = (wid == 0) ? s0i : ((wid == 1) ? s1i : ((wid == 2) ? s2i : s3i));
            size_t qidx = ((size_t)ti * n_used + si) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[wid][lane], lane, act_q + qidx, sm_q[wid]);
        }
        return;
    }

    for (int t0 = z_id * 8; t0 < cnt; t0 += z_n * 8) {
        int nt = min(8, cnt - t0);
        int tids[8];
        int sids[8];
        for (int m = 0; m < nt; ++m) {
            tids[m] = tok_sorted[off0 + t0 + m];
            sids[m] = slot_sorted[off0 + t0 + m];
        }
        float ag[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float au[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (row < n_ff) {
            if (nt == 8) {
                const q8_1_blk* xps[8] = {
                    Xq + (size_t)tids[0] * nblk, Xq + (size_t)tids[1] * nblk,
                    Xq + (size_t)tids[2] * nblk, Xq + (size_t)tids[3] * nblk,
                    Xq + (size_t)tids[4] * nblk, Xq + (size_t)tids[5] * nblk,
                    Xq + (size_t)tids[6] * nblk, Xq + (size_t)tids[7] * nblk
                };
                for (int b = lane8; b < nblk; b += 8) {
                    uint4 qg = qg_row[b];
                    uint4 qu = qu_row[b];
                    float dg = __half2float(dg_row[b]);
                    float du = __half2float(du_row[b]);
                    int32_t g0 = (int32_t)(qg.x & 0x0F0F0F0Fu), g1 = (int32_t)(qg.y & 0x0F0F0F0Fu);
                    int32_t g2 = (int32_t)(qg.z & 0x0F0F0F0Fu), g3 = (int32_t)(qg.w & 0x0F0F0F0Fu);
                    int32_t g4 = (int32_t)((qg.x >> 4) & 0x0F0F0F0Fu), g5 = (int32_t)((qg.y >> 4) & 0x0F0F0F0Fu);
                    int32_t g6 = (int32_t)((qg.z >> 4) & 0x0F0F0F0Fu), g7 = (int32_t)((qg.w >> 4) & 0x0F0F0F0Fu);
                    int32_t u0 = (int32_t)(qu.x & 0x0F0F0F0Fu), u1 = (int32_t)(qu.y & 0x0F0F0F0Fu);
                    int32_t u2 = (int32_t)(qu.z & 0x0F0F0F0Fu), u3 = (int32_t)(qu.w & 0x0F0F0F0Fu);
                    int32_t u4 = (int32_t)((qu.x >> 4) & 0x0F0F0F0Fu), u5 = (int32_t)((qu.y >> 4) & 0x0F0F0F0Fu);
                    int32_t u6 = (int32_t)((qu.z >> 4) & 0x0F0F0F0Fu), u7 = (int32_t)((qu.w >> 4) & 0x0F0F0F0Fu);
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        const q8_1_blk xb = xps[m][b];
                        int32_t sg = 0, su = 0;
                        sg = __builtin_amdgcn_sudot4(false, g0, true, xb.q[0], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g1, true, xb.q[1], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g2, true, xb.q[2], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g3, true, xb.q[3], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g4, true, xb.q[4], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g5, true, xb.q[5], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g6, true, xb.q[6], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g7, true, xb.q[7], sg, false);
                        ag[m] = fmaf(dg * xb.d, (float)(sg - 8 * xb.s), ag[m]);
                        su = __builtin_amdgcn_sudot4(false, u0, true, xb.q[0], su, false);
                        su = __builtin_amdgcn_sudot4(false, u1, true, xb.q[1], su, false);
                        su = __builtin_amdgcn_sudot4(false, u2, true, xb.q[2], su, false);
                        su = __builtin_amdgcn_sudot4(false, u3, true, xb.q[3], su, false);
                        su = __builtin_amdgcn_sudot4(false, u4, true, xb.q[4], su, false);
                        su = __builtin_amdgcn_sudot4(false, u5, true, xb.q[5], su, false);
                        su = __builtin_amdgcn_sudot4(false, u6, true, xb.q[6], su, false);
                        su = __builtin_amdgcn_sudot4(false, u7, true, xb.q[7], su, false);
                        au[m] = fmaf(du * xb.d, (float)(su - 8 * xb.s), au[m]);
                    }
                }
            } else {
                for (int b = lane8; b < nblk; b += 8) {
                    uint4 qg = qg_row[b];
                    uint4 qu = qu_row[b];
                    float dg = __half2float(dg_row[b]);
                    float du = __half2float(du_row[b]);
                    int32_t g0 = (int32_t)(qg.x & 0x0F0F0F0Fu), g1 = (int32_t)(qg.y & 0x0F0F0F0Fu);
                    int32_t g2 = (int32_t)(qg.z & 0x0F0F0F0Fu), g3 = (int32_t)(qg.w & 0x0F0F0F0Fu);
                    int32_t g4 = (int32_t)((qg.x >> 4) & 0x0F0F0F0Fu), g5 = (int32_t)((qg.y >> 4) & 0x0F0F0F0Fu);
                    int32_t g6 = (int32_t)((qg.z >> 4) & 0x0F0F0F0Fu), g7 = (int32_t)((qg.w >> 4) & 0x0F0F0F0Fu);
                    int32_t u0 = (int32_t)(qu.x & 0x0F0F0F0Fu), u1 = (int32_t)(qu.y & 0x0F0F0F0Fu);
                    int32_t u2 = (int32_t)(qu.z & 0x0F0F0F0Fu), u3 = (int32_t)(qu.w & 0x0F0F0F0Fu);
                    int32_t u4 = (int32_t)((qu.x >> 4) & 0x0F0F0F0Fu), u5 = (int32_t)((qu.y >> 4) & 0x0F0F0F0Fu);
                    int32_t u6 = (int32_t)((qu.z >> 4) & 0x0F0F0F0Fu), u7 = (int32_t)((qu.w >> 4) & 0x0F0F0F0Fu);
                    for (int m = 0; m < nt; ++m) {
                        const q8_1_blk xb = Xq[(size_t)tids[m] * nblk + b];
                        int32_t sg = 0, su = 0;
                        sg = __builtin_amdgcn_sudot4(false, g0, true, xb.q[0], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g1, true, xb.q[1], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g2, true, xb.q[2], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g3, true, xb.q[3], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g4, true, xb.q[4], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g5, true, xb.q[5], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g6, true, xb.q[6], sg, false);
                        sg = __builtin_amdgcn_sudot4(false, g7, true, xb.q[7], sg, false);
                        ag[m] = fmaf(dg * xb.d, (float)(sg - 8 * xb.s), ag[m]);
                        su = __builtin_amdgcn_sudot4(false, u0, true, xb.q[0], su, false);
                        su = __builtin_amdgcn_sudot4(false, u1, true, xb.q[1], su, false);
                        su = __builtin_amdgcn_sudot4(false, u2, true, xb.q[2], su, false);
                        su = __builtin_amdgcn_sudot4(false, u3, true, xb.q[3], su, false);
                        su = __builtin_amdgcn_sudot4(false, u4, true, xb.q[4], su, false);
                        su = __builtin_amdgcn_sudot4(false, u5, true, xb.q[5], su, false);
                        su = __builtin_amdgcn_sudot4(false, u6, true, xb.q[6], su, false);
                        su = __builtin_amdgcn_sudot4(false, u7, true, xb.q[7], su, false);
                        au[m] = fmaf(du * xb.d, (float)(su - 8 * xb.s), au[m]);
                    }
                }
            }
        }
        for (int m = 0; m < nt; ++m) {
            float g = s_warp_sum8(ag[m]) * sc;
            float u = s_warp_sum8(au[m]) * sc;
            if (lane8 == 0) {
                if (row < n_ff) {
                    float sg = 1.0f / (1.0f + expf(-g));
                    float act = beta * tanhf(g / beta) * sg;
                    float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
                    sm_v[m][grp] = act * up;
                } else {
                    sm_v[m][grp] = 0.0f;
                }
            }
        }
        __syncthreads();
        if (wid < nt) {
            size_t qidx = ((size_t)tids[wid] * n_used + sids[wid]) * qblk_per_slot + blockIdx.x;
            store_q8_1_from_warp(sm_v[wid][lane], lane, act_q + qidx, sm_q[wid]);
        }
        __syncthreads();
    }
}

extern "C" int ink_moe_gu_q4_0_sudot4_to_q8(hipStream_t s, const uint8_t* W, const float* scales, const void* Xq, const int* exp_counts, const int* exp_offsets, const int* tok_sorted, const int* slot_sorted, void* act_q, int n_ff, int k, int n_used, float beta, float lb, int B) {
    int gy = (B * n_used < 128) ? (B * n_used) : 128;
    int gz = (B >= 40) ? 4 : ((B >= 20) ? 2 : 1);
    moe_gu_q4_0_sudot4_to_q8_k<<<dim3((n_ff + 31) / 32, gy, gz), dim3(256), 0, s>>>(W, scales, (const q8_1_blk*)Xq, exp_counts, exp_offsets, tok_sorted, slot_sorted, (q8_1_blk*)act_q, n_ff, k, n_used, beta, lb);
    return 0;
}

extern "C" __global__ void kv_conv_qkvr_store_multi_snap_k(float* __restrict__ Q,
                                                           const float* __restrict__ K_raw,
                                                           const float* __restrict__ V_raw,
                                                           float* __restrict__ R,
                                                           const float* __restrict__ k_conv_w,
                                                           const float* __restrict__ v_conv_w,
                                                           const float* __restrict__ k_dyn,
                                                           const float* __restrict__ v_dyn,
                                                           float* __restrict__ k_hist_pool,
                                                           float* __restrict__ v_hist_pool,
                                                           float* __restrict__ k_snap_pool,
                                                           float* __restrict__ v_snap_pool,
                                                           const float* __restrict__ q_clip,
                                                           const float* __restrict__ rel_clip,
                                                           __half* __restrict__ k_cache_pool,
                                                           __half* __restrict__ v_cache_pool,
                                                           const int* __restrict__ d_seq_ids,
                                                           const int* __restrict__ d_positions,
                                                           int K, int is_swa, int n_head, int n_head_kv,
                                                           int dh, int d_rel, int max_ctx,
                                                           float qk_thresh, float rel_thresh,
                                                           float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int u = blockIdx.y;
    int lane = threadIdx.x;
    int seq = d_seq_ids[u * K];
    int pos0 = d_positions[u * K];
    int bt_base = u * K;
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (h < n_head) {
        const float* qch = q_clip + (size_t)h * dh;
        const float* rch = rel_clip + (size_t)h * d_rel;
        for (int t = 0; t < K; ++t) {
            int bt = bt_base + t;
            int pos = pos0 + t;
            float tau = 1.0f;
            if (!is_swa) {
                float ratio = ((float)pos + 1.0f) / log_floor;
                if (ratio < 1.0f) ratio = 1.0f;
                tau = 1.0f + log_alpha * logf(ratio);
            }
            float* qh = Q + (size_t)bt * q_dim + (size_t)h * dh;
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

            float* rh = R + (size_t)bt * r_dim + (size_t)h * d_rel;
            float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
            float rss = s_warp_sum32(rv * rv);
            float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
            if (lane < d_rel) rh[lane] = rv * rfactor;
        }
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float* k_hist = k_hist_pool + (size_t)seq * kv_dim * 3;
        float* v_hist = v_hist_pool + (size_t)seq * kv_dim * 3;
        float* k_snap = k_snap_pool ? (k_snap_pool + (size_t)u * K * kv_dim * 3) : nullptr;
        float* v_snap = v_snap_pool ? (v_snap_pool + (size_t)u * K * kv_dim * 3) : nullptr;
        size_t seq_kv_stride = (size_t)n_head_kv * max_ctx * dh;
        __half* kc_base = k_cache_pool + (size_t)seq * seq_kv_stride;
        __half* vc_base = v_cache_pool + (size_t)seq * seq_kv_stride;
        float kh0[4], kh1[4], kh2[4], vh0[4], vh1[4], vh2[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            kh0[p] = k_hist[c * 3 + 0]; kh1[p] = k_hist[c * 3 + 1]; kh2[p] = k_hist[c * 3 + 2];
            vh0[p] = v_hist[c * 3 + 0]; vh1[p] = v_hist[c * 3 + 1]; vh2[p] = v_hist[c * 3 + 2];
        }
        for (int t = 0; t < K; ++t) {
            int bt = bt_base + t;
            int pos = pos0 + t;
            int slot = is_swa ? (pos % max_ctx) : pos;
            float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
            float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
            if (k_dyn) {
                const float* dk = k_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
                const float* dv = v_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
                kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
                vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
            }
            float kv_v[4], vv_v[4];
            float kss = 0.0f, vss = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int c = kh_idx * dh + lane + p * 32;
                float kx = K_raw[(size_t)bt * kv_dim + c];
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

                float vx = V_raw[(size_t)bt * kv_dim + c];
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
            __half* kc_slot = kc_base + ((size_t)kh_idx * max_ctx + slot) * dh;
            __half* vc_slot = vc_base + ((size_t)kh_idx * max_ctx + slot) * dh;
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

extern "C" __global__ void shortconv_res_multi_snap_k(const float* __restrict__ raw,
                                                      const float* __restrict__ conv_w,
                                                      float* __restrict__ hist_pool,
                                                      float* __restrict__ snap_pool,
                                                      const float* __restrict__ add_src,
                                                      float* __restrict__ block_dst,
                                                      float* __restrict__ out,
                                                      const int* __restrict__ d_seq_ids,
                                                      int n_embd, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;
    if (c >= n_embd) return;
    int seq = d_seq_ids[u * K];
    int bt_base = u * K;
    float* hc = hist_pool + ((size_t)seq * n_embd + c) * 3;
    float* sn_base = snap_pool ? (snap_pool + (size_t)u * K * n_embd * 3) : nullptr;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = conv_w + (size_t)c * 4;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    for (int t = 0; t < K; ++t) {
        size_t idx = (size_t)(bt_base + t) * n_embd + c;
        float x = raw[idx];
        float conv_out = (x + w0 * h0 + w1 * h1 + w2 * h2 + w3 * x) * rsqrtf(5.0f);
        h0 = h1; h1 = h2; h2 = x;
        if (sn_base) {
            float* sn = sn_base + ((size_t)t * n_embd + c) * 3;
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

extern "C" int ink_kv_conv_qkvr_store_multi_snap(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist_pool, float* v_hist_pool, float* k_snap_pool, float* v_snap_pool, const float* q_clip, const float* rel_clip, __half* kc_pool, __half* vc_pool, const int* d_seq_ids, const int* d_positions, int n_users, int K, int is_swa, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_multi_snap_k<<<dim3(n_head + n_head_kv, n_users), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist_pool, v_hist_pool, k_snap_pool, v_snap_pool, q_clip, rel_clip, kc_pool, vc_pool, d_seq_ids, d_positions, K, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}

extern "C" __global__ void shortconv_res_multi_snap_clear_k(float* __restrict__ raw,
                                                            const float* __restrict__ conv_w,
                                                            float* __restrict__ hist_pool,
                                                            float* __restrict__ snap_pool,
                                                            const float* __restrict__ add_src,
                                                            float* __restrict__ block_dst,
                                                            float* __restrict__ out,
                                                            const int* __restrict__ d_seq_ids,
                                                            int n_embd, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;
    if (c >= n_embd) return;
    int seq = d_seq_ids[u * K];
    int bt_base = u * K;
    float* hc = hist_pool + ((size_t)seq * n_embd + c) * 3;
    float* sn_base = snap_pool ? (snap_pool + (size_t)u * K * n_embd * 3) : nullptr;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float* wc = conv_w + (size_t)c * 4;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    for (int t = 0; t < K; ++t) {
        size_t idx = (size_t)(bt_base + t) * n_embd + c;
        float x = raw[idx];
        raw[idx] = 0.0f;
        float conv_out = (x + w0 * h0 + w1 * h1 + w2 * h2 + w3 * x) * rsqrtf(5.0f);
        h0 = h1; h1 = h2; h2 = x;
        if (sn_base) {
            float* sn = sn_base + ((size_t)t * n_embd + c) * 3;
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

extern "C" __global__ void rewind_multi_all_slots_k(const uint64_t* __restrict__ slot_ptrs,
                                                    const int* __restrict__ d_rw_u,
                                                    const int* __restrict__ d_rw_seq,
                                                    const int* __restrict__ d_rw_idx,
                                                    int kv_n, int e_n, int k_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sl = blockIdx.y;
    int r = blockIdx.z;
    int u = d_rw_u[r];
    int su = d_rw_seq[r];
    int idx = d_rw_idx[r];
    const uint64_t* sp = slot_ptrs + (size_t)sl * 8;
    if (i < kv_n) {
        size_t s_off = (size_t)(u * k_stride + idx) * kv_n + i;
        size_t d_off = (size_t)su * kv_n + i;
        reinterpret_cast<float*>(sp[1])[d_off] = reinterpret_cast<const float*>(sp[0])[s_off];
        reinterpret_cast<float*>(sp[3])[d_off] = reinterpret_cast<const float*>(sp[2])[s_off];
    }
    if (i < e_n) {
        size_t s_off = (size_t)(u * k_stride + idx) * e_n + i;
        size_t d_off = (size_t)su * e_n + i;
        reinterpret_cast<float*>(sp[5])[d_off] = reinterpret_cast<const float*>(sp[4])[s_off];
        reinterpret_cast<float*>(sp[7])[d_off] = reinterpret_cast<const float*>(sp[6])[s_off];
    }
}

extern "C" int ink_shortconv_res_multi_snap(hipStream_t s, const float* raw, const float* conv_w, float* hist_pool, float* snap_pool, const float* add_src, float* block_dst, float* out, const int* d_seq_ids, int n_embd, int n_users, int K) {
    shortconv_res_multi_snap_k<<<dim3((n_embd + 255) / 256, n_users), dim3(256), 0, s>>>(raw, conv_w, hist_pool, snap_pool, add_src, block_dst, out, d_seq_ids, n_embd, K);
    return 0;
}

extern "C" int ink_shortconv_res_multi_snap_clear(hipStream_t s, float* raw, const float* conv_w, float* hist_pool, float* snap_pool, const float* add_src, float* block_dst, float* out, const int* d_seq_ids, int n_embd, int n_users, int K) {
    shortconv_res_multi_snap_clear_k<<<dim3((n_embd + 255) / 256, n_users), dim3(256), 0, s>>>(raw, conv_w, hist_pool, snap_pool, add_src, block_dst, out, d_seq_ids, n_embd, K);
    return 0;
}

extern "C" int ink_rewind_multi_all_slots(hipStream_t s, const uint64_t* slot_ptrs, const int* d_rw_u, const int* d_rw_seq, const int* d_rw_idx, int n_slots, int n_rw, int kv_n, int e_n, int k_stride) {
    int max_n = (kv_n > e_n) ? kv_n : e_n;
    rewind_multi_all_slots_k<<<dim3((max_n + 255) / 256, n_slots, n_rw), dim3(256), 0, s>>>(slot_ptrs, d_rw_u, d_rw_seq, d_rw_idx, kv_n, e_n, k_stride);
    return 0;
}

extern "C" __global__ void rewind_single_all_slots_k(const uint64_t* __restrict__ slot_ptrs,
                                                     int su, int idx,
                                                     int kv_n, int e_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sl = blockIdx.y;
    const uint64_t* sp = slot_ptrs + (size_t)sl * 8;
    if (i < kv_n) {
        size_t s_off = (size_t)idx * kv_n + i;
        size_t d_off = (size_t)su * kv_n + i;
        reinterpret_cast<float*>(sp[1])[d_off] = reinterpret_cast<const float*>(sp[0])[s_off];
        reinterpret_cast<float*>(sp[3])[d_off] = reinterpret_cast<const float*>(sp[2])[s_off];
    }
    if (i < e_n) {
        size_t s_off = (size_t)idx * e_n + i;
        size_t d_off = (size_t)su * e_n + i;
        reinterpret_cast<float*>(sp[5])[d_off] = reinterpret_cast<const float*>(sp[4])[s_off];
        reinterpret_cast<float*>(sp[7])[d_off] = reinterpret_cast<const float*>(sp[6])[s_off];
    }
}

extern "C" int ink_rewind_single_all_slots(hipStream_t s, const uint64_t* slot_ptrs, int su, int idx, int n_slots, int kv_n, int e_n) {
    int max_n = (kv_n > e_n) ? kv_n : e_n;
    rewind_single_all_slots_k<<<dim3((max_n + 255) / 256, n_slots), dim3(256), 0, s>>>(slot_ptrs, su, idx, kv_n, e_n);
    return 0;
}

extern "C" __global__ void inkling_attn_batch_gated_q8_k(const float* __restrict__ Q,
                                                         const float* __restrict__ R,
                                                         const float* __restrict__ sinks,
                                                         const float* __restrict__ rel_bias,
                                                         const __half* __restrict__ k_cache,
                                                         const __half* __restrict__ v_cache,
                                                         const float* __restrict__ gate,
                                                         const int* __restrict__ d_pos0,
                                                         q8_1_blk* __restrict__ out_q8,
                                                         int n_head, int n_head_kv,
                                                         int dh, int d_rel,
                                                         int rel_extent, int swa_window,
                                                         int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int pos = (*d_pos0) + bt;
    int kv_h = h / (n_head / n_head_kv);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    __shared__ float qs[128];
    __shared__ float rs[16];
    float q_scale = rsqrtf((float)dh);
    for (int d = threadIdx.x; d < dh; d += blockDim.x) {
        qs[d] = Q[(size_t)bt * q_dim + (size_t)h * dh + d] * q_scale;
    }
    if (threadIdx.x < d_rel) {
        rs[threadIdx.x] = R[(size_t)bt * r_dim + (size_t)h * d_rel + threadIdx.x];
    }
    __syncthreads();

    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[(size_t)bt * n_head + h];

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int p0 = lane * 2;
    int p1 = lane * 2 + 64;
    float q0a = qs[p0], q0b = qs[p0 + 1], q1a = qs[p1], q1b = qs[p1 + 1];

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half2* kt2 = reinterpret_cast<const __half2*>(kc_h + (size_t)kv_slot * dh);
        float2 k0 = __half22float2(kt2[lane]);
        float2 k1 = __half22float2(kt2[lane + 32]);
        float dot = q0a * k0.x + q0b * k0.y + q1a * k1.x + q1b * k1.y;
        dot = s_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = s_warp_sum32(rp);
            rbias = __shfl(rp, 0, 32);
        }
        score += rbias - sink_h;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        const __half2* vt2 = reinterpret_cast<const __half2*>(vc_h + (size_t)kv_slot * dh);
        float2 v0 = __half22float2(vt2[lane]);
        float2 v1 = __half22float2(vt2[lane + 32]);
        o0 = fmaf(p, v0.x, o0 * alpha);
        o1 = fmaf(p, v0.y, o1 * alpha);
        o2 = fmaf(p, v1.x, o2 * alpha);
        o3 = fmaf(p, v1.y, o3 * alpha);
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    sm_o[wid][p0] = o0; sm_o[wid][p0 + 1] = o1; sm_o[wid][p1] = o2; sm_o[wid][p1 + 1] = o3;
    __syncthreads();

    if (wid == 0) {
        float gm = sm_m[0];
        for (int w = 1; w < nw; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
        int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;
        for (int w = 0; w < nw; ++w) {
            float sc = expf(sm_m[w] - gm);
            gl += sm_l[w] * sc;
            go0 += sm_o[w][d0] * sc;
            go1 += sm_o[w][d1] * sc;
            go2 += sm_o[w][d2] * sc;
            go3 += sm_o[w][d3] * sc;
        }
        float inv_l = 1.0f / gl;
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        float v0 = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        float v1 = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        float v2 = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        float v3 = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
        __shared__ int8_t smq[32];
        size_t qbase = ((size_t)bt * n_head + h) * 4;
        store_q8_1_from_warp(v0, lane, out_q8 + qbase + 0, smq);
        store_q8_1_from_warp(v1, lane, out_q8 + qbase + 1, smq);
        store_q8_1_from_warp(v2, lane, out_q8 + qbase + 2, smq);
        store_q8_1_from_warp(v3, lane, out_q8 + qbase + 3, smq);
    }
}

extern "C" int ink_attn_b8_flash_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B);
extern "C" int ink_attn_batch_gated_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    if (B >= 4 && B <= 64 && dh == 128 && d_rel == 16) {
        return ink_attn_b8_flash_q8(s, Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B);
    }
    inkling_attn_batch_gated_q8_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, (q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

template <int NT>
__global__ void dflash2_gu_swiglu_q8_nt_k(const uint8_t* __restrict__ wg,
                                          const uint8_t* __restrict__ wu,
                                          const q8_1_blk* __restrict__ Xq,
                                          q8_1_blk* __restrict__ out_q8,
                                          int n_ff, int k) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 32 + grp;
    int nblk = k >> 5;
    int qblk_out = n_ff >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    int r_safe = min(row, n_ff - 1);
    const uint8_t* wgr = wg + (size_t)r_safe * rbytes;
    const uint8_t* wur = wu + (size_t)r_safe * rbytes;
    const uint4* qgr = reinterpret_cast<const uint4*>(wgr);
    const __half* dgr = reinterpret_cast<const __half*>(wgr + (size_t)nblk * 16);
    const uint4* qur = reinterpret_cast<const uint4*>(wur);
    const __half* dur = reinterpret_cast<const __half*>(wur + (size_t)nblk * 16);

    __shared__ q8_1_blk sm_x[NT][8];
    float ag[NT];
    float au[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) { ag[m] = 0.0f; au[m] = 0.0f; }
    for (int b0 = 0; b0 < nblk; b0 += 8) {
        if (threadIdx.x < NT * 8) {
            int tm = threadIdx.x >> 3;
            int tb = threadIdx.x & 7;
            if (b0 + tb < nblk) sm_x[tm][tb] = Xq[(size_t)tm * nblk + b0 + tb];
        }
        __syncthreads();
        int b = b0 + lane8;
        if (row < n_ff && b < nblk) {
            uint4 qg = qgr[b];
            uint4 qu = qur[b];
            float dg = __half2float(dgr[b]);
            float du = __half2float(dur[b]);
            int32_t g0 = (int32_t)(qg.x & 0x0F0F0F0Fu), g1 = (int32_t)(qg.y & 0x0F0F0F0Fu);
            int32_t g2 = (int32_t)(qg.z & 0x0F0F0F0Fu), g3 = (int32_t)(qg.w & 0x0F0F0F0Fu);
            int32_t g4 = (int32_t)((qg.x >> 4) & 0x0F0F0F0Fu), g5 = (int32_t)((qg.y >> 4) & 0x0F0F0F0Fu);
            int32_t g6 = (int32_t)((qg.z >> 4) & 0x0F0F0F0Fu), g7 = (int32_t)((qg.w >> 4) & 0x0F0F0F0Fu);
            int32_t u0 = (int32_t)(qu.x & 0x0F0F0F0Fu), u1 = (int32_t)(qu.y & 0x0F0F0F0Fu);
            int32_t u2 = (int32_t)(qu.z & 0x0F0F0F0Fu), u3 = (int32_t)(qu.w & 0x0F0F0F0Fu);
            int32_t u4 = (int32_t)((qu.x >> 4) & 0x0F0F0F0Fu), u5 = (int32_t)((qu.y >> 4) & 0x0F0F0F0Fu);
            int32_t u6 = (int32_t)((qu.z >> 4) & 0x0F0F0F0Fu), u7 = (int32_t)((qu.w >> 4) & 0x0F0F0F0Fu);
#pragma unroll
            for (int m = 0; m < NT; ++m) {
                const q8_1_blk xb = sm_x[m][lane8];
                int32_t sg = 0, su = 0;
                sg = __builtin_amdgcn_sudot4(false, g0, true, xb.q[0], sg, false);
                sg = __builtin_amdgcn_sudot4(false, g1, true, xb.q[1], sg, false);
                sg = __builtin_amdgcn_sudot4(false, g2, true, xb.q[2], sg, false);
                sg = __builtin_amdgcn_sudot4(false, g3, true, xb.q[3], sg, false);
                sg = __builtin_amdgcn_sudot4(false, g4, true, xb.q[4], sg, false);
                sg = __builtin_amdgcn_sudot4(false, g5, true, xb.q[5], sg, false);
                sg = __builtin_amdgcn_sudot4(false, g6, true, xb.q[6], sg, false);
                sg = __builtin_amdgcn_sudot4(false, g7, true, xb.q[7], sg, false);
                ag[m] = fmaf(dg * xb.d, (float)(sg - 8 * xb.s), ag[m]);
                su = __builtin_amdgcn_sudot4(false, u0, true, xb.q[0], su, false);
                su = __builtin_amdgcn_sudot4(false, u1, true, xb.q[1], su, false);
                su = __builtin_amdgcn_sudot4(false, u2, true, xb.q[2], su, false);
                su = __builtin_amdgcn_sudot4(false, u3, true, xb.q[3], su, false);
                su = __builtin_amdgcn_sudot4(false, u4, true, xb.q[4], su, false);
                su = __builtin_amdgcn_sudot4(false, u5, true, xb.q[5], su, false);
                su = __builtin_amdgcn_sudot4(false, u6, true, xb.q[6], su, false);
                su = __builtin_amdgcn_sudot4(false, u7, true, xb.q[7], su, false);
                au[m] = fmaf(du * xb.d, (float)(su - 8 * xb.s), au[m]);
            }
        }
        __syncthreads();
    }
    __shared__ float sm_v[NT][32];
    __shared__ int8_t sm_q[NT][32];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        float g = s_warp_sum8(ag[m]);
        float u = s_warp_sum8(au[m]);
        if (lane8 == 0) {
            sm_v[m][grp] = (row < n_ff) ? ((g / (1.0f + exp2f(-g * S_LOG2E))) * u) : 0.0f;
        }
    }
    __syncthreads();
    if (wid < NT) {
        size_t qidx = (size_t)wid * qblk_out + blockIdx.x;
        store_q8_1_from_warp(sm_v[wid][lane], lane, out_q8 + qidx, sm_q[wid]);
    }
}

extern "C" int ink_dflash2_gu_swiglu_q8(hipStream_t s, const uint8_t* wg, const uint8_t* wu, const void* Xq, void* out_q8, int n_ff, int k, int B) {
    if (B == 8) {
        dflash2_gu_swiglu_q8_nt_k<8><<<dim3((n_ff + 31) / 32), dim3(256), 0, s>>>(wg, wu, (const q8_1_blk*)Xq, (q8_1_blk*)out_q8, n_ff, k);
        return 0;
    }
    if (B == 6) {
        dflash2_gu_swiglu_q8_nt_k<6><<<dim3((n_ff + 31) / 32), dim3(256), 0, s>>>(wg, wu, (const q8_1_blk*)Xq, (q8_1_blk*)out_q8, n_ff, k);
        return 0;
    }
    if (B == 4) {
        dflash2_gu_swiglu_q8_nt_k<4><<<dim3((n_ff + 31) / 32), dim3(256), 0, s>>>(wg, wu, (const q8_1_blk*)Xq, (q8_1_blk*)out_q8, n_ff, k);
        return 0;
    }
    return 1;
}

template <int NT>
__global__ void dflash2_proj_conv_add_q4_nt_k(const uint8_t* __restrict__ W,
                                              const q8_1_blk* __restrict__ Xq,
                                              const float* __restrict__ dynamic,
                                              const float* __restrict__ base_kernel,
                                              float* __restrict__ x_inout,
                                              int n_embd, int k,
                                              int group_size, int side) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 32 + grp;
    int nblk = k >> 5;
    size_t rbytes = q4_soa_row_bytes(nblk);
    int r_safe = min(row, n_embd - 1);
    const uint8_t* wrow = W + (size_t)r_safe * rbytes;
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
        if (row < n_embd && b < nblk) {
            uint4 q4 = qrow[b];
            float dw = __half2float(drow[b]);
            int32_t w0 = (int32_t)(q4.x & 0x0F0F0F0Fu), w1 = (int32_t)(q4.y & 0x0F0F0F0Fu);
            int32_t w2 = (int32_t)(q4.z & 0x0F0F0F0Fu), w3 = (int32_t)(q4.w & 0x0F0F0F0Fu);
            int32_t w4 = (int32_t)((q4.x >> 4) & 0x0F0F0F0Fu), w5 = (int32_t)((q4.y >> 4) & 0x0F0F0F0Fu);
            int32_t w6 = (int32_t)((q4.z >> 4) & 0x0F0F0F0Fu), w7 = (int32_t)((q4.w >> 4) & 0x0F0F0F0Fu);
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

    float vals[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) vals[m] = s_warp_sum8(acc[m]);
    if (lane8 == 0 && row < n_embd) {
        int n_groups = n_embd / group_size;
        int g = row / group_size;
        const float* base_s = base_kernel + (size_t)(side * 2 * n_embd);
        float b0 = base_s[row];
        float b1 = base_s[n_embd + row];
        float prev = 0.0f;
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            const float* dyn_t = dynamic + (size_t)m * (size_t)(4 * n_groups) + (size_t)(side * 2 * n_groups);
            float w0 = b0 + dyn_t[g];
            float w1 = b1 + dyn_t[n_groups + g];
            float v = vals[m];
            x_inout[(size_t)m * n_embd + row] += w0 * v + w1 * prev;
            prev = v;
        }
    }
}

extern "C" int ink_dflash2_proj_conv_add_q4(hipStream_t s, const uint8_t* W, const void* Xq, const float* dynamic, const float* base_kernel, float* x_inout, int n_embd, int k, int group_size, int side, int B) {
    if (B == 8) {
        dflash2_proj_conv_add_q4_nt_k<8><<<dim3((n_embd + 31) / 32), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, dynamic, base_kernel, x_inout, n_embd, k, group_size, side);
        return 0;
    }
    if (B == 6) {
        dflash2_proj_conv_add_q4_nt_k<6><<<dim3((n_embd + 31) / 32), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, dynamic, base_kernel, x_inout, n_embd, k, group_size, side);
        return 0;
    }
    if (B == 4) {
        dflash2_proj_conv_add_q4_nt_k<4><<<dim3((n_embd + 31) / 32), dim3(256), 0, s>>>(W, (const q8_1_blk*)Xq, dynamic, base_kernel, x_inout, n_embd, k, group_size, side);
        return 0;
    }
    return 1;
}
