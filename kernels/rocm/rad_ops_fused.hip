#include "rad_ops_common.cuh"

extern "C" int ink_mix_batch(hipStream_t s, const float* blocks, int n_blocks, int max_B, const float* cur, const float* query, const float* w0, const float* w1, const float* w2, float* out, int n_embd, int n_queries, int B, float eps);

extern "C" __global__ void qkvrg_gemv_k(const uint8_t* __restrict__ wq,
                                        const uint8_t* __restrict__ wk,
                                        const uint8_t* __restrict__ wv,
                                        const uint8_t* __restrict__ wrel,
                                        const uint8_t* __restrict__ wgate,
                                        const uint16_t* __restrict__ wsink,
                                        const uint16_t* __restrict__ wkdyn,
                                        const uint16_t* __restrict__ wvdyn,
                                        const float* __restrict__ inp0,
                                        const float* __restrict__ inp1,
                                        const float* __restrict__ inp2,
                                        float* __restrict__ q_out,
                                        float* __restrict__ k_out,
                                        float* __restrict__ v_out,
                                        float* __restrict__ r_out,
                                        float* __restrict__ gate_out,
                                        float* __restrict__ sink_out,
                                        float* __restrict__ kdyn_out,
                                        float* __restrict__ vdyn_out,
                                        int k_dim, int q_dim, int kv_dim,
                                        int r_dim, int n_head, int dyn_dim) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g_row = blockIdx.x * 8 + warp;
    int nblk = k_dim >> 5;
    size_t rb = (size_t)nblk * 34;

    if (g_row < q_dim) {
        float v = dot_q8_0_row32(wq + (size_t)g_row * rb, inp0, nblk, lane);
        if (lane == 0) q_out[g_row] = v;
        return;
    }
    g_row -= q_dim;
    if (g_row < kv_dim) {
        float v = dot_q8_0_row32(wk + (size_t)g_row * rb, inp1, nblk, lane);
        if (lane == 0) k_out[g_row] = v;
        return;
    }
    g_row -= kv_dim;
    if (g_row < kv_dim) {
        float v = dot_q8_0_row32(wv + (size_t)g_row * rb, inp2, nblk, lane);
        if (lane == 0) v_out[g_row] = v;
        return;
    }
    g_row -= kv_dim;
    if (g_row < r_dim) {
        float v = dot_q8_0_row32(wrel + (size_t)g_row * rb, inp0, nblk, lane);
        if (lane == 0) r_out[g_row] = v;
        return;
    }
    g_row -= r_dim;
    if (g_row < q_dim) {
        float v = dot_q8_0_row32(wgate + (size_t)g_row * rb, inp0, nblk, lane);
        if (lane == 0) gate_out[g_row] = v;
        return;
    }
    g_row -= q_dim;
    if (g_row < n_head) {
        const uint16_t* wr = wsink + (size_t)g_row * k_dim;
        float acc = 0.0f;
        for (int j = lane; j < k_dim; j += 32) acc = fmaf(bf16_to_fp32(wr[j]), inp0[j], acc);
        acc = warp_sum32(acc);
        if (lane == 0) sink_out[g_row] = acc;
        return;
    }
    g_row -= n_head;
    if (dyn_dim > 0) {
        if (g_row < dyn_dim) {
            const uint16_t* wr = wkdyn + (size_t)g_row * k_dim;
            float acc = 0.0f;
            for (int j = lane; j < k_dim; j += 32) acc = fmaf(bf16_to_fp32(wr[j]), inp1[j], acc);
            acc = warp_sum32(acc);
            if (lane == 0) kdyn_out[g_row] = acc;
            return;
        }
        g_row -= dyn_dim;
        if (g_row < dyn_dim) {
            const uint16_t* wr = wvdyn + (size_t)g_row * k_dim;
            float acc = 0.0f;
            for (int j = lane; j < k_dim; j += 32) acc = fmaf(bf16_to_fp32(wr[j]), inp2[j], acc);
            acc = warp_sum32(acc);
            if (lane == 0) vdyn_out[g_row] = acc;
            return;
        }
    }
}

extern "C" __global__ void kv_conv_qkvr_store_k(float* __restrict__ Q,
                                                const float* __restrict__ K_raw,
                                                const float* __restrict__ V_raw,
                                                float* __restrict__ R,
                                                const float* __restrict__ k_conv_w,
                                                const float* __restrict__ v_conv_w,
                                                const float* __restrict__ k_dyn,
                                                const float* __restrict__ v_dyn,
                                                float* __restrict__ k_hist,
                                                float* __restrict__ v_hist,
                                                const float* __restrict__ q_clip,
                                                const float* __restrict__ rel_clip,
                                                __half* __restrict__ k_cache,
                                                __half* __restrict__ v_cache,
                                                const int* __restrict__ d_pos,
                                                int is_swa, int n_head, int n_head_kv,
                                                int dh, int d_rel, int max_ctx,
                                                float qk_thresh, float rel_thresh,
                                                float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos = *d_pos;
    float tau = 1.0f;
    if (!is_swa) {
        float ratio = ((float)pos + 1.0f) / log_floor;
        if (ratio < 1.0f) ratio = 1.0f;
        tau = 1.0f + log_alpha * logf(ratio);
    }

    if (h < n_head) {
        float* qh = Q + (size_t)h * dh;
        const float* qch = q_clip + (size_t)h * dh;
        float ss = 0.0f;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        ss = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
        ss = warp_sum32(ss);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;

        float* rh = R + (size_t)h * d_rel;
        const float* rch = rel_clip + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = warp_sum32(rv * rv);
        float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
        if (lane < d_rel) rh[lane] = rv * rfactor;
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        int slot = is_swa ? (pos % max_ctx) : pos;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float kd0 = 0.0f, kd1 = 0.0f, kd2 = 0.0f, kd3 = 0.0f;
        float vd0 = 0.0f, vd1 = 0.0f, vd2 = 0.0f, vd3 = 0.0f;
        if (k_dyn) {
            const float* dk = k_dyn + (size_t)kh_idx * 4;
            const float* dv = v_dyn + (size_t)kh_idx * 4;
            kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
            vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
        }
        float kv_v[4];
        float vv_v[4];
        float kss = 0.0f, vss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int d = lane + p * 32;
            int c = kh_idx * dh + d;
            float* khc = k_hist + (size_t)c * 3;
            float kh0 = khc[0], kh1 = khc[1], kh2 = khc[2];
            float kx = K_raw[c];
            const float* kwc = k_conv_w + (size_t)c * 4;
            float kacc = kwc[0] * kh0 + kwc[1] * kh1 + kwc[2] * kh2 + kwc[3] * kx;
            if (k_dyn) kacc += kd0 * kx + kd1 * kh2 + kd2 * kh1 + kd3 * kh0;
            khc[0] = kh1; khc[1] = kh2; khc[2] = kx;
            float kout = (kx + kacc) * sc;
            kv_v[p] = kout;
            kss += kout * kout;

            float* vhc = v_hist + (size_t)c * 3;
            float vh0 = vhc[0], vh1 = vhc[1], vh2 = vhc[2];
            float vx = V_raw[c];
            const float* vwc = v_conv_w + (size_t)c * 4;
            float vacc = vwc[0] * vh0 + vwc[1] * vh1 + vwc[2] * vh2 + vwc[3] * vx;
            if (v_dyn) vacc += vd0 * vx + vd1 * vh2 + vd2 * vh1 + vd3 * vh0;
            vhc[0] = vh1; vhc[1] = vh2; vhc[2] = vx;
            float vout = (vx + vacc) * sc;
            vv_v[p] = vout;
            vss += vout * vout;
        }
        kss = warp_sum32(kss);
        vss = warp_sum32(vss);
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

extern "C" __global__ void inkling_attn_decode_gated_k(const float* __restrict__ Q,
                                                       const float* __restrict__ R,
                                                       const float* __restrict__ sinks,
                                                       const float* __restrict__ rel_bias,
                                                       const __half* __restrict__ k_cache,
                                                       const __half* __restrict__ v_cache,
                                                       const float* __restrict__ gate,
                                                       const int* __restrict__ d_pos,
                                                       float* __restrict__ out,
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
    int d0 = lane;
    int d1 = lane + 32;
    int d2 = lane + 64;
    int d3 = lane + 96;

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half* kt = kc_h + (size_t)kv_slot * dh;
        float dot = qs[d0] * __half2float(kt[d0])
                  + qs[d1] * __half2float(kt[d1])
                  + qs[d2] * __half2float(kt[d2])
                  + qs[d3] * __half2float(kt[d3]);
        dot = warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = warp_sum32(rp);
            rbias = __shfl(rp, 0, 32);
        }
        score += rbias - sink_h;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        const __half* vt = vc_h + (size_t)kv_slot * dh;
        o0 = o0 * alpha + p * __half2float(vt[d0]);
        o1 = o1 * alpha + p * __half2float(vt[d1]);
        o2 = o2 * alpha + p * __half2float(vt[d2]);
        o3 = o3 * alpha + p * __half2float(vt[d3]);
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    if (lane == 0) {
        sm_m[wid] = m;
        sm_l[wid] = l;
    }
    sm_o[wid][d0] = o0;
    sm_o[wid][d1] = o1;
    sm_o[wid][d2] = o2;
    sm_o[wid][d3] = o3;
    __syncthreads();

    if (wid == 0) {
        float gm = sm_m[0];
        for (int w = 1; w < nw; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
        for (int w = 0; w < nw; ++w) {
            float sc = expf(sm_m[w] - gm);
            gl += sm_l[w] * sc;
            go0 += sm_o[w][d0] * sc;
            go1 += sm_o[w][d1] * sc;
            go2 += sm_o[w][d2] * sc;
            go3 += sm_o[w][d3] * sc;
        }
        float inv_l = 1.0f / gl;
        float* oh = out + (size_t)h * dh;
        const float* gh = gate + (size_t)h * dh;
        oh[d0] = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        oh[d1] = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        oh[d2] = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        oh[d3] = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
    }
}

extern "C" __global__ void wo_conv_res_k(const uint8_t* __restrict__ wo,
                                         const float* __restrict__ x,
                                         const float* __restrict__ conv_w,
                                         float* __restrict__ hist,
                                         const float* __restrict__ add_src,
                                         float* __restrict__ block_dst,
                                         float* __restrict__ out,
                                         int n_embd, int q_dim) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n_embd) return;
    int nblk = q_dim >> 5;
    float o = dot_q8_0_row32(wo + (size_t)row * ((size_t)nblk * 34), x, nblk, lane);
    if (lane == 0) {
        float* hc = hist + (size_t)row * 3;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float* wc = conv_w + (size_t)row * 4;
        float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * o;
        hc[0] = h1; hc[1] = h2; hc[2] = o;
        float conv_out = (o + acc) * rsqrtf(5.0f);
        if (block_dst) {
            block_dst[row] = add_src[row];
            out[row] = conv_out;
        } else {
            out[row] = add_src[row] + conv_out;
        }
    }
}

extern "C" __global__ void moe_router_gemv_k(const float* __restrict__ w_router,
                                             const float* __restrict__ x,
                                             const float* __restrict__ bias,
                                             float* __restrict__ wtop,
                                             int* __restrict__ itop,
                                             int n_expert, int k_dim, int n_used,
                                             float route_scale) {
    __shared__ float probs[256];
    __shared__ float sel[256];
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    for (int e = wid; e < n_expert; e += nw) {
        const float* wr = w_router + (size_t)e * k_dim;
        float acc = 0.0f;
        for (int j = lane; j < k_dim; j += 32) {
            acc = fmaf(wr[j], x[j], acc);
        }
        acc = warp_sum32(acc);
        if (lane == 0) {
            float p = 1.0f / (1.0f + expf(-acc));
            probs[e] = p;
            sel[e] = bias ? (p + bias[e]) : p;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float wsum = 0.0f;
        for (int t = 0; t < n_used; ++t) {
            float best = -1e30f;
            int bi = 0;
            for (int i = 0; i < n_expert; ++i) {
                if (sel[i] > best) {
                    best = sel[i];
                    bi = i;
                }
            }
            sel[bi] = -1e30f;
            itop[t] = bi;
            wtop[t] = probs[bi];
            wsum += probs[bi];
        }
        float inv = route_scale / (wsum + 1e-20f);
        for (int t = 0; t < n_used; ++t) {
            wtop[t] *= inv;
        }
    }
}

extern "C" __global__ void moe_down_conv_add_nvfp4_k(const uint8_t* __restrict__ W,
                                                     const float* __restrict__ scales,
                                                     const float* __restrict__ act_in,
                                                     const int* __restrict__ itop,
                                                     const float* __restrict__ wtop,
                                                     const float* __restrict__ conv_w,
                                                     float* __restrict__ hist,
                                                     const float* __restrict__ ffn_inp,
                                                     float* __restrict__ cur_out,
                                                     int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    if (row >= n_embd) return;
    int nblk = n_ff >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[slot];
        float w = wtop[slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = dot_nvfp4_row8(wrow, act_in + (size_t)slot * n_ff, nblk, lane8);
        total = fmaf(v, w, total);
    }
    if (lane8 == 0) {
        float* hc = hist + (size_t)row * 3;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float* wc = conv_w + (size_t)row * 4;
        float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * total;
        hc[0] = h1; hc[1] = h2; hc[2] = total;
        cur_out[row] = ffn_inp[row] + (total + acc) * rsqrtf(5.0f);
    }
}

extern "C" __global__ void moe_down_conv_add_q8_0_k(const uint8_t* __restrict__ W,
                                                    const float* __restrict__ scales,
                                                    const float* __restrict__ act_in,
                                                    const int* __restrict__ itop,
                                                    const float* __restrict__ wtop,
                                                    const float* __restrict__ conv_w,
                                                    float* __restrict__ hist,
                                                    const float* __restrict__ ffn_inp,
                                                    float* __restrict__ cur_out,
                                                    int n_embd, int n_ff, int n_used) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n_embd) return;
    int nblk = n_ff >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[slot];
        float w = wtop[slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = dot_q8_0_row32(wrow, act_in + (size_t)slot * n_ff, nblk, lane);
        total = fmaf(v, w, total);
    }
    if (lane == 0) {
        float* hc = hist + (size_t)row * 3;
        float h0 = hc[0], h1 = hc[1], h2 = hc[2];
        const float* wc = conv_w + (size_t)row * 4;
        float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * total;
        hc[0] = h1; hc[1] = h2; hc[2] = total;
        cur_out[row] = ffn_inp[row] + (total + acc) * rsqrtf(5.0f);
    }
}

extern "C" int ink_qkvrg_gemv(hipStream_t s,
                              const uint8_t* wq, const uint8_t* wk, const uint8_t* wv,
                              const uint8_t* wrel, const uint8_t* wgate, const uint16_t* wsink,
                              const uint16_t* wkdyn, const uint16_t* wvdyn,
                              const float* inp0, const float* inp1, const float* inp2,
                              float* q_out, float* k_out, float* v_out,
                              float* r_out, float* gate_out, float* sink_out,
                              float* kdyn_out, float* vdyn_out,
                              int k_dim, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim) {
    int total = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    qkvrg_gemv_k<<<dim3((total + 7) / 8), dim3(256), 0, s>>>(
        wq, wk, wv, wrel, wgate, wsink, wkdyn, wvdyn,
        inp0, inp1, inp2, q_out, k_out, v_out, r_out, gate_out, sink_out, kdyn_out, vdyn_out,
        k_dim, q_dim, kv_dim, r_dim, n_head, dyn_dim);
    return 0;
}

extern "C" int ink_kv_conv_qkvr_store(hipStream_t s,
                                      float* Q, const float* K_raw, const float* V_raw, float* R,
                                      const float* k_conv_w, const float* v_conv_w,
                                      const float* k_dyn, const float* v_dyn,
                                      float* k_hist, float* v_hist,
                                      const float* q_clip, const float* rel_clip,
                                      __half* k_cache, __half* v_cache,
                                      const int* d_pos, int is_swa, int n_head, int n_head_kv,
                                      int dh, int d_rel, int max_ctx,
                                      float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_k<<<dim3(n_head + n_head_kv), dim3(32), 0, s>>>(
        Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist,
        q_clip, rel_clip, k_cache, v_cache, d_pos, is_swa, n_head, n_head_kv,
        dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}

extern "C" int ink_attn_decode_gated(hipStream_t s,
                                     const float* Q, const float* R, const float* sinks, const float* rel_bias,
                                     const __half* k_cache, const __half* v_cache, const float* gate,
                                     const int* d_pos, float* out,
                                     int n_head, int n_head_kv, int dh, int d_rel,
                                     int rel_extent, int swa_window, int is_swa, int max_ctx) {
    inkling_attn_decode_gated_k<<<dim3(n_head), dim3(256), 0, s>>>(
        Q, R, sinks, rel_bias, k_cache, v_cache, gate, d_pos, out,
        n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" int ink_wo_conv_res(hipStream_t s,
                               const uint8_t* wo, const float* x, const float* conv_w,
                               float* hist, const float* add_src, float* block_dst, float* out,
                               int n_embd, int q_dim) {
    wo_conv_res_k<<<dim3((n_embd + 7) / 8), dim3(256), 0, s>>>(
        wo, x, conv_w, hist, add_src, block_dst, out, n_embd, q_dim);
    return 0;
}

extern "C" int ink_moe_router_fused(hipStream_t s,
                                    const float* w_router, const float* x, const float* bias,
                                    float* wtop, int* itop,
                                    int n_expert, int k_dim, int n_used, float route_scale) {
    moe_router_gemv_k<<<dim3(1), dim3(256), 0, s>>>(
        w_router, x, bias, wtop, itop, n_expert, k_dim, n_used, route_scale);
    return 0;
}

extern "C" int ink_moe_down_conv_add_nvfp4(hipStream_t s,
                                           const uint8_t* W, const float* scales, const float* act_in,
                                           const int* itop, const float* wtop,
                                           const float* conv_w, float* hist,
                                           const float* ffn_inp, float* cur_out,
                                           int n_embd, int n_ff, int n_used) {
    moe_down_conv_add_nvfp4_k<<<dim3((n_embd + 15) / 16), dim3(128), 0, s>>>(
        W, scales, act_in, itop, wtop, conv_w, hist, ffn_inp, cur_out, n_embd, n_ff, n_used);
    return 0;
}

extern "C" __global__ void mix_rmsnorm_k(const float* __restrict__ blocks,
                                         int n_blocks,
                                         const float* __restrict__ cur,
                                         const float* __restrict__ query,
                                         const float* __restrict__ w0,
                                         const float* __restrict__ w1,
                                         const float* __restrict__ w2,
                                         float* __restrict__ out,
                                         int n_embd, float eps) {
    int q = blockIdx.x;
    const float* qv = query + (size_t)q * n_embd;
    float* ov = out + (size_t)q * n_embd;
    const float* nw = (q == 0) ? w0 : ((q == 1) ? w1 : w2);
    int n_states = n_blocks + 1;
    __shared__ float red[8];
    __shared__ float invs[16];
    __shared__ float scores[16];
    int wid = threadIdx.x >> 5;
    int nw_w = blockDim.x >> 5;
    float inv_sqrt_d = rsqrtf((float)n_embd);

    for (int s = 0; s < n_states; ++s) {
        const float* sv = (s < n_blocks) ? (blocks + (size_t)s * n_embd) : cur;
        float ss = 0.0f;
        float dot = 0.0f;
        for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
            float v = sv[j];
            ss += v * v;
            dot += v * qv[j];
        }
        ss = warp_sum32(ss);
        if ((threadIdx.x & 31) == 0) red[wid] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int i = 0; i < nw_w; ++i) t += red[i];
            invs[s] = rsqrtf(t / (float)n_embd + 1e-6f);
        }
        __syncthreads();
        dot = warp_sum32(dot);
        if ((threadIdx.x & 31) == 0) red[wid] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int i = 0; i < nw_w; ++i) t += red[i];
            scores[s] = t * invs[s] * inv_sqrt_d;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mx = scores[0];
        for (int s = 1; s < n_states; ++s) mx = fmaxf(mx, scores[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            scores[s] = expf(scores[s] - mx);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        for (int s = 0; s < n_states; ++s) {
            scores[s] *= inv_sum * invs[s];
        }
    }
    __syncthreads();

    float out_ss = 0.0f;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_blocks; ++s) {
            acc += scores[s] * blocks[(size_t)s * n_embd + j];
        }
        acc += scores[n_blocks] * cur[j];
        ov[j] = acc;
        out_ss += acc * acc;
    }
    out_ss = warp_sum32(out_ss);
    if ((threadIdx.x & 31) == 0) red[wid] = out_ss;
    __syncthreads();
    __shared__ float inv_out;
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw_w; ++i) t += red[i];
        inv_out = rsqrtf(t / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv_out;
    for (int j = threadIdx.x; j < n_embd; j += blockDim.x) {
        ov[j] = ov[j] * io * nw[j];
    }
}

extern "C" __global__ void loop_ctrl_down_batch_k(const float* __restrict__ cur0,
                                                  const float* __restrict__ cur1,
                                                  const float* __restrict__ cur2,
                                                  const float* __restrict__ msgs,
                                                  const float* __restrict__ wd0,
                                                  const float* __restrict__ wd1,
                                                  const float* __restrict__ wd2,
                                                  float* __restrict__ h_out,
                                                  int n_embd, int rank) {
    int r = blockIdx.x;
    int c = blockIdx.y;
    if (r >= rank) return;
    const float* cur = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * n_embd;
    const float* wd = (c == 0) ? wd0 : ((c == 1) ? wd1 : wd2);
    const float* wr = wd + (size_t)r * (2 * n_embd);
    __shared__ float red_sc[8];
    __shared__ float red_sm[8];
    __shared__ float red_ac[8];
    __shared__ float red_am[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int e4 = n_embd >> 2;
    const float4* c4 = reinterpret_cast<const float4*>(cur);
    const float4* m4 = reinterpret_cast<const float4*>(msg);
    const float4* wc4 = reinterpret_cast<const float4*>(wr);
    const float4* wm4 = reinterpret_cast<const float4*>(wr + n_embd);

    float sc = 0.0f, sm = 0.0f, ac = 0.0f, am = 0.0f;
    for (int j = tid; j < e4; j += 256) {
        float4 vc = c4[j];
        float4 vm = m4[j];
        float4 wcv = wc4[j];
        float4 wmv = wm4[j];
        sc = fmaf(vc.x, vc.x, fmaf(vc.y, vc.y, fmaf(vc.z, vc.z, fmaf(vc.w, vc.w, sc))));
        sm = fmaf(vm.x, vm.x, fmaf(vm.y, vm.y, fmaf(vm.z, vm.z, fmaf(vm.w, vm.w, sm))));
        ac = fmaf(wcv.x, vc.x, fmaf(wcv.y, vc.y, fmaf(wcv.z, vc.z, fmaf(wcv.w, vc.w, ac))));
        am = fmaf(wmv.x, vm.x, fmaf(wmv.y, vm.y, fmaf(wmv.z, vm.z, fmaf(wmv.w, vm.w, am))));
    }
    sc = warp_sum32(sc);
    sm = warp_sum32(sm);
    ac = warp_sum32(ac);
    am = warp_sum32(am);
    if (lane == 0) {
        red_sc[wid] = sc;
        red_sm[wid] = sm;
        red_ac[wid] = ac;
        red_am[wid] = am;
    }
    __syncthreads();
    if (tid == 0) {
        float tsc = 0.0f, tsm = 0.0f, tac = 0.0f, tam = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            tsc += red_sc[i];
            tsm += red_sm[i];
            tac += red_ac[i];
            tam += red_am[i];
        }
        float rc = rsqrtf(tsc / (float)n_embd + 1e-6f);
        float rm = rsqrtf(tsm / (float)n_embd + 1e-6f);
        h_out[(size_t)c * rank + r] = tac * rc + tam * rm;
    }
}

extern "C" __global__ void loop_ctrl_up_norm_batch_k(const float* __restrict__ cur0,
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
                                                     const float* __restrict__ nw0,
                                                     const float* __restrict__ nw1,
                                                     const float* __restrict__ nw2,
                                                     float* __restrict__ out0,
                                                     float* __restrict__ out1,
                                                     float* __restrict__ out2,
                                                     int n_embd, int rank, float eps) {
    int c = blockIdx.x;
    const float* cur = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * n_embd;
    const float* h = h_in + (size_t)c * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    float* out = (c == 0) ? out0 : ((c == 1) ? out1 : out2);

    __shared__ float hs[128];
    __shared__ float red[8];
    if (threadIdx.x < rank) hs[threadIdx.x] = h[threadIdx.x];
    __syncthreads();

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float* wgi = wg + (size_t)i * rank;
        const float* wfi = wf + (size_t)i * rank;
        float g = 0.0f, f = 0.0f;
        for (int r = 0; r < rank; ++r) {
            float hr = hs[r];
            g = fmaf(wgi[r], hr, g);
            f = fmaf(wfi[r], hr, f);
        }
        float cv = cur[i];
        float v = cv - tanhf(f) * cv + tanhf(g) * msg[i];
        out[i] = v;
        ss += v * v;
    }
    ss = warp_sum32(ss);
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nwarps; ++i) t += red[i];
        inv = rsqrtf(t / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        out[i] = out[i] * io * nw[i];
    }
}

extern "C" __global__ void loop_ctrl_up_fast_k(const float* __restrict__ cur0,
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
                                               int n_embd, int rank) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int i = blockIdx.x * 8 + warp;
    int c = blockIdx.y;
    if (i >= n_embd) return;
    const float* cur = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const float* msg = msgs + (size_t)c * n_embd;
    const float* h = h_in + (size_t)c * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    float* out = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const float2* wgi = reinterpret_cast<const float2*>(wg + (size_t)i * rank);
    const float2* wfi = reinterpret_cast<const float2*>(wf + (size_t)i * rank);
    const float2* h2 = reinterpret_cast<const float2*>(h);
    int r2 = rank >> 1;
    float g = 0.0f, f = 0.0f;
    for (int r = lane; r < r2; r += 32) {
        float2 hr = h2[r];
        float2 gv = wgi[r];
        float2 fv = wfi[r];
        g = fmaf(gv.x, hr.x, fmaf(gv.y, hr.y, g));
        f = fmaf(fv.x, hr.x, fmaf(fv.y, hr.y, f));
    }
    g = warp_sum32(g);
    f = warp_sum32(f);
    if (lane == 0) {
        float cv = cur[i];
        out[i] = cv - tanhf(f) * cv + tanhf(g) * msg[i];
    }
}

extern "C" __global__ void rmsnorm_multi_inplace_k(float* __restrict__ out0,
                                                   float* __restrict__ out1,
                                                   float* __restrict__ out2,
                                                   const float* __restrict__ nw0,
                                                   const float* __restrict__ nw1,
                                                   const float* __restrict__ nw2,
                                                   int n_embd, float eps) {
    int c = blockIdx.x;
    float* out = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    __shared__ float red[8];
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int e4 = n_embd >> 2;
    float4* o4 = reinterpret_cast<float4*>(out);
    const float4* w4 = reinterpret_cast<const float4*>(nw);
    float4 reg[3];
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        int j = tid + p * 256;
        if (j < e4) {
            float4 v = o4[j];
            reg[p] = v;
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
    }
    ss = warp_sum32(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += red[i];
        inv = rsqrtf(t / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        int j = tid + p * 256;
        if (j < e4) {
            float4 v = reg[p];
            float4 w = w4[j];
            o4[j] = make_float4(v.x * io * w.x, v.y * io * w.y, v.z * io * w.z, v.w * io * w.w);
        }
    }
}

extern "C" int ink_mix_rmsnorm(hipStream_t s, const float* blocks, int n_blocks, const float* cur, const float* query, const float* w0, const float* w1, const float* w2, float* out, int n_embd, int n_queries, float eps) {
    return ink_mix_batch(s, blocks, n_blocks, 1, cur, query, w0, w1, w2, out, n_embd, n_queries, 1, eps);
}

extern "C" int ink_loop_ctrl_3_norm(hipStream_t s, const float* cur0, const float* cur1, const float* cur2, const float* msgs, const float* wd0, const float* wd1, const float* wd2, const float* wg0, const float* wg1, const float* wg2, const float* wf0, const float* wf1, const float* wf2, const float* nw0, const float* nw1, const float* nw2, float* h_buf, float* out0, float* out1, float* out2, int n_embd, int rank, float eps) {
    loop_ctrl_down_batch_k<<<dim3(rank, 3), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, wd0, wd1, wd2, h_buf, n_embd, rank);
    loop_ctrl_up_fast_k<<<dim3((n_embd + 7) / 8, 3), dim3(256), 0, s>>>(cur0, cur1, cur2, msgs, h_buf, wg0, wg1, wg2, wf0, wf1, wf2, out0, out1, out2, n_embd, rank);
    rmsnorm_multi_inplace_k<<<dim3(3), dim3(256), 0, s>>>(out0, out1, out2, nw0, nw1, nw2, n_embd, eps);
    return 0;
}

extern "C" int ink_loop_ctrl_1_norm(hipStream_t s, const float* cur, const float* msg, const float* wd, const float* wg, const float* wf, const float* nw, float* h_buf, float* out, int n_embd, int rank, float eps) {
    loop_ctrl_down_batch_k<<<dim3(rank, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, wd, wd, wd, h_buf, n_embd, rank);
    loop_ctrl_up_fast_k<<<dim3((n_embd + 7) / 8, 1), dim3(256), 0, s>>>(cur, cur, cur, msg, h_buf, wg, wg, wg, wf, wf, wf, out, out, out, n_embd, rank);
    rmsnorm_multi_inplace_k<<<dim3(1), dim3(256), 0, s>>>(out, out, out, nw, nw, nw, n_embd, eps);
    return 0;
}

extern "C" int ink_moe_down_conv_add_q8_0(hipStream_t s,
                                          const uint8_t* W, const float* scales, const float* act_in,
                                          const int* itop, const float* wtop,
                                          const float* conv_w, float* hist,
                                          const float* ffn_inp, float* cur_out,
                                          int n_embd, int n_ff, int n_used) {
    moe_down_conv_add_q8_0_k<<<dim3((n_embd + 7) / 8), dim3(256), 0, s>>>(
        W, scales, act_in, itop, wtop, conv_w, hist, ffn_inp, cur_out, n_embd, n_ff, n_used);
    return 0;
}
