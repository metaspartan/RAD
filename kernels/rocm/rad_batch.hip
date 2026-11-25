#include "rad_batch_common.cuh"



extern "C" __global__ void gemm_q8_0_k(const uint8_t* __restrict__ W,
                                       const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= n) return;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    int nblk = k >> 5;
    const uint8_t* wrow = W + (size_t)row * ((size_t)nblk * 34);
    gemm_q8_0_row_tile8(wrow, X, Y, row, n, k, t0, n_t, lane);
}


extern "C" __global__ void gemm_bf16_k(const uint16_t* __restrict__ W,
                                       const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       int n, int k, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= n) return;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    const uint16_t* wrow = W + (size_t)row * k;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = lane; j < k; j += 32) {
        float wv = b_bf16_to_fp32(wrow[j]);
        for (int m = 0; m < n_t; ++m) {
            acc[m] = fmaf(wv, X[(size_t)(t0 + m) * k + j], acc[m]);
        }
    }
    for (int m = 0; m < n_t; ++m) {
        float v = b_warp_sum32(acc[m]);
        if (lane == 0) Y[(size_t)(t0 + m) * n + row] = v;
    }
}


extern "C" __global__ void qkvrg_gemm_k(const uint8_t* __restrict__ wq,
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
                                        int r_dim, int n_head, int dyn_dim, int B) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g_row = blockIdx.x * 8 + warp;
    int t0 = blockIdx.y * 8;
    int n_t = min(8, B - t0);
    if (n_t <= 0) return;
    int nblk = k_dim >> 5;
    size_t rb = (size_t)nblk * 34;

    if (g_row < q_dim) {
        gemm_q8_0_row_tile8(wq + (size_t)g_row * rb, inp0, q_out, g_row, q_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= q_dim;
    if (g_row < kv_dim) {
        gemm_q8_0_row_tile8(wk + (size_t)g_row * rb, inp1, k_out, g_row, kv_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= kv_dim;
    if (g_row < kv_dim) {
        gemm_q8_0_row_tile8(wv + (size_t)g_row * rb, inp2, v_out, g_row, kv_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= kv_dim;
    if (g_row < r_dim) {
        gemm_q8_0_row_tile8(wrel + (size_t)g_row * rb, inp0, r_out, g_row, r_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= r_dim;
    if (g_row < q_dim) {
        gemm_q8_0_row_tile8(wgate + (size_t)g_row * rb, inp0, gate_out, g_row, q_dim, k_dim, t0, n_t, lane);
        return;
    }
    g_row -= q_dim;
    if (g_row < n_head) {
        const uint16_t* wr = wsink + (size_t)g_row * k_dim;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int j = lane; j < k_dim; j += 32) {
            float wval = b_bf16_to_fp32(wr[j]);
            for (int m = 0; m < n_t; ++m) acc[m] = fmaf(wval, inp0[(size_t)(t0 + m) * k_dim + j], acc[m]);
        }
        for (int m = 0; m < n_t; ++m) {
            float v = b_warp_sum32(acc[m]);
            if (lane == 0) sink_out[(size_t)(t0 + m) * n_head + g_row] = v;
        }
        return;
    }
    g_row -= n_head;
    if (dyn_dim > 0) {
        if (g_row < dyn_dim) {
            const uint16_t* wr = wkdyn + (size_t)g_row * k_dim;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = lane; j < k_dim; j += 32) {
                float wval = b_bf16_to_fp32(wr[j]);
                for (int m = 0; m < n_t; ++m) acc[m] = fmaf(wval, inp1[(size_t)(t0 + m) * k_dim + j], acc[m]);
            }
            for (int m = 0; m < n_t; ++m) {
                float v = b_warp_sum32(acc[m]);
                if (lane == 0) kdyn_out[(size_t)(t0 + m) * dyn_dim + g_row] = v;
            }
            return;
        }
        g_row -= dyn_dim;
        if (g_row < dyn_dim) {
            const uint16_t* wr = wvdyn + (size_t)g_row * k_dim;
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int j = lane; j < k_dim; j += 32) {
                float wval = b_bf16_to_fp32(wr[j]);
                for (int m = 0; m < n_t; ++m) acc[m] = fmaf(wval, inp2[(size_t)(t0 + m) * k_dim + j], acc[m]);
            }
            for (int m = 0; m < n_t; ++m) {
                float v = b_warp_sum32(acc[m]);
                if (lane == 0) vdyn_out[(size_t)(t0 + m) * dyn_dim + g_row] = v;
            }
            return;
        }
    }
}


extern "C" __global__ void embed_q8_0_norm_batch_k(const uint8_t* __restrict__ tok_embd,
                                                   const float* __restrict__ tok_norm,
                                                   const int* __restrict__ d_toks,
                                                   float* __restrict__ out,
                                                   int n_embd) {
    int t = blockIdx.x;
    int tok = d_toks[t];
    int nblk = n_embd >> 5;
    const uint8_t* row = tok_embd + (size_t)tok * ((size_t)nblk * 34);
    float* ov = out + (size_t)t * n_embd;
    float ss = 0.0f;
    for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) {
        int b = idx >> 5;
        int j = idx & 31;
        const uint8_t* bp = row + (size_t)b * 34;
        uint16_t dh_bits = (uint16_t)bp[0] | ((uint16_t)bp[1] << 8);
        float d = __half2float(*reinterpret_cast<const __half*>(&dh_bits));
        float v = d * (float)(reinterpret_cast<const int8_t*>(bp + 2)[j]);
        ov[idx] = v;
        ss += v * v;
    }
    ss = b_warp_sum32(ss);
    __shared__ float red[8];
    int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n_embd + 1e-6f);
    }
    __syncthreads();
    float io = inv;
    if (tok_norm) {
        for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) ov[idx] = ov[idx] * io * tok_norm[idx];
    } else {
        for (int idx = threadIdx.x; idx < n_embd; idx += blockDim.x) ov[idx] = ov[idx] * io;
    }
}


extern "C" __global__ void rmsnorm_batch_k(const float* __restrict__ X,
                                           const float* __restrict__ w,
                                           float* __restrict__ Y,
                                           int n, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    const float* xv = X + (size_t)t * n;
    float* yv = Y + (size_t)t * n;
    __shared__ float red[8];
    __shared__ float inv;
    if ((n & 3) == 0 && n <= 3072) {
        int e4 = n >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(xv);
        float4* y4 = reinterpret_cast<float4*>(yv);
        float4 reg[3];
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int j = tid + p * 256;
            if (j < e4) {
                float4 v = x4[j];
                reg[p] = v;
                ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
            }
        }
        ss = b_warp_sum32(ss);
        if (lane == 0) red[wid] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) tot += red[i];
            inv = rsqrtf(tot / (float)n + eps);
        }
        __syncthreads();
        float io = inv;
        if (w) {
            const float4* w4 = reinterpret_cast<const float4*>(w);
#pragma unroll
            for (int p = 0; p < 3; ++p) {
                int j = tid + p * 256;
                if (j < e4) {
                    float4 v = reg[p];
                    float4 wv = w4[j];
                    y4[j] = make_float4(v.x * io * wv.x, v.y * io * wv.y, v.z * io * wv.z, v.w * io * wv.w);
                }
            }
        } else {
#pragma unroll
            for (int p = 0; p < 3; ++p) {
                int j = tid + p * 256;
                if (j < e4) {
                    float4 v = reg[p];
                    y4[j] = make_float4(v.x * io, v.y * io, v.z * io, v.w * io);
                }
            }
        }
        return;
    }
    float ss = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float v = xv[j];
        ss += v * v;
    }
    ss = b_warp_sum32(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n + eps);
    }
    __syncthreads();
    float io = inv;
    if (w) {
        for (int j = tid; j < n; j += blockDim.x) yv[j] = xv[j] * io * w[j];
    } else {
        for (int j = tid; j < n; j += blockDim.x) yv[j] = xv[j] * io;
    }
}


extern "C" __global__ void loop_ctrl_up_norm_b_k(const float* __restrict__ cur0,
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
                                                 int n_embd, int rank, int max_B, float eps) {
    int c = blockIdx.x;
    int t = blockIdx.y;
    const float* cur = ((c == 0) ? cur0 : ((c == 1) ? cur1 : cur2)) + (size_t)t * n_embd;
    const float* msg = msgs + ((size_t)c * max_B + t) * n_embd;
    const float* h = h_in + ((size_t)c * max_B + t) * rank;
    const float* wg = (c == 0) ? wg0 : ((c == 1) ? wg1 : wg2);
    const float* wf = (c == 0) ? wf0 : ((c == 1) ? wf1 : wf2);
    const float* nw = (c == 0) ? nw0 : ((c == 1) ? nw1 : nw2);
    float* out = ((c == 0) ? out0 : ((c == 1) ? out1 : out2)) + (size_t)t * n_embd;

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
    ss = b_warp_sum32(ss);
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) red[wid] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nwarps; ++i) tot += red[i];
        inv = rsqrtf(tot / (float)n_embd + eps);
    }
    __syncthreads();
    float io = inv;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        out[i] = out[i] * io * nw[i];
    }
}


extern "C" __global__ void batch_hist_update_3_k(const float* __restrict__ raw,
                                                 float* __restrict__ hist,
                                                 int dim, int B) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    float* hc = hist + (size_t)c * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    if (B >= 3) {
        hc[0] = raw[(size_t)(B - 3) * dim + c];
        hc[1] = raw[(size_t)(B - 2) * dim + c];
        hc[2] = raw[(size_t)(B - 1) * dim + c];
    } else if (B == 2) {
        hc[0] = h2;
        hc[1] = raw[c];
        hc[2] = raw[(size_t)dim + c];
    } else if (B == 1) {
        hc[0] = h1;
        hc[1] = h2;
        hc[2] = raw[c];
    }
}


extern "C" __global__ void kv_conv_qkvr_store_pf_b_k(float* __restrict__ Q,
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
    int pos0 = *d_pos0;
    int pos = pos0 + t;
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
        const float* rch = rel_clip + (size_t)h * d_rel;
        float* qh = Q + (size_t)t * q_dim + (size_t)h * dh;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        float ss = b_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;

        float* rh = R + (size_t)t * r_dim + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = b_warp_sum32(rv * rv);
        float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
        if (lane < d_rel) rh[lane] = rv * rfactor;
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
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
        kss = b_warp_sum32(kss);
        vss = b_warp_sum32(vss);
        float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
        float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
        int slot = is_swa ? (pos % max_ctx) : pos;
        __half* kdst = k_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
        __half* vdst = v_cache + ((size_t)kh_idx * max_ctx + slot) * dh;
        kdst[lane] = __float2half(kv_v[0] * kfac);
        kdst[lane + 32] = __float2half(kv_v[1] * kfac);
        kdst[lane + 64] = __float2half(kv_v[2] * kfac);
        kdst[lane + 96] = __float2half(kv_v[3] * kfac);
        vdst[lane] = __float2half(vv_v[0] * vinv);
        vdst[lane + 32] = __float2half(vv_v[1] * vinv);
        vdst[lane + 64] = __float2half(vv_v[2] * vinv);
        vdst[lane + 96] = __float2half(vv_v[3] * vinv);
    }
}


extern "C" __global__ void kv_conv_qkvr_store_batch_k(float* __restrict__ Q,
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
                                                      const int* __restrict__ d_pos0,
                                                      int B, int is_swa, int n_head, int n_head_kv,
                                                      int dh, int d_rel, int max_ctx,
                                                      float qk_thresh, float rel_thresh,
                                                      float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (h < n_head) {
        const float* qch = q_clip + (size_t)h * dh;
        const float* rch = rel_clip + (size_t)h * d_rel;
        for (int t = 0; t < B; ++t) {
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
            float ss = b_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
            float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
            qh[lane] = q0 * factor;
            qh[lane + 32] = q1 * factor;
            qh[lane + 64] = q2 * factor;
            qh[lane + 96] = q3 * factor;

            float* rh = R + (size_t)t * r_dim + (size_t)h * d_rel;
            float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
            float rss = b_warp_sum32(rv * rv);
            float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
            if (lane < d_rel) rh[lane] = rv * rfactor;
        }
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
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
                float kout = (kx + kacc) * sc;
                kv_v[p] = kout;
                kss += kout * kout;

                float vx = V_raw[(size_t)t * kv_dim + c];
                const float* vwc = v_conv_w + (size_t)c * 4;
                float vacc = vwc[0] * vh0[p] + vwc[1] * vh1[p] + vwc[2] * vh2[p] + vwc[3] * vx;
                if (v_dyn) vacc += vd0 * vx + vd1 * vh2[p] + vd2 * vh1[p] + vd3 * vh0[p];
                vh0[p] = vh1[p]; vh1[p] = vh2[p]; vh2[p] = vx;
                float vout = (vx + vacc) * sc;
                vv_v[p] = vout;
                vss += vout * vout;
            }
            kss = b_warp_sum32(kss);
            vss = b_warp_sum32(vss);
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


extern "C" __global__ void inkling_attn_batch_gated_k(const float* __restrict__ Q,
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
                                                      int is_swa, int max_ctx) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
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

    int pos = (*d_pos0) + bt;
    int t_start = is_swa ? max(0, pos - swa_window + 1) : 0;
    int t_end = pos + 1;
    float sink_h = sinks[(size_t)bt * n_head + h];

    const __half* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const __half* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int d0 = lane, d1 = lane + 32, d2 = lane + 64, d3 = lane + 96;

    for (int t = t_start + wid; t < t_end; t += nw) {
        int kv_slot = is_swa ? (t % max_ctx) : t;
        const __half* kt = kc_h + (size_t)kv_slot * dh;
        float dot = qs[d0] * __half2float(kt[d0])
                  + qs[d1] * __half2float(kt[d1])
                  + qs[d2] * __half2float(kt[d2])
                  + qs[d3] * __half2float(kt[d3]);
        dot = b_warp_sum32(dot);
        float score = __shfl(dot, 0, 32);

        int dist = pos - t;
        float rbias = 0.0f;
        if (dist < rel_extent) {
            const float* rb = rel_bias + (size_t)dist * d_rel;
            float rp = (lane < d_rel) ? (rb[lane] * rs[lane]) : 0.0f;
            rp = b_warp_sum32(rp);
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
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    sm_o[wid][d0] = o0; sm_o[wid][d1] = o1; sm_o[wid][d2] = o2; sm_o[wid][d3] = o3;
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
        float* oh = out + (size_t)bt * q_dim + (size_t)h * dh;
        const float* gh = gate + (size_t)bt * q_dim + (size_t)h * dh;
        oh[d0] = (go0 * inv_l) / (1.0f + expf(-gh[d0]));
        oh[d1] = (go1 * inv_l) / (1.0f + expf(-gh[d1]));
        oh[d2] = (go2 * inv_l) / (1.0f + expf(-gh[d2]));
        oh[d3] = (go3 * inv_l) / (1.0f + expf(-gh[d3]));
    }
}


extern "C" __global__ void shortconv_res_pf_b_k(const float* __restrict__ raw,
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
    float h2 = (t >= 1) ? raw[(size_t)(t - 1) * n_embd + c] : hist[(size_t)c * 3 + 2];
    float h1 = (t >= 2) ? raw[(size_t)(t - 2) * n_embd + c] : hist[(size_t)c * 3 + 1 + t];
    float h0 = (t >= 3) ? raw[(size_t)(t - 3) * n_embd + c] : hist[(size_t)c * 3 + t];
    const float* wc = conv_w + (size_t)c * 4;
    float conv_out = (x + wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * x) * rsqrtf(5.0f);
    if (block_dst) {
        block_dst[idx] = add_src[idx];
        out[idx] = conv_out;
    } else {
        out[idx] = add_src[idx] + conv_out;
    }
}


extern "C" __global__ void shortconv_res_batch_k(const float* __restrict__ raw,
                                                 const float* __restrict__ conv_w,
                                                 float* __restrict__ hist,
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
    float sc = rsqrtf(5.0f);
    for (int t = 0; t < B; ++t) {
        size_t idx = (size_t)t * n_embd + c;
        float x = raw[idx];
        float conv_out = (x + w0 * h0 + w1 * h1 + w2 * h2 + w3 * x) * sc;
        h0 = h1; h1 = h2; h2 = x;
        if (block_dst) {
            block_dst[idx] = add_src[idx];
            out[idx] = conv_out;
        } else {
            out[idx] = add_src[idx] + conv_out;
        }
    }
    hc[0] = h0; hc[1] = h1; hc[2] = h2;
}


extern "C" __global__ void moe_router_batch_k(const float* __restrict__ logits,
                                              const float* __restrict__ bias,
                                              float* __restrict__ wtop,
                                              int* __restrict__ itop,
                                              int n_expert, int k, float route_scale) {
    int bt = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    const float* lg = logits + (size_t)bt * n_expert;
    float* wt = wtop + (size_t)bt * k;
    int* it = itop + (size_t)bt * k;
    __shared__ float sp[128];
    __shared__ float ss[128];
    __shared__ float wv[4];
    __shared__ int wi[4];
    __shared__ float chosen_p[16];
    if (tid < n_expert) {
        float p = 1.0f / (1.0f + expf(-lg[tid]));
        sp[tid] = p;
        ss[tid] = bias ? (p + bias[tid]) : p;
    } else {
        sp[tid] = 0.0f;
        ss[tid] = -1e30f;
    }
    __syncthreads();

    for (int t = 0; t < k; ++t) {
        float v = ss[tid];
        int idx = tid;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down(v, off, 32);
            int oi = __shfl_down(idx, off, 32);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0) {
            wv[wid] = v;
            wi[wid] = idx;
        }
        __syncthreads();
        if (tid == 0) {
            float bv = wv[0];
            int bi = wi[0];
            for (int w = 1; w < 4; ++w) {
                if (wv[w] > bv || (wv[w] == bv && wi[w] < bi)) {
                    bv = wv[w];
                    bi = wi[w];
                }
            }
            ss[bi] = -1e30f;
            it[t] = bi;
            chosen_p[t] = sp[bi];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float wsum = 0.0f;
        for (int t = 0; t < k; ++t) wsum += chosen_p[t];
        float inv = route_scale / (wsum + 1e-20f);
        for (int t = 0; t < k; ++t) wt[t] = chosen_p[t] * inv;
    }
}


extern "C" __global__ void moe_gu_nvfp4_batch_k(const uint8_t* __restrict__ W,
                                                const float* __restrict__ scales,
                                                const float* __restrict__ X,
                                                const int* __restrict__ itop,
                                                float* __restrict__ act_out,
                                                int n_ff, int k, int n_used,
                                                float beta, float lb) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    int slot = blockIdx.y;
    int bt = blockIdx.z;
    if (row >= n_ff) return;
    int exp_id = itop[(size_t)bt * n_used + slot];
    int nblk = k >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* base = W + (size_t)exp_id * exp_bytes;
    const float* xt = X + (size_t)bt * k;
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = b_dot_nvfp4_row8(base + (size_t)row * row_bytes, xt, nblk, lane8) * sc;
    float u = b_dot_nvfp4_row8(base + (size_t)(n_ff + row) * row_bytes, xt, nblk, lane8) * sc;
    if (lane8 == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[((size_t)bt * n_used + slot) * n_ff + row] = act * up;
    }
}


extern "C" __global__ void moe_dn_nvfp4_batch_k(const uint8_t* __restrict__ W,
                                                const float* __restrict__ scales,
                                                const float* __restrict__ act_in,
                                                const int* __restrict__ itop,
                                                const float* __restrict__ wtop,
                                                float* __restrict__ y_out,
                                                int n_embd, int n_ff, int n_used) {
    int grp = threadIdx.x >> 3;
    int lane8 = threadIdx.x & 7;
    int row = blockIdx.x * 16 + grp;
    int bt = blockIdx.y;
    if (row >= n_embd) return;
    int nblk = n_ff >> 6;
    size_t row_bytes = (size_t)nblk * 36;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[(size_t)bt * n_used + slot];
        float w = wtop[(size_t)bt * n_used + slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = b_dot_nvfp4_row8(wrow, act_in + ((size_t)bt * n_used + slot) * n_ff, nblk, lane8);
        total = fmaf(v, w, total);
    }
    if (lane8 == 0) y_out[(size_t)bt * n_embd + row] = total;
}


extern "C" __global__ void moe_gu_q8_0_batch_k(const uint8_t* __restrict__ W,
                                               const float* __restrict__ scales,
                                               const float* __restrict__ X,
                                               const int* __restrict__ itop,
                                               float* __restrict__ act_out,
                                               int n_ff, int k, int n_used,
                                               float beta, float lb) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int slot = blockIdx.y;
    int bt = blockIdx.z;
    if (row >= n_ff) return;
    int exp_id = itop[(size_t)bt * n_used + slot];
    int nblk = k >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    size_t exp_bytes = (size_t)(2 * n_ff) * row_bytes;
    const uint8_t* base = W + (size_t)exp_id * exp_bytes;
    const float* xt = X + (size_t)bt * k;
    float sc = scales ? scales[exp_id] : 1.0f;
    float g = b_dot_q8_0_row32(base + (size_t)row * row_bytes, xt, nblk, lane) * sc;
    float u = b_dot_q8_0_row32(base + (size_t)(n_ff + row) * row_bytes, xt, nblk, lane) * sc;
    if (lane == 0) {
        float sg = 1.0f / (1.0f + expf(-g));
        float act = beta * tanhf(g / beta) * sg;
        float up = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
        act_out[((size_t)bt * n_used + slot) * n_ff + row] = act * up;
    }
}


extern "C" __global__ void moe_dn_q8_0_batch_k(const uint8_t* __restrict__ W,
                                               const float* __restrict__ scales,
                                               const float* __restrict__ act_in,
                                               const int* __restrict__ itop,
                                               const float* __restrict__ wtop,
                                               float* __restrict__ y_out,
                                               int n_embd, int n_ff, int n_used) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int bt = blockIdx.y;
    if (row >= n_embd) return;
    int nblk = n_ff >> 5;
    size_t row_bytes = (size_t)nblk * 34;
    size_t exp_bytes = (size_t)n_embd * row_bytes;
    float total = 0.0f;
    for (int slot = 0; slot < n_used; ++slot) {
        int exp_id = itop[(size_t)bt * n_used + slot];
        float w = wtop[(size_t)bt * n_used + slot] * (scales ? scales[exp_id] : 1.0f);
        const uint8_t* wrow = W + (size_t)exp_id * exp_bytes + (size_t)row * row_bytes;
        float v = b_dot_q8_0_row32(wrow, act_in + ((size_t)bt * n_used + slot) * n_ff, nblk, lane);
        total = fmaf(v, w, total);
    }
    if (lane == 0) y_out[(size_t)bt * n_embd + row] = total;
}


extern "C" int ink_gemm_q8_0(hipStream_t s, const uint8_t* W, const float* X, float* Y, int n, int k, int B) {
    gemm_q8_0_k<<<dim3((n + 7) / 8, (B + 7) / 8), dim3(256), 0, s>>>(W, X, Y, n, k, B);
    return 0;
}

extern "C" int ink_gemm_bf16(hipStream_t s, const uint16_t* W, const float* X, float* Y, int n, int k, int B) {
    gemm_bf16_k<<<dim3((n + 3) / 4, (B + 7) / 8), dim3(128), 0, s>>>(W, X, Y, n, k, B);
    return 0;
}

extern "C" int ink_qkvrg_gemm(hipStream_t s, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wrel, const uint8_t* wgate, const uint16_t* wsink, const uint16_t* wkdyn, const uint16_t* wvdyn, const float* inp0, const float* inp1, const float* inp2, float* q_out, float* k_out, float* v_out, float* r_out, float* gate_out, float* sink_out, float* kdyn_out, float* vdyn_out, int k_dim, int q_dim, int kv_dim, int r_dim, int n_head, int dyn_dim, int B) {
    int total = q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head + 2 * dyn_dim;
    qkvrg_gemm_k<<<dim3((total + 7) / 8, (B + 7) / 8), dim3(256), 0, s>>>(wq, wk, wv, wrel, wgate, wsink, wkdyn, wvdyn, inp0, inp1, inp2, q_out, k_out, v_out, r_out, gate_out, sink_out, kdyn_out, vdyn_out, k_dim, q_dim, kv_dim, r_dim, n_head, dyn_dim, B);
    return 0;
}

extern "C" int ink_embed_q8_0_norm_batch(hipStream_t s, const uint8_t* tok_embd, const float* tok_norm, const int* d_toks, float* out, int n_embd, int B) {
    embed_q8_0_norm_batch_k<<<dim3(B), dim3(256), 0, s>>>(tok_embd, tok_norm, d_toks, out, n_embd);
    return 0;
}

extern "C" int ink_rmsnorm_batch(hipStream_t s, const float* X, const float* w, float* Y, int n, int B, float eps) {
    rmsnorm_batch_k<<<dim3(B), dim3(256), 0, s>>>(X, w, Y, n, eps);
    return 0;
}

extern "C" int ink_kv_conv_qkvr_store_batch(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist, float* v_hist, const float* q_clip, const float* rel_clip, __half* k_cache, __half* v_cache, const int* d_pos0, int B, int is_swa, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    if (B > 16) {
        kv_conv_qkvr_store_pf_b_k<<<dim3(n_head + n_head_kv, B), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, q_clip, rel_clip, k_cache, v_cache, d_pos0, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
        int kv_dim = n_head_kv * dh;
        batch_hist_update_3_k<<<dim3((kv_dim + 255) / 256), dim3(256), 0, s>>>(K_raw, k_hist, kv_dim, B);
        batch_hist_update_3_k<<<dim3((kv_dim + 255) / 256), dim3(256), 0, s>>>(V_raw, v_hist, kv_dim, B);
        return 0;
    }
    kv_conv_qkvr_store_batch_k<<<dim3(n_head + n_head_kv), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, q_clip, rel_clip, k_cache, v_cache, d_pos0, B, is_swa, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}

extern "C" int ink_attn_batch_gated(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* k_cache, const __half* v_cache, const float* gate, const int* d_pos0, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    inkling_attn_batch_gated_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, k_cache, v_cache, gate, d_pos0, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx);
    return 0;
}

extern "C" int ink_shortconv_res_batch(hipStream_t s, const float* raw, const float* conv_w, float* hist, const float* add_src, float* block_dst, float* out, int n_embd, int B) {
    if (B > 16) {
        shortconv_res_pf_b_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(raw, conv_w, hist, add_src, block_dst, out, n_embd);
        batch_hist_update_3_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(raw, hist, n_embd, B);
        return 0;
    }
    shortconv_res_batch_k<<<dim3((n_embd + 255) / 256), dim3(256), 0, s>>>(raw, conv_w, hist, add_src, block_dst, out, n_embd, B);
    return 0;
}

extern "C" int ink_moe_router_batch(hipStream_t s, const float* logits, const float* bias, float* wtop, int* itop, int n_expert, int k, float route_scale, int B) {
    moe_router_batch_k<<<dim3(B), dim3(128), 0, s>>>(logits, bias, wtop, itop, n_expert, k, route_scale);
    return 0;
}

extern "C" int ink_moe_gu_q8_0_batch(hipStream_t s, const uint8_t* W, const float* scales, const float* X, const int* itop, float* act_out, int n_ff, int k, int n_used, float beta, float lb, int B) {
    moe_gu_q8_0_batch_k<<<dim3((n_ff + 7) / 8, n_used, B), dim3(256), 0, s>>>(W, scales, X, itop, act_out, n_ff, k, n_used, beta, lb);
    return 0;
}

extern "C" int ink_moe_dn_q8_0_batch(hipStream_t s, const uint8_t* W, const float* scales, const float* act_in, const int* itop, const float* wtop, float* y_out, int n_embd, int n_ff, int n_used, int B) {
    moe_dn_q8_0_batch_k<<<dim3((n_embd + 7) / 8, B), dim3(256), 0, s>>>(W, scales, act_in, itop, wtop, y_out, n_embd, n_ff, n_used);
    return 0;
}
