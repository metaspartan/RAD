#include "rad_q8kv_common.cuh"


extern "C" __global__ void kv_conv_qkvr_store_b1_q8kv_k(float* __restrict__ Q,
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
                                                        int8_t* __restrict__ k_cache,
                                                        int8_t* __restrict__ v_cache,
                                                        __half* __restrict__ k_scale,
                                                        __half* __restrict__ v_scale,
                                                        const int* __restrict__ d_pos,
                                                        int n_head, int n_head_kv,
                                                        int dh, int d_rel, int max_ctx,
                                                        float qk_thresh, float rel_thresh,
                                                        float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos = *d_pos;
    if (h < n_head) {
        float ratio = ((float)pos + 1.0f) / log_floor;
        if (ratio < 1.0f) ratio = 1.0f;
        float tau = 1.0f + log_alpha * logf(ratio);
        float* qh = Q + (size_t)h * dh;
        const float* qch = q_clip + (size_t)h * dh;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        float ss = q8kv_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;
        float* rh = R + (size_t)h * d_rel;
        const float* rch = rel_clip + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = q8kv_warp_sum32(rv * rv);
        float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
        if (lane < d_rel) rh[lane] = rv * rfactor;
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
        float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
        if (k_dyn) {
            const float* dk = k_dyn + (size_t)kh_idx * 4;
            const float* dv = v_dyn + (size_t)kh_idx * 4;
            kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
            vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
        }
        float kv_v[4], vv_v[4];
        float kss = 0.0f, vss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
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
        kss = q8kv_warp_sum32(kss);
        vss = q8kv_warp_sum32(vss);
        float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
        float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
        size_t slot_idx = (size_t)kh_idx * max_ctx + pos;
        q8kv_quantize_store_128(kv_v[0] * kfac, kv_v[1] * kfac, kv_v[2] * kfac, kv_v[3] * kfac, lane, k_cache + slot_idx * dh, k_scale + slot_idx);
        q8kv_quantize_store_128(vv_v[0] * vinv, vv_v[1] * vinv, vv_v[2] * vinv, vv_v[3] * vinv, lane, v_cache + slot_idx * dh, v_scale + slot_idx);
    }
}


extern "C" __global__ void kv_conv_qkvr_store_batch_snap_q8kv_k(float* __restrict__ Q,
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
                                                                int8_t* __restrict__ k_cache,
                                                                int8_t* __restrict__ v_cache,
                                                                __half* __restrict__ k_scale,
                                                                __half* __restrict__ v_scale,
                                                                const int* __restrict__ d_pos0,
                                                                int B, int n_head, int n_head_kv,
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
        float ratio = ((float)pos + 1.0f) / log_floor;
        if (ratio < 1.0f) ratio = 1.0f;
        float tau = 1.0f + log_alpha * logf(ratio);
        float* qh = Q + (size_t)t * q_dim + (size_t)h * dh;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        float ss = q8kv_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;
        float* rh = R + (size_t)t * r_dim + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = q8kv_warp_sum32(rv * rv);
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
            kss = q8kv_warp_sum32(kss);
            vss = q8kv_warp_sum32(vss);
            float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
            float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
            size_t slot_idx = (size_t)kh_idx * max_ctx + pos;
            q8kv_quantize_store_128(kv_v[0] * kfac, kv_v[1] * kfac, kv_v[2] * kfac, kv_v[3] * kfac, lane, k_cache + slot_idx * dh, k_scale + slot_idx);
            q8kv_quantize_store_128(vv_v[0] * vinv, vv_v[1] * vinv, vv_v[2] * vinv, vv_v[3] * vinv, lane, v_cache + slot_idx * dh, v_scale + slot_idx);
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            k_hist[c * 3 + 0] = kh0[p]; k_hist[c * 3 + 1] = kh1[p]; k_hist[c * 3 + 2] = kh2[p];
            v_hist[c * 3 + 0] = vh0[p]; v_hist[c * 3 + 1] = vh1[p]; v_hist[c * 3 + 2] = vh2[p];
        }
    }
}


extern "C" __global__ void kv_conv_qkvr_store_prefill_q8kv_k(float* __restrict__ Q,
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
                                                             int8_t* __restrict__ k_cache,
                                                             int8_t* __restrict__ v_cache,
                                                             __half* __restrict__ k_scale,
                                                             __half* __restrict__ v_scale,
                                                             const int* __restrict__ d_pos0,
                                                             int n_head, int n_head_kv,
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
        float ratio = ((float)pos + 1.0f) / log_floor;
        if (ratio < 1.0f) ratio = 1.0f;
        float tau = 1.0f + log_alpha * logf(ratio);
        const float* qch = q_clip + (size_t)h * dh;
        float* qh = Q + (size_t)t * q_dim + (size_t)h * dh;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        float ss = q8kv_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;
        const float* rch = rel_clip + (size_t)h * d_rel;
        float* rh = R + (size_t)t * r_dim + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = q8kv_warp_sum32(rv * rv);
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
        kss = q8kv_warp_sum32(kss);
        vss = q8kv_warp_sum32(vss);
        float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
        float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
        size_t slot_idx = (size_t)kh_idx * max_ctx + pos;
        q8kv_quantize_store_128(kv_v[0] * kfac, kv_v[1] * kfac, kv_v[2] * kfac, kv_v[3] * kfac, lane, k_cache + slot_idx * dh, k_scale + slot_idx);
        q8kv_quantize_store_128(vv_v[0] * vinv, vv_v[1] * vinv, vv_v[2] * vinv, vv_v[3] * vinv, lane, v_cache + slot_idx * dh, v_scale + slot_idx);
    }
}


extern "C" __global__ void q8kv_hist_update_3_k(const float* __restrict__ raw, float* __restrict__ hist, int dim, int B) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    float* hc = hist + (size_t)c * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    hc[0] = (B >= 3) ? raw[(size_t)(B - 3) * dim + c] : ((B == 2) ? h2 : h1);
    hc[1] = (B >= 2) ? raw[(size_t)(B - 2) * dim + c] : h2;
    hc[2] = raw[(size_t)(B - 1) * dim + c];
}


extern "C" __global__ void kv_conv_qkvr_store_multi_q8kv_k(float* __restrict__ Q,
                                                           const float* __restrict__ K_raw,
                                                           const float* __restrict__ V_raw,
                                                           float* __restrict__ R,
                                                           const float* __restrict__ k_conv_w,
                                                           const float* __restrict__ v_conv_w,
                                                           const float* __restrict__ k_dyn,
                                                           const float* __restrict__ v_dyn,
                                                           float* __restrict__ k_hist_pool,
                                                           float* __restrict__ v_hist_pool,
                                                           const float* __restrict__ q_clip,
                                                           const float* __restrict__ rel_clip,
                                                           int8_t* __restrict__ k_cache_pool,
                                                           int8_t* __restrict__ v_cache_pool,
                                                           __half* __restrict__ k_scale_pool,
                                                           __half* __restrict__ v_scale_pool,
                                                           const int* __restrict__ d_seq_ids,
                                                           const int* __restrict__ d_positions,
                                                           int n_head, int n_head_kv,
                                                           int dh, int d_rel, int max_ctx,
                                                           float qk_thresh, float rel_thresh,
                                                           float log_floor, float log_alpha) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int lane = threadIdx.x;
    int seq = d_seq_ids[bt];
    int pos = d_positions[bt];
    int q_dim = n_head * dh;
    int kv_dim = n_head_kv * dh;
    int r_dim = n_head * d_rel;
    int dyn_dim = n_head_kv * 4;

    if (h < n_head) {
        float ratio = ((float)pos + 1.0f) / log_floor;
        if (ratio < 1.0f) ratio = 1.0f;
        float tau = 1.0f + log_alpha * logf(ratio);
        float* qh = Q + (size_t)bt * q_dim + (size_t)h * dh;
        const float* qch = q_clip + (size_t)h * dh;
        float q0 = qh[lane] * qch[lane];
        float q1 = qh[lane + 32] * qch[lane + 32];
        float q2 = qh[lane + 64] * qch[lane + 64];
        float q3 = qh[lane + 96] * qch[lane + 96];
        float ss = q8kv_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
        qh[lane] = q0 * factor;
        qh[lane + 32] = q1 * factor;
        qh[lane + 64] = q2 * factor;
        qh[lane + 96] = q3 * factor;
        float* rh = R + (size_t)bt * r_dim + (size_t)h * d_rel;
        const float* rch = rel_clip + (size_t)h * d_rel;
        float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
        float rss = q8kv_warp_sum32(rv * rv);
        float rfactor = fminf(1.0f, rel_thresh / (sqrtf(__shfl(rss, 0, 32)) + 1e-6f)) * tau;
        if (lane < d_rel) rh[lane] = rv * rfactor;
    } else if (h < n_head + n_head_kv) {
        int kh_idx = h - n_head;
        float sc = k_dyn ? rsqrtf(9.0f) : rsqrtf(5.0f);
        float kd0 = 0.f, kd1 = 0.f, kd2 = 0.f, kd3 = 0.f;
        float vd0 = 0.f, vd1 = 0.f, vd2 = 0.f, vd3 = 0.f;
        if (k_dyn) {
            const float* dk = k_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
            const float* dv = v_dyn + (size_t)bt * dyn_dim + (size_t)kh_idx * 4;
            kd0 = dk[0]; kd1 = dk[1]; kd2 = dk[2]; kd3 = dk[3];
            vd0 = dv[0]; vd1 = dv[1]; vd2 = dv[2]; vd3 = dv[3];
        }
        float* k_hist = k_hist_pool + (size_t)seq * kv_dim * 3;
        float* v_hist = v_hist_pool + (size_t)seq * kv_dim * 3;
        float kv_v[4], vv_v[4];
        float kss = 0.0f, vss = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            float* khc = k_hist + (size_t)c * 3;
            float kh0 = khc[0], kh1 = khc[1], kh2 = khc[2];
            float kx = K_raw[(size_t)bt * kv_dim + c];
            const float* kwc = k_conv_w + (size_t)c * 4;
            float kacc = kwc[0] * kh0 + kwc[1] * kh1 + kwc[2] * kh2 + kwc[3] * kx;
            if (k_dyn) kacc += kd0 * kx + kd1 * kh2 + kd2 * kh1 + kd3 * kh0;
            khc[0] = kh1; khc[1] = kh2; khc[2] = kx;
            float kout = (kx + kacc) * sc;
            kv_v[p] = kout;
            kss += kout * kout;

            float* vhc = v_hist + (size_t)c * 3;
            float vh0 = vhc[0], vh1 = vhc[1], vh2 = vhc[2];
            float vx = V_raw[(size_t)bt * kv_dim + c];
            const float* vwc = v_conv_w + (size_t)c * 4;
            float vacc = vwc[0] * vh0 + vwc[1] * vh1 + vwc[2] * vh2 + vwc[3] * vx;
            if (v_dyn) vacc += vd0 * vx + vd1 * vh2 + vd2 * vh1 + vd3 * vh0;
            vhc[0] = vh1; vhc[1] = vh2; vhc[2] = vx;
            float vout = (vx + vacc) * sc;
            vv_v[p] = vout;
            vss += vout * vout;
        }
        kss = q8kv_warp_sum32(kss);
        vss = q8kv_warp_sum32(vss);
        float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
        float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
        size_t seq_sc_stride = (size_t)n_head_kv * max_ctx;
        size_t slot_idx = (size_t)seq * seq_sc_stride + (size_t)kh_idx * max_ctx + pos;
        q8kv_quantize_store_128(kv_v[0] * kfac, kv_v[1] * kfac, kv_v[2] * kfac, kv_v[3] * kfac, lane, k_cache_pool + slot_idx * dh, k_scale_pool + slot_idx);
        q8kv_quantize_store_128(vv_v[0] * vinv, vv_v[1] * vinv, vv_v[2] * vinv, vv_v[3] * vinv, lane, v_cache_pool + slot_idx * dh, v_scale_pool + slot_idx);
    }
}


extern "C" __global__ void kv_conv_qkvr_store_multi_snap_q8kv_k(float* __restrict__ Q,
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
                                                                int8_t* __restrict__ k_cache_pool,
                                                                int8_t* __restrict__ v_cache_pool,
                                                                __half* __restrict__ k_scale_pool,
                                                                __half* __restrict__ v_scale_pool,
                                                                const int* __restrict__ d_seq_ids,
                                                                const int* __restrict__ d_positions,
                                                                int K, int n_head, int n_head_kv,
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
            float ratio = ((float)pos + 1.0f) / log_floor;
            if (ratio < 1.0f) ratio = 1.0f;
            float tau = 1.0f + log_alpha * logf(ratio);
            float* qh = Q + (size_t)bt * q_dim + (size_t)h * dh;
            float q0 = qh[lane] * qch[lane];
            float q1 = qh[lane + 32] * qch[lane + 32];
            float q2 = qh[lane + 64] * qch[lane + 64];
            float q3 = qh[lane + 96] * qch[lane + 96];
            float ss = q8kv_warp_sum32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
            float factor = fminf(1.0f, qk_thresh / (sqrtf(__shfl(ss, 0, 32)) + 1e-6f)) * tau;
            qh[lane] = q0 * factor;
            qh[lane + 32] = q1 * factor;
            qh[lane + 64] = q2 * factor;
            qh[lane + 96] = q3 * factor;
            float* rh = R + (size_t)bt * r_dim + (size_t)h * d_rel;
            float rv = (lane < d_rel) ? (rh[lane] * rch[lane]) : 0.0f;
            float rss = q8kv_warp_sum32(rv * rv);
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
        float kh0[4], kh1[4], kh2[4], vh0[4], vh1[4], vh2[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            kh0[p] = k_hist[c * 3 + 0]; kh1[p] = k_hist[c * 3 + 1]; kh2[p] = k_hist[c * 3 + 2];
            vh0[p] = v_hist[c * 3 + 0]; vh1[p] = v_hist[c * 3 + 1]; vh2[p] = v_hist[c * 3 + 2];
        }
        size_t seq_sc_stride = (size_t)n_head_kv * max_ctx;
        for (int t = 0; t < K; ++t) {
            int bt = bt_base + t;
            int pos = pos0 + t;
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
            kss = q8kv_warp_sum32(kss);
            vss = q8kv_warp_sum32(vss);
            float kfac = fminf(1.0f, qk_thresh / (sqrtf(__shfl(kss, 0, 32)) + 1e-6f));
            float vinv = rsqrtf(__shfl(vss, 0, 32) / (float)dh + 1e-6f);
            size_t slot_idx = (size_t)seq * seq_sc_stride + (size_t)kh_idx * max_ctx + pos;
            q8kv_quantize_store_128(kv_v[0] * kfac, kv_v[1] * kfac, kv_v[2] * kfac, kv_v[3] * kfac, lane, k_cache_pool + slot_idx * dh, k_scale_pool + slot_idx);
            q8kv_quantize_store_128(vv_v[0] * vinv, vv_v[1] * vinv, vv_v[2] * vinv, vv_v[3] * vinv, lane, v_cache_pool + slot_idx * dh, v_scale_pool + slot_idx);
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            int c = kh_idx * dh + lane + p * 32;
            k_hist[c * 3 + 0] = kh0[p]; k_hist[c * 3 + 1] = kh1[p]; k_hist[c * 3 + 2] = kh2[p];
            v_hist[c * 3 + 0] = vh0[p]; v_hist[c * 3 + 1] = vh1[p]; v_hist[c * 3 + 2] = vh2[p];
        }
    }
}


extern "C" int ink_kv_conv_qkvr_store_b1_q8kv(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist, float* v_hist, const float* q_clip, const float* rel_clip, int8_t* kc, int8_t* vc, __half* ks, __half* vs, const int* d_pos, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_b1_q8kv_k<<<dim3(n_head + n_head_kv), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, q_clip, rel_clip, kc, vc, ks, vs, d_pos, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}


extern "C" int ink_kv_conv_qkvr_store_batch_snap_q8kv(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist, float* v_hist, float* k_snap, float* v_snap, const float* q_clip, const float* rel_clip, int8_t* kc, int8_t* vc, __half* ks, __half* vs, const int* d_pos0, int B, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    if (B > 8 && k_snap == nullptr && v_snap == nullptr) {
        kv_conv_qkvr_store_prefill_q8kv_k<<<dim3(n_head + n_head_kv, B), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, q_clip, rel_clip, kc, vc, ks, vs, d_pos0, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
        int kv_dim = n_head_kv * dh;
        q8kv_hist_update_3_k<<<dim3((kv_dim + 255) / 256), dim3(256), 0, s>>>(K_raw, k_hist, kv_dim, B);
        q8kv_hist_update_3_k<<<dim3((kv_dim + 255) / 256), dim3(256), 0, s>>>(V_raw, v_hist, kv_dim, B);
        return 0;
    }
    kv_conv_qkvr_store_batch_snap_q8kv_k<<<dim3(n_head * B + n_head_kv), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist, v_hist, k_snap, v_snap, q_clip, rel_clip, kc, vc, ks, vs, d_pos0, B, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}


extern "C" int ink_kv_conv_qkvr_store_multi_q8kv(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist_pool, float* v_hist_pool, const float* q_clip, const float* rel_clip, int8_t* kc_pool, int8_t* vc_pool, __half* ks_pool, __half* vs_pool, const int* d_seq_ids, const int* d_positions, int B, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_multi_q8kv_k<<<dim3(n_head + n_head_kv, B), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist_pool, v_hist_pool, q_clip, rel_clip, kc_pool, vc_pool, ks_pool, vs_pool, d_seq_ids, d_positions, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}


extern "C" int ink_kv_conv_qkvr_store_multi_snap_q8kv(hipStream_t s, float* Q, const float* K_raw, const float* V_raw, float* R, const float* k_conv_w, const float* v_conv_w, const float* k_dyn, const float* v_dyn, float* k_hist_pool, float* v_hist_pool, float* k_snap_pool, float* v_snap_pool, const float* q_clip, const float* rel_clip, int8_t* kc_pool, int8_t* vc_pool, __half* ks_pool, __half* vs_pool, const int* d_seq_ids, const int* d_positions, int n_users, int K, int n_head, int n_head_kv, int dh, int d_rel, int max_ctx, float qk_thresh, float rel_thresh, float log_floor, float log_alpha) {
    kv_conv_qkvr_store_multi_snap_q8kv_k<<<dim3(n_head + n_head_kv, n_users), dim3(32), 0, s>>>(Q, K_raw, V_raw, R, k_conv_w, v_conv_w, k_dyn, v_dyn, k_hist_pool, v_hist_pool, k_snap_pool, v_snap_pool, q_clip, rel_clip, kc_pool, vc_pool, ks_pool, vs_pool, d_seq_ids, d_positions, K, n_head, n_head_kv, dh, d_rel, max_ctx, qk_thresh, rel_thresh, log_floor, log_alpha);
    return 0;
}
