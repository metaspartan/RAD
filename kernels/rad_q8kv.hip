#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

typedef _Float16 q8kv_h8_v __attribute__((ext_vector_type(8)));
typedef float q8kv_f8_v __attribute__((ext_vector_type(8)));

struct q8kv_q8_1_blk {
    float d;
    int32_t s;
    int32_t q[8];
};

__device__ __forceinline__ float q8kv_warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ float q8kv_warp_max32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
    return v;
}

__device__ __forceinline__ int32_t q8kv_warp_sum32_i(int32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ int32_t q8kv_pack4_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (int32_t)((uint32_t)(uint8_t)a
                   | ((uint32_t)(uint8_t)b << 8)
                   | ((uint32_t)(uint8_t)c << 16)
                   | ((uint32_t)(uint8_t)d << 24));
}

__device__ __forceinline__ void q8kv_store_q8_1(float v, int lane, q8kv_q8_1_blk* dst, int8_t* smq) {
    float amax = __shfl(q8kv_warp_max32(fabsf(v)), 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    smq[lane] = (int8_t)qi;
    int32_t qsum = q8kv_warp_sum32_i(qi);
    __builtin_amdgcn_wave_barrier();
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if (lane < 8) {
        int i = lane * 4;
        dst->q[lane] = q8kv_pack4_i8(smq[i], smq[i + 1], smq[i + 2], smq[i + 3]);
    }
}

__device__ __forceinline__ void q8kv_quantize_store_128(float v0, float v1, float v2, float v3, int lane, int8_t* dst, __half* dst_scale) {
    float lm = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = __shfl(q8kv_warp_max32(lm), 0, 32);
    float sc = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    if (lane == 0) *dst_scale = __float2half(sc);
    int q0 = max(-127, min(127, __float2int_rn(v0 * inv)));
    int q1 = max(-127, min(127, __float2int_rn(v1 * inv)));
    int q2 = max(-127, min(127, __float2int_rn(v2 * inv)));
    int q3 = max(-127, min(127, __float2int_rn(v3 * inv)));
    dst[lane] = (int8_t)q0;
    dst[lane + 32] = (int8_t)q1;
    dst[lane + 64] = (int8_t)q2;
    dst[lane + 96] = (int8_t)q3;
}

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

__device__ float g_q8kv_b8_split_buf[24 * 96 * 64 * 130];

extern "C" __global__ void attn_b8_gqa3_flash_split_q8kv_k(const float* __restrict__ Q,
                                                           const float* __restrict__ R,
                                                           const float* __restrict__ sinks,
                                                           const float* __restrict__ rel_bias,
                                                           const int8_t* __restrict__ k_cache,
                                                           const int8_t* __restrict__ v_cache,
                                                           const __half* __restrict__ k_scale,
                                                           const __half* __restrict__ v_scale,
                                                           const int* __restrict__ d_pos0,
                                                           float* __restrict__ split_buf,
                                                           int n_head, int n_head_kv,
                                                           int dh, int d_rel,
                                                           int rel_extent, int max_ctx, int B, int n_splits) {
    int kv_h = blockIdx.x;
    int sp = blockIdx.y;
    int qb = blockIdx.z;
    int q_off = qb * 16;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int tg = wid / 3;
    int g = wid % 3;
    int h = kv_h * 3 + g;
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    int q_idx = q_off + col;
    bool vq = (q_idx < B);
    int pos_q = pos0 + q_idx;
    int n_q_tile = min(16, B - q_off);
    int t_max = pos0 + q_off + n_q_tile;
    int b_stride = (B > 64) ? 128 : 64;
    int eff_splits = min(n_splits, max(16, (pos0 + B + 255) >> 8));
    if (sp >= eff_splits) return;

    int total_tiles = (t_max + 15) / 16;
    int tiles_per_sp = (total_tiles + eff_splits - 1) / eff_splits;
    int tile_s = sp * tiles_per_sp * 16;
    int tile_e = min(t_max, tile_s + tiles_per_sp * 16);

    if (tile_s >= tile_e) {
        if (tg == 0 && half == 0 && vq) {
            float* dst = split_buf + (((size_t)h * n_splits + sp) * b_stride + q_idx) * 130;
            dst[0] = -1e30f;
            dst[1] = 0.0f;
        }
        return;
    }

    float q_scale = rsqrtf((float)dh);
    q8kv_h8_v q_reg[8];
    float4 r_reg0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg3 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sink_q = 0.0f;
    if (vq) {
        const float* qh = Q + (size_t)q_idx * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)q_idx * r_dim + (size_t)h * d_rel);
        r_reg0 = rh4[0]; r_reg1 = rh4[1]; r_reg2 = rh4[2]; r_reg3 = rh4[3];
        sink_q = sinks[(size_t)q_idx * n_head + h];
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = (sp == 0 && tg == 0) ? 0.0f : -1e30f;
    float l_q = (sp == 0 && tg == 0) ? 1.0f : 0.0f;
    q8kv_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int8_t* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const int8_t* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;
    const __half* ks_h = k_scale + (size_t)kv_h * max_ctx;
    const __half* vs_h = v_scale + (size_t)kv_h * max_ctx;

    __shared__ union {
        struct {
            _Float16 k[2][16][136];
            _Float16 vt[2][128][24];
        } kv;
        float o[2][16][128];
    } sm_u;

    int local_tid = g * 32 + lane;
    for (int tile_base = tile_s; tile_base < tile_e; tile_base += 32) {
        int tile = tile_base + tg * 16;
        int n_kv = (tile < tile_e) ? min(16, tile_e - tile) : 0;
        if (n_kv > 0) {
            for (int idx = local_tid; idx < 128; idx += 96) {
                int tk = idx >> 3;
                int d16 = (idx & 7) * 16;
                if (tk < n_kv) {
                    int kv_slot = tile + tk;
                    uint4 kq = *reinterpret_cast<const uint4*>(kc_h + (size_t)kv_slot * dh + d16);
                    uint4 vq = *reinterpret_cast<const uint4*>(vc_h + (size_t)kv_slot * dh + d16);
                    float ks = __half2float(ks_h[kv_slot]);
                    float vs = __half2float(vs_h[kv_slot]);
                    uint32_t kw[4] = {kq.x, kq.y, kq.z, kq.w};
                    uint32_t vw[4] = {vq.x, vq.y, vq.z, vq.w};
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        int d0 = d16 + w * 4;
                        sm_u.kv.k[tg][tk][d0 + 0] = (_Float16)((float)(int8_t)(kw[w] & 0xFF) * ks);
                        sm_u.kv.k[tg][tk][d0 + 1] = (_Float16)((float)(int8_t)((kw[w] >> 8) & 0xFF) * ks);
                        sm_u.kv.k[tg][tk][d0 + 2] = (_Float16)((float)(int8_t)((kw[w] >> 16) & 0xFF) * ks);
                        sm_u.kv.k[tg][tk][d0 + 3] = (_Float16)((float)(int8_t)(kw[w] >> 24) * ks);
                        sm_u.kv.vt[tg][d0 + 0][tk] = (_Float16)((float)(int8_t)(vw[w] & 0xFF) * vs);
                        sm_u.kv.vt[tg][d0 + 1][tk] = (_Float16)((float)(int8_t)((vw[w] >> 8) & 0xFF) * vs);
                        sm_u.kv.vt[tg][d0 + 2][tk] = (_Float16)((float)(int8_t)((vw[w] >> 16) & 0xFF) * vs);
                        sm_u.kv.vt[tg][d0 + 3][tk] = (_Float16)((float)(int8_t)(vw[w] >> 24) * vs);
                    }
                } else {
                    *reinterpret_cast<uint4*>(&sm_u.kv.k[tg][tk][d16]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&sm_u.kv.k[tg][tk][d16 + 8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int j = 0; j < 16; ++j) sm_u.kv.vt[tg][d16 + j][tk] = (_Float16)0.0f;
                }
            }
        }
        __syncthreads();

        if (n_kv > 0) {
            q8kv_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                q8kv_h8_v kv_v;
                int dbase = db * 16 + half * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) kv_v[j] = sm_u.kv.k[tg][col][dbase + j];
                s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
            }

            float local_max = -1e30f;
            float sc_arr[8];
            bool tile_all_far = ((pos0 + q_off) - (tile + n_kv - 1) >= rel_extent) && (tile + n_kv - 1 <= pos0 + q_off);
            if (tile_all_far) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int tk = half * 8 + i;
                    if (vq && tk < n_kv) {
                        float sc = s_v[i] - sink_q;
                        sc_arr[i] = sc;
                        local_max = fmaxf(local_max, sc);
                    } else {
                        sc_arr[i] = -1e30f;
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int tk = half * 8 + i;
                    int t = tile + tk;
                    if (vq && tk < n_kv && t <= pos_q) {
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
            }
            float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
            float m_new = fmaxf(m_q, tile_max);
            float alpha = (m_q > -1e20f) ? expf(m_q - m_new) : 0.0f;

            q8kv_h8_v p_reg;
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
                q8kv_h8_v vt_v;
                int d_row = db * 16 + col;
#pragma unroll
                for (int j = 0; j < 8; ++j) vt_v[j] = sm_u.kv.vt[tg][d_row][half * 8 + j];
                o_acc[db] *= alpha;
                o_acc[db] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_v, p_reg, o_acc[db]);
            }
        }
        __syncthreads();
    }

    __shared__ float sm_m[2][3][16];
    __shared__ float sm_l[2][3][16];
    if (half == 0 && vq) {
        sm_m[tg][g][col] = m_q;
        sm_l[tg][g][col] = l_q;
    }

    for (int gg = 0; gg < 3; ++gg) {
        __syncthreads();
        if (g == gg && vq) {
#pragma unroll
            for (int db = 0; db < 8; ++db) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sm_u.o[tg][col][db * 16 + half * 8 + i] = o_acc[db][i];
                }
            }
        }
        __syncthreads();
        int h_out = kv_h * 3 + gg;
        for (int bt = wid; bt < n_q_tile; bt += 6) {
            float m0 = sm_m[0][gg][bt];
            float m1 = sm_m[1][gg][bt];
            float gm = fmaxf(m0, m1);
            float sc0 = (m0 > -1e20f) ? expf(m0 - gm) : 0.0f;
            float sc1 = (m1 > -1e20f) ? expf(m1 - gm) : 0.0f;
            float gl = sm_l[0][gg][bt] * sc0 + sm_l[1][gg][bt] * sc1;
            float* dst = split_buf + (((size_t)h_out * n_splits + sp) * b_stride + (q_off + bt)) * 130;
            if (lane == 0) { dst[0] = gm; dst[1] = gl; }
#pragma unroll
            for (int blk = 0; blk < 4; ++blk) {
                int d = blk * 32 + lane;
                dst[2 + d] = sm_u.o[0][bt][d] * sc0 + sm_u.o[1][bt][d] * sc1;
            }
        }
    }
}

extern "C" __global__ void attn_b48_gqa3_flash_split_q8kv_k(const float* __restrict__ Q,
                                                            const float* __restrict__ R,
                                                            const float* __restrict__ sinks,
                                                            const float* __restrict__ rel_bias,
                                                            const int8_t* __restrict__ k_cache,
                                                            const int8_t* __restrict__ v_cache,
                                                            const __half* __restrict__ k_scale,
                                                            const __half* __restrict__ v_scale,
                                                            const int* __restrict__ d_pos0,
                                                            float* __restrict__ split_buf,
                                                            int n_head, int n_head_kv,
                                                            int dh, int d_rel,
                                                            int rel_extent, int max_ctx, int B, int n_splits) {
    int kv_h = blockIdx.x;
    int sp = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int qsub = wid / 3;
    int g = wid % 3;
    int h = kv_h * 3 + g;
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    int q_idx = blockIdx.z * 64 + qsub * 16 + col;
    bool vq = (q_idx < B);
    int pos_q = pos0 + q_idx;
    int t_max = pos0 + B;
    int b_stride = (B > 64) ? 128 : 64;
    int eff_splits = min(n_splits, max(16, (t_max + 255) >> 8));
    if (sp >= eff_splits) return;

    int total_tiles = (t_max + 15) / 16;
    int tiles_per_sp = (total_tiles + eff_splits - 1) / eff_splits;
    int tile_s = sp * tiles_per_sp * 16;
    int tile_e = min(t_max, tile_s + tiles_per_sp * 16);

    if (tile_s >= tile_e) {
        if (half == 0 && vq) {
            float* dst = split_buf + (((size_t)h * n_splits + sp) * b_stride + q_idx) * 130;
            dst[0] = -1e30f;
            dst[1] = 0.0f;
        }
        return;
    }

    float q_scale = rsqrtf((float)dh);
    q8kv_h8_v q_reg[8];
    float4 r_reg0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg3 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sink_q = 0.0f;
    if (vq) {
        const float* qh = Q + (size_t)q_idx * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)q_idx * r_dim + (size_t)h * d_rel);
        r_reg0 = rh4[0]; r_reg1 = rh4[1]; r_reg2 = rh4[2]; r_reg3 = rh4[3];
        sink_q = sinks[(size_t)q_idx * n_head + h];
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = (sp == 0) ? 0.0f : -1e30f;
    float l_q = (sp == 0) ? 1.0f : 0.0f;
    q8kv_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int8_t* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const int8_t* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;
    const __half* ks_h = k_scale + (size_t)kv_h * max_ctx;
    const __half* vs_h = v_scale + (size_t)kv_h * max_ctx;

    __shared__ _Float16 sm_k[32][136];
    __shared__ _Float16 sm_vt[128][32];

    for (int tile_base = tile_s; tile_base < tile_e; tile_base += 32) {
        int n_kv_big = min(32, tile_e - tile_base);
        if (tid < 256) {
            int tk = tid >> 3;
            int d16 = (tid & 7) * 16;
            if (tk < n_kv_big) {
                int kv_slot = tile_base + tk;
                uint4 kq = *reinterpret_cast<const uint4*>(kc_h + (size_t)kv_slot * dh + d16);
                uint4 vq4 = *reinterpret_cast<const uint4*>(vc_h + (size_t)kv_slot * dh + d16);
                float ks = __half2float(ks_h[kv_slot]);
                float vs = __half2float(vs_h[kv_slot]);
                uint32_t kw[4] = {kq.x, kq.y, kq.z, kq.w};
                uint32_t vw[4] = {vq4.x, vq4.y, vq4.z, vq4.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    int d0 = d16 + w * 4;
                    sm_k[tk][d0 + 0] = (_Float16)((float)(int8_t)(kw[w] & 0xFF) * ks);
                    sm_k[tk][d0 + 1] = (_Float16)((float)(int8_t)((kw[w] >> 8) & 0xFF) * ks);
                    sm_k[tk][d0 + 2] = (_Float16)((float)(int8_t)((kw[w] >> 16) & 0xFF) * ks);
                    sm_k[tk][d0 + 3] = (_Float16)((float)(int8_t)(kw[w] >> 24) * ks);
                    sm_vt[d0 + 0][tk] = (_Float16)((float)(int8_t)(vw[w] & 0xFF) * vs);
                    sm_vt[d0 + 1][tk] = (_Float16)((float)(int8_t)((vw[w] >> 8) & 0xFF) * vs);
                    sm_vt[d0 + 2][tk] = (_Float16)((float)(int8_t)((vw[w] >> 16) & 0xFF) * vs);
                    sm_vt[d0 + 3][tk] = (_Float16)((float)(int8_t)(vw[w] >> 24) * vs);
                }
            } else {
                *reinterpret_cast<uint4*>(&sm_k[tk][d16]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sm_k[tk][d16 + 8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 16; ++j) sm_vt[d16 + j][tk] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        for (int sub = 0; sub < 2; ++sub) {
            int sub_off = sub * 16;
            int n_kv = min(16, n_kv_big - sub_off);
            if (n_kv <= 0) break;
            int tile = tile_base + sub_off;
            q8kv_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                q8kv_h8_v kv_v;
                int dbase = db * 16 + half * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) kv_v[j] = sm_k[sub_off + col][dbase + j];
                s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
            }

            float local_max = -1e30f;
            float sc_arr[8];
            bool tile_all_far = (pos0 - (tile + n_kv - 1) >= rel_extent) && (tile + n_kv - 1 <= pos0);
            if (tile_all_far) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int tk = half * 8 + i;
                    if (vq && tk < n_kv) {
                        float sc = s_v[i] - sink_q;
                        sc_arr[i] = sc;
                        local_max = fmaxf(local_max, sc);
                    } else {
                        sc_arr[i] = -1e30f;
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int tk = half * 8 + i;
                    int t = tile + tk;
                    if (vq && tk < n_kv && t <= pos_q) {
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
            }
            float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
            float m_new = fmaxf(m_q, tile_max);
            float alpha = (m_q > -1e20f) ? expf(m_q - m_new) : 0.0f;

            q8kv_h8_v p_reg;
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
                q8kv_h8_v vt_v;
                int d_row = db * 16 + col;
#pragma unroll
                for (int j = 0; j < 8; ++j) vt_v[j] = sm_vt[d_row][sub_off + half * 8 + j];
                o_acc[db] *= alpha;
                o_acc[db] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_v, p_reg, o_acc[db]);
            }
        }
        __syncthreads();
    }

    if (vq) {
        float* dst = split_buf + (((size_t)h * n_splits + sp) * b_stride + q_idx) * 130;
        if (half == 0) {
            dst[0] = m_q;
            dst[1] = l_q;
        }
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dst[2 + db * 16 + half * 8 + i] = o_acc[db][i];
            }
        }
    }
}

extern "C" __global__ void attn_b8_flash_reduce_q8kv_k(const float* __restrict__ split_buf,
                                                       const float* __restrict__ gate,
                                                       const int* __restrict__ d_pos0,
                                                       q8kv_q8_1_blk* __restrict__ out_q8,
                                                       int n_head, int B, int b_stride, int n_splits) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = wid * 32 + lane;
    int q_dim = n_head * 128;
    int eff_splits = min(n_splits, max(16, (*d_pos0 + B + 255) >> 8));

    float gm = -1e30f;
    for (int sp = 0; sp < eff_splits; ++sp) {
        const float* p = split_buf + (((size_t)h * n_splits + sp) * b_stride + bt) * 130;
        gm = fmaxf(gm, p[0]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    for (int sp = 0; sp < eff_splits; ++sp) {
        const float* p = split_buf + (((size_t)h * n_splits + sp) * b_stride + bt) * 130;
        if (p[0] > -1e20f) {
            float sc = expf(p[0] - gm);
            gl += p[1] * sc;
            go += p[2 + d] * sc;
        }
    }
    float v = (go / gl) / (1.0f + expf(-gate[(size_t)bt * q_dim + (size_t)h * 128 + d]));
    __shared__ int8_t smq[4][32];
    size_t qbase = ((size_t)bt * n_head + h) * 4;
    q8kv_store_q8_1(v, lane, out_q8 + qbase + wid, smq[wid]);
}

__device__ float g_q8kv_b1_split_buf[32 * 64 * 130];

extern "C" __global__ void attn_decode_gqa_splitkv_q8kv_k(const float* __restrict__ Q,
                                                          const float* __restrict__ R,
                                                          const float* __restrict__ sinks,
                                                          const float* __restrict__ rel_bias,
                                                          const int8_t* __restrict__ k_cache,
                                                          const int8_t* __restrict__ v_cache,
                                                          const __half* __restrict__ k_scale,
                                                          const __half* __restrict__ v_scale,
                                                          const int* __restrict__ d_pos,
                                                          float* __restrict__ split_buf,
                                                          int n_head, int n_head_kv,
                                                          int dh, int d_rel,
                                                          int rel_extent, int max_ctx, int n_splits) {
    int kv_h = blockIdx.x;
    int sp = blockIdx.y;
    int h0 = kv_h * 3;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int pos = *d_pos;
    int t_end = pos + 1;
    int chunk = (t_end + n_splits - 1) / n_splits;
    int ts = sp * chunk;
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

    float q_scale = rsqrtf((float)dh);
    float q0[3], q1[3], q2[3], q3[3];
    float rv[3], sk[3];
    float m[3], l[3], o0[3], o1[3], o2[3], o3[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h0 + g;
        float4 qv = reinterpret_cast<const float4*>(Q + (size_t)h * dh)[lane];
        q0[g] = qv.x * q_scale;
        q1[g] = qv.y * q_scale;
        q2[g] = qv.z * q_scale;
        q3[g] = qv.w * q_scale;
        rv[g] = (lane < d_rel) ? R[(size_t)h * d_rel + lane] : 0.0f;
        sk[g] = sinks[h];
        m[g] = (sp == 0 && wid == 0) ? 0.0f : -1e30f;
        l[g] = (sp == 0 && wid == 0) ? 1.0f : 0.0f;
        o0[g] = 0.0f; o1[g] = 0.0f; o2[g] = 0.0f; o3[g] = 0.0f;
    }

    const int8_t* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const int8_t* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;
    const __half* ks_h = k_scale + (size_t)kv_h * max_ctx;
    const __half* vs_h = v_scale + (size_t)kv_h * max_ctx;

    for (int t = ts + wid; t < te; t += 8) {
        uint32_t k4 = reinterpret_cast<const uint32_t*>(kc_h + (size_t)t * dh)[lane];
        uint32_t v4 = reinterpret_cast<const uint32_t*>(vc_h + (size_t)t * dh)[lane];
        float ks = __half2float(ks_h[t]);
        float vs = __half2float(vs_h[t]);
        float k0 = (float)(int8_t)(k4 & 0xFF);
        float k1 = (float)(int8_t)((k4 >> 8) & 0xFF);
        float k2 = (float)(int8_t)((k4 >> 16) & 0xFF);
        float k3 = (float)(int8_t)(k4 >> 24);
        float v0 = (float)(int8_t)(v4 & 0xFF);
        float v1 = (float)(int8_t)((v4 >> 8) & 0xFF);
        float v2 = (float)(int8_t)((v4 >> 16) & 0xFF);
        float v3 = (float)(int8_t)(v4 >> 24);
        int dist = pos - t;
        bool has_rb = (dist < rel_extent);
        float rb_l = (has_rb && lane < d_rel) ? rel_bias[(size_t)dist * d_rel + lane] : 0.0f;
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = fmaf(q0[g], k0, fmaf(q1[g], k1, fmaf(q2[g], k2, q3[g] * k3)));
            float score = __shfl(q8kv_warp_sum32(dot), 0, 32) * ks;
            if (has_rb) score += __shfl(q8kv_warp_sum32(rb_l * rv[g]), 0, 32);
            score -= sk[g];
            float m_new = fmaxf(m[g], score);
            float alpha = (m[g] > -1e20f) ? expf(m[g] - m_new) : 0.0f;
            float p = expf(score - m_new);
            l[g] = l[g] * alpha + p;
            m[g] = m_new;
            float pvs = p * vs;
            o0[g] = fmaf(pvs, v0, o0[g] * alpha);
            o1[g] = fmaf(pvs, v1, o1[g] * alpha);
            o2[g] = fmaf(pvs, v2, o2[g] * alpha);
            o3[g] = fmaf(pvs, v3, o3[g] * alpha);
        }
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    int d4 = lane * 4;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[wid] = m[g]; sm_l[wid] = l[g]; }
        sm_o[wid][d4 + 0] = o0[g]; sm_o[wid][d4 + 1] = o1[g];
        sm_o[wid][d4 + 2] = o2[g]; sm_o[wid][d4 + 3] = o3[g];
        __syncthreads();
        if (wid == 0) {
            float gm = sm_m[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) gm = fmaxf(gm, sm_m[w]);
            float gl = 0.0f;
            float go0 = 0.0f, go1 = 0.0f, go2 = 0.0f, go3 = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                float sc = (sm_m[w] > -1e20f) ? expf(sm_m[w] - gm) : 0.0f;
                gl += sm_l[w] * sc;
                go0 += sm_o[w][d4 + 0] * sc;
                go1 += sm_o[w][d4 + 1] * sc;
                go2 += sm_o[w][d4 + 2] * sc;
                go3 += sm_o[w][d4 + 3] * sc;
            }
            int h = h0 + g;
            float* dst = split_buf + ((size_t)h * n_splits + sp) * 130;
            if (lane == 0) { dst[0] = gm; dst[1] = gl; }
            dst[2 + d4 + 0] = go0;
            dst[2 + d4 + 1] = go1;
            dst[2 + d4 + 2] = go2;
            dst[2 + d4 + 3] = go3;
        }
        __syncthreads();
    }
}

extern "C" __global__ void attn_decode_splitkv_reduce_q8kv_k(const float* __restrict__ split_buf,
                                                             const float* __restrict__ gate,
                                                             q8kv_q8_1_blk* __restrict__ out_q8,
                                                             int n_splits) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int d = wid * 32 + lane;
    const float* base = split_buf + (size_t)h * n_splits * 130;
    __shared__ float sm_w[96];
    __shared__ float sm_gl;
    if (wid == 0) {
        float lm = -1e30f;
        for (int sp = lane; sp < n_splits; sp += 32) {
            lm = fmaxf(lm, base[(size_t)sp * 130]);
        }
        float gm = __shfl(q8kv_warp_max32(lm), 0, 32);
        float ll = 0.0f;
        for (int sp = lane; sp < n_splits; sp += 32) {
            float ms = base[(size_t)sp * 130];
            float sc = (ms > -1e20f) ? expf(ms - gm) : 0.0f;
            sm_w[sp] = sc;
            ll += base[(size_t)sp * 130 + 1] * sc;
        }
        float gl = q8kv_warp_sum32(ll);
        if (lane == 0) sm_gl = 1.0f / gl;
    }
    __syncthreads();
    float inv_gl = sm_gl;
    float go = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        float w = sm_w[sp];
        if (w != 0.0f) {
            go = fmaf(base[(size_t)sp * 130 + 2 + d], w, go);
        }
    }
    float v = (go * inv_gl) / (1.0f + expf(-gate[(size_t)h * 128 + d]));
    __shared__ int8_t smq[4][32];
    q8kv_store_q8_1(v, lane, out_q8 + h * 4 + wid, smq[wid]);
}

template <bool MULTI>
__global__ void attn_batch_gqa3_flash_q8kv_t(const float* __restrict__ Q,
                                             const float* __restrict__ R,
                                             const float* __restrict__ sinks,
                                             const float* __restrict__ rel_bias,
                                             const int8_t* __restrict__ k_cache,
                                             const int8_t* __restrict__ v_cache,
                                             const __half* __restrict__ k_scale,
                                             const __half* __restrict__ v_scale,
                                             const float* __restrict__ gate,
                                             const int* __restrict__ d_pos0,
                                             const int* __restrict__ d_seq_ids,
                                             float* __restrict__ out,
                                             int n_head, int n_head_kv,
                                             int dh, int d_rel,
                                             int rel_extent, int max_ctx, int B) {
    int kv_h = blockIdx.x;
    int qb = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int qsub = wid / 3;
    int g = wid % 3;
    int h = kv_h * 3 + g;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;
    int pos0;
    if constexpr (MULTI) {
        int tok_off = blockIdx.z * B;
        int seq = d_seq_ids[tok_off];
        pos0 = d_pos0[tok_off];
        Q += (size_t)tok_off * q_dim;
        R += (size_t)tok_off * r_dim;
        sinks += (size_t)tok_off * n_head;
        gate += (size_t)tok_off * q_dim;
        out += (size_t)tok_off * q_dim;
        size_t sc_seq_off = (size_t)seq * n_head_kv * max_ctx;
        k_cache += sc_seq_off * dh;
        v_cache += sc_seq_off * dh;
        k_scale += sc_seq_off;
        v_scale += sc_seq_off;
    } else {
        (void)d_seq_ids;
        pos0 = *d_pos0;
    }

    int q_row = qb * 64 + qsub * 16 + col;
    bool vq = (q_row < B);
    int pos_q = pos0 + q_row;
    float q_scale = rsqrtf((float)dh);

    q8kv_h8_v q_reg[8];
    float4 r_reg0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg3 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sink_q = 0.0f;
    if (vq) {
        const float* qh = Q + (size_t)q_row * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)q_row * r_dim + (size_t)h * d_rel);
        r_reg0 = rh4[0]; r_reg1 = rh4[1]; r_reg2 = rh4[2]; r_reg3 = rh4[3];
        sink_q = sinks[(size_t)q_row * n_head + h];
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = 0.0f;
    float l_q = 1.0f;
    q8kv_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    int block_q_min = pos0 + qb * 64;
    int t_max = min(pos0 + B, block_q_min + 64);
    const int8_t* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const int8_t* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;
    const __half* ks_h = k_scale + (size_t)kv_h * max_ctx;
    const __half* vs_h = v_scale + (size_t)kv_h * max_ctx;

    __shared__ _Float16 sm_k[16][136];
    __shared__ _Float16 sm_vt[128][24];

    for (int tile = 0; tile < t_max; tile += 16) {
        int n_kv = min(16, t_max - tile);
        if (tid < 128) {
            int tk = tid >> 3;
            int d16 = (tid & 7) * 16;
            if (tk < n_kv) {
                int kv_slot = tile + tk;
                uint4 kq = *reinterpret_cast<const uint4*>(kc_h + (size_t)kv_slot * dh + d16);
                uint4 vq4 = *reinterpret_cast<const uint4*>(vc_h + (size_t)kv_slot * dh + d16);
                float ks = __half2float(ks_h[kv_slot]);
                float vs = __half2float(vs_h[kv_slot]);
                uint32_t kw[4] = {kq.x, kq.y, kq.z, kq.w};
                uint32_t vw[4] = {vq4.x, vq4.y, vq4.z, vq4.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    int d0 = d16 + w * 4;
                    sm_k[tk][d0 + 0] = (_Float16)((float)(int8_t)(kw[w] & 0xFF) * ks);
                    sm_k[tk][d0 + 1] = (_Float16)((float)(int8_t)((kw[w] >> 8) & 0xFF) * ks);
                    sm_k[tk][d0 + 2] = (_Float16)((float)(int8_t)((kw[w] >> 16) & 0xFF) * ks);
                    sm_k[tk][d0 + 3] = (_Float16)((float)(int8_t)(kw[w] >> 24) * ks);
                    sm_vt[d0 + 0][tk] = (_Float16)((float)(int8_t)(vw[w] & 0xFF) * vs);
                    sm_vt[d0 + 1][tk] = (_Float16)((float)(int8_t)((vw[w] >> 8) & 0xFF) * vs);
                    sm_vt[d0 + 2][tk] = (_Float16)((float)(int8_t)((vw[w] >> 16) & 0xFF) * vs);
                    sm_vt[d0 + 3][tk] = (_Float16)((float)(int8_t)(vw[w] >> 24) * vs);
                }
            } else {
                *reinterpret_cast<uint4*>(&sm_k[tk][d16]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sm_k[tk][d16 + 8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 16; ++j) sm_vt[d16 + j][tk] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        q8kv_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            q8kv_h8_v kv_v;
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) kv_v[j] = sm_k[col][dbase + j];
            s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
        }

        float local_max = -1e30f;
        float sc_arr[8];
        bool tile_all_far = (block_q_min - (tile + n_kv - 1) >= rel_extent) && (tile + n_kv - 1 <= block_q_min);
        if (tile_all_far) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int tk = half * 8 + i;
                if (vq && tk < n_kv) {
                    float sc = s_v[i] - sink_q;
                    sc_arr[i] = sc;
                    local_max = fmaxf(local_max, sc);
                } else {
                    sc_arr[i] = -1e30f;
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int tk = half * 8 + i;
                int t = tile + tk;
                if (vq && tk < n_kv && t <= pos_q) {
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
        }
        float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
        float m_new = fmaxf(m_q, tile_max);
        float alpha = expf(m_q - m_new);

        q8kv_h8_v p_reg;
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
            q8kv_h8_v vt_v;
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
        float* oh = out + (size_t)q_row * q_dim + (size_t)h * dh;
        const float* gh = gate + (size_t)q_row * q_dim + (size_t)h * dh;
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

extern "C" __global__ void attn_batch_gated_flash_q8kv_k(const float* __restrict__ Q,
                                                         const float* __restrict__ R,
                                                         const float* __restrict__ sinks,
                                                         const float* __restrict__ rel_bias,
                                                         const int8_t* __restrict__ k_cache,
                                                         const int8_t* __restrict__ v_cache,
                                                         const __half* __restrict__ k_scale,
                                                         const __half* __restrict__ v_scale,
                                                         const float* __restrict__ gate,
                                                         const int* __restrict__ d_pos0,
                                                         float* __restrict__ out,
                                                         int n_head, int n_head_kv,
                                                         int dh, int d_rel,
                                                         int rel_extent, int max_ctx, int B) {
    int h = blockIdx.x;
    int qb = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int kv_h = h / (n_head / n_head_kv);
    int pos0 = *d_pos0;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    int q_row = qb * 64 + wid * 16 + col;
    bool vq = (q_row < B);
    int pos_q = pos0 + q_row;
    float q_scale = rsqrtf((float)dh);

    q8kv_h8_v q_reg[8];
    float4 r_reg0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 r_reg3 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sink_q = 0.0f;
    if (vq) {
        const float* qh = Q + (size_t)q_row * q_dim + (size_t)h * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)q_row * r_dim + (size_t)h * d_rel);
        r_reg0 = rh4[0]; r_reg1 = rh4[1]; r_reg2 = rh4[2]; r_reg3 = rh4[3];
        sink_q = sinks[(size_t)q_row * n_head + h];
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = 0.0f;
    float l_q = 1.0f;
    q8kv_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    int t_max = min(pos0 + B, pos0 + qb * 64 + 64);
    const int8_t* kc_h = k_cache + (size_t)kv_h * max_ctx * dh;
    const int8_t* vc_h = v_cache + (size_t)kv_h * max_ctx * dh;
    const __half* ks_h = k_scale + (size_t)kv_h * max_ctx;
    const __half* vs_h = v_scale + (size_t)kv_h * max_ctx;

    __shared__ _Float16 sm_k[16][136];
    __shared__ _Float16 sm_vt[128][24];

    for (int tile = 0; tile < t_max; tile += 16) {
        int n_kv = min(16, t_max - tile);
        for (int idx = tid; idx < 256; idx += 128) {
            int tk = idx >> 4;
            int d8 = (idx & 15) * 8;
            if (tk < n_kv) {
                int kv_slot = tile + tk;
                uint2 kq = *reinterpret_cast<const uint2*>(kc_h + (size_t)kv_slot * dh + d8);
                uint2 vq2 = *reinterpret_cast<const uint2*>(vc_h + (size_t)kv_slot * dh + d8);
                float ks = __half2float(ks_h[kv_slot]);
                float vs = __half2float(vs_h[kv_slot]);
                sm_k[tk][d8 + 0] = (_Float16)((float)(int8_t)(kq.x & 0xFF) * ks);
                sm_k[tk][d8 + 1] = (_Float16)((float)(int8_t)((kq.x >> 8) & 0xFF) * ks);
                sm_k[tk][d8 + 2] = (_Float16)((float)(int8_t)((kq.x >> 16) & 0xFF) * ks);
                sm_k[tk][d8 + 3] = (_Float16)((float)(int8_t)(kq.x >> 24) * ks);
                sm_k[tk][d8 + 4] = (_Float16)((float)(int8_t)(kq.y & 0xFF) * ks);
                sm_k[tk][d8 + 5] = (_Float16)((float)(int8_t)((kq.y >> 8) & 0xFF) * ks);
                sm_k[tk][d8 + 6] = (_Float16)((float)(int8_t)((kq.y >> 16) & 0xFF) * ks);
                sm_k[tk][d8 + 7] = (_Float16)((float)(int8_t)(kq.y >> 24) * ks);
                sm_vt[d8 + 0][tk] = (_Float16)((float)(int8_t)(vq2.x & 0xFF) * vs);
                sm_vt[d8 + 1][tk] = (_Float16)((float)(int8_t)((vq2.x >> 8) & 0xFF) * vs);
                sm_vt[d8 + 2][tk] = (_Float16)((float)(int8_t)((vq2.x >> 16) & 0xFF) * vs);
                sm_vt[d8 + 3][tk] = (_Float16)((float)(int8_t)(vq2.x >> 24) * vs);
                sm_vt[d8 + 4][tk] = (_Float16)((float)(int8_t)(vq2.y & 0xFF) * vs);
                sm_vt[d8 + 5][tk] = (_Float16)((float)(int8_t)((vq2.y >> 8) & 0xFF) * vs);
                sm_vt[d8 + 6][tk] = (_Float16)((float)(int8_t)((vq2.y >> 16) & 0xFF) * vs);
                sm_vt[d8 + 7][tk] = (_Float16)((float)(int8_t)(vq2.y >> 24) * vs);
            } else {
                *reinterpret_cast<uint4*>(&sm_k[tk][d8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 8; ++j) sm_vt[d8 + j][tk] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        q8kv_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            q8kv_h8_v kv_v;
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
            if (vq && tk < n_kv && t <= pos_q) {
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
        float alpha = expf(m_q - m_new);

        q8kv_h8_v p_reg;
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
            q8kv_h8_v vt_v;
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
        float* oh = out + (size_t)q_row * q_dim + (size_t)h * dh;
        const float* gh = gate + (size_t)q_row * q_dim + (size_t)h * dh;
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

extern "C" __global__ void inkling_attn_multi_gated_q8_q8kv_k(const float* __restrict__ Q,
                                                              const float* __restrict__ R,
                                                              const float* __restrict__ sinks,
                                                              const float* __restrict__ rel_bias,
                                                              const int8_t* __restrict__ k_cache_pool,
                                                              const int8_t* __restrict__ v_cache_pool,
                                                              const __half* __restrict__ k_scale_pool,
                                                              const __half* __restrict__ v_scale_pool,
                                                              const float* __restrict__ gate,
                                                              const int* __restrict__ d_seq_ids,
                                                              const int* __restrict__ d_positions,
                                                              q8kv_q8_1_blk* __restrict__ out_q8,
                                                              int n_head, int n_head_kv,
                                                              int dh, int d_rel,
                                                              int rel_extent, int max_ctx) {
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

    int t_end = pos + 1;
    float sink_h = sinks[(size_t)bt * n_head + h];
    size_t seq_sc_stride = (size_t)n_head_kv * max_ctx;
    const int8_t* kc_h = k_cache_pool + ((size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx) * dh;
    const int8_t* vc_h = v_cache_pool + ((size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx) * dh;
    const __half* ks_h = k_scale_pool + (size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx;
    const __half* vs_h = v_scale_pool + (size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx;

    float m = (wid == 0) ? 0.0f : -1e30f;
    float l = (wid == 0) ? 1.0f : 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float4 qv = reinterpret_cast<const float4*>(qs)[lane];
    float q0 = qv.x, q1 = qv.y, q2 = qv.z, q3 = qv.w;
    int d4 = lane * 4;

    for (int t = wid; t < t_end; t += nw) {
        uint32_t k4 = reinterpret_cast<const uint32_t*>(kc_h + (size_t)t * dh)[lane];
        uint32_t v4 = reinterpret_cast<const uint32_t*>(vc_h + (size_t)t * dh)[lane];
        float ks = __half2float(ks_h[t]);
        float vs = __half2float(vs_h[t]);
        float k0 = (float)(int8_t)(k4 & 0xFF);
        float k1 = (float)(int8_t)((k4 >> 8) & 0xFF);
        float k2 = (float)(int8_t)((k4 >> 16) & 0xFF);
        float k3 = (float)(int8_t)(k4 >> 24);
        float dot = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))) * ks;
        float score = __shfl(q8kv_warp_sum32(dot), 0, 32);
        int dist = pos - t;
        if (dist < rel_extent) {
            float rp = (lane < d_rel) ? (rel_bias[(size_t)dist * d_rel + lane] * rs[lane]) : 0.0f;
            score += __shfl(q8kv_warp_sum32(rp), 0, 32);
        }
        score -= sink_h;
        float m_new = fmaxf(m, score);
        float alpha = (m > -1e20f) ? expf(m - m_new) : 0.0f;
        float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;
        float pvs = p * vs;
        o0 = fmaf(pvs, (float)(int8_t)(v4 & 0xFF), o0 * alpha);
        o1 = fmaf(pvs, (float)(int8_t)((v4 >> 8) & 0xFF), o1 * alpha);
        o2 = fmaf(pvs, (float)(int8_t)((v4 >> 16) & 0xFF), o2 * alpha);
        o3 = fmaf(pvs, (float)(int8_t)(v4 >> 24), o3 * alpha);
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    sm_o[wid][d4 + 0] = o0; sm_o[wid][d4 + 1] = o1; sm_o[wid][d4 + 2] = o2; sm_o[wid][d4 + 3] = o3;
    __syncthreads();

    if (wid < 4) {
        int d = wid * 32 + lane;
        float gm = sm_m[0];
        for (int w = 1; w < nw; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f, go = 0.0f;
        for (int w = 0; w < nw; ++w) {
            float sc = (sm_m[w] > -1e20f) ? expf(sm_m[w] - gm) : 0.0f;
            gl += sm_l[w] * sc;
            go += sm_o[w][d] * sc;
        }
        float v = (go / gl) / (1.0f + expf(-gate[(size_t)bt * q_dim + (size_t)h * dh + d]));
        __shared__ int8_t smq[4][32];
        size_t qbase = ((size_t)bt * n_head + h) * 4;
        q8kv_store_q8_1(v, lane, out_q8 + qbase + wid, smq[wid]);
    }
}

extern "C" __global__ void inkling_attn_multi_gqa3_q8_q8kv_k(const float* __restrict__ Q,
                                                             const float* __restrict__ R,
                                                             const float* __restrict__ sinks,
                                                             const float* __restrict__ rel_bias,
                                                             const int8_t* __restrict__ k_cache_pool,
                                                             const int8_t* __restrict__ v_cache_pool,
                                                             const __half* __restrict__ k_scale_pool,
                                                             const __half* __restrict__ v_scale_pool,
                                                             const float* __restrict__ gate,
                                                             const int* __restrict__ d_seq_ids,
                                                             const int* __restrict__ d_positions,
                                                             q8kv_q8_1_blk* __restrict__ out_q8,
                                                             int n_head, int n_head_kv,
                                                             int dh, int d_rel,
                                                             int rel_extent, int max_ctx) {
    int kv_h = blockIdx.x;
    int bt = blockIdx.y;
    int h0 = kv_h * 3;
    int seq = d_seq_ids[bt];
    int pos = d_positions[bt];
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int q_dim = n_head * dh;
    int r_dim = n_head * d_rel;

    float q_scale = rsqrtf((float)dh);
    float q0[3], q1[3], q2[3], q3[3];
    float rv[3], sk[3];
    float m[3], l[3], o0[3], o1[3], o2[3], o3[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = h0 + g;
        float4 qv = reinterpret_cast<const float4*>(Q + (size_t)bt * q_dim + (size_t)h * dh)[lane];
        q0[g] = qv.x * q_scale;
        q1[g] = qv.y * q_scale;
        q2[g] = qv.z * q_scale;
        q3[g] = qv.w * q_scale;
        rv[g] = (lane < d_rel) ? R[(size_t)bt * r_dim + (size_t)h * d_rel + lane] : 0.0f;
        sk[g] = sinks[(size_t)bt * n_head + h];
        m[g] = (wid == 0) ? 0.0f : -1e30f;
        l[g] = (wid == 0) ? 1.0f : 0.0f;
        o0[g] = 0.0f; o1[g] = 0.0f; o2[g] = 0.0f; o3[g] = 0.0f;
    }

    int t_end = pos + 1;
    size_t seq_sc_stride = (size_t)n_head_kv * max_ctx;
    const int8_t* kc_h = k_cache_pool + ((size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx) * dh;
    const int8_t* vc_h = v_cache_pool + ((size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx) * dh;
    const __half* ks_h = k_scale_pool + (size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx;
    const __half* vs_h = v_scale_pool + (size_t)seq * seq_sc_stride + (size_t)kv_h * max_ctx;

    for (int t = wid; t < t_end; t += 8) {
        uint32_t k4 = reinterpret_cast<const uint32_t*>(kc_h + (size_t)t * dh)[lane];
        uint32_t v4 = reinterpret_cast<const uint32_t*>(vc_h + (size_t)t * dh)[lane];
        float ks = __half2float(ks_h[t]);
        float vs = __half2float(vs_h[t]);
        float k0 = (float)(int8_t)(k4 & 0xFF);
        float k1 = (float)(int8_t)((k4 >> 8) & 0xFF);
        float k2 = (float)(int8_t)((k4 >> 16) & 0xFF);
        float k3 = (float)(int8_t)(k4 >> 24);
        float v0 = (float)(int8_t)(v4 & 0xFF);
        float v1 = (float)(int8_t)((v4 >> 8) & 0xFF);
        float v2 = (float)(int8_t)((v4 >> 16) & 0xFF);
        float v3 = (float)(int8_t)(v4 >> 24);
        int dist = pos - t;
        bool has_rb = (dist < rel_extent);
        float rb_l = (has_rb && lane < d_rel) ? rel_bias[(size_t)dist * d_rel + lane] : 0.0f;
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = fmaf(q0[g], k0, fmaf(q1[g], k1, fmaf(q2[g], k2, q3[g] * k3)));
            float score = __shfl(q8kv_warp_sum32(dot), 0, 32) * ks;
            if (has_rb) score += __shfl(q8kv_warp_sum32(rb_l * rv[g]), 0, 32);
            score -= sk[g];
            float m_new = fmaxf(m[g], score);
            float alpha = (m[g] > -1e20f) ? expf(m[g] - m_new) : 0.0f;
            float p = expf(score - m_new);
            l[g] = l[g] * alpha + p;
            m[g] = m_new;
            float pvs = p * vs;
            o0[g] = fmaf(pvs, v0, o0[g] * alpha);
            o1[g] = fmaf(pvs, v1, o1[g] * alpha);
            o2[g] = fmaf(pvs, v2, o2[g] * alpha);
            o3[g] = fmaf(pvs, v3, o3[g] * alpha);
        }
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_o[8][128];
    __shared__ int8_t smq[4][32];
    int d4 = lane * 4;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[wid] = m[g]; sm_l[wid] = l[g]; }
        sm_o[wid][d4 + 0] = o0[g]; sm_o[wid][d4 + 1] = o1[g];
        sm_o[wid][d4 + 2] = o2[g]; sm_o[wid][d4 + 3] = o3[g];
        __syncthreads();
        if (wid < 4) {
            int d = wid * 32 + lane;
            float gm = sm_m[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) gm = fmaxf(gm, sm_m[w]);
            float gl = 0.0f, go = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                float sc = (sm_m[w] > -1e20f) ? expf(sm_m[w] - gm) : 0.0f;
                gl += sm_l[w] * sc;
                go += sm_o[w][d] * sc;
            }
            int h = h0 + g;
            float v = (go / gl) / (1.0f + expf(-gate[(size_t)bt * q_dim + (size_t)h * dh + d]));
            size_t qbase = ((size_t)bt * n_head + h) * 4;
            q8kv_store_q8_1(v, lane, out_q8 + qbase + wid, smq[wid]);
        }
        __syncthreads();
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

extern "C" int ink_attn_b8_flash_q8_q8kv(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const int8_t* kc, const int8_t* vc, const __half* ks, const __half* vs, const float* gate, const int* d_pos0, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int max_ctx, int B) {
    int b_stride = (B > 64) ? 128 : 64;
    int max_sp = (B > 64) ? 48 : 96;
    int n_splits = (max_ctx > 65536) ? max_sp : ((max_ctx > 32768) ? ((64 < max_sp) ? 64 : max_sp) : ((max_ctx > 2048) ? 32 : 8));
    float* sbuf = nullptr;
    (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_q8kv_b8_split_buf));
    if (B > 16 && B <= 128 && n_head == n_head_kv * 3 && dh == 128) {
        int nthreads = (B <= 48) ? 288 : 384;
        attn_b48_gqa3_flash_split_q8kv_k<<<dim3(n_head_kv, n_splits, (B + 63) / 64), dim3(nthreads), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, ks, vs, d_pos0, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx, B, n_splits);
    } else {
        attn_b8_gqa3_flash_split_q8kv_k<<<dim3(n_head_kv, n_splits, (B + 15) / 16), dim3(192), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, ks, vs, d_pos0, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx, B, n_splits);
    }
    attn_b8_flash_reduce_q8kv_k<<<dim3(n_head, B), dim3(128), 0, s>>>(sbuf, gate, d_pos0, (q8kv_q8_1_blk*)out_q8, n_head, B, b_stride, n_splits);
    return 0;
}

extern "C" int ink_attn_decode_gated_q8_q8kv(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const int8_t* kc, const int8_t* vc, const __half* ks, const __half* vs, const float* gate, const int* d_pos, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int max_ctx) {
    int n_splits = (max_ctx > 32768) ? 64 : ((max_ctx > 2048) ? 32 : 8);
    float* sbuf = nullptr;
    (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_q8kv_b1_split_buf));
    attn_decode_gqa_splitkv_q8kv_k<<<dim3(n_head_kv, n_splits), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, ks, vs, d_pos, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx, n_splits);
    attn_decode_splitkv_reduce_q8kv_k<<<dim3(n_head), dim3(128), 0, s>>>(sbuf, gate, (q8kv_q8_1_blk*)out_q8, n_splits);
    return 0;
}

extern "C" int ink_attn_batch_gated_flash_q8kv(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const int8_t* kc, const int8_t* vc, const __half* ks, const __half* vs, const float* gate, const int* d_pos0, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int max_ctx, int B) {
    if (n_head == n_head_kv * 3 && dh == 128) {
        attn_batch_gqa3_flash_q8kv_t<false><<<dim3(n_head_kv, (B + 63) / 64), dim3(384), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, ks, vs, gate, d_pos0, nullptr, out, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx, B);
        return 0;
    }
    attn_batch_gated_flash_q8kv_k<<<dim3(n_head, (B + 63) / 64), dim3(128), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, ks, vs, gate, d_pos0, out, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx, B);
    return 0;
}

extern "C" int ink_attn_multi_gated_flash_q8kv(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const int8_t* kc_pool, const int8_t* vc_pool, const __half* ks_pool, const __half* vs_pool, const float* gate, const int* d_seq_ids, const int* d_positions, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int max_ctx, int n_users, int K) {
    attn_batch_gqa3_flash_q8kv_t<true><<<dim3(n_head_kv, (K + 63) / 64, n_users), dim3(384), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, ks_pool, vs_pool, gate, d_positions, d_seq_ids, out, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx, K);
    return 0;
}

extern "C" int ink_attn_multi_gated_q8_q8kv(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const int8_t* kc_pool, const int8_t* vc_pool, const __half* ks_pool, const __half* vs_pool, const float* gate, const int* d_seq_ids, const int* d_positions, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int max_ctx, int B) {
    if (n_head == n_head_kv * 3 && dh == 128 && B >= 8) {
        inkling_attn_multi_gqa3_q8_q8kv_k<<<dim3(n_head_kv, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, ks_pool, vs_pool, gate, d_seq_ids, d_positions, (q8kv_q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx);
        return 0;
    }
    inkling_attn_multi_gated_q8_q8kv_k<<<dim3(n_head, B), dim3(256), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, ks_pool, vs_pool, gate, d_seq_ids, d_positions, (q8kv_q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, max_ctx);
    return 0;
}
