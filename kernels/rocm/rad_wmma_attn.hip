#include "rad_wmma_common.cuh"


template <bool MULTI>
__global__ void attn_batch_gqa3_flash_fp16_t(const float* __restrict__ Q,
                                             const float* __restrict__ R,
                                             const float* __restrict__ sinks,
                                             const float* __restrict__ rel_bias,
                                             const __half* __restrict__ k_cache,
                                             const __half* __restrict__ v_cache,
                                             const float* __restrict__ gate,
                                             const int* __restrict__ d_pos0,
                                             const int* __restrict__ d_seq_ids,
                                             float* __restrict__ out,
                                             int n_head, int n_head_kv,
                                             int dh, int d_rel,
                                             int rel_extent, int swa_window,
                                             int is_swa, int max_ctx, int B) {
    int kv_h = blockIdx.x;
    int bt_blk = (gridDim.y - 1 - blockIdx.y) * 64;
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
        size_t kv_seq_off = (size_t)seq * n_head_kv * max_ctx * dh;
        k_cache += kv_seq_off;
        v_cache += kv_seq_off;
    } else {
        (void)d_seq_ids;
        pos0 = *d_pos0;
    }

    int bt_q = bt_blk + qsub * 16 + col;
    bool vq = (bt_q < B);
    int pos_q = pos0 + bt_q;
    int t_start_q = is_swa ? max(0, pos_q - swa_window + 1) : 0;

    int t_min = is_swa ? max(0, pos0 + bt_blk - swa_window + 1) : 0;
    int t_max = pos0 + min(bt_blk + 64, B);

    const float LOG2E = 1.4426950408889634f;
    float q_scale = rsqrtf((float)dh) * LOG2E;
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
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
        const float4* rh4 = reinterpret_cast<const float4*>(R + (size_t)bt_q * r_dim + (size_t)h * d_rel);
        r_reg0 = make_float4(rh4[0].x * LOG2E, rh4[0].y * LOG2E, rh4[0].z * LOG2E, rh4[0].w * LOG2E);
        r_reg1 = make_float4(rh4[1].x * LOG2E, rh4[1].y * LOG2E, rh4[1].z * LOG2E, rh4[1].w * LOG2E);
        r_reg2 = make_float4(rh4[2].x * LOG2E, rh4[2].y * LOG2E, rh4[2].z * LOG2E, rh4[2].w * LOG2E);
        r_reg3 = make_float4(rh4[3].x * LOG2E, rh4[3].y * LOG2E, rh4[3].z * LOG2E, rh4[3].w * LOG2E);
        sink_q = sinks[(size_t)bt_q * n_head + h] * LOG2E;
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
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const _Float16* kc_h = reinterpret_cast<const _Float16*>(k_cache) + (size_t)kv_h * max_ctx * dh;
    const _Float16* vc_h = reinterpret_cast<const _Float16*>(v_cache) + (size_t)kv_h * max_ctx * dh;

    __shared__ _Float16 sm_k[32][136];
    __shared__ _Float16 sm_vt[128][40];

    for (int tile_base = t_min; tile_base < t_max; tile_base += 32) {
        int n_kv_big = min(32, t_max - tile_base);
        if (tid < 256) {
            int tk_l = tid >> 3;
            int d8_l = (tid & 7) * 16;
            if (tk_l < n_kv_big) {
                int kv_slot = is_swa ? ((tile_base + tk_l) % max_ctx) : (tile_base + tk_l);
                const _Float16* kp = kc_h + (size_t)kv_slot * dh + d8_l;
                const _Float16* vp = vc_h + (size_t)kv_slot * dh + d8_l;
                *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l]) = *reinterpret_cast<const uint4*>(kp);
                *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l + 8]) = *reinterpret_cast<const uint4*>(kp + 8);
#pragma unroll
                for (int j = 0; j < 16; ++j) sm_vt[d8_l + j][tk_l] = vp[j];
            } else {
                *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sm_k[tk_l][d8_l + 8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 16; ++j) sm_vt[d8_l + j][tk_l] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        for (int sub = 0; sub < 2; ++sub) {
            int sub_off = sub * 16;
            int n_kv = min(16, n_kv_big - sub_off);
            if (n_kv <= 0) break;
            int tile = tile_base + sub_off;
            wmma_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                wmma_h8_v kv_v;
                int dbase = db * 16 + half * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) kv_v[j] = sm_k[sub_off + col][dbase + j];
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
            bool need_rescale = (m_new > m_q);
            float alpha = need_rescale ? exp2f(m_q - m_new) : 1.0f;

            wmma_h8_v p_reg;
            float local_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float p = (sc_arr[i] > -1e20f) ? exp2f(sc_arr[i] - m_new) : 0.0f;
                p_reg[i] = (_Float16)p;
                local_sum += p;
            }
            l_q = l_q * alpha + local_sum + __shfl_xor(local_sum, 16, 32);
            m_q = m_new;

            if (need_rescale) {
#pragma unroll
                for (int db = 0; db < 8; ++db) o_acc[db] *= alpha;
            }
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                wmma_h8_v vt_v;
                int d_row = db * 16 + col;
#pragma unroll
                for (int j = 0; j < 8; ++j) vt_v[j] = sm_vt[d_row][sub_off + half * 8 + j];
                o_acc[db] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_v, p_reg, o_acc[db]);
            }
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
                oh[d] = (o_acc[db][i] * inv_l) / (1.0f + exp2f(-gh[d] * LOG2E));
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


__device__ float g_b8_split_buf[32 * 32 * 64 * 130];


extern "C" __global__ void attn_b8_gqa3_flash_split_k(const float* __restrict__ Q,
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
    int t_start_q = is_swa ? max(0, pos_q - swa_window + 1) : 0;
    int t_min = is_swa ? max(0, (pos0 + q_off) - swa_window + 1) : 0;
    int t_max = pos0 + q_off + n_q_tile;

    int total_tiles = (t_max - t_min + 15) / 16;
    int tiles_per_sp = (total_tiles + n_splits - 1) / n_splits;
    int tile_s = t_min + sp * tiles_per_sp * 16;
    int tile_e = min(t_max, tile_s + tiles_per_sp * 16);

    if (tile_s >= tile_e) {
        if (tg == 0 && half == 0 && vq) {
            float* dst = split_buf + (((size_t)h * n_splits + sp) * 64 + q_idx) * 130;
            dst[0] = -1e30f;
            dst[1] = 0.0f;
        }
        return;
    }

    float q_scale = rsqrtf((float)dh) * LOG2E;
    wmma_h8_v q_reg[8];
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
        float4 t0 = rh4[0], t1 = rh4[1], t2 = rh4[2], t3 = rh4[3];
        r_reg0 = make_float4(t0.x * LOG2E, t0.y * LOG2E, t0.z * LOG2E, t0.w * LOG2E);
        r_reg1 = make_float4(t1.x * LOG2E, t1.y * LOG2E, t1.z * LOG2E, t1.w * LOG2E);
        r_reg2 = make_float4(t2.x * LOG2E, t2.y * LOG2E, t2.z * LOG2E, t2.w * LOG2E);
        r_reg3 = make_float4(t3.x * LOG2E, t3.y * LOG2E, t3.z * LOG2E, t3.w * LOG2E);
        sink_q = sinks[(size_t)q_idx * n_head + h] * LOG2E;
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = (sp == 0 && tg == 0) ? 0.0f : -1e30f;
    float l_q = (sp == 0 && tg == 0) ? 1.0f : 0.0f;
    wmma_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const _Float16* kc_h = reinterpret_cast<const _Float16*>(k_cache) + (size_t)kv_h * max_ctx * dh;
    const _Float16* vc_h = reinterpret_cast<const _Float16*>(v_cache) + (size_t)kv_h * max_ctx * dh;

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
            for (int idx = local_tid; idx < 256; idx += 96) {
                int tk = idx >> 4;
                int d8 = (idx & 15) * 8;
                if (tk < n_kv) {
                    int kv_slot = is_swa ? ((tile + tk) % max_ctx) : (tile + tk);
                    const _Float16* kp = kc_h + (size_t)kv_slot * dh + d8;
                    const _Float16* vp = vc_h + (size_t)kv_slot * dh + d8;
                    *reinterpret_cast<uint4*>(&sm_u.kv.k[tg][tk][d8]) = *reinterpret_cast<const uint4*>(kp);
#pragma unroll
                    for (int j = 0; j < 8; ++j) sm_u.kv.vt[tg][d8 + j][tk] = vp[j];
                } else {
                    *reinterpret_cast<uint4*>(&sm_u.kv.k[tg][tk][d8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int j = 0; j < 8; ++j) sm_u.kv.vt[tg][d8 + j][tk] = (_Float16)0.0f;
                }
            }
        }
        __syncthreads();

        if (n_kv > 0) {
            wmma_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                wmma_h8_v kv_v;
                int dbase = db * 16 + half * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) kv_v[j] = sm_u.kv.k[tg][col][dbase + j];
                s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
            }

            float local_max = -1e30f;
            float sc_arr[8];
            bool tile_all_far = ((pos0 + q_off) - (tile + n_kv - 1) >= rel_extent) && (tile + n_kv - 1 <= pos0 + q_off) && (!is_swa || tile >= (pos0 + q_off + n_q_tile - 1 - swa_window + 1));
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
            }
            float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
            float m_new = fmaxf(m_q, tile_max);
            bool need_rescale = (m_new > m_q && m_q > -1e20f);
            float alpha = (m_q > -1e20f) ? (need_rescale ? exp2f(m_q - m_new) : 1.0f) : 0.0f;

            wmma_h8_v p_reg;
            float local_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float p = (sc_arr[i] > -1e20f) ? exp2f(sc_arr[i] - m_new) : 0.0f;
                p_reg[i] = (_Float16)p;
                local_sum += p;
            }
            l_q = l_q * alpha + local_sum + __shfl_xor(local_sum, 16, 32);
            m_q = m_new;

            if (need_rescale) {
#pragma unroll
                for (int db = 0; db < 8; ++db) o_acc[db] *= alpha;
            }
#pragma unroll
            for (int db = 0; db < 8; ++db) {
                wmma_h8_v vt_v;
                int d_row = db * 16 + col;
#pragma unroll
                for (int j = 0; j < 8; ++j) vt_v[j] = sm_u.kv.vt[tg][d_row][half * 8 + j];
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
            float sc0 = (m0 > -1e20f) ? exp2f(m0 - gm) : 0.0f;
            float sc1 = (m1 > -1e20f) ? exp2f(m1 - gm) : 0.0f;
            float gl = sm_l[0][gg][bt] * sc0 + sm_l[1][gg][bt] * sc1;
            float* dst = split_buf + (((size_t)h_out * n_splits + sp) * 64 + (q_off + bt)) * 130;
            if (lane == 0) { dst[0] = (gm > -1e20f) ? (gm * 0.6931471805599453f) : -1e30f; dst[1] = gl; }
#pragma unroll
            for (int blk = 0; blk < 4; ++blk) {
                int d = blk * 32 + lane;
                dst[2 + d] = sm_u.o[0][bt][d] * sc0 + sm_u.o[1][bt][d] * sc1;
            }
        }
    }
}


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
        const float* p = split_buf + (((size_t)h * n_splits + sp) * 64 + bt) * 130;
        gm = fmaxf(gm, p[0]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* p = split_buf + (((size_t)h * n_splits + sp) * 64 + bt) * 130;
        if (p[0] > -1e20f) {
            float sc = exp2f((p[0] - gm) * LOG2E);
            gl += p[1] * sc;
            go += p[2 + d] * sc;
        }
    }
    float v = (go / gl) / (1.0f + exp2f(-gate[(size_t)bt * q_dim + (size_t)h * 128 + d] * LOG2E));
    __shared__ int8_t smq[4][32];
    size_t qbase = ((size_t)bt * n_head + h) * 4;
    w_q8_store(v, lane, out_q8 + qbase + wid, smq[wid]);
}


extern "C" int ink_attn_b8_flash_q8(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, void* out_q8, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    if (n_head <= 32) {
        float* sbuf = nullptr;
        (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_b8_split_buf));
        if (n_head == 3 * n_head_kv) {
            int n_splits = (!is_swa && max_ctx > 2048) ? 32 : 8;
            attn_b8_gqa3_flash_split_k<<<dim3(n_head_kv, n_splits, (B + 15) / 16), dim3(192), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, d_pos0, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B, n_splits);
            attn_b8_flash_reduce_q8_k<<<dim3(n_head, B), dim3(128), 0, s>>>(sbuf, gate, (w_q8_1_blk*)out_q8, n_head, n_splits);
            return 0;
        }
        int n_splits = 4;
        attn_b8_flash_split_k<<<dim3(n_head, n_splits), dim3(128), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, d_pos0, sbuf, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B, n_splits);
        attn_b8_flash_reduce_q8_k<<<dim3(n_head, B), dim3(128), 0, s>>>(sbuf, gate, (w_q8_1_blk*)out_q8, n_head, n_splits);
        return 0;
    }
    attn_b8_flash_q8_k<<<dim3(n_head), dim3(128), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, (w_q8_1_blk*)out_q8, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B);
    return 0;
}


extern "C" int ink_attn_batch_gated_flash(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc, const __half* vc, const float* gate, const int* d_pos0, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int B) {
    if (n_head == n_head_kv * 3 && dh == 128) {
        attn_batch_gqa3_flash_fp16_t<false><<<dim3(n_head_kv, (B + 63) / 64), dim3(384), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, nullptr, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B);
        return 0;
    }
    attn_batch_gated_flash_k<<<dim3(n_head, (B + 63) / 64), dim3(128), 0, s>>>(Q, R, sinks, rel_bias, kc, vc, gate, d_pos0, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, B);
    return 0;
}


extern "C" int ink_attn_multi_gated_flash(hipStream_t s, const float* Q, const float* R, const float* sinks, const float* rel_bias, const __half* kc_pool, const __half* vc_pool, const float* gate, const int* d_seq_ids, const int* d_positions, float* out, int n_head, int n_head_kv, int dh, int d_rel, int rel_extent, int swa_window, int is_swa, int max_ctx, int n_users, int K) {
    attn_batch_gqa3_flash_fp16_t<true><<<dim3(n_head_kv, (K + 63) / 64, n_users), dim3(384), 0, s>>>(Q, R, sinks, rel_bias, kc_pool, vc_pool, gate, d_positions, d_seq_ids, out, n_head, n_head_kv, dh, d_rel, rel_extent, swa_window, is_swa, max_ctx, K);
    return 0;
}


__device__ float g_df2_wmma_split_buf[32 * 16 * 8 * 130];


extern "C" __global__ void dspark_attn_wmma_gqa4_split_k(const float* __restrict__ Q,
                                                         const float* __restrict__ K_blk,
                                                         const float* __restrict__ V_blk,
                                                         const __half* __restrict__ kc,
                                                         const __half* __restrict__ vc,
                                                         const int* __restrict__ d_ctx_len,
                                                         float* __restrict__ split_buf,
                                                         int K_len, int n_head, int n_head_kv,
                                                         int dh, int swa_window, int is_swa,
                                                         int max_ctx, int n_splits) {
    int kv_h = blockIdx.x;
    int sp = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int col = lane & 15;
    int half = lane >> 4;
    int h = kv_h * 4 + wid;
    int ctx_len = *d_ctx_len;
    int t_start = is_swa ? max(0, ctx_len - swa_window) : 0;
    int ctx_cnt = ctx_len - t_start;
    int total_kv = ctx_cnt + K_len;

    int total_tiles = (total_kv + 15) / 16;
    int tiles_per_sp = (total_tiles + n_splits - 1) / n_splits;
    int tile_s = sp * tiles_per_sp * 16;
    int tile_e = min(total_kv, tile_s + tiles_per_sp * 16);

    bool vq = (col < K_len);
    if (tile_s >= tile_e) {
        if (half == 0 && vq) {
            float* dst = split_buf + (((size_t)h * n_splits + sp) * 8 + col) * 130;
            dst[0] = -1e30f;
            dst[1] = 0.0f;
        }
        return;
    }

    float q_scale = rsqrtf((float)dh) * LOG2E;
    wmma_h8_v q_reg[8];
    if (vq) {
        const float* qh = Q + ((size_t)col * n_head + h) * dh;
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)(qh[dbase + j] * q_scale);
        }
    } else {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[db][j] = (_Float16)0.0f;
        }
    }

    float m_q = -1e30f;
    float l_q = 0.0f;
    wmma_f8_v o_acc[8];
#pragma unroll
    for (int db = 0; db < 8; ++db) o_acc[db] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const _Float16* kc_h = reinterpret_cast<const _Float16*>(kc) + (size_t)kv_h * max_ctx * dh;
    const _Float16* vc_h = reinterpret_cast<const _Float16*>(vc) + (size_t)kv_h * max_ctx * dh;

    __shared__ union {
        struct {
            _Float16 k[16][136];
            _Float16 vt[128][24];
        } kv;
        float o[4][8][128];
    } sm_u;

    for (int tile = tile_s; tile < tile_e; tile += 16) {
        int n_kv = min(16, tile_e - tile);
        for (int idx = tid; idx < 256; idx += 128) {
            int tk = idx >> 4;
            int d8 = (idx & 15) * 8;
            if (tk < n_kv) {
                int kidx = tile + tk;
                if (kidx < ctx_cnt) {
                    int t = t_start + kidx;
                    int kv_slot = is_swa ? (t % max_ctx) : t;
                    const _Float16* kp = kc_h + (size_t)kv_slot * dh + d8;
                    const _Float16* vp = vc_h + (size_t)kv_slot * dh + d8;
                    *reinterpret_cast<uint4*>(&sm_u.kv.k[tk][d8]) = *reinterpret_cast<const uint4*>(kp);
                    uint4 vq = *reinterpret_cast<const uint4*>(vp);
                    const _Float16* vh = reinterpret_cast<const _Float16*>(&vq);
#pragma unroll
                    for (int j = 0; j < 8; ++j) sm_u.kv.vt[d8 + j][tk] = vh[j];
                } else {
                    int bi = kidx - ctx_cnt;
                    const float* kp = K_blk + ((size_t)bi * n_head_kv + kv_h) * dh + d8;
                    const float* vp = V_blk + ((size_t)bi * n_head_kv + kv_h) * dh + d8;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sm_u.kv.k[tk][d8 + j] = (_Float16)kp[j];
                        sm_u.kv.vt[d8 + j][tk] = (_Float16)vp[j];
                    }
                }
            } else {
                *reinterpret_cast<uint4*>(&sm_u.kv.k[tk][d8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 8; ++j) sm_u.kv.vt[d8 + j][tk] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        wmma_f8_v s_v = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v kv_v;
            int dbase = db * 16 + half * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) kv_v[j] = sm_u.kv.k[col][dbase + j];
            s_v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv_v, q_reg[db], s_v);
        }

        float local_max = -1e30f;
        float sc_arr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int tk = half * 8 + i;
            if (vq && tk < n_kv) {
                sc_arr[i] = s_v[i];
                local_max = fmaxf(local_max, s_v[i]);
            } else {
                sc_arr[i] = -1e30f;
            }
        }
        float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));
        float m_new = fmaxf(m_q, tile_max);
        bool need_rescale = (m_new > m_q && m_q > -1e20f);
        float alpha = (m_q > -1e20f) ? (need_rescale ? exp2f(m_q - m_new) : 1.0f) : 0.0f;

        wmma_h8_v p_reg;
        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float p = (sc_arr[i] > -1e20f) ? exp2f(sc_arr[i] - m_new) : 0.0f;
            p_reg[i] = (_Float16)p;
            local_sum += p;
        }
        l_q = l_q * alpha + local_sum + __shfl_xor(local_sum, 16, 32);
        m_q = m_new;

        if (need_rescale) {
#pragma unroll
            for (int db = 0; db < 8; ++db) o_acc[db] *= alpha;
        }
#pragma unroll
        for (int db = 0; db < 8; ++db) {
            wmma_h8_v vt_v;
            int d_row = db * 16 + col;
#pragma unroll
            for (int j = 0; j < 8; ++j) vt_v[j] = sm_u.kv.vt[d_row][half * 8 + j];
            o_acc[db] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_v, p_reg, o_acc[db]);
        }
        __syncthreads();
    }

    if (vq) {
#pragma unroll
        for (int db = 0; db < 8; ++db) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sm_u.o[wid][col][db * 16 + half * 8 + i] = o_acc[db][i];
            }
        }
        float* dst = split_buf + (((size_t)h * n_splits + sp) * 8 + col) * 130;
        if (half == 0) {
            dst[0] = m_q;
            dst[1] = l_q;
        }
    }
    __builtin_amdgcn_wave_barrier();
    for (int bt = 0; bt < K_len; ++bt) {
        float* dst = split_buf + (((size_t)h * n_splits + sp) * 8 + bt) * 130;
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            int d = blk * 32 + lane;
            dst[2 + d] = sm_u.o[wid][bt][d];
        }
    }
}


extern "C" __global__ void dspark_attn_wmma_reduce_q8_k(const float* __restrict__ split_buf,
                                                        w_q8_1_blk* __restrict__ out_q8,
                                                        int n_head, int n_splits) {
    int h = blockIdx.x;
    int bt = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = wid * 32 + lane;

    float gm = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* p = split_buf + (((size_t)h * n_splits + sp) * 8 + bt) * 130;
        gm = fmaxf(gm, p[0]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* p = split_buf + (((size_t)h * n_splits + sp) * 8 + bt) * 130;
        if (p[0] > -1e20f) {
            float sc = exp2f(p[0] - gm);
            gl += p[1] * sc;
            go += p[2 + d] * sc;
        }
    }
    float v = (gl > 0.0f) ? (go / gl) : 0.0f;
    __shared__ int8_t smq[4][32];
    size_t qbase = ((size_t)bt * n_head + h) * 4;
    w_q8_store(v, lane, out_q8 + qbase + wid, smq[wid]);
}


extern "C" int ink_dspark_attn_wmma_q8(hipStream_t s, const float* Q, const float* K_blk, const float* V_blk, const __half* kc, const __half* vc, const int* d_ctx_len, void* out_q8, int K_len, int n_head, int n_head_kv, int dh, int swa_window, int is_swa, int max_ctx) {
    int n_splits = 16;
    float* sbuf = nullptr;
    (void)hipGetSymbolAddress((void**)&sbuf, HIP_SYMBOL(g_df2_wmma_split_buf));
    dspark_attn_wmma_gqa4_split_k<<<dim3(n_head_kv, n_splits), dim3(128), 0, s>>>(Q, K_blk, V_blk, kc, vc, d_ctx_len, sbuf, K_len, n_head, n_head_kv, dh, swa_window, is_swa, max_ctx, n_splits);
    dspark_attn_wmma_reduce_q8_k<<<dim3(n_head, K_len), dim3(128), 0, s>>>(sbuf, (w_q8_1_blk*)out_q8, n_head, n_splits);
    return 0;
}
