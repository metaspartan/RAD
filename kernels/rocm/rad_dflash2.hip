#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float df2_bf16_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ size_t df2_q4_soa_row_bytes(int nblk) {
    return (size_t)nblk * 16 + (((size_t)nblk * 2 + 15) & ~(size_t)15);
}

__device__ __forceinline__ float df2_warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 32);
    }
    return v;
}

extern "C" __global__ void bf16_to_q4_0_k(const uint16_t* __restrict__ src,
                                          uint8_t* __restrict__ dst,
                                          size_t n_rows, int nblk) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_rows * (size_t)nblk;
    if (idx >= total) return;
    size_t row = idx / (size_t)nblk;
    int b = (int)(idx % (size_t)nblk);
    const uint16_t* sp = src + idx * 32;
    size_t rbytes = df2_q4_soa_row_bytes(nblk);
    uint8_t* drow = dst + row * rbytes;
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float v = df2_bf16_f32(sp[j]);
        vals[j] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    float d_out = amax / 7.5f;
    float inv = (d_out > 0.0f) ? (1.0f / d_out) : 0.0f;
    reinterpret_cast<__half*>(drow + (size_t)nblk * 16)[b] = __float2half(d_out);
    uint8_t* qo = drow + (size_t)b * 16;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int q0 = max(0, min(15, (int)floorf(vals[j] * inv + 8.5f)));
        int q1 = max(0, min(15, (int)floorf(vals[j + 16] * inv + 8.5f)));
        qo[j] = (uint8_t)(q0 | (q1 << 4));
    }
}

extern "C" __global__ void bf16_to_f32_k(const uint16_t* __restrict__ src,
                                         float* __restrict__ dst,
                                         size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = df2_bf16_f32(src[i]);
}

extern "C" __global__ void dflash2_conv_k(const float* __restrict__ hidden,
                                          const float* __restrict__ dynamic,
                                          const float* __restrict__ base_kernel,
                                          float* __restrict__ out,
                                          int n_embd, int block_size,
                                          int group_size, int side) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (c >= n_embd) return;
    int pos_in_blk = bt % block_size;
    int n_groups = n_embd / group_size;
    int g = c / group_size;
    const float* dyn_t = dynamic + (size_t)bt * (size_t)(4 * n_groups) + (size_t)(side * 2 * n_groups);
    const float* base_s = base_kernel + (size_t)(side * 2 * n_embd);
    float w0 = base_s[c] + dyn_t[g];
    float w1 = base_s[n_embd + c] + dyn_t[n_groups + g];
    float v0 = hidden[(size_t)bt * n_embd + c];
    float v1 = (pos_in_blk > 0) ? hidden[(size_t)(bt - 1) * n_embd + c] : 0.0f;
    out[(size_t)bt * n_embd + c] = w0 * v0 + w1 * v1;
}

extern "C" __global__ void dflash2_conv_add_k(const float* __restrict__ hidden,
                                              const float* __restrict__ dynamic,
                                              const float* __restrict__ base_kernel,
                                              float* __restrict__ x_inout,
                                              int n_embd, int block_size,
                                              int group_size, int side) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = blockIdx.y;
    if (c >= n_embd) return;
    int pos_in_blk = bt % block_size;
    int n_groups = n_embd / group_size;
    int g = c / group_size;
    const float* dyn_t = dynamic + (size_t)bt * (size_t)(4 * n_groups) + (size_t)(side * 2 * n_groups);
    const float* base_s = base_kernel + (size_t)(side * 2 * n_embd);
    float w0 = base_s[c] + dyn_t[g];
    float w1 = base_s[n_embd + c] + dyn_t[n_groups + g];
    float v0 = hidden[(size_t)bt * n_embd + c];
    float v1 = (pos_in_blk > 0) ? hidden[(size_t)(bt - 1) * n_embd + c] : 0.0f;
    x_inout[(size_t)bt * n_embd + c] += w0 * v0 + w1 * v1;
}

struct df2_q8_1_blk {
    float d;
    int32_t s;
    int32_t q[8];
};

__device__ __forceinline__ float df2_warp_max32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off, 32));
    return v;
}

__device__ __forceinline__ int32_t df2_warp_sum32_i(int32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off, 32);
    return v;
}

__device__ __forceinline__ int32_t df2_pack4_i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int32_t)(uint8_t)a) | (((int32_t)(uint8_t)b) << 8) | (((int32_t)(uint8_t)c) << 16) | (((int32_t)(uint8_t)d) << 24);
}

extern "C" __global__ void dspark_swiglu_q8_k(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              df2_q8_1_blk* __restrict__ out_q8,
                                              int k) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nblk = k >> 5;
    int b = blockIdx.x * 4 + wid;
    if (b >= nblk) return;
    int t = blockIdx.y;
    size_t idx = (size_t)t * k + (b << 5) + lane;
    float g = gate[idx];
    float v = (g / (1.0f + expf(-g))) * up[idx];
    float amax = __shfl(df2_warp_max32(fabsf(v)), 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    int32_t qsum = df2_warp_sum32_i(qi);
    int q1 = __shfl_down(qi, 1, 32);
    int q2 = __shfl_down(qi, 2, 32);
    int q3 = __shfl_down(qi, 3, 32);
    df2_q8_1_blk* dst = out_q8 + (size_t)t * nblk + b;
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if ((lane & 3) == 0) {
        dst->q[lane >> 2] = df2_pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
    }
}

extern "C" __global__ void dflash2_topk16_k(const float* __restrict__ logits,
                                            int* __restrict__ cand_ids,
                                            float* __restrict__ cand_vals,
                                            int vocab_padded, int vocab_unpadded) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_p = logits + (size_t)row * vocab_padded;

    float best_v[16];
    int best_i[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        best_v[k] = -1e30f;
        best_i[k] = 0;
    }
    for (int i = tid; i < vocab_unpadded; i += 256) {
        float v = row_p[i];
        if (v > best_v[15]) {
            int pos = 15;
            while (pos > 0 && v > best_v[pos - 1]) {
                best_v[pos] = best_v[pos - 1];
                best_i[pos] = best_i[pos - 1];
                --pos;
            }
            best_v[pos] = v;
            best_i[pos] = i;
        }
    }

    __shared__ float sm_v[256 * 16];
    __shared__ int sm_i[256 * 16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        sm_v[tid * 16 + k] = best_v[k];
        sm_i[tid * 16 + k] = best_i[k];
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float mv[16];
            int mi[16];
            int a = tid * 16;
            int b = (tid + stride) * 16;
            int pa = 0, pb = 0;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (sm_v[a + pa] >= sm_v[b + pb]) {
                    mv[k] = sm_v[a + pa];
                    mi[k] = sm_i[a + pa];
                    ++pa;
                } else {
                    mv[k] = sm_v[b + pb];
                    mi[k] = sm_i[b + pb];
                    ++pb;
                }
            }
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                sm_v[a + k] = mv[k];
                sm_i[a + k] = mi[k];
            }
        }
        __syncthreads();
    }

    if (tid < 16) {
        cand_ids[row * 16 + tid] = sm_i[tid];
        cand_vals[row * 16 + tid] = sm_v[tid];
    }
}

extern "C" __global__ void dflash2_select_path_k(const int* __restrict__ cand_ids,
                                                 const float* __restrict__ cand_vals,
                                                 const float* __restrict__ h_proj,
                                                 const uint16_t* __restrict__ pred_cb,
                                                 const uint16_t* __restrict__ succ_cb,
                                                 int* __restrict__ d_draft_toks,
                                                 int K_len, int rank, float softcap) {
    int tid = threadIdx.x;
    int u = blockIdx.x;
    int* out_toks = d_draft_toks + u * K_len;
    const int* cids = cand_ids + u * (K_len - 1) * 16;
    const float* cvals = cand_vals + u * (K_len - 1) * 16;
    const float* hp_base = h_proj + (size_t)u * (K_len - 1) * rank;

    __shared__ float cond_s[256];
    __shared__ float scores_s[16];
    __shared__ int prev_tok;

    if (tid == 0) prev_tok = out_toks[0];
    __syncthreads();

    for (int slot = 1; slot < K_len; ++slot) {
        int pt = prev_tok;
        const float* hp = hp_base + (size_t)(slot - 1) * rank;
        const uint16_t* pb = pred_cb + (size_t)pt * rank;
        if (tid < rank) {
            cond_s[tid] = df2_bf16_f32(pb[tid]) * hp[tid];
        }
        __syncthreads();

        int cand_idx = tid >> 4;
        int lane16 = tid & 15;
        int cid = cids[(slot - 1) * 16 + cand_idx];
        const uint16_t* sb = succ_cb + (size_t)cid * rank;
        float dot = 0.0f;
        for (int r = lane16; r < rank; r += 16) {
            dot = fmaf(df2_bf16_f32(sb[r]), cond_s[r], dot);
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            dot += __shfl_down(dot, off, 16);
        }
        if (lane16 == 0) {
            float sc = cvals[(slot - 1) * 16 + cand_idx] + dot;
            if (softcap > 0.0f) sc = softcap * tanhf(sc / softcap);
            scores_s[cand_idx] = sc;
        }
        __syncthreads();

        if (tid == 0) {
            int best = 0;
            float bv = scores_s[0];
            for (int j = 1; j < 16; ++j) {
                if (scores_s[j] > bv) {
                    bv = scores_s[j];
                    best = j;
                }
            }
            int chosen = cids[(slot - 1) * 16 + best];
            out_toks[slot] = chosen;
            prev_tok = chosen;
        }
        __syncthreads();
    }
}

extern "C" __global__ void bf16_to_q8_0_k(const uint16_t* __restrict__ src,
                                          uint8_t* __restrict__ dst,
                                          size_t n_rows, int nblk) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_rows * (size_t)nblk;
    if (idx >= total) return;
    const uint16_t* sp = src + idx * 32;
    uint8_t* dp = dst + idx * 34;
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float v = df2_bf16_f32(sp[j]);
        vals[j] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    float d = amax / 127.0f;
    float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
    __half dh = __float2half(d);
    memcpy(dp, &dh, 2);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        int q = max(-127, min(127, __float2int_rn(vals[j] * inv)));
        dp[2 + j] = (uint8_t)(int8_t)q;
    }
}

extern "C" int ink_bf16_to_q8_0(hipStream_t s, const uint16_t* src, uint8_t* dst, size_t n_rows, int nblk) {
    size_t total = n_rows * (size_t)nblk;
    bf16_to_q8_0_k<<<dim3((unsigned int)((total + 255) / 256)), dim3(256), 0, s>>>(src, dst, n_rows, nblk);
    return 0;
}

extern "C" int ink_bf16_to_q4_0(hipStream_t s, const uint16_t* src, uint8_t* dst, size_t n_rows, int nblk) {
    size_t total = n_rows * (size_t)nblk;
    bf16_to_q4_0_k<<<dim3((unsigned int)((total + 255) / 256)), dim3(256), 0, s>>>(src, dst, n_rows, nblk);
    return 0;
}

extern "C" int ink_bf16_to_f32(hipStream_t s, const uint16_t* src, float* dst, size_t n) {
    bf16_to_f32_k<<<dim3((unsigned int)((n + 255) / 256)), dim3(256), 0, s>>>(src, dst, n);
    return 0;
}

extern "C" int ink_dflash2_conv(hipStream_t s, const float* hidden, const float* dynamic, const float* base_kernel, float* out, int n_embd, int block_size, int group_size, int side, int n_tokens) {
    dflash2_conv_k<<<dim3((n_embd + 255) / 256, n_tokens), dim3(256), 0, s>>>(hidden, dynamic, base_kernel, out, n_embd, block_size, group_size, side);
    return 0;
}

__device__ float g_df2_topk_v[64 * 16 * 16];
__device__ int g_df2_topk_i[64 * 16 * 16];

extern "C" __global__ void dflash2_topk16_part_k(const float* __restrict__ logits,
                                                 int* __restrict__ part_ids,
                                                 float* __restrict__ part_vals,
                                                 int vocab_padded, int vocab_unpadded, int n_chunks) {
    int row = blockIdx.x;
    int ch = blockIdx.y;
    int tid = threadIdx.x;
    int chunk_sz = (((vocab_unpadded + n_chunks - 1) / n_chunks) + 3) & ~3;
    int i_beg = ch * chunk_sz;
    int i_end = min(vocab_unpadded, i_beg + chunk_sz);
    const float* row_p = logits + (size_t)row * vocab_padded;

    float best_v[16];
    int best_i[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        best_v[k] = -1e30f;
        best_i[k] = 0;
    }
    int i_end4 = i_end & ~3;
    const float4* row_p4 = reinterpret_cast<const float4*>(row_p);
    for (int i = i_beg + tid * 4; i < i_end4; i += 512) {
        float4 v4 = row_p4[i >> 2];
        float vals[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v = vals[j];
            if (v > best_v[15]) {
                int pos = 15;
                while (pos > 0 && v > best_v[pos - 1]) {
                    best_v[pos] = best_v[pos - 1];
                    best_i[pos] = best_i[pos - 1];
                    --pos;
                }
                best_v[pos] = v;
                best_i[pos] = i + j;
            }
        }
    }
    for (int i = i_end4 + tid; i < i_end; i += 128) {
        float v = row_p[i];
        if (v > best_v[15]) {
            int pos = 15;
            while (pos > 0 && v > best_v[pos - 1]) {
                best_v[pos] = best_v[pos - 1];
                best_i[pos] = best_i[pos - 1];
                --pos;
            }
            best_v[pos] = v;
            best_i[pos] = i;
        }
    }

    __shared__ float sm_v[128 * 16];
    __shared__ int sm_i[128 * 16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        sm_v[tid * 16 + k] = best_v[k];
        sm_i[tid * 16 + k] = best_i[k];
    }
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float mv[16];
            int mi[16];
            int a = tid * 16;
            int b = (tid + stride) * 16;
            int pa = 0, pb = 0;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (sm_v[a + pa] >= sm_v[b + pb]) {
                    mv[k] = sm_v[a + pa];
                    mi[k] = sm_i[a + pa];
                    ++pa;
                } else {
                    mv[k] = sm_v[b + pb];
                    mi[k] = sm_i[b + pb];
                    ++pb;
                }
            }
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                sm_v[a + k] = mv[k];
                sm_i[a + k] = mi[k];
            }
        }
        __syncthreads();
    }

    if (tid < 16) {
        int out_off = (row * n_chunks + ch) * 16 + tid;
        part_ids[out_off] = sm_i[tid];
        part_vals[out_off] = sm_v[tid];
    }
}

extern "C" __global__ void dflash2_topk16_merge_k(const int* __restrict__ part_ids,
                                                  const float* __restrict__ part_vals,
                                                  int* __restrict__ cand_ids,
                                                  float* __restrict__ cand_vals,
                                                  int n_chunks) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float sm_v[16 * 16];
    __shared__ int sm_i[16 * 16];
    if (tid < n_chunks * 16) {
        int src_off = row * n_chunks * 16 + tid;
        sm_v[tid] = part_vals[src_off];
        sm_i[tid] = part_ids[src_off];
    }
    __syncthreads();
    for (int stride = n_chunks >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float mv[16];
            int mi[16];
            int a = tid * 16;
            int b = (tid + stride) * 16;
            int pa = 0, pb = 0;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (sm_v[a + pa] >= sm_v[b + pb]) {
                    mv[k] = sm_v[a + pa];
                    mi[k] = sm_i[a + pa];
                    ++pa;
                } else {
                    mv[k] = sm_v[b + pb];
                    mi[k] = sm_i[b + pb];
                    ++pb;
                }
            }
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                sm_v[a + k] = mv[k];
                sm_i[a + k] = mi[k];
            }
        }
        __syncthreads();
    }
    if (tid < 16) {
        cand_ids[row * 16 + tid] = sm_i[tid];
        cand_vals[row * 16 + tid] = sm_v[tid];
    }
}

extern "C" __global__ void dflash2_conv_q8_k(const float* __restrict__ hidden,
                                             const float* __restrict__ dynamic,
                                             const float* __restrict__ base_kernel,
                                             df2_q8_1_blk* __restrict__ out_q8,
                                             int n_embd, int block_size,
                                             int group_size, int side) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nblk = n_embd >> 5;
    int b = blockIdx.x * 4 + wid;
    if (b >= nblk) return;
    int bt = blockIdx.y;
    int c = (b << 5) + lane;
    int pos_in_blk = bt % block_size;
    int n_groups = n_embd / group_size;
    int g = c / group_size;
    const float* dyn_t = dynamic + (size_t)bt * (size_t)(4 * n_groups) + (size_t)(side * 2 * n_groups);
    const float* base_s = base_kernel + (size_t)(side * 2 * n_embd);
    float w0 = base_s[c] + dyn_t[g];
    float w1 = base_s[n_embd + c] + dyn_t[n_groups + g];
    float v0 = hidden[(size_t)bt * n_embd + c];
    float v1 = (pos_in_blk > 0) ? hidden[(size_t)(bt - 1) * n_embd + c] : 0.0f;
    float v = w0 * v0 + w1 * v1;
    float amax = __shfl(df2_warp_max32(fabsf(v)), 0, 32);
    float d = amax / 127.0f;
    float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    int32_t qsum = df2_warp_sum32_i(qi);
    int q1 = __shfl_down(qi, 1, 32);
    int q2 = __shfl_down(qi, 2, 32);
    int q3 = __shfl_down(qi, 3, 32);
    df2_q8_1_blk* dst = out_q8 + (size_t)bt * nblk + b;
    if (lane == 0) {
        dst->d = d;
        dst->s = qsum;
    }
    if ((lane & 3) == 0) {
        dst->q[lane >> 2] = df2_pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
    }
}

extern "C" __global__ void dflash2_rmsnorm_q8_k(const float* __restrict__ x,
                                                const float* __restrict__ w,
                                                float* __restrict__ y_f32,
                                                df2_q8_1_blk* __restrict__ y_q8,
                                                int n_embd, float eps) {
    int bt = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    const float* xr = x + (size_t)bt * n_embd;
    float ss = 0.0f;
    for (int i = tid; i < n_embd; i += 256) {
        float v = xr[i];
        ss += v * v;
    }
    ss = df2_warp_sum32(ss);
    __shared__ float sm_ss[8];
    if (lane == 0) sm_ss[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm_ss[i];
        sm_ss[0] = rsqrtf(tot / (float)n_embd + eps);
    }
    __syncthreads();
    float inv_rms = sm_ss[0];
    int nblk = n_embd >> 5;
    float* yr = y_f32 ? (y_f32 + (size_t)bt * n_embd) : nullptr;
    for (int b = wid; b < nblk; b += 8) {
        int c = (b << 5) + lane;
        float v = xr[c] * inv_rms * w[c];
        if (yr) yr[c] = v;
        float amax = __shfl(df2_warp_max32(fabsf(v)), 0, 32);
        float d = amax / 127.0f;
        float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int qi = __float2int_rn(v * inv);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        int32_t qsum = df2_warp_sum32_i(qi);
        int q1 = __shfl_down(qi, 1, 32);
        int q2 = __shfl_down(qi, 2, 32);
        int q3 = __shfl_down(qi, 3, 32);
        df2_q8_1_blk* dst = y_q8 + (size_t)bt * nblk + b;
        if (lane == 0) {
            dst->d = d;
            dst->s = qsum;
        }
        if ((lane & 3) == 0) {
            dst->q[lane >> 2] = df2_pack4_i8((int8_t)qi, (int8_t)q1, (int8_t)q2, (int8_t)q3);
        }
    }
}

extern "C" int ink_dflash2_topk16(hipStream_t s, const float* logits, int* cand_ids, float* cand_vals, int vocab_padded, int vocab_unpadded, int n_rows) {
    if (n_rows <= 64) {
        int n_chunks = 16;
        int* pids = nullptr;
        float* pvals = nullptr;
        (void)hipGetSymbolAddress((void**)&pids, HIP_SYMBOL(g_df2_topk_i));
        (void)hipGetSymbolAddress((void**)&pvals, HIP_SYMBOL(g_df2_topk_v));
        dflash2_topk16_part_k<<<dim3(n_rows, n_chunks), dim3(128), 0, s>>>(logits, pids, pvals, vocab_padded, vocab_unpadded, n_chunks);
        dflash2_topk16_merge_k<<<dim3(n_rows), dim3(256), 0, s>>>(pids, pvals, cand_ids, cand_vals, n_chunks);
        return 0;
    }
    dflash2_topk16_k<<<dim3(n_rows), dim3(256), 0, s>>>(logits, cand_ids, cand_vals, vocab_padded, vocab_unpadded);
    return 0;
}

extern "C" int ink_dflash2_conv_q8(hipStream_t s, const float* hidden, const float* dynamic, const float* base_kernel, void* out_q8, int n_embd, int block_size, int group_size, int side, int n_tokens) {
    int nblk = n_embd >> 5;
    dflash2_conv_q8_k<<<dim3((nblk + 3) / 4, n_tokens), dim3(128), 0, s>>>(hidden, dynamic, base_kernel, (df2_q8_1_blk*)out_q8, n_embd, block_size, group_size, side);
    return 0;
}

extern "C" int ink_dflash2_rmsnorm_q8(hipStream_t s, const float* x, const float* w, float* y_f32, void* y_q8, int n_embd, int n_tokens, float eps) {
    dflash2_rmsnorm_q8_k<<<dim3(n_tokens), dim3(256), 0, s>>>(x, w, y_f32, (df2_q8_1_blk*)y_q8, n_embd, eps);
    return 0;
}

extern "C" __global__ void gather_dspark_feats_batch_k(const float* __restrict__ bsc_feats,
                                                       float* __restrict__ dst,
                                                       int start_tok, int max_batch,
                                                       int n_embd, int n_tok) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= n_embd || t >= n_tok) return;
    int src_tok = start_tok + t;
    float* dp = dst + (size_t)t * (4 * n_embd);
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        dp[(size_t)f * n_embd + c] = bsc_feats[((size_t)f * max_batch + src_tok) * n_embd + c];
    }
}

extern "C" __global__ void dflash2_rope_norm_ctx_batch_k(const float* __restrict__ K_in,
                                                         const float* __restrict__ V_in,
                                                         const float* __restrict__ k_norm_w,
                                                         __half* __restrict__ kc,
                                                         __half* __restrict__ vc,
                                                         int pos0, int max_ctx,
                                                         int n_head_kv, int dh,
                                                         float freq_base, int is_swa) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int lane = threadIdx.x;
    int pos = pos0 + t;
    int slot = is_swa ? (pos % max_ctx) : pos;
    int kv_dim = n_head_kv * dh;
    const float* kh = K_in + (size_t)t * kv_dim + h * dh;
    const float* vh = V_in + (size_t)t * kv_dim + h * dh;
    __half* kd = kc + ((size_t)h * max_ctx + slot) * dh;
    __half* vd = vc + ((size_t)h * max_ctx + slot) * dh;
    float kv[4];
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int d = lane + p * 32;
        kv[p] = kh[d];
        ss += kv[p] * kv[p];
        vd[d] = __float2half(vh[d]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor(ss, off, 32);
    }
    float inv = rsqrtf(ss / (float)dh + 1e-6f);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int d = lane + p * 32;
        kv[p] = kv[p] * inv * k_norm_w[d];
    }
    int half_dh = dh >> 1;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        int d = lane + p * 32;
        float freq = powf(freq_base, -2.0f * (float)d / (float)dh);
        float ang = (float)pos * freq;
        float ca = cosf(ang);
        float sa = sinf(ang);
        float x0 = kv[p];
        float x1 = kv[p + 2];
        kd[d] = __float2half(x0 * ca - x1 * sa);
        kd[d + half_dh] = __float2half(x0 * sa + x1 * ca);
    }
}

extern "C" int ink_gather_dspark_feats_batch(hipStream_t s, const float* bsc_feats, float* dst, int start_tok, int max_batch, int n_embd, int n_tok) {
    gather_dspark_feats_batch_k<<<dim3((n_embd + 255) / 256, n_tok), dim3(256), 0, s>>>(bsc_feats, dst, start_tok, max_batch, n_embd, n_tok);
    return 0;
}

extern "C" int ink_dflash2_rope_norm_ctx_batch(hipStream_t s, const float* k_in, const float* v_in, const float* k_norm_w, __half* kc, __half* vc, int pos0, int max_ctx, int n_head_kv, int dh, float freq_base, int is_swa, int n_tok) {
    dflash2_rope_norm_ctx_batch_k<<<dim3(n_head_kv, n_tok), dim3(32), 0, s>>>(k_in, v_in, k_norm_w, kc, vc, pos0, max_ctx, n_head_kv, dh, freq_base, is_swa);
    return 0;
}

extern "C" __global__ void dflash2_rope_norm_ctx_batch_dpos_k(const float* __restrict__ K_in,
                                                              const float* __restrict__ V_in,
                                                              const float* __restrict__ k_norm_w,
                                                              __half* __restrict__ kc,
                                                              __half* __restrict__ vc,
                                                              const int* __restrict__ d_pos0, int max_ctx,
                                                              int n_head_kv, int dh,
                                                              float freq_base, int is_swa) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int lane = threadIdx.x;
    int pos = d_pos0[0] + t;
    int slot = is_swa ? (pos % max_ctx) : pos;
    int kv_dim = n_head_kv * dh;
    const float* kh = K_in + (size_t)t * kv_dim + h * dh;
    const float* vh = V_in + (size_t)t * kv_dim + h * dh;
    __half* kd = kc + ((size_t)h * max_ctx + slot) * dh;
    __half* vd = vc + ((size_t)h * max_ctx + slot) * dh;
    float kv[4];
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int d = lane + p * 32;
        kv[p] = kh[d];
        ss += kv[p] * kv[p];
        vd[d] = __float2half(vh[d]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor(ss, off, 32);
    }
    float inv = rsqrtf(ss / (float)dh + 1e-6f);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        int d = lane + p * 32;
        kv[p] = kv[p] * inv * k_norm_w[d];
    }
    int half_dh = dh >> 1;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        int d = lane + p * 32;
        float freq = powf(freq_base, -2.0f * (float)d / (float)dh);
        float ang = (float)pos * freq;
        float ca = cosf(ang);
        float sa = sinf(ang);
        float x0 = kv[p];
        float x1 = kv[p + 2];
        kd[d] = __float2half(x0 * ca - x1 * sa);
        kd[d + half_dh] = __float2half(x0 * sa + x1 * ca);
    }
}

extern "C" int ink_dflash2_rope_norm_ctx_batch_dpos(hipStream_t s, const float* k_in, const float* v_in, const float* k_norm_w, __half* kc, __half* vc, const int* d_pos0, int max_ctx, int n_head_kv, int dh, float freq_base, int is_swa, int n_tok) {
    dflash2_rope_norm_ctx_batch_dpos_k<<<dim3(n_head_kv, n_tok), dim3(32), 0, s>>>(k_in, v_in, k_norm_w, kc, vc, d_pos0, max_ctx, n_head_kv, dh, freq_base, is_swa);
    return 0;
}

extern "C" int ink_dflash2_select_path(hipStream_t s, const int* cand_ids, const float* cand_vals, const float* h_proj, const uint16_t* pred_cb, const uint16_t* succ_cb, int* d_draft_toks, int n_users, int K_len, int rank, float softcap) {
    dflash2_select_path_k<<<dim3(n_users), dim3(256), 0, s>>>(cand_ids, cand_vals, h_proj, pred_cb, succ_cb, d_draft_toks, K_len, rank, softcap);
    return 0;
}

extern "C" int ink_dflash2_conv_add(hipStream_t s, const float* hidden, const float* dynamic, const float* base_kernel, float* x_inout, int n_embd, int block_size, int group_size, int side, int B) {
    dflash2_conv_add_k<<<dim3((n_embd + 255) / 256, B), dim3(256), 0, s>>>(hidden, dynamic, base_kernel, x_inout, n_embd, block_size, group_size, side);
    return 0;
}

extern "C" int ink_dspark_swiglu_q8(hipStream_t s, const float* gate, const float* up, void* out_q8, int k, int B) {
    int nblk = k >> 5;
    dspark_swiglu_q8_k<<<dim3((nblk + 3) / 4, B), dim3(128), 0, s>>>(gate, up, (df2_q8_1_blk*)out_q8, k);
    return 0;
}
