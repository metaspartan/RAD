#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

extern "C" __global__ void gemm_f16_bias_k(const __half* __restrict__ W,
                                           const float* __restrict__ bias,
                                           const float* __restrict__ X,
                                           float* __restrict__ Y,
                                           int N, int K, int B) {
    int col = blockIdx.x * 16 + (threadIdx.x & 15);
    int row = blockIdx.y * 16 + (threadIdx.x >> 4);
    __shared__ float smA[16][16];
    __shared__ float smB[16][16];
    int tx = threadIdx.x & 15;
    int ty = threadIdx.x >> 4;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        smA[ty][tx] = (row < B && (k0 + tx) < K) ? X[(size_t)row * K + (k0 + tx)] : 0.0f;
        smB[ty][tx] = (col < N && (k0 + ty) < K) ? __half2float(W[(size_t)col * K + (k0 + ty)]) : 0.0f;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc = fmaf(smA[ty][i], smB[i][tx], acc);
        }
        __syncthreads();
    }
    if (row < B && col < N) {
        if (bias != nullptr) acc += bias[col];
        Y[(size_t)row * N + col] = acc;
    }
}

extern "C" __global__ void layernorm_f32_k(const float* __restrict__ X,
                                           const float* __restrict__ W,
                                           const float* __restrict__ B,
                                           float* __restrict__ Y,
                                           int N, float eps) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = X + (size_t)r * N;
    float* yr = Y + (size_t)r * N;
    __shared__ float sm_sum[256];
    __shared__ float sm_sq[256];
    float s = 0.0f;
    float sq = 0.0f;
    for (int i = tid; i < N; i += 256) {
        float v = xr[i];
        s += v;
        sq += v * v;
    }
    sm_sum[tid] = s;
    sm_sq[tid] = sq;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            sm_sum[tid] += sm_sum[tid + st];
            sm_sq[tid] += sm_sq[tid + st];
        }
        __syncthreads();
    }
    float mean = sm_sum[0] / (float)N;
    float var = fmaxf(0.0f, sm_sq[0] / (float)N - mean * mean);
    float inv = rsqrtf(var + eps);
    for (int i = tid; i < N; i += 256) {
        float v = (xr[i] - mean) * inv * W[i];
        if (B != nullptr) v += B[i];
        yr[i] = v;
    }
}

extern "C" __global__ void gelu_f32_k(float* __restrict__ X, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = X[i];
        X[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
}

extern "C" __global__ void ls_res_add_f32_k(float* __restrict__ X,
                                            const float* __restrict__ delta,
                                            const float* __restrict__ ls_w,
                                            int N, int rows) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (c < N && r < rows) {
        size_t idx = (size_t)r * N + c;
        X[idx] += delta[idx] * ls_w[c];
    }
}

extern "C" __global__ void vit_attn_f32_k(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O,
                                          int seq_len, int n_head, int dh) {
    int h = blockIdx.x;
    int tq = blockIdx.y;
    int lane = threadIdx.x;
    int dim = n_head * dh;
    const float* qp = Q + (size_t)tq * dim + h * dh;
    float q0 = qp[lane];
    float q1 = qp[lane + 32];
    float scale = rsqrtf((float)dh);
    float m = -1e30f;
    float l = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;
    for (int tk = 0; tk < seq_len; ++tk) {
        const float* kp = K + (size_t)tk * dim + h * dh;
        const float* vp = V + (size_t)tk * dim + h * dh;
        float dot = q0 * kp[lane] + q1 * kp[lane + 32];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor(dot, off, 32);
        }
        float s = dot * scale;
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float p = expf(s - m_new);
        o0 = o0 * alpha + p * vp[lane];
        o1 = o1 * alpha + p * vp[lane + 32];
        l = l * alpha + p;
        m = m_new;
    }
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float* op = O + (size_t)tq * dim + h * dh;
    op[lane] = o0 * inv_l;
    op[lane + 32] = o1 * inv_l;
}

extern "C" __global__ void vit_merge_2x2_k(const float* __restrict__ X,
                                           float* __restrict__ Y,
                                           int grid_h, int grid_w, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int out_idx = blockIdx.y;
    int out_w = grid_w / 2;
    int oh = out_idx / out_w;
    int ow = out_idx % out_w;
    if (c >= C) return;
    float* yp = Y + (size_t)out_idx * (4 * C);
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int in_idx = (oh * 2 + dy) * grid_w + (ow * 2 + dx);
            yp[(dy * 2 + dx) * C + c] = X[(size_t)in_idx * C + c];
        }
    }
}

extern "C" int ink_gemm_f16_bias(hipStream_t s, const __half* W, const float* bias, const float* X, float* Y, int N, int K, int B) {
    dim3 grid((N + 15) / 16, (B + 15) / 16);
    gemm_f16_bias_k<<<grid, dim3(256), 0, s>>>(W, bias, X, Y, N, K, B);
    return 0;
}

extern "C" int ink_layernorm_f32(hipStream_t s, const float* X, const float* W, const float* B, float* Y, int N, int rows, float eps) {
    layernorm_f32_k<<<dim3(rows), dim3(256), 0, s>>>(X, W, B, Y, N, eps);
    return 0;
}

extern "C" int ink_gelu_f32(hipStream_t s, float* X, size_t n) {
    gelu_f32_k<<<dim3((unsigned int)((n + 255) / 256)), dim3(256), 0, s>>>(X, n);
    return 0;
}

extern "C" int ink_ls_res_add_f32(hipStream_t s, float* X, const float* delta, const float* ls_w, int N, int rows) {
    ls_res_add_f32_k<<<dim3((N + 255) / 256, rows), dim3(256), 0, s>>>(X, delta, ls_w, N, rows);
    return 0;
}

extern "C" int ink_vit_attn_f32(hipStream_t s, const float* Q, const float* K, const float* V, float* O, int seq_len, int n_head, int dh) {
    vit_attn_f32_k<<<dim3(n_head, seq_len), dim3(32), 0, s>>>(Q, K, V, O, seq_len, n_head, dh);
    return 0;
}

extern "C" int ink_vit_merge_2x2(hipStream_t s, const float* X, float* Y, int grid_h, int grid_w, int C) {
    int out_tokens = (grid_h / 2) * (grid_w / 2);
    vit_merge_2x2_k<<<dim3((C + 255) / 256, out_tokens), dim3(256), 0, s>>>(X, Y, grid_h, grid_w, C);
    return 0;
}
