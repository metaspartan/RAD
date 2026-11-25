use core::ffi::c_void;
use rad_cuda::CudaStream;

extern "C" {
    pub fn rad_cuda_wht128_pre_mul_f16(
        x: *const c_void,
        suh: *const c_void,
        y: *mut c_void,
        m: i32,
        k: i32,
        scale: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_exl3_reconstruct_f16(
        out: *mut c_void,
        trellis: *const c_void,
        k: i32,
        n: i32,
        bits: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_exl3_gemv_f16(
        x_rot: *const c_void,
        trellis: *const c_void,
        svh: *const c_void,
        z_scratch_f32: *mut c_void,
        y: *mut c_void,
        m: i32,
        k: i32,
        n: i32,
        bits: i32,
        wht_scale: f32,
        add_to_y: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_exl3_moe_gemv_f16(
        x_in: *const c_void,
        trellis_ptrs: *const *const c_void,
        suh_ptrs: *const *const c_void,
        svh_ptrs: *const *const c_void,
        expert_ids: *const i32,
        y_out: *mut c_void,
        top_k: i32,
        k: i32,
        n: i32,
        bits: i32,
        wht_scale: f32,
        x_per_expert: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_rms_norm_bf16w_f16(
        x: *const c_void,
        w: *const c_void,
        y: *mut c_void,
        m: i32,
        d: i32,
        eps: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_clamped_swiglu_f16(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        n: i32,
        limit: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_moe_weighted_sum_f16(
        expert_outs: *const c_void,
        weights: *const f32,
        shared_out: *const c_void,
        dst: *mut c_void,
        top_k: i32,
        d: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_embed_bf16_to_f16(
        table: *const c_void,
        tok_ids: *const i32,
        out: *mut c_void,
        m: i32,
        d: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_rope_yarn_f16(
        q: *mut c_void,
        k: *mut c_void,
        n_q_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        pos: i32,
        theta: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mhc_sinkhorn_mix_f16(
        streams: *mut c_void,
        branch_out: *const c_void,
        h_coeff: *const f32,
        d: i32,
        sinkhorn_iters: i32,
        eps: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mqa_attn_sink_decode_f16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        attn_sink: *const f32,
        out: *mut c_void,
        n_heads: i32,
        seq_len: i32,
        head_dim: i32,
        scale: f32,
        stream: CudaStream,
    );
}
