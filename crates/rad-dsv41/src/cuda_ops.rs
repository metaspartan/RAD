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

    pub fn rad_cuda_exl3_mma_f16(
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

    pub fn rad_cuda_exl3_moe_mma_f16(
        x_in: *const c_void,
        trellis_ptrs: *const *const c_void,
        suh_ptrs: *const *const c_void,
        svh_ptrs: *const *const c_void,
        expert_ids: *const i32,
        z_scratch_f32: *mut c_void,
        y_out: *mut c_void,
        top_k: i32,
        m: i32,
        k: i32,
        n: i32,
        bits: i32,
        wht_scale: f32,
        x_per_expert: i32,
        expert_start: i32,
        n_local_experts: i32,
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

    pub fn rad_cuda_rms_norm_f32_bf16w_f16(
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
        m: i32,
        top_k: i32,
        d: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_embed_to_mhc_f16(
        table: *const c_void,
        tok_ids: *const i32,
        streams: *mut c_void,
        m: i32,
        d: i32,
        stream: CudaStream,
    );

    #[allow(dead_code)]
    pub fn rad_cuda_mhc_collapse_mean_f16(
        streams: *const c_void,
        out: *mut c_void,
        m: i32,
        d: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_kv_scatter_f16(
        k_in: *const c_void,
        v_in: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        m: i32,
        start_pos: i32,
        max_ctx: i32,
        head_dim: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_rope_yarn_f16(
        q: *mut c_void,
        k: *mut c_void,
        m: i32,
        n_q_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        start_pos: i32,
        theta: f32,
        orig_len: i32,
        sign: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mhc_sinkhorn_mix_f16(
        streams: *mut c_void,
        branch_out: *const c_void,
        peer_branch_out: *const c_void,
        h_coeff: *const f32,
        m: i32,
        d: i32,
        sinkhorn_iters: i32,
        eps: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mqa_attn_sink_causal_f16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        attn_sink: *const f32,
        out: *mut c_void,
        m: i32,
        n_heads: i32,
        start_pos: i32,
        max_ctx: i32,
        sliding_window: i32,
        head_dim: i32,
        scale: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_engram_gate_f32(
        streams: *mut c_void,
        kv: *const c_void,
        q_weight: *const c_void,
        k_weight: *const c_void,
        m: i32,
        d: i32,
        eps: f32,
        stream: CudaStream,
    );

    #[allow(dead_code)]
    pub fn rad_cuda_mtp_add_proj_to_mhc_f32(
        embed: *const c_void,
        tok_ids: *const i32,
        proj: *const c_void,
        streams: *mut c_void,
        m: i32,
        d: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mhc_collapse_mean_to_f16(
        streams: *const c_void,
        out: *mut c_void,
        m: i32,
        d: i32,
        out_stride: i32,
        out_col_off: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mqa_attn_sink_dspark_f16(
        q: *const c_void,
        k_ring: *const c_void,
        v_ring: *const c_void,
        kv_blk: *const c_void,
        attn_sink: *const f32,
        out: *mut c_void,
        m: i32,
        n_heads: i32,
        q0: i32,
        max_ctx: i32,
        sliding_window: i32,
        head_dim: i32,
        scale: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_markov_bias_add_f16(
        logits_row: *const c_void,
        markov_head: *const c_void,
        markov_embed: *const c_void,
        prev_tok: i32,
        out_row: *mut c_void,
        vocab_local: i32,
        rank: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_wo_a_gather_group_f16(
        attn_out: *const c_void,
        grp_in: *mut c_void,
        m: i32,
        n_heads_local: i32,
        heads_per_grp: i32,
        head_dim: i32,
        gi: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_wo_a_scatter_group_f16(
        grp_out: *const c_void,
        wo_mid: *mut c_void,
        m: i32,
        groups_local: i32,
        o_lora_rank: i32,
        gi: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_moe_route_sqrtsoftplus(
        x: *const c_void,
        gate_w: *const c_void,
        gate_bias: *const c_void,
        raw_scores: *mut c_void,
        sel_scores: *mut c_void,
        topk_ids: *mut c_void,
        topk_weights: *mut c_void,
        m: i32,
        n_experts: i32,
        d: i32,
        top_k: i32,
        scaling_factor: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mhc_pre_and_coeff(
        streams: *const c_void,
        hc_fn: *const c_void,
        hc_base: *const c_void,
        hc_scale: *const c_void,
        prev_pre: *const c_void,
        h_coeff: *mut c_void,
        next_pre: *mut c_void,
        branch_in: *mut c_void,
        m: i32,
        d: i32,
        rms_eps: f32,
        hc_eps: f32,
        stream: CudaStream,
    );

    pub fn rad_cuda_mhc_pre_mix_only(
        streams: *const c_void,
        prev_pre: *const c_void,
        branch_in: *mut c_void,
        m: i32,
        d: i32,
        stream: CudaStream,
    );

    pub fn rad_cuda_argmax_f16(
        logits: *const c_void,
        blk_vals: *mut c_void,
        blk_ids: *mut c_void,
        out_val: *mut c_void,
        out_id: *mut c_void,
        m: i32,
        n: i32,
        stream: CudaStream,
    );
}
