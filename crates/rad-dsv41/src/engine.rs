use anyhow::Result;
use core::ffi::c_void;
use rad_cuda::comm::Tp2Comm;
use rad_cuda::{CudaBuf, CudaFail, CudaStream};
use std::io::Write;
use std::path::Path;
use crate::config::DeepseekV41Config;
use crate::cuda_ops;
use crate::mtp::ngram_draft;
use crate::safetensors::SafetensorsIndex;
use crate::weights::{load_raw_buf_replicated, Exl3Linear, LayerWeights};

pub const MAX_BATCH: usize = 64;

pub struct KvCacheLayer {
    pub k_cache: CudaBuf,
    pub v_cache: CudaBuf,
}

pub struct Dsv41Engine {
    pub cfg: DeepseekV41Config,
    pub rank: usize,
    pub world: usize,
    pub max_ctx: usize,
    pub embed: CudaBuf,
    pub final_norm: CudaBuf,
    pub head: Exl3Linear,
    pub layers: Vec<LayerWeights>,
    pub kv_caches: Vec<KvCacheLayer>,
    pub d_streams: CudaBuf,
    pub d_branch_in: CudaBuf,
    pub d_normed: CudaBuf,
    pub d_xrot: CudaBuf,
    pub d_z_scratch: CudaBuf,
    pub d_q_lora: CudaBuf,
    pub d_q_normed: CudaBuf,
    pub d_q: CudaBuf,
    pub d_kv: CudaBuf,
    pub d_kv_normed: CudaBuf,
    pub d_v_tmp: CudaBuf,
    pub d_attn_out: CudaBuf,
    pub d_wo_grp_in: CudaBuf,
    pub d_wo_grp_out: CudaBuf,
    pub d_wo_mid: CudaBuf,
    pub d_branch_out: CudaBuf,
    pub d_h_coeff: CudaBuf,
    pub d_gate_raw: CudaBuf,
    pub d_gate_sel: CudaBuf,
    pub d_topk_ids: CudaBuf,
    pub d_topk_weights: CudaBuf,
    pub d_moe_gate: CudaBuf,
    pub d_moe_up: CudaBuf,
    pub d_moe_act: CudaBuf,
    pub d_moe_outs: CudaBuf,
    pub d_shared_g: CudaBuf,
    pub d_shared_u: CudaBuf,
    pub d_shared_out: CudaBuf,
    pub d_logits: CudaBuf,
    pub d_tok_ids: CudaBuf,
    pub d_arg_vals: CudaBuf,
    pub d_arg_ids: CudaBuf,
    pub d_best_val: CudaBuf,
    pub d_best_id: CudaBuf,
    pub h_ar_buf: Vec<u16>,
    pub pos: usize,
    pub wht_scale: f32,
}

impl Dsv41Engine {
    pub fn load(
        model_dir: &Path,
        max_ctx: usize,
        comm: &mut Tp2Comm,
        max_layers: Option<usize>,
    ) -> Result<Self> {
        let rank = comm.rank;
        let world = comm.world_size();
        comm.set_nodelay(false);
        let cfg: DeepseekV41Config = if world > 1 && rank != 0 {
            let bytes = comm.recv_bytes()?;
            serde_json::from_slice(&bytes)?
        } else {
            let text = std::fs::read_to_string(model_dir.join("config.json"))?;
            if world > 1 {
                comm.send_bytes(text.as_bytes())?;
            }
            serde_json::from_str(&text)?
        };
        let tc = cfg.text_config.clone();
        let mut idx_opt = if world > 1 && rank != 0 {
            None
        } else {
            Some(SafetensorsIndex::open(model_dir)?)
        };

        let embed = load_raw_buf_replicated(idx_opt.as_mut(), "embed.weight", comm)?;
        let final_norm = load_raw_buf_replicated(idx_opt.as_mut(), "norm.weight", comm)?;
        let head = Exl3Linear::load_col_shard(idx_opt.as_mut(), "head", comm)?;

        let n_layers = max_layers.unwrap_or(tc.num_hidden_layers).min(tc.num_hidden_layers);
        let mut layers = Vec::with_capacity(n_layers);
        let mut kv_caches = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(LayerWeights::load(idx_opt.as_mut(), &cfg, l, comm)?);
            kv_caches.push(KvCacheLayer {
                k_cache: CudaBuf::new(max_ctx * tc.head_dim * 2).map_err(CudaFail)?,
                v_cache: CudaBuf::new(max_ctx * tc.head_dim * 2).map_err(CudaFail)?,
            });
            if rank == 0 && (l == 0 || (l + 1) % 5 == 0 || l + 1 == n_layers) {
                println!("[load] layer {}/{n_layers} ready (world={world})", l + 1);
                let _ = std::io::stdout().flush();
            }
        }

        let d = tc.hidden_size;
        comm.set_nodelay(true);
        let n_heads_local = tc.num_attention_heads / world;
        let groups_local = tc.o_groups / world;
        let heads_per_grp = tc.num_attention_heads / tc.o_groups;
        let vocab_local = tc.vocab_size / world;
        let max_head_m = 16usize;
        let z_elems = (MAX_BATCH * n_heads_local * tc.head_dim)
            .max(MAX_BATCH * tc.num_experts_per_tok * d)
            .max(max_head_m * vocab_local);

        Ok(Self {
            cfg,
            rank,
            world,
            max_ctx,
            embed,
            final_norm,
            head,
            layers,
            kv_caches,
            d_streams: CudaBuf::new(MAX_BATCH * 4 * d * 4).map_err(CudaFail)?,
            d_branch_in: CudaBuf::new(MAX_BATCH * d * 4).map_err(CudaFail)?,
            d_normed: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_xrot: CudaBuf::new(MAX_BATCH * (n_heads_local * tc.head_dim).max(d) * 2).map_err(CudaFail)?,
            d_z_scratch: CudaBuf::new(z_elems * 4).map_err(CudaFail)?,
            d_q_lora: CudaBuf::new(MAX_BATCH * tc.q_lora_rank * 2).map_err(CudaFail)?,
            d_q_normed: CudaBuf::new(MAX_BATCH * tc.q_lora_rank * 2).map_err(CudaFail)?,
            d_q: CudaBuf::new(MAX_BATCH * n_heads_local * tc.head_dim * 2).map_err(CudaFail)?,
            d_kv: CudaBuf::new(MAX_BATCH * tc.head_dim * 2).map_err(CudaFail)?,
            d_kv_normed: CudaBuf::new(MAX_BATCH * tc.head_dim * 2).map_err(CudaFail)?,
            d_v_tmp: CudaBuf::new(MAX_BATCH * tc.head_dim * 2).map_err(CudaFail)?,
            d_attn_out: CudaBuf::new(MAX_BATCH * n_heads_local * tc.head_dim * 2).map_err(CudaFail)?,
            d_wo_grp_in: CudaBuf::new(MAX_BATCH * heads_per_grp * tc.head_dim * 2).map_err(CudaFail)?,
            d_wo_grp_out: CudaBuf::new(MAX_BATCH * tc.o_lora_rank * 2).map_err(CudaFail)?,
            d_wo_mid: CudaBuf::new(MAX_BATCH * groups_local * tc.o_lora_rank * 2).map_err(CudaFail)?,
            d_branch_out: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_h_coeff: CudaBuf::new(MAX_BATCH * 24 * 4).map_err(CudaFail)?,
            d_gate_raw: CudaBuf::new(MAX_BATCH * tc.n_routed_experts * 4).map_err(CudaFail)?,
            d_gate_sel: CudaBuf::new(MAX_BATCH * tc.n_routed_experts * 4).map_err(CudaFail)?,
            d_topk_ids: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * 4).map_err(CudaFail)?,
            d_topk_weights: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * 4).map_err(CudaFail)?,
            d_moe_gate: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_up: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_act: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_outs: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * d * 2).map_err(CudaFail)?,
            d_shared_g: CudaBuf::new(MAX_BATCH * (tc.moe_intermediate_size / world) * 2).map_err(CudaFail)?,
            d_shared_u: CudaBuf::new(MAX_BATCH * (tc.moe_intermediate_size / world) * 2).map_err(CudaFail)?,
            d_shared_out: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_logits: CudaBuf::new(max_head_m * vocab_local * 2).map_err(CudaFail)?,
            d_tok_ids: CudaBuf::new(MAX_BATCH * 4).map_err(CudaFail)?,
            d_arg_vals: CudaBuf::new(max_head_m * 256 * 4).map_err(CudaFail)?,
            d_arg_ids: CudaBuf::new(max_head_m * 256 * 4).map_err(CudaFail)?,
            d_best_val: CudaBuf::new(max_head_m * 4).map_err(CudaFail)?,
            d_best_id: CudaBuf::new(max_head_m * 4).map_err(CudaFail)?,
            h_ar_buf: vec![0u16; MAX_BATCH * d],
            pos: 0,
            wht_scale: 1.0f32 / 128.0f32.sqrt(),
        })
    }

    unsafe fn exl3_proj(
        &self,
        x: *const c_void,
        w: &Exl3Linear,
        y: *mut c_void,
        m: i32,
        add_to_y: i32,
        stream: CudaStream,
    ) {
        cuda_ops::rad_cuda_wht128_pre_mul_f16(x, w.suh.ptr, self.d_xrot.ptr, m, w.k as i32, self.wht_scale, stream);
        cuda_ops::rad_cuda_exl3_mma_f16(
            self.d_xrot.ptr,
            w.trellis.ptr,
            w.svh.ptr,
            self.d_z_scratch.ptr,
            y,
            m,
            w.k as i32,
            w.n as i32,
            w.bits as i32,
            self.wht_scale,
            add_to_y,
            stream,
        );
    }

    fn all_reduce_branch(&mut self, m: usize, comm: &mut Tp2Comm) -> Result<()> {
        if self.world <= 1 {
            return Ok(());
        }
        let d = self.cfg.text_config.hidden_size;
        let elems = m * d;
        rad_cuda::sync_device().map_err(CudaFail)?;
        let raw_out = unsafe { std::slice::from_raw_parts_mut(self.h_ar_buf.as_mut_ptr() as *mut u8, elems * 2) };
        self.d_branch_out.copy_to_host(raw_out).map_err(CudaFail)?;
        comm.all_reduce_sum_f16(&mut self.h_ar_buf[..elems])?;
        let raw_in = unsafe { std::slice::from_raw_parts(self.h_ar_buf.as_ptr() as *const u8, elems * 2) };
        self.d_branch_out.copy_from(raw_in, None).map_err(CudaFail)?;
        Ok(())
    }

    fn forward_layers(&mut self, tokens: &[i32], comm: &mut Tp2Comm) -> Result<()> {
        let m = tokens.len();
        let mi = m as i32;
        let tc = self.cfg.text_config.clone();
        let d = tc.hidden_size;
        let n_heads_local = tc.num_attention_heads / self.world;
        let groups_local = tc.o_groups / self.world;
        let heads_per_grp = tc.num_attention_heads / tc.o_groups;
        let stream: CudaStream = std::ptr::null_mut();

        let tok_bytes = unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, m * 4) };
        self.d_tok_ids.copy_from(tok_bytes, None).map_err(CudaFail)?;
        unsafe {
            cuda_ops::rad_cuda_embed_to_mhc_f16(self.embed.ptr, self.d_tok_ids.ptr as *const i32, self.d_streams.ptr, mi, d as i32, stream);
        }

        let start_pos = self.pos as i32;
        let attn_scale = 1.0f32 / (tc.head_dim as f32).sqrt();

        for l in 0..self.layers.len() {
            unsafe {
                let lw = &self.layers[l];
                cuda_ops::rad_cuda_mhc_pre_and_coeff(
                    self.d_streams.ptr,
                    lw.hc_attn_fn.ptr,
                    lw.hc_attn_base.ptr,
                    lw.hc_attn_scale.ptr,
                    self.d_h_coeff.ptr,
                    self.d_branch_in.ptr,
                    mi,
                    d as i32,
                    tc.hc_eps,
                    stream,
                );
                cuda_ops::rad_cuda_rms_norm_f32_bf16w_f16(self.d_branch_in.ptr, lw.attn_norm.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, stream);
                self.exl3_proj(self.d_normed.ptr, &lw.wq_a, self.d_q_lora.ptr, mi, 0, stream);
                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_q_lora.ptr, lw.q_norm.ptr, self.d_q_normed.ptr, mi, tc.q_lora_rank as i32, tc.rms_norm_eps, stream);
                self.exl3_proj(self.d_q_normed.ptr, &lw.wq_b, self.d_q.ptr, mi, 0, stream);

                self.exl3_proj(self.d_normed.ptr, &lw.wkv, self.d_kv.ptr, mi, 0, stream);
                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, lw.kv_norm.ptr, self.d_kv_normed.ptr, mi, tc.head_dim as i32, tc.rms_norm_eps, stream);
                cuda_ops::rad_cuda_rope_yarn_f16(
                    self.d_q.ptr,
                    self.d_kv_normed.ptr,
                    mi,
                    n_heads_local as i32,
                    1,
                    tc.head_dim as i32,
                    tc.qk_rope_head_dim as i32,
                    start_pos,
                    tc.rope_theta,
                    1.0f32,
                    stream,
                );
                cuda_ops::rad_cuda_kv_scatter_f16(
                    self.d_kv_normed.ptr,
                    self.d_kv_normed.ptr,
                    self.kv_caches[l].k_cache.ptr,
                    self.kv_caches[l].v_cache.ptr,
                    mi,
                    start_pos,
                    self.max_ctx as i32,
                    tc.head_dim as i32,
                    stream,
                );

                let sink_ptr = (lw.attn_sink.ptr as *const f32).add(self.rank * n_heads_local);
                cuda_ops::rad_cuda_mqa_attn_sink_causal_f16(
                    self.d_q.ptr,
                    self.kv_caches[l].k_cache.ptr,
                    self.kv_caches[l].v_cache.ptr,
                    sink_ptr,
                    self.d_attn_out.ptr,
                    mi,
                    n_heads_local as i32,
                    start_pos,
                    self.max_ctx as i32,
                    tc.head_dim as i32,
                    attn_scale,
                    stream,
                );
                cuda_ops::rad_cuda_rope_yarn_f16(
                    self.d_attn_out.ptr,
                    std::ptr::null_mut(),
                    mi,
                    n_heads_local as i32,
                    0,
                    tc.head_dim as i32,
                    tc.qk_rope_head_dim as i32,
                    start_pos,
                    tc.rope_theta,
                    -1.0f32,
                    stream,
                );

                if m == 1 {
                    for (gi, wo_slice) in lw.wo_a_slices.iter().enumerate() {
                        let x_ptr = (self.d_attn_out.ptr as *const u8).add(gi * heads_per_grp * tc.head_dim * 2) as *const c_void;
                        let y_ptr = (self.d_wo_mid.ptr as *mut u8).add(gi * tc.o_lora_rank * 2) as *mut c_void;
                        self.exl3_proj(x_ptr, wo_slice, y_ptr, 1, 0, stream);
                    }
                } else {
                    for (gi, wo_slice) in lw.wo_a_slices.iter().enumerate() {
                        cuda_ops::rad_cuda_wo_a_gather_group_f16(
                            self.d_attn_out.ptr,
                            self.d_wo_grp_in.ptr,
                            mi,
                            n_heads_local as i32,
                            heads_per_grp as i32,
                            tc.head_dim as i32,
                            gi as i32,
                            stream,
                        );
                        self.exl3_proj(self.d_wo_grp_in.ptr, wo_slice, self.d_wo_grp_out.ptr, mi, 0, stream);
                        cuda_ops::rad_cuda_wo_a_scatter_group_f16(
                            self.d_wo_grp_out.ptr,
                            self.d_wo_mid.ptr,
                            mi,
                            groups_local as i32,
                            tc.o_lora_rank as i32,
                            gi as i32,
                            stream,
                        );
                    }
                }
                self.exl3_proj(self.d_wo_mid.ptr, &lw.wo_b, self.d_branch_out.ptr, mi, 0, stream);
            }

            self.all_reduce_branch(m, comm)?;

            unsafe {
                let lw = &self.layers[l];
                cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(
                    self.d_streams.ptr,
                    self.d_branch_out.ptr,
                    self.d_h_coeff.ptr as *const f32,
                    mi,
                    d as i32,
                    tc.hc_sinkhorn_iters as i32,
                    tc.hc_eps,
                    stream,
                );
                cuda_ops::rad_cuda_mhc_pre_and_coeff(
                    self.d_streams.ptr,
                    lw.hc_ffn_fn.ptr,
                    lw.hc_ffn_base.ptr,
                    lw.hc_ffn_scale.ptr,
                    self.d_h_coeff.ptr,
                    self.d_branch_in.ptr,
                    mi,
                    d as i32,
                    tc.hc_eps,
                    stream,
                );
                cuda_ops::rad_cuda_rms_norm_f32_bf16w_f16(self.d_branch_in.ptr, lw.ffn_norm.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, stream);
                cuda_ops::rad_cuda_moe_route_sqrtsoftplus(
                    self.d_normed.ptr,
                    lw.gate_w.ptr,
                    lw.gate_bias.ptr,
                    self.d_gate_raw.ptr,
                    self.d_gate_sel.ptr,
                    self.d_topk_ids.ptr,
                    self.d_topk_weights.ptr,
                    mi,
                    tc.n_routed_experts as i32,
                    d as i32,
                    tc.num_experts_per_tok as i32,
                    tc.routed_scaling_factor,
                    stream,
                );
                self.exl3_proj(self.d_normed.ptr, &lw.shared_w1, self.d_shared_g.ptr, mi, 0, stream);
                self.exl3_proj(self.d_normed.ptr, &lw.shared_w3, self.d_shared_u.ptr, mi, 0, stream);
                cuda_ops::rad_cuda_clamped_swiglu_f16(
                    self.d_shared_g.ptr,
                    self.d_shared_u.ptr,
                    self.d_shared_g.ptr,
                    mi * (tc.moe_intermediate_size / self.world) as i32,
                    tc.swiglu_limit,
                    stream,
                );
                self.exl3_proj(self.d_shared_g.ptr, &lw.shared_w2, self.d_shared_out.ptr, mi, 0, stream);

                let tk = tc.num_experts_per_tok as i32;
                let inter = tc.moe_intermediate_size as i32;
                let bits = lw.experts[0].w1.bits as i32;
                let exp_start = lw.expert_start as i32;
                let n_local = lw.experts.len() as i32;

                cuda_ops::rad_cuda_exl3_moe_mma_f16(
                    self.d_normed.ptr,
                    lw.d_w1_trel.ptr as *const *const c_void,
                    lw.d_w1_suh.ptr as *const *const c_void,
                    lw.d_w1_svh.ptr as *const *const c_void,
                    self.d_topk_ids.ptr as *const i32,
                    self.d_z_scratch.ptr,
                    self.d_moe_gate.ptr,
                    tk,
                    mi,
                    d as i32,
                    inter,
                    bits,
                    self.wht_scale,
                    0,
                    exp_start,
                    n_local,
                    stream,
                );
                cuda_ops::rad_cuda_exl3_moe_mma_f16(
                    self.d_normed.ptr,
                    lw.d_w3_trel.ptr as *const *const c_void,
                    lw.d_w3_suh.ptr as *const *const c_void,
                    lw.d_w3_svh.ptr as *const *const c_void,
                    self.d_topk_ids.ptr as *const i32,
                    self.d_z_scratch.ptr,
                    self.d_moe_up.ptr,
                    tk,
                    mi,
                    d as i32,
                    inter,
                    bits,
                    self.wht_scale,
                    0,
                    exp_start,
                    n_local,
                    stream,
                );
                cuda_ops::rad_cuda_clamped_swiglu_f16(
                    self.d_moe_gate.ptr,
                    self.d_moe_up.ptr,
                    self.d_moe_act.ptr,
                    mi * tk * inter,
                    tc.swiglu_limit,
                    stream,
                );
                cuda_ops::rad_cuda_exl3_moe_mma_f16(
                    self.d_moe_act.ptr,
                    lw.d_w2_trel.ptr as *const *const c_void,
                    lw.d_w2_suh.ptr as *const *const c_void,
                    lw.d_w2_svh.ptr as *const *const c_void,
                    self.d_topk_ids.ptr as *const i32,
                    self.d_z_scratch.ptr,
                    self.d_moe_outs.ptr,
                    tk,
                    mi,
                    inter,
                    d as i32,
                    bits,
                    self.wht_scale,
                    1,
                    exp_start,
                    n_local,
                    stream,
                );
                cuda_ops::rad_cuda_moe_weighted_sum_f16(
                    self.d_moe_outs.ptr,
                    self.d_topk_weights.ptr as *const f32,
                    self.d_shared_out.ptr,
                    self.d_branch_out.ptr,
                    mi,
                    tk,
                    d as i32,
                    stream,
                );
            }

            self.all_reduce_branch(m, comm)?;

            unsafe {
                cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(
                    self.d_streams.ptr,
                    self.d_branch_out.ptr,
                    self.d_h_coeff.ptr as *const f32,
                    mi,
                    d as i32,
                    tc.hc_sinkhorn_iters as i32,
                    tc.hc_eps,
                    stream,
                );
            }
        }
        self.pos += m;
        Ok(())
    }

    pub fn step_batch_no_head(&mut self, tokens: &[i32], comm: &mut Tp2Comm) -> Result<()> {
        self.forward_layers(tokens, comm)
    }

    pub fn step_batch(&mut self, tokens: &[i32], comm: &mut Tp2Comm) -> Result<Vec<i32>> {
        let m = tokens.len();
        self.forward_layers(tokens, comm)?;
        let tc = &self.cfg.text_config;
        let d = tc.hidden_size;
        let stream: CudaStream = std::ptr::null_mut();
        unsafe {
            cuda_ops::rad_cuda_mhc_collapse_mean_f16(self.d_streams.ptr, self.d_branch_in.ptr, m as i32, d as i32, stream);
            cuda_ops::rad_cuda_rms_norm_f32_bf16w_f16(self.d_branch_in.ptr, self.final_norm.ptr, self.d_normed.ptr, m as i32, d as i32, tc.rms_norm_eps, stream);
            self.exl3_proj(self.d_normed.ptr, &self.head, self.d_logits.ptr, m as i32, 0, stream);
            cuda_ops::rad_cuda_argmax_f16(
                self.d_logits.ptr,
                self.d_arg_vals.ptr,
                self.d_arg_ids.ptr,
                self.d_best_val.ptr,
                self.d_best_id.ptr,
                m as i32,
                self.head.n as i32,
                stream,
            );
        }
        rad_cuda::sync_device().map_err(CudaFail)?;
        let mut best_vals = vec![0.0f32; m];
        let mut best_ids = vec![0i32; m];
        self.d_best_val.copy_to_host(unsafe { std::slice::from_raw_parts_mut(best_vals.as_mut_ptr() as *mut u8, m * 4) }).map_err(CudaFail)?;
        self.d_best_id.copy_to_host(unsafe { std::slice::from_raw_parts_mut(best_ids.as_mut_ptr() as *mut u8, m * 4) }).map_err(CudaFail)?;
        let offset = (self.rank * self.head.n) as i32;
        for id in &mut best_ids {
            *id += offset;
        }
        comm.all_reduce_argmax(&mut best_vals, &mut best_ids)?;
        Ok(best_ids)
    }

    pub fn step_decode(&mut self, token_id: i32, comm: &mut Tp2Comm) -> Result<i32> {
        let out = self.step_batch(&[token_id], comm)?;
        Ok(out[0])
    }

    pub fn profile_step(&mut self, token_id: i32, comm: &mut Tp2Comm) -> Result<()> {
        let tc = self.cfg.text_config.clone();
        let d = tc.hidden_size;
        let n_heads_local = tc.num_attention_heads / self.world;
        let heads_per_grp = tc.num_attention_heads / tc.o_groups;
        let stream: CudaStream = std::ptr::null_mut();
        let start_pos = self.pos as i32;
        let attn_scale = 1.0f32 / (tc.head_dim as f32).sqrt();
        let sync_t = || -> f64 {
            let _ = rad_cuda::sync_device();
            0.0
        };
        self.d_tok_ids.copy_from(&token_id.to_le_bytes(), None).map_err(CudaFail)?;
        unsafe {
            cuda_ops::rad_cuda_embed_to_mhc_f16(self.embed.ptr, self.d_tok_ids.ptr as *const i32, self.d_streams.ptr, 1, d as i32, stream);
        }
        let _ = sync_t();
        let lw = &self.layers[0];
        let t0 = std::time::Instant::now();
        unsafe {
            cuda_ops::rad_cuda_mhc_pre_and_coeff(self.d_streams.ptr, lw.hc_attn_fn.ptr, lw.hc_attn_base.ptr, lw.hc_attn_scale.ptr, self.d_h_coeff.ptr, self.d_branch_in.ptr, 1, d as i32, tc.hc_eps, stream);
            cuda_ops::rad_cuda_rms_norm_f32_bf16w_f16(self.d_branch_in.ptr, lw.attn_norm.ptr, self.d_normed.ptr, 1, d as i32, tc.rms_norm_eps, stream);
        }
        let _ = sync_t();
        let t_mhc1 = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = std::time::Instant::now();
        unsafe {
            self.exl3_proj(self.d_normed.ptr, &lw.wq_a, self.d_q_lora.ptr, 1, 0, stream);
            cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_q_lora.ptr, lw.q_norm.ptr, self.d_q_normed.ptr, 1, tc.q_lora_rank as i32, tc.rms_norm_eps, stream);
            self.exl3_proj(self.d_q_normed.ptr, &lw.wq_b, self.d_q.ptr, 1, 0, stream);
            self.exl3_proj(self.d_normed.ptr, &lw.wkv, self.d_kv.ptr, 1, 0, stream);
            cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, lw.kv_norm.ptr, self.d_kv_normed.ptr, 1, tc.head_dim as i32, tc.rms_norm_eps, stream);
            cuda_ops::rad_cuda_rope_yarn_f16(self.d_q.ptr, self.d_kv_normed.ptr, 1, n_heads_local as i32, 1, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, tc.rope_theta, 1.0f32, stream);
            cuda_ops::rad_cuda_kv_scatter_f16(self.d_kv_normed.ptr, self.d_kv_normed.ptr, self.kv_caches[0].k_cache.ptr, self.kv_caches[0].v_cache.ptr, 1, start_pos, self.max_ctx as i32, tc.head_dim as i32, stream);
        }
        let _ = sync_t();
        let t_qkv = t1.elapsed().as_secs_f64() * 1e6;

        let t2 = std::time::Instant::now();
        unsafe {
            let sink_ptr = (lw.attn_sink.ptr as *const f32).add(self.rank * n_heads_local);
            cuda_ops::rad_cuda_mqa_attn_sink_causal_f16(self.d_q.ptr, self.kv_caches[0].k_cache.ptr, self.kv_caches[0].v_cache.ptr, sink_ptr, self.d_attn_out.ptr, 1, n_heads_local as i32, start_pos, self.max_ctx as i32, tc.head_dim as i32, attn_scale, stream);
            cuda_ops::rad_cuda_rope_yarn_f16(self.d_attn_out.ptr, std::ptr::null_mut(), 1, n_heads_local as i32, 0, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, tc.rope_theta, -1.0f32, stream);
        }
        let _ = sync_t();
        let t_attn = t2.elapsed().as_secs_f64() * 1e6;

        let t3 = std::time::Instant::now();
        unsafe {
            for (gi, wo_slice) in lw.wo_a_slices.iter().enumerate() {
                let x_ptr = (self.d_attn_out.ptr as *const u8).add(gi * heads_per_grp * tc.head_dim * 2) as *const c_void;
                let y_ptr = (self.d_wo_mid.ptr as *mut u8).add(gi * tc.o_lora_rank * 2) as *mut c_void;
                self.exl3_proj(x_ptr, wo_slice, y_ptr, 1, 0, stream);
            }
            self.exl3_proj(self.d_wo_mid.ptr, &lw.wo_b, self.d_branch_out.ptr, 1, 0, stream);
        }
        let _ = sync_t();
        let t_wo = t3.elapsed().as_secs_f64() * 1e6;

        let t4 = std::time::Instant::now();
        self.all_reduce_branch(1, comm)?;
        let _ = sync_t();
        let t_ar1 = t4.elapsed().as_secs_f64() * 1e6;

        let lw = &self.layers[0];
        let t5 = std::time::Instant::now();
        unsafe {
            cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(self.d_streams.ptr, self.d_branch_out.ptr, self.d_h_coeff.ptr as *const f32, 1, d as i32, tc.hc_sinkhorn_iters as i32, tc.hc_eps, stream);
            cuda_ops::rad_cuda_mhc_pre_and_coeff(self.d_streams.ptr, lw.hc_ffn_fn.ptr, lw.hc_ffn_base.ptr, lw.hc_ffn_scale.ptr, self.d_h_coeff.ptr, self.d_branch_in.ptr, 1, d as i32, tc.hc_eps, stream);
            cuda_ops::rad_cuda_rms_norm_f32_bf16w_f16(self.d_branch_in.ptr, lw.ffn_norm.ptr, self.d_normed.ptr, 1, d as i32, tc.rms_norm_eps, stream);
            cuda_ops::rad_cuda_moe_route_sqrtsoftplus(self.d_normed.ptr, lw.gate_w.ptr, lw.gate_bias.ptr, self.d_gate_raw.ptr, self.d_gate_sel.ptr, self.d_topk_ids.ptr, self.d_topk_weights.ptr, 1, tc.n_routed_experts as i32, d as i32, tc.num_experts_per_tok as i32, tc.routed_scaling_factor, stream);
        }
        let _ = sync_t();
        let t_route = t5.elapsed().as_secs_f64() * 1e6;

        let t6 = std::time::Instant::now();
        unsafe {
            self.exl3_proj(self.d_normed.ptr, &lw.shared_w1, self.d_shared_g.ptr, 1, 0, stream);
            self.exl3_proj(self.d_normed.ptr, &lw.shared_w3, self.d_shared_u.ptr, 1, 0, stream);
            cuda_ops::rad_cuda_clamped_swiglu_f16(self.d_shared_g.ptr, self.d_shared_u.ptr, self.d_shared_g.ptr, (tc.moe_intermediate_size / self.world) as i32, tc.swiglu_limit, stream);
            self.exl3_proj(self.d_shared_g.ptr, &lw.shared_w2, self.d_shared_out.ptr, 1, 0, stream);
        }
        let _ = sync_t();
        let t_shared = t6.elapsed().as_secs_f64() * 1e6;

        let t7 = std::time::Instant::now();
        unsafe {
            let tk = tc.num_experts_per_tok as i32;
            let inter = tc.moe_intermediate_size as i32;
            let bits = lw.experts[0].w1.bits as i32;
            let exp_start = lw.expert_start as i32;
            let n_local = lw.experts.len() as i32;
            cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w1_trel.ptr as *const *const c_void, lw.d_w1_suh.ptr as *const *const c_void, lw.d_w1_svh.ptr as *const *const c_void, self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, self.d_moe_gate.ptr, tk, 1, d as i32, inter, bits, self.wht_scale, 0, exp_start, n_local, stream);
            cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w3_trel.ptr as *const *const c_void, lw.d_w3_suh.ptr as *const *const c_void, lw.d_w3_svh.ptr as *const *const c_void, self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, self.d_moe_up.ptr, tk, 1, d as i32, inter, bits, self.wht_scale, 0, exp_start, n_local, stream);
            cuda_ops::rad_cuda_clamped_swiglu_f16(self.d_moe_gate.ptr, self.d_moe_up.ptr, self.d_moe_act.ptr, tk * inter, tc.swiglu_limit, stream);
            cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_moe_act.ptr, lw.d_w2_trel.ptr as *const *const c_void, lw.d_w2_suh.ptr as *const *const c_void, lw.d_w2_svh.ptr as *const *const c_void, self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, self.d_moe_outs.ptr, tk, 1, inter, d as i32, bits, self.wht_scale, 1, exp_start, n_local, stream);
            cuda_ops::rad_cuda_moe_weighted_sum_f16(self.d_moe_outs.ptr, self.d_topk_weights.ptr as *const f32, self.d_shared_out.ptr, self.d_branch_out.ptr, 1, tk, d as i32, stream);
        }
        let _ = sync_t();
        let t_moe = t7.elapsed().as_secs_f64() * 1e6;

        let t8 = std::time::Instant::now();
        self.all_reduce_branch(1, comm)?;
        let _ = sync_t();
        let t_ar2 = t8.elapsed().as_secs_f64() * 1e6;

        if self.rank == 0 {
            let tot = t_mhc1 + t_qkv + t_attn + t_wo + t_ar1 + t_route + t_shared + t_moe + t_ar2;
            println!(
                "PROFILE L0 (us): mhc1={t_mhc1:.0} qkv={t_qkv:.0} attn={t_attn:.0} wo={t_wo:.0} ar1={t_ar1:.0} route={t_route:.0} shared={t_shared:.0} moe={t_moe:.0} ar2={t_ar2:.0} | SUM={tot:.0} us ({:.1} ms x40)",
                tot * 40.0 / 1000.0
            );
        }
        Ok(())
    }

    pub fn prefill_tokens(&mut self, tokens: &[i32], comm: &mut Tp2Comm) -> Result<i32> {
        if tokens.is_empty() {
            return Ok(self.cfg.bos_token_id);
        }
        let n = tokens.len();
        let mut idx = 0usize;
        while idx + 1 < n {
            let chunk_end = (idx + MAX_BATCH).min(n - 1);
            let mut chunk = tokens[idx..chunk_end].to_vec();
            let mut cmd = 1u32;
            comm.broadcast_u32(&mut cmd)?;
            comm.broadcast_i32_vec(&mut chunk)?;
            self.step_batch_no_head(&chunk, comm)?;
            idx = chunk_end;
        }
        let mut last = vec![tokens[n - 1]];
        let mut cmd = 2u32;
        comm.broadcast_u32(&mut cmd)?;
        comm.broadcast_i32_vec(&mut last)?;
        let preds = self.step_batch(&last, comm)?;
        Ok(preds[0])
    }

    pub fn step_speculative(&mut self, history: &[i32], anchor: i32, comm: &mut Tp2Comm) -> Result<Vec<i32>> {
        if let Some(mut draft) = ngram_draft(history, anchor, 16) {
            let start_pos = self.pos;
            let mut cmd = 2u32;
            comm.broadcast_u32(&mut cmd)?;
            comm.broadcast_i32_vec(&mut draft)?;
            let preds = self.step_batch(&draft, comm)?;
            let mut accepted = Vec::with_capacity(draft.len());
            let mut k_acc = 0usize;
            for i in 0..draft.len() {
                let pred = preds[i];
                accepted.push(pred);
                k_acc += 1;
                if i + 1 < draft.len() && pred != draft[i + 1] {
                    break;
                }
            }
            let new_pos = start_pos + k_acc;
            self.pos = new_pos;
            let mut pcmd = 3u32;
            comm.broadcast_u32(&mut pcmd)?;
            let mut pval = new_pos as u32;
            comm.broadcast_u32(&mut pval)?;
            return Ok(accepted);
        }
        let mut one = vec![anchor];
        let mut cmd = 2u32;
        comm.broadcast_u32(&mut cmd)?;
        comm.broadcast_i32_vec(&mut one)?;
        let preds = self.step_batch(&one, comm)?;
        Ok(preds)
    }
}
