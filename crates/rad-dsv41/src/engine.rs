use anyhow::Result;
use core::ffi::c_void;
use rad_cuda::comm::Tp2Comm;
use rad_cuda::{CudaBuf, CudaCapturedGraph, CudaFail, CudaPinnedBuf, CudaStream};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;
use crate::config::DeepseekV41Config;
use crate::cuda_ops;
use crate::engram::{EngramHost, EngramLayerWeights};
use crate::mtp::{ngram_draft_cfg, DsparkMtp};
use crate::safetensors::SafetensorsIndex;
use crate::weights::{load_raw_buf_replicated, CompressorWeights, Exl3Linear, LayerWeights};

pub const MAX_BATCH: usize = 2048;
pub const MAX_DECODE_BATCH: usize = 256;

pub struct KvCacheLayer {
    pub k_cache: CudaBuf,
}

pub struct CompKvSlot {
    pub layer_id: usize,
    pub weights: CompressorWeights,
    pub raw_kv: CudaBuf,
    pub raw_gate: Option<CudaBuf>,
    pub comp_kv: CudaBuf,
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
    pub comp_slots: Vec<CompKvSlot>,
    pub engram_layers: Vec<EngramLayerWeights>,
    pub engram_host: Option<EngramHost>,
    pub d_engram_in: CudaBuf,
    pub d_engram_kv: CudaBuf,
    pub tok_history: Vec<i32>,
    pub mtp: Option<DsparkMtp>,
    pub d_streams: CudaBuf,
    pub d_normed: CudaBuf,
    pub d_xrot: CudaBuf,
    pub d_z_scratch: CudaBuf,
    pub d_q_lora: CudaBuf,
    pub d_q_normed: CudaBuf,
    pub d_q: CudaBuf,
    pub d_kv: CudaBuf,
    pub d_kv_normed: CudaBuf,
    pub d_attn_out: CudaBuf,
    pub d_wo_mid: CudaBuf,
    pub d_branch_out: CudaBuf,
    pub d_h_coeff: CudaBuf,
    pub d_pre_a: CudaBuf,
    pub d_pre_b: CudaBuf,
    pub d_gate_raw: CudaBuf,
    pub d_gate_sel: CudaBuf,
    pub d_topk_ids: CudaBuf,
    pub d_topk_weights: CudaBuf,
    pub d_moe_gate: CudaBuf,
    pub d_moe_act: CudaBuf,
    pub d_moe_outs: CudaBuf,
    pub d_shared_g: CudaBuf,
    pub d_shared_out: CudaBuf,
    pub d_logits: CudaBuf,
    pub d_tok_ids: CudaPinnedBuf,
    pub d_arg_vals: CudaBuf,
    pub d_arg_ids: CudaBuf,
    pub d_best_val: CudaPinnedBuf,
    pub d_best_id: CudaPinnedBuf,
    pub h_ar_buf: Vec<u16>,
    pub qkv_graphs: HashMap<(usize, usize), CudaCapturedGraph>,
    pub wo_graphs: HashMap<(usize, usize), CudaCapturedGraph>,
    pub ffn_graphs: HashMap<(usize, usize), CudaCapturedGraph>,
    pub mtp_qkv_graphs: HashMap<usize, CudaCapturedGraph>,
    pub mtp_wo_graphs: HashMap<usize, CudaCapturedGraph>,
    pub mtp_ffn_graphs: HashMap<usize, CudaCapturedGraph>,
    pub last_draft_len: usize,
    pub last_acc: usize,
    pub dspark_blocks: usize,
    pub pos: usize,
    pub wht_scale: f32,
}

impl Dsv41Engine {
    pub fn load(
        model_dir: &Path,
        engram_dir: Option<&Path>,
        max_ctx: usize,
        comm: &mut Tp2Comm,
        max_layers: Option<usize>,
        load_mtp: bool,
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
            });
            if rank == 0 && (l == 0 || (l + 1) % 5 == 0 || l + 1 == n_layers) {
                println!("[load] layer {}/{n_layers} ready (world={world})", l + 1);
                let _ = std::io::stdout().flush();
            }
        }
        let mut comp_slots = Vec::new();
        for &lid in &tc.kv_source_layer_ids {
            if lid < n_layers {
                let ratio = tc.compress_ratios.get(lid).copied().unwrap_or(1).max(1);
                let weights = CompressorWeights::load(idx_opt.as_mut(), lid, ratio, comm)?;
                let raw_kv = CudaBuf::new(max_ctx * tc.head_dim * 2).map_err(CudaFail)?;
                let raw_gate = if ratio > 1 {
                    Some(CudaBuf::new(max_ctx * tc.head_dim * 2).map_err(CudaFail)?)
                } else {
                    None
                };
                let comp_cap = (max_ctx / ratio).max(1);
                let comp_kv = CudaBuf::new(comp_cap * tc.head_dim * 2).map_err(CudaFail)?;
                comp_slots.push(CompKvSlot {
                    layer_id: lid,
                    weights,
                    raw_kv,
                    raw_gate,
                    comp_kv,
                });
            }
        }
        let mut engram_layers = Vec::new();
        let engram_host = if let Some(edir) = engram_dir {
            for (tidx, &lid) in tc.engram_layer_ids.iter().enumerate() {
                if lid < n_layers {
                    engram_layers.push(EngramLayerWeights::load(idx_opt.as_mut(), lid, tidx, comm)?);
                }
            }
            if rank == 0 && !engram_layers.is_empty() {
                let tok = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                    .map_err(|e| anyhow::anyhow!("tokenizer load for engram failed: {e}"))?;
                Some(EngramHost::open(edir, &tok, tc.vocab_size, cfg.pad_token_id as usize)?)
            } else {
                None
            }
        } else {
            None
        };
        let mtp = if load_mtp && n_layers == tc.num_hidden_layers && tc.num_nextn_predict_layers > 0 {
            let m = DsparkMtp::load(idx_opt.as_mut(), &cfg, max_ctx, comm)?;
            if rank == 0 {
                println!("[load] DSpark MTP ({} layers, block={}) ready", m.layers.len(), tc.dspark_block_size);
                let _ = std::io::stdout().flush();
            }
            Some(m)
        } else {
            None
        };

        let d = tc.hidden_size;
        unsafe {
            cuda_ops::rad_cuda_moe_init_meta();
        }
        comm.set_nodelay(true);
        let n_heads_local = tc.num_attention_heads / world;
        let groups_local = tc.o_groups / world;
        let vocab_local = tc.vocab_size / world;
        let max_head_m = MAX_DECODE_BATCH;
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
            comp_slots,
            engram_layers,
            engram_host,
            d_engram_in: CudaBuf::new(MAX_BATCH * 6144 * 2).map_err(CudaFail)?,
            d_engram_kv: CudaBuf::new(MAX_BATCH * 5 * d * 2).map_err(CudaFail)?,
            tok_history: Vec::with_capacity(max_ctx),
            mtp,
            d_streams: CudaBuf::new(MAX_BATCH * 4 * d * 4).map_err(CudaFail)?,
            d_normed: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_xrot: CudaBuf::new(MAX_BATCH * (n_heads_local * tc.head_dim).max(3 * d) * 2).map_err(CudaFail)?,
            d_z_scratch: CudaBuf::new(z_elems * 4).map_err(CudaFail)?,
            d_q_lora: CudaBuf::new(MAX_BATCH * tc.q_lora_rank * 2).map_err(CudaFail)?,
            d_q_normed: CudaBuf::new(MAX_BATCH * tc.q_lora_rank * 2).map_err(CudaFail)?,
            d_q: CudaBuf::new(MAX_BATCH * n_heads_local * tc.head_dim * 2).map_err(CudaFail)?,
            d_kv: CudaBuf::new(MAX_BATCH * tc.head_dim * 2).map_err(CudaFail)?,
            d_kv_normed: CudaBuf::new(MAX_BATCH * tc.head_dim * 2).map_err(CudaFail)?,
            d_attn_out: CudaBuf::new(MAX_BATCH * n_heads_local * tc.head_dim * 2).map_err(CudaFail)?,
            d_wo_mid: CudaBuf::new(MAX_BATCH * groups_local * tc.o_lora_rank * 2).map_err(CudaFail)?,
            d_branch_out: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_h_coeff: CudaBuf::new(MAX_BATCH * 24 * 4).map_err(CudaFail)?,
            d_pre_a: CudaBuf::new(MAX_BATCH * 4 * 4).map_err(CudaFail)?,
            d_pre_b: CudaBuf::new(MAX_BATCH * 4 * 4).map_err(CudaFail)?,
            d_gate_raw: CudaBuf::new(MAX_BATCH * tc.n_routed_experts * 4).map_err(CudaFail)?,
            d_gate_sel: CudaBuf::new(MAX_BATCH * tc.n_routed_experts * 4).map_err(CudaFail)?,
            d_topk_ids: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * 4).map_err(CudaFail)?,
            d_topk_weights: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * 4).map_err(CudaFail)?,
            d_moe_gate: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_act: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_outs: CudaBuf::new(MAX_BATCH * tc.num_experts_per_tok * d * 2).map_err(CudaFail)?,
            d_shared_g: CudaBuf::new(MAX_BATCH * (tc.moe_intermediate_size / world) * 2).map_err(CudaFail)?,
            d_shared_out: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_logits: CudaBuf::new(max_head_m * vocab_local * 2).map_err(CudaFail)?,
            d_tok_ids: CudaPinnedBuf::new(MAX_BATCH * 4).map_err(CudaFail)?,
            d_arg_vals: CudaBuf::new(max_head_m * 256 * 4).map_err(CudaFail)?,
            d_arg_ids: CudaBuf::new(max_head_m * 256 * 4).map_err(CudaFail)?,
            d_best_val: CudaPinnedBuf::new(max_head_m * 4).map_err(CudaFail)?,
            d_best_id: CudaPinnedBuf::new(max_head_m * 4).map_err(CudaFail)?,
            h_ar_buf: vec![0u16; MAX_BATCH * d],
            qkv_graphs: HashMap::new(),
            wo_graphs: HashMap::new(),
            ffn_graphs: HashMap::new(),
            mtp_qkv_graphs: HashMap::new(),
            mtp_wo_graphs: HashMap::new(),
            mtp_ffn_graphs: HashMap::new(),
            last_draft_len: 1,
            last_acc: 0,
            dspark_blocks: 1,
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

    unsafe fn exl3_proj_prerot(
        &self,
        x_rot: *const c_void,
        w: &Exl3Linear,
        y: *mut c_void,
        m: i32,
        add_to_y: i32,
        stream: CudaStream,
    ) {
        cuda_ops::rad_cuda_exl3_mma_f16(
            x_rot,
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

    unsafe fn exl3_proj_swiglu(
        &self,
        x: *const c_void,
        w: &Exl3Linear,
        gate_in: *const c_void,
        next_suh: *const c_void,
        y: *mut c_void,
        m: i32,
        swiglu_limit: f32,
        stream: CudaStream,
    ) {
        cuda_ops::rad_cuda_wht128_pre_mul_f16(x, w.suh.ptr, self.d_xrot.ptr, m, w.k as i32, self.wht_scale, stream);
        cuda_ops::rad_cuda_exl3_mma_swiglu_f16(
            self.d_xrot.ptr,
            w.trellis.ptr,
            w.svh.ptr,
            self.d_z_scratch.ptr,
            gate_in,
            next_suh,
            y,
            m,
            w.k as i32,
            w.n as i32,
            w.bits as i32,
            self.wht_scale,
            swiglu_limit,
            stream,
        );
    }

    unsafe fn enqueue_qkv_proj(&self, l: usize, mi: i32, stream: CudaStream) {
        let tc = &self.cfg.text_config;
        let d = tc.hidden_size;
        if let Some(ref mtp) = self.mtp {
            if let Some(slot) = tc.dspark_target_layer_ids.iter().position(|&id| id == l) {
                cuda_ops::rad_cuda_mhc_collapse_mean_to_f16(
                    self.d_streams.ptr,
                    mtp.d_taps_f16.ptr,
                    mi,
                    d as i32,
                    (3 * d) as i32,
                    (slot * d) as i32,
                    stream,
                );
            }
        }
        let lw = &self.layers[l];
        if l > 0 && self.engram_layers.is_empty() {
            cuda_ops::rad_cuda_mhc_coeff_proj_only(
                self.d_streams.ptr,
                lw.hc_attn_fn.ptr,
                lw.hc_attn_base.ptr,
                lw.hc_attn_scale.ptr,
                self.d_h_coeff.ptr,
                self.d_pre_a.ptr,
                mi,
                d as i32,
                tc.rms_norm_eps,
                tc.hc_eps,
                stream,
            );
        } else {
            let prev_pre = if l == 0 { std::ptr::null() } else { self.d_pre_b.ptr as *const c_void };
            cuda_ops::rad_cuda_mhc_pre_coeff_rmsnorm(
                self.d_streams.ptr,
                lw.hc_attn_fn.ptr,
                lw.hc_attn_base.ptr,
                lw.hc_attn_scale.ptr,
                prev_pre,
                lw.attn_norm.ptr,
                self.d_h_coeff.ptr,
                self.d_pre_a.ptr,
                self.d_normed.ptr,
                mi,
                d as i32,
                tc.rms_norm_eps,
                tc.hc_eps,
                stream,
            );
        }
        self.exl3_proj(self.d_normed.ptr, &lw.wq_a, self.d_q_lora.ptr, mi, 0, stream);
        cuda_ops::rad_cuda_rms_norm_bf16w_wht128_f16(self.d_q_lora.ptr, lw.q_norm.ptr, lw.wq_b.suh.ptr, self.d_xrot.ptr, mi, tc.q_lora_rank as i32, tc.rms_norm_eps, self.wht_scale, stream);
        self.exl3_proj_prerot(self.d_xrot.ptr, &lw.wq_b, self.d_q.ptr, mi, 0, stream);
        self.exl3_proj(self.d_normed.ptr, &lw.wkv, self.d_kv.ptr, mi, 0, stream);
        cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, lw.kv_norm.ptr, self.d_kv_normed.ptr, mi, tc.head_dim as i32, tc.rms_norm_eps, stream);
    }

    unsafe fn enqueue_wo_proj(&self, l: usize, m: usize, branch_dst: *mut c_void, stream: CudaStream) {
        let tc = &self.cfg.text_config;
        let mi = m as i32;
        let groups_local = (tc.o_groups / self.world) as i32;
        let lw = &self.layers[l];
        let s0 = &lw.wo_a_slices[0];
        cuda_ops::rad_cuda_exl3_batched_group_mma_f16(
            self.d_attn_out.ptr,
            lw.d_wo_a_trel.ptr as *const *const c_void,
            lw.d_wo_a_suh.ptr as *const *const c_void,
            lw.d_wo_a_svh.ptr as *const *const c_void,
            self.d_z_scratch.ptr,
            self.d_wo_mid.ptr,
            self.d_xrot.ptr,
            mi,
            groups_local,
            s0.k as i32,
            s0.n as i32,
            s0.bits as i32,
            self.wht_scale,
            stream,
        );
        self.exl3_proj(self.d_wo_mid.ptr, &lw.wo_b, branch_dst, mi, 0, stream);
    }

    unsafe fn enqueue_ffn_half(
        &self,
        l: usize,
        mi: i32,
        b_self: *const c_void,
        b_peer: *const c_void,
        branch_dst: *mut c_void,
        stream: CudaStream,
    ) {
        let tc = &self.cfg.text_config;
        let d = tc.hidden_size;
        let lw = &self.layers[l];
        cuda_ops::rad_cuda_mhc_sinkhorn_pre_coeff_rmsnorm(
            self.d_streams.ptr,
            b_self,
            b_peer,
            lw.hc_ffn_fn.ptr,
            lw.hc_ffn_base.ptr,
            lw.hc_ffn_scale.ptr,
            self.d_pre_a.ptr,
            lw.ffn_norm.ptr,
            self.d_h_coeff.ptr,
            self.d_pre_b.ptr,
            self.d_normed.ptr,
            mi,
            d as i32,
            tc.hc_sinkhorn_iters as i32,
            tc.rms_norm_eps,
            tc.hc_eps,
            stream,
        );
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
        self.exl3_proj_swiglu(self.d_normed.ptr, &lw.shared_w3, self.d_shared_g.ptr, lw.shared_w2.suh.ptr, self.d_shared_g.ptr, mi, tc.swiglu_limit, stream);
        self.exl3_proj_prerot(self.d_shared_g.ptr, &lw.shared_w2, self.d_shared_out.ptr, mi, 0, stream);

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
            std::ptr::null(),
            self.d_topk_ids.ptr as *const i32,
            self.d_z_scratch.ptr,
            std::ptr::null(),
            self.d_moe_gate.ptr,
            self.d_moe_outs.ptr,
            tk,
            mi,
            d as i32,
            inter,
            bits,
            self.wht_scale,
            tc.swiglu_limit,
            0,
            exp_start,
            n_local,
            0,
            stream,
        );
        cuda_ops::rad_cuda_exl3_moe_mma_f16(
            self.d_normed.ptr,
            lw.d_w3_trel.ptr as *const *const c_void,
            lw.d_w3_suh.ptr as *const *const c_void,
            lw.d_w3_svh.ptr as *const *const c_void,
            lw.d_w2_suh.ptr as *const *const c_void,
            self.d_topk_ids.ptr as *const i32,
            self.d_z_scratch.ptr,
            self.d_moe_gate.ptr,
            self.d_moe_act.ptr,
            self.d_moe_outs.ptr,
            tk,
            mi,
            d as i32,
            inter,
            bits,
            self.wht_scale,
            tc.swiglu_limit,
            0,
            exp_start,
            n_local,
            1,
            stream,
        );
        let w2_suh_ptr = if mi >= 512 { std::ptr::null() } else { lw.d_w2_suh.ptr as *const *const c_void };
        cuda_ops::rad_cuda_exl3_moe_mma_f16(
            self.d_moe_act.ptr,
            lw.d_w2_trel.ptr as *const *const c_void,
            w2_suh_ptr,
            lw.d_w2_svh.ptr as *const *const c_void,
            std::ptr::null(),
            self.d_topk_ids.ptr as *const i32,
            self.d_z_scratch.ptr,
            std::ptr::null(),
            self.d_moe_outs.ptr,
            self.d_moe_act.ptr,
            tk,
            mi,
            inter,
            d as i32,
            bits,
            self.wht_scale,
            tc.swiglu_limit,
            1,
            exp_start,
            n_local,
            1,
            stream,
        );
        cuda_ops::rad_cuda_moe_weighted_sum_f16(
            self.d_moe_outs.ptr,
            self.d_topk_weights.ptr as *const f32,
            self.d_topk_ids.ptr as *const i32,
            self.d_shared_out.ptr,
            branch_dst,
            mi,
            tk,
            d as i32,
            exp_start,
            n_local,
            stream,
        );
    }

    fn exchange_branch(&mut self, m: usize, comm: &mut Tp2Comm) -> Result<(*const c_void, *const c_void)> {
        if self.world <= 1 {
            return Ok((self.d_branch_out.ptr, std::ptr::null()));
        }
        let d = self.cfg.text_config.hidden_size;
        let elems = m * d;
        rad_cuda::sync_device().map_err(CudaFail)?;
        if let Some((s_ptr, r_ptr)) = comm.rdma_ptrs() {
            comm.rdma_exchange_mapped(elems * 2)?;
            return Ok((s_ptr, r_ptr));
        }
        let raw_out = unsafe { std::slice::from_raw_parts_mut(self.h_ar_buf.as_mut_ptr() as *mut u8, elems * 2) };
        self.d_branch_out.copy_to_host(raw_out).map_err(CudaFail)?;
        comm.all_reduce_sum_f16(&mut self.h_ar_buf[..elems])?;
        let raw_in = unsafe { std::slice::from_raw_parts(self.h_ar_buf.as_ptr() as *const u8, elems * 2) };
        self.d_branch_out.copy_from(raw_in, None).map_err(CudaFail)?;
        Ok((self.d_branch_out.ptr, std::ptr::null()))
    }

    fn forward_layers(&mut self, tokens: &[i32], comm: &mut Tp2Comm, update_mtp: bool) -> Result<()> {
        let m = tokens.len();
        let mi = m as i32;
        let tc = self.cfg.text_config.clone();
        let d = tc.hidden_size;
        let n_heads_local = tc.num_attention_heads / self.world;
        let stream: CudaStream = std::ptr::null_mut();

        let tok_bytes = unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, m * 4) };
        self.d_tok_ids.as_bytes_mut()[..m * 4].copy_from_slice(tok_bytes);
        if self.tok_history.len() > self.pos {
            self.tok_history.truncate(self.pos);
        }
        self.tok_history.extend_from_slice(tokens);
        unsafe {
            cuda_ops::rad_cuda_embed_to_mhc_f16(self.embed.ptr, self.d_tok_ids.ptr as *const i32, self.d_streams.ptr, mi, d as i32, stream);
        }

        let start_pos = self.pos as i32;
        let attn_scale = 1.0f32 / (tc.head_dim as f32).sqrt();
        let branch_dst = comm.rdma_ptrs().map(|(s, _)| s).unwrap_or(self.d_branch_out.ptr);

        for l in 0..self.layers.len() {
            if let Some(e_idx) = self.engram_layers.iter().position(|e| e.layer_id == l) {
                let table_idx = self.engram_layers[e_idx].table_idx;
                let row_bytes = if self.rank == 0 {
                    let vals = self
                        .engram_host
                        .as_ref()
                        .unwrap()
                        .lookup_batch_f16(table_idx, &self.tok_history, self.pos, m)?;
                    let bytes = unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 2) }.to_vec();
                    if self.world > 1 {
                        comm.send_bytes(&bytes)?;
                    }
                    bytes
                } else {
                    comm.recv_bytes()?
                };
                self.d_engram_in.copy_from(&row_bytes, None).map_err(CudaFail)?;
                unsafe {
                    let ew = &self.engram_layers[e_idx];
                    self.exl3_proj(self.d_engram_in.ptr, &ew.wkv, self.d_engram_kv.ptr, mi, 0, stream);
                    cuda_ops::rad_cuda_engram_gate_f32(
                        self.d_streams.ptr,
                        self.d_engram_kv.ptr,
                        ew.q_weight.ptr,
                        ew.k_weight.ptr,
                        mi,
                        d as i32,
                        tc.rms_norm_eps,
                        stream,
                    );
                }
            }
            let use_graph = (m == 1 || m == 6 || m == 16 || m == 32 || m == 64 || m == 128 || m == 512 || m == 1024 || m == 2048) && comm.rdma_ptrs().is_some();
            if use_graph {
                let gkey = (l, m);
                if !self.qkv_graphs.contains_key(&gkey) {
                    let g = unsafe {
                        CudaCapturedGraph::capture(|gs| {
                            self.enqueue_qkv_proj(l, mi, gs);
                        })
                    }
                    .map_err(CudaFail)?;
                    self.qkv_graphs.insert(gkey, g);
                }
                self.qkv_graphs.get(&gkey).unwrap().launch_on(stream).map_err(CudaFail)?;
            } else {
                unsafe {
                    self.enqueue_qkv_proj(l, mi, stream);
                }
            }
            unsafe {
                let lw = &self.layers[l];
                let is_comp = tc.compress_ratios.get(l).copied().unwrap_or(0) > 0;
                let (r_theta, r_orig) = if is_comp {
                    (tc.compress_rope_theta, tc.rope_scaling.original_max_position_embeddings as i32)
                } else {
                    (tc.rope_theta, 0)
                };
                cuda_ops::rad_cuda_rope_yarn_f16(
                    self.d_q.ptr,
                    self.d_kv_normed.ptr,
                    mi,
                    n_heads_local as i32,
                    1,
                    tc.head_dim as i32,
                    tc.qk_rope_head_dim as i32,
                    start_pos,
                    r_theta,
                    r_orig,
                    1.0f32,
                    stream,
                );
                cuda_ops::rad_cuda_kv_scatter_f16(
                    self.d_kv_normed.ptr,
                    self.d_kv_normed.ptr,
                    self.kv_caches[l].k_cache.ptr,
                    self.kv_caches[l].k_cache.ptr,
                    mi,
                    start_pos,
                    self.max_ctx as i32,
                    tc.head_dim as i32,
                    stream,
                );

                if let Some(cs_idx) = self.comp_slots.iter().position(|s| s.layer_id == l) {
                    let ratio = self.comp_slots[cs_idx].weights.ratio;
                    self.exl3_proj(self.d_normed.ptr, &self.comp_slots[cs_idx].weights.wkv, self.d_kv.ptr, mi, 0, stream);
                    if ratio == 1 {
                        cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, self.comp_slots[cs_idx].weights.norm.ptr, self.d_kv_normed.ptr, mi, tc.head_dim as i32, tc.rms_norm_eps, stream);
                        cuda_ops::rad_cuda_rope_yarn_f16(std::ptr::null_mut(), self.d_kv_normed.ptr, mi, 0, 1, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, r_theta, r_orig, 1.0f32, stream);
                        cuda_ops::rad_cuda_kv_scatter_f16(self.d_kv_normed.ptr, self.d_kv_normed.ptr, self.comp_slots[cs_idx].comp_kv.ptr, self.comp_slots[cs_idx].comp_kv.ptr, mi, start_pos, self.max_ctx as i32, tc.head_dim as i32, stream);
                    } else {
                        cuda_ops::rad_cuda_kv_scatter_f16(self.d_kv.ptr, self.d_kv.ptr, self.comp_slots[cs_idx].raw_kv.ptr, self.comp_slots[cs_idx].raw_kv.ptr, mi, start_pos, self.max_ctx as i32, tc.head_dim as i32, stream);
                        if let (Some(ref wg), Some(ref rg)) = (&self.comp_slots[cs_idx].weights.wgate, &self.comp_slots[cs_idx].raw_gate) {
                            self.exl3_proj(self.d_normed.ptr, wg, self.d_kv.ptr, mi, 0, stream);
                            cuda_ops::rad_cuda_kv_scatter_f16(self.d_kv.ptr, self.d_kv.ptr, rg.ptr, rg.ptr, mi, start_pos, self.max_ctx as i32, tc.head_dim as i32, stream);
                            let c_start = start_pos / 2;
                            let c_end = (start_pos + mi) / 2;
                            if c_end > c_start {
                                cuda_ops::rad_cuda_csa2_compress_pair_f16(
                                    self.comp_slots[cs_idx].raw_kv.ptr,
                                    rg.ptr,
                                    self.comp_slots[cs_idx].weights.norm.ptr,
                                    self.comp_slots[cs_idx].comp_kv.ptr,
                                    c_start,
                                    c_end - c_start,
                                    tc.head_dim as i32,
                                    tc.qk_rope_head_dim as i32,
                                    r_theta,
                                    r_orig,
                                    tc.rms_norm_eps,
                                    stream,
                                );
                            }
                        }
                    }
                }

                let sink_ptr = (lw.attn_sink.ptr as *const f32).add(self.rank * n_heads_local);
                let sw = if is_comp { tc.sliding_window as i32 } else { 0 };
                let (comp_ptr, comp_ratio) = if is_comp {
                    if let Some(cs) = self.comp_slots.iter().rfind(|s| s.layer_id <= l) {
                        (cs.comp_kv.ptr as *const c_void, cs.weights.ratio as i32)
                    } else {
                        (std::ptr::null(), 0)
                    }
                } else {
                    (std::ptr::null(), 0)
                };
                cuda_ops::rad_cuda_mqa_attn_sink_causal_f16(
                    self.d_q.ptr,
                    self.kv_caches[l].k_cache.ptr,
                    self.kv_caches[l].k_cache.ptr,
                    comp_ptr,
                    sink_ptr,
                    self.d_attn_out.ptr,
                    self.d_z_scratch.ptr,
                    mi,
                    n_heads_local as i32,
                    start_pos,
                    self.max_ctx as i32,
                    sw,
                    comp_ratio,
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
                    r_theta,
                    r_orig,
                    -1.0f32,
                    stream,
                );
            }
            if use_graph {
                let gkey = (l, m);
                if !self.wo_graphs.contains_key(&gkey) {
                    let g = unsafe {
                        CudaCapturedGraph::capture(|gs| {
                            self.enqueue_wo_proj(l, m, branch_dst, gs);
                        })
                    }
                    .map_err(CudaFail)?;
                    self.wo_graphs.insert(gkey, g);
                }
                self.wo_graphs.get(&gkey).unwrap().launch_on(stream).map_err(CudaFail)?;
            } else {
                unsafe {
                    self.enqueue_wo_proj(l, m, branch_dst, stream);
                }
            }

            let (b_self, b_peer) = self.exchange_branch(m, comm)?;

            if use_graph {
                let gkey = (l, m);
                if !self.ffn_graphs.contains_key(&gkey) {
                    let g = unsafe {
                        CudaCapturedGraph::capture(|gs| {
                            self.enqueue_ffn_half(l, mi, b_self, b_peer, branch_dst, gs);
                        })
                    }
                    .map_err(CudaFail)?;
                    self.ffn_graphs.insert(gkey, g);
                }
                self.ffn_graphs.get(&gkey).unwrap().launch_on(stream).map_err(CudaFail)?;
            } else {
                unsafe {
                    self.enqueue_ffn_half(l, mi, b_self, b_peer, branch_dst, stream);
                }
            }

            let (b_self2, b_peer2) = self.exchange_branch(m, comm)?;

            unsafe {
                if l + 1 < self.layers.len() && self.engram_layers.is_empty() {
                    cuda_ops::rad_cuda_mhc_sinkhorn_mix_pre_rmsnorm_only(
                        self.d_streams.ptr,
                        b_self2,
                        b_peer2,
                        self.d_h_coeff.ptr as *const f32,
                        self.d_pre_b.ptr,
                        self.layers[l + 1].attn_norm.ptr,
                        self.d_normed.ptr,
                        mi,
                        d as i32,
                        tc.hc_sinkhorn_iters as i32,
                        tc.rms_norm_eps,
                        tc.hc_eps,
                        stream,
                    );
                } else {
                    cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(
                        self.d_streams.ptr,
                        b_self2,
                        b_peer2,
                        self.d_h_coeff.ptr as *const f32,
                        mi,
                        d as i32,
                        tc.hc_sinkhorn_iters as i32,
                        tc.hc_eps,
                        stream,
                    );
                }
            }
        }
        if update_mtp {
            if let Some(mtp) = self.mtp.take() {
                unsafe {
                    self.exl3_proj(mtp.d_taps_f16.ptr, &mtp.main_proj, mtp.d_main_proj.ptr, mi, 0, stream);
                    cuda_ops::rad_cuda_rms_norm_bf16w_f16(mtp.d_main_proj.ptr, mtp.main_norm.ptr, mtp.d_main_x.ptr, mi, d as i32, tc.rms_norm_eps / 256.0f32, stream);
                    for ml in 0..mtp.layers.len() {
                        let mlw = &mtp.layers[ml];
                        self.exl3_proj(mtp.d_main_x.ptr, &mlw.wkv, self.d_kv.ptr, mi, 0, stream);
                        cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, mlw.kv_norm.ptr, self.d_kv_normed.ptr, mi, tc.head_dim as i32, tc.rms_norm_eps, stream);
                        cuda_ops::rad_cuda_rope_yarn_f16(std::ptr::null_mut(), self.d_kv_normed.ptr, mi, 0, 1, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, tc.rope_theta, 0, 1.0f32, stream);
                        cuda_ops::rad_cuda_kv_scatter_f16(self.d_kv_normed.ptr, self.d_kv_normed.ptr, mtp.kv_caches[ml].k_cache.ptr, mtp.kv_caches[ml].k_cache.ptr, mi, start_pos, self.max_ctx as i32, tc.head_dim as i32, stream);
                    }
                }
                self.mtp = Some(mtp);
            }
        }
        self.pos += m;
        Ok(())
    }

    pub fn step_batch_no_head(&mut self, tokens: &[i32], comm: &mut Tp2Comm, update_mtp: bool) -> Result<()> {
        self.forward_layers(tokens, comm, update_mtp)
    }

    pub fn step_batch(&mut self, tokens: &[i32], comm: &mut Tp2Comm) -> Result<Vec<i32>> {
        let m = tokens.len();
        self.forward_layers(tokens, comm, true)?;
        let tc = &self.cfg.text_config;
        let d = tc.hidden_size;
        let stream: CudaStream = std::ptr::null_mut();
        unsafe {
            cuda_ops::rad_cuda_mhc_pre_mix_rmsnorm_only(self.d_streams.ptr, self.d_pre_b.ptr, self.final_norm.ptr, self.d_normed.ptr, m as i32, d as i32, tc.rms_norm_eps, stream);
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
        let mut best_vals = self.d_best_val.as_f32_slice()[..m].to_vec();
        let mut best_ids: Vec<i32> = unsafe { std::slice::from_raw_parts(self.d_best_id.ptr as *const i32, m) }.to_vec();
        let offset = (self.rank * self.head.n) as i32;
        for id in &mut best_ids {
            *id += offset;
        }
        comm.all_reduce_argmax(&mut best_vals, &mut best_ids)?;
        Ok(best_ids)
    }

    pub fn step_batch_last(&mut self, tokens: &[i32], comm: &mut Tp2Comm) -> Result<i32> {
        let m = tokens.len();
        self.forward_layers(tokens, comm, true)?;
        let tc = &self.cfg.text_config;
        let d = tc.hidden_size;
        let stream: CudaStream = std::ptr::null_mut();
        let last = m - 1;
        unsafe {
            let s_last = (self.d_streams.ptr as *const u8).add(last * 4 * d * 4) as *const c_void;
            let p_last = (self.d_pre_b.ptr as *const u8).add(last * 4 * 4) as *const c_void;
            cuda_ops::rad_cuda_mhc_pre_mix_rmsnorm_only(s_last, p_last, self.final_norm.ptr, self.d_normed.ptr, 1, d as i32, tc.rms_norm_eps, stream);
            self.exl3_proj(self.d_normed.ptr, &self.head, self.d_logits.ptr, 1, 0, stream);
            cuda_ops::rad_cuda_argmax_f16(
                self.d_logits.ptr,
                self.d_arg_vals.ptr,
                self.d_arg_ids.ptr,
                self.d_best_val.ptr,
                self.d_best_id.ptr,
                1,
                self.head.n as i32,
                stream,
            );
        }
        rad_cuda::sync_device().map_err(CudaFail)?;
        let mut best_vals = [self.d_best_val.as_f32_slice()[0]];
        let mut best_ids = [unsafe { *(self.d_best_id.ptr as *const i32) }];
        best_ids[0] += (self.rank * self.head.n) as i32;
        comm.all_reduce_argmax(&mut best_vals, &mut best_ids)?;
        Ok(best_ids[0])
    }

    pub fn step_decode(&mut self, token_id: i32, comm: &mut Tp2Comm) -> Result<i32> {
        let out = self.step_batch(&[token_id], comm)?;
        Ok(out[0])
    }

    pub fn profile_step(&mut self, token_id: i32, comm: &mut Tp2Comm) -> Result<()> {
        for &(m_prof, sp_prof) in &[(1usize, self.pos as i32), (2048usize, 2048i32)] {
            self.profile_batch_m(token_id, m_prof, sp_prof, comm)?;
        }
        Ok(())
    }

    fn profile_batch_m(&mut self, token_id: i32, m: usize, start_pos: i32, comm: &mut Tp2Comm) -> Result<()> {
        let tc = self.cfg.text_config.clone();
        let d = tc.hidden_size;
        let mi = m as i32;
        let n_heads_local = tc.num_attention_heads / self.world;
        let stream: CudaStream = std::ptr::null_mut();
        let attn_scale = 1.0f32 / (tc.head_dim as f32).sqrt();
        let branch_dst = comm.rdma_ptrs().map(|(s, _)| s).unwrap_or(self.d_branch_out.ptr);
        let sync_t = || -> f64 {
            let _ = rad_cuda::sync_device();
            0.0
        };
        let toks = vec![token_id; m];
        let tok_bytes = unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u8, m * 4) };
        self.d_tok_ids.as_bytes_mut()[..m * 4].copy_from_slice(tok_bytes);
        unsafe {
            cuda_ops::rad_cuda_embed_to_mhc_f16(self.embed.ptr, self.d_tok_ids.ptr as *const i32, self.d_streams.ptr, mi, d as i32, stream);
        }
        let _ = sync_t();
        let lw = &self.layers[0];
        let t0 = std::time::Instant::now();
        unsafe {
            cuda_ops::rad_cuda_mhc_pre_coeff_rmsnorm(self.d_streams.ptr, lw.hc_attn_fn.ptr, lw.hc_attn_base.ptr, lw.hc_attn_scale.ptr, std::ptr::null(), lw.attn_norm.ptr, self.d_h_coeff.ptr, self.d_pre_a.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, tc.hc_eps, stream);
        }
        let _ = sync_t();
        let t_mhc1 = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = std::time::Instant::now();
        unsafe {
            self.exl3_proj(self.d_normed.ptr, &lw.wq_a, self.d_q_lora.ptr, mi, 0, stream);
            cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_q_lora.ptr, lw.q_norm.ptr, self.d_q_normed.ptr, mi, tc.q_lora_rank as i32, tc.rms_norm_eps, stream);
            self.exl3_proj(self.d_q_normed.ptr, &lw.wq_b, self.d_q.ptr, mi, 0, stream);
            self.exl3_proj(self.d_normed.ptr, &lw.wkv, self.d_kv.ptr, mi, 0, stream);
            cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, lw.kv_norm.ptr, self.d_kv_normed.ptr, mi, tc.head_dim as i32, tc.rms_norm_eps, stream);
            cuda_ops::rad_cuda_rope_yarn_f16(self.d_q.ptr, self.d_kv_normed.ptr, mi, n_heads_local as i32, 1, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, tc.rope_theta, 0, 1.0f32, stream);
            cuda_ops::rad_cuda_kv_scatter_f16(self.d_kv_normed.ptr, self.d_kv_normed.ptr, self.kv_caches[0].k_cache.ptr, self.kv_caches[0].k_cache.ptr, mi, start_pos, self.max_ctx as i32, tc.head_dim as i32, stream);
        }
        let _ = sync_t();
        let t_qkv = t1.elapsed().as_secs_f64() * 1e6;

        let t2 = std::time::Instant::now();
        unsafe {
            let sink_ptr = (lw.attn_sink.ptr as *const f32).add(self.rank * n_heads_local);
            cuda_ops::rad_cuda_mqa_attn_sink_causal_f16(self.d_q.ptr, self.kv_caches[0].k_cache.ptr, self.kv_caches[0].k_cache.ptr, std::ptr::null(), sink_ptr, self.d_attn_out.ptr, self.d_z_scratch.ptr, mi, n_heads_local as i32, start_pos, self.max_ctx as i32, tc.sliding_window as i32, 0, tc.head_dim as i32, attn_scale, stream);
            cuda_ops::rad_cuda_rope_yarn_f16(self.d_attn_out.ptr, std::ptr::null_mut(), mi, n_heads_local as i32, 0, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, tc.rope_theta, 0, -1.0f32, stream);
        }
        let _ = sync_t();
        let t_attn = t2.elapsed().as_secs_f64() * 1e6;

        let t3 = std::time::Instant::now();
        unsafe {
            self.enqueue_wo_proj(0, m, branch_dst, stream);
        }
        let _ = sync_t();
        let t_wo = t3.elapsed().as_secs_f64() * 1e6;

        let t4 = std::time::Instant::now();
        let (b_self, b_peer) = self.exchange_branch(m, comm)?;
        let _ = sync_t();
        let t_ar1 = t4.elapsed().as_secs_f64() * 1e6;

        let lw = &self.layers[0];
        let t5 = std::time::Instant::now();
        unsafe {
            cuda_ops::rad_cuda_mhc_sinkhorn_pre_coeff_rmsnorm(self.d_streams.ptr, b_self, b_peer, lw.hc_ffn_fn.ptr, lw.hc_ffn_base.ptr, lw.hc_ffn_scale.ptr, self.d_pre_a.ptr, lw.ffn_norm.ptr, self.d_h_coeff.ptr, self.d_pre_b.ptr, self.d_normed.ptr, mi, d as i32, tc.hc_sinkhorn_iters as i32, tc.rms_norm_eps, tc.hc_eps, stream);
            cuda_ops::rad_cuda_moe_route_sqrtsoftplus(self.d_normed.ptr, lw.gate_w.ptr, lw.gate_bias.ptr, self.d_gate_raw.ptr, self.d_gate_sel.ptr, self.d_topk_ids.ptr, self.d_topk_weights.ptr, mi, tc.n_routed_experts as i32, d as i32, tc.num_experts_per_tok as i32, tc.routed_scaling_factor, stream);
        }
        let _ = sync_t();
        let t_route = t5.elapsed().as_secs_f64() * 1e6;

        let t6 = std::time::Instant::now();
        unsafe {
            self.exl3_proj(self.d_normed.ptr, &lw.shared_w1, self.d_shared_g.ptr, mi, 0, stream);
            self.exl3_proj_swiglu(self.d_normed.ptr, &lw.shared_w3, self.d_shared_g.ptr, lw.shared_w2.suh.ptr, self.d_shared_g.ptr, mi, tc.swiglu_limit, stream);
            self.exl3_proj_prerot(self.d_shared_g.ptr, &lw.shared_w2, self.d_shared_out.ptr, mi, 0, stream);
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
            cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w1_trel.ptr as *const *const c_void, lw.d_w1_suh.ptr as *const *const c_void, lw.d_w1_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, std::ptr::null(), self.d_moe_gate.ptr, self.d_moe_outs.ptr, tk, mi, d as i32, inter, bits, self.wht_scale, tc.swiglu_limit, 0, exp_start, n_local, 0, stream);
            cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w3_trel.ptr as *const *const c_void, lw.d_w3_suh.ptr as *const *const c_void, lw.d_w3_svh.ptr as *const *const c_void, lw.d_w2_suh.ptr as *const *const c_void, self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, self.d_moe_gate.ptr, self.d_moe_act.ptr, self.d_moe_outs.ptr, tk, mi, d as i32, inter, bits, self.wht_scale, tc.swiglu_limit, 0, exp_start, n_local, 1, stream);
            let w2_suh_ptr = if mi >= 512 { std::ptr::null() } else { lw.d_w2_suh.ptr as *const *const c_void };
            cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_moe_act.ptr, lw.d_w2_trel.ptr as *const *const c_void, w2_suh_ptr, lw.d_w2_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, std::ptr::null(), self.d_moe_outs.ptr, self.d_moe_act.ptr, tk, mi, inter, d as i32, bits, self.wht_scale, tc.swiglu_limit, 1, exp_start, n_local, 1, stream);
            cuda_ops::rad_cuda_moe_weighted_sum_f16(self.d_moe_outs.ptr, self.d_topk_weights.ptr as *const f32, self.d_topk_ids.ptr as *const i32, self.d_shared_out.ptr, branch_dst, mi, tk, d as i32, exp_start, n_local, stream);
        }
        let _ = sync_t();
        let t_moe = t7.elapsed().as_secs_f64() * 1e6;

        let t8 = std::time::Instant::now();
        let _ = self.exchange_branch(m, comm)?;
        let _ = sync_t();
        let t_ar2 = t8.elapsed().as_secs_f64() * 1e6;

        if self.rank == 0 {
            let tot = t_mhc1 + t_qkv + t_attn + t_wo + t_ar1 + t_route + t_shared + t_moe + t_ar2;
            println!(
                "PROFILE L0 M={m} pos={start_pos} (us): mhc1={t_mhc1:.0} qkv={t_qkv:.0} attn={t_attn:.0} wo={t_wo:.0} ar1={t_ar1:.0} route={t_route:.0} shared={t_shared:.0} moe={t_moe:.0} ar2={t_ar2:.0} | SUM={tot:.0} us ({:.1} ms x40)",
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
        let sw = self.cfg.text_config.sliding_window.max(4096);
        let mut idx = 0usize;
        while idx + MAX_BATCH < n {
            let chunk_end = idx + MAX_BATCH;
            let mut chunk = tokens[idx..chunk_end].to_vec();
            let need_mtp = n.saturating_sub(chunk_end) < sw;
            let mut cmd = if need_mtp { 1u32 } else { 7u32 };
            comm.broadcast_u32(&mut cmd)?;
            comm.broadcast_i32_vec(&mut chunk)?;
            self.step_batch_no_head(&chunk, comm, need_mtp)?;
            idx = chunk_end;
        }
        let mut last_chunk = tokens[idx..n].to_vec();
        let mut cmd = 6u32;
        comm.broadcast_u32(&mut cmd)?;
        comm.broadcast_i32_vec(&mut last_chunk)?;
        self.step_batch_last(&last_chunk, comm)
    }

    pub fn draft_dspark(&mut self, anchor: i32, n_blocks: usize, comm: &mut Tp2Comm) -> Result<Vec<i32>> {
        let Some(mtp) = self.mtp.take() else {
            return Ok(vec![anchor]);
        };
        let tc = self.cfg.text_config.clone();
        let d = tc.hidden_size;
        let blk = tc.dspark_block_size.max(2).min(8);
        let n_blk = n_blocks.max(1).min(3);
        let mi = blk as i32;
        let n_heads_local = tc.num_attention_heads / self.world;
        let groups_local = tc.o_groups / self.world;
        let stream: CudaStream = std::ptr::null_mut();
        let attn_scale = 1.0f32 / (tc.head_dim as f32).sqrt();
        let branch_dst = comm.rdma_ptrs().map(|(s, _)| s).unwrap_or(self.d_branch_out.ptr);
        let use_mtp_graph = comm.rdma_ptrs().is_some();
        let v_local = self.head.n;
        let m_rank = tc.dspark_markov_rank.max(1) as i32;
        let offset = (self.rank * v_local) as i32;
        let mut out = Vec::with_capacity(n_blk * blk + 1);
        out.push(anchor);

        for b_idx in 0..n_blk {
            let start_pos = (self.pos + b_idx * blk) as i32;
            let mut draft_in = vec![tc.dspark_noise_token_id; blk];
            draft_in[0] = *out.last().unwrap();
            let tok_bytes = unsafe { std::slice::from_raw_parts(draft_in.as_ptr() as *const u8, blk * 4) };
            self.d_tok_ids.as_bytes_mut()[..blk * 4].copy_from_slice(tok_bytes);

            unsafe {
                cuda_ops::rad_cuda_embed_to_mhc_f16(self.embed.ptr, self.d_tok_ids.ptr as *const i32, self.d_streams.ptr, mi, d as i32, stream);
            }

            for l in 0..mtp.layers.len() {
                if use_mtp_graph {
                    if !self.mtp_qkv_graphs.contains_key(&l) {
                        let g = unsafe {
                            CudaCapturedGraph::capture(|gs| {
                                let lw = &mtp.layers[l];
                                let prev_pre = if l == 0 { std::ptr::null() } else { self.d_pre_b.ptr as *const c_void };
                                cuda_ops::rad_cuda_mhc_pre_coeff_rmsnorm(self.d_streams.ptr, lw.hc_attn_fn.ptr, lw.hc_attn_base.ptr, lw.hc_attn_scale.ptr, prev_pre, lw.attn_norm.ptr, self.d_h_coeff.ptr, self.d_pre_a.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, tc.hc_eps, gs);
                                self.exl3_proj(self.d_normed.ptr, &lw.wq_a, self.d_q_lora.ptr, mi, 0, gs);
                                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_q_lora.ptr, lw.q_norm.ptr, self.d_q_normed.ptr, mi, tc.q_lora_rank as i32, tc.rms_norm_eps, gs);
                                self.exl3_proj(self.d_q_normed.ptr, &lw.wq_b, self.d_q.ptr, mi, 0, gs);
                                self.exl3_proj(self.d_normed.ptr, &lw.wkv, self.d_kv.ptr, mi, 0, gs);
                                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, lw.kv_norm.ptr, self.d_kv_normed.ptr, mi, tc.head_dim as i32, tc.rms_norm_eps, gs);
                            })
                        }
                        .map_err(CudaFail)?;
                        self.mtp_qkv_graphs.insert(l, g);
                    }
                    self.mtp_qkv_graphs.get(&l).unwrap().launch_on(stream).map_err(CudaFail)?;
                } else {
                    unsafe {
                        let lw = &mtp.layers[l];
                        let prev_pre = if l == 0 { std::ptr::null() } else { self.d_pre_b.ptr as *const c_void };
                        cuda_ops::rad_cuda_mhc_pre_coeff_rmsnorm(self.d_streams.ptr, lw.hc_attn_fn.ptr, lw.hc_attn_base.ptr, lw.hc_attn_scale.ptr, prev_pre, lw.attn_norm.ptr, self.d_h_coeff.ptr, self.d_pre_a.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, tc.hc_eps, stream);
                        self.exl3_proj(self.d_normed.ptr, &lw.wq_a, self.d_q_lora.ptr, mi, 0, stream);
                        cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_q_lora.ptr, lw.q_norm.ptr, self.d_q_normed.ptr, mi, tc.q_lora_rank as i32, tc.rms_norm_eps, stream);
                        self.exl3_proj(self.d_q_normed.ptr, &lw.wq_b, self.d_q.ptr, mi, 0, stream);
                        self.exl3_proj(self.d_normed.ptr, &lw.wkv, self.d_kv.ptr, mi, 0, stream);
                        cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, lw.kv_norm.ptr, self.d_kv_normed.ptr, mi, tc.head_dim as i32, tc.rms_norm_eps, stream);
                    }
                }
                unsafe {
                    let lw = &mtp.layers[l];
                    cuda_ops::rad_cuda_rope_yarn_f16(self.d_q.ptr, self.d_kv_normed.ptr, mi, n_heads_local as i32, 1, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, tc.rope_theta, 0, 1.0f32, stream);
                    let sink_ptr = (lw.attn_sink.ptr as *const f32).add(self.rank * n_heads_local);
                    cuda_ops::rad_cuda_mqa_attn_sink_dspark_f16(self.d_q.ptr, mtp.kv_caches[l].k_cache.ptr, self.d_z_scratch.ptr, self.d_kv_normed.ptr, sink_ptr, self.d_attn_out.ptr, mi, n_heads_local as i32, start_pos, self.max_ctx as i32, tc.sliding_window as i32, tc.head_dim as i32, attn_scale, stream);
                    if b_idx + 1 < n_blk {
                        cuda_ops::rad_cuda_kv_scatter_f16(self.d_kv_normed.ptr, self.d_kv_normed.ptr, mtp.kv_caches[l].k_cache.ptr, mtp.kv_caches[l].k_cache.ptr, mi, start_pos, self.max_ctx as i32, tc.head_dim as i32, stream);
                    }
                    cuda_ops::rad_cuda_rope_yarn_f16(self.d_attn_out.ptr, std::ptr::null_mut(), mi, n_heads_local as i32, 0, tc.head_dim as i32, tc.qk_rope_head_dim as i32, start_pos, tc.rope_theta, 0, -1.0f32, stream);
                }
                if use_mtp_graph {
                    if !self.mtp_wo_graphs.contains_key(&l) {
                        let g = unsafe {
                            CudaCapturedGraph::capture(|gs| {
                                let lw = &mtp.layers[l];
                                let s0 = &lw.wo_a_slices[0];
                                cuda_ops::rad_cuda_exl3_batched_group_mma_f16(
                                    self.d_attn_out.ptr,
                                    lw.d_wo_a_trel.ptr as *const *const c_void,
                                    lw.d_wo_a_suh.ptr as *const *const c_void,
                                    lw.d_wo_a_svh.ptr as *const *const c_void,
                                    self.d_z_scratch.ptr,
                                    self.d_wo_mid.ptr,
                                    std::ptr::null_mut(),
                                    mi,
                                    groups_local as i32,
                                    s0.k as i32,
                                    s0.n as i32,
                                    s0.bits as i32,
                                    self.wht_scale,
                                    gs,
                                );
                                self.exl3_proj(self.d_wo_mid.ptr, &lw.wo_b, branch_dst, mi, 0, gs);
                            })
                        }
                        .map_err(CudaFail)?;
                        self.mtp_wo_graphs.insert(l, g);
                    }
                    self.mtp_wo_graphs.get(&l).unwrap().launch_on(stream).map_err(CudaFail)?;
                } else {
                    unsafe {
                        let lw = &mtp.layers[l];
                        let s0 = &lw.wo_a_slices[0];
                        cuda_ops::rad_cuda_exl3_batched_group_mma_f16(
                            self.d_attn_out.ptr,
                            lw.d_wo_a_trel.ptr as *const *const c_void,
                            lw.d_wo_a_suh.ptr as *const *const c_void,
                            lw.d_wo_a_svh.ptr as *const *const c_void,
                            self.d_z_scratch.ptr,
                            self.d_wo_mid.ptr,
                            std::ptr::null_mut(),
                            mi,
                            groups_local as i32,
                            s0.k as i32,
                            s0.n as i32,
                            s0.bits as i32,
                            self.wht_scale,
                            stream,
                        );
                        self.exl3_proj(self.d_wo_mid.ptr, &lw.wo_b, branch_dst, mi, 0, stream);
                    }
                }
                let (b_self, b_peer) = self.exchange_branch(blk, comm)?;
                if use_mtp_graph {
                    if !self.mtp_ffn_graphs.contains_key(&l) {
                        let g = unsafe {
                            CudaCapturedGraph::capture(|gs| {
                                let lw = &mtp.layers[l];
                                cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(self.d_streams.ptr, b_self, b_peer, self.d_h_coeff.ptr as *const f32, mi, d as i32, tc.hc_sinkhorn_iters as i32, tc.hc_eps, gs);
                                cuda_ops::rad_cuda_mhc_pre_coeff_rmsnorm(self.d_streams.ptr, lw.hc_ffn_fn.ptr, lw.hc_ffn_base.ptr, lw.hc_ffn_scale.ptr, self.d_pre_a.ptr, lw.ffn_norm.ptr, self.d_h_coeff.ptr, self.d_pre_b.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, tc.hc_eps, gs);
                                let tk = tc.dspark_num_experts_per_tok as i32;
                                let inter = tc.moe_intermediate_size as i32;
                                let bits = lw.experts[0].w1.bits as i32;
                                let exp_start = lw.expert_start as i32;
                                let n_local = lw.experts.len() as i32;
                                cuda_ops::rad_cuda_moe_route_sqrtsoftplus(self.d_normed.ptr, lw.gate_w.ptr, lw.gate_bias.ptr, self.d_gate_raw.ptr, self.d_gate_sel.ptr, self.d_topk_ids.ptr, self.d_topk_weights.ptr, mi, tc.dspark_n_routed_experts as i32, d as i32, tk, tc.routed_scaling_factor, gs);
                                self.exl3_proj(self.d_normed.ptr, &lw.shared_w1, self.d_shared_g.ptr, mi, 0, gs);
                                self.exl3_proj_swiglu(self.d_normed.ptr, &lw.shared_w3, self.d_shared_g.ptr, lw.shared_w2.suh.ptr, self.d_shared_g.ptr, mi, tc.swiglu_limit, gs);
                                self.exl3_proj_prerot(self.d_shared_g.ptr, &lw.shared_w2, self.d_shared_out.ptr, mi, 0, gs);
                                cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w1_trel.ptr as *const *const c_void, lw.d_w1_suh.ptr as *const *const c_void, lw.d_w1_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, std::ptr::null(), self.d_moe_gate.ptr, std::ptr::null_mut(), tk, mi, d as i32, inter, bits, self.wht_scale, tc.swiglu_limit, 0, exp_start, n_local, 0, gs);
                                cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w3_trel.ptr as *const *const c_void, lw.d_w3_suh.ptr as *const *const c_void, lw.d_w3_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, self.d_moe_gate.ptr, self.d_moe_act.ptr, std::ptr::null_mut(), tk, mi, d as i32, inter, bits, self.wht_scale, tc.swiglu_limit, 0, exp_start, n_local, 1, gs);
                                cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_moe_act.ptr, lw.d_w2_trel.ptr as *const *const c_void, lw.d_w2_suh.ptr as *const *const c_void, lw.d_w2_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, std::ptr::null(), self.d_moe_outs.ptr, std::ptr::null_mut(), tk, mi, inter, d as i32, bits, self.wht_scale, tc.swiglu_limit, 1, exp_start, n_local, 1, gs);
                                cuda_ops::rad_cuda_moe_weighted_sum_f16(self.d_moe_outs.ptr, self.d_topk_weights.ptr as *const f32, self.d_topk_ids.ptr as *const i32, self.d_shared_out.ptr, branch_dst, mi, tk, d as i32, exp_start, n_local, gs);
                            })
                        }
                        .map_err(CudaFail)?;
                        self.mtp_ffn_graphs.insert(l, g);
                    }
                    self.mtp_ffn_graphs.get(&l).unwrap().launch_on(stream).map_err(CudaFail)?;
                } else {
                    unsafe {
                        let lw = &mtp.layers[l];
                        cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(self.d_streams.ptr, b_self, b_peer, self.d_h_coeff.ptr as *const f32, mi, d as i32, tc.hc_sinkhorn_iters as i32, tc.hc_eps, stream);
                        cuda_ops::rad_cuda_mhc_pre_coeff_rmsnorm(self.d_streams.ptr, lw.hc_ffn_fn.ptr, lw.hc_ffn_base.ptr, lw.hc_ffn_scale.ptr, self.d_pre_a.ptr, lw.ffn_norm.ptr, self.d_h_coeff.ptr, self.d_pre_b.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, tc.hc_eps, stream);
                        let tk = tc.dspark_num_experts_per_tok as i32;
                        let inter = tc.moe_intermediate_size as i32;
                        let bits = lw.experts[0].w1.bits as i32;
                        let exp_start = lw.expert_start as i32;
                        let n_local = lw.experts.len() as i32;
                        cuda_ops::rad_cuda_moe_route_sqrtsoftplus(self.d_normed.ptr, lw.gate_w.ptr, lw.gate_bias.ptr, self.d_gate_raw.ptr, self.d_gate_sel.ptr, self.d_topk_ids.ptr, self.d_topk_weights.ptr, mi, tc.dspark_n_routed_experts as i32, d as i32, tk, tc.routed_scaling_factor, stream);
                        self.exl3_proj(self.d_normed.ptr, &lw.shared_w1, self.d_shared_g.ptr, mi, 0, stream);
                        self.exl3_proj_swiglu(self.d_normed.ptr, &lw.shared_w3, self.d_shared_g.ptr, lw.shared_w2.suh.ptr, self.d_shared_g.ptr, mi, tc.swiglu_limit, stream);
                        self.exl3_proj_prerot(self.d_shared_g.ptr, &lw.shared_w2, self.d_shared_out.ptr, mi, 0, stream);
                        cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w1_trel.ptr as *const *const c_void, lw.d_w1_suh.ptr as *const *const c_void, lw.d_w1_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, std::ptr::null(), self.d_moe_gate.ptr, std::ptr::null_mut(), tk, mi, d as i32, inter, bits, self.wht_scale, tc.swiglu_limit, 0, exp_start, n_local, 0, stream);
                        cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_normed.ptr, lw.d_w3_trel.ptr as *const *const c_void, lw.d_w3_suh.ptr as *const *const c_void, lw.d_w3_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, self.d_moe_gate.ptr, self.d_moe_act.ptr, std::ptr::null_mut(), tk, mi, d as i32, inter, bits, self.wht_scale, tc.swiglu_limit, 0, exp_start, n_local, 1, stream);
                        cuda_ops::rad_cuda_exl3_moe_mma_f16(self.d_moe_act.ptr, lw.d_w2_trel.ptr as *const *const c_void, lw.d_w2_suh.ptr as *const *const c_void, lw.d_w2_svh.ptr as *const *const c_void, std::ptr::null(), self.d_topk_ids.ptr as *const i32, self.d_z_scratch.ptr, std::ptr::null(), self.d_moe_outs.ptr, std::ptr::null_mut(), tk, mi, inter, d as i32, bits, self.wht_scale, tc.swiglu_limit, 1, exp_start, n_local, 1, stream);
                        cuda_ops::rad_cuda_moe_weighted_sum_f16(self.d_moe_outs.ptr, self.d_topk_weights.ptr as *const f32, self.d_topk_ids.ptr as *const i32, self.d_shared_out.ptr, branch_dst, mi, tk, d as i32, exp_start, n_local, stream);
                    }
                }
                let (b_self2, b_peer2) = self.exchange_branch(blk, comm)?;
                unsafe {
                    cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(self.d_streams.ptr, b_self2, b_peer2, self.d_h_coeff.ptr as *const f32, mi, d as i32, tc.hc_sinkhorn_iters as i32, tc.hc_eps, stream);
                }
            }
            unsafe {
                cuda_ops::rad_cuda_mhc_pre_mix_rmsnorm_only(self.d_streams.ptr, self.d_pre_b.ptr, mtp.final_norm.ptr, self.d_normed.ptr, mi, d as i32, tc.rms_norm_eps, stream);
                self.exl3_proj(self.d_normed.ptr, &self.head, self.d_logits.ptr, mi, 0, stream);
            }
            for i in 0..blk {
                let prev_tok = *out.last().unwrap();
                unsafe {
                    let row_ptr = (self.d_logits.ptr as *const u8).add(i * v_local * 2) as *const c_void;
                    cuda_ops::rad_cuda_markov_argmax_f16(
                        row_ptr,
                        mtp.markov_head.ptr,
                        mtp.markov_embed.ptr,
                        prev_tok,
                        self.d_arg_vals.ptr,
                        self.d_arg_ids.ptr,
                        self.d_best_val.ptr,
                        self.d_best_id.ptr,
                        v_local as i32,
                        m_rank,
                        stream,
                    );
                }
                rad_cuda::sync_device().map_err(CudaFail)?;
                let mut bval = [self.d_best_val.as_f32_slice()[0]];
                let mut bid = [unsafe { *(self.d_best_id.ptr as *const i32) }];
                bid[0] += offset;
                comm.all_reduce_argmax(&mut bval, &mut bid)?;
                out.push(bid[0]);
            }
        }
        self.mtp = Some(mtp);
        Ok(out)
    }

    pub fn step_speculative(&mut self, history: &[i32], anchor: i32, comm: &mut Tp2Comm) -> Result<Vec<i32>> {
        static TRACE_SPEC: OnceLock<bool> = OnceLock::new();
        let trace = *TRACE_SPEC.get_or_init(|| std::env::var("RAD_TRACE_SPEC").map(|v| v == "1").unwrap_or(false));
        let max_k_lim = if self.last_acc >= 64 { MAX_DECODE_BATCH } else { 128 };
        let max_k = max_k_lim.min(self.max_ctx.saturating_sub(self.pos + 1));
        let t_draft = Instant::now();
        let in_edit = self.last_acc >= 10;
        let (mut draft, mode) = if max_k >= 4 {
            if let Some(ng) = ngram_draft_cfg(history, anchor, max_k, in_edit) {
                (ng, "ngram")
            } else if self.mtp.is_some() {
                let n_blk = 1usize;
                let mut dcmd = 5u32;
                comm.broadcast_u32(&mut dcmd)?;
                let mut dargs = [anchor as u32, n_blk as u32];
                comm.broadcast_u32_array(&mut dargs)?;
                (self.draft_dspark(anchor, n_blk, comm)?, "dspark")
            } else {
                (vec![anchor], "single")
            }
        } else {
            (vec![anchor], "single")
        };
        let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;
        self.last_draft_len = draft.len();
        if draft.len() > 1 {
            let start_pos = self.pos;
            let t_ver = Instant::now();
            let mut cmd = 2u32;
            comm.broadcast_u32(&mut cmd)?;
            comm.broadcast_i32_vec(&mut draft)?;
            let preds = self.step_batch(&draft, comm)?;
            let ver_ms = t_ver.elapsed().as_secs_f64() * 1000.0;
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
            if mode == "dspark" {
                if k_acc == draft.len() {
                    self.dspark_blocks = (self.dspark_blocks + 1).min(3);
                } else if k_acc <= 3 {
                    self.dspark_blocks = 1;
                } else if k_acc + 2 < draft.len() {
                    self.dspark_blocks = self.dspark_blocks.saturating_sub(1).max(1);
                }
            }
            if trace && self.rank == 0 {
                println!(
                    "[spec] mode={mode} draft={} acc={k_acc} draft_ms={draft_ms:.1} ver_ms={ver_ms:.1}",
                    draft.len()
                );
            }
            self.last_acc = k_acc;
            let new_pos = start_pos + k_acc;
            self.pos = new_pos;
            if let Some(ref mut mtp) = self.mtp {
                mtp.last_ctx_row = k_acc.saturating_sub(1).min(15);
            }
            let mut pcmd = 3u32;
            comm.broadcast_u32(&mut pcmd)?;
            let mut pargs = [new_pos as u32, k_acc.saturating_sub(1).min(15) as u32];
            comm.broadcast_u32_array(&mut pargs)?;
            return Ok(accepted);
        }
        let mut one = vec![anchor];
        let mut cmd = 2u32;
        comm.broadcast_u32(&mut cmd)?;
        comm.broadcast_i32_vec(&mut one)?;
        let preds = self.step_batch(&one, comm)?;
        if let Some(ref mut mtp) = self.mtp {
            mtp.last_ctx_row = 0;
        }
        Ok(preds)
    }
}
