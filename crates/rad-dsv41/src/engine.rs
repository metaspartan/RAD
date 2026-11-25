use anyhow::Result;
use core::ffi::c_void;
use rad_cuda::comm::Tp2Comm;
use rad_cuda::{CudaBuf, CudaFail, CudaStream};
use std::path::Path;
use crate::config::DeepseekV41Config;
use crate::cuda_ops;
use crate::safetensors::SafetensorsIndex;
use crate::weights::{load_raw_buf, Exl3Linear, LayerWeights};

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
    pub d_attn_out: CudaBuf,
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
    pub d_tok_id: CudaBuf,
    pub d_arg_vals: CudaBuf,
    pub d_arg_ids: CudaBuf,
    pub d_best_val: CudaBuf,
    pub d_best_id: CudaBuf,
    pub pos: usize,
    pub wht_scale: f32,
}

impl Dsv41Engine {
    pub fn load(
        model_dir: &Path,
        max_ctx: usize,
        rank: usize,
        world: usize,
        max_layers: Option<usize>,
    ) -> Result<Self> {
        let cfg = DeepseekV41Config::load(model_dir)?;
        let tc = cfg.text_config.clone();
        let mut idx = SafetensorsIndex::open(model_dir)?;

        let embed = load_raw_buf(&mut idx, "embed.weight")?;
        let final_norm = load_raw_buf(&mut idx, "norm.weight")?;
        let head = Exl3Linear::load_col_shard(&mut idx, "head", rank, world)?;

        let n_layers = max_layers.unwrap_or(tc.num_hidden_layers).min(tc.num_hidden_layers);
        let mut layers = Vec::with_capacity(n_layers);
        let mut kv_caches = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(LayerWeights::load(&mut idx, &cfg, l, rank, world)?);
            kv_caches.push(KvCacheLayer {
                k_cache: CudaBuf::new(max_ctx * tc.head_dim * 2).map_err(CudaFail)?,
                v_cache: CudaBuf::new(max_ctx * tc.head_dim * 2).map_err(CudaFail)?,
            });
        }

        let d = tc.hidden_size;
        let n_heads_local = tc.num_attention_heads / world;
        let max_n = tc.vocab_size / world;
        let max_m = 16usize;

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
            d_streams: CudaBuf::new(4 * d * 2).map_err(CudaFail)?,
            d_branch_in: CudaBuf::new(d * 2).map_err(CudaFail)?,
            d_normed: CudaBuf::new(d * 2).map_err(CudaFail)?,
            d_xrot: CudaBuf::new(max_m * 32768 * 2).map_err(CudaFail)?,
            d_z_scratch: CudaBuf::new(max_m * max_n * 4).map_err(CudaFail)?,
            d_q_lora: CudaBuf::new(tc.q_lora_rank * 2).map_err(CudaFail)?,
            d_q_normed: CudaBuf::new(tc.q_lora_rank * 2).map_err(CudaFail)?,
            d_q: CudaBuf::new(n_heads_local * tc.head_dim * 2).map_err(CudaFail)?,
            d_kv: CudaBuf::new(tc.head_dim * 2).map_err(CudaFail)?,
            d_kv_normed: CudaBuf::new(tc.head_dim * 2).map_err(CudaFail)?,
            d_attn_out: CudaBuf::new(n_heads_local * tc.head_dim * 2).map_err(CudaFail)?,
            d_wo_mid: CudaBuf::new((tc.o_groups / world) * tc.o_lora_rank * 2).map_err(CudaFail)?,
            d_branch_out: CudaBuf::new(d * 2).map_err(CudaFail)?,
            d_h_coeff: CudaBuf::new(24 * 4).map_err(CudaFail)?,
            d_gate_raw: CudaBuf::new(tc.n_routed_experts * 4).map_err(CudaFail)?,
            d_gate_sel: CudaBuf::new(tc.n_routed_experts * 4).map_err(CudaFail)?,
            d_topk_ids: CudaBuf::new(tc.num_experts_per_tok * 4).map_err(CudaFail)?,
            d_topk_weights: CudaBuf::new(tc.num_experts_per_tok * 4).map_err(CudaFail)?,
            d_moe_gate: CudaBuf::new(tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_up: CudaBuf::new(tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_act: CudaBuf::new(tc.moe_intermediate_size * 2).map_err(CudaFail)?,
            d_moe_outs: CudaBuf::new(tc.num_experts_per_tok * d * 2).map_err(CudaFail)?,
            d_shared_g: CudaBuf::new((tc.moe_intermediate_size / world) * 2).map_err(CudaFail)?,
            d_shared_u: CudaBuf::new((tc.moe_intermediate_size / world) * 2).map_err(CudaFail)?,
            d_shared_out: CudaBuf::new(d * 2).map_err(CudaFail)?,
            d_logits: CudaBuf::new(max_n * 2).map_err(CudaFail)?,
            d_tok_id: CudaBuf::new(4).map_err(CudaFail)?,
            d_arg_vals: CudaBuf::new(256 * 4).map_err(CudaFail)?,
            d_arg_ids: CudaBuf::new(256 * 4).map_err(CudaFail)?,
            d_best_val: CudaBuf::new(4).map_err(CudaFail)?,
            d_best_id: CudaBuf::new(4).map_err(CudaFail)?,
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

    pub fn step_decode(&mut self, token_id: i32, comm: &mut Tp2Comm) -> Result<i32> {
        let tc = &self.cfg.text_config;
        let d = tc.hidden_size;
        let n_heads_local = tc.num_attention_heads / self.world;
        let stream: CudaStream = std::ptr::null_mut();

        self.d_tok_id.copy_from(&token_id.to_le_bytes(), None).map_err(CudaFail)?;
        unsafe {
            cuda_ops::rad_cuda_embed_bf16_to_f16(self.embed.ptr, self.d_tok_id.ptr as *const i32, self.d_branch_in.ptr, 1, d as i32, stream);
            for r in 0..4 {
                let dst = (self.d_streams.ptr as *mut u8).add(r * d * 2) as *mut c_void;
                rad_cuda::cudaMemcpyAsync(dst, self.d_branch_in.ptr, d * 2, rad_cuda::CUDA_MEMCPY_DEVICE_TO_DEVICE, stream);
            }
        }

        let seq_len = (self.pos + 1).min(self.max_ctx);
        let attn_scale = 1.0f32 / (tc.head_dim as f32).sqrt();
        let mut host_ids = vec![0i32; tc.num_experts_per_tok];
        let mut host_weights = vec![0.0f32; tc.num_experts_per_tok];
        let mut ar_buf = vec![0.0f32; d];
        let mut f16_tmp = vec![0u16; d];

        for l in 0..self.layers.len() {
            let lw = &self.layers[l];
            unsafe {
                cuda_ops::rad_cuda_mhc_pre_and_coeff(
                    self.d_streams.ptr,
                    lw.hc_attn_fn.ptr,
                    lw.hc_attn_base.ptr,
                    lw.hc_attn_scale.ptr,
                    self.d_h_coeff.ptr,
                    self.d_branch_in.ptr,
                    d as i32,
                    stream,
                );
                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_branch_in.ptr, lw.attn_norm.ptr, self.d_normed.ptr, 1, d as i32, tc.rms_norm_eps, stream);
                self.exl3_proj(self.d_normed.ptr, &lw.wq_a, self.d_q_lora.ptr, 1, 0, stream);
                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_q_lora.ptr, lw.q_norm.ptr, self.d_q_normed.ptr, 1, tc.q_lora_rank as i32, tc.rms_norm_eps, stream);
                self.exl3_proj(self.d_q_normed.ptr, &lw.wq_b, self.d_q.ptr, 1, 0, stream);

                self.exl3_proj(self.d_normed.ptr, &lw.wkv, self.d_kv.ptr, 1, 0, stream);
                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_kv.ptr, lw.kv_norm.ptr, self.d_kv_normed.ptr, 1, tc.head_dim as i32, tc.rms_norm_eps, stream);
                cuda_ops::rad_cuda_rope_yarn_f16(
                    self.d_q.ptr,
                    self.d_kv_normed.ptr,
                    n_heads_local as i32,
                    1,
                    tc.head_dim as i32,
                    tc.qk_rope_head_dim as i32,
                    self.pos as i32,
                    tc.rope_theta,
                    stream,
                );
                let kv_off = (self.pos % self.max_ctx) * tc.head_dim * 2;
                let k_dst = (self.kv_caches[l].k_cache.ptr as *mut u8).add(kv_off) as *mut c_void;
                let v_dst = (self.kv_caches[l].v_cache.ptr as *mut u8).add(kv_off) as *mut c_void;
                rad_cuda::cudaMemcpyAsync(k_dst, self.d_kv_normed.ptr, tc.head_dim * 2, rad_cuda::CUDA_MEMCPY_DEVICE_TO_DEVICE, stream);
                rad_cuda::cudaMemcpyAsync(v_dst, self.d_kv.ptr, tc.head_dim * 2, rad_cuda::CUDA_MEMCPY_DEVICE_TO_DEVICE, stream);

                let sink_ptr = (lw.attn_sink.ptr as *const f32).add(self.rank * n_heads_local);
                cuda_ops::rad_cuda_mqa_attn_sink_decode_f16(
                    self.d_q.ptr,
                    self.kv_caches[l].k_cache.ptr,
                    self.kv_caches[l].v_cache.ptr,
                    sink_ptr,
                    self.d_attn_out.ptr,
                    n_heads_local as i32,
                    seq_len as i32,
                    tc.head_dim as i32,
                    attn_scale,
                    stream,
                );

                let heads_per_grp = tc.num_attention_heads / tc.o_groups;
                for (gi, wo_slice) in lw.wo_a_slices.iter().enumerate() {
                    let x_ptr = (self.d_attn_out.ptr as *const u8).add(gi * heads_per_grp * tc.head_dim * 2) as *const c_void;
                    let y_ptr = (self.d_wo_mid.ptr as *mut u8).add(gi * tc.o_lora_rank * 2) as *mut c_void;
                    self.exl3_proj(x_ptr, wo_slice, y_ptr, 1, 0, stream);
                }
                self.exl3_proj(self.d_wo_mid.ptr, &lw.wo_b, self.d_branch_out.ptr, 1, 0, stream);
            }

            if self.world > 1 {
                rad_cuda::sync_device().map_err(CudaFail)?;
                let raw_bytes = unsafe { std::slice::from_raw_parts_mut(f16_tmp.as_mut_ptr() as *mut u8, d * 2) };
                self.d_branch_out.copy_to_host(raw_bytes).map_err(CudaFail)?;
                for i in 0..d {
                    ar_buf[i] = f16_to_f32(f16_tmp[i]);
                }
                comm.all_reduce_sum_f32(&mut ar_buf)?;
                for i in 0..d {
                    f16_tmp[i] = f32_to_f16(ar_buf[i]);
                }
                let raw_in = unsafe { std::slice::from_raw_parts(f16_tmp.as_ptr() as *const u8, d * 2) };
                self.d_branch_out.copy_from(raw_in, None).map_err(CudaFail)?;
            }

            unsafe {
                cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(
                    self.d_streams.ptr,
                    self.d_branch_out.ptr,
                    self.d_h_coeff.ptr as *const f32,
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
                    d as i32,
                    stream,
                );
                cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_branch_in.ptr, lw.ffn_norm.ptr, self.d_normed.ptr, 1, d as i32, tc.rms_norm_eps, stream);
                cuda_ops::rad_cuda_moe_route_sqrtsoftplus(
                    self.d_normed.ptr,
                    lw.gate_w.ptr,
                    lw.gate_bias.ptr,
                    self.d_gate_raw.ptr,
                    self.d_gate_sel.ptr,
                    self.d_topk_ids.ptr,
                    self.d_topk_weights.ptr,
                    tc.n_routed_experts as i32,
                    d as i32,
                    tc.num_experts_per_tok as i32,
                    tc.routed_scaling_factor,
                    stream,
                );
                self.exl3_proj(self.d_normed.ptr, &lw.shared_w1, self.d_shared_g.ptr, 1, 0, stream);
                self.exl3_proj(self.d_normed.ptr, &lw.shared_w3, self.d_shared_u.ptr, 1, 0, stream);
                cuda_ops::rad_cuda_clamped_swiglu_f16(
                    self.d_shared_g.ptr,
                    self.d_shared_u.ptr,
                    self.d_shared_g.ptr,
                    (tc.moe_intermediate_size / self.world) as i32,
                    tc.swiglu_limit,
                    stream,
                );
                self.exl3_proj(self.d_shared_g.ptr, &lw.shared_w2, self.d_shared_out.ptr, 1, 0, stream);
            }

            rad_cuda::sync_device().map_err(CudaFail)?;
            let id_bytes = unsafe { std::slice::from_raw_parts_mut(host_ids.as_mut_ptr() as *mut u8, host_ids.len() * 4) };
            let w_bytes = unsafe { std::slice::from_raw_parts_mut(host_weights.as_mut_ptr() as *mut u8, host_weights.len() * 4) };
            self.d_topk_ids.copy_to_host(id_bytes).map_err(CudaFail)?;
            self.d_topk_weights.copy_to_host(w_bytes).map_err(CudaFail)?;

            let exp_end = lw.expert_start + lw.experts.len();
            let mut local_k = 0usize;
            let mut local_weights = Vec::with_capacity(tc.num_experts_per_tok);
            for slot in 0..tc.num_experts_per_tok {
                let eid = host_ids[slot] as usize;
                if eid >= lw.expert_start && eid < exp_end {
                    let ew = &lw.experts[eid - lw.expert_start];
                    unsafe {
                        self.exl3_proj(self.d_normed.ptr, &ew.w1, self.d_moe_gate.ptr, 1, 0, stream);
                        self.exl3_proj(self.d_normed.ptr, &ew.w3, self.d_moe_up.ptr, 1, 0, stream);
                        cuda_ops::rad_cuda_clamped_swiglu_f16(
                            self.d_moe_gate.ptr,
                            self.d_moe_up.ptr,
                            self.d_moe_act.ptr,
                            tc.moe_intermediate_size as i32,
                            tc.swiglu_limit,
                            stream,
                        );
                        let out_ptr = (self.d_moe_outs.ptr as *mut u8).add(local_k * d * 2) as *mut c_void;
                        self.exl3_proj(self.d_moe_act.ptr, &ew.w2, out_ptr, 1, 0, stream);
                    }
                    local_weights.push(host_weights[slot]);
                    local_k += 1;
                }
            }
            if local_k > 0 {
                let lw_bytes = unsafe { std::slice::from_raw_parts(local_weights.as_ptr() as *const u8, local_k * 4) };
                self.d_topk_weights.copy_from(lw_bytes, None).map_err(CudaFail)?;
                unsafe {
                    cuda_ops::rad_cuda_moe_weighted_sum_f16(
                        self.d_moe_outs.ptr,
                        self.d_topk_weights.ptr as *const f32,
                        self.d_shared_out.ptr,
                        self.d_branch_out.ptr,
                        local_k as i32,
                        d as i32,
                        stream,
                    );
                }
            } else {
                unsafe {
                    rad_cuda::cudaMemcpyAsync(self.d_branch_out.ptr, self.d_shared_out.ptr, d * 2, rad_cuda::CUDA_MEMCPY_DEVICE_TO_DEVICE, stream);
                }
            }

            if self.world > 1 {
                rad_cuda::sync_device().map_err(CudaFail)?;
                let raw_bytes = unsafe { std::slice::from_raw_parts_mut(f16_tmp.as_mut_ptr() as *mut u8, d * 2) };
                self.d_branch_out.copy_to_host(raw_bytes).map_err(CudaFail)?;
                for i in 0..d {
                    ar_buf[i] = f16_to_f32(f16_tmp[i]);
                }
                comm.all_reduce_sum_f32(&mut ar_buf)?;
                for i in 0..d {
                    f16_tmp[i] = f32_to_f16(ar_buf[i]);
                }
                let raw_in = unsafe { std::slice::from_raw_parts(f16_tmp.as_ptr() as *const u8, d * 2) };
                self.d_branch_out.copy_from(raw_in, None).map_err(CudaFail)?;
            }

            unsafe {
                cuda_ops::rad_cuda_mhc_sinkhorn_mix_f16(
                    self.d_streams.ptr,
                    self.d_branch_out.ptr,
                    self.d_h_coeff.ptr as *const f32,
                    d as i32,
                    tc.hc_sinkhorn_iters as i32,
                    tc.hc_eps,
                    stream,
                );
            }
        }

        unsafe {
            cuda_ops::rad_cuda_rms_norm_bf16w_f16(self.d_streams.ptr, self.final_norm.ptr, self.d_normed.ptr, 1, d as i32, tc.rms_norm_eps, stream);
            self.exl3_proj(self.d_normed.ptr, &self.head, self.d_logits.ptr, 1, 0, stream);
            cuda_ops::rad_cuda_argmax_f16(
                self.d_logits.ptr,
                self.d_arg_vals.ptr,
                self.d_arg_ids.ptr,
                self.d_best_val.ptr,
                self.d_best_id.ptr,
                self.head.n as i32,
                stream,
            );
        }
        rad_cuda::sync_device().map_err(CudaFail)?;
        let mut best_id = 0i32;
        self.d_best_id.copy_to_host(unsafe { std::slice::from_raw_parts_mut(&mut best_id as *mut i32 as *mut u8, 4) }).map_err(CudaFail)?;
        self.pos += 1;
        Ok(best_id + (self.rank * self.head.n) as i32)
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let f = (bits & 0x3ff) as u32;
    let out = if e == 0 {
        if f == 0 {
            s << 31
        } else {
            let mut m = f;
            let mut exp = 127 - 14;
            while (m & 0x400) == 0 {
                m <<= 1;
                exp -= 1;
            }
            (s << 31) | (exp << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | 0x7f800000 | (f << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (f << 13)
    };
    f32::from_bits(out)
}

fn f32_to_f16(val: f32) -> u16 {
    let b = val.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let f = b & 0x7fffff;
    if e <= 0 {
        s
    } else if e >= 31 {
        s | 0x7c00
    } else {
        s | ((e as u16) << 10) | ((f >> 13) as u16)
    }
}
