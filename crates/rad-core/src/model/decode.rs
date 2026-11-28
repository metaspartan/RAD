use super::*;

impl InklingEngine {
    pub fn reset(&mut self) -> Result<()> {
        for sl in &mut self.slots {
            sl.k_hist.zero().map_err(he)?;
            sl.v_hist.zero().map_err(he)?;
            sl.attn_post_hist.zero().map_err(he)?;
            sl.ffn_post_hist.zero().map_err(he)?;
        }
        unsafe { rad_hip::hipStreamSynchronize(std::ptr::null_mut()) };
        self.pos = 0;
        for p in &mut self.seq_pos {
            *p = 0;
        }
        for t in &mut self.seq_tokens {
            t.clear();
        }
        for s in 0..self.max_seqs {
            self.paged_kv.free_seq(s);
        }
        Ok(())
    }

    unsafe fn enqueue_layer(&self, s: HipStream, il: usize, ip: usize, n_blocks: &mut usize, loop_idx: Option<usize>) {
        let l = &self.layers[ip];
        let sl = &self.slots[il];
        let e = self.cfg.n_embd as i32;
        let kv_dim = (self.cfg.n_head_kv * self.cfg.dh) as i32;
        let q_dim = (self.cfg.n_head * self.cfg.dh) as i32;
        let r_dim = (self.cfg.n_head * self.cfg.d_rel) as i32;

        let cur = self.sc.cur.ptr as *mut f32;
        let ffn_inp = self.sc.ffn_inp.ptr as *mut f32;
        let blocks = self.sc.blocks.ptr as *mut f32;
        let mixed = self.sc.mixed.ptr as *mut f32;
        let msgs = self.sc.msgs.ptr as *const f32;
        let inp0 = self.sc.inp0.ptr as *mut f32;
        let inp1 = self.sc.inp1.ptr as *mut f32;
        let inp2 = self.sc.inp2.ptr as *mut f32;

        let (p0, p1, p2): (*const f32, *const f32, *const f32);
        if let Some(ref rq) = l.attn_res_query {
            let nq = if l.is_swa { 1 } else { 3 };
            if let Some(li) = loop_idx {
                ops::ink_attn_res_mix(s, blocks, *n_blocks as i32, cur, rq.ptr as *const f32, mixed, e, nq);
                let ll = &self.loop_layers[li];
                let rank = self.cfg.gate_rank as i32;
                let ch = self.sc.ctrl_h.ptr as *mut f32;
                let s0 = mixed as *const f32;
                let s1 = if l.is_swa { mixed as *const f32 } else { mixed.add(self.cfg.n_embd) as *const f32 };
                let s2 = if l.is_swa { mixed as *const f32 } else { mixed.add(2 * self.cfg.n_embd) as *const f32 };
                let (nw0, nw1, nw2) = if l.is_swa {
                    let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
                    (nw, nw, nw)
                } else {
                    (
                        l.attn_q_inp_norm.as_ref().unwrap().ptr as *const f32,
                        l.attn_k_inp_norm.as_ref().unwrap().ptr as *const f32,
                        l.attn_v_inp_norm.as_ref().unwrap().ptr as *const f32,
                    )
                };
                ops::ink_loop_ctrl_3_norm(
                    s, s0, s1, s2, msgs,
                    ll.ctrl_down[0].ptr as *const f32, ll.ctrl_down[1].ptr as *const f32, ll.ctrl_down[2].ptr as *const f32,
                    ll.ctrl_gain[0].ptr as *const f32, ll.ctrl_gain[1].ptr as *const f32, ll.ctrl_gain[2].ptr as *const f32,
                    ll.ctrl_forget[0].ptr as *const f32, ll.ctrl_forget[1].ptr as *const f32, ll.ctrl_forget[2].ptr as *const f32,
                    nw0, nw1, nw2, ch, inp0, inp1, inp2, e, rank, self.cfg.eps,
                );
                p0 = inp0;
                p1 = inp1;
                p2 = inp2;
            } else if l.is_swa {
                let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
                ops::ink_mix_rmsnorm(s, blocks, *n_blocks as i32, cur, rq.ptr as *const f32, nw, nw, nw, inp0, e, 1, self.cfg.eps);
                p0 = inp0;
                p1 = inp0;
                p2 = inp0;
            } else {
                ops::ink_mix_rmsnorm(
                    s, blocks, *n_blocks as i32, cur, rq.ptr as *const f32,
                    l.attn_q_inp_norm.as_ref().unwrap().ptr as *const f32,
                    l.attn_k_inp_norm.as_ref().unwrap().ptr as *const f32,
                    l.attn_v_inp_norm.as_ref().unwrap().ptr as *const f32,
                    mixed, e, 3, self.cfg.eps,
                );
                p0 = mixed;
                p1 = mixed.add(self.cfg.n_embd);
                p2 = mixed.add(2 * self.cfg.n_embd);
            }
        } else {
            let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
            ops::ink_rmsnorm(s, cur, nw, inp0, e, self.cfg.eps);
            p0 = inp0;
            p1 = inp0;
            p2 = inp0;
        }

        let q = self.sc.q.ptr as *mut f32;
        let k_raw = self.sc.k_raw.ptr as *mut f32;
        let v_raw = self.sc.v_raw.ptr as *mut f32;
        let r = self.sc.r.ptr as *mut f32;
        let sinks = self.sc.sinks.ptr as *mut f32;
        let gate_out = self.sc.gate_out.ptr as *mut f32;
        let k_dyn = self.sc.k_dyn.ptr as *mut f32;
        let v_dyn = self.sc.v_dyn.ptr as *mut f32;

        let (wkdyn_ptr, wvdyn_ptr, dyn_dim) = if l.is_swa {
            (std::ptr::null(), std::ptr::null(), 0)
        } else {
            (
                l.attn_k_conv_proj.as_ref().unwrap().buf.ptr as *const u16,
                l.attn_v_conv_proj.as_ref().unwrap().buf.ptr as *const u16,
                (4 * self.cfg.n_head_kv) as i32,
            )
        };

        let xq_stride = MAX_BATCH * ((q_dim as usize) / 32) * 40;
        let xq0 = self.bsc.xq.ptr;
        let xq1 = (self.bsc.xq.ptr as *mut u8).add(xq_stride) as *mut core::ffi::c_void;
        let xq2 = (self.bsc.xq.ptr as *mut u8).add(2 * xq_stride) as *mut core::ffi::c_void;
        if l.wq.ty == WeightType::Q4_0 || self.b1_sudot4 {
            let (q1, q2) = if p0 == p1 && p0 == p2 {
                ops::ink_quantize_q8_1(s, p0, xq0, e, 1);
                (xq0 as *const core::ffi::c_void, xq0 as *const core::ffi::c_void)
            } else {
                ops::ink_quantize_q8_1_x3(s, p0, p1, p2, xq0, xq1, xq2, e, 1);
                (xq1 as *const core::ffi::c_void, xq2 as *const core::ffi::c_void)
            };
            ops::ink_qkvrg_gemm_sudot4(
                s,
                l.wq.buf.ptr as *const u8,
                l.wk.buf.ptr as *const u8,
                l.wv.buf.ptr as *const u8,
                l.wrel.buf.ptr as *const u8,
                l.wqkv_gate.buf.ptr as *const u8,
                l.attn_sink_proj.buf.ptr as *const u16,
                wkdyn_ptr,
                wvdyn_ptr,
                xq0, q1, q2,
                p0, p1, p2,
                q, k_raw, v_raw, r, gate_out, sinks, k_dyn, v_dyn,
                e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, dyn_dim,
                1, if l.wq.ty == WeightType::Q4_0 { 1 } else { 0 },
            );
        } else {
            ops::ink_qkvrg_gemv(
                s,
                l.wq.buf.ptr as *const u8,
                l.wk.buf.ptr as *const u8,
                l.wv.buf.ptr as *const u8,
                l.wrel.buf.ptr as *const u8,
                l.wqkv_gate.buf.ptr as *const u8,
                l.attn_sink_proj.buf.ptr as *const u16,
                wkdyn_ptr,
                wvdyn_ptr,
                p0, p1, p2,
                q, k_raw, v_raw, r, gate_out, sinks, k_dyn, v_dyn,
                e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, dyn_dim,
            );
        }

        let d_pos = self.sc.d_pos.ptr as *const i32;
        if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
            ops::ink_kv_conv_qkvr_store_b1_q8kv(
                s, q, k_raw, v_raw, r,
                l.attn_k_conv.ptr as *const f32,
                l.attn_v_conv.ptr as *const f32,
                if l.is_swa { std::ptr::null() } else { k_dyn },
                if l.is_swa { std::ptr::null() } else { v_dyn },
                sl.k_hist.ptr as *mut f32,
                sl.v_hist.ptr as *mut f32,
                l.attn_q_clip.ptr as *const f32,
                l.attn_rel_clip.ptr as *const f32,
                sl.k_cache.ptr as *mut i8,
                sl.v_cache.ptr as *mut i8,
                ks.ptr as *mut u16,
                vs.ptr as *mut u16,
                d_pos,
                self.cfg.n_head as i32,
                self.cfg.n_head_kv as i32,
                self.cfg.dh as i32,
                self.cfg.d_rel as i32,
                sl.max_ctx as i32,
                self.cfg.qk_clip,
                self.cfg.rel_clip,
                self.cfg.log_floor,
                self.cfg.log_alpha,
            );
        } else {
            ops::ink_kv_conv_qkvr_store(
                s, q, k_raw, v_raw, r,
                l.attn_k_conv.ptr as *const f32,
                l.attn_v_conv.ptr as *const f32,
                if l.is_swa { std::ptr::null() } else { k_dyn },
                if l.is_swa { std::ptr::null() } else { v_dyn },
                sl.k_hist.ptr as *mut f32,
                sl.v_hist.ptr as *mut f32,
                l.attn_q_clip.ptr as *const f32,
                l.attn_rel_clip.ptr as *const f32,
                sl.k_cache.ptr as *mut u16,
                sl.v_cache.ptr as *mut u16,
                d_pos,
                if sl.is_swa { 1 } else { 0 },
                self.cfg.n_head as i32,
                self.cfg.n_head_kv as i32,
                self.cfg.dh as i32,
                self.cfg.d_rel as i32,
                sl.max_ctx as i32,
                self.cfg.qk_clip,
                self.cfg.rel_clip,
                self.cfg.log_floor,
                self.cfg.log_alpha,
            );
        }

        let block_dst = if ip % self.cfg.attn_res_block == 0 {
            let dst = blocks.add((*n_blocks) * self.cfg.n_embd);
            *n_blocks += 1;
            dst
        } else {
            std::ptr::null_mut()
        };

        if l.wo.ty == WeightType::Q4_0 {
            if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
                ops::ink_attn_decode_gated_q8_q8kv(
                    s, q, r, sinks,
                    l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const i8,
                    sl.v_cache.ptr as *const i8,
                    ks.ptr as *const u16,
                    vs.ptr as *const u16,
                    gate_out,
                    d_pos, xq0,
                    self.cfg.n_head as i32,
                    self.cfg.n_head_kv as i32,
                    self.cfg.dh as i32,
                    self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32,
                    sl.max_ctx as i32,
                );
            } else {
                ops::ink_attn_decode_gated_q8(
                    s, q, r, sinks,
                    l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const u16,
                    sl.v_cache.ptr as *const u16,
                    gate_out,
                    d_pos, xq0,
                    self.cfg.n_head as i32,
                    self.cfg.n_head_kv as i32,
                    self.cfg.dh as i32,
                    self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32,
                    self.cfg.swa_window as i32,
                    if sl.is_swa { 1 } else { 0 },
                    sl.max_ctx as i32,
                );
            }
            ops::ink_wo_conv_res_q4_0_sudot4(
                s,
                l.wo.buf.ptr as *const u8,
                xq0,
                l.attn_post_conv.ptr as *const f32,
                sl.attn_post_hist.ptr as *mut f32,
                cur,
                block_dst,
                ffn_inp,
                e, q_dim,
            );
        } else {
            let attn_out = self.sc.attn_out.ptr as *mut f32;
            ops::ink_attn_decode_gated(
                s, q, r, sinks,
                l.attn_rel_bias.ptr as *const f32,
                sl.k_cache.ptr as *const u16,
                sl.v_cache.ptr as *const u16,
                gate_out,
                d_pos, attn_out,
                self.cfg.n_head as i32,
                self.cfg.n_head_kv as i32,
                self.cfg.dh as i32,
                self.cfg.d_rel as i32,
                self.cfg.rel_extent as i32,
                self.cfg.swa_window as i32,
                if sl.is_swa { 1 } else { 0 },
                sl.max_ctx as i32,
            );
            ops::ink_wo_conv_res(
                s,
                l.wo.buf.ptr as *const u8,
                attn_out,
                l.attn_post_conv.ptr as *const f32,
                sl.attn_post_hist.ptr as *mut f32,
                cur,
                block_dst,
                ffn_inp,
                e, q_dim,
            );
        }

        let ffn_normed = self.sc.ffn_normed.ptr as *mut f32;
        if let Some(li) = loop_idx {
            ops::ink_attn_res_mix(s, blocks, *n_blocks as i32, ffn_inp, l.ffn_res_query.ptr as *const f32, mixed, e, 1);
            let ll = &self.loop_layers[li];
            let rank = self.cfg.gate_rank as i32;
            let ch = self.sc.ctrl_h.ptr as *mut f32;
            ops::ink_loop_ctrl_1_norm(
                s, mixed, msgs.add(3 * self.cfg.n_embd),
                ll.ctrl_down[3].ptr as *const f32,
                ll.ctrl_gain[3].ptr as *const f32,
                ll.ctrl_forget[3].ptr as *const f32,
                l.ffn_norm.ptr as *const f32,
                ch, ffn_normed, e, rank, self.cfg.eps,
            );
        } else {
            let fnw = l.ffn_norm.ptr as *const f32;
            ops::ink_mix_rmsnorm(s, blocks, *n_blocks as i32, ffn_inp, l.ffn_res_query.ptr as *const f32, fnw, fnw, fnw, ffn_normed, e, 1, self.cfg.eps);
        }

        let router_logits = self.sc.router_logits.ptr as *mut f32;
        l.ffn_gate_inp.gemv(s, ffn_normed, router_logits);
        let bias_ptr = l.ffn_exp_probs_b.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let b1_fused_rq = l.ffn_gate_up_exps.ty == WeightType::Q4_0 && self.cfg.n_expert == 128 && self.cfg.n_expert_used == 8;
        if b1_fused_rq {
            ops::ink_b1_router_and_q8(
                s, router_logits, bias_ptr,
                self.sc.wtop.ptr as *mut f32, self.sc.itop.ptr as *mut i32,
                ffn_normed, xq0, self.cfg.route_scale, e,
            );
        } else {
            ops::ink_moe_router(
                s,
                router_logits,
                bias_ptr,
                self.sc.wtop.ptr as *mut f32,
                self.sc.itop.ptr as *mut i32,
                self.cfg.n_expert as i32,
                self.cfg.n_expert_used as i32,
                self.cfg.route_scale,
            );
        }

        let gu_sc = l.ffn_gate_up_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let dn_sc = l.ffn_down_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let moe_act = self.sc.moe_act.ptr as *mut f32;

        match l.ffn_gate_up_exps.ty {
            WeightType::Nvfp4 => {
                if self.b1_sudot4 {
                    ops::ink_quantize_q8_1(s, ffn_normed, xq0, e, 1);
                    ops::ink_moe_gu_nvfp4_b1_sudot4(
                        s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, xq0,
                        self.sc.itop.ptr as *const i32, moe_act,
                        self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                        self.cfg.situ_beta, self.cfg.situ_lb,
                    );
                } else {
                    ops::ink_moe_gate_up_situ_nvfp4(
                        s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, ffn_normed,
                        self.sc.itop.ptr as *const i32, moe_act,
                        self.cfg.n_ff_exp as i32, e, self.cfg.situ_beta, self.cfg.situ_lb,
                    );
                }
            }
            WeightType::Q4_0 => {
                if !b1_fused_rq {
                    ops::ink_quantize_q8_1(s, ffn_normed, xq0, e, 1);
                }
                ops::ink_moe_gu_q4_0_b1_to_q8(
                    s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, xq0,
                    self.sc.itop.ptr as *const i32, self.bsc.act_q.ptr,
                    self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                    self.cfg.situ_beta, self.cfg.situ_lb,
                );
            }
            WeightType::Q8_0 => {
                ops::ink_moe_gate_up_situ_q8_0(
                    s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, ffn_normed,
                    self.sc.itop.ptr as *const i32, moe_act,
                    self.cfg.n_ff_exp as i32, e, self.cfg.situ_beta, self.cfg.situ_lb,
                );
            }
            _ => unreachable!(),
        }

        match l.ffn_down_exps.ty {
            WeightType::Nvfp4 => {
                if self.b1_sudot4 {
                    ops::ink_quantize_q8_1(s, moe_act, self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32);
                    ops::ink_moe_dn_conv_add_nvfp4_b1_sudot4(
                        s, l.ffn_down_exps.buf.ptr as *const u8, dn_sc, self.bsc.act_q.ptr,
                        self.sc.itop.ptr as *const i32, self.sc.wtop.ptr as *const f32,
                        l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32,
                        ffn_inp, cur, e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32,
                    );
                } else {
                    ops::ink_moe_down_conv_add_nvfp4(
                        s, l.ffn_down_exps.buf.ptr as *const u8, dn_sc, moe_act,
                        self.sc.itop.ptr as *const i32, self.sc.wtop.ptr as *const f32,
                        l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32,
                        ffn_inp, cur, e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32,
                    );
                }
            }
            WeightType::Q4_0 => {
                ops::ink_moe_dn_conv_add_q4_0_b1_sudot4(
                    s, l.ffn_down_exps.buf.ptr as *const u8, dn_sc, self.bsc.act_q.ptr,
                    self.sc.itop.ptr as *const i32, self.sc.wtop.ptr as *const f32,
                    l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32,
                    ffn_inp, cur, e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32,
                );
            }
            WeightType::Q8_0 => {
                ops::ink_moe_down_conv_add_q8_0(
                    s, l.ffn_down_exps.buf.ptr as *const u8, dn_sc, moe_act,
                    self.sc.itop.ptr as *const i32, self.sc.wtop.ptr as *const f32,
                    l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32,
                    ffn_inp, cur, e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32,
                );
            }
            _ => unreachable!(),
        }
    }

    pub unsafe fn enqueue_forward(&self, s: HipStream) {
        let e = self.cfg.n_embd as i32;
        let cur = self.sc.cur.ptr as *mut f32;
        let blocks = self.sc.blocks.ptr as *mut f32;
        let prelude = self.sc.prelude.ptr as *mut f32;
        let bank = self.sc.bank.ptr as *mut f32;
        let bank_out = self.sc.bank_out.ptr as *mut f32;
        let msgs = self.sc.msgs.ptr as *mut f32;
        let feats = self.sc.dspark_feats.ptr as *mut f32;

        ops::ink_embed_q8_0_norm(
            s,
            self.tok_embd.buf.ptr as *const u8,
            self.tok_norm.ptr as *const f32,
            self.sc.d_tok.ptr as *const i32,
            cur,
            e,
        );

        let mut n_blocks = 0usize;
        let mut visit = 0usize;

        for il in 0..self.cfg.core_start {
            self.enqueue_layer(s, il, il, &mut n_blocks, None);
            visit += 1;
            if visit == 5 {
                ops::ink_copy_f32(s, cur, feats, e);
            }
        }

        let prelude_n_blocks = n_blocks;
        ops::ink_copy_f32(s, cur, prelude, e);

        for pass in 0..self.cfg.n_passes {
            let mut bank_len = 0usize;
            if pass > 0 {
                let start_b = prelude_n_blocks + 1;
                bank_len = n_blocks - start_b;
                for b in 0..bank_len {
                    ops::ink_copy_f32(s, blocks.add((start_b + b) * self.cfg.n_embd), bank.add(b * self.cfg.n_embd), e);
                }
                ops::ink_copy_f32(s, cur, bank_out, e);
                ops::ink_copy_f32(s, prelude, cur, e);
                n_blocks = prelude_n_blocks;
            }
            for ip in self.cfg.core_start..self.cfg.core_end {
                let il = if pass == 0 {
                    ip
                } else {
                    self.cfg.n_phys + (pass - 1) * (self.cfg.core_end - self.cfg.core_start) + (ip - self.cfg.core_start)
                };
                if pass == 0 {
                    self.enqueue_layer(s, il, ip, &mut n_blocks, None);
                } else {
                    let li = ip - self.cfg.core_start;
                    let ll = &self.loop_layers[li];
                    ops::ink_attn_res_mix(s, bank, bank_len as i32, bank_out, ll.msg_query.ptr as *const f32, msgs, e, 4);
                    self.enqueue_layer(s, il, ip, &mut n_blocks, Some(li));
                }
                visit += 1;
                if visit == 23 {
                    ops::ink_copy_f32(s, cur, feats.add(self.cfg.n_embd), e);
                } else if visit == 41 {
                    ops::ink_copy_f32(s, cur, feats.add(2 * self.cfg.n_embd), e);
                }
            }
        }

        for il in self.cfg.core_end..self.cfg.n_phys {
            self.enqueue_layer(s, il, il, &mut n_blocks, None);
            visit += 1;
            if visit == 48 {
                ops::ink_copy_f32(s, cur, feats.add(3 * self.cfg.n_embd), e);
            }
        }

        let mixed = self.sc.mixed.ptr as *mut f32;
        ops::ink_attn_res_mix(s, blocks, n_blocks as i32, cur, self.output_res_query.ptr as *const f32, mixed, e, 1);
        ops::ink_rmsnorm(s, mixed, self.output_norm.ptr as *const f32, cur, e, self.cfg.eps);
        match self.output.ty {
            WeightType::Q4_0 => {
                ops::ink_quantize_q8_1(s, cur, self.bsc.xq.ptr, e, 1);
                ops::ink_gemm_q4_0_sudot4(s, self.output.buf.ptr as *const u8, self.bsc.xq.ptr, self.sc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, 1);
            }
            _ => {
                self.output.gemv(s, cur, self.sc.logits.ptr as *mut f32);
            }
        }
        ops::ink_softcap_argmax(
            s,
            self.sc.logits.ptr as *mut f32,
            self.cfg.final_softcap,
            self.cfg.vocab_unpadded as i32,
            self.sc.d_out_tok.ptr as *mut i32,
        );
    }

    pub fn capture_decode_graph(&mut self) -> Result<()> {
        let g = unsafe { CapturedGraph::capture(|s| self.enqueue_forward(s)) }.map_err(he)?;
        self.decode_graph = Some(g);
        Ok(())
    }

    pub fn step(&mut self, token: i32) -> Result<i32> {
        let pos = self.pos;
        if let Some(ref g) = self.decode_graph {
            self.sc.d_tok.copy_from(&token.to_le_bytes(), Some(g.stream)).map_err(he)?;
            self.sc.d_pos.copy_from(&pos.to_le_bytes(), Some(g.stream)).map_err(he)?;
            g.launch().map_err(he)?;
            g.sync().map_err(he)?;
        } else {
            self.sc.d_tok.copy_from(&token.to_le_bytes(), Some(self.stream)).map_err(he)?;
            self.sc.d_pos.copy_from(&pos.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe {
                self.enqueue_forward(self.stream);
                rad_hip::hipStreamSynchronize(self.stream);
            }
        }
        let mut out_b = [0u8; 4];
        self.sc.d_out_tok.copy_to_host(&mut out_b).map_err(he)?;
        self.pos += 1;
        Ok(i32::from_le_bytes(out_b))
    }

    unsafe fn enqueue_layer_batch(&self, s: HipStream, il: usize, ip: usize, n_blocks: &mut usize, loop_idx: Option<usize>, b: usize) {
        let l = &self.layers[ip];
        let sl = &self.slots[il];
        let e = self.cfg.n_embd as i32;
        let mb = MAX_BATCH as i32;
        let bi = b as i32;
        let kv_dim = (self.cfg.n_head_kv * self.cfg.dh) as i32;
        let q_dim = (self.cfg.n_head * self.cfg.dh) as i32;
        let r_dim = (self.cfg.n_head * self.cfg.d_rel) as i32;

        let cur = self.bsc.cur.ptr as *mut f32;
        let ffn_inp = self.bsc.ffn_inp.ptr as *mut f32;
        let blocks = self.bsc.blocks.ptr as *mut f32;
        let mixed = self.bsc.mixed.ptr as *mut f32;
        let msgs = self.bsc.msgs.ptr as *const f32;
        let inp0 = self.bsc.inp0.ptr as *mut f32;
        let inp1 = self.bsc.inp1.ptr as *mut f32;
        let inp2 = self.bsc.inp2.ptr as *mut f32;

        let (p0, p1, p2): (*const f32, *const f32, *const f32);
        if let Some(ref rq) = l.attn_res_query {
            let nq = if l.is_swa { 1 } else { 3 };
            if let Some(li) = loop_idx {
                ops::ink_mix_batch(s, blocks, *n_blocks as i32, mb, cur, rq.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), mixed, e, nq, bi, self.cfg.eps);
                let ll = &self.loop_layers[li];
                let rank = self.cfg.gate_rank as i32;
                let ch = self.bsc.ctrl_h.ptr as *mut f32;
                let s0 = mixed as *const f32;
                let s1 = if l.is_swa { mixed as *const f32 } else { mixed.add(MAX_BATCH * self.cfg.n_embd) as *const f32 };
                let s2 = if l.is_swa { mixed as *const f32 } else { mixed.add(2 * MAX_BATCH * self.cfg.n_embd) as *const f32 };
                let (nw0, nw1, nw2) = if l.is_swa {
                    let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
                    (nw, nw, nw)
                } else {
                    (
                        l.attn_q_inp_norm.as_ref().unwrap().ptr as *const f32,
                        l.attn_k_inp_norm.as_ref().unwrap().ptr as *const f32,
                        l.attn_v_inp_norm.as_ref().unwrap().ptr as *const f32,
                    )
                };
                ops::ink_loop_ctrl_3_norm_batch(
                    s, s0, s1, s2, msgs,
                    ll.ctrl_down[0].ptr as *const f32, ll.ctrl_down[1].ptr as *const f32, ll.ctrl_down[2].ptr as *const f32,
                    ll.ctrl_gain[0].ptr as *const f32, ll.ctrl_gain[1].ptr as *const f32, ll.ctrl_gain[2].ptr as *const f32,
                    ll.ctrl_forget[0].ptr as *const f32, ll.ctrl_forget[1].ptr as *const f32, ll.ctrl_forget[2].ptr as *const f32,
                    nw0, nw1, nw2, ch, inp0, inp1, inp2, e, rank, mb, bi, self.cfg.eps,
                );
                p0 = inp0; p1 = inp1; p2 = inp2;
            } else if l.is_swa {
                let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
                ops::ink_mix_batch(s, blocks, *n_blocks as i32, mb, cur, rq.ptr as *const f32, nw, nw, nw, inp0, e, 1, bi, self.cfg.eps);
                p0 = inp0; p1 = inp0; p2 = inp0;
            } else {
                ops::ink_mix_batch(
                    s, blocks, *n_blocks as i32, mb, cur, rq.ptr as *const f32,
                    l.attn_q_inp_norm.as_ref().unwrap().ptr as *const f32,
                    l.attn_k_inp_norm.as_ref().unwrap().ptr as *const f32,
                    l.attn_v_inp_norm.as_ref().unwrap().ptr as *const f32,
                    mixed, e, 3, bi, self.cfg.eps,
                );
                p0 = mixed;
                p1 = mixed.add(MAX_BATCH * self.cfg.n_embd);
                p2 = mixed.add(2 * MAX_BATCH * self.cfg.n_embd);
            }
        } else {
            let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
            ops::ink_rmsnorm_batch(s, cur, nw, inp0, e, bi, self.cfg.eps);
            p0 = inp0; p1 = inp0; p2 = inp0;
        }

        let q = self.bsc.q.ptr as *mut f32;
        let k_raw = self.bsc.k_raw.ptr as *mut f32;
        let v_raw = self.bsc.v_raw.ptr as *mut f32;
        let r = self.bsc.r.ptr as *mut f32;
        let sinks = self.bsc.sinks.ptr as *mut f32;
        let gate_out = self.bsc.gate_out.ptr as *mut f32;
        let k_dyn = self.bsc.k_dyn.ptr as *mut f32;
        let v_dyn = self.bsc.v_dyn.ptr as *mut f32;

        let (wkdyn_ptr, wvdyn_ptr, dyn_dim) = if l.is_swa {
            (std::ptr::null(), std::ptr::null(), 0)
        } else {
            (
                l.attn_k_conv_proj.as_ref().unwrap().buf.ptr as *const u16,
                l.attn_v_conv_proj.as_ref().unwrap().buf.ptr as *const u16,
                (4 * self.cfg.n_head_kv) as i32,
            )
        };

        let xq_stride = MAX_BATCH * ((q_dim as usize) / 32) * 40;
        let xq0 = self.bsc.xq.ptr;
        let xq1 = (self.bsc.xq.ptr as *mut u8).add(xq_stride) as *mut core::ffi::c_void;
        let xq2 = (self.bsc.xq.ptr as *mut u8).add(2 * xq_stride) as *mut core::ffi::c_void;
        let (q1, q2) = if p0 == p1 && p0 == p2 {
            ops::ink_quantize_q8_1(s, p0, xq0, e, bi);
            (xq0 as *const core::ffi::c_void, xq0 as *const core::ffi::c_void)
        } else {
            ops::ink_quantize_q8_1_x3(s, p0, p1, p2, xq0, xq1, xq2, e, bi);
            (xq1 as *const core::ffi::c_void, xq2 as *const core::ffi::c_void)
        };
        if l.wq.ty == WeightType::Q4_0 && b >= 12 {
            ops::ink_qkvrg_gemm_q4_wmma(
                s,
                l.wq.buf.ptr as *const u8,
                l.wk.buf.ptr as *const u8,
                l.wv.buf.ptr as *const u8,
                l.wrel.buf.ptr as *const u8,
                l.wqkv_gate.buf.ptr as *const u8,
                l.attn_sink_proj.buf.ptr as *const u16,
                wkdyn_ptr, wvdyn_ptr,
                xq0, q1, q2,
                p0, p1, p2,
                q, k_raw, v_raw, r, gate_out, sinks, k_dyn, v_dyn,
                e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, dyn_dim,
                bi,
            );
        } else {
            ops::ink_qkvrg_gemm_sudot4(
                s,
                l.wq.buf.ptr as *const u8,
                l.wk.buf.ptr as *const u8,
                l.wv.buf.ptr as *const u8,
                l.wrel.buf.ptr as *const u8,
                l.wqkv_gate.buf.ptr as *const u8,
                l.attn_sink_proj.buf.ptr as *const u16,
                wkdyn_ptr, wvdyn_ptr,
                xq0, q1, q2,
                p0, p1, p2,
                q, k_raw, v_raw, r, gate_out, sinks, k_dyn, v_dyn,
                e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, dyn_dim,
                bi, if l.wq.ty == WeightType::Q4_0 { 1 } else { 0 },
            );
        }

        let d_pos0 = self.bsc.d_pos0.ptr as *const i32;
        let snap_lim = self.multi_verify_cap.min(512);
        let k_snap = if b <= snap_lim { sl.k_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
        let v_snap = if b <= snap_lim { sl.v_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
        if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
            ops::ink_kv_conv_qkvr_store_batch_snap_q8kv(
                s, q, k_raw, v_raw, r,
                l.attn_k_conv.ptr as *const f32,
                l.attn_v_conv.ptr as *const f32,
                if l.is_swa { std::ptr::null() } else { k_dyn },
                if l.is_swa { std::ptr::null() } else { v_dyn },
                sl.k_hist.ptr as *mut f32,
                sl.v_hist.ptr as *mut f32,
                k_snap,
                v_snap,
                l.attn_q_clip.ptr as *const f32,
                l.attn_rel_clip.ptr as *const f32,
                sl.k_cache.ptr as *mut i8,
                sl.v_cache.ptr as *mut i8,
                ks.ptr as *mut u16,
                vs.ptr as *mut u16,
                d_pos0, bi,
                self.cfg.n_head as i32,
                self.cfg.n_head_kv as i32,
                self.cfg.dh as i32,
                self.cfg.d_rel as i32,
                sl.max_ctx as i32,
                self.cfg.qk_clip,
                self.cfg.rel_clip,
                self.cfg.log_floor,
                self.cfg.log_alpha,
            );
        } else {
            ops::ink_kv_conv_qkvr_store_batch_snap(
                s, q, k_raw, v_raw, r,
                l.attn_k_conv.ptr as *const f32,
                l.attn_v_conv.ptr as *const f32,
                if l.is_swa { std::ptr::null() } else { k_dyn },
                if l.is_swa { std::ptr::null() } else { v_dyn },
                sl.k_hist.ptr as *mut f32,
                sl.v_hist.ptr as *mut f32,
                k_snap,
                v_snap,
                l.attn_q_clip.ptr as *const f32,
                l.attn_rel_clip.ptr as *const f32,
                sl.k_cache.ptr as *mut u16,
                sl.v_cache.ptr as *mut u16,
                d_pos0, bi,
                if sl.is_swa { 1 } else { 0 },
                self.cfg.n_head as i32,
                self.cfg.n_head_kv as i32,
                self.cfg.dh as i32,
                self.cfg.d_rel as i32,
                sl.max_ctx as i32,
                self.cfg.qk_clip,
                self.cfg.rel_clip,
                self.cfg.log_floor,
                self.cfg.log_alpha,
            );
        }

        let attn_out = self.bsc.attn_out.ptr as *mut f32;
        if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
            if b > 128 {
                ops::ink_attn_batch_gated_flash_q8kv(
                    s, q, r, sinks,
                    l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const i8,
                    sl.v_cache.ptr as *const i8,
                    ks.ptr as *const u16,
                    vs.ptr as *const u16,
                    gate_out, d_pos0, attn_out,
                    self.cfg.n_head as i32,
                    self.cfg.n_head_kv as i32,
                    self.cfg.dh as i32,
                    self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32,
                    sl.max_ctx as i32, bi,
                );
                ops::ink_quantize_q8_1(s, attn_out, self.bsc.xq.ptr, q_dim, bi);
            } else {
                ops::ink_attn_b8_flash_q8_q8kv(
                    s, q, r, sinks,
                    l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const i8,
                    sl.v_cache.ptr as *const i8,
                    ks.ptr as *const u16,
                    vs.ptr as *const u16,
                    gate_out, d_pos0, self.bsc.xq.ptr,
                    self.cfg.n_head as i32,
                    self.cfg.n_head_kv as i32,
                    self.cfg.dh as i32,
                    self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32,
                    sl.max_ctx as i32, bi,
                );
            }
        } else if b >= 16 {
            ops::ink_attn_batch_gated_flash(
                s, q, r, sinks,
                l.attn_rel_bias.ptr as *const f32,
                sl.k_cache.ptr as *const u16,
                sl.v_cache.ptr as *const u16,
                gate_out, d_pos0, attn_out,
                self.cfg.n_head as i32,
                self.cfg.n_head_kv as i32,
                self.cfg.dh as i32,
                self.cfg.d_rel as i32,
                self.cfg.rel_extent as i32,
                self.cfg.swa_window as i32,
                if sl.is_swa { 1 } else { 0 },
                sl.max_ctx as i32, bi,
            );
            ops::ink_quantize_q8_1(s, attn_out, self.bsc.xq.ptr, q_dim, bi);
        } else {
            ops::ink_attn_batch_gated_q8(
                s, q, r, sinks,
                l.attn_rel_bias.ptr as *const f32,
                sl.k_cache.ptr as *const u16,
                sl.v_cache.ptr as *const u16,
                gate_out, d_pos0, self.bsc.xq.ptr,
                self.cfg.n_head as i32,
                self.cfg.n_head_kv as i32,
                self.cfg.dh as i32,
                self.cfg.d_rel as i32,
                self.cfg.rel_extent as i32,
                self.cfg.swa_window as i32,
                if sl.is_swa { 1 } else { 0 },
                sl.max_ctx as i32, bi,
            );
        }

        let o_raw = self.bsc.o_raw.ptr as *mut f32;
        let block_dst = if ip % self.cfg.attn_res_block == 0 {
            let dst = blocks.add((*n_blocks) * MAX_BATCH * self.cfg.n_embd);
            *n_blocks += 1;
            dst
        } else {
            std::ptr::null_mut()
        };
        let ap_snap = if b <= self.multi_verify_cap.min(512) { sl.attn_post_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
        if l.wo.ty == WeightType::Q4_0 && (b == 2 || b == 3 || b == 4 || b == 6 || b == 8) {
            ops::ink_wo_conv_res_q4_batch_snap(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, ap_snap, cur, block_dst, ffn_inp, e, q_dim, bi);
        } else {
            match l.wo.ty {
                WeightType::Q4_0 if b >= 12 => ops::ink_gemm_q4_0_wmma(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
                WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
                _ => ops::ink_gemm_q8_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
            };
            ops::ink_shortconv_res_batch_snap(s, o_raw, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, ap_snap, cur, block_dst, ffn_inp, e, bi);
        }

        let ffn_normed = self.bsc.ffn_normed.ptr as *mut f32;
        if let Some(li) = loop_idx {
            ops::ink_mix_batch(s, blocks, *n_blocks as i32, mb, ffn_inp, l.ffn_res_query.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), mixed, e, 1, bi, self.cfg.eps);
            let ll = &self.loop_layers[li];
            let rank = self.cfg.gate_rank as i32;
            let ch = self.bsc.ctrl_h.ptr as *mut f32;
            ops::ink_loop_ctrl_1_norm_batch(
                s, mixed, msgs.add(3 * MAX_BATCH * self.cfg.n_embd),
                ll.ctrl_down[3].ptr as *const f32,
                ll.ctrl_gain[3].ptr as *const f32,
                ll.ctrl_forget[3].ptr as *const f32,
                l.ffn_norm.ptr as *const f32,
                ch, ffn_normed, e, rank, mb, bi, self.cfg.eps,
            );
        } else {
            let fnw = l.ffn_norm.ptr as *const f32;
            ops::ink_mix_batch(s, blocks, *n_blocks as i32, mb, ffn_inp, l.ffn_res_query.ptr as *const f32, fnw, fnw, fnw, ffn_normed, e, 1, bi, self.cfg.eps);
        }

        let router_logits = self.bsc.router_logits.ptr as *mut f32;
        ops::ink_gemm_f32(s, l.ffn_gate_inp.buf.ptr as *const f32, ffn_normed, router_logits, self.cfg.n_expert as i32, e, bi);
        let bias_ptr = l.ffn_exp_probs_b.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());

        let gu_sc = l.ffn_gate_up_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let dn_sc = l.ffn_down_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let moe_act = self.bsc.moe_act.ptr as *mut f32;
        let moe_raw = self.bsc.moe_raw.ptr as *mut f32;

        let fused_rg = l.ffn_gate_up_exps.ty == WeightType::Q4_0 && b <= 64 && self.cfg.n_expert == 128 && self.cfg.n_expert_used == 8;
        if fused_rg {
            ops::ink_moe_router_groups_and_q8(
                s, router_logits, bias_ptr, dn_sc,
                self.bsc.exp_counts.ptr as *mut i32, self.bsc.exp_offsets.ptr as *mut i32,
                self.bsc.tok_sorted.ptr as *mut i32, self.bsc.slot_sorted.ptr as *mut i32,
                self.bsc.wt_sorted.ptr as *mut f32, ffn_normed, self.bsc.xq.ptr,
                self.cfg.route_scale, e, bi,
            );
        } else {
            ops::ink_moe_router_batch(s, router_logits, bias_ptr, self.bsc.wtop.ptr as *mut f32, self.bsc.itop.ptr as *mut i32, self.cfg.n_expert as i32, self.cfg.n_expert_used as i32, self.cfg.route_scale, bi);
        }

        match l.ffn_gate_up_exps.ty {
            WeightType::Nvfp4 => {
                ops::ink_quantize_q8_1(s, ffn_normed, self.bsc.xq.ptr, e, bi);
                ops::ink_moe_build_groups(
                    s, self.bsc.itop.ptr as *const i32, self.bsc.wtop.ptr as *const f32, dn_sc,
                    self.bsc.exp_counts.ptr as *mut i32, self.bsc.exp_offsets.ptr as *mut i32,
                    self.bsc.tok_sorted.ptr as *mut i32, self.bsc.slot_sorted.ptr as *mut i32,
                    self.bsc.wt_sorted.ptr as *mut f32, self.cfg.n_expert_used as i32, bi,
                );
                ops::ink_moe_gu_nvfp4_sudot4(
                    s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, self.bsc.xq.ptr,
                    self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                    self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                    moe_act, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                    self.cfg.situ_beta, self.cfg.situ_lb,
                );
            }
            WeightType::Q4_0 => {
                if !fused_rg {
                    ops::ink_quantize_q8_1(s, ffn_normed, self.bsc.xq.ptr, e, bi);
                    ops::ink_moe_build_groups(
                        s, self.bsc.itop.ptr as *const i32, self.bsc.wtop.ptr as *const f32, dn_sc,
                        self.bsc.exp_counts.ptr as *mut i32, self.bsc.exp_offsets.ptr as *mut i32,
                        self.bsc.tok_sorted.ptr as *mut i32, self.bsc.slot_sorted.ptr as *mut i32,
                        self.bsc.wt_sorted.ptr as *mut f32, self.cfg.n_expert_used as i32, bi,
                    );
                }
                if b >= 48 {
                    ops::ink_moe_gu_q4_0_wmma_to_q8(
                        s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, self.bsc.xq.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                        self.cfg.situ_beta, self.cfg.situ_lb, bi,
                    );
                } else {
                    ops::ink_moe_gu_q4_0_sudot4_to_q8(
                        s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, self.bsc.xq.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                        self.cfg.situ_beta, self.cfg.situ_lb, bi,
                    );
                }
            }
            WeightType::Q8_0 => {
                ops::ink_moe_gu_q8_0_batch(s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, ffn_normed, self.bsc.itop.ptr as *const i32, moe_act, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32, self.cfg.situ_beta, self.cfg.situ_lb, bi);
            }
            _ => unreachable!(),
        };
        match l.ffn_down_exps.ty {
            WeightType::Nvfp4 => {
                ops::ink_quantize_q8_1(s, moe_act, self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, bi * self.cfg.n_expert_used as i32);
                ops::ink_moe_dn_nvfp4_sudot4(
                    s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                    self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                    self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                    self.bsc.wt_sorted.ptr as *const f32, moe_raw,
                    e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi,
                );
            }
            WeightType::Q4_0 => {
                if b >= 48 {
                    ops::ink_moe_dn_q4_0_wmma(
                        s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        self.bsc.wt_sorted.ptr as *const f32, self.bsc.moe_dn_slots.ptr as *mut f32, moe_raw,
                        e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi,
                    );
                } else if b <= 128 {
                    ops::ink_moe_dn_q4_0_sudot4_nozero(
                        s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        self.bsc.wt_sorted.ptr as *const f32, moe_raw,
                        e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi,
                    );
                } else {
                    ops::ink_moe_dn_q4_0_sudot4(
                        s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        self.bsc.wt_sorted.ptr as *const f32, moe_raw,
                        e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi,
                    );
                }
            }
            WeightType::Q8_0 => {
                ops::ink_moe_dn_q8_0_batch(s, l.ffn_down_exps.buf.ptr as *const u8, dn_sc, moe_act, self.bsc.itop.ptr as *const i32, self.bsc.wtop.ptr as *const f32, moe_raw, e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi);
            }
            _ => unreachable!(),
        };
        let fp_snap = if b <= self.multi_verify_cap.min(512) { sl.ffn_post_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
        if b <= 128 && l.ffn_down_exps.ty == WeightType::Q4_0 {
            ops::ink_shortconv_res_batch_snap_clear(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, fp_snap, ffn_inp, std::ptr::null_mut(), cur, e, bi);
        } else {
            ops::ink_shortconv_res_batch_snap(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, fp_snap, ffn_inp, std::ptr::null_mut(), cur, e, bi);
        }
    }

    pub unsafe fn enqueue_forward_batch(&self, s: HipStream, b: usize, all_logits: bool) {
        let e = self.cfg.n_embd as i32;
        let mb = MAX_BATCH as i32;
        let bi = b as i32;
        let be = (b * self.cfg.n_embd) as i32;
        if b <= 128 {
            ops::ink_zero_f32(s, self.bsc.moe_raw.ptr as *mut f32, be);
        }
        let cur = self.bsc.cur.ptr as *mut f32;
        let blocks = self.bsc.blocks.ptr as *mut f32;
        let prelude = self.bsc.prelude.ptr as *mut f32;
        let bank = self.bsc.bank.ptr as *mut f32;
        let bank_out = self.bsc.bank_out.ptr as *mut f32;
        let msgs = self.bsc.msgs.ptr as *mut f32;
        let feats = self.bsc.dspark_feats.ptr as *mut f32;

        ops::ink_embed_q8_0_norm_batch(s, self.tok_embd.buf.ptr as *const u8, self.tok_norm.ptr as *const f32, self.bsc.d_toks.ptr as *const i32, cur, e, bi);

        let mut n_blocks = 0usize;
        let mut visit = 0usize;
        for il in 0..self.cfg.core_start {
            self.enqueue_layer_batch(s, il, il, &mut n_blocks, None, b);
            visit += 1;
            if visit == 5 { ops::ink_copy_f32(s, cur, feats, be); }
        }
        let prelude_n_blocks = n_blocks;
        ops::ink_copy_f32(s, cur, prelude, be);

        for pass in 0..self.cfg.n_passes {
            let mut bank_len = 0usize;
            if pass > 0 {
                let start_b = prelude_n_blocks + 1;
                bank_len = n_blocks - start_b;
                for bk in 0..bank_len {
                    ops::ink_copy_f32(s, blocks.add((start_b + bk) * MAX_BATCH * self.cfg.n_embd), bank.add(bk * MAX_BATCH * self.cfg.n_embd), be);
                }
                ops::ink_copy_f32(s, cur, bank_out, be);
                ops::ink_copy_f32(s, prelude, cur, be);
                n_blocks = prelude_n_blocks;
            }
            for ip in self.cfg.core_start..self.cfg.core_end {
                let il = if pass == 0 {
                    ip
                } else {
                    self.cfg.n_phys + (pass - 1) * (self.cfg.core_end - self.cfg.core_start) + (ip - self.cfg.core_start)
                };
                if pass == 0 {
                    self.enqueue_layer_batch(s, il, ip, &mut n_blocks, None, b);
                } else {
                    let li = ip - self.cfg.core_start;
                    let ll = &self.loop_layers[li];
                    ops::ink_mix_batch(s, bank, bank_len as i32, mb, bank_out, ll.msg_query.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), msgs, e, 4, bi, self.cfg.eps);
                    self.enqueue_layer_batch(s, il, ip, &mut n_blocks, Some(li), b);
                }
                visit += 1;
                if visit == 23 { ops::ink_copy_f32(s, cur, feats.add(MAX_BATCH * self.cfg.n_embd), be); }
                else if visit == 41 { ops::ink_copy_f32(s, cur, feats.add(2 * MAX_BATCH * self.cfg.n_embd), be); }
            }
        }
        for il in self.cfg.core_end..self.cfg.n_phys {
            self.enqueue_layer_batch(s, il, il, &mut n_blocks, None, b);
            visit += 1;
            if visit == 48 { ops::ink_copy_f32(s, cur, feats.add(3 * MAX_BATCH * self.cfg.n_embd), be); }
        }

        let onw = self.output_norm.ptr as *const f32;
        ops::ink_mix_batch(s, blocks, n_blocks as i32, mb, cur, self.output_res_query.ptr as *const f32, onw, onw, onw, self.bsc.mixed.ptr as *mut f32, e, 1, bi, self.cfg.eps);
        if all_logits {
            ops::ink_quantize_q8_1(s, self.bsc.mixed.ptr as *const f32, self.bsc.xq.ptr, e, bi);
            match self.output.ty {
                WeightType::Q4_0 if b >= 12 => ops::ink_gemm_q4_0_wmma(s, self.output.buf.ptr as *const u8, self.bsc.xq.ptr, self.bsc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, bi),
                WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, self.output.buf.ptr as *const u8, self.bsc.xq.ptr, self.bsc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, bi),
                _ => ops::ink_gemm_q8_0_sudot4(s, self.output.buf.ptr as *const u8, self.bsc.xq.ptr, self.bsc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, bi),
            };
            ops::ink_softcap_argmax_batch(s, self.bsc.logits.ptr as *mut f32, self.cfg.final_softcap, self.cfg.vocab_padded as i32, self.cfg.vocab_unpadded as i32, self.bsc.d_out_toks.ptr as *mut i32, bi);
        } else {
            let last_x = (self.bsc.mixed.ptr as *const f32).add((b - 1) * self.cfg.n_embd);
            ops::ink_quantize_q8_1(s, last_x, self.bsc.xq.ptr, e, 1);
            match self.output.ty {
                WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, self.output.buf.ptr as *const u8, self.bsc.xq.ptr, self.bsc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, 1),
                _ => ops::ink_gemm_q8_0_sudot4(s, self.output.buf.ptr as *const u8, self.bsc.xq.ptr, self.bsc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, 1),
            };
            ops::ink_softcap_argmax(s, self.bsc.logits.ptr as *mut f32, self.cfg.final_softcap, self.cfg.vocab_unpadded as i32, self.bsc.d_out_toks.ptr as *mut i32);
        }
    }

    pub fn capture_verify_graph(&mut self, b: usize) -> Result<()> {
        let g = unsafe { CapturedGraph::capture(|s| self.enqueue_forward_batch(s, b, true)) }.map_err(he)?;
        self.seq_verify_graphs.insert((0usize, b), g);
        Ok(())
    }

    pub fn verify_batch(&mut self, tokens: &[i32]) -> Result<Vec<i32>> {
        let b = tokens.len();
        assert!(b >= 1 && b <= self.multi_verify_cap.min(512));
        let pos0 = self.pos;
        let tok_bytes = unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, b * 4) };
        let key = (0usize, b);
        if !self.seq_verify_graphs.contains_key(&key) {
            let g = unsafe { CapturedGraph::capture(|s| self.enqueue_forward_batch(s, b, true)) }.map_err(he)?;
            self.seq_verify_graphs.insert(key, g);
        }
        let g = self.seq_verify_graphs.get(&key).unwrap();
        self.bsc.d_toks.copy_from(tok_bytes, Some(g.stream)).map_err(he)?;
        self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(g.stream)).map_err(he)?;
        g.launch().map_err(he)?;
        let mut out = vec![0i32; b];
        let rc = unsafe {
            rad_hip::hipMemcpyAsync(out.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, b * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST, g.stream)
        };
        if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
        g.sync().map_err(he)?;
        self.pos += b as i32;
        Ok(out)
    }

    pub fn prefill_tokens(&mut self, tokens: &[i32]) -> Result<i32> {
        let mut last = 0;
        let mut i = 0;
        while i < tokens.len() {
            let chunk = &tokens[i..(i + MAX_BATCH).min(tokens.len())];
            let b = chunk.len();
            if b == 1 {
                last = self.step(chunk[0])?;
            } else {
                let pos0 = self.pos;
                let tok_bytes = unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, b * 4) };
                self.bsc.d_toks.copy_from(tok_bytes, Some(self.stream)).map_err(he)?;
                self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
                unsafe {
                    self.enqueue_forward_batch(self.stream, b, false);
                    rad_hip::hipStreamSynchronize(self.stream);
                }
                let mut out_b = [0u8; 4];
                let rc = unsafe {
                    rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
                };
                if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
                last = i32::from_le_bytes(out_b);
                self.pos += b as i32;
            }
            i += b;
        }
        Ok(last)
    }

    pub fn tok_embd_ptr(&self) -> *const u8 {
        self.tok_embd.buf.ptr as *const u8
    }
    pub fn tok_norm_ptr(&self) -> *const f32 {
        self.tok_norm.ptr as *const f32
    }
    pub fn output_ptr(&self) -> *const u8 {
        self.output.buf.ptr as *const u8
    }
    pub fn output_is_q4(&self) -> bool {
        self.output.ty == WeightType::Q4_0
    }
    pub fn single_dspark_feats_ptr(&self) -> *const f32 {
        self.sc.dspark_feats.ptr as *const f32
    }
    pub fn batch_dspark_feats_ptr(&self) -> (*const f32, usize) {
        (self.bsc.dspark_feats.ptr as *const f32, MAX_BATCH)
    }
    pub fn gather_batch_dspark_feats_async(&self, s: HipStream, tok_idx: usize) -> *const f32 {
        unsafe {
            ops::ink_gather_dspark_feats(
                s,
                self.bsc.dspark_feats.ptr as *const f32,
                self.sc.dspark_feats.ptr as *mut f32,
                tok_idx as i32,
                MAX_BATCH as i32,
                self.cfg.n_embd as i32,
            );
        }
        self.sc.dspark_feats.ptr as *const f32
    }
    pub fn gather_batch_dspark_feats(&self, tok_idx: usize) -> *const f32 {
        let p = self.gather_batch_dspark_feats_async(self.stream, tok_idx);
        unsafe { rad_hip::hipStreamSynchronize(self.stream); }
        p
    }
    pub fn verify_batch_seq_from_dev(
        &mut self,
        seq: usize,
        d_toks_src: *const core::ffi::c_void,
        b: usize,
        stream: HipStream,
        draft_out: &mut [i32],
    ) -> Result<Vec<i32>> {
        assert!(b >= 1 && b <= self.multi_verify_cap.min(512));
        let pos0 = self.seq_pos[seq];
        let key = (seq, b);
        if !self.seq_verify_graphs.contains_key(&key) {
            let g = if seq == 0 {
                unsafe { CapturedGraph::capture(|s| self.enqueue_forward_batch(s, b, true)) }.map_err(he)?
            } else {
                unsafe { CapturedGraph::capture(|s| self.enqueue_forward_ex(s, b, true, Some(seq), b)) }.map_err(he)?
            };
            self.seq_verify_graphs.insert(key, g);
        }
        let g = self.seq_verify_graphs.get(&key).unwrap();
        let rc_cp = unsafe {
            rad_hip::hipMemcpyAsync(
                self.bsc.d_toks.ptr,
                d_toks_src,
                b * 4,
                rad_hip::HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream,
            )
        };
        if rc_cp != rad_hip::HIP_SUCCESS {
            return Err(he(rc_cp));
        }
        self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(stream)).map_err(he)?;
        g.launch_on(stream).map_err(he)?;
        let pin_d = self.h_io_pinned.ptr as *mut i32;
        let pin_v = unsafe { pin_d.add(512) };
        let rc_d = unsafe {
            rad_hip::hipMemcpyAsync(
                pin_d as *mut core::ffi::c_void,
                d_toks_src,
                b * 4,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };
        if rc_d != rad_hip::HIP_SUCCESS {
            return Err(he(rc_d));
        }
        let rc = unsafe {
            rad_hip::hipMemcpyAsync(
                pin_v as *mut core::ffi::c_void,
                self.bsc.d_out_toks.ptr,
                b * 4,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        let rc_s = unsafe { rad_hip::hipStreamSynchronize(stream) };
        if rc_s != rad_hip::HIP_SUCCESS {
            return Err(he(rc_s));
        }
        let out = unsafe {
            draft_out[..b].copy_from_slice(std::slice::from_raw_parts(pin_d, b));
            std::slice::from_raw_parts(pin_v, b).to_vec()
        };
        self.seq_pos[seq] += b as i32;
        if seq == 0 {
            self.pos = self.seq_pos[0];
        }
        Ok(out)
    }

    pub fn verify_batch_seq_on(&mut self, seq: usize, tokens: &[i32], stream: HipStream) -> Result<Vec<i32>> {
        let b = tokens.len();
        assert!(b >= 1 && b <= self.multi_verify_cap.min(512));
        let pos0 = self.seq_pos[seq];
        let tok_bytes = unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, b * 4) };
        let key = (seq, b);
        if !self.seq_verify_graphs.contains_key(&key) {
            let g = if seq == 0 {
                unsafe { CapturedGraph::capture(|s| self.enqueue_forward_batch(s, b, true)) }.map_err(he)?
            } else {
                unsafe { CapturedGraph::capture(|s| self.enqueue_forward_ex(s, b, true, Some(seq), b)) }.map_err(he)?
            };
            self.seq_verify_graphs.insert(key, g);
        }
        let g = self.seq_verify_graphs.get(&key).unwrap();
        self.bsc.d_toks.copy_from(tok_bytes, Some(stream)).map_err(he)?;
        self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(stream)).map_err(he)?;
        g.launch_on(stream).map_err(he)?;
        let pin_v = self.h_io_pinned.ptr as *mut i32;
        let rc = unsafe {
            rad_hip::hipMemcpyAsync(
                pin_v as *mut core::ffi::c_void,
                self.bsc.d_out_toks.ptr,
                b * 4,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        let rc_s = unsafe { rad_hip::hipStreamSynchronize(stream) };
        if rc_s != rad_hip::HIP_SUCCESS {
            return Err(he(rc_s));
        }
        let out = unsafe { std::slice::from_raw_parts(pin_v, b).to_vec() };
        self.seq_pos[seq] += b as i32;
        if seq == 0 {
            self.pos = self.seq_pos[0];
        }
        Ok(out)
    }

    pub fn verify_batch_seq(&mut self, seq: usize, tokens: &[i32]) -> Result<Vec<i32>> {
        if seq == 0 {
            self.pos = self.seq_pos[0];
            let out = self.verify_batch(tokens)?;
            self.seq_pos[0] = self.pos;
            return Ok(out);
        }
        let b = tokens.len();
        assert!(b >= 1 && b <= self.multi_verify_cap.min(512));
        let pos0 = self.seq_pos[seq];
        let tok_bytes = unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, b * 4) };
        let key = (seq, b);
        if !self.seq_verify_graphs.contains_key(&key) {
            let g = unsafe { CapturedGraph::capture(|s| self.enqueue_forward_ex(s, b, true, Some(seq), b)) }.map_err(he)?;
            self.seq_verify_graphs.insert(key, g);
        }
        let g = self.seq_verify_graphs.get(&key).unwrap();
        self.bsc.d_toks.copy_from(tok_bytes, Some(g.stream)).map_err(he)?;
        self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(g.stream)).map_err(he)?;
        g.launch().map_err(he)?;
        let mut out = vec![0i32; b];
        let rc = unsafe {
            rad_hip::hipMemcpyAsync(out.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, b * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST, g.stream)
        };
        if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
        g.sync().map_err(he)?;
        self.seq_pos[seq] += b as i32;
        Ok(out)
    }

    pub fn rewind_verify_seq_to_on(&mut self, seq: usize, n_acc: usize, b: usize, stream: HipStream) -> Result<()> {
        if n_acc < b {
            let idx = (n_acc - 1) as i32;
            let kv_n = (self.cfg.n_head_kv * self.cfg.dh * 3) as i32;
            let e_n = (self.cfg.n_embd * 3) as i32;
            unsafe {
                ops::ink_rewind_single_all_slots(
                    stream,
                    self.d_rewind_ptrs.ptr as *const u64,
                    seq as i32,
                    idx,
                    self.slots.len() as i32,
                    kv_n,
                    e_n,
                );
            }
            self.seq_pos[seq] -= (b - n_acc) as i32;
            if seq == 0 {
                self.pos = self.seq_pos[0];
            }
        }
        Ok(())
    }

    pub fn rewind_verify_seq_to(&mut self, seq: usize, n_acc: usize, b: usize) -> Result<()> {
        if n_acc < b {
            let idx = (n_acc - 1) as i32;
            let kv_n = (self.cfg.n_head_kv * self.cfg.dh * 3) as i32;
            let e_n = (self.cfg.n_embd * 3) as i32;
            let target_s = self.seq_verify_graphs.get(&(seq, b)).map(|vg| vg.stream).unwrap_or(self.stream);
            unsafe {
                ops::ink_rewind_single_all_slots(
                    target_s,
                    self.d_rewind_ptrs.ptr as *const u64,
                    seq as i32,
                    idx,
                    self.slots.len() as i32,
                    kv_n,
                    e_n,
                );
            }
            self.seq_pos[seq] -= (b - n_acc) as i32;
            if seq == 0 {
                self.pos = self.seq_pos[0];
            }
        }
        Ok(())
    }

    pub fn rewind_verify_to(&mut self, n_acc: usize, b: usize) -> Result<()> {
        self.seq_pos[0] = self.pos;
        self.rewind_verify_seq_to(0, n_acc, b)
    }

    pub fn reset_seq(&mut self, seq: usize) -> Result<()> {
        if let Some((_, ref vg)) = self.verify_graph {
            let _ = vg.sync();
        }
        for ((s_id, _), vg) in &self.seq_verify_graphs {
            if *s_id == seq {
                let _ = vg.sync();
            }
        }
        self.paged_kv.free_seq(seq);
        let kv_dim = self.cfg.n_head_kv * self.cfg.dh;
        let e = self.cfg.n_embd;
        let kv_bytes = kv_dim * 3 * 4;
        let e_bytes = e * 3 * 4;
        for sl in &mut self.slots {
            unsafe {
                rad_hip::hipMemsetD32Async((sl.k_hist.ptr as *mut u8).add(seq * kv_bytes) as *mut core::ffi::c_void, 0, kv_bytes / 4, self.stream);
                rad_hip::hipMemsetD32Async((sl.v_hist.ptr as *mut u8).add(seq * kv_bytes) as *mut core::ffi::c_void, 0, kv_bytes / 4, self.stream);
                rad_hip::hipMemsetD32Async((sl.attn_post_hist.ptr as *mut u8).add(seq * e_bytes) as *mut core::ffi::c_void, 0, e_bytes / 4, self.stream);
                rad_hip::hipMemsetD32Async((sl.ffn_post_hist.ptr as *mut u8).add(seq * e_bytes) as *mut core::ffi::c_void, 0, e_bytes / 4, self.stream);
            }
        }
        unsafe { rad_hip::hipStreamSynchronize(self.stream) };
        self.seq_pos[seq] = 0;
        self.seq_tokens[seq].clear();
        if seq == 0 {
            self.pos = 0;
        }
        Ok(())
    }

}
