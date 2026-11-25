use super::*;

impl InklingEngine {
    pub fn layer_moe_summary(&self) {
        let mut q8_layers = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            if l.ffn_gate_up_exps.ty == WeightType::Q8_0 || l.ffn_down_exps.ty == WeightType::Q8_0 {
                q8_layers.push(i);
            }
        }
        println!("Q8_0 MoE physical layers: {:?}", q8_layers);
    }

    pub fn profile_single_step(&self) -> Result<()> {
        let s = self.stream;
        let e = self.cfg.n_embd as i32;
        let kv_dim = (self.cfg.n_head_kv * self.cfg.dh) as i32;
        let q_dim = (self.cfg.n_head * self.cfg.dh) as i32;
        let r_dim = (self.cfg.n_head * self.cfg.d_rel) as i32;
        let _l = &self.layers[6];
        let _sl = &self.slots[6];
        let _ll = &self.loop_layers[0];
        let iters = 200;

        let _bench = |name: &str, mult: f64, f: &dyn Fn()| {
            for _ in 0..5 { f(); }
            unsafe { rad_hip::hipStreamSynchronize(s); }
            let t0 = std::time::Instant::now();
            for _ in 0..iters { f(); }
            unsafe { rad_hip::hipStreamSynchronize(s); }
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("  B=1 {:<22}: {:7.1} us/call  x{:2.0} = {:6.2} ms", name, us, mult, us * mult / 1000.0);
        };

        println!("=== B=1 COLD-HBM 48-VISIT TOTALS ===");
        let mut t_mix = 0.0f64;
        let mut t_qkvrg = 0.0f64;
        let mut t_kvconv = 0.0f64;
        let mut t_attn = 0.0f64;
        let mut t_wo = 0.0f64;
        let mut t_fmix = 0.0f64;
        let mut t_router = 0.0f64;
        let mut t_mgu = 0.0f64;
        let mut t_mdn = 0.0f64;
        let mut t_lm = 0.0f64;
        let sync_t = |st: HipStream| -> std::time::Instant {
            unsafe { rad_hip::hipStreamSynchronize(st); }
            std::time::Instant::now()
        };
        let xq0 = self.bsc.xq.ptr;
        let xq_stride = MAX_BATCH * ((q_dim as usize) / 32) * 40;
        let xq1 = unsafe { (self.bsc.xq.ptr as *mut u8).add(xq_stride) as *mut core::ffi::c_void };
        let xq2 = unsafe { (self.bsc.xq.ptr as *mut u8).add(2 * xq_stride) as *mut core::ffi::c_void };
        let runs = 5;
        for _ in 0..runs {
            let mut n_blocks = 0usize;
            for ip in 0..self.cfg.n_phys {
                let l = &self.layers[ip];
                let sl = &self.slots[ip];
                let t0 = sync_t(s);
                unsafe {
                    if let Some(ref nw) = l.attn_norm {
                        ops::ink_rmsnorm(s, self.sc.cur.ptr as *const f32, nw.ptr as *const f32, self.sc.inp0.ptr as *mut f32, e, self.cfg.eps);
                    }
                }
                let t1 = sync_t(s);
                t_mix += (t1 - t0).as_secs_f64() * 1000.0;
                unsafe {
                    ops::ink_quantize_q8_1(s, self.sc.inp0.ptr as *const f32, xq0, e, 1);
                    let (wkd, wvd, dd) = if l.is_swa { (std::ptr::null(), std::ptr::null(), 0) } else {
                        (l.attn_k_conv_proj.as_ref().unwrap().buf.ptr as *const u16, l.attn_v_conv_proj.as_ref().unwrap().buf.ptr as *const u16, (4 * self.cfg.n_head_kv) as i32)
                    };
                    ops::ink_qkvrg_gemm_sudot4(
                        s, l.wq.buf.ptr as *const u8, l.wk.buf.ptr as *const u8, l.wv.buf.ptr as *const u8,
                        l.wrel.buf.ptr as *const u8, l.wqkv_gate.buf.ptr as *const u8, l.attn_sink_proj.buf.ptr as *const u16,
                        wkd, wvd, xq0, xq1, xq2,
                        self.sc.inp0.ptr as *const f32, self.sc.inp0.ptr as *const f32, self.sc.inp0.ptr as *const f32,
                        self.sc.q.ptr as *mut f32, self.sc.k_raw.ptr as *mut f32, self.sc.v_raw.ptr as *mut f32,
                        self.sc.r.ptr as *mut f32, self.sc.gate_out.ptr as *mut f32, self.sc.sinks.ptr as *mut f32,
                        self.sc.k_dyn.ptr as *mut f32, self.sc.v_dyn.ptr as *mut f32,
                        e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, dd, 1, if l.wq.ty == WeightType::Q4_0 { 1 } else { 0 },
                    );
                }
                let t2 = sync_t(s);
                t_qkvrg += (t2 - t1).as_secs_f64() * 1000.0;
                unsafe {
                    ops::ink_kv_conv_qkvr_store(
                        s, self.sc.q.ptr as *mut f32, self.sc.k_raw.ptr as *const f32, self.sc.v_raw.ptr as *const f32, self.sc.r.ptr as *mut f32,
                        l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                        if l.is_swa { std::ptr::null() } else { self.sc.k_dyn.ptr as *const f32 },
                        if l.is_swa { std::ptr::null() } else { self.sc.v_dyn.ptr as *const f32 },
                        sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32, l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                        sl.k_cache.ptr as *mut u16, sl.v_cache.ptr as *mut u16, self.sc.d_pos.ptr as *const i32,
                        if sl.is_swa { 1 } else { 0 }, self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                    );
                }
                let t3 = sync_t(s);
                t_kvconv += (t3 - t2).as_secs_f64() * 1000.0;
                unsafe {
                    ops::ink_attn_decode_gated(
                        s, self.sc.q.ptr as *const f32, self.sc.r.ptr as *const f32, self.sc.sinks.ptr as *const f32,
                        l.attn_rel_bias.ptr as *const f32, sl.k_cache.ptr as *const u16, sl.v_cache.ptr as *const u16,
                        self.sc.gate_out.ptr as *const f32, self.sc.d_pos.ptr as *const i32, self.sc.attn_out.ptr as *mut f32,
                        self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 }, sl.max_ctx as i32,
                    );
                }
                let t4 = sync_t(s);
                t_attn += (t4 - t3).as_secs_f64() * 1000.0;
                unsafe {
                    ops::ink_quantize_q8_1(s, self.sc.attn_out.ptr as *const f32, xq0, q_dim, 1);
                    ops::ink_wo_conv_res_q4_0_sudot4(
                        s, l.wo.buf.ptr as *const u8, xq0, l.attn_post_conv.ptr as *const f32,
                        sl.attn_post_hist.ptr as *mut f32, self.sc.cur.ptr as *const f32, std::ptr::null_mut(),
                        self.sc.ffn_inp.ptr as *mut f32, e, q_dim,
                    );
                }
                let t5 = sync_t(s);
                t_wo += (t5 - t4).as_secs_f64() * 1000.0;
                unsafe {
                    let fnw = l.ffn_norm.ptr as *const f32;
                    ops::ink_mix_rmsnorm(s, self.sc.blocks.ptr as *const f32, n_blocks as i32, self.sc.ffn_inp.ptr as *const f32, l.ffn_res_query.ptr as *const f32, fnw, fnw, fnw, self.sc.ffn_normed.ptr as *mut f32, e, 1, self.cfg.eps);
                }
                let t6 = sync_t(s);
                t_fmix += (t6 - t5).as_secs_f64() * 1000.0;
                unsafe {
                    l.ffn_gate_inp.gemv(s, self.sc.ffn_normed.ptr as *const f32, self.sc.router_logits.ptr as *mut f32);
                    let bp = l.ffn_exp_probs_b.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
                    ops::ink_moe_router(s, self.sc.router_logits.ptr as *const f32, bp, self.sc.wtop.ptr as *mut f32, self.sc.itop.ptr as *mut i32, self.cfg.n_expert as i32, self.cfg.n_expert_used as i32, self.cfg.route_scale);
                }
                let t7 = sync_t(s);
                t_router += (t7 - t6).as_secs_f64() * 1000.0;
                let gu_sc = l.ffn_gate_up_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
                let dn_sc = l.ffn_down_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
                unsafe {
                    ops::ink_quantize_q8_1(s, self.sc.ffn_normed.ptr as *const f32, xq0, e, 1);
                    ops::ink_moe_gu_q4_0_b1_sudot4(
                        s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, xq0,
                        self.sc.itop.ptr as *const i32, self.sc.moe_act.ptr as *mut f32,
                        self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32, self.cfg.situ_beta, self.cfg.situ_lb,
                    );
                }
                let t8 = sync_t(s);
                t_mgu += (t8 - t7).as_secs_f64() * 1000.0;
                unsafe {
                    ops::ink_quantize_q8_1(s, self.sc.moe_act.ptr as *const f32, self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32);
                    ops::ink_moe_dn_conv_add_q4_0_b1_sudot4(
                        s, l.ffn_down_exps.buf.ptr as *const u8, dn_sc, self.bsc.act_q.ptr,
                        self.sc.itop.ptr as *const i32, self.sc.wtop.ptr as *const f32,
                        l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32,
                        self.sc.ffn_inp.ptr as *const f32, self.sc.cur.ptr as *mut f32,
                        e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32,
                    );
                }
                let t9 = sync_t(s);
                t_mdn += (t9 - t8).as_secs_f64() * 1000.0;
                if ip % self.cfg.attn_res_block == 0 { n_blocks += 1; }
            }
            let t0 = sync_t(s);
            unsafe {
                ops::ink_quantize_q8_1(s, self.sc.cur.ptr as *const f32, xq0, e, 1);
                ops::ink_gemm_q4_0_sudot4(s, self.output.buf.ptr as *const u8, xq0, self.sc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, 1);
            }
            let t1 = sync_t(s);
            t_lm += (t1 - t0).as_secs_f64() * 1000.0;
        }
        let sc2 = 2.0 / runs as f64;
        println!("  mix={:.2}ms qkvrg={:.2}ms kvconv={:.2}ms attn={:.2}ms wo={:.2}ms fmix={:.2}ms router={:.2}ms mgu={:.2}ms mdn={:.2}ms lm={:.2}ms | sum48={:.2}ms",
            t_mix * sc2, t_qkvrg * sc2, t_kvconv * sc2, t_attn * sc2, t_wo * sc2, t_fmix * sc2, t_router * sc2, t_mgu * sc2, t_mdn * sc2, t_lm / runs as f64,
            (t_mix + t_qkvrg + t_kvconv + t_attn + t_wo + t_fmix + t_router + t_mgu + t_mdn) * sc2 + t_lm / runs as f64);
        Ok(())
    }

    pub fn profile_batch_step(&self, b: usize) -> Result<()> {
        let s = self.stream;
        let e = self.cfg.n_embd as i32;
        let kv_dim = (self.cfg.n_head_kv * self.cfg.dh) as i32;
        let q_dim = (self.cfg.n_head * self.cfg.dh) as i32;
        let r_dim = (self.cfg.n_head * self.cfg.d_rel) as i32;
        let bi = b as i32;
        let l = &self.layers[6];
        let sl = &self.slots[6];
        let iters = 50;

        let bench = |name: &str, mult: f64, f: &dyn Fn()| {
            for _ in 0..3 { f(); }
            unsafe { rad_hip::hipStreamSynchronize(s); }
            let t0 = std::time::Instant::now();
            for _ in 0..iters { f(); }
            unsafe { rad_hip::hipStreamSynchronize(s); }
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("  B={:<2} {:<22}: {:7.1} us/call  x{:2.0} = {:6.2} ms", b, name, us, mult, us * mult / 1000.0);
        };

        println!("=== B={b} PER-STAGE BREAKDOWN (48 visits) ===");
        let xq0 = self.bsc.xq.ptr;
        bench("mix_batch", 96.0, &|| unsafe {
            let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
            ops::ink_mix_batch(s, self.bsc.blocks.ptr as *const f32, 4, MAX_BATCH as i32, self.bsc.cur.ptr as *const f32, l.ffn_res_query.ptr as *const f32, nw, nw, nw, self.bsc.inp0.ptr as *mut f32, e, 1, bi, self.cfg.eps);
        });
        let ll = &self.loop_layers[0];
        bench("loop_ctrl_3_batch", 24.0, &|| unsafe {
            let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
            ops::ink_loop_ctrl_3_norm_batch(
                s, self.bsc.mixed.ptr as *const f32, self.bsc.mixed.ptr as *const f32, self.bsc.mixed.ptr as *const f32,
                self.bsc.msgs.ptr as *const f32,
                ll.ctrl_down[0].ptr as *const f32, ll.ctrl_down[1].ptr as *const f32, ll.ctrl_down[2].ptr as *const f32,
                ll.ctrl_gain[0].ptr as *const f32, ll.ctrl_gain[1].ptr as *const f32, ll.ctrl_gain[2].ptr as *const f32,
                ll.ctrl_forget[0].ptr as *const f32, ll.ctrl_forget[1].ptr as *const f32, ll.ctrl_forget[2].ptr as *const f32,
                nw, nw, nw, self.bsc.ctrl_h.ptr as *mut f32,
                self.bsc.inp0.ptr as *mut f32, self.bsc.inp1.ptr as *mut f32, self.bsc.inp2.ptr as *mut f32,
                e, self.cfg.gate_rank as i32, MAX_BATCH as i32, bi, self.cfg.eps,
            );
        });
        bench("qkvrg_wmma", 48.0, &|| unsafe {
            ops::ink_quantize_q8_1(s, self.bsc.inp0.ptr as *const f32, xq0, e, bi);
            ops::ink_qkvrg_gemm_q4_wmma(
                s, l.wq.buf.ptr as *const u8, l.wk.buf.ptr as *const u8, l.wv.buf.ptr as *const u8,
                l.wrel.buf.ptr as *const u8, l.wqkv_gate.buf.ptr as *const u8, l.attn_sink_proj.buf.ptr as *const u16,
                std::ptr::null(), std::ptr::null(), xq0, xq0, xq0,
                self.bsc.inp0.ptr as *const f32, self.bsc.inp0.ptr as *const f32, self.bsc.inp0.ptr as *const f32,
                self.bsc.q.ptr as *mut f32, self.bsc.k_raw.ptr as *mut f32, self.bsc.v_raw.ptr as *mut f32,
                self.bsc.r.ptr as *mut f32, self.bsc.gate_out.ptr as *mut f32, self.bsc.sinks.ptr as *mut f32,
                self.bsc.k_dyn.ptr as *mut f32, self.bsc.v_dyn.ptr as *mut f32,
                e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, 0, bi,
            );
        });
        bench("kv_conv_store_batch", 48.0, &|| unsafe {
            ops::ink_kv_conv_qkvr_store_batch(
                s, self.bsc.q.ptr as *mut f32, self.bsc.k_raw.ptr as *const f32, self.bsc.v_raw.ptr as *const f32, self.bsc.r.ptr as *mut f32,
                l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32, std::ptr::null(), std::ptr::null(),
                sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32, l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                sl.k_cache.ptr as *mut u16, sl.v_cache.ptr as *mut u16, self.bsc.d_pos0.ptr as *const i32,
                bi, 1, self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
            );
        });
        bench("attn_batch_flash", 48.0, &|| unsafe {
            ops::ink_attn_batch_gated_flash(
                s, self.bsc.q.ptr as *const f32, self.bsc.r.ptr as *const f32, self.bsc.sinks.ptr as *const f32,
                l.attn_rel_bias.ptr as *const f32, sl.k_cache.ptr as *const u16, sl.v_cache.ptr as *const u16,
                self.bsc.gate_out.ptr as *const f32, self.bsc.d_pos0.ptr as *const i32, self.bsc.attn_out.ptr as *mut f32,
                self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                self.cfg.rel_extent as i32, self.cfg.swa_window as i32, 1, sl.max_ctx as i32, bi,
            );
        });
        bench("wo_q4_wmma", 48.0, &|| unsafe {
            ops::ink_quantize_q8_1(s, self.bsc.attn_out.ptr as *const f32, xq0, q_dim, bi);
            ops::ink_gemm_q4_0_wmma(s, l.wo.buf.ptr as *const u8, xq0, self.bsc.o_raw.ptr as *mut f32, e, q_dim, bi);
        });
        bench("shortconv_res_batch", 96.0, &|| unsafe {
            ops::ink_shortconv_res_batch(s, self.bsc.o_raw.ptr as *const f32, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, self.bsc.cur.ptr as *const f32, std::ptr::null_mut(), self.bsc.ffn_inp.ptr as *mut f32, e, bi);
        });
        bench("router_gemm_f32", 48.0, &|| unsafe {
            ops::ink_gemm_f32(s, l.ffn_gate_inp.buf.ptr as *const f32, self.bsc.ffn_normed.ptr as *const f32, self.bsc.router_logits.ptr as *mut f32, self.cfg.n_expert as i32, e, bi);
            let bp = l.ffn_exp_probs_b.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
            ops::ink_moe_router_batch(s, self.bsc.router_logits.ptr as *const f32, bp, self.bsc.wtop.ptr as *mut f32, self.bsc.itop.ptr as *mut i32, self.cfg.n_expert as i32, self.cfg.n_expert_used as i32, self.cfg.route_scale, bi);
        });
        let gu_sc = l.ffn_gate_up_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let dn_sc = l.ffn_down_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        unsafe {
            ops::ink_moe_build_groups(
                s, self.bsc.itop.ptr as *const i32, self.bsc.wtop.ptr as *const f32, dn_sc,
                self.bsc.exp_counts.ptr as *mut i32, self.bsc.exp_offsets.ptr as *mut i32,
                self.bsc.tok_sorted.ptr as *mut i32, self.bsc.slot_sorted.ptr as *mut i32,
                self.bsc.wt_sorted.ptr as *mut f32, self.cfg.n_expert_used as i32, bi,
            );
        }
        bench("moe_gu_q4_wmma", 48.0, &|| unsafe {
            ops::ink_quantize_q8_1(s, self.bsc.ffn_normed.ptr as *const f32, xq0, e, bi);
            ops::ink_moe_gu_q4_0_wmma(
                s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, xq0,
                self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                self.bsc.moe_act.ptr as *mut f32, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                self.cfg.situ_beta, self.cfg.situ_lb, bi,
            );
        });
        bench("moe_dn_q4_wmma", 48.0, &|| unsafe {
            ops::ink_quantize_q8_1(s, self.bsc.moe_act.ptr as *const f32, self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, bi * self.cfg.n_expert_used as i32);
            ops::ink_moe_dn_q4_0_wmma(
                s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                self.bsc.wt_sorted.ptr as *const f32, self.bsc.moe_dn_slots.ptr as *mut f32, self.bsc.moe_raw.ptr as *mut f32,
                e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi,
            );
        });
        Ok(())
    }

    pub fn profile_batch_cold(&mut self, b: usize) -> Result<()> {
        let s = self.stream;
        let e = self.cfg.n_embd as i32;
        let kv_dim = (self.cfg.n_head_kv * self.cfg.dh) as i32;
        let q_dim = (self.cfg.n_head * self.cfg.dh) as i32;
        let r_dim = (self.cfg.n_head * self.cfg.d_rel) as i32;
        let bi = b as i32;
        let mb = MAX_BATCH as i32;
        let dummy: Vec<i32> = (0..b).map(|i| (100 + i) as i32).collect();
        let tok_bytes = unsafe { std::slice::from_raw_parts(dummy.as_ptr() as *const u8, b * 4) };
        let pos0: i32 = std::env::var("RAD_PROF_POS").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        self.bsc.d_toks.copy_from(tok_bytes, Some(s)).map_err(he)?;
        self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(s)).map_err(he)?;
        unsafe {
            self.enqueue_forward_batch(s, b, b <= 8);
            rad_hip::hipStreamSynchronize(s);
        }
        let sync_t = |st: HipStream| -> std::time::Instant {
            unsafe { rad_hip::hipStreamSynchronize(st); }
            std::time::Instant::now()
        };
        let mut t_amix = 0.0f64;
        let mut t_qkvrg = 0.0f64;
        let mut t_kvconv = 0.0f64;
        let mut t_attn = 0.0f64;
        let mut t_wo = 0.0f64;
        let mut t_aconv = 0.0f64;
        let mut t_fmix = 0.0f64;
        let mut t_router = 0.0f64;
        let mut t_mgu = 0.0f64;
        let mut t_mdn = 0.0f64;
        let mut t_fconv = 0.0f64;
        let mut t_lm = 0.0f64;
        let xq0 = self.bsc.xq.ptr;
        let xq_stride = MAX_BATCH * ((q_dim as usize) / 32) * 40;
        let xq1 = unsafe { (self.bsc.xq.ptr as *mut u8).add(xq_stride) as *mut core::ffi::c_void };
        let xq2 = unsafe { (self.bsc.xq.ptr as *mut u8).add(2 * xq_stride) as *mut core::ffi::c_void };
        let runs = 3;
        for _ in 0..runs {
            let mut n_blocks = 0usize;
            for pass in 0..2 {
                for ip in 0..self.cfg.n_phys {
                    let l = &self.layers[ip];
                    let sl = &self.slots[ip];
                    let is_loop = pass == 1 && ip >= self.cfg.core_start && ip < self.cfg.core_end;
                    let t0 = sync_t(s);
                    unsafe {
                        if let Some(ref rq) = l.attn_res_query {
                            let nq = if l.is_swa { 1 } else { 3 };
                            if is_loop {
                                let ll = &self.loop_layers[ip - self.cfg.core_start];
                                ops::ink_mix_batch(s, self.bsc.blocks.ptr as *const f32, n_blocks.max(1) as i32, mb, self.bsc.cur.ptr as *const f32, rq.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), self.bsc.mixed.ptr as *mut f32, e, nq, bi, self.cfg.eps);
                                let nw = l.attn_norm.as_ref().or(l.attn_q_inp_norm.as_ref()).unwrap().ptr as *const f32;
                                ops::ink_loop_ctrl_3_norm_batch(
                                    s, self.bsc.mixed.ptr as *const f32, self.bsc.mixed.ptr as *const f32, self.bsc.mixed.ptr as *const f32, self.bsc.msgs.ptr as *const f32,
                                    ll.ctrl_down[0].ptr as *const f32, ll.ctrl_down[1].ptr as *const f32, ll.ctrl_down[2].ptr as *const f32,
                                    ll.ctrl_gain[0].ptr as *const f32, ll.ctrl_gain[1].ptr as *const f32, ll.ctrl_gain[2].ptr as *const f32,
                                    ll.ctrl_forget[0].ptr as *const f32, ll.ctrl_forget[1].ptr as *const f32, ll.ctrl_forget[2].ptr as *const f32,
                                    nw, nw, nw, self.bsc.ctrl_h.ptr as *mut f32, self.bsc.inp0.ptr as *mut f32, self.bsc.inp1.ptr as *mut f32, self.bsc.inp2.ptr as *mut f32,
                                    e, self.cfg.gate_rank as i32, mb, bi, self.cfg.eps,
                                );
                            } else {
                                let nw = l.attn_norm.as_ref().or(l.attn_q_inp_norm.as_ref()).unwrap().ptr as *const f32;
                                ops::ink_mix_batch(s, self.bsc.blocks.ptr as *const f32, n_blocks.max(1) as i32, mb, self.bsc.cur.ptr as *const f32, rq.ptr as *const f32, nw, nw, nw, self.bsc.mixed.ptr as *mut f32, e, nq, bi, self.cfg.eps);
                            }
                        } else {
                            let nw = l.attn_norm.as_ref().unwrap().ptr as *const f32;
                            ops::ink_rmsnorm_batch(s, self.bsc.cur.ptr as *const f32, nw, self.bsc.inp0.ptr as *mut f32, e, bi, self.cfg.eps);
                        }
                    }
                    let t1 = sync_t(s);
                    t_amix += (t1 - t0).as_secs_f64() * 1000.0;
                    unsafe {
                        ops::ink_quantize_q8_1(s, self.bsc.inp0.ptr as *const f32, xq0, e, bi);
                        let (wkd, wvd, dd) = if l.is_swa { (std::ptr::null(), std::ptr::null(), 0) } else {
                            (l.attn_k_conv_proj.as_ref().unwrap().buf.ptr as *const u16, l.attn_v_conv_proj.as_ref().unwrap().buf.ptr as *const u16, (4 * self.cfg.n_head_kv) as i32)
                        };
                        if l.wq.ty == WeightType::Q4_0 && b >= 12 {
                            ops::ink_qkvrg_gemm_q4_wmma(
                                s, l.wq.buf.ptr as *const u8, l.wk.buf.ptr as *const u8, l.wv.buf.ptr as *const u8,
                                l.wrel.buf.ptr as *const u8, l.wqkv_gate.buf.ptr as *const u8, l.attn_sink_proj.buf.ptr as *const u16,
                                wkd, wvd, xq0, xq1, xq2,
                                self.bsc.inp0.ptr as *const f32, self.bsc.inp0.ptr as *const f32, self.bsc.inp0.ptr as *const f32,
                                self.bsc.q.ptr as *mut f32, self.bsc.k_raw.ptr as *mut f32, self.bsc.v_raw.ptr as *mut f32,
                                self.bsc.r.ptr as *mut f32, self.bsc.gate_out.ptr as *mut f32, self.bsc.sinks.ptr as *mut f32,
                                self.bsc.k_dyn.ptr as *mut f32, self.bsc.v_dyn.ptr as *mut f32,
                                e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, dd, bi,
                            );
                        } else {
                            ops::ink_qkvrg_gemm_sudot4(
                                s, l.wq.buf.ptr as *const u8, l.wk.buf.ptr as *const u8, l.wv.buf.ptr as *const u8,
                                l.wrel.buf.ptr as *const u8, l.wqkv_gate.buf.ptr as *const u8, l.attn_sink_proj.buf.ptr as *const u16,
                                wkd, wvd, xq0, xq1, xq2,
                                self.bsc.inp0.ptr as *const f32, self.bsc.inp0.ptr as *const f32, self.bsc.inp0.ptr as *const f32,
                                self.bsc.q.ptr as *mut f32, self.bsc.k_raw.ptr as *mut f32, self.bsc.v_raw.ptr as *mut f32,
                                self.bsc.r.ptr as *mut f32, self.bsc.gate_out.ptr as *mut f32, self.bsc.sinks.ptr as *mut f32,
                                self.bsc.k_dyn.ptr as *mut f32, self.bsc.v_dyn.ptr as *mut f32,
                                e, q_dim, kv_dim, r_dim, self.cfg.n_head as i32, dd, bi, if l.wq.ty == WeightType::Q4_0 { 1 } else { 0 },
                            );
                        }
                    }
                    let t2 = sync_t(s);
                    t_qkvrg += (t2 - t1).as_secs_f64() * 1000.0;
                    unsafe {
                        let k_snap = if b <= 16 { sl.k_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
                        let v_snap = if b <= 16 { sl.v_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
                        if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
                            ops::ink_kv_conv_qkvr_store_batch_snap_q8kv(
                                s, self.bsc.q.ptr as *mut f32, self.bsc.k_raw.ptr as *const f32, self.bsc.v_raw.ptr as *const f32, self.bsc.r.ptr as *mut f32,
                                l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                                if l.is_swa { std::ptr::null() } else { self.bsc.k_dyn.ptr as *const f32 },
                                if l.is_swa { std::ptr::null() } else { self.bsc.v_dyn.ptr as *const f32 },
                                sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32, k_snap, v_snap,
                                l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                                sl.k_cache.ptr as *mut i8, sl.v_cache.ptr as *mut i8, ks.ptr as *mut u16, vs.ptr as *mut u16, self.bsc.d_pos0.ptr as *const i32,
                                bi, self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                                sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                            );
                        } else {
                            ops::ink_kv_conv_qkvr_store_batch_snap(
                                s, self.bsc.q.ptr as *mut f32, self.bsc.k_raw.ptr as *const f32, self.bsc.v_raw.ptr as *const f32, self.bsc.r.ptr as *mut f32,
                                l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                                if l.is_swa { std::ptr::null() } else { self.bsc.k_dyn.ptr as *const f32 },
                                if l.is_swa { std::ptr::null() } else { self.bsc.v_dyn.ptr as *const f32 },
                                sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32, k_snap, v_snap,
                                l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                                sl.k_cache.ptr as *mut u16, sl.v_cache.ptr as *mut u16, self.bsc.d_pos0.ptr as *const i32,
                                bi, if sl.is_swa { 1 } else { 0 }, self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                                sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                            );
                        }
                    }
                    let t3 = sync_t(s);
                    t_kvconv += (t3 - t2).as_secs_f64() * 1000.0;
                    unsafe {
                        if b > 64 {
                            if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
                                ops::ink_attn_batch_gated_flash_q8kv(
                                    s, self.bsc.q.ptr as *const f32, self.bsc.r.ptr as *const f32, self.bsc.sinks.ptr as *const f32,
                                    l.attn_rel_bias.ptr as *const f32, sl.k_cache.ptr as *const i8, sl.v_cache.ptr as *const i8,
                                    ks.ptr as *const u16, vs.ptr as *const u16,
                                    self.bsc.gate_out.ptr as *const f32, self.bsc.d_pos0.ptr as *const i32, self.bsc.attn_out.ptr as *mut f32,
                                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                                    self.cfg.rel_extent as i32, sl.max_ctx as i32, bi,
                                );
                            } else {
                                ops::ink_attn_batch_gated_flash(
                                    s, self.bsc.q.ptr as *const f32, self.bsc.r.ptr as *const f32, self.bsc.sinks.ptr as *const f32,
                                    l.attn_rel_bias.ptr as *const f32, sl.k_cache.ptr as *const u16, sl.v_cache.ptr as *const u16,
                                    self.bsc.gate_out.ptr as *const f32, self.bsc.d_pos0.ptr as *const i32, self.bsc.attn_out.ptr as *mut f32,
                                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                                    self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 }, sl.max_ctx as i32, bi,
                                );
                            }
                            ops::ink_quantize_q8_1(s, self.bsc.attn_out.ptr as *const f32, xq0, q_dim, bi);
                        } else if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
                            ops::ink_attn_b8_flash_q8_q8kv(
                                s, self.bsc.q.ptr as *const f32, self.bsc.r.ptr as *const f32, self.bsc.sinks.ptr as *const f32,
                                l.attn_rel_bias.ptr as *const f32, sl.k_cache.ptr as *const i8, sl.v_cache.ptr as *const i8,
                                ks.ptr as *const u16, vs.ptr as *const u16,
                                self.bsc.gate_out.ptr as *const f32, self.bsc.d_pos0.ptr as *const i32, xq0,
                                self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                                self.cfg.rel_extent as i32, sl.max_ctx as i32, bi,
                            );
                        } else {
                            ops::ink_attn_batch_gated_q8(
                                s, self.bsc.q.ptr as *const f32, self.bsc.r.ptr as *const f32, self.bsc.sinks.ptr as *const f32,
                                l.attn_rel_bias.ptr as *const f32, sl.k_cache.ptr as *const u16, sl.v_cache.ptr as *const u16,
                                self.bsc.gate_out.ptr as *const f32, self.bsc.d_pos0.ptr as *const i32, xq0,
                                self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                                self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 }, sl.max_ctx as i32, bi,
                            );
                        }
                    }
                    let t4 = sync_t(s);
                    t_attn += (t4 - t3).as_secs_f64() * 1000.0;
                    unsafe {
                        match l.wo.ty {
                            WeightType::Q4_0 if b >= 12 => ops::ink_gemm_q4_0_wmma(s, l.wo.buf.ptr as *const u8, xq0, self.bsc.o_raw.ptr as *mut f32, e, q_dim, bi),
                            WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, l.wo.buf.ptr as *const u8, xq0, self.bsc.o_raw.ptr as *mut f32, e, q_dim, bi),
                            _ => ops::ink_gemm_q8_0_sudot4(s, l.wo.buf.ptr as *const u8, xq0, self.bsc.o_raw.ptr as *mut f32, e, q_dim, bi),
                        };
                    }
                    let t5 = sync_t(s);
                    t_wo += (t5 - t4).as_secs_f64() * 1000.0;
                    unsafe {
                        let ap_snap = if b <= 8 { sl.attn_post_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
                        ops::ink_shortconv_res_batch_snap(s, self.bsc.o_raw.ptr as *const f32, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, ap_snap, self.bsc.cur.ptr as *const f32, std::ptr::null_mut(), self.bsc.ffn_inp.ptr as *mut f32, e, bi);
                    }
                    let t5b = sync_t(s);
                    t_aconv += (t5b - t5).as_secs_f64() * 1000.0;
                    unsafe {
                        let fnw = l.ffn_norm.ptr as *const f32;
                        if is_loop {
                            let ll = &self.loop_layers[ip - self.cfg.core_start];
                            ops::ink_mix_batch(s, self.bsc.blocks.ptr as *const f32, n_blocks.max(1) as i32, mb, self.bsc.ffn_inp.ptr as *const f32, l.ffn_res_query.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), self.bsc.mixed.ptr as *mut f32, e, 1, bi, self.cfg.eps);
                            ops::ink_loop_ctrl_1_norm_batch(
                                s, self.bsc.mixed.ptr as *const f32, self.bsc.msgs.ptr as *const f32,
                                ll.ctrl_down[3].ptr as *const f32, ll.ctrl_gain[3].ptr as *const f32, ll.ctrl_forget[3].ptr as *const f32,
                                fnw, self.bsc.ctrl_h.ptr as *mut f32, self.bsc.ffn_normed.ptr as *mut f32, e, self.cfg.gate_rank as i32, mb, bi, self.cfg.eps,
                            );
                        } else {
                            ops::ink_mix_batch(s, self.bsc.blocks.ptr as *const f32, n_blocks.max(1) as i32, mb, self.bsc.ffn_inp.ptr as *const f32, l.ffn_res_query.ptr as *const f32, fnw, fnw, fnw, self.bsc.ffn_normed.ptr as *mut f32, e, 1, bi, self.cfg.eps);
                        }
                    }
                    let t6 = sync_t(s);
                    t_fmix += (t6 - t5b).as_secs_f64() * 1000.0;
                    unsafe {
                        ops::ink_gemm_f32(s, l.ffn_gate_inp.buf.ptr as *const f32, self.bsc.ffn_normed.ptr as *const f32, self.bsc.router_logits.ptr as *mut f32, self.cfg.n_expert as i32, e, bi);
                        let bp = l.ffn_exp_probs_b.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
                        ops::ink_moe_router_batch(s, self.bsc.router_logits.ptr as *const f32, bp, self.bsc.wtop.ptr as *mut f32, self.bsc.itop.ptr as *mut i32, self.cfg.n_expert as i32, self.cfg.n_expert_used as i32, self.cfg.route_scale, bi);
                    }
                    let t7 = sync_t(s);
                    t_router += (t7 - t6).as_secs_f64() * 1000.0;
                    let gu_sc = l.ffn_gate_up_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
                    let dn_sc = l.ffn_down_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
                    unsafe {
                        ops::ink_quantize_q8_1(s, self.bsc.ffn_normed.ptr as *const f32, xq0, e, bi);
                        ops::ink_moe_build_groups(
                            s, self.bsc.itop.ptr as *const i32, self.bsc.wtop.ptr as *const f32, dn_sc,
                            self.bsc.exp_counts.ptr as *mut i32, self.bsc.exp_offsets.ptr as *mut i32,
                            self.bsc.tok_sorted.ptr as *mut i32, self.bsc.slot_sorted.ptr as *mut i32,
                            self.bsc.wt_sorted.ptr as *mut f32, self.cfg.n_expert_used as i32, bi,
                        );
                        if b >= 48 {
                            ops::ink_moe_gu_q4_0_wmma_to_q8(
                                s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, xq0,
                                self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                                self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                                self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                                self.cfg.situ_beta, self.cfg.situ_lb, bi,
                            );
                        } else {
                            ops::ink_moe_gu_q4_0_sudot4_to_q8(
                                s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, xq0,
                                self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                                self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                                self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
                                self.cfg.situ_beta, self.cfg.situ_lb, bi,
                            );
                        }
                    }
                    let t8 = sync_t(s);
                    t_mgu += (t8 - t7).as_secs_f64() * 1000.0;
                    unsafe {
                        if b >= 48 {
                            ops::ink_moe_dn_q4_0_wmma(
                                s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                                self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                                self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                                self.bsc.wt_sorted.ptr as *const f32, self.bsc.moe_dn_slots.ptr as *mut f32, self.bsc.moe_raw.ptr as *mut f32,
                                e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi,
                            );
                        } else {
                            ops::ink_moe_dn_q4_0_sudot4(
                                s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                                self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                                self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                                self.bsc.wt_sorted.ptr as *const f32, self.bsc.moe_raw.ptr as *mut f32,
                                e, self.cfg.n_ff_exp as i32, self.cfg.n_expert_used as i32, bi,
                            );
                        }
                    }
                    let t9 = sync_t(s);
                    t_mdn += (t9 - t8).as_secs_f64() * 1000.0;
                    unsafe {
                        let fp_snap = if b <= 8 { sl.ffn_post_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
                        ops::ink_shortconv_res_batch_snap(s, self.bsc.moe_raw.ptr as *const f32, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, fp_snap, self.bsc.ffn_inp.ptr as *const f32, std::ptr::null_mut(), self.bsc.cur.ptr as *mut f32, e, bi);
                    }
                    let t9b = sync_t(s);
                    t_fconv += (t9b - t9).as_secs_f64() * 1000.0;
                    if pass == 0 && ip % self.cfg.attn_res_block == 0 { n_blocks += 1; }
                }
            }
            let t0 = sync_t(s);
            unsafe {
                if b <= 8 {
                    ops::ink_quantize_q8_1(s, self.bsc.mixed.ptr as *const f32, xq0, e, bi);
                    ops::ink_gemm_q4_0_sudot4(s, self.output.buf.ptr as *const u8, xq0, self.bsc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, bi);
                } else {
                    ops::ink_quantize_q8_1(s, self.bsc.mixed.ptr as *const f32, xq0, e, 1);
                    ops::ink_gemm_q4_0_sudot4(s, self.output.buf.ptr as *const u8, xq0, self.bsc.logits.ptr as *mut f32, self.cfg.vocab_padded as i32, e, 1);
                }
            }
            let t1 = sync_t(s);
            t_lm += (t1 - t0).as_secs_f64() * 1000.0;
        }
        let inv = 1.0 / runs as f64;
        let tot = (t_amix + t_qkvrg + t_kvconv + t_attn + t_wo + t_aconv + t_fmix + t_router + t_mgu + t_mdn + t_fconv + t_lm) * inv;
        println!(
            "COLD B={:<4}: amix={:.2} qkvrg={:.2} kvconv={:.2} attn={:.2} wo={:.2} aconv={:.2} fmix={:.2} router={:.2} mgu={:.2} mdn={:.2} fconv={:.2} lm={:.2} | sum={:.2}ms",
            b, t_amix*inv, t_qkvrg*inv, t_kvconv*inv, t_attn*inv, t_wo*inv, t_aconv*inv, t_fmix*inv, t_router*inv, t_mgu*inv, t_mdn*inv, t_fconv*inv, t_lm*inv, tot
        );
        Ok(())
    }

}
