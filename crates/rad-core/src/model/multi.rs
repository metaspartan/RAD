use super::*;

impl InklingEngine {
    pub fn prefill_seq(&mut self, seq: usize, tokens: &[i32]) -> Result<(i32, usize)> {
        let mut cached = 0usize;
        if !self.seq_tokens[seq].is_empty()
            && tokens.len() > self.seq_tokens[seq].len()
            && tokens.starts_with(&self.seq_tokens[seq])
        {
            cached = self.seq_tokens[seq].len();
        } else {
            self.reset_seq(seq)?;
        }
        let suffix = &tokens[cached..];
        let mut last = 0i32;
        let mut i = 0usize;
        while i < suffix.len() {
            let chunk = &suffix[i..(i + MAX_BATCH).min(suffix.len())];
            let b = chunk.len();
            let pos0 = self.seq_pos[seq];
            let tok_bytes = unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, b * 4) };
            self.bsc.d_toks.copy_from(tok_bytes, Some(self.stream)).map_err(he)?;
            self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe {
                self.enqueue_forward_batch_seq(self.stream, seq, b, false);
                rad_hip::hipStreamSynchronize(self.stream);
            }
            let mut out_b = [0u8; 4];
            let rc = unsafe {
                rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
            };
            if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
            last = i32::from_le_bytes(out_b);
            self.seq_pos[seq] += b as i32;
            self.seq_tokens[seq].extend_from_slice(chunk);
            i += b;
        }
        if seq == 0 {
            self.pos = self.seq_pos[0];
        }
        Ok((last, cached))
    }

    pub fn prefill_seq_dspark(
        &mut self,
        seq: usize,
        tokens: &[i32],
        mut drafter: Option<&mut crate::dspark::DsparkDrafter>,
    ) -> Result<(i32, usize)> {
        let mut cached = 0usize;
        if !self.seq_tokens[seq].is_empty()
            && tokens.len() > self.seq_tokens[seq].len()
            && tokens.starts_with(&self.seq_tokens[seq])
        {
            cached = self.seq_tokens[seq].len();
        } else {
            self.reset_seq(seq)?;
            if seq == 0 {
                if let Some(d) = drafter.as_deref_mut() {
                    d.reset();
                }
            }
        }
        let suffix = &tokens[cached..];
        let mut last = 0i32;
        let mut i = 0usize;
        while i < suffix.len() {
            let chunk = &suffix[i..(i + MAX_BATCH).min(suffix.len())];
            let b = chunk.len();
            let pos0 = self.seq_pos[seq];
            let tok_bytes = unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, b * 4) };
            self.bsc.d_toks.copy_from(tok_bytes, Some(self.stream)).map_err(he)?;
            self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe {
                self.enqueue_forward_batch_seq(self.stream, seq, b, false);
                rad_hip::hipStreamSynchronize(self.stream);
            }
            if seq == 0 {
                if let Some(d) = drafter.as_deref_mut() {
                    for j in 0..b {
                        let fptr = self.gather_batch_dspark_feats(j);
                        d.inject_target_feats(fptr, pos0 + j as i32)?;
                    }
                }
            }
            let mut out_b = [0u8; 4];
            let rc = unsafe {
                rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
            };
            if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
            last = i32::from_le_bytes(out_b);
            self.seq_pos[seq] += b as i32;
            self.seq_tokens[seq].extend_from_slice(chunk);
            i += b;
        }
        if seq == 0 {
            self.pos = self.seq_pos[0];
        }
        Ok((last, cached))
    }

    pub fn step_multi(&mut self, seq_ids: &[i32], tokens: &[i32]) -> Result<Vec<i32>> {
        let b = seq_ids.len();
        assert_eq!(b, tokens.len());
        if b == 1 && seq_ids[0] == 0 && self.decode_graph.is_some() {
            self.pos = self.seq_pos[0];
            let nxt = self.step(tokens[0])?;
            self.seq_pos[0] = self.pos;
            self.seq_tokens[0].push(tokens[0]);
            self.last_logits_in_sc = true;
            return Ok(vec![nxt]);
        }
        self.last_logits_in_sc = false;
        let positions: Vec<i32> = seq_ids.iter().map(|&s| self.seq_pos[s as usize]).collect();
        let tok_bytes = unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, b * 4) };
        let seq_bytes = unsafe { std::slice::from_raw_parts(seq_ids.as_ptr() as *const u8, b * 4) };
        let pos_bytes = unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, b * 4) };
        if !self.multi_graphs.contains_key(&b) {
            let g = unsafe { CapturedGraph::capture(|s| self.enqueue_forward_multi(s, b)) }.map_err(he)?;
            self.multi_graphs.insert(b, g);
        }
        let g = self.multi_graphs.get(&b).unwrap();
        self.bsc.d_toks.copy_from(tok_bytes, Some(g.stream)).map_err(he)?;
        self.bsc.d_seq_ids.copy_from(seq_bytes, Some(g.stream)).map_err(he)?;
        self.bsc.d_positions.copy_from(pos_bytes, Some(g.stream)).map_err(he)?;
        g.launch().map_err(he)?;
        g.sync().map_err(he)?;
        let mut out_b = vec![0u8; b * 4];
        let rc = unsafe {
            rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, b * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
        for (idx, &sid) in seq_ids.iter().enumerate() {
            let su = sid as usize;
            self.seq_pos[su] += 1;
            self.seq_tokens[su].push(tokens[idx]);
            if su == 0 { self.pos = self.seq_pos[0]; }
        }
        Ok(out_b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn verify_multi(&mut self, seq_ids: &[i32], draft_tokens: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
        let n_users = seq_ids.len();
        assert_eq!(n_users, draft_tokens.len());
        assert!(n_users >= 1 && n_users <= 16);
        let k_len = draft_tokens[0].len();
        assert!(k_len >= 1 && k_len <= self.multi_verify_cap.min(256));
        let total_b = n_users * k_len;
        assert!(total_b <= self.multi_verify_cap);
        let mut flat_toks = Vec::with_capacity(total_b);
        let mut flat_seqs = Vec::with_capacity(total_b);
        let mut flat_pos = Vec::with_capacity(total_b);
        for u in 0..n_users {
            assert_eq!(draft_tokens[u].len(), k_len);
            let seq = seq_ids[u];
            let pos0 = self.seq_pos[seq as usize];
            for k in 0..k_len {
                flat_toks.push(draft_tokens[u][k]);
                flat_seqs.push(seq);
                flat_pos.push(pos0 + k as i32);
            }
        }
        let tok_bytes = unsafe { std::slice::from_raw_parts(flat_toks.as_ptr() as *const u8, total_b * 4) };
        let seq_bytes = unsafe { std::slice::from_raw_parts(flat_seqs.as_ptr() as *const u8, total_b * 4) };
        let pos_bytes = unsafe { std::slice::from_raw_parts(flat_pos.as_ptr() as *const u8, total_b * 4) };
        let key = (n_users, k_len);
        if !self.multi_verify_graphs.contains_key(&key) {
            let g = unsafe { CapturedGraph::capture(|s| self.enqueue_forward_ex(s, total_b, true, None, k_len)) }.map_err(he)?;
            self.multi_verify_graphs.insert(key, g);
        }
        let g = self.multi_verify_graphs.get(&key).unwrap();
        self.bsc.d_toks.copy_from(tok_bytes, Some(g.stream)).map_err(he)?;
        self.bsc.d_seq_ids.copy_from(seq_bytes, Some(g.stream)).map_err(he)?;
        self.bsc.d_positions.copy_from(pos_bytes, Some(g.stream)).map_err(he)?;
        g.launch().map_err(he)?;
        g.sync().map_err(he)?;
        let mut out_b = vec![0u8; total_b * 4];
        let rc = unsafe {
            rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, total_b * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
        for &sid in seq_ids {
            let su = sid as usize;
            self.seq_pos[su] += k_len as i32;
            if su == 0 { self.pos = self.seq_pos[0]; }
        }
        let flat_out: Vec<i32> = out_b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(flat_out.chunks_exact(k_len).map(|c| c.to_vec()).collect())
    }

    pub fn rewind_verify_multi(&mut self, seq_ids: &[i32], n_acc: &[usize], k_len: usize) -> Result<()> {
        let kv_n = (self.cfg.n_head_kv * self.cfg.dh * 3) as i32;
        let e_n = (self.cfg.n_embd * 3) as i32;
        let mut rw_u: Vec<i32> = Vec::with_capacity(seq_ids.len());
        let mut rw_seq: Vec<i32> = Vec::with_capacity(seq_ids.len());
        let mut rw_idx: Vec<i32> = Vec::with_capacity(seq_ids.len());
        for (u, &sid) in seq_ids.iter().enumerate() {
            let acc = n_acc[u];
            if acc < k_len {
                let su = sid as usize;
                rw_u.push(u as i32);
                rw_seq.push(sid);
                rw_idx.push((acc - 1) as i32);
                self.seq_pos[su] -= (k_len - acc) as i32;
                if su == 0 { self.pos = self.seq_pos[0]; }
            }
        }
        if !rw_u.is_empty() {
            let n_rw = rw_u.len();
            let u_b = unsafe { std::slice::from_raw_parts(rw_u.as_ptr() as *const u8, n_rw * 4) };
            let s_b = unsafe { std::slice::from_raw_parts(rw_seq.as_ptr() as *const u8, n_rw * 4) };
            let i_b = unsafe { std::slice::from_raw_parts(rw_idx.as_ptr() as *const u8, n_rw * 4) };
            self.d_rw_u.copy_from(u_b, Some(self.stream)).map_err(he)?;
            self.d_rw_seq.copy_from(s_b, Some(self.stream)).map_err(he)?;
            self.d_rw_idx.copy_from(i_b, Some(self.stream)).map_err(he)?;
            unsafe {
                ops::ink_rewind_multi_all_slots(
                    self.stream,
                    self.d_rewind_ptrs.ptr as *const u64,
                    self.d_rw_u.ptr as *const i32,
                    self.d_rw_seq.ptr as *const i32,
                    self.d_rw_idx.ptr as *const i32,
                    self.slots.len() as i32,
                    n_rw as i32,
                    kv_n,
                    e_n,
                    k_len as i32,
                );
                rad_hip::hipStreamSynchronize(self.stream);
            }
        }
        Ok(())
    }

    unsafe fn enqueue_layer_batch_ex(&self, s: HipStream, il: usize, ip: usize, n_blocks: &mut usize, loop_idx: Option<usize>, b: usize, seq_opt: Option<usize>, multi_k: usize, _multi_seqs: Option<&[i32]>) {
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

        let attn_out = self.bsc.attn_out.ptr as *mut f32;
        let mut attn_already_q8 = false;
        if let Some(seq) = seq_opt {
            let seq_kv_off = seq * self.cfg.n_head_kv * sl.max_ctx * self.cfg.dh;
            let seq_kh_off = seq * (kv_dim as usize) * 3;
            let seq_sc_off = seq * self.cfg.n_head_kv * sl.max_ctx;
            let d_pos0 = self.bsc.d_pos0.ptr as *const i32;
            if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
                let k_snap_p = if multi_k > 1 { sl.k_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
                let v_snap_p = if multi_k > 1 { sl.v_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
                ops::ink_kv_conv_qkvr_store_batch_snap_q8kv(
                    s, q, k_raw, v_raw, r,
                    l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                    if l.is_swa { std::ptr::null() } else { k_dyn },
                    if l.is_swa { std::ptr::null() } else { v_dyn },
                    (sl.k_hist.ptr as *mut f32).add(seq_kh_off),
                    (sl.v_hist.ptr as *mut f32).add(seq_kh_off),
                    k_snap_p, v_snap_p,
                    l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                    (sl.k_cache.ptr as *mut i8).add(seq_kv_off),
                    (sl.v_cache.ptr as *mut i8).add(seq_kv_off),
                    (ks.ptr as *mut u16).add(seq_sc_off),
                    (vs.ptr as *mut u16).add(seq_sc_off),
                    d_pos0, bi,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                );
                if b > 128 {
                    ops::ink_attn_batch_gated_flash_q8kv(
                        s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                        (sl.k_cache.ptr as *const i8).add(seq_kv_off),
                        (sl.v_cache.ptr as *const i8).add(seq_kv_off),
                        (ks.ptr as *const u16).add(seq_sc_off),
                        (vs.ptr as *const u16).add(seq_sc_off),
                        gate_out, d_pos0, attn_out,
                        self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        self.cfg.rel_extent as i32, sl.max_ctx as i32, bi,
                    );
                } else {
                    ops::ink_attn_b8_flash_q8_q8kv(
                        s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                        (sl.k_cache.ptr as *const i8).add(seq_kv_off),
                        (sl.v_cache.ptr as *const i8).add(seq_kv_off),
                        (ks.ptr as *const u16).add(seq_sc_off),
                        (vs.ptr as *const u16).add(seq_sc_off),
                        gate_out, d_pos0, self.bsc.xq.ptr,
                        self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        self.cfg.rel_extent as i32, sl.max_ctx as i32, bi,
                    );
                    attn_already_q8 = true;
                }
            } else {
                if multi_k > 1 {
                    ops::ink_kv_conv_qkvr_store_batch_snap(
                        s, q, k_raw, v_raw, r,
                        l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                        if l.is_swa { std::ptr::null() } else { k_dyn },
                        if l.is_swa { std::ptr::null() } else { v_dyn },
                        (sl.k_hist.ptr as *mut f32).add(seq_kh_off),
                        (sl.v_hist.ptr as *mut f32).add(seq_kh_off),
                        sl.k_snap.ptr as *mut f32, sl.v_snap.ptr as *mut f32,
                        l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                        (sl.k_cache.ptr as *mut u16).add(seq_kv_off),
                        (sl.v_cache.ptr as *mut u16).add(seq_kv_off),
                        d_pos0, bi, if sl.is_swa { 1 } else { 0 },
                        self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                    );
                } else {
                    ops::ink_kv_conv_qkvr_store_batch(
                        s, q, k_raw, v_raw, r,
                        l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                        if l.is_swa { std::ptr::null() } else { k_dyn },
                        if l.is_swa { std::ptr::null() } else { v_dyn },
                        (sl.k_hist.ptr as *mut f32).add(seq_kh_off),
                        (sl.v_hist.ptr as *mut f32).add(seq_kh_off),
                        l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                        (sl.k_cache.ptr as *mut u16).add(seq_kv_off),
                        (sl.v_cache.ptr as *mut u16).add(seq_kv_off),
                        d_pos0, bi, if sl.is_swa { 1 } else { 0 },
                        self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                    );
                }
                if b >= 16 {
                    ops::ink_attn_batch_gated_flash(
                        s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                        (sl.k_cache.ptr as *const u16).add(seq_kv_off),
                        (sl.v_cache.ptr as *const u16).add(seq_kv_off),
                        gate_out, d_pos0, attn_out,
                        self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 },
                        sl.max_ctx as i32, bi,
                    );
                } else {
                    ops::ink_attn_batch_gated_q8(
                        s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                        (sl.k_cache.ptr as *const u16).add(seq_kv_off),
                        (sl.v_cache.ptr as *const u16).add(seq_kv_off),
                        gate_out, d_pos0, self.bsc.xq.ptr,
                        self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                        self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 },
                        sl.max_ctx as i32, bi,
                    );
                    attn_already_q8 = true;
                }
            }
        } else if let (Some(ref ks), Some(ref vs)) = (&sl.k_scale, &sl.v_scale) {
            if multi_k > 1 {
                let n_users = (b / multi_k) as i32;
                ops::ink_kv_conv_qkvr_store_multi_snap_q8kv(
                    s, q, k_raw, v_raw, r,
                    l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                    if l.is_swa { std::ptr::null() } else { k_dyn },
                    if l.is_swa { std::ptr::null() } else { v_dyn },
                    sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32,
                    sl.k_snap.ptr as *mut f32, sl.v_snap.ptr as *mut f32,
                    l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                    sl.k_cache.ptr as *mut i8, sl.v_cache.ptr as *mut i8,
                    ks.ptr as *mut u16, vs.ptr as *mut u16,
                    self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    n_users, multi_k as i32,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                );
            } else {
                ops::ink_kv_conv_qkvr_store_multi_q8kv(
                    s, q, k_raw, v_raw, r,
                    l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                    if l.is_swa { std::ptr::null() } else { k_dyn },
                    if l.is_swa { std::ptr::null() } else { v_dyn },
                    sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32,
                    l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                    sl.k_cache.ptr as *mut i8, sl.v_cache.ptr as *mut i8,
                    ks.ptr as *mut u16, vs.ptr as *mut u16,
                    self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    bi,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                );
            }
            if multi_k >= 2 && multi_k <= 16 && b <= 128 && self.cfg.n_head == self.cfg.n_head_kv * 3 && self.cfg.dh == 128 {
                let n_users = (b / multi_k) as i32;
                ops::ink_attn_multi_b8_flash_q8_q8kv(
                    s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const i8, sl.v_cache.ptr as *const i8,
                    ks.ptr as *const u16, vs.ptr as *const u16,
                    gate_out, self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    self.bsc.xq.ptr,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32, sl.max_ctx as i32, n_users, multi_k as i32,
                );
                attn_already_q8 = true;
            } else if multi_k >= 8 && self.cfg.n_head == self.cfg.n_head_kv * 3 && self.cfg.dh == 128 {
                let n_users = (b / multi_k) as i32;
                ops::ink_attn_multi_gated_flash_q8kv(
                    s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const i8, sl.v_cache.ptr as *const i8,
                    ks.ptr as *const u16, vs.ptr as *const u16,
                    gate_out, self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    attn_out,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32, sl.max_ctx as i32, n_users, multi_k as i32,
                );
            } else {
                ops::ink_attn_multi_gated_q8_q8kv(
                    s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const i8, sl.v_cache.ptr as *const i8,
                    ks.ptr as *const u16, vs.ptr as *const u16,
                    gate_out, self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    self.bsc.xq.ptr,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32, sl.max_ctx as i32, bi,
                );
                attn_already_q8 = true;
            }
        } else {
            if multi_k > 1 {
                let n_users = (b / multi_k) as i32;
                ops::ink_kv_conv_qkvr_store_multi_snap(
                    s, q, k_raw, v_raw, r,
                    l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                    if l.is_swa { std::ptr::null() } else { k_dyn },
                    if l.is_swa { std::ptr::null() } else { v_dyn },
                    sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32,
                    sl.k_snap.ptr as *mut f32, sl.v_snap.ptr as *mut f32,
                    l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                    sl.k_cache.ptr as *mut u16, sl.v_cache.ptr as *mut u16,
                    self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    n_users, multi_k as i32, if sl.is_swa { 1 } else { 0 },
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                );
            } else {
                ops::ink_kv_conv_qkvr_store_multi(
                    s, q, k_raw, v_raw, r,
                    l.attn_k_conv.ptr as *const f32, l.attn_v_conv.ptr as *const f32,
                    if l.is_swa { std::ptr::null() } else { k_dyn },
                    if l.is_swa { std::ptr::null() } else { v_dyn },
                    sl.k_hist.ptr as *mut f32, sl.v_hist.ptr as *mut f32,
                    l.attn_q_clip.ptr as *const f32, l.attn_rel_clip.ptr as *const f32,
                    sl.k_cache.ptr as *mut u16, sl.v_cache.ptr as *mut u16,
                    self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    bi, if sl.is_swa { 1 } else { 0 },
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    sl.max_ctx as i32, self.cfg.qk_clip, self.cfg.rel_clip, self.cfg.log_floor, self.cfg.log_alpha,
                );
            }
            if multi_k >= 8 && self.cfg.n_head == self.cfg.n_head_kv * 3 && self.cfg.dh == 128 {
                let n_users = (b / multi_k) as i32;
                ops::ink_attn_multi_gated_flash(
                    s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const u16, sl.v_cache.ptr as *const u16,
                    gate_out, self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    attn_out,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 },
                    sl.max_ctx as i32, n_users, multi_k as i32,
                );
            } else {
                ops::ink_attn_multi_gated_q8(
                    s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                    sl.k_cache.ptr as *const u16, sl.v_cache.ptr as *const u16,
                    gate_out, self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                    self.bsc.xq.ptr,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 },
                    sl.max_ctx as i32, bi,
                );
                attn_already_q8 = true;
            }
        }

        let o_raw = self.bsc.o_raw.ptr as *mut f32;
        if !attn_already_q8 {
            ops::ink_quantize_q8_1(s, attn_out, self.bsc.xq.ptr, q_dim, bi);
        }
        let block_dst = if ip % self.cfg.attn_res_block == 0 {
            let dst = blocks.add((*n_blocks) * MAX_BATCH * self.cfg.n_embd);
            *n_blocks += 1;
            dst
        } else {
            std::ptr::null_mut()
        };
        if let Some(seq) = seq_opt {
            let seq_eh_off = seq * self.cfg.n_embd * 3;
            let ap_snap = if multi_k > 1 { sl.attn_post_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
            if l.wo.ty == WeightType::Q4_0 && (b == 4 || b == 6 || b == 8) {
                ops::ink_wo_conv_res_q4_batch_snap(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, l.attn_post_conv.ptr as *const f32, (sl.attn_post_hist.ptr as *mut f32).add(seq_eh_off), ap_snap, cur, block_dst, ffn_inp, e, q_dim, bi);
            } else {
                match l.wo.ty {
                    WeightType::Q4_0 if b >= 12 => ops::ink_gemm_q4_0_wmma(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
                    WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
                    _ => ops::ink_gemm_q8_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
                };
                if multi_k > 1 {
                    ops::ink_shortconv_res_batch_snap(s, o_raw, l.attn_post_conv.ptr as *const f32, (sl.attn_post_hist.ptr as *mut f32).add(seq_eh_off), ap_snap, cur, block_dst, ffn_inp, e, bi);
                } else {
                    ops::ink_shortconv_res_batch(s, o_raw, l.attn_post_conv.ptr as *const f32, (sl.attn_post_hist.ptr as *mut f32).add(seq_eh_off), cur, block_dst, ffn_inp, e, bi);
                }
            }
        } else {
            match l.wo.ty {
                WeightType::Q4_0 if b >= 12 => ops::ink_gemm_q4_0_wmma(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
                WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
                _ => ops::ink_gemm_q8_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
            };
            if multi_k > 1 {
                ops::ink_shortconv_res_multi_snap(s, o_raw, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, sl.attn_post_snap.ptr as *mut f32, cur, block_dst, ffn_inp, self.bsc.d_seq_ids.ptr as *const i32, e, (b / multi_k) as i32, multi_k as i32);
            } else {
                ops::ink_shortconv_res_multi(s, o_raw, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, cur, block_dst, ffn_inp, self.bsc.d_seq_ids.ptr as *const i32, e, bi);
            }
        }

        let ffn_normed = self.bsc.ffn_normed.ptr as *mut f32;
        if let Some(li) = loop_idx {
            ops::ink_mix_batch(s, blocks, *n_blocks as i32, mb, ffn_inp, l.ffn_res_query.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), mixed, e, 1, bi, self.cfg.eps);
            let ll = &self.loop_layers[li];
            let rank = self.cfg.gate_rank as i32;
            let ch = self.bsc.ctrl_h.ptr as *mut f32;
            ops::ink_loop_ctrl_1_norm_batch(
                s, mixed, msgs.add(3 * MAX_BATCH * self.cfg.n_embd),
                ll.ctrl_down[3].ptr as *const f32, ll.ctrl_gain[3].ptr as *const f32, ll.ctrl_forget[3].ptr as *const f32,
                l.ffn_norm.ptr as *const f32, ch, ffn_normed, e, rank, mb, bi, self.cfg.eps,
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
        let use_nozero = (seq_opt.is_none() || multi_k > 1) && b <= 128;
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
                } else if use_nozero {
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
        if let Some(seq) = seq_opt {
            let seq_eh_off = seq * self.cfg.n_embd * 3;
            if multi_k > 1 {
                if use_nozero && l.ffn_down_exps.ty == WeightType::Q4_0 {
                    ops::ink_shortconv_res_batch_snap_clear(s, moe_raw, l.ffn_post_conv.ptr as *const f32, (sl.ffn_post_hist.ptr as *mut f32).add(seq_eh_off), sl.ffn_post_snap.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, e, bi);
                } else {
                    ops::ink_shortconv_res_batch_snap(s, moe_raw, l.ffn_post_conv.ptr as *const f32, (sl.ffn_post_hist.ptr as *mut f32).add(seq_eh_off), sl.ffn_post_snap.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, e, bi);
                }
            } else {
                ops::ink_shortconv_res_batch(s, moe_raw, l.ffn_post_conv.ptr as *const f32, (sl.ffn_post_hist.ptr as *mut f32).add(seq_eh_off), ffn_inp, std::ptr::null_mut(), cur, e, bi);
            }
        } else if multi_k > 1 {
            if use_nozero && l.ffn_down_exps.ty == WeightType::Q4_0 {
                ops::ink_shortconv_res_multi_snap_clear(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, sl.ffn_post_snap.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, self.bsc.d_seq_ids.ptr as *const i32, e, (b / multi_k) as i32, multi_k as i32);
            } else {
                ops::ink_shortconv_res_multi_snap(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, sl.ffn_post_snap.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, self.bsc.d_seq_ids.ptr as *const i32, e, (b / multi_k) as i32, multi_k as i32);
            }
        } else if use_nozero && l.ffn_down_exps.ty == WeightType::Q4_0 {
            ops::ink_shortconv_res_multi_clear(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, self.bsc.d_seq_ids.ptr as *const i32, e, bi);
        } else {
            ops::ink_shortconv_res_multi(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, self.bsc.d_seq_ids.ptr as *const i32, e, bi);
        }
    }

    pub unsafe fn enqueue_forward_batch_seq(&self, s: HipStream, seq: usize, b: usize, all_logits: bool) {
        self.enqueue_forward_ex_emb(s, b, all_logits, Some(seq), 1, false);
    }

    pub unsafe fn enqueue_forward_multi(&self, s: HipStream, b: usize) {
        self.enqueue_forward_ex_emb(s, b, true, None, 1, false);
    }

    pub fn prefill_tokens_seq(&mut self, seq: usize, tokens: &[i32]) -> Result<i32> {
        self.last_logits_in_sc = false;
        let mut last = 0i32;
        let mut i = 0usize;
        while i < tokens.len() {
            let chunk = &tokens[i..(i + MAX_BATCH).min(tokens.len())];
            let b = chunk.len();
            let pos0 = self.seq_pos[seq];
            let tok_bytes = unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, b * 4) };
            self.bsc.d_toks.copy_from(tok_bytes, Some(self.stream)).map_err(he)?;
            self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe {
                self.enqueue_forward_ex_emb(self.stream, b, false, Some(seq), 1, false);
                rad_hip::hipStreamSynchronize(self.stream);
            }
            let mut out_b = [0u8; 4];
            let rc = unsafe {
                rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
            };
            if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
            last = i32::from_le_bytes(out_b);
            self.seq_pos[seq] += b as i32;
            self.seq_tokens[seq].extend_from_slice(chunk);
            if seq == 0 {
                self.pos = self.seq_pos[0];
            }
            i += b;
        }
        Ok(last)
    }

    pub fn prefill_embeds_seq(&mut self, seq: usize, embeds: &[f32], n_tok: usize) -> Result<i32> {
        assert_eq!(embeds.len(), n_tok * self.cfg.n_embd);
        let mut last = 0i32;
        let mut i = 0usize;
        while i < n_tok {
            let chunk_n = (n_tok - i).min(MAX_BATCH);
            let pos0 = self.seq_pos[seq];
            let slice = &embeds[(i * self.cfg.n_embd)..((i + chunk_n) * self.cfg.n_embd)];
            let ebytes = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
            self.bsc.cur.copy_from(ebytes, Some(self.stream)).map_err(he)?;
            self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe {
                self.enqueue_forward_ex_emb(self.stream, chunk_n, false, Some(seq), 1, true);
                rad_hip::hipStreamSynchronize(self.stream);
            }
            let mut out_b = [0u8; 4];
            let rc = unsafe {
                rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
            };
            if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
            last = i32::from_le_bytes(out_b);
            self.seq_pos[seq] += chunk_n as i32;
            for _ in 0..chunk_n {
                self.seq_tokens[seq].push(-1);
            }
            if seq == 0 {
                self.pos = self.seq_pos[0];
            }
            i += chunk_n;
        }
        Ok(last)
    }

    pub(super) unsafe fn enqueue_forward_ex(&self, s: HipStream, b: usize, all_logits: bool, seq_opt: Option<usize>, multi_k: usize) {
        self.enqueue_forward_ex_emb(s, b, all_logits, seq_opt, multi_k, false);
    }

    unsafe fn enqueue_forward_ex_emb(&self, s: HipStream, b: usize, all_logits: bool, seq_opt: Option<usize>, multi_k: usize, skip_embed: bool) {
        self.enqueue_forward_ex_emb_m(s, b, all_logits, seq_opt, multi_k, skip_embed, None);
    }

    unsafe fn enqueue_forward_ex_emb_m(&self, s: HipStream, b: usize, all_logits: bool, seq_opt: Option<usize>, multi_k: usize, skip_embed: bool, multi_seqs: Option<&[i32]>) {
        let e = self.cfg.n_embd as i32;
        let mb = MAX_BATCH as i32;
        let bi = b as i32;
        let be = (b * self.cfg.n_embd) as i32;
        if (seq_opt.is_none() || multi_k > 1) && b <= 128 {
            ops::ink_zero_f32(s, self.bsc.moe_raw.ptr as *mut f32, be);
        }
        let cur = self.bsc.cur.ptr as *mut f32;
        let blocks = self.bsc.blocks.ptr as *mut f32;
        let prelude = self.bsc.prelude.ptr as *mut f32;
        let bank = self.bsc.bank.ptr as *mut f32;
        let bank_out = self.bsc.bank_out.ptr as *mut f32;
        let msgs = self.bsc.msgs.ptr as *mut f32;
        let feats = self.bsc.dspark_feats.ptr as *mut f32;

        if !skip_embed {
            ops::ink_embed_q8_0_norm_batch(s, self.tok_embd.buf.ptr as *const u8, self.tok_norm.ptr as *const f32, self.bsc.d_toks.ptr as *const i32, cur, e, bi);
        }

        let mut n_blocks = 0usize;
        let mut visit = 0usize;
        for il in 0..self.cfg.core_start {
            self.enqueue_layer_batch_ex(s, il, il, &mut n_blocks, None, b, seq_opt, multi_k, multi_seqs);
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
                    self.enqueue_layer_batch_ex(s, il, ip, &mut n_blocks, None, b, seq_opt, multi_k, multi_seqs);
                } else {
                    let li = ip - self.cfg.core_start;
                    let ll = &self.loop_layers[li];
                    ops::ink_mix_batch(s, bank, bank_len as i32, mb, bank_out, ll.msg_query.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), msgs, e, 4, bi, self.cfg.eps);
                    self.enqueue_layer_batch_ex(s, il, ip, &mut n_blocks, Some(li), b, seq_opt, multi_k, multi_seqs);
                }
                visit += 1;
                if visit == 23 { ops::ink_copy_f32(s, cur, feats.add(MAX_BATCH * self.cfg.n_embd), be); }
                else if visit == 41 { ops::ink_copy_f32(s, cur, feats.add(2 * MAX_BATCH * self.cfg.n_embd), be); }
            }
        }
        for il in self.cfg.core_end..self.cfg.n_phys {
            self.enqueue_layer_batch_ex(s, il, il, &mut n_blocks, None, b, seq_opt, multi_k, multi_seqs);
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

}
