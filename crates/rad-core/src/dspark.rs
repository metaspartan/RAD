use crate::model::{DevWeight, InklingEngine, WeightType};
use crate::ops;
use anyhow::{anyhow, Result};
use rad_gguf::GgufFile;
use rad_hip::{CapturedGraph, DevBuf, HipError, HipStream};
use std::path::Path;

fn he(e: HipError) -> anyhow::Error {
    anyhow!("hip error rc={e}")
}

struct DraftLayer {
    is_swa: bool,
    max_ctx: usize,
    attn_norm: DevBuf,
    dflare_norm: DevBuf,
    wq: DevWeight,
    wk: DevWeight,
    wv: DevWeight,
    wo: DevWeight,
    wk_ctx: DevWeight,
    wv_ctx: DevWeight,
    q_norm: DevBuf,
    k_norm: DevBuf,
    k_ctx_norm: DevBuf,
    ffn_norm: DevBuf,
    ffn_gate: DevWeight,
    ffn_up: DevWeight,
    ffn_down: DevWeight,
    k_cache: DevBuf,
    v_cache: DevBuf,
}

pub struct DsparkDrafter {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub dh: usize,
    pub n_ff: usize,
    pub rank: usize,
    pub block_size: usize,
    pub swa_window: usize,
    pub freq_base: f32,
    pub eps: f32,
    pub max_seqs: usize,
    pub ctx_len: i32,
    pub seq_ctx_len: Vec<i32>,
    dspark_fuse: DevBuf,
    dspark_norm: DevBuf,
    mask_embd: DevBuf,
    output_norm: DevBuf,
    joint_proj: DevWeight,
    joint_proj_b: DevBuf,
    hidden_out: DevWeight,
    hidden_out_b: DevBuf,
    markov_w1: DevBuf,
    layers: Vec<DraftLayer>,
    fused_ctx: DevBuf,
    ctx_normed: DevBuf,
    ctx_k: DevBuf,
    ctx_v: DevBuf,
    ctx_m: DevBuf,
    x: DevBuf,
    x_norm: DevBuf,
    xq: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    attn_out: DevBuf,
    o_out: DevBuf,
    gate: DevBuf,
    up: DevBuf,
    ffn_out: DevBuf,
    hidden: DevBuf,
    state: DevBuf,
    markov_inp: DevBuf,
    joint: DevBuf,
    markov_out: DevBuf,
    delta_h: DevBuf,
    final_h: DevBuf,
    logits: DevBuf,
    d_draft_toks: DevBuf,
    d_pos: DevBuf,
    d_ctx_len: DevBuf,
    d_seq_ids: DevBuf,
    d_anchors: DevBuf,
    d_slot_out: DevBuf,
    anchors_emb: DevBuf,
    stream: HipStream,
    draft_graph: Option<(usize, CapturedGraph)>,
    seq_draft_graphs: std::collections::HashMap<(usize, usize), CapturedGraph>,
    multi_draft_graphs: std::collections::HashMap<(usize, usize), CapturedGraph>,
}

unsafe impl Send for DsparkDrafter {}

impl DsparkDrafter {
    pub fn load(path: &Path, max_ctx: usize) -> Result<Self> {
        Self::load_multi(path, max_ctx, 1)
    }

    pub fn load_multi(path: &Path, max_ctx: usize, max_seqs: usize) -> Result<Self> {
        let max_seqs = max_seqs.max(1);
        let g = GgufFile::open(path)?;
        let q4_draft = std::env::var("RAD_DSPARK_Q4").map(|v| v != "0").unwrap_or(true);
        let n_layers = g.kv_u64("dflash.block_count").unwrap_or(3) as usize;
        let n_embd = g.kv_u64("dflash.embedding_length").unwrap_or(2304) as usize;
        let n_head = g.kv_u64("dflash.attention.head_count").unwrap_or(24) as usize;
        let n_head_kv = g.kv_u64("dflash.attention.head_count_kv").unwrap_or(8) as usize;
        let dh = g.kv_u64("dflash.attention.key_length").unwrap_or(128) as usize;
        let n_ff = g.kv_u64("dflash.feed_forward_length").unwrap_or(9216) as usize;
        let rank = g.kv_u64("dflash.dspark.markov_rank").unwrap_or(256) as usize;
        let block_size = g.kv_u64("dflash.block_size").unwrap_or(8) as usize;
        let swa_window = g.kv_u64("dflash.attention.sliding_window").unwrap_or(1024) as usize;
        let freq_base = g.kv_f32("dflash.rope.freq_base").unwrap_or(1000000.0);
        let eps = g.kv_f32("dflash.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let swa_pat = g.kv_bool_arr("dflash.attention.sliding_window_pattern").unwrap_or_else(|| vec![true, false, true]);

        let load_raw = |name: &str| -> Result<DevBuf> {
            let bytes = g.tensor_bytes(name).ok_or_else(|| anyhow!("missing {name}"))?;
            let mut b = DevBuf::new(bytes.len()).map_err(he)?;
            b.copy_from(bytes, None).map_err(he)?;
            Ok(b)
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let is_swa = swa_pat.get(i).copied().unwrap_or(false);
            let l_ctx = if is_swa { (swa_window + 16).min(max_ctx) } else { max_ctx };
            let kv_bytes = max_seqs * n_head_kv * l_ctx * dh * 2;
            let mut k_cache = DevBuf::new(kv_bytes).map_err(he)?;
            let mut v_cache = DevBuf::new(kv_bytes).map_err(he)?;
            k_cache.zero().map_err(he)?;
            v_cache.zero().map_err(he)?;
            layers.push(DraftLayer {
                is_swa,
                max_ctx: l_ctx,
                attn_norm: load_raw(&p("attn_norm.weight"))?,
                dflare_norm: load_raw(&p("dflare_norm.weight"))?,
                wq: DevWeight::load_maybe_q4(&g, &p("attn_q.weight"), q4_draft)?,
                wk: DevWeight::load_maybe_q4(&g, &p("attn_k.weight"), q4_draft)?,
                wv: DevWeight::load_maybe_q4(&g, &p("attn_v.weight"), q4_draft)?,
                wo: DevWeight::load_maybe_q4(&g, &p("attn_output.weight"), q4_draft)?,
                wk_ctx: DevWeight::load(&g, &p("attn_k_ctx.weight"))?,
                wv_ctx: DevWeight::load(&g, &p("attn_v_ctx.weight"))?,
                q_norm: load_raw(&p("attn_q_norm.weight"))?,
                k_norm: load_raw(&p("attn_k_norm.weight"))?,
                k_ctx_norm: load_raw(&p("attn_k_ctx_norm.weight"))?,
                ffn_norm: load_raw(&p("ffn_norm.weight"))?,
                ffn_gate: DevWeight::load_maybe_q4(&g, &p("ffn_gate.weight"), q4_draft)?,
                ffn_up: DevWeight::load_maybe_q4(&g, &p("ffn_up.weight"), q4_draft)?,
                ffn_down: DevWeight::load_maybe_q4(&g, &p("ffn_down.weight"), q4_draft)?,
                k_cache,
                v_cache,
            });
        }

        let kmax = block_size;
        let q_dim = n_head * dh;
        let kv_dim = n_head_kv * dh;
        let max_k_dim = n_ff.max(q_dim).max(n_embd);

        extern "C" {
            fn hipStreamCreate(s: *mut HipStream) -> i32;
        }
        let mut stream: HipStream = std::ptr::null_mut();
        unsafe { hipStreamCreate(&mut stream) };

        Ok(Self {
            n_layers,
            n_embd,
            n_head,
            n_head_kv,
            dh,
            n_ff,
            rank,
            block_size,
            swa_window,
            freq_base,
            eps,
            max_seqs,
            ctx_len: 0,
            seq_ctx_len: vec![0; max_seqs],
            dspark_fuse: load_raw("dspark_fuse.weight")?,
            dspark_norm: load_raw("dspark_norm.weight")?,
            mask_embd: load_raw("mask_embd.weight")?,
            output_norm: load_raw("output_norm.weight")?,
            joint_proj: DevWeight::load(&g, "joint_proj.weight")?,
            joint_proj_b: load_raw("joint_proj.bias")?,
            hidden_out: DevWeight::load(&g, "dspark_hidden_out.weight")?,
            hidden_out_b: load_raw("dspark_hidden_out.bias")?,
            markov_w1: load_raw("markov_w1.weight")?,
            layers,
            fused_ctx: DevBuf::new(4 * n_embd * 4).map_err(he)?,
            ctx_normed: DevBuf::new(n_embd * 4).map_err(he)?,
            ctx_k: DevBuf::new(kv_dim * 4).map_err(he)?,
            ctx_v: DevBuf::new(kv_dim * 4).map_err(he)?,
            ctx_m: DevBuf::new(max_seqs * n_embd * 4).map_err(he)?,
            x: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            x_norm: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            xq: DevBuf::new(max_seqs * kmax * (max_k_dim / 32) * 40).map_err(he)?,
            q: DevBuf::new(max_seqs * kmax * q_dim * 4).map_err(he)?,
            k: DevBuf::new(max_seqs * kmax * kv_dim * 4).map_err(he)?,
            v: DevBuf::new(max_seqs * kmax * kv_dim * 4).map_err(he)?,
            attn_out: DevBuf::new(max_seqs * kmax * q_dim * 4).map_err(he)?,
            o_out: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            gate: DevBuf::new(max_seqs * kmax * n_ff * 4).map_err(he)?,
            up: DevBuf::new(max_seqs * kmax * n_ff * 4).map_err(he)?,
            ffn_out: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            hidden: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            state: DevBuf::new(max_seqs * rank * 4).map_err(he)?,
            markov_inp: DevBuf::new(max_seqs * (2 * rank + 2 * n_embd) * 4).map_err(he)?,
            joint: DevBuf::new(max_seqs * 3 * rank * 4).map_err(he)?,
            markov_out: DevBuf::new(max_seqs * rank * 4).map_err(he)?,
            delta_h: DevBuf::new(max_seqs * n_embd * 4).map_err(he)?,
            final_h: DevBuf::new(max_seqs * n_embd * 4).map_err(he)?,
            logits: DevBuf::new(max_seqs * 201024 * 4).map_err(he)?,
            d_draft_toks: DevBuf::new(max_seqs * kmax * 4).map_err(he)?,
            d_pos: DevBuf::new(max_seqs * 4).map_err(he)?,
            d_ctx_len: DevBuf::new(max_seqs * 4).map_err(he)?,
            d_seq_ids: DevBuf::new(max_seqs * 4).map_err(he)?,
            d_anchors: DevBuf::new(max_seqs * 4).map_err(he)?,
            d_slot_out: DevBuf::new(max_seqs * 4).map_err(he)?,
            anchors_emb: DevBuf::new(max_seqs * n_embd * 4).map_err(he)?,
            stream,
            draft_graph: None,
            seq_draft_graphs: std::collections::HashMap::new(),
            multi_draft_graphs: std::collections::HashMap::new(),
        })
    }

    pub fn reset(&mut self) {
        self.ctx_len = 0;
        self.seq_ctx_len[0] = 0;
    }

    pub fn reset_seq(&mut self, seq: usize) {
        self.seq_ctx_len[seq] = 0;
        if seq == 0 {
            self.ctx_len = 0;
        }
    }

    pub fn inject_target_feats_seq(&mut self, seq: usize, feats_ptr: *const f32, pos: i32) -> Result<()> {
        let e = self.n_embd as i32;
        let kv_dim = (self.n_head_kv * self.dh) as i32;
        self.d_pos.copy_from(&pos.to_le_bytes(), Some(self.stream)).map_err(he)?;
        unsafe {
            ops::ink_dspark_fuse(
                self.stream,
                feats_ptr,
                self.dspark_fuse.ptr as *const f32,
                self.fused_ctx.ptr as *mut f32,
                e,
                1,
            );
            for (il, l) in self.layers.iter().enumerate() {
                let kv_off = seq * self.n_head_kv * l.max_ctx * self.dh;
                let f_row = (self.fused_ctx.ptr as *const f32).add(il * self.n_embd);
                ops::ink_rmsnorm(self.stream, f_row, l.dflare_norm.ptr as *const f32, self.ctx_normed.ptr as *mut f32, e, self.eps);
                ops::ink_gemv_q8_0(self.stream, l.wk_ctx.buf.ptr as *const u8, self.ctx_normed.ptr as *const f32, self.ctx_k.ptr as *mut f32, kv_dim, e);
                ops::ink_gemv_q8_0(self.stream, l.wv_ctx.buf.ptr as *const u8, self.ctx_normed.ptr as *const f32, self.ctx_v.ptr as *mut f32, kv_dim, e);
                ops::ink_dspark_rope_norm_ctx(
                    self.stream,
                    self.ctx_k.ptr as *const f32,
                    self.ctx_v.ptr as *const f32,
                    l.k_ctx_norm.ptr as *const f32,
                    (l.k_cache.ptr as *mut u16).add(kv_off),
                    (l.v_cache.ptr as *mut u16).add(kv_off),
                    self.d_pos.ptr as *const i32,
                    l.max_ctx as i32,
                    self.n_head_kv as i32,
                    self.dh as i32,
                    self.freq_base,
                    if l.is_swa { 1 } else { 0 },
                );
            }
            let f_m = (self.fused_ctx.ptr as *const f32).add(3 * self.n_embd);
            ops::ink_rmsnorm(self.stream, f_m, self.dspark_norm.ptr as *const f32, (self.ctx_m.ptr as *mut f32).add(seq * self.n_embd), e, self.eps);
            rad_hip::hipStreamSynchronize(self.stream);
        }
        self.seq_ctx_len[seq] = pos + 1;
        if seq == 0 {
            self.ctx_len = pos + 1;
        }
        Ok(())
    }

    pub fn inject_target_feats_cfg(&mut self, feats_ptr: *const f32, pos: i32, use_softmax: bool) -> Result<()> {
        let e = self.n_embd as i32;
        let kv_dim = (self.n_head_kv * self.dh) as i32;
        self.d_pos.copy_from(&pos.to_le_bytes(), Some(self.stream)).map_err(he)?;
        unsafe {
            ops::ink_dspark_fuse(
                self.stream,
                feats_ptr,
                self.dspark_fuse.ptr as *const f32,
                self.fused_ctx.ptr as *mut f32,
                e,
                if use_softmax { 1 } else { 0 },
            );
            for (il, l) in self.layers.iter().enumerate() {
                let f_row = (self.fused_ctx.ptr as *const f32).add(il * self.n_embd);
                ops::ink_rmsnorm(self.stream, f_row, l.dflare_norm.ptr as *const f32, self.ctx_normed.ptr as *mut f32, e, self.eps);
                ops::ink_gemv_q8_0(self.stream, l.wk_ctx.buf.ptr as *const u8, self.ctx_normed.ptr as *const f32, self.ctx_k.ptr as *mut f32, kv_dim, e);
                ops::ink_gemv_q8_0(self.stream, l.wv_ctx.buf.ptr as *const u8, self.ctx_normed.ptr as *const f32, self.ctx_v.ptr as *mut f32, kv_dim, e);
                ops::ink_dspark_rope_norm_ctx(
                    self.stream,
                    self.ctx_k.ptr as *const f32,
                    self.ctx_v.ptr as *const f32,
                    l.k_ctx_norm.ptr as *const f32,
                    l.k_cache.ptr as *mut u16,
                    l.v_cache.ptr as *mut u16,
                    self.d_pos.ptr as *const i32,
                    l.max_ctx as i32,
                    self.n_head_kv as i32,
                    self.dh as i32,
                    self.freq_base,
                    if l.is_swa { 1 } else { 0 },
                );
            }
            let f_m = (self.fused_ctx.ptr as *const f32).add(3 * self.n_embd);
            ops::ink_rmsnorm(self.stream, f_m, self.dspark_norm.ptr as *const f32, self.ctx_m.ptr as *mut f32, e, self.eps);
            rad_hip::hipStreamSynchronize(self.stream);
        }
        self.ctx_len = pos + 1;
        Ok(())
    }

    pub fn inject_target_feats(&mut self, feats_ptr: *const f32, pos: i32) -> Result<()> {
        self.inject_target_feats_cfg(feats_ptr, pos, true)
    }

    unsafe fn gemm_w(&self, s: HipStream, w: &DevWeight, xq: *const core::ffi::c_void, y: *mut f32, n: i32, k: i32, b: i32) {
        match w.ty {
            WeightType::Q4_0 if b >= 12 => {
                ops::ink_gemm_q4_0_wmma(s, w.buf.ptr as *const u8, xq, y, n, k, b);
            }
            WeightType::Q4_0 => {
                ops::ink_gemm_q4_0_sudot4(s, w.buf.ptr as *const u8, xq, y, n, k, b);
            }
            _ => {
                ops::ink_gemm_q8_0_sudot4(s, w.buf.ptr as *const u8, xq, y, n, k, b);
            }
        }
    }

    unsafe fn enqueue_draft_multi(&self, s: HipStream, target: &InklingEngine, n_users: usize, k_len: usize) {
        let e = self.n_embd as i32;
        let q_dim = (self.n_head * self.dh) as i32;
        let kv_dim = (self.n_head_kv * self.dh) as i32;
        let nu = n_users as i32;
        let ki = k_len as i32;
        let tot_b = (n_users * k_len) as i32;
        let d_toks = self.d_draft_toks.ptr as *mut i32;

        ops::ink_dspark_scatter_toks(s, self.d_anchors.ptr as *const i32, d_toks, nu, ki, 0);
        ops::ink_embed_q8_0_norm_batch(
            s,
            target.tok_embd_ptr(),
            target.tok_norm_ptr(),
            self.d_anchors.ptr as *const i32,
            self.anchors_emb.ptr as *mut f32,
            e,
            nu,
        );
        ops::ink_dspark_init_multi(
            s,
            self.anchors_emb.ptr as *const f32,
            self.mask_embd.ptr as *const f32,
            self.x.ptr as *mut f32,
            e,
            nu,
            ki,
        );

        for l in &self.layers {
            ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, l.attn_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, e, tot_b, self.eps);
            ops::ink_quantize_q8_1(s, self.x_norm.ptr as *const f32, self.xq.ptr, e, tot_b);
            self.gemm_w(s, &l.wq, self.xq.ptr, self.q.ptr as *mut f32, q_dim, e, tot_b);
            self.gemm_w(s, &l.wk, self.xq.ptr, self.k.ptr as *mut f32, kv_dim, e, tot_b);
            self.gemm_w(s, &l.wv, self.xq.ptr, self.v.ptr as *mut f32, kv_dim, e, tot_b);
            ops::ink_dspark_rope_norm_qk_multi(
                s,
                self.q.ptr as *mut f32,
                self.k.ptr as *mut f32,
                l.q_norm.ptr as *const f32,
                l.k_norm.ptr as *const f32,
                self.d_pos.ptr as *const i32,
                nu,
                ki,
                self.n_head as i32,
                self.n_head_kv as i32,
                self.dh as i32,
                self.freq_base,
            );
            ops::ink_dspark_attn_multi(
                s,
                self.q.ptr as *const f32,
                self.k.ptr as *const f32,
                self.v.ptr as *const f32,
                l.k_cache.ptr as *const u16,
                l.v_cache.ptr as *const u16,
                self.d_seq_ids.ptr as *const i32,
                self.d_ctx_len.ptr as *const i32,
                self.attn_out.ptr as *mut f32,
                nu,
                ki,
                self.n_head as i32,
                self.n_head_kv as i32,
                self.dh as i32,
                self.swa_window as i32,
                if l.is_swa { 1 } else { 0 },
                l.max_ctx as i32,
            );
            ops::ink_quantize_q8_1(s, self.attn_out.ptr as *const f32, self.xq.ptr, q_dim, tot_b);
            self.gemm_w(s, &l.wo, self.xq.ptr, self.o_out.ptr as *mut f32, e, q_dim, tot_b);
            ops::ink_vec_add(s, self.x.ptr as *const f32, self.o_out.ptr as *const f32, self.x.ptr as *mut f32, tot_b * e);

            ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, l.ffn_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, e, tot_b, self.eps);
            ops::ink_quantize_q8_1(s, self.x_norm.ptr as *const f32, self.xq.ptr, e, tot_b);
            self.gemm_w(s, &l.ffn_gate, self.xq.ptr, self.gate.ptr as *mut f32, self.n_ff as i32, e, tot_b);
            self.gemm_w(s, &l.ffn_up, self.xq.ptr, self.up.ptr as *mut f32, self.n_ff as i32, e, tot_b);
            ops::ink_dspark_swiglu(s, self.gate.ptr as *const f32, self.up.ptr as *const f32, self.gate.ptr as *mut f32, tot_b * self.n_ff as i32);
            ops::ink_quantize_q8_1(s, self.gate.ptr as *const f32, self.xq.ptr, self.n_ff as i32, tot_b);
            self.gemm_w(s, &l.ffn_down, self.xq.ptr, self.ffn_out.ptr as *mut f32, e, self.n_ff as i32, tot_b);
            ops::ink_vec_add(s, self.x.ptr as *const f32, self.ffn_out.ptr as *const f32, self.x.ptr as *mut f32, tot_b * e);
        }

        ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, self.output_norm.ptr as *const f32, self.hidden.ptr as *mut f32, e, tot_b, self.eps);
        ops::ink_zero_f32(s, self.state.ptr as *mut f32, nu * self.rank as i32);

        for slot in 1..k_len {
            let si = slot as i32;
            ops::ink_dspark_markov_inp_multi(
                s,
                self.state.ptr as *const f32,
                self.markov_w1.ptr as *const u16,
                d_toks as *const i32,
                self.hidden.ptr as *const f32,
                self.ctx_m.ptr as *const f32,
                self.d_seq_ids.ptr as *const i32,
                self.markov_inp.ptr as *mut f32,
                e,
                self.rank as i32,
                nu,
                ki,
                si,
            );
            ops::ink_gemm_bf16(
                s,
                self.joint_proj.buf.ptr as *const u16,
                self.markov_inp.ptr as *const f32,
                self.joint.ptr as *mut f32,
                (3 * self.rank) as i32,
                (2 * self.rank + 2 * self.n_embd) as i32,
                nu,
            );
            ops::ink_dspark_markov_update_multi(
                s,
                self.joint.ptr as *const f32,
                self.joint_proj_b.ptr as *const f32,
                self.state.ptr as *mut f32,
                self.markov_out.ptr as *mut f32,
                self.rank as i32,
                nu,
            );
            ops::ink_gemm_bf16(
                s,
                self.hidden_out.buf.ptr as *const u16,
                self.markov_out.ptr as *const f32,
                self.delta_h.ptr as *mut f32,
                e,
                self.rank as i32,
                nu,
            );
            ops::ink_dspark_add_bias_multi(
                s,
                self.hidden.ptr as *const f32,
                self.delta_h.ptr as *const f32,
                self.hidden_out_b.ptr as *const f32,
                self.final_h.ptr as *mut f32,
                e,
                nu,
                ki,
                si,
            );
            ops::ink_quantize_q8_1(s, self.final_h.ptr as *const f32, self.xq.ptr, e, nu);
            if target.output_is_q4() {
                if n_users >= 12 {
                    ops::ink_gemm_q4_0_wmma(s, target.output_ptr(), self.xq.ptr, self.logits.ptr as *mut f32, target.cfg.vocab_padded as i32, e, nu);
                } else {
                    ops::ink_gemm_q4_0_sudot4(s, target.output_ptr(), self.xq.ptr, self.logits.ptr as *mut f32, target.cfg.vocab_padded as i32, e, nu);
                }
            } else {
                ops::ink_gemm_q8_0_sudot4(s, target.output_ptr(), self.xq.ptr, self.logits.ptr as *mut f32, target.cfg.vocab_padded as i32, e, nu);
            }
            ops::ink_softcap_argmax_batch(
                s,
                self.logits.ptr as *mut f32,
                target.cfg.final_softcap,
                target.cfg.vocab_padded as i32,
                target.cfg.vocab_unpadded as i32,
                self.d_slot_out.ptr as *mut i32,
                nu,
            );
            ops::ink_dspark_scatter_toks(s, self.d_slot_out.ptr as *const i32, d_toks, nu, ki, si);
        }
    }

    pub fn draft_multi(&mut self, seq_ids: &[i32], anchor_toks: &[i32], target: &InklingEngine, k_len: usize) -> Result<Vec<Vec<i32>>> {
        let n_users = seq_ids.len();
        assert_eq!(n_users, anchor_toks.len());
        let k_len = k_len.min(self.block_size);
        let ctx_lens: Vec<i32> = seq_ids.iter().map(|&s| self.seq_ctx_len[s as usize]).collect();
        let seq_bytes = unsafe { std::slice::from_raw_parts(seq_ids.as_ptr() as *const u8, n_users * 4) };
        let anc_bytes = unsafe { std::slice::from_raw_parts(anchor_toks.as_ptr() as *const u8, n_users * 4) };
        let ctx_bytes = unsafe { std::slice::from_raw_parts(ctx_lens.as_ptr() as *const u8, n_users * 4) };
        let key = (n_users, k_len);
        if !self.multi_draft_graphs.contains_key(&key) {
            let g = unsafe { CapturedGraph::capture(|s| self.enqueue_draft_multi(s, target, n_users, k_len)) }.map_err(he)?;
            self.multi_draft_graphs.insert(key, g);
        }
        let g = self.multi_draft_graphs.get(&key).unwrap();
        self.d_seq_ids.copy_from(seq_bytes, Some(g.stream)).map_err(he)?;
        self.d_anchors.copy_from(anc_bytes, Some(g.stream)).map_err(he)?;
        self.d_pos.copy_from(ctx_bytes, Some(g.stream)).map_err(he)?;
        self.d_ctx_len.copy_from(ctx_bytes, Some(g.stream)).map_err(he)?;
        g.launch().map_err(he)?;
        g.sync().map_err(he)?;
        let mut out_b = vec![0u8; n_users * k_len * 4];
        let rc = unsafe {
            rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.d_draft_toks.ptr, n_users * k_len * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        let flat: Vec<i32> = out_b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(flat.chunks_exact(k_len).map(|c| c.to_vec()).collect())
    }

    unsafe fn enqueue_draft_seq(&self, s: HipStream, target: &InklingEngine, seq: usize, k_len: usize, mode: i32, add_hidden: bool, norm_anchor: bool) {
        let e = self.n_embd as i32;
        let q_dim = (self.n_head * self.dh) as i32;
        let kv_dim = (self.n_head_kv * self.dh) as i32;
        let ki = k_len as i32;
        let d_toks = self.d_draft_toks.ptr as *mut i32;

        ops::ink_embed_q8_0_norm(
            s,
            target.tok_embd_ptr(),
            if norm_anchor { target.tok_norm_ptr() } else { std::ptr::null() },
            d_toks as *const i32,
            self.x.ptr as *mut f32,
            e,
        );
        for slot in 1..k_len {
            ops::ink_copy_f32(
                s,
                self.mask_embd.ptr as *const f32,
                (self.x.ptr as *mut f32).add(slot * self.n_embd),
                e,
            );
        }

        for l in &self.layers {
            ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, l.attn_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, e, ki, self.eps);
            ops::ink_quantize_q8_1(s, self.x_norm.ptr as *const f32, self.xq.ptr, e, ki);
            self.gemm_w(s, &l.wq, self.xq.ptr, self.q.ptr as *mut f32, q_dim, e, ki);
            self.gemm_w(s, &l.wk, self.xq.ptr, self.k.ptr as *mut f32, kv_dim, e, ki);
            self.gemm_w(s, &l.wv, self.xq.ptr, self.v.ptr as *mut f32, kv_dim, e, ki);
            ops::ink_dspark_rope_norm_qk(
                s,
                self.q.ptr as *mut f32,
                self.k.ptr as *mut f32,
                l.q_norm.ptr as *const f32,
                l.k_norm.ptr as *const f32,
                self.d_pos.ptr as *const i32,
                ki,
                self.n_head as i32,
                self.n_head_kv as i32,
                self.dh as i32,
                self.freq_base,
            );
            let kv_off = seq * self.n_head_kv * l.max_ctx * self.dh;
            ops::ink_dspark_attn(
                s,
                self.q.ptr as *const f32,
                self.k.ptr as *const f32,
                self.v.ptr as *const f32,
                (l.k_cache.ptr as *const u16).add(kv_off),
                (l.v_cache.ptr as *const u16).add(kv_off),
                self.d_ctx_len.ptr as *const i32,
                self.attn_out.ptr as *mut f32,
                ki,
                self.n_head as i32,
                self.n_head_kv as i32,
                self.dh as i32,
                self.swa_window as i32,
                if l.is_swa { 1 } else { 0 },
                l.max_ctx as i32,
            );
            ops::ink_quantize_q8_1(s, self.attn_out.ptr as *const f32, self.xq.ptr, q_dim, ki);
            self.gemm_w(s, &l.wo, self.xq.ptr, self.o_out.ptr as *mut f32, e, q_dim, ki);
            ops::ink_vec_add(s, self.x.ptr as *const f32, self.o_out.ptr as *const f32, self.x.ptr as *mut f32, ki * e);

            ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, l.ffn_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, e, ki, self.eps);
            ops::ink_quantize_q8_1(s, self.x_norm.ptr as *const f32, self.xq.ptr, e, ki);
            self.gemm_w(s, &l.ffn_gate, self.xq.ptr, self.gate.ptr as *mut f32, self.n_ff as i32, e, ki);
            self.gemm_w(s, &l.ffn_up, self.xq.ptr, self.up.ptr as *mut f32, self.n_ff as i32, e, ki);
            ops::ink_dspark_swiglu(s, self.gate.ptr as *const f32, self.up.ptr as *const f32, self.gate.ptr as *mut f32, ki * self.n_ff as i32);
            ops::ink_quantize_q8_1(s, self.gate.ptr as *const f32, self.xq.ptr, self.n_ff as i32, ki);
            self.gemm_w(s, &l.ffn_down, self.xq.ptr, self.ffn_out.ptr as *mut f32, e, self.n_ff as i32, ki);
            ops::ink_vec_add(s, self.x.ptr as *const f32, self.ffn_out.ptr as *const f32, self.x.ptr as *mut f32, ki * e);
        }

        ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, self.output_norm.ptr as *const f32, self.hidden.ptr as *mut f32, e, ki, self.eps);
        ops::ink_zero_f32(s, self.state.ptr as *mut f32, self.rank as i32);

        for slot in 1..k_len {
            let hi = (self.hidden.ptr as *const f32).add(slot * self.n_embd);
            ops::ink_dspark_markov_inp(
                s,
                self.state.ptr as *const f32,
                self.markov_w1.ptr as *const u16,
                d_toks.add(slot - 1) as *const i32,
                hi,
                (self.ctx_m.ptr as *const f32).add(seq * self.n_embd),
                self.markov_inp.ptr as *mut f32,
                e,
                self.rank as i32,
                mode,
            );
            ops::ink_gemv_bf16(
                s,
                self.joint_proj.buf.ptr as *const u16,
                self.markov_inp.ptr as *const f32,
                self.joint.ptr as *mut f32,
                (3 * self.rank) as i32,
                (2 * self.rank + 2 * self.n_embd) as i32,
            );
            ops::ink_dspark_markov_update(
                s,
                self.joint.ptr as *const f32,
                self.joint_proj_b.ptr as *const f32,
                self.state.ptr as *mut f32,
                self.markov_out.ptr as *mut f32,
                self.rank as i32,
            );
            ops::ink_gemv_bf16(
                s,
                self.hidden_out.buf.ptr as *const u16,
                self.markov_out.ptr as *const f32,
                self.delta_h.ptr as *mut f32,
                e,
                self.rank as i32,
            );
            if add_hidden {
                ops::ink_dspark_add_bias(
                    s,
                    hi,
                    self.delta_h.ptr as *const f32,
                    self.hidden_out_b.ptr as *const f32,
                    self.final_h.ptr as *mut f32,
                    e,
                );
            } else {
                ops::ink_vec_add(
                    s,
                    self.delta_h.ptr as *const f32,
                    self.hidden_out_b.ptr as *const f32,
                    self.final_h.ptr as *mut f32,
                    e,
                );
            }
            ops::ink_quantize_q8_1(s, self.final_h.ptr as *const f32, self.xq.ptr, e, 1);
            if target.output_is_q4() {
                ops::ink_gemm_q4_0_sudot4(
                    s,
                    target.output_ptr(),
                    self.xq.ptr,
                    self.logits.ptr as *mut f32,
                    target.cfg.vocab_padded as i32,
                    e,
                    1,
                );
            } else {
                ops::ink_gemm_q8_0_sudot4(
                    s,
                    target.output_ptr(),
                    self.xq.ptr,
                    self.logits.ptr as *mut f32,
                    target.cfg.vocab_padded as i32,
                    e,
                    1,
                );
            }
            ops::ink_softcap_argmax(
                s,
                self.logits.ptr as *mut f32,
                target.cfg.final_softcap,
                target.cfg.vocab_unpadded as i32,
                d_toks.add(slot),
            );
        }
    }

    pub fn capture_draft_graph(&mut self, target: &InklingEngine, k_len: usize) -> Result<()> {
        let k_len = k_len.min(self.block_size);
        let g = unsafe { CapturedGraph::capture(|s| self.enqueue_draft_seq(s, target, 0, k_len, 0, true, true)) }.map_err(he)?;
        self.draft_graph = Some((k_len, g));
        Ok(())
    }

    pub fn draft_seq(&mut self, seq: usize, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> Result<Vec<i32>> {
        let k_len = k_len.min(self.block_size);
        let ctx_len = self.seq_ctx_len[seq];
        let pos0 = ctx_len;
        let key = (seq, k_len);
        if !self.seq_draft_graphs.contains_key(&key) {
            let g = unsafe { CapturedGraph::capture(|s| self.enqueue_draft_seq(s, target, seq, k_len, 0, true, true)) }.map_err(he)?;
            self.seq_draft_graphs.insert(key, g);
        }
        let g = self.seq_draft_graphs.get(&key).unwrap();
        self.d_draft_toks.copy_from(&anchor_tok.to_le_bytes(), Some(g.stream)).map_err(he)?;
        self.d_pos.copy_from(&pos0.to_le_bytes(), Some(g.stream)).map_err(he)?;
        self.d_ctx_len.copy_from(&ctx_len.to_le_bytes(), Some(g.stream)).map_err(he)?;
        g.launch().map_err(he)?;
        g.sync().map_err(he)?;
        let mut out_b = vec![0u8; k_len * 4];
        let rc = unsafe {
            rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.d_draft_toks.ptr, k_len * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        Ok(out_b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn draft_with_mode(&mut self, anchor_tok: i32, target: &InklingEngine, k_len: usize, mode: i32, add_hidden: bool, norm_anchor: bool) -> Result<Vec<i32>> {
        let k_len = k_len.min(self.block_size);
        let pos0 = self.ctx_len;
        let use_graph = mode == 0 && add_hidden && norm_anchor
            && self.draft_graph.as_ref().map(|(gk, _)| *gk == k_len).unwrap_or(false);
        if use_graph {
            let (_, ref g) = self.draft_graph.as_ref().unwrap();
            self.d_draft_toks.copy_from(&anchor_tok.to_le_bytes(), Some(g.stream)).map_err(he)?;
            self.d_pos.copy_from(&pos0.to_le_bytes(), Some(g.stream)).map_err(he)?;
            self.d_ctx_len.copy_from(&self.ctx_len.to_le_bytes(), Some(g.stream)).map_err(he)?;
            g.launch().map_err(he)?;
            g.sync().map_err(he)?;
        } else {
            self.d_draft_toks.copy_from(&anchor_tok.to_le_bytes(), Some(self.stream)).map_err(he)?;
            self.d_pos.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            self.d_ctx_len.copy_from(&self.ctx_len.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe {
                self.enqueue_draft_seq(self.stream, target, 0, k_len, mode, add_hidden, norm_anchor);
                rad_hip::hipStreamSynchronize(self.stream);
            }
        }
        let mut out_b = vec![0u8; k_len * 4];
        let rc = unsafe {
            rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.d_draft_toks.ptr, k_len * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        Ok(out_b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn draft(&mut self, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> Result<Vec<i32>> {
        self.draft_with_mode(anchor_tok, target, k_len, 0, true, true)
    }
}
