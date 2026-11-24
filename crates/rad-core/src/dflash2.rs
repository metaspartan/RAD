use crate::model::{DevWeight, InklingEngine, WeightType};
use crate::ops;
use crate::safetensors::SafetensorsFile;
use anyhow::{anyhow, Result};
use rad_hip::{CapturedGraph, DevBuf, HipError, HipStream};
use std::collections::HashMap;
use std::path::Path;

fn he(e: HipError) -> anyhow::Error {
    anyhow!("hip error rc={e}")
}

fn load_st_bf16_raw(st: &SafetensorsFile, name: &str) -> Result<DevBuf> {
    let (_, bytes) = st.tensor_bytes(name)?;
    let mut b = DevBuf::new(bytes.len()).map_err(he)?;
    b.copy_from(bytes, None).map_err(he)?;
    Ok(b)
}

fn load_st_f32(st: &SafetensorsFile, name: &str) -> Result<DevBuf> {
    let (t, bytes) = st.tensor_bytes(name)?;
    let n: usize = t.shape.iter().product();
    let mut src = DevBuf::new(bytes.len()).map_err(he)?;
    src.copy_from(bytes, None).map_err(he)?;
    let dst = DevBuf::new(n * 4).map_err(he)?;
    unsafe {
        ops::ink_bf16_to_f32(std::ptr::null_mut(), src.ptr as *const u16, dst.ptr as *mut f32, n);
        rad_hip::hipStreamSynchronize(std::ptr::null_mut());
    }
    drop(src);
    Ok(dst)
}

fn load_st_linear_q8(st: &SafetensorsFile, name: &str) -> Result<DevWeight> {
    let (t, bytes) = st.tensor_bytes(name)?;
    if t.shape.len() != 2 {
        return Err(anyhow!("expected 2D tensor for {name}, got {:?}", t.shape));
    }
    let n = t.shape[0];
    let k = t.shape[1];
    let nblk = k / 32;
    let mut src = DevBuf::new(bytes.len()).map_err(he)?;
    src.copy_from(bytes, None).map_err(he)?;
    let dst = DevBuf::new(n * nblk * 34).map_err(he)?;
    unsafe {
        ops::ink_bf16_to_q8_0(std::ptr::null_mut(), src.ptr as *const u16, dst.ptr as *mut u8, n, nblk as i32);
        rad_hip::hipStreamSynchronize(std::ptr::null_mut());
    }
    drop(src);
    Ok(DevWeight { buf: dst, ty: WeightType::Q8_0, n: n as i32, k: k as i32 })
}

fn load_st_linear(st: &SafetensorsFile, name: &str, to_q4: bool) -> Result<DevWeight> {
    let (t, bytes) = st.tensor_bytes(name)?;
    if t.shape.len() != 2 {
        return Err(anyhow!("expected 2D tensor for {name}, got {:?}", t.shape));
    }
    let n = t.shape[0];
    let k = t.shape[1];
    let mut src = DevBuf::new(bytes.len()).map_err(he)?;
    src.copy_from(bytes, None).map_err(he)?;
    if !to_q4 {
        return Ok(DevWeight { buf: src, ty: WeightType::Bf16, n: n as i32, k: k as i32 });
    }
    let nblk = k / 32;
    let rbytes = nblk * 16 + ((nblk * 2 + 15) & !15);
    let dst = DevBuf::new(n * rbytes).map_err(he)?;
    unsafe {
        ops::ink_bf16_to_q4_0(std::ptr::null_mut(), src.ptr as *const u16, dst.ptr as *mut u8, n, nblk as i32);
        rad_hip::hipStreamSynchronize(std::ptr::null_mut());
    }
    drop(src);
    Ok(DevWeight { buf: dst, ty: WeightType::Q4_0, n: n as i32, k: k as i32 })
}

struct DFlash2Layer {
    max_ctx: usize,
    input_norm: DevBuf,
    attn_conv_base: DevBuf,
    attn_conv_proj: DevWeight,
    wq: DevWeight,
    wk: DevWeight,
    wv: DevWeight,
    wo: DevWeight,
    q_norm: DevBuf,
    k_norm: DevBuf,
    post_attn_norm: DevBuf,
    mlp_conv_base: DevBuf,
    mlp_conv_proj: DevWeight,
    gate_proj: DevWeight,
    up_proj: DevWeight,
    down_proj: DevWeight,
    k_cache: DevBuf,
    v_cache: DevBuf,
}

pub struct DFlash2Drafter {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub dh: usize,
    pub n_ff: usize,
    pub block_size: usize,
    pub swa_window: usize,
    pub conv_group_size: usize,
    pub mask_token_id: i32,
    pub selector_rank: usize,
    pub final_softcap: f32,
    pub freq_base: f32,
    pub eps: f32,
    pub max_seqs: usize,
    pub ctx_len: i32,
    pub seq_ctx_len: Vec<i32>,
    fc: DevWeight,
    hidden_norm: DevBuf,
    output_norm: DevBuf,
    sel_hidden_proj: DevWeight,
    sel_pred_cb: DevBuf,
    sel_succ_cb: DevBuf,
    layers: Vec<DFlash2Layer>,
    ctx_proj: DevBuf,
    ctx_normed: DevBuf,
    ctx_k: DevBuf,
    ctx_v: DevBuf,
    inj_feats: DevBuf,
    inj_proj: DevBuf,
    inj_normed: DevBuf,
    inj_k: DevBuf,
    inj_v: DevBuf,
    inj_xq: DevBuf,
    d_mask_tok: DevBuf,
    mask_embd: DevBuf,
    x: DevBuf,
    x_norm: DevBuf,
    x_conv: DevBuf,
    dyn_buf: DevBuf,
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
    h_pred: DevBuf,
    h_proj: DevBuf,
    logits: DevBuf,
    cand_ids: DevBuf,
    cand_vals: DevBuf,
    d_draft_toks: DevBuf,
    d_pos: DevBuf,
    d_ctx_len: DevBuf,
    d_seq_ids: DevBuf,
    d_anchors: DevBuf,
    anchors_emb: DevBuf,
    stream: HipStream,
    draft_graph: Option<(usize, CapturedGraph)>,
    seq_draft_graphs: HashMap<(usize, usize), CapturedGraph>,
    multi_draft_graphs: HashMap<(usize, usize), CapturedGraph>,
}

unsafe impl Send for DFlash2Drafter {}

impl DFlash2Drafter {
    pub fn load(path: &Path, max_ctx: usize) -> Result<Self> {
        Self::load_multi(path, max_ctx, 1)
    }

    pub fn load_multi(path: &Path, max_ctx: usize, max_seqs: usize) -> Result<Self> {
        let max_seqs = max_seqs.max(1);
        let (st_path, cfg_path) = if path.is_dir() {
            (path.join("model.safetensors"), path.join("config.json"))
        } else {
            (path.to_path_buf(), path.parent().unwrap_or_else(|| Path::new(".")).join("config.json"))
        };
        let st = SafetensorsFile::open(&st_path)?;
        let cfg_val: serde_json::Value = if cfg_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?
        } else {
            serde_json::json!({})
        };
        let df = &cfg_val["dflash_config"];
        let q4_draft = std::env::var("RAD_DFLASH2_Q4").map(|v| v != "0").unwrap_or(true);
        let n_layers = cfg_val["num_hidden_layers"].as_u64().unwrap_or(5) as usize;
        let n_embd = cfg_val["hidden_size"].as_u64().unwrap_or(2304) as usize;
        let n_head = cfg_val["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let n_head_kv = cfg_val["num_key_value_heads"].as_u64().unwrap_or(8) as usize;
        let dh = cfg_val["head_dim"].as_u64().unwrap_or(128) as usize;
        let n_ff = cfg_val["intermediate_size"].as_u64().unwrap_or(6912) as usize;
        let block_size = df["block_size"].as_u64().unwrap_or(8) as usize;
        let swa_window = cfg_val["sliding_window"].as_u64().unwrap_or(2048) as usize;
        let conv_group_size = df["conv_group_size"].as_u64().unwrap_or(16) as usize;
        let mask_token_id = df["mask_token_id"].as_i64().unwrap_or(200056) as i32;
        let selector_rank = df["selector_rank"].as_u64().unwrap_or(256) as usize;
        let final_softcap = df["final_logit_softcapping"].as_f64().unwrap_or(30.0) as f32;
        let freq_base = cfg_val["rope_parameters"]["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
        let eps = cfg_val["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;

        let l_ctx = (swa_window + 16).min(max_ctx);
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            let kv_bytes = max_seqs * n_head_kv * l_ctx * dh * 2;
            let mut k_cache = DevBuf::new(kv_bytes).map_err(he)?;
            let mut v_cache = DevBuf::new(kv_bytes).map_err(he)?;
            k_cache.zero().map_err(he)?;
            v_cache.zero().map_err(he)?;
            layers.push(DFlash2Layer {
                max_ctx: l_ctx,
                input_norm: load_st_f32(&st, &p("input_layernorm.weight"))?,
                attn_conv_base: load_st_f32(&st, &p("attention_conv.base_kernel"))?,
                attn_conv_proj: load_st_linear_q8(&st, &p("attention_conv.kernel_projection.weight"))?,
                wq: load_st_linear(&st, &p("self_attn.q_proj.weight"), q4_draft)?,
                wk: load_st_linear(&st, &p("self_attn.k_proj.weight"), q4_draft)?,
                wv: load_st_linear(&st, &p("self_attn.v_proj.weight"), q4_draft)?,
                wo: load_st_linear(&st, &p("self_attn.o_proj.weight"), q4_draft)?,
                q_norm: load_st_f32(&st, &p("self_attn.q_norm.weight"))?,
                k_norm: load_st_f32(&st, &p("self_attn.k_norm.weight"))?,
                post_attn_norm: load_st_f32(&st, &p("post_attention_layernorm.weight"))?,
                mlp_conv_base: load_st_f32(&st, &p("mlp_conv.base_kernel"))?,
                mlp_conv_proj: load_st_linear_q8(&st, &p("mlp_conv.kernel_projection.weight"))?,
                gate_proj: load_st_linear(&st, &p("mlp.gate_proj.weight"), q4_draft)?,
                up_proj: load_st_linear(&st, &p("mlp.up_proj.weight"), q4_draft)?,
                down_proj: load_st_linear(&st, &p("mlp.down_proj.weight"), q4_draft)?,
                k_cache,
                v_cache,
            });
        }

        let kmax = block_size;
        let q_dim = n_head * dh;
        let kv_dim = n_head_kv * dh;
        let n_groups = n_embd / conv_group_size;
        let dyn_dim = 4 * n_groups;
        let max_k_dim = n_ff.max(q_dim).max(4 * n_embd);

        let mut d_mask_tok = DevBuf::new(4).map_err(he)?;
        d_mask_tok.copy_from(&mask_token_id.to_le_bytes(), None).map_err(he)?;

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
            block_size,
            swa_window,
            conv_group_size,
            mask_token_id,
            selector_rank,
            final_softcap,
            freq_base,
            eps,
            max_seqs,
            ctx_len: 0,
            seq_ctx_len: vec![0; max_seqs],
            fc: load_st_linear(&st, "fc.weight", q4_draft)?,
            hidden_norm: load_st_f32(&st, "hidden_norm.weight")?,
            output_norm: load_st_f32(&st, "norm.weight")?,
            sel_hidden_proj: load_st_linear_q8(&st, "candidate_selector.hidden_projection.weight")?,
            sel_pred_cb: load_st_bf16_raw(&st, "candidate_selector.predecessor_codebook")?,
            sel_succ_cb: load_st_bf16_raw(&st, "candidate_selector.successor_codebook")?,
            layers,
            ctx_proj: DevBuf::new(n_embd * 4).map_err(he)?,
            ctx_normed: DevBuf::new(n_embd * 4).map_err(he)?,
            ctx_k: DevBuf::new(kv_dim * 4).map_err(he)?,
            ctx_v: DevBuf::new(kv_dim * 4).map_err(he)?,
            inj_feats: DevBuf::new(256 * 4 * n_embd * 4).map_err(he)?,
            inj_proj: DevBuf::new(256 * n_embd * 4).map_err(he)?,
            inj_normed: DevBuf::new(256 * n_embd * 4).map_err(he)?,
            inj_k: DevBuf::new(256 * kv_dim * 4).map_err(he)?,
            inj_v: DevBuf::new(256 * kv_dim * 4).map_err(he)?,
            inj_xq: DevBuf::new(256 * (4 * n_embd / 32) * 40).map_err(he)?,
            d_mask_tok,
            mask_embd: DevBuf::new(n_embd * 4).map_err(he)?,
            x: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            x_norm: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            x_conv: DevBuf::new(max_seqs * kmax * n_embd * 4).map_err(he)?,
            dyn_buf: DevBuf::new(max_seqs * kmax * dyn_dim * 4).map_err(he)?,
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
            h_pred: DevBuf::new(max_seqs * (kmax - 1) * n_embd * 4).map_err(he)?,
            h_proj: DevBuf::new(max_seqs * (kmax - 1) * selector_rank * 4).map_err(he)?,
            logits: DevBuf::new(max_seqs * (kmax - 1) * 201024 * 4).map_err(he)?,
            cand_ids: DevBuf::new(max_seqs * (kmax - 1) * 16 * 4).map_err(he)?,
            cand_vals: DevBuf::new(max_seqs * (kmax - 1) * 16 * 4).map_err(he)?,
            d_draft_toks: DevBuf::new(max_seqs * kmax * 4).map_err(he)?,
            d_pos: DevBuf::new(max_seqs * 4).map_err(he)?,
            d_ctx_len: DevBuf::new(max_seqs * 4).map_err(he)?,
            d_seq_ids: DevBuf::new(max_seqs * 4).map_err(he)?,
            d_anchors: DevBuf::new(max_seqs * 4).map_err(he)?,
            anchors_emb: DevBuf::new(max_seqs * n_embd * 4).map_err(he)?,
            stream,
            draft_graph: None,
            seq_draft_graphs: HashMap::new(),
            multi_draft_graphs: HashMap::new(),
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

    unsafe fn gemm_w(&self, s: HipStream, w: &DevWeight, x_f32: *const f32, xq: *const core::ffi::c_void, y: *mut f32, n: i32, k: i32, b: i32) {
        match w.ty {
            WeightType::Q4_0 if b >= 12 => {
                ops::ink_gemm_q4_0_wmma(s, w.buf.ptr as *const u8, xq, y, n, k, b);
            }
            WeightType::Q4_0 => {
                ops::ink_gemm_q4_0_sudot4(s, w.buf.ptr as *const u8, xq, y, n, k, b);
            }
            WeightType::Bf16 => {
                ops::ink_gemm_bf16(s, w.buf.ptr as *const u16, x_f32, y, n, k, b);
            }
            _ => {
                ops::ink_gemm_q8_0_sudot4(s, w.buf.ptr as *const u8, xq, y, n, k, b);
            }
        }
    }

    pub fn inject_target_feats_seq(&mut self, seq: usize, feats_ptr: *const f32, pos: i32) -> Result<()> {
        self.inject_target_feats_seq_async(seq, feats_ptr, pos)?;
        unsafe { rad_hip::hipStreamSynchronize(self.stream); }
        Ok(())
    }

    pub fn inject_target_feats_seq_async(&mut self, seq: usize, feats_ptr: *const f32, pos: i32) -> Result<()> {
        let e = self.n_embd as i32;
        let kv_dim = (self.n_head_kv * self.dh) as i32;
        self.d_pos.copy_from(&pos.to_le_bytes(), Some(self.stream)).map_err(he)?;
        unsafe {
            if self.fc.ty == WeightType::Q4_0 {
                ops::ink_quantize_q8_1(self.stream, feats_ptr, self.xq.ptr, 4 * e, 1);
                ops::ink_gemm_q4_0_sudot4(self.stream, self.fc.buf.ptr as *const u8, self.xq.ptr, self.ctx_proj.ptr as *mut f32, e, 4 * e, 1);
            } else {
                ops::ink_gemv_bf16(self.stream, self.fc.buf.ptr as *const u16, feats_ptr, self.ctx_proj.ptr as *mut f32, e, 4 * e);
            }
            ops::ink_rmsnorm(self.stream, self.ctx_proj.ptr as *const f32, self.hidden_norm.ptr as *const f32, self.ctx_normed.ptr as *mut f32, e, self.eps);
            if self.layers[0].wk.ty == WeightType::Q4_0 {
                ops::ink_quantize_q8_1(self.stream, self.ctx_normed.ptr as *const f32, self.xq.ptr, e, 1);
            }
            for l in &self.layers {
                let kv_off = seq * self.n_head_kv * l.max_ctx * self.dh;
                self.gemm_w(self.stream, &l.wk, self.ctx_normed.ptr as *const f32, self.xq.ptr, self.ctx_k.ptr as *mut f32, kv_dim, e, 1);
                self.gemm_w(self.stream, &l.wv, self.ctx_normed.ptr as *const f32, self.xq.ptr, self.ctx_v.ptr as *mut f32, kv_dim, e, 1);
                ops::ink_dspark_rope_norm_ctx(
                    self.stream,
                    self.ctx_k.ptr as *const f32,
                    self.ctx_v.ptr as *const f32,
                    l.k_norm.ptr as *const f32,
                    (l.k_cache.ptr as *mut u16).add(kv_off),
                    (l.v_cache.ptr as *mut u16).add(kv_off),
                    self.d_pos.ptr as *const i32,
                    l.max_ctx as i32,
                    self.n_head_kv as i32,
                    self.dh as i32,
                    self.freq_base,
                    1,
                );
            }
        }
        self.seq_ctx_len[seq] = pos + 1;
        if seq == 0 {
            self.ctx_len = pos + 1;
        }
        Ok(())
    }

    pub fn stream(&self) -> HipStream {
        self.stream
    }

    pub fn sync(&self) {
        unsafe { rad_hip::hipStreamSynchronize(self.stream); }
    }

    pub fn advance_ctx_len_seq(&mut self, seq: usize, new_ctx_len: i32) {
        self.seq_ctx_len[seq] = new_ctx_len;
        if seq == 0 {
            self.ctx_len = new_ctx_len;
        }
    }

    pub fn inject_target_feats_batch_seq_async(&mut self, seq: usize, bsc_feats: *const f32, max_batch: usize, start_tok: usize, pos0: i32, n_tok: usize) -> Result<()> {
        if n_tok == 0 {
            return Ok(());
        }
        let e = self.n_embd as i32;
        let kv_dim = (self.n_head_kv * self.dh) as i32;
        let mut off = 0usize;
        while off < n_tok {
            let chunk = (n_tok - off).min(256);
            let ci = chunk as i32;
            unsafe {
                ops::ink_gather_dspark_feats_batch(
                    self.stream,
                    bsc_feats,
                    self.inj_feats.ptr as *mut f32,
                    (start_tok + off) as i32,
                    max_batch as i32,
                    e,
                    ci,
                );
                if self.fc.ty == WeightType::Q4_0 {
                    ops::ink_quantize_q8_1(self.stream, self.inj_feats.ptr as *const f32, self.inj_xq.ptr, 4 * e, ci);
                }
                self.gemm_w(self.stream, &self.fc, self.inj_feats.ptr as *const f32, self.inj_xq.ptr, self.inj_proj.ptr as *mut f32, e, 4 * e, ci);
                ops::ink_rmsnorm_batch(self.stream, self.inj_proj.ptr as *const f32, self.hidden_norm.ptr as *const f32, self.inj_normed.ptr as *mut f32, e, ci, self.eps);
                if self.layers[0].wk.ty == WeightType::Q4_0 {
                    ops::ink_quantize_q8_1(self.stream, self.inj_normed.ptr as *const f32, self.inj_xq.ptr, e, ci);
                }
                for l in &self.layers {
                    let kv_off = seq * self.n_head_kv * l.max_ctx * self.dh;
                    self.gemm_w(self.stream, &l.wk, self.inj_normed.ptr as *const f32, self.inj_xq.ptr, self.inj_k.ptr as *mut f32, kv_dim, e, ci);
                    self.gemm_w(self.stream, &l.wv, self.inj_normed.ptr as *const f32, self.inj_xq.ptr, self.inj_v.ptr as *mut f32, kv_dim, e, ci);
                    ops::ink_dflash2_rope_norm_ctx_batch(
                        self.stream,
                        self.inj_k.ptr as *const f32,
                        self.inj_v.ptr as *const f32,
                        l.k_norm.ptr as *const f32,
                        (l.k_cache.ptr as *mut u16).add(kv_off),
                        (l.v_cache.ptr as *mut u16).add(kv_off),
                        pos0 + off as i32,
                        l.max_ctx as i32,
                        self.n_head_kv as i32,
                        self.dh as i32,
                        self.freq_base,
                        1,
                        ci,
                    );
                }
            }
            off += chunk;
        }
        self.seq_ctx_len[seq] = pos0 + n_tok as i32;
        if seq == 0 {
            self.ctx_len = pos0 + n_tok as i32;
        }
        Ok(())
    }

    pub fn inject_target_feats_batch_seq(&mut self, seq: usize, bsc_feats: *const f32, max_batch: usize, start_tok: usize, pos0: i32, n_tok: usize) -> Result<()> {
        self.inject_target_feats_batch_seq_async(seq, bsc_feats, max_batch, start_tok, pos0, n_tok)?;
        unsafe { rad_hip::hipStreamSynchronize(self.stream); }
        Ok(())
    }

    pub fn inject_target_feats(&mut self, feats_ptr: *const f32, pos: i32) -> Result<()> {
        self.inject_target_feats_seq(0, feats_ptr, pos)
    }

    unsafe fn enqueue_backbone(&self, s: HipStream, n_users: usize, k_len: usize, single_seq: Option<usize>) {
        let e = self.n_embd as i32;
        let q_dim = (self.n_head * self.dh) as i32;
        let kv_dim = (self.n_head_kv * self.dh) as i32;
        let dyn_dim = (4 * (self.n_embd / self.conv_group_size)) as i32;
        let ki = k_len as i32;
        let nu = n_users as i32;
        let tot_b = (n_users * k_len) as i32;
        let gs = self.conv_group_size as i32;

        for l in &self.layers {
            if l.attn_conv_proj.ty != WeightType::Bf16 {
                ops::ink_dflash2_rmsnorm_q8(s, self.x.ptr as *const f32, l.input_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, self.xq.ptr, e, tot_b, self.eps);
            } else {
                ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, l.input_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, e, tot_b, self.eps);
            }
            self.gemm_w(s, &l.attn_conv_proj, self.x_norm.ptr as *const f32, self.xq.ptr, self.dyn_buf.ptr as *mut f32, dyn_dim, e, tot_b);
            if l.wq.ty == WeightType::Q4_0 {
                ops::ink_dflash2_conv_q8(s, self.x_norm.ptr as *const f32, self.dyn_buf.ptr as *const f32, l.attn_conv_base.ptr as *const f32, self.xq.ptr, e, ki, gs, 0, tot_b);
            } else {
                ops::ink_dflash2_conv(s, self.x_norm.ptr as *const f32, self.dyn_buf.ptr as *const f32, l.attn_conv_base.ptr as *const f32, self.x_conv.ptr as *mut f32, e, ki, gs, 0, tot_b);
            }
            if l.wq.ty == WeightType::Q4_0 && (tot_b == 4 || tot_b == 6 || tot_b == 8) {
                ops::ink_dflash2_qkv_q4(s, l.wq.buf.ptr as *const u8, l.wk.buf.ptr as *const u8, l.wv.buf.ptr as *const u8, self.xq.ptr, self.q.ptr as *mut f32, self.k.ptr as *mut f32, self.v.ptr as *mut f32, e, q_dim, kv_dim, tot_b);
            } else {
                self.gemm_w(s, &l.wq, self.x_conv.ptr as *const f32, self.xq.ptr, self.q.ptr as *mut f32, q_dim, e, tot_b);
                self.gemm_w(s, &l.wk, self.x_conv.ptr as *const f32, self.xq.ptr, self.k.ptr as *mut f32, kv_dim, e, tot_b);
                self.gemm_w(s, &l.wv, self.x_conv.ptr as *const f32, self.xq.ptr, self.v.ptr as *mut f32, kv_dim, e, tot_b);
            }

            if let Some(seq) = single_seq {
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
                if l.wo.ty == WeightType::Q4_0 && self.dh == 128 && k_len <= 8 && self.n_head == 4 * self.n_head_kv && self.n_head <= 32 {
                    ops::ink_dspark_attn_wmma_q8(
                        s,
                        self.q.ptr as *const f32,
                        self.k.ptr as *const f32,
                        self.v.ptr as *const f32,
                        (l.k_cache.ptr as *const u16).add(kv_off),
                        (l.v_cache.ptr as *const u16).add(kv_off),
                        self.d_ctx_len.ptr as *const i32,
                        self.xq.ptr,
                        ki,
                        self.n_head as i32,
                        self.n_head_kv as i32,
                        self.dh as i32,
                        self.swa_window as i32,
                        1,
                        l.max_ctx as i32,
                    );
                } else if l.wo.ty == WeightType::Q4_0 && self.dh == 128 && k_len <= 8 && self.n_head <= 32 {
                    ops::ink_dspark_attn_q8(
                        s,
                        self.q.ptr as *const f32,
                        self.k.ptr as *const f32,
                        self.v.ptr as *const f32,
                        (l.k_cache.ptr as *const u16).add(kv_off),
                        (l.v_cache.ptr as *const u16).add(kv_off),
                        self.d_ctx_len.ptr as *const i32,
                        self.xq.ptr,
                        ki,
                        self.n_head as i32,
                        self.n_head_kv as i32,
                        self.dh as i32,
                        self.swa_window as i32,
                        1,
                        l.max_ctx as i32,
                    );
                } else {
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
                        1,
                        l.max_ctx as i32,
                    );
                    if l.wo.ty == WeightType::Q4_0 {
                        ops::ink_quantize_q8_1(s, self.attn_out.ptr as *const f32, self.xq.ptr, q_dim, tot_b);
                    }
                }
            } else {
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
                    1,
                    l.max_ctx as i32,
                );
                if l.wo.ty == WeightType::Q4_0 {
                    ops::ink_quantize_q8_1(s, self.attn_out.ptr as *const f32, self.xq.ptr, q_dim, tot_b);
                }
            }

            if l.wo.ty == WeightType::Q4_0 && n_users == 1 && (tot_b == 4 || tot_b == 6 || tot_b == 8) {
                ops::ink_dflash2_proj_conv_add_q4(s, l.wo.buf.ptr as *const u8, self.xq.ptr, self.dyn_buf.ptr as *const f32, l.attn_conv_base.ptr as *const f32, self.x.ptr as *mut f32, e, q_dim, gs, 1, tot_b);
            } else {
                self.gemm_w(s, &l.wo, self.attn_out.ptr as *const f32, self.xq.ptr, self.o_out.ptr as *mut f32, e, q_dim, tot_b);
                ops::ink_dflash2_conv_add(s, self.o_out.ptr as *const f32, self.dyn_buf.ptr as *const f32, l.attn_conv_base.ptr as *const f32, self.x.ptr as *mut f32, e, ki, gs, 1, tot_b);
            }

            if l.mlp_conv_proj.ty != WeightType::Bf16 {
                ops::ink_dflash2_rmsnorm_q8(s, self.x.ptr as *const f32, l.post_attn_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, self.xq.ptr, e, tot_b, self.eps);
            } else {
                ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, l.post_attn_norm.ptr as *const f32, self.x_norm.ptr as *mut f32, e, tot_b, self.eps);
            }
            self.gemm_w(s, &l.mlp_conv_proj, self.x_norm.ptr as *const f32, self.xq.ptr, self.dyn_buf.ptr as *mut f32, dyn_dim, e, tot_b);
            if l.gate_proj.ty == WeightType::Q4_0 {
                ops::ink_dflash2_conv_q8(s, self.x_norm.ptr as *const f32, self.dyn_buf.ptr as *const f32, l.mlp_conv_base.ptr as *const f32, self.xq.ptr, e, ki, gs, 0, tot_b);
            } else {
                ops::ink_dflash2_conv(s, self.x_norm.ptr as *const f32, self.dyn_buf.ptr as *const f32, l.mlp_conv_base.ptr as *const f32, self.x_conv.ptr as *mut f32, e, ki, gs, 0, tot_b);
            }
            if l.gate_proj.ty == WeightType::Q4_0 && l.down_proj.ty == WeightType::Q4_0 && (tot_b == 4 || tot_b == 6 || tot_b == 8) {
                ops::ink_dflash2_gu_swiglu_q8(s, l.gate_proj.buf.ptr as *const u8, l.up_proj.buf.ptr as *const u8, self.xq.ptr, self.gate.ptr, self.n_ff as i32, e, tot_b);
                if n_users == 1 {
                    ops::ink_dflash2_proj_conv_add_q4(s, l.down_proj.buf.ptr as *const u8, self.gate.ptr, self.dyn_buf.ptr as *const f32, l.mlp_conv_base.ptr as *const f32, self.x.ptr as *mut f32, e, self.n_ff as i32, gs, 1, tot_b);
                } else {
                    self.gemm_w(s, &l.down_proj, self.gate.ptr as *const f32, self.gate.ptr, self.ffn_out.ptr as *mut f32, e, self.n_ff as i32, tot_b);
                    ops::ink_dflash2_conv_add(s, self.ffn_out.ptr as *const f32, self.dyn_buf.ptr as *const f32, l.mlp_conv_base.ptr as *const f32, self.x.ptr as *mut f32, e, ki, gs, 1, tot_b);
                }
            } else {
                self.gemm_w(s, &l.gate_proj, self.x_conv.ptr as *const f32, self.xq.ptr, self.gate.ptr as *mut f32, self.n_ff as i32, e, tot_b);
                self.gemm_w(s, &l.up_proj, self.x_conv.ptr as *const f32, self.xq.ptr, self.up.ptr as *mut f32, self.n_ff as i32, e, tot_b);
                if l.down_proj.ty == WeightType::Q4_0 {
                    ops::ink_dspark_swiglu_q8(s, self.gate.ptr as *const f32, self.up.ptr as *const f32, self.xq.ptr, self.n_ff as i32, tot_b);
                } else {
                    ops::ink_dspark_swiglu(s, self.gate.ptr as *const f32, self.up.ptr as *const f32, self.gate.ptr as *mut f32, tot_b * self.n_ff as i32);
                }
                self.gemm_w(s, &l.down_proj, self.gate.ptr as *const f32, self.xq.ptr, self.ffn_out.ptr as *mut f32, e, self.n_ff as i32, tot_b);
                ops::ink_dflash2_conv_add(s, self.ffn_out.ptr as *const f32, self.dyn_buf.ptr as *const f32, l.mlp_conv_base.ptr as *const f32, self.x.ptr as *mut f32, e, ki, gs, 1, tot_b);
            }
        }

        ops::ink_rmsnorm_batch(s, self.x.ptr as *const f32, self.output_norm.ptr as *const f32, self.hidden.ptr as *mut f32, e, tot_b, self.eps);
    }

    unsafe fn enqueue_head_and_selector(&self, s: HipStream, target: &InklingEngine, n_users: usize, k_len: usize) {
        let e = self.n_embd as i32;
        let pred_per_u = k_len - 1;
        let n_pred = (n_users * pred_per_u) as i32;
        let h_pred_ptr: *const f32 = if n_users == 1 {
            (self.hidden.ptr as *const f32).add(self.n_embd)
        } else {
            for u in 0..n_users {
                ops::ink_copy_f32(
                    s,
                    (self.hidden.ptr as *const f32).add((u * k_len + 1) * self.n_embd),
                    (self.h_pred.ptr as *mut f32).add(u * pred_per_u * self.n_embd),
                    (pred_per_u * self.n_embd) as i32,
                );
            }
            self.h_pred.ptr as *const f32
        };

        ops::ink_quantize_q8_1(s, h_pred_ptr, self.xq.ptr, e, n_pred);
        if target.output_is_q4() {
            if n_pred >= 12 {
                ops::ink_gemm_q4_0_wmma(s, target.output_ptr(), self.xq.ptr, self.logits.ptr as *mut f32, target.cfg.vocab_padded as i32, e, n_pred);
            } else {
                ops::ink_gemm_q4_0_sudot4(s, target.output_ptr(), self.xq.ptr, self.logits.ptr as *mut f32, target.cfg.vocab_padded as i32, e, n_pred);
            }
        } else {
            ops::ink_gemm_q8_0_sudot4(s, target.output_ptr(), self.xq.ptr, self.logits.ptr as *mut f32, target.cfg.vocab_padded as i32, e, n_pred);
        }

        ops::ink_dflash2_topk16(
            s,
            self.logits.ptr as *const f32,
            self.cand_ids.ptr as *mut i32,
            self.cand_vals.ptr as *mut f32,
            target.cfg.vocab_padded as i32,
            target.cfg.vocab_unpadded as i32,
            n_pred,
        );

        self.gemm_w(
            s,
            &self.sel_hidden_proj,
            h_pred_ptr,
            self.xq.ptr,
            self.h_proj.ptr as *mut f32,
            self.selector_rank as i32,
            e,
            n_pred,
        );

        ops::ink_dflash2_select_path(
            s,
            self.cand_ids.ptr as *const i32,
            self.cand_vals.ptr as *const f32,
            self.h_proj.ptr as *const f32,
            self.sel_pred_cb.ptr as *const u16,
            self.sel_succ_cb.ptr as *const u16,
            self.d_draft_toks.ptr as *mut i32,
            n_users as i32,
            k_len as i32,
            self.selector_rank as i32,
            self.final_softcap,
        );
    }

    unsafe fn enqueue_draft_seq(&self, s: HipStream, target: &InklingEngine, seq: usize, k_len: usize) {
        let e = self.n_embd as i32;
        ops::ink_embed_q8_0_norm(
            s,
            target.tok_embd_ptr(),
            target.tok_norm_ptr(),
            self.d_draft_toks.ptr as *const i32,
            self.anchors_emb.ptr as *mut f32,
            e,
        );
        ops::ink_dspark_init_multi(
            s,
            self.anchors_emb.ptr as *const f32,
            self.mask_embd.ptr as *const f32,
            self.x.ptr as *mut f32,
            e,
            1,
            k_len as i32,
        );
        self.enqueue_backbone(s, 1, k_len, Some(seq));
        self.enqueue_head_and_selector(s, target, 1, k_len);
    }

    unsafe fn enqueue_draft_multi(&self, s: HipStream, target: &InklingEngine, n_users: usize, k_len: usize) {
        let e = self.n_embd as i32;
        let nu = n_users as i32;
        let ki = k_len as i32;
        ops::ink_dspark_scatter_toks(s, self.d_anchors.ptr as *const i32, self.d_draft_toks.ptr as *mut i32, nu, ki, 0);
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
        self.enqueue_backbone(s, n_users, k_len, None);
        self.enqueue_head_and_selector(s, target, n_users, k_len);
    }

    fn init_mask_embd(&self, target: &InklingEngine) {
        unsafe {
            ops::ink_embed_q8_0_norm(
                self.stream,
                target.tok_embd_ptr(),
                target.tok_norm_ptr(),
                self.d_mask_tok.ptr as *const i32,
                self.mask_embd.ptr as *mut f32,
                self.n_embd as i32,
            );
            rad_hip::hipStreamSynchronize(self.stream);
        }
    }

    pub fn capture_draft_graph(&mut self, target: &InklingEngine, k_len: usize) -> Result<()> {
        self.init_mask_embd(target);
        let k_len = k_len.min(self.block_size);
        let g = unsafe { CapturedGraph::capture(|s| self.enqueue_draft_seq(s, target, 0, k_len)) }.map_err(he)?;
        self.draft_graph = Some((k_len, g));
        Ok(())
    }

    pub fn draft(&mut self, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> Result<Vec<i32>> {
        let k_len = k_len.min(self.block_size);
        let pos0 = self.ctx_len;
        let use_graph = self.draft_graph.as_ref().map(|(gk, _)| *gk == k_len).unwrap_or(false);
        if use_graph {
            let (_, ref g) = self.draft_graph.as_ref().unwrap();
            self.d_draft_toks.copy_from(&anchor_tok.to_le_bytes(), Some(self.stream)).map_err(he)?;
            self.d_pos.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            self.d_ctx_len.copy_from(&self.ctx_len.to_le_bytes(), Some(self.stream)).map_err(he)?;
            g.launch_on(self.stream).map_err(he)?;
            unsafe { rad_hip::hipStreamSynchronize(self.stream); }
        } else {
            self.d_draft_toks.copy_from(&anchor_tok.to_le_bytes(), Some(self.stream)).map_err(he)?;
            self.d_pos.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            self.d_ctx_len.copy_from(&self.ctx_len.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe {
                self.enqueue_draft_seq(self.stream, target, 0, k_len);
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

    pub fn draft_seq(&mut self, seq: usize, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> Result<Vec<i32>> {
        let k_len = k_len.min(self.block_size);
        let ctx_len = self.seq_ctx_len[seq];
        let pos0 = ctx_len;
        let key = (seq, k_len);
        if !self.seq_draft_graphs.contains_key(&key) {
            let g = unsafe { CapturedGraph::capture(|s| self.enqueue_draft_seq(s, target, seq, k_len)) }.map_err(he)?;
            self.seq_draft_graphs.insert(key, g);
        }
        let g = self.seq_draft_graphs.get(&key).unwrap();
        self.d_draft_toks.copy_from(&anchor_tok.to_le_bytes(), Some(self.stream)).map_err(he)?;
        self.d_pos.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
        self.d_ctx_len.copy_from(&ctx_len.to_le_bytes(), Some(self.stream)).map_err(he)?;
        g.launch_on(self.stream).map_err(he)?;
        unsafe { rad_hip::hipStreamSynchronize(self.stream); }
        let mut out_b = vec![0u8; k_len * 4];
        let rc = unsafe {
            rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.d_draft_toks.ptr, k_len * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        Ok(out_b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
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
            self.init_mask_embd(target);
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
}
