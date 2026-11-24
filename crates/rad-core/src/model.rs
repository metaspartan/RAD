use crate::ops;
use anyhow::{anyhow, bail, Result};
use rad_gguf::GgufFile;
use rad_hip::{CapturedGraph, DevBuf, HipError, HipStream};
use std::path::Path;

fn he(e: HipError) -> anyhow::Error {
    anyhow!("hip error rc={e}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightType {
    F32,
    Bf16,
    Q8_0,
    Q4_0,
    Nvfp4,
}

pub struct DevWeight {
    pub buf: DevBuf,
    pub ty: WeightType,
    pub n: i32,
    pub k: i32,
}

impl DevWeight {
    pub fn load(g: &GgufFile, name: &str) -> Result<Self> {
        let info = g.tensors.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let bytes = g.tensor_bytes(name).ok_or_else(|| anyhow!("missing bytes {name}"))?;
        let ty = match info.ggml_type {
            0 => WeightType::F32,
            8 => WeightType::Q8_0,
            30 => WeightType::Bf16,
            40 => WeightType::Nvfp4,
            other => bail!("unsupported weight type {other} for {name}"),
        };
        let mut buf = DevBuf::new(bytes.len()).map_err(he)?;
        buf.copy_from(bytes, None).map_err(he)?;
        let k = info.dims[0] as i32;
        let n = if info.n_dims >= 2 { info.dims[1] as i32 } else { 1 };
        Ok(Self { buf, ty, n, k })
    }

    pub fn load_maybe_q4(g: &GgufFile, name: &str, to_q4: bool) -> Result<Self> {
        if !to_q4 {
            return Self::load(g, name);
        }
        let info = g.tensors.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let k = info.dims[0] as i32;
        let n = if info.n_dims >= 2 { info.dims[1] as i32 } else { 1 };
        let nblk = (k as usize) / 32;
        let rbytes = nblk * 16 + ((nblk * 2 + 15) & !15);
        if info.ggml_type == 8 {
            let src = g.tensor_bytes(name).ok_or_else(|| anyhow!("missing bytes {name}"))?;
            let n_blocks = src.len() / 34;
            let n_rows = n_blocks / nblk;
            let mut src_dev = DevBuf::new(src.len()).map_err(he)?;
            src_dev.copy_from(src, None).map_err(he)?;
            let dst_dev = DevBuf::new(n_rows * rbytes).map_err(he)?;
            unsafe {
                ops::ink_q8_0_to_q4_0(std::ptr::null_mut(), src_dev.ptr as *const u8, dst_dev.ptr as *mut u8, n_rows, nblk as i32);
                rad_hip::hipStreamSynchronize(std::ptr::null_mut());
            }
            drop(src_dev);
            return Ok(Self { buf: dst_dev, ty: WeightType::Q4_0, n, k });
        }
        if info.ggml_type == 40 {
            let src = g.tensor_bytes(name).ok_or_else(|| anyhow!("missing bytes {name}"))?;
            let n_nv = src.len() / 36;
            let nblk_nv = (k as usize) / 64;
            let n_rows = n_nv / nblk_nv;
            let mut src_dev = DevBuf::new(src.len()).map_err(he)?;
            src_dev.copy_from(src, None).map_err(he)?;
            let dst_dev = DevBuf::new(n_rows * rbytes).map_err(he)?;
            unsafe {
                ops::ink_nvfp4_to_q4_0(std::ptr::null_mut(), src_dev.ptr as *const u8, dst_dev.ptr as *mut u8, n_rows, nblk_nv as i32);
                rad_hip::hipStreamSynchronize(std::ptr::null_mut());
            }
            drop(src_dev);
            return Ok(Self { buf: dst_dev, ty: WeightType::Q4_0, n, k });
        }
        Self::load(g, name)
    }

    fn load_f32(g: &GgufFile, name: &str) -> Result<DevBuf> {
        let info = g.tensors.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        if info.ggml_type != 0 {
            bail!("expected F32 for {name}, got {}", info.ggml_type);
        }
        let bytes = g.tensor_bytes(name).ok_or_else(|| anyhow!("missing bytes {name}"))?;
        let mut buf = DevBuf::new(bytes.len()).map_err(he)?;
        buf.copy_from(bytes, None).map_err(he)?;
        Ok(buf)
    }

    fn load_f32_opt(g: &GgufFile, name: &str) -> Result<Option<DevBuf>> {
        if !g.tensors.contains_key(name) {
            return Ok(None);
        }
        Ok(Some(Self::load_f32(g, name)?))
    }

    unsafe fn gemv(&self, s: HipStream, x: *const f32, y: *mut f32) {
        match self.ty {
            WeightType::F32 => {
                ops::ink_gemv_f32(s, self.buf.ptr as *const f32, x, y, self.n, self.k);
            }
            WeightType::Bf16 => {
                ops::ink_gemv_bf16(s, self.buf.ptr as *const u16, x, y, self.n, self.k);
            }
            WeightType::Q8_0 => {
                ops::ink_gemv_q8_0(s, self.buf.ptr as *const u8, x, y, self.n, self.k);
            }
            WeightType::Q4_0 => {
                ops::ink_gemv_q8_0(s, self.buf.ptr as *const u8, x, y, self.n, self.k);
            }
            WeightType::Nvfp4 => {
                ops::ink_gemv_nvfp4(s, self.buf.ptr as *const u8, x, y, self.n, self.k, 1.0);
            }
        }
    }
}

pub struct InklingConfig {
    pub n_phys: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub dh: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub swa_window: usize,
    pub is_swa: Vec<bool>,
    pub core_start: usize,
    pub core_end: usize,
    pub n_passes: usize,
    pub gate_rank: usize,
    pub attn_res_block: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub vocab_padded: usize,
    pub vocab_unpadded: usize,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub route_scale: f32,
    pub final_softcap: f32,
}

impl InklingConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g.kv_str("general.architecture").unwrap_or("");
        let pfx = if arch == "inkling2_loop" { "inkling2_loop" } else if arch == "inkling2" { "inkling2" } else {
            bail!("unexpected architecture: {arch}");
        };
        let k = |s: &str| format!("{pfx}.{s}");
        let n_phys = if pfx == "inkling2_loop" {
            g.kv_u64(&k("block_count")).unwrap_or(24) as usize
        } else {
            24
        };
        let swa_full = g.kv_bool_arr(&k("attention.sliding_window_pattern"))
            .ok_or_else(|| anyhow!("missing sliding_window_pattern"))?;
        let is_swa = swa_full[..n_phys].to_vec();
        Ok(Self {
            n_phys,
            n_embd: g.kv_u64(&k("embedding_length")).unwrap_or(2304) as usize,
            n_head: 24,
            n_head_kv: 8,
            dh: g.kv_u64(&k("attention.key_length")).unwrap_or(128) as usize,
            d_rel: g.kv_u64(&k("attention.relative_length")).unwrap_or(16) as usize,
            rel_extent: g.kv_u64(&k("attention.relative_extent")).unwrap_or(1024) as usize,
            swa_window: g.kv_u64(&k("attention.sliding_window")).unwrap_or(1024) as usize,
            is_swa,
            core_start: g.kv_u64(&k("loop.start")).or_else(|| g.kv_u64(&k("loop.core_start"))).unwrap_or(6) as usize,
            core_end: g.kv_u64(&k("loop.end")).or_else(|| g.kv_u64(&k("loop.core_end"))).unwrap_or(18) as usize,
            n_passes: g.kv_u64(&k("loop.count")).or_else(|| g.kv_u64(&k("num_loops"))).unwrap_or(3) as usize,
            gate_rank: g.kv_u64(&k("loop.gate_rank")).unwrap_or(64) as usize,
            attn_res_block: g.kv_u64(&k("attn_res.block_size")).unwrap_or(6) as usize,
            n_expert: g.kv_u64(&k("expert_count")).unwrap_or(128) as usize,
            n_expert_used: g.kv_u64(&k("expert_used_count")).unwrap_or(8) as usize,
            n_ff_exp: g.kv_u64(&k("expert_feed_forward_length")).unwrap_or(1152) as usize,
            vocab_padded: g.kv_u64(&k("vocab_size")).unwrap_or(201024) as usize,
            vocab_unpadded: g.kv_u64(&k("vocab_size_unpadded")).unwrap_or(200058) as usize,
            eps: g.kv_f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            qk_clip: g.kv_f32(&k("attention.qk_clip")).unwrap_or(22.627417),
            rel_clip: g.kv_f32(&k("attention.relative_clip")).unwrap_or(8.0),
            log_floor: g.kv_u64(&k("attention.log_scale_floor")).unwrap_or(8192) as f32,
            log_alpha: g.kv_f32(&k("attention.log_scale_alpha")).unwrap_or(0.1),
            situ_beta: g.kv_f32(&k("activation.situ_beta")).unwrap_or(4.0),
            situ_lb: g.kv_f32(&k("activation.situ_linear_beta")).unwrap_or(25.0),
            route_scale: g.kv_f32(&k("expert_weights_scale")).unwrap_or(8.0),
            final_softcap: g.kv_f32(&k("final_logit_softcapping")).unwrap_or(30.0),
        })
    }

    pub fn n_slots(&self) -> usize {
        self.n_phys + (self.n_passes - 1) * (self.core_end - self.core_start)
    }
}

struct PhysLayer {
    is_swa: bool,
    attn_norm: Option<DevBuf>,
    attn_q_inp_norm: Option<DevBuf>,
    attn_k_inp_norm: Option<DevBuf>,
    attn_v_inp_norm: Option<DevBuf>,
    attn_res_query: Option<DevBuf>,
    ffn_res_query: DevBuf,
    wq: DevWeight,
    wk: DevWeight,
    wv: DevWeight,
    wrel: DevWeight,
    wo: DevWeight,
    wqkv_gate: DevWeight,
    attn_sink_proj: DevWeight,
    attn_k_conv: DevBuf,
    attn_v_conv: DevBuf,
    attn_k_conv_proj: Option<DevWeight>,
    attn_v_conv_proj: Option<DevWeight>,
    attn_post_conv: DevBuf,
    ffn_post_conv: DevBuf,
    attn_q_clip: DevBuf,
    attn_rel_clip: DevBuf,
    attn_rel_bias: DevBuf,
    ffn_norm: DevBuf,
    ffn_gate_inp: DevWeight,
    ffn_exp_probs_b: Option<DevBuf>,
    ffn_gate_up_exps: DevWeight,
    ffn_gate_up_scale: Option<DevBuf>,
    ffn_down_exps: DevWeight,
    ffn_down_scale: Option<DevBuf>,
}

struct LoopLayer {
    msg_query: DevBuf,
    ctrl_down: [DevBuf; 4],
    ctrl_gain: [DevBuf; 4],
    ctrl_forget: [DevBuf; 4],
}

struct SlotState {
    is_swa: bool,
    max_ctx: usize,
    k_cache: DevBuf,
    v_cache: DevBuf,
    k_hist: DevBuf,
    v_hist: DevBuf,
    attn_post_hist: DevBuf,
    ffn_post_hist: DevBuf,
    k_snap: DevBuf,
    v_snap: DevBuf,
    attn_post_snap: DevBuf,
    ffn_post_snap: DevBuf,
}

struct Scratch {
    d_tok: DevBuf,
    d_pos: DevBuf,
    d_out_tok: DevBuf,
    cur: DevBuf,
    inp_sa: DevBuf,
    ffn_inp: DevBuf,
    prelude: DevBuf,
    blocks: DevBuf,
    bank: DevBuf,
    bank_out: DevBuf,
    mixed: DevBuf,
    msgs: DevBuf,
    inp0: DevBuf,
    inp1: DevBuf,
    inp2: DevBuf,
    ctrl_h: DevBuf,
    q: DevBuf,
    k_raw: DevBuf,
    v_raw: DevBuf,
    k_conv: DevBuf,
    v_conv: DevBuf,
    k_dyn: DevBuf,
    v_dyn: DevBuf,
    r: DevBuf,
    sinks: DevBuf,
    attn_out: DevBuf,
    gate_out: DevBuf,
    o_raw: DevBuf,
    o_conv: DevBuf,
    ffn_normed: DevBuf,
    router_logits: DevBuf,
    wtop: DevBuf,
    itop: DevBuf,
    moe_act: DevBuf,
    moe_raw: DevBuf,
    moe_conv: DevBuf,
    logits: DevBuf,
    dspark_feats: DevBuf,
}

const MAX_BATCH: usize = 1024;

struct BatchScratch {
    d_toks: DevBuf,
    d_pos0: DevBuf,
    d_seq_ids: DevBuf,
    d_positions: DevBuf,
    d_out_toks: DevBuf,
    cur: DevBuf,
    ffn_inp: DevBuf,
    prelude: DevBuf,
    blocks: DevBuf,
    bank: DevBuf,
    bank_out: DevBuf,
    mixed: DevBuf,
    msgs: DevBuf,
    inp0: DevBuf,
    inp1: DevBuf,
    inp2: DevBuf,
    ctrl_h: DevBuf,
    q: DevBuf,
    k_raw: DevBuf,
    v_raw: DevBuf,
    k_dyn: DevBuf,
    v_dyn: DevBuf,
    r: DevBuf,
    sinks: DevBuf,
    attn_out: DevBuf,
    gate_out: DevBuf,
    o_raw: DevBuf,
    ffn_normed: DevBuf,
    router_logits: DevBuf,
    wtop: DevBuf,
    itop: DevBuf,
    moe_act: DevBuf,
    moe_dn_slots: DevBuf,
    moe_raw: DevBuf,
    logits: DevBuf,
    dspark_feats: DevBuf,
    xq: DevBuf,
    act_q: DevBuf,
    exp_counts: DevBuf,
    exp_offsets: DevBuf,
    tok_sorted: DevBuf,
    slot_sorted: DevBuf,
    wt_sorted: DevBuf,
}

pub struct InklingEngine {
    pub cfg: InklingConfig,
    pub max_ctx: usize,
    pub max_seqs: usize,
    tok_embd: DevWeight,
    tok_norm: DevBuf,
    output_res_query: DevBuf,
    output_norm: DevBuf,
    output: DevWeight,
    layers: Vec<PhysLayer>,
    loop_layers: Vec<LoopLayer>,
    slots: Vec<SlotState>,
    sc: Scratch,
    bsc: BatchScratch,
    stream: HipStream,
    decode_graph: Option<CapturedGraph>,
    verify_graph: Option<(usize, CapturedGraph)>,
    multi_graphs: std::collections::HashMap<usize, CapturedGraph>,
    multi_verify_graphs: std::collections::HashMap<(usize, usize), CapturedGraph>,
    pub pos: i32,
    pub seq_pos: Vec<i32>,
    pub seq_tokens: Vec<Vec<i32>>,
    b1_sudot4: bool,
}

unsafe impl Send for InklingEngine {}

impl InklingEngine {
    pub fn load(path: &Path, max_ctx: usize) -> Result<Self> {
        Self::load_multi(path, max_ctx, 1)
    }

    pub fn load_multi(path: &Path, max_ctx: usize, max_seqs: usize) -> Result<Self> {
        let max_seqs = max_seqs.max(1).min(MAX_BATCH);
        let g = GgufFile::open(path)?;
        let cfg = InklingConfig::from_gguf(&g)?;
        let fast_q4 = std::env::var("RAD_FAST_QUANT").map(|v| v != "0").unwrap_or(true);
        let attn_q4_mode = std::env::var("RAD_ATTN_Q4").unwrap_or_else(|_| if fast_q4 { "all".into() } else { "none".into() });
        let moe_q4_mode = std::env::var("RAD_MOE_Q4").unwrap_or_else(|_| if fast_q4 { "all".into() } else { "0".into() });
        let b1_sudot4 = std::env::var("RAD_B1_SUDOT4").map(|v| v == "1").unwrap_or(false);
        let tok_embd = DevWeight::load(&g, "token_embd.weight")?;
        let tok_norm = DevWeight::load_f32(&g, "token_embd_norm.weight")?;
        let output_res_query = DevWeight::load_f32(&g, "output_res_query.weight")?;
        let output_norm = DevWeight::load_f32(&g, "output_norm.weight")?;
        let output = DevWeight::load_maybe_q4(&g, "output.weight", fast_q4)?;

        let mut layers = Vec::with_capacity(cfg.n_phys);
        for i in 0..cfg.n_phys {
            let p = |s: &str| format!("blk.{i}.{s}");
            let is_swa = cfg.is_swa[i];
            let is_outer = i < cfg.core_start || i >= cfg.core_end;
            let q4_attn = match attn_q4_mode.as_str() {
                "all" => true,
                "outer" => is_outer,
                _ => false,
            };
            layers.push(PhysLayer {
                is_swa,
                attn_norm: if is_swa { Some(DevWeight::load_f32(&g, &p("attn_norm.weight"))?) } else { None },
                attn_q_inp_norm: if !is_swa { Some(DevWeight::load_f32(&g, &p("attn_q_inp_norm.weight"))?) } else { None },
                attn_k_inp_norm: if !is_swa { Some(DevWeight::load_f32(&g, &p("attn_k_inp_norm.weight"))?) } else { None },
                attn_v_inp_norm: if !is_swa { Some(DevWeight::load_f32(&g, &p("attn_v_inp_norm.weight"))?) } else { None },
                attn_res_query: if i > 0 { Some(DevWeight::load_f32(&g, &p("attn_res_query.weight"))?) } else { None },
                ffn_res_query: DevWeight::load_f32(&g, &p("ffn_res_query.weight"))?,
                wq: DevWeight::load_maybe_q4(&g, &p("attn_q.weight"), q4_attn)?,
                wk: DevWeight::load_maybe_q4(&g, &p("attn_k.weight"), q4_attn)?,
                wv: DevWeight::load_maybe_q4(&g, &p("attn_v.weight"), q4_attn)?,
                wrel: DevWeight::load_maybe_q4(&g, &p("attn_rel.weight"), q4_attn)?,
                wo: DevWeight::load_maybe_q4(&g, &p("attn_output.weight"), q4_attn)?,
                wqkv_gate: DevWeight::load_maybe_q4(&g, &p("attn_gate.weight"), q4_attn)?,
                attn_sink_proj: DevWeight::load(&g, &p("attn_sink_proj.weight"))?,
                attn_k_conv: DevWeight::load_f32(&g, &p("attn_k_conv.weight"))?,
                attn_v_conv: DevWeight::load_f32(&g, &p("attn_v_conv.weight"))?,
                attn_k_conv_proj: if !is_swa { Some(DevWeight::load(&g, &p("attn_k_conv_proj.weight"))?) } else { None },
                attn_v_conv_proj: if !is_swa { Some(DevWeight::load(&g, &p("attn_v_conv_proj.weight"))?) } else { None },
                attn_post_conv: DevWeight::load_f32(&g, &p("attn_post_conv.weight"))?,
                ffn_post_conv: DevWeight::load_f32(&g, &p("ffn_post_conv.weight"))?,
                attn_q_clip: DevWeight::load_f32(&g, &p("attn_q_clip.weight"))?,
                attn_rel_clip: DevWeight::load_f32(&g, &p("attn_rel_clip.weight"))?,
                attn_rel_bias: DevWeight::load_f32(&g, &p("attn_rel_bias.weight"))?,
                ffn_norm: DevWeight::load_f32(&g, &p("ffn_norm.weight"))?,
                ffn_gate_inp: DevWeight::load(&g, &p("ffn_gate_inp.weight"))?,
                ffn_exp_probs_b: DevWeight::load_f32_opt(&g, &p("exp_probs_b.bias"))?,
                ffn_gate_up_exps: DevWeight::load_maybe_q4(&g, &p("ffn_gate_up_exps.weight"), moe_q4_mode == "1" || moe_q4_mode == "all" || (moe_q4_mode == "q8" && i >= 20))?,
                ffn_gate_up_scale: DevWeight::load_f32_opt(&g, &p("ffn_gate_up_exps.scale"))?,
                ffn_down_exps: DevWeight::load_maybe_q4(&g, &p("ffn_down_exps.weight"), moe_q4_mode == "1" || moe_q4_mode == "all" || (moe_q4_mode == "q8" && i >= 20))?,
                ffn_down_scale: DevWeight::load_f32_opt(&g, &p("ffn_down_exps.scale"))?,
            });
        }

        let mut loop_layers = Vec::with_capacity(cfg.core_end - cfg.core_start);
        for i in cfg.core_start..cfg.core_end {
            let p = |s: &str| format!("blk.{i}.{s}");
            let names = ["q", "k", "v", "ffn"];
            let mut ctrl_down = Vec::with_capacity(4);
            let mut ctrl_gain = Vec::with_capacity(4);
            let mut ctrl_forget = Vec::with_capacity(4);
            for nm in names {
                ctrl_down.push(DevWeight::load_f32(&g, &p(&format!("loop_{nm}_ctrl_down.weight")))?);
                ctrl_gain.push(DevWeight::load_f32(&g, &p(&format!("loop_{nm}_ctrl_gain.weight")))?);
                ctrl_forget.push(DevWeight::load_f32(&g, &p(&format!("loop_{nm}_ctrl_forget.weight")))?);
            }
            loop_layers.push(LoopLayer {
                msg_query: DevWeight::load_f32(&g, &p("loop_msg_query.weight"))?,
                ctrl_down: ctrl_down.try_into().ok().unwrap(),
                ctrl_gain: ctrl_gain.try_into().ok().unwrap(),
                ctrl_forget: ctrl_forget.try_into().ok().unwrap(),
            });
        }

        let n_slots = cfg.n_slots();
        let mut slots = Vec::with_capacity(n_slots);
        let kv_dim = cfg.n_head_kv * cfg.dh;
        for il in 0..n_slots {
            let ip = if il < cfg.n_phys {
                il
            } else {
                cfg.core_start + ((il - cfg.n_phys) % (cfg.core_end - cfg.core_start))
            };
            let is_swa = cfg.is_swa[ip];
            let slot_ctx = if is_swa { (cfg.swa_window + 16).min(max_ctx) } else { max_ctx };
            let kv_bytes = max_seqs * cfg.n_head_kv * slot_ctx * cfg.dh * 2;
            let mut k_cache = DevBuf::new(kv_bytes).map_err(he)?;
            let mut v_cache = DevBuf::new(kv_bytes).map_err(he)?;
            let mut k_hist = DevBuf::new(max_seqs * kv_dim * 3 * 4).map_err(he)?;
            let mut v_hist = DevBuf::new(max_seqs * kv_dim * 3 * 4).map_err(he)?;
            let mut attn_post_hist = DevBuf::new(max_seqs * cfg.n_embd * 3 * 4).map_err(he)?;
            let mut ffn_post_hist = DevBuf::new(max_seqs * cfg.n_embd * 3 * 4).map_err(he)?;
            let snap_users = max_seqs.min(8);
            let k_snap = DevBuf::new(snap_users * 8 * kv_dim * 3 * 4).map_err(he)?;
            let v_snap = DevBuf::new(snap_users * 8 * kv_dim * 3 * 4).map_err(he)?;
            let attn_post_snap = DevBuf::new(snap_users * 8 * cfg.n_embd * 3 * 4).map_err(he)?;
            let ffn_post_snap = DevBuf::new(snap_users * 8 * cfg.n_embd * 3 * 4).map_err(he)?;
            k_cache.zero().map_err(he)?;
            v_cache.zero().map_err(he)?;
            k_hist.zero().map_err(he)?;
            v_hist.zero().map_err(he)?;
            attn_post_hist.zero().map_err(he)?;
            ffn_post_hist.zero().map_err(he)?;
            slots.push(SlotState {
                is_swa,
                max_ctx: slot_ctx,
                k_cache,
                v_cache,
                k_hist,
                v_hist,
                attn_post_hist,
                ffn_post_hist,
                k_snap,
                v_snap,
                attn_post_snap,
                ffn_post_snap,
            });
        }

        let e = cfg.n_embd;
        let q_dim = cfg.n_head * cfg.dh;
        let r_dim = cfg.n_head * cfg.d_rel;
        let sc = Scratch {
            d_tok: DevBuf::new(4).map_err(he)?,
            d_pos: DevBuf::new(4).map_err(he)?,
            d_out_tok: DevBuf::new(4).map_err(he)?,
            cur: DevBuf::new(e * 4).map_err(he)?,
            inp_sa: DevBuf::new(e * 4).map_err(he)?,
            ffn_inp: DevBuf::new(e * 4).map_err(he)?,
            prelude: DevBuf::new(e * 4).map_err(he)?,
            blocks: DevBuf::new(8 * e * 4).map_err(he)?,
            bank: DevBuf::new(4 * e * 4).map_err(he)?,
            bank_out: DevBuf::new(e * 4).map_err(he)?,
            mixed: DevBuf::new(4 * e * 4).map_err(he)?,
            msgs: DevBuf::new(4 * e * 4).map_err(he)?,
            inp0: DevBuf::new(e * 4).map_err(he)?,
            inp1: DevBuf::new(e * 4).map_err(he)?,
            inp2: DevBuf::new(e * 4).map_err(he)?,
            ctrl_h: DevBuf::new(3 * cfg.gate_rank * 4).map_err(he)?,
            q: DevBuf::new(q_dim * 4).map_err(he)?,
            k_raw: DevBuf::new(kv_dim * 4).map_err(he)?,
            v_raw: DevBuf::new(kv_dim * 4).map_err(he)?,
            k_conv: DevBuf::new(kv_dim * 4).map_err(he)?,
            v_conv: DevBuf::new(kv_dim * 4).map_err(he)?,
            k_dyn: DevBuf::new(4 * cfg.n_head_kv * 4).map_err(he)?,
            v_dyn: DevBuf::new(4 * cfg.n_head_kv * 4).map_err(he)?,
            r: DevBuf::new(r_dim * 4).map_err(he)?,
            sinks: DevBuf::new(cfg.n_head * 4).map_err(he)?,
            attn_out: DevBuf::new(q_dim * 4).map_err(he)?,
            gate_out: DevBuf::new(q_dim * 4).map_err(he)?,
            o_raw: DevBuf::new(e * 4).map_err(he)?,
            o_conv: DevBuf::new(e * 4).map_err(he)?,
            ffn_normed: DevBuf::new(e * 4).map_err(he)?,
            router_logits: DevBuf::new(cfg.n_expert * 4).map_err(he)?,
            wtop: DevBuf::new(cfg.n_expert_used * 4).map_err(he)?,
            itop: DevBuf::new(cfg.n_expert_used * 4).map_err(he)?,
            moe_act: DevBuf::new(cfg.n_expert_used * cfg.n_ff_exp * 4).map_err(he)?,
            moe_raw: DevBuf::new(e * 4).map_err(he)?,
            moe_conv: DevBuf::new(e * 4).map_err(he)?,
            logits: DevBuf::new(cfg.vocab_padded * 4).map_err(he)?,
            dspark_feats: DevBuf::new(4 * e * 4).map_err(he)?,
        };

        let mb = MAX_BATCH;
        let bsc = BatchScratch {
            d_toks: DevBuf::new(mb * 4).map_err(he)?,
            d_pos0: DevBuf::new(4).map_err(he)?,
            d_seq_ids: DevBuf::new(mb * 4).map_err(he)?,
            d_positions: DevBuf::new(mb * 4).map_err(he)?,
            d_out_toks: DevBuf::new(mb * 4).map_err(he)?,
            cur: DevBuf::new(mb * e * 4).map_err(he)?,
            ffn_inp: DevBuf::new(mb * e * 4).map_err(he)?,
            prelude: DevBuf::new(mb * e * 4).map_err(he)?,
            blocks: DevBuf::new(8 * mb * e * 4).map_err(he)?,
            bank: DevBuf::new(4 * mb * e * 4).map_err(he)?,
            bank_out: DevBuf::new(mb * e * 4).map_err(he)?,
            mixed: DevBuf::new(4 * mb * e * 4).map_err(he)?,
            msgs: DevBuf::new(4 * mb * e * 4).map_err(he)?,
            inp0: DevBuf::new(mb * e * 4).map_err(he)?,
            inp1: DevBuf::new(mb * e * 4).map_err(he)?,
            inp2: DevBuf::new(mb * e * 4).map_err(he)?,
            ctrl_h: DevBuf::new(3 * mb * cfg.gate_rank * 4).map_err(he)?,
            q: DevBuf::new(mb * q_dim * 4).map_err(he)?,
            k_raw: DevBuf::new(mb * kv_dim * 4).map_err(he)?,
            v_raw: DevBuf::new(mb * kv_dim * 4).map_err(he)?,
            k_dyn: DevBuf::new(mb * 4 * cfg.n_head_kv * 4).map_err(he)?,
            v_dyn: DevBuf::new(mb * 4 * cfg.n_head_kv * 4).map_err(he)?,
            r: DevBuf::new(mb * r_dim * 4).map_err(he)?,
            sinks: DevBuf::new(mb * cfg.n_head * 4).map_err(he)?,
            attn_out: DevBuf::new(mb * q_dim * 4).map_err(he)?,
            gate_out: DevBuf::new(mb * q_dim * 4).map_err(he)?,
            o_raw: DevBuf::new(mb * e * 4).map_err(he)?,
            ffn_normed: DevBuf::new(mb * e * 4).map_err(he)?,
            router_logits: DevBuf::new(mb * cfg.n_expert * 4).map_err(he)?,
            wtop: DevBuf::new(mb * cfg.n_expert_used * 4).map_err(he)?,
            itop: DevBuf::new(mb * cfg.n_expert_used * 4).map_err(he)?,
            moe_act: DevBuf::new(mb * cfg.n_expert_used * cfg.n_ff_exp * 4).map_err(he)?,
            moe_dn_slots: DevBuf::new(mb * cfg.n_expert_used * e * 4).map_err(he)?,
            moe_raw: DevBuf::new(mb * e * 4).map_err(he)?,
            logits: DevBuf::new(max_seqs.max(32) * cfg.vocab_padded * 4).map_err(he)?,
            dspark_feats: DevBuf::new(4 * mb * e * 4).map_err(he)?,
            xq: DevBuf::new(3 * mb * (q_dim / 32) * 40).map_err(he)?,
            act_q: DevBuf::new(mb * cfg.n_expert_used * (cfg.n_ff_exp / 32) * 40).map_err(he)?,
            exp_counts: DevBuf::new(256 * 4).map_err(he)?,
            exp_offsets: DevBuf::new(129 * 4).map_err(he)?,
            tok_sorted: DevBuf::new(mb * cfg.n_expert_used * 4).map_err(he)?,
            slot_sorted: DevBuf::new(mb * cfg.n_expert_used * 4).map_err(he)?,
            wt_sorted: DevBuf::new(mb * cfg.n_expert_used * 4).map_err(he)?,
        };

        extern "C" {
            fn hipStreamCreate(s: *mut HipStream) -> i32;
        }
        let mut stream: HipStream = std::ptr::null_mut();
        unsafe { hipStreamCreate(&mut stream) };

        Ok(Self {
            cfg,
            max_ctx,
            max_seqs,
            tok_embd,
            tok_norm,
            output_res_query,
            output_norm,
            output,
            layers,
            loop_layers,
            slots,
            sc,
            bsc,
            stream,
            decode_graph: None,
            verify_graph: None,
            multi_graphs: std::collections::HashMap::new(),
            multi_verify_graphs: std::collections::HashMap::new(),
            pos: 0,
            seq_pos: vec![0; max_seqs],
            seq_tokens: vec![Vec::new(); max_seqs],
            b1_sudot4,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        for sl in &mut self.slots {
            sl.k_hist.zero().map_err(he)?;
            sl.v_hist.zero().map_err(he)?;
            sl.attn_post_hist.zero().map_err(he)?;
            sl.ffn_post_hist.zero().map_err(he)?;
        }
        unsafe { rad_hip::hipStreamSynchronize(std::ptr::null_mut()) };
        self.pos = 0;
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
            ops::ink_quantize_q8_1(s, p0, xq0, e, 1);
            let (q1, q2) = if p0 == p1 && p0 == p2 {
                (xq0 as *const core::ffi::c_void, xq0 as *const core::ffi::c_void)
            } else {
                ops::ink_quantize_q8_1(s, p1, xq1, e, 1);
                ops::ink_quantize_q8_1(s, p2, xq2, e, 1);
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

        let block_dst = if ip % self.cfg.attn_res_block == 0 {
            let dst = blocks.add((*n_blocks) * self.cfg.n_embd);
            *n_blocks += 1;
            dst
        } else {
            std::ptr::null_mut()
        };

        if l.wo.ty == WeightType::Q4_0 {
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
                ops::ink_quantize_q8_1(s, ffn_normed, xq0, e, 1);
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
        ops::ink_quantize_q8_1(s, p0, xq0, e, bi);
        let (q1, q2) = if p0 == p1 && p0 == p2 {
            (xq0 as *const core::ffi::c_void, xq0 as *const core::ffi::c_void)
        } else {
            ops::ink_quantize_q8_1(s, p1, xq1, e, bi);
            ops::ink_quantize_q8_1(s, p2, xq2, e, bi);
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
        let k_snap = if b <= 8 { sl.k_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
        let v_snap = if b <= 8 { sl.v_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
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

        let attn_out = self.bsc.attn_out.ptr as *mut f32;
        if b >= 16 {
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
        } else {
            ops::ink_attn_batch_gated(
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
        }

        let o_raw = self.bsc.o_raw.ptr as *mut f32;
        ops::ink_quantize_q8_1(s, attn_out, self.bsc.xq.ptr, q_dim, bi);
        match l.wo.ty {
            WeightType::Q4_0 if b >= 12 => ops::ink_gemm_q4_0_wmma(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
            WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
            _ => ops::ink_gemm_q8_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
        };

        let block_dst = if ip % self.cfg.attn_res_block == 0 {
            let dst = blocks.add((*n_blocks) * MAX_BATCH * self.cfg.n_embd);
            *n_blocks += 1;
            dst
        } else {
            std::ptr::null_mut()
        };
        let ap_snap = if b <= 8 { sl.attn_post_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
        ops::ink_shortconv_res_batch_snap(s, o_raw, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, ap_snap, cur, block_dst, ffn_inp, e, bi);

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
        ops::ink_moe_router_batch(s, router_logits, bias_ptr, self.bsc.wtop.ptr as *mut f32, self.bsc.itop.ptr as *mut i32, self.cfg.n_expert as i32, self.cfg.n_expert_used as i32, self.cfg.route_scale, bi);

        let gu_sc = l.ffn_gate_up_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let dn_sc = l.ffn_down_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let moe_act = self.bsc.moe_act.ptr as *mut f32;
        let moe_raw = self.bsc.moe_raw.ptr as *mut f32;

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
                ops::ink_quantize_q8_1(s, ffn_normed, self.bsc.xq.ptr, e, bi);
                ops::ink_moe_build_groups(
                    s, self.bsc.itop.ptr as *const i32, self.bsc.wtop.ptr as *const f32, dn_sc,
                    self.bsc.exp_counts.ptr as *mut i32, self.bsc.exp_offsets.ptr as *mut i32,
                    self.bsc.tok_sorted.ptr as *mut i32, self.bsc.slot_sorted.ptr as *mut i32,
                    self.bsc.wt_sorted.ptr as *mut f32, self.cfg.n_expert_used as i32, bi,
                );
                if b >= 48 {
                    ops::ink_moe_gu_q4_0_wmma(
                        s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, self.bsc.xq.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        moe_act, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
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
                    ops::ink_quantize_q8_1(s, moe_act, self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, bi * self.cfg.n_expert_used as i32);
                    ops::ink_moe_dn_q4_0_wmma(
                        s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        self.bsc.wt_sorted.ptr as *const f32, self.bsc.moe_dn_slots.ptr as *mut f32, moe_raw,
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
        let fp_snap = if b <= 8 { sl.ffn_post_snap.ptr as *mut f32 } else { std::ptr::null_mut() };
        ops::ink_shortconv_res_batch_snap(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, fp_snap, ffn_inp, std::ptr::null_mut(), cur, e, bi);
    }

    pub unsafe fn enqueue_forward_batch(&self, s: HipStream, b: usize, all_logits: bool) {
        let e = self.cfg.n_embd as i32;
        let mb = MAX_BATCH as i32;
        let bi = b as i32;
        let be = (b * self.cfg.n_embd) as i32;
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
        self.verify_graph = Some((b, g));
        Ok(())
    }

    pub fn verify_batch(&mut self, tokens: &[i32]) -> Result<Vec<i32>> {
        let b = tokens.len();
        assert!(b >= 1 && b <= 8);
        let pos0 = self.pos;
        let tok_bytes = unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, b * 4) };
        if let Some((gb, ref g)) = self.verify_graph {
            if gb == b {
                self.bsc.d_toks.copy_from(tok_bytes, Some(g.stream)).map_err(he)?;
                self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(g.stream)).map_err(he)?;
                g.launch().map_err(he)?;
                g.sync().map_err(he)?;
            } else {
                self.bsc.d_toks.copy_from(tok_bytes, Some(self.stream)).map_err(he)?;
                self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
                unsafe { self.enqueue_forward_batch(self.stream, b, true); rad_hip::hipStreamSynchronize(self.stream); }
            }
        } else {
            self.bsc.d_toks.copy_from(tok_bytes, Some(self.stream)).map_err(he)?;
            self.bsc.d_pos0.copy_from(&pos0.to_le_bytes(), Some(self.stream)).map_err(he)?;
            unsafe { self.enqueue_forward_batch(self.stream, b, true); rad_hip::hipStreamSynchronize(self.stream); }
        }
        let mut out_b = vec![0u8; b * 4];
        let rc = unsafe {
            rad_hip::hipMemcpy(out_b.as_mut_ptr() as *mut core::ffi::c_void, self.bsc.d_out_toks.ptr, b * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS { return Err(he(rc)); }
        self.pos += b as i32;
        Ok(out_b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
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
    pub fn gather_batch_dspark_feats(&self, tok_idx: usize) -> *const f32 {
        unsafe {
            ops::ink_gather_dspark_feats(
                self.stream,
                self.bsc.dspark_feats.ptr as *const f32,
                self.sc.dspark_feats.ptr as *mut f32,
                tok_idx as i32,
                MAX_BATCH as i32,
                self.cfg.n_embd as i32,
            );
            rad_hip::hipStreamSynchronize(self.stream);
        }
        self.sc.dspark_feats.ptr as *const f32
    }
    pub fn rewind_verify_to(&mut self, n_acc: usize, b: usize) -> Result<()> {
        if n_acc < b {
            let idx = n_acc - 1;
            let kv_n = (self.cfg.n_head_kv * self.cfg.dh * 3) as i32;
            let e_n = (self.cfg.n_embd * 3) as i32;
            let kv_off = idx * (kv_n as usize);
            let e_off = idx * (e_n as usize);
            unsafe {
                for sl in &self.slots {
                    ops::ink_copy_f32(self.stream, (sl.k_snap.ptr as *const f32).add(kv_off), sl.k_hist.ptr as *mut f32, kv_n);
                    ops::ink_copy_f32(self.stream, (sl.v_snap.ptr as *const f32).add(kv_off), sl.v_hist.ptr as *mut f32, kv_n);
                    ops::ink_copy_f32(self.stream, (sl.attn_post_snap.ptr as *const f32).add(e_off), sl.attn_post_hist.ptr as *mut f32, e_n);
                    ops::ink_copy_f32(self.stream, (sl.ffn_post_snap.ptr as *const f32).add(e_off), sl.ffn_post_hist.ptr as *mut f32, e_n);
                }
                rad_hip::hipStreamSynchronize(self.stream);
            }
            self.pos -= (b - n_acc) as i32;
            self.seq_pos[0] = self.pos;
        }
        Ok(())
    }

    pub fn reset_seq(&mut self, seq: usize) -> Result<()> {
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
            return Ok(vec![nxt]);
        }
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
        assert!(k_len >= 1 && k_len <= 8);
        let total_b = n_users * k_len;
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
        let mut any = false;
        unsafe {
            for (u, &sid) in seq_ids.iter().enumerate() {
                let acc = n_acc[u];
                if acc < k_len {
                    any = true;
                    let su = sid as usize;
                    let idx = acc - 1;
                    let kv_src_off = (u * 8 + idx) * (kv_n as usize);
                    let e_src_off = (u * 8 + idx) * (e_n as usize);
                    let kv_dst_off = su * (kv_n as usize);
                    let e_dst_off = su * (e_n as usize);
                    for sl in &self.slots {
                        ops::ink_copy_f32(self.stream, (sl.k_snap.ptr as *const f32).add(kv_src_off), (sl.k_hist.ptr as *mut f32).add(kv_dst_off), kv_n);
                        ops::ink_copy_f32(self.stream, (sl.v_snap.ptr as *const f32).add(kv_src_off), (sl.v_hist.ptr as *mut f32).add(kv_dst_off), kv_n);
                        ops::ink_copy_f32(self.stream, (sl.attn_post_snap.ptr as *const f32).add(e_src_off), (sl.attn_post_hist.ptr as *mut f32).add(e_dst_off), e_n);
                        ops::ink_copy_f32(self.stream, (sl.ffn_post_snap.ptr as *const f32).add(e_src_off), (sl.ffn_post_hist.ptr as *mut f32).add(e_dst_off), e_n);
                    }
                    self.seq_pos[su] -= (k_len - acc) as i32;
                    if su == 0 { self.pos = self.seq_pos[0]; }
                }
            }
            if any {
                rad_hip::hipStreamSynchronize(self.stream);
            }
        }
        Ok(())
    }

    unsafe fn enqueue_layer_batch_ex(&self, s: HipStream, il: usize, ip: usize, n_blocks: &mut usize, loop_idx: Option<usize>, b: usize, seq_opt: Option<usize>, multi_k: usize) {
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
        ops::ink_quantize_q8_1(s, p0, xq0, e, bi);
        let (q1, q2) = if p0 == p1 && p0 == p2 {
            (xq0 as *const core::ffi::c_void, xq0 as *const core::ffi::c_void)
        } else {
            ops::ink_quantize_q8_1(s, p1, xq1, e, bi);
            ops::ink_quantize_q8_1(s, p2, xq2, e, bi);
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
        if let Some(seq) = seq_opt {
            let seq_kv_off = seq * self.cfg.n_head_kv * sl.max_ctx * self.cfg.dh;
            let seq_kh_off = seq * (kv_dim as usize) * 3;
            let d_pos0 = self.bsc.d_pos0.ptr as *const i32;
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
                ops::ink_attn_batch_gated(
                    s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                    (sl.k_cache.ptr as *const u16).add(seq_kv_off),
                    (sl.v_cache.ptr as *const u16).add(seq_kv_off),
                    gate_out, d_pos0, attn_out,
                    self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 },
                    sl.max_ctx as i32, bi,
                );
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
            ops::ink_attn_multi_gated_q8(
                s, q, r, sinks, l.attn_rel_bias.ptr as *const f32,
                sl.k_cache.ptr as *const u16, sl.v_cache.ptr as *const u16,
                gate_out, self.bsc.d_seq_ids.ptr as *const i32, self.bsc.d_positions.ptr as *const i32,
                self.bsc.xq.ptr,
                self.cfg.n_head as i32, self.cfg.n_head_kv as i32, self.cfg.dh as i32, self.cfg.d_rel as i32,
                self.cfg.rel_extent as i32, self.cfg.swa_window as i32, if sl.is_swa { 1 } else { 0 },
                sl.max_ctx as i32, bi,
            );
        }

        let o_raw = self.bsc.o_raw.ptr as *mut f32;
        if seq_opt.is_some() {
            ops::ink_quantize_q8_1(s, attn_out, self.bsc.xq.ptr, q_dim, bi);
        }
        match l.wo.ty {
            WeightType::Q4_0 if b >= 12 => ops::ink_gemm_q4_0_wmma(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
            WeightType::Q4_0 => ops::ink_gemm_q4_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
            _ => ops::ink_gemm_q8_0_sudot4(s, l.wo.buf.ptr as *const u8, self.bsc.xq.ptr, o_raw, e, q_dim, bi),
        };

        let block_dst = if ip % self.cfg.attn_res_block == 0 {
            let dst = blocks.add((*n_blocks) * MAX_BATCH * self.cfg.n_embd);
            *n_blocks += 1;
            dst
        } else {
            std::ptr::null_mut()
        };
        if let Some(seq) = seq_opt {
            let seq_eh_off = seq * self.cfg.n_embd * 3;
            ops::ink_shortconv_res_batch(s, o_raw, l.attn_post_conv.ptr as *const f32, (sl.attn_post_hist.ptr as *mut f32).add(seq_eh_off), cur, block_dst, ffn_inp, e, bi);
        } else if multi_k > 1 {
            ops::ink_shortconv_res_multi_snap(s, o_raw, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, sl.attn_post_snap.ptr as *mut f32, cur, block_dst, ffn_inp, self.bsc.d_seq_ids.ptr as *const i32, e, (b / multi_k) as i32, multi_k as i32);
        } else {
            ops::ink_shortconv_res_multi(s, o_raw, l.attn_post_conv.ptr as *const f32, sl.attn_post_hist.ptr as *mut f32, cur, block_dst, ffn_inp, self.bsc.d_seq_ids.ptr as *const i32, e, bi);
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
        ops::ink_moe_router_batch(s, router_logits, bias_ptr, self.bsc.wtop.ptr as *mut f32, self.bsc.itop.ptr as *mut i32, self.cfg.n_expert as i32, self.cfg.n_expert_used as i32, self.cfg.route_scale, bi);

        let gu_sc = l.ffn_gate_up_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let dn_sc = l.ffn_down_scale.as_ref().map(|b| b.ptr as *const f32).unwrap_or(std::ptr::null());
        let moe_act = self.bsc.moe_act.ptr as *mut f32;
        let moe_raw = self.bsc.moe_raw.ptr as *mut f32;

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
                ops::ink_quantize_q8_1(s, ffn_normed, self.bsc.xq.ptr, e, bi);
                ops::ink_moe_build_groups(
                    s, self.bsc.itop.ptr as *const i32, self.bsc.wtop.ptr as *const f32, dn_sc,
                    self.bsc.exp_counts.ptr as *mut i32, self.bsc.exp_offsets.ptr as *mut i32,
                    self.bsc.tok_sorted.ptr as *mut i32, self.bsc.slot_sorted.ptr as *mut i32,
                    self.bsc.wt_sorted.ptr as *mut f32, self.cfg.n_expert_used as i32, bi,
                );
                if b >= 48 {
                    ops::ink_moe_gu_q4_0_wmma(
                        s, l.ffn_gate_up_exps.buf.ptr as *const u8, gu_sc, self.bsc.xq.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        moe_act, self.cfg.n_ff_exp as i32, e, self.cfg.n_expert_used as i32,
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
                    ops::ink_quantize_q8_1(s, moe_act, self.bsc.act_q.ptr, self.cfg.n_ff_exp as i32, bi * self.cfg.n_expert_used as i32);
                    ops::ink_moe_dn_q4_0_wmma(
                        s, l.ffn_down_exps.buf.ptr as *const u8, self.bsc.act_q.ptr,
                        self.bsc.exp_counts.ptr as *const i32, self.bsc.exp_offsets.ptr as *const i32,
                        self.bsc.tok_sorted.ptr as *const i32, self.bsc.slot_sorted.ptr as *const i32,
                        self.bsc.wt_sorted.ptr as *const f32, self.bsc.moe_dn_slots.ptr as *mut f32, moe_raw,
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
            ops::ink_shortconv_res_batch(s, moe_raw, l.ffn_post_conv.ptr as *const f32, (sl.ffn_post_hist.ptr as *mut f32).add(seq_eh_off), ffn_inp, std::ptr::null_mut(), cur, e, bi);
        } else if multi_k > 1 {
            ops::ink_shortconv_res_multi_snap(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, sl.ffn_post_snap.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, self.bsc.d_seq_ids.ptr as *const i32, e, (b / multi_k) as i32, multi_k as i32);
        } else {
            ops::ink_shortconv_res_multi(s, moe_raw, l.ffn_post_conv.ptr as *const f32, sl.ffn_post_hist.ptr as *mut f32, ffn_inp, std::ptr::null_mut(), cur, self.bsc.d_seq_ids.ptr as *const i32, e, bi);
        }
    }

    pub unsafe fn enqueue_forward_batch_seq(&self, s: HipStream, seq: usize, b: usize, all_logits: bool) {
        self.enqueue_forward_ex(s, b, all_logits, Some(seq), 1);
    }

    pub unsafe fn enqueue_forward_multi(&self, s: HipStream, b: usize) {
        self.enqueue_forward_ex(s, b, true, None, 1);
    }

    unsafe fn enqueue_forward_ex(&self, s: HipStream, b: usize, all_logits: bool, seq_opt: Option<usize>, multi_k: usize) {
        let e = self.cfg.n_embd as i32;
        let mb = MAX_BATCH as i32;
        let bi = b as i32;
        let be = (b * self.cfg.n_embd) as i32;
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
            self.enqueue_layer_batch_ex(s, il, il, &mut n_blocks, None, b, seq_opt, multi_k);
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
                    self.enqueue_layer_batch_ex(s, il, ip, &mut n_blocks, None, b, seq_opt, multi_k);
                } else {
                    let li = ip - self.cfg.core_start;
                    let ll = &self.loop_layers[li];
                    ops::ink_mix_batch(s, bank, bank_len as i32, mb, bank_out, ll.msg_query.ptr as *const f32, std::ptr::null(), std::ptr::null(), std::ptr::null(), msgs, e, 4, bi, self.cfg.eps);
                    self.enqueue_layer_batch_ex(s, il, ip, &mut n_blocks, Some(li), b, seq_opt, multi_k);
                }
                visit += 1;
                if visit == 23 { ops::ink_copy_f32(s, cur, feats.add(MAX_BATCH * self.cfg.n_embd), be); }
                else if visit == 41 { ops::ink_copy_f32(s, cur, feats.add(2 * MAX_BATCH * self.cfg.n_embd), be); }
            }
        }
        for il in self.cfg.core_end..self.cfg.n_phys {
            self.enqueue_layer_batch_ex(s, il, il, &mut n_blocks, None, b, seq_opt, multi_k);
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
        let l = &self.layers[6];
        let sl = &self.slots[6];
        let ll = &self.loop_layers[0];
        let iters = 200;

        let bench = |name: &str, mult: f64, f: &dyn Fn()| {
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
        let mut n_blocks = 0usize;
        let runs = 5;
        for _ in 0..runs {
            n_blocks = 0;
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

    pub fn read_logits(&self) -> Result<Vec<f32>> {
        let n = self.cfg.vocab_unpadded;
        let mut raw = vec![0u8; n * 4];
        let slice_len = raw.len();
        let rc = unsafe {
            rad_hip::hipMemcpy(
                raw.as_mut_ptr() as *mut core::ffi::c_void,
                self.sc.logits.ptr,
                slice_len,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
