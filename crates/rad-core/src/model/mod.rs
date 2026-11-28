use crate::ops;
use anyhow::{anyhow, bail, Result};
use rad_gguf::GgufFile;
use rad_hip::{CapturedGraph, DevBuf, HipError, HipStream};
use std::path::Path;


mod decode;
mod multi;
mod profile;

fn he(e: HipError) -> anyhow::Error {
    anyhow!("hip error rc={e}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightType {
    F32,
    Bf16,
    Q8_0,
    Q8_0Soa,
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
            WeightType::Q8_0 | WeightType::Q8_0Soa => {
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
    k_scale: Option<DevBuf>,
    v_scale: Option<DevBuf>,
    k_hist: DevBuf,
    v_hist: DevBuf,
    attn_post_hist: DevBuf,
    ffn_post_hist: DevBuf,
    k_snap: DevBuf,
    v_snap: DevBuf,
    attn_post_snap: DevBuf,
    ffn_post_snap: DevBuf,
}

#[allow(dead_code)]
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
    topk_vals: DevBuf,
    topk_ids: DevBuf,
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
    pub multi_verify_cap: usize,
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
    seq_verify_graphs: std::collections::HashMap<(usize, usize), CapturedGraph>,
    multi_graphs: std::collections::HashMap<usize, CapturedGraph>,
    multi_verify_graphs: std::collections::HashMap<(usize, usize), CapturedGraph>,
    pub pos: i32,
    pub seq_pos: Vec<i32>,
    pub seq_tokens: Vec<Vec<i32>>,
    pub paged_kv: crate::paged_kv::PagedKvManager,
    d_rewind_ptrs: DevBuf,
    d_rw_u: DevBuf,
    d_rw_seq: DevBuf,
    d_rw_idx: DevBuf,
    h_io_pinned: rad_hip::PinnedBuf,
    last_logits_in_sc: bool,
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
        let mut est_kv_bytes = 0usize;
        for il in 0..n_slots {
            let ip = if il < cfg.n_phys {
                il
            } else {
                cfg.core_start + ((il - cfg.n_phys) % (cfg.core_end - cfg.core_start))
            };
            let is_swa = cfg.is_swa[ip];
            let slot_ctx = if is_swa { (cfg.swa_window + 16).min(max_ctx) } else { max_ctx };
            let kv_bytes = if is_swa {
                max_seqs * cfg.n_head_kv * slot_ctx * cfg.dh * 2
            } else {
                max_seqs * cfg.n_head_kv * slot_ctx * cfg.dh
            };
            est_kv_bytes += kv_bytes * 2;
            if !is_swa {
                est_kv_bytes += max_seqs * cfg.n_head_kv * slot_ctx * 4;
            }
        }
        let dev_id = rad_hip::get_device().unwrap_or(0);
        let free_vram = rad_hip::mem_info(dev_id).map(|(f, _)| f).unwrap_or(usize::MAX);
        let rem_after_kv = free_vram.saturating_sub(est_kv_bytes);
        let snap_cap = if max_seqs >= 8 && max_ctx <= 32768 && rem_after_kv >= 5_500_000_000 {
            1024
        } else if max_seqs >= 8 && max_ctx <= 65536 && rem_after_kv >= 4_000_000_000 {
            512
        } else if max_seqs >= 4 && max_ctx <= 65536 && rem_after_kv >= 3_000_000_000 {
            256
        } else {
            128
        };
        for il in 0..n_slots {
            let ip = if il < cfg.n_phys {
                il
            } else {
                cfg.core_start + ((il - cfg.n_phys) % (cfg.core_end - cfg.core_start))
            };
            let is_swa = cfg.is_swa[ip];
            let slot_ctx = if is_swa { (cfg.swa_window + 16).min(max_ctx) } else { max_ctx };
            let kv_bytes = if is_swa {
                max_seqs * cfg.n_head_kv * slot_ctx * cfg.dh * 2
            } else {
                max_seqs * cfg.n_head_kv * slot_ctx * cfg.dh
            };
            let mut k_cache = DevBuf::new(kv_bytes).map_err(he)?;
            let mut v_cache = DevBuf::new(kv_bytes).map_err(he)?;
            let (k_scale, v_scale) = if is_swa {
                (None, None)
            } else {
                let sc_bytes = max_seqs * cfg.n_head_kv * slot_ctx * 2;
                let mut ks = DevBuf::new(sc_bytes).map_err(he)?;
                let mut vs = DevBuf::new(sc_bytes).map_err(he)?;
                ks.zero().map_err(he)?;
                vs.zero().map_err(he)?;
                (Some(ks), Some(vs))
            };
            let mut k_hist = DevBuf::new(max_seqs * kv_dim * 3 * 4).map_err(he)?;
            let mut v_hist = DevBuf::new(max_seqs * kv_dim * 3 * 4).map_err(he)?;
            let mut attn_post_hist = DevBuf::new(max_seqs * cfg.n_embd * 3 * 4).map_err(he)?;
            let mut ffn_post_hist = DevBuf::new(max_seqs * cfg.n_embd * 3 * 4).map_err(he)?;
            let k_snap = DevBuf::new(snap_cap * kv_dim * 3 * 4).map_err(he)?;
            let v_snap = DevBuf::new(snap_cap * kv_dim * 3 * 4).map_err(he)?;
            let attn_post_snap = DevBuf::new(snap_cap * cfg.n_embd * 3 * 4).map_err(he)?;
            let ffn_post_snap = DevBuf::new(snap_cap * cfg.n_embd * 3 * 4).map_err(he)?;
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
                k_scale,
                v_scale,
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
        let multi_verify_cap = snap_cap;
        let bsc = BatchScratch {
            d_toks: DevBuf::new(mb * 4).map_err(he)?,
            d_pos0: DevBuf::new(4).map_err(he)?,
            d_seq_ids: DevBuf::new(mb * 4).map_err(he)?,
            d_positions: DevBuf::new(mb * 4).map_err(he)?,
            d_out_toks: DevBuf::new(mb * 4).map_err(he)?,
            topk_vals: DevBuf::new(multi_verify_cap.max(128) * 64 * 4).map_err(he)?,
            topk_ids: DevBuf::new(multi_verify_cap.max(128) * 64 * 4).map_err(he)?,
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
            logits: DevBuf::new(multi_verify_cap.max(max_seqs) * cfg.vocab_padded * 4).map_err(he)?,
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

        let mut rw_ptrs: Vec<u64> = Vec::with_capacity(slots.len() * 8);
        for sl in &slots {
            rw_ptrs.push(sl.k_snap.ptr as u64);
            rw_ptrs.push(sl.k_hist.ptr as u64);
            rw_ptrs.push(sl.v_snap.ptr as u64);
            rw_ptrs.push(sl.v_hist.ptr as u64);
            rw_ptrs.push(sl.attn_post_snap.ptr as u64);
            rw_ptrs.push(sl.attn_post_hist.ptr as u64);
            rw_ptrs.push(sl.ffn_post_snap.ptr as u64);
            rw_ptrs.push(sl.ffn_post_hist.ptr as u64);
        }
        let mut d_rewind_ptrs = DevBuf::new(rw_ptrs.len() * 8).map_err(he)?;
        let rw_bytes = unsafe { std::slice::from_raw_parts(rw_ptrs.as_ptr() as *const u8, rw_ptrs.len() * 8) };
        d_rewind_ptrs.copy_from(rw_bytes, None).map_err(he)?;
        let d_rw_u = DevBuf::new(MAX_BATCH * 4).map_err(he)?;
        let d_rw_seq = DevBuf::new(MAX_BATCH * 4).map_err(he)?;
        let d_rw_idx = DevBuf::new(MAX_BATCH * 4).map_err(he)?;

        Ok(Self {
            cfg,
            max_ctx,
            max_seqs,
            multi_verify_cap,
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
            seq_verify_graphs: std::collections::HashMap::new(),
            multi_graphs: std::collections::HashMap::new(),
            multi_verify_graphs: std::collections::HashMap::new(),
            pos: 0,
            seq_pos: vec![0; max_seqs],
            seq_tokens: vec![Vec::new(); max_seqs],
            paged_kv: crate::paged_kv::PagedKvManager::new(
                max_seqs,
                max_ctx,
                std::env::var("RAD_KV_BLOCK_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(16),
                std::env::var("RAD_KV_BLOCKS").ok().and_then(|s| s.parse().ok()),
            ),
            d_rewind_ptrs,
            d_rw_u,
            d_rw_seq,
            d_rw_idx,
            h_io_pinned: rad_hip::PinnedBuf::new(4096).map_err(he)?,
            last_logits_in_sc: false,
            b1_sudot4,
        })
    }

    pub fn read_batch_logits(&self, tok_idx: usize) -> Result<Vec<f32>> {
        if self.last_logits_in_sc && tok_idx == 0 {
            return self.read_logits();
        }
        let n = self.cfg.vocab_unpadded;
        let stride = self.cfg.vocab_padded;
        let mut raw = vec![0u8; n * 4];
        let slice_len = raw.len();
        let rc = unsafe {
            let src = (self.bsc.logits.ptr as *const f32).add(tok_idx * stride) as *const core::ffi::c_void;
            rad_hip::hipMemcpy(
                raw.as_mut_ptr() as *mut core::ffi::c_void,
                src,
                slice_len,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        let cap = self.cfg.final_softcap;
        Ok(raw.chunks_exact(4).map(|c| {
            let v = f32::from_le_bytes(c.try_into().unwrap());
            if cap > 0.0 { cap * (v / cap).tanh() } else { v }
        }).collect())
    }

    pub fn read_batch_topk64(&self, b: usize) -> Result<(Vec<f32>, Vec<i32>)> {
        let count = b * 64;
        let mut vals = vec![0.0f32; count];
        let mut ids = vec![0i32; count];
        unsafe {
            ops::ink_logits_topk64_batch(
                self.stream,
                self.bsc.logits.ptr as *const f32,
                self.cfg.vocab_padded as i32,
                self.cfg.vocab_unpadded as i32,
                self.bsc.topk_vals.ptr as *mut f32,
                self.bsc.topk_ids.ptr as *mut i32,
                b as i32,
            );
            let rc1 = rad_hip::hipMemcpyAsync(
                vals.as_mut_ptr() as *mut core::ffi::c_void,
                self.bsc.topk_vals.ptr,
                count * 4,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
                self.stream,
            );
            if rc1 != rad_hip::HIP_SUCCESS { return Err(he(rc1)); }
            let rc2 = rad_hip::hipMemcpyAsync(
                ids.as_mut_ptr() as *mut core::ffi::c_void,
                self.bsc.topk_ids.ptr,
                count * 4,
                rad_hip::HIP_MEMCPY_DEVICE_TO_HOST,
                self.stream,
            );
            if rc2 != rad_hip::HIP_SUCCESS { return Err(he(rc2)); }
            rad_hip::hipStreamSynchronize(self.stream);
        }
        let cap = self.cfg.final_softcap;
        if cap > 0.0 {
            for v in &mut vals {
                *v = cap * (*v / cap).tanh();
            }
        }
        Ok((vals, ids))
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
