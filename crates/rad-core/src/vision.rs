use crate::ops;
use anyhow::{anyhow, Result};
use base64::Engine;
use image::imageops::FilterType;
use rad_gguf::GgufFile;
use rad_hip::{DevBuf, HipError, HipStream};
use std::path::Path;

fn he(e: HipError) -> anyhow::Error {
    anyhow!("hip error rc={e}")
}

fn load_raw(g: &GgufFile, name: &str) -> Result<DevBuf> {
    let bytes = g.tensor_bytes(name).ok_or_else(|| anyhow!("missing vision tensor {name}"))?;
    let mut b = DevBuf::new(bytes.len()).map_err(he)?;
    b.copy_from(bytes, None).map_err(he)?;
    Ok(b)
}

struct VitLayer {
    ln1_w: DevBuf,
    ln1_b: DevBuf,
    ln2_w: DevBuf,
    ln2_b: DevBuf,
    ls1_w: DevBuf,
    ls2_w: DevBuf,
    wq: DevBuf,
    bq: DevBuf,
    wk: DevBuf,
    bk: DevBuf,
    wv: DevBuf,
    bv: DevBuf,
    wo: DevBuf,
    bo: DevBuf,
    ffn_up_w: DevBuf,
    ffn_up_b: DevBuf,
    ffn_dn_w: DevBuf,
    ffn_dn_b: DevBuf,
}

pub struct VisionEncoder {
    pub image_size: usize,
    pub patch_size: usize,
    pub grid_size: usize,
    pub n_patches: usize,
    pub n_tokens: usize,
    pub out_tokens: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub dh: usize,
    pub n_layers: usize,
    pub proj_dim: usize,
    pub merge_size: usize,
    pub eps: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    class_embd: DevBuf,
    class_pos_embd: DevBuf,
    patch_embd: DevBuf,
    pos_embd: DevBuf,
    pre_ln_w: DevBuf,
    pre_ln_b: DevBuf,
    layers: Vec<VitLayer>,
    mm1: DevBuf,
    mm2: DevBuf,
    patches_buf: DevBuf,
    x: DevBuf,
    h: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    attn_out: DevBuf,
    o_out: DevBuf,
    ff_mid: DevBuf,
    merged: DevBuf,
    proj_mid: DevBuf,
    proj_out: DevBuf,
    stream: HipStream,
}

unsafe impl Send for VisionEncoder {}

impl VisionEncoder {
    pub fn load(path: &Path) -> Result<Self> {
        let g = GgufFile::open(path)?;
        let image_size = g.kv_u64("clip.vision.image_size").unwrap_or(640) as usize;
        let patch_size = g.kv_u64("clip.vision.patch_size").unwrap_or(20) as usize;
        let n_embd = g.kv_u64("clip.vision.embedding_length").unwrap_or(1024) as usize;
        let n_ff = g.kv_u64("clip.vision.feed_forward_length").unwrap_or(4096) as usize;
        let n_layers = g.kv_u64("clip.vision.block_count").unwrap_or(23) as usize;
        let n_head = g.kv_u64("clip.vision.attention.head_count").unwrap_or(16) as usize;
        let proj_dim = g.kv_u64("clip.vision.projection_dim").unwrap_or(2304) as usize;
        let merge_size = g.kv_u64("clip.vision.spatial_merge_size").unwrap_or(2) as usize;
        let eps = g.kv_f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-5);
        let grid_size = image_size / patch_size;
        let n_patches = grid_size * grid_size;
        let n_tokens = n_patches + 1;
        let out_tokens = (grid_size / merge_size) * (grid_size / merge_size);
        let dh = n_embd / n_head;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            layers.push(VitLayer {
                ln1_w: load_raw(&g, &p("ln1.weight"))?,
                ln1_b: load_raw(&g, &p("ln1.bias"))?,
                ln2_w: load_raw(&g, &p("ln2.weight"))?,
                ln2_b: load_raw(&g, &p("ln2.bias"))?,
                ls1_w: load_raw(&g, &p("ls1.weight"))?,
                ls2_w: load_raw(&g, &p("ls2.weight"))?,
                wq: load_raw(&g, &p("attn_q.weight"))?,
                bq: load_raw(&g, &p("attn_q.bias"))?,
                wk: load_raw(&g, &p("attn_k.weight"))?,
                bk: load_raw(&g, &p("attn_k.bias"))?,
                wv: load_raw(&g, &p("attn_v.weight"))?,
                bv: load_raw(&g, &p("attn_v.bias"))?,
                wo: load_raw(&g, &p("attn_out.weight"))?,
                bo: load_raw(&g, &p("attn_out.bias"))?,
                ffn_up_w: load_raw(&g, &p("ffn_up.weight"))?,
                ffn_up_b: load_raw(&g, &p("ffn_up.bias"))?,
                ffn_dn_w: load_raw(&g, &p("ffn_down.weight"))?,
                ffn_dn_b: load_raw(&g, &p("ffn_down.bias"))?,
            });
        }

        extern "C" {
            fn hipStreamCreate(s: *mut HipStream) -> i32;
        }
        let mut stream: HipStream = std::ptr::null_mut();
        unsafe { hipStreamCreate(&mut stream) };

        let patch_dim = 3 * patch_size * patch_size;
        let merged_dim = merge_size * merge_size * n_embd;

        Ok(Self {
            image_size,
            patch_size,
            grid_size,
            n_patches,
            n_tokens,
            out_tokens,
            n_embd,
            n_ff,
            n_head,
            dh,
            n_layers,
            proj_dim,
            merge_size,
            eps,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            class_embd: load_raw(&g, "v.class_embd")?,
            class_pos_embd: load_raw(&g, "v.class_position_embd.weight")?,
            patch_embd: load_raw(&g, "v.patch_embd.weight")?,
            pos_embd: load_raw(&g, "v.position_embd.weight")?,
            pre_ln_w: load_raw(&g, "v.pre_ln.weight")?,
            pre_ln_b: load_raw(&g, "v.pre_ln.bias")?,
            layers,
            mm1: load_raw(&g, "mm.1.weight")?,
            mm2: load_raw(&g, "mm.2.weight")?,
            patches_buf: DevBuf::new(n_patches * patch_dim * 4).map_err(he)?,
            x: DevBuf::new(n_tokens * n_embd * 4).map_err(he)?,
            h: DevBuf::new(n_tokens * n_embd * 4).map_err(he)?,
            q: DevBuf::new(n_tokens * n_embd * 4).map_err(he)?,
            k: DevBuf::new(n_tokens * n_embd * 4).map_err(he)?,
            v: DevBuf::new(n_tokens * n_embd * 4).map_err(he)?,
            attn_out: DevBuf::new(n_tokens * n_embd * 4).map_err(he)?,
            o_out: DevBuf::new(n_tokens * n_embd * 4).map_err(he)?,
            ff_mid: DevBuf::new(n_tokens * n_ff * 4).map_err(he)?,
            merged: DevBuf::new(out_tokens * merged_dim * 4).map_err(he)?,
            proj_mid: DevBuf::new(out_tokens * merged_dim * 4).map_err(he)?,
            proj_out: DevBuf::new(out_tokens * proj_dim * 4).map_err(he)?,
            stream,
        })
    }

    pub fn encode_image_path(&mut self, path: &Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        self.encode_image_bytes(&bytes)
    }

    pub fn encode_data_url(&mut self, url: &str) -> Result<Vec<f32>> {
        if let Some((_, b64)) = url.split_once("base64,") {
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim())?;
            return self.encode_image_bytes(&bytes);
        }
        let p = Path::new(url.strip_prefix("file://").unwrap_or(url));
        if p.exists() {
            return self.encode_image_path(p);
        }
        Err(anyhow!("unsupported image url (provide local path or data:image/...;base64,...)"))
    }

    pub fn encode_image_bytes(&mut self, bytes: &[u8]) -> Result<Vec<f32>> {
        let img = image::load_from_memory(bytes)?
            .resize_exact(self.image_size as u32, self.image_size as u32, FilterType::CatmullRom)
            .to_rgb8();
        let raw = img.as_raw();
        let ps = self.patch_size;
        let gs = self.grid_size;
        let patch_dim = 3 * ps * ps;
        let mut patches = vec![0.0f32; self.n_patches * patch_dim];
        for py in 0..gs {
            for px in 0..gs {
                let p_idx = py * gs + px;
                let base = p_idx * patch_dim;
                for c in 0..3 {
                    for dy in 0..ps {
                        let y = py * ps + dy;
                        for dx in 0..ps {
                            let x = px * ps + dx;
                            let pix = raw[(y * self.image_size + x) * 3 + c] as f32 / 255.0;
                            let norm = (pix - self.mean[c]) / self.std[c];
                            patches[base + (c * ps + dy) * ps + dx] = norm;
                        }
                    }
                }
            }
        }
        self.encode_patches(&patches)
    }

    fn encode_patches(&mut self, patches: &[f32]) -> Result<Vec<f32>> {
        let e = self.n_embd as i32;
        let nt = self.n_tokens as i32;
        let np = self.n_patches as i32;
        let patch_dim = (3 * self.patch_size * self.patch_size) as i32;
        let merged_dim = (self.merge_size * self.merge_size * self.n_embd) as i32;
        let pbytes = unsafe { std::slice::from_raw_parts(patches.as_ptr() as *const u8, patches.len() * 4) };
        self.patches_buf.copy_from(pbytes, Some(self.stream)).map_err(he)?;

        unsafe {
            ops::ink_vec_add(self.stream, self.class_embd.ptr as *const f32, self.class_pos_embd.ptr as *const f32, self.x.ptr as *mut f32, e);
            let x_patches = (self.x.ptr as *mut f32).add(self.n_embd);
            ops::ink_gemm_f16_bias(self.stream, self.patch_embd.ptr as *const u16, std::ptr::null(), self.patches_buf.ptr as *const f32, x_patches, e, patch_dim, np);
            ops::ink_vec_add(self.stream, x_patches as *const f32, self.pos_embd.ptr as *const f32, x_patches, np * e);
            ops::ink_layernorm_f32(self.stream, self.x.ptr as *const f32, self.pre_ln_w.ptr as *const f32, self.pre_ln_b.ptr as *const f32, self.x.ptr as *mut f32, e, nt, self.eps);

            for l in &self.layers {
                ops::ink_layernorm_f32(self.stream, self.x.ptr as *const f32, l.ln1_w.ptr as *const f32, l.ln1_b.ptr as *const f32, self.h.ptr as *mut f32, e, nt, self.eps);
                ops::ink_gemm_f16_bias(self.stream, l.wq.ptr as *const u16, l.bq.ptr as *const f32, self.h.ptr as *const f32, self.q.ptr as *mut f32, e, e, nt);
                ops::ink_gemm_f16_bias(self.stream, l.wk.ptr as *const u16, l.bk.ptr as *const f32, self.h.ptr as *const f32, self.k.ptr as *mut f32, e, e, nt);
                ops::ink_gemm_f16_bias(self.stream, l.wv.ptr as *const u16, l.bv.ptr as *const f32, self.h.ptr as *const f32, self.v.ptr as *mut f32, e, e, nt);
                ops::ink_vit_attn_f32(self.stream, self.q.ptr as *const f32, self.k.ptr as *const f32, self.v.ptr as *const f32, self.attn_out.ptr as *mut f32, nt, self.n_head as i32, self.dh as i32);
                ops::ink_gemm_f16_bias(self.stream, l.wo.ptr as *const u16, l.bo.ptr as *const f32, self.attn_out.ptr as *const f32, self.o_out.ptr as *mut f32, e, e, nt);
                ops::ink_ls_res_add_f32(self.stream, self.x.ptr as *mut f32, self.o_out.ptr as *const f32, l.ls1_w.ptr as *const f32, e, nt);

                ops::ink_layernorm_f32(self.stream, self.x.ptr as *const f32, l.ln2_w.ptr as *const f32, l.ln2_b.ptr as *const f32, self.h.ptr as *mut f32, e, nt, self.eps);
                ops::ink_gemm_f16_bias(self.stream, l.ffn_up_w.ptr as *const u16, l.ffn_up_b.ptr as *const f32, self.h.ptr as *const f32, self.ff_mid.ptr as *mut f32, self.n_ff as i32, e, nt);
                ops::ink_gelu_f32(self.stream, self.ff_mid.ptr as *mut f32, self.n_tokens * self.n_ff);
                ops::ink_gemm_f16_bias(self.stream, l.ffn_dn_w.ptr as *const u16, l.ffn_dn_b.ptr as *const f32, self.ff_mid.ptr as *const f32, self.o_out.ptr as *mut f32, e, self.n_ff as i32, nt);
                ops::ink_ls_res_add_f32(self.stream, self.x.ptr as *mut f32, self.o_out.ptr as *const f32, l.ls2_w.ptr as *const f32, e, nt);
            }

            ops::ink_vit_merge_2x2(self.stream, (self.x.ptr as *const f32).add(self.n_embd), self.merged.ptr as *mut f32, self.grid_size as i32, self.grid_size as i32, e);
            ops::ink_gemm_f16_bias(self.stream, self.mm1.ptr as *const u16, std::ptr::null(), self.merged.ptr as *const f32, self.proj_mid.ptr as *mut f32, merged_dim, merged_dim, self.out_tokens as i32);
            ops::ink_gelu_f32(self.stream, self.proj_mid.ptr as *mut f32, self.out_tokens * (merged_dim as usize));
            ops::ink_gemm_f16_bias(self.stream, self.mm2.ptr as *const u16, std::ptr::null(), self.proj_mid.ptr as *const f32, self.proj_out.ptr as *mut f32, self.proj_dim as i32, merged_dim, self.out_tokens as i32);
            rad_hip::hipStreamSynchronize(self.stream);
        }

        let total_f32 = self.out_tokens * self.proj_dim;
        let mut out = vec![0.0f32; total_f32];
        let rc = unsafe {
            rad_hip::hipMemcpy(out.as_mut_ptr() as *mut core::ffi::c_void, self.proj_out.ptr, total_f32 * 4, rad_hip::HIP_MEMCPY_DEVICE_TO_HOST)
        };
        if rc != rad_hip::HIP_SUCCESS {
            return Err(he(rc));
        }
        Ok(out)
    }
}
