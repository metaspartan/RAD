use anyhow::{anyhow, Result};
use rad_cuda::{CudaBuf, CudaFail};
use crate::config::DeepseekV41Config;
use crate::safetensors::SafetensorsIndex;

pub struct Exl3Linear {
    pub trellis: CudaBuf,
    pub suh: CudaBuf,
    pub svh: CudaBuf,
    pub k: usize,
    pub n: usize,
    pub bits: usize,
}

impl Exl3Linear {
    pub fn load(idx: &mut SafetensorsIndex, prefix: &str) -> Result<Self> {
        let (k, n, bits) = idx.exl3_shape_and_bits(prefix)?;
        let (_, trel_bytes) = idx.get_tensor(&format!("{prefix}.trellis"))?;
        let mut trellis = CudaBuf::new(trel_bytes.len()).map_err(CudaFail)?;
        trellis.copy_from(trel_bytes, None).map_err(CudaFail)?;

        let (_, suh_bytes) = idx.get_tensor(&format!("{prefix}.suh"))?;
        let mut suh = CudaBuf::new(suh_bytes.len()).map_err(CudaFail)?;
        suh.copy_from(suh_bytes, None).map_err(CudaFail)?;

        let (_, svh_bytes) = idx.get_tensor(&format!("{prefix}.svh"))?;
        let mut svh = CudaBuf::new(svh_bytes.len()).map_err(CudaFail)?;
        svh.copy_from(svh_bytes, None).map_err(CudaFail)?;

        Ok(Self { trellis, suh, svh, k, n, bits })
    }

    pub fn load_col_shard(
        idx: &mut SafetensorsIndex,
        prefix: &str,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        if world <= 1 {
            return Self::load(idx, prefix);
        }
        let (k, n, bits) = idx.exl3_shape_and_bits(prefix)?;
        let n_local = n / world;
        if n_local % 128 != 0 {
            return Err(anyhow!("{prefix} col shard n_local={n_local} not multiple of 128"));
        }
        let k_tiles = k / 16;
        let n_tiles = n / 16;
        let n_tiles_local = n_local / 16;
        let n_tile_start = rank * n_tiles_local;
        let tile_bytes = 16 * bits * 2;
        let (_, trel_all) = idx.get_tensor(&format!("{prefix}.trellis"))?;
        let mut trel_host = vec![0u8; k_tiles * n_tiles_local * tile_bytes];
        for kt in 0..k_tiles {
            let src_off = (kt * n_tiles + n_tile_start) * tile_bytes;
            let dst_off = kt * n_tiles_local * tile_bytes;
            let row_len = n_tiles_local * tile_bytes;
            trel_host[dst_off..dst_off + row_len].copy_from_slice(&trel_all[src_off..src_off + row_len]);
        }
        let mut trellis = CudaBuf::new(trel_host.len()).map_err(CudaFail)?;
        trellis.copy_from(&trel_host, None).map_err(CudaFail)?;

        let (_, suh_bytes) = idx.get_tensor(&format!("{prefix}.suh"))?;
        let mut suh = CudaBuf::new(suh_bytes.len()).map_err(CudaFail)?;
        suh.copy_from(suh_bytes, None).map_err(CudaFail)?;

        let (_, svh_all) = idx.get_tensor(&format!("{prefix}.svh"))?;
        let svh_off = rank * n_local * 2;
        let mut svh = CudaBuf::new(n_local * 2).map_err(CudaFail)?;
        svh.copy_from(&svh_all[svh_off..svh_off + n_local * 2], None).map_err(CudaFail)?;

        Ok(Self { trellis, suh, svh, k, n: n_local, bits })
    }

    pub fn load_row_shard(
        idx: &mut SafetensorsIndex,
        prefix: &str,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        if world <= 1 {
            return Self::load(idx, prefix);
        }
        let (k, n, bits) = idx.exl3_shape_and_bits(prefix)?;
        let k_local = k / world;
        if k_local % 128 != 0 {
            return Err(anyhow!("{prefix} row shard k_local={k_local} not multiple of 128"));
        }
        let k_tiles_local = k_local / 16;
        let n_tiles = n / 16;
        let tile_bytes = 16 * bits * 2;
        let shard_bytes = k_tiles_local * n_tiles * tile_bytes;
        let (_, trel_all) = idx.get_tensor(&format!("{prefix}.trellis"))?;
        let src_off = rank * shard_bytes;
        let mut trellis = CudaBuf::new(shard_bytes).map_err(CudaFail)?;
        trellis.copy_from(&trel_all[src_off..src_off + shard_bytes], None).map_err(CudaFail)?;

        let (_, suh_all) = idx.get_tensor(&format!("{prefix}.suh"))?;
        let suh_off = rank * k_local * 2;
        let mut suh = CudaBuf::new(k_local * 2).map_err(CudaFail)?;
        suh.copy_from(&suh_all[suh_off..suh_off + k_local * 2], None).map_err(CudaFail)?;

        let (_, svh_bytes) = idx.get_tensor(&format!("{prefix}.svh"))?;
        let mut svh = CudaBuf::new(svh_bytes.len()).map_err(CudaFail)?;
        svh.copy_from(svh_bytes, None).map_err(CudaFail)?;

        Ok(Self { trellis, suh, svh, k: k_local, n, bits })
    }
}

pub fn load_raw_buf(idx: &mut SafetensorsIndex, name: &str) -> Result<CudaBuf> {
    let (_, bytes) = idx.get_tensor(name)?;
    let mut buf = CudaBuf::new(bytes.len()).map_err(CudaFail)?;
    buf.copy_from(bytes, None).map_err(CudaFail)?;
    Ok(buf)
}

pub struct ExpertWeights {
    pub w1: Exl3Linear,
    pub w2: Exl3Linear,
    pub w3: Exl3Linear,
}

pub struct LayerWeights {
    pub attn_norm: CudaBuf,
    pub ffn_norm: CudaBuf,
    pub q_norm: CudaBuf,
    pub kv_norm: CudaBuf,
    pub attn_sink: CudaBuf,
    pub wq_a: Exl3Linear,
    pub wq_b: Exl3Linear,
    pub wkv: Exl3Linear,
    pub wo_a_slices: Vec<Exl3Linear>,
    pub wo_b: Exl3Linear,
    pub gate_w: CudaBuf,
    pub gate_bias: CudaBuf,
    pub shared_w1: Exl3Linear,
    pub shared_w2: Exl3Linear,
    pub shared_w3: Exl3Linear,
    pub hc_attn_fn: CudaBuf,
    pub hc_attn_base: CudaBuf,
    pub hc_attn_scale: CudaBuf,
    pub hc_ffn_fn: CudaBuf,
    pub hc_ffn_base: CudaBuf,
    pub hc_ffn_scale: CudaBuf,
    pub experts: Vec<ExpertWeights>,
    pub expert_start: usize,
    pub d_w1_trel: CudaBuf,
    pub d_w1_suh: CudaBuf,
    pub d_w1_svh: CudaBuf,
    pub d_w2_trel: CudaBuf,
    pub d_w2_suh: CudaBuf,
    pub d_w2_svh: CudaBuf,
    pub d_w3_trel: CudaBuf,
    pub d_w3_suh: CudaBuf,
    pub d_w3_svh: CudaBuf,
}

fn upload_ptr_table(ptrs: &[u64]) -> Result<CudaBuf> {
    let bytes = unsafe { std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, ptrs.len() * 8) };
    let mut buf = CudaBuf::new(bytes.len()).map_err(CudaFail)?;
    buf.copy_from(bytes, None).map_err(CudaFail)?;
    Ok(buf)
}

impl LayerWeights {
    pub fn load(
        idx: &mut SafetensorsIndex,
        cfg: &DeepseekV41Config,
        layer_idx: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        let p = format!("layers.{layer_idx}");
        let tc = &cfg.text_config;
        let groups_per_rank = tc.o_groups / world;
        let g_start = rank * groups_per_rank;
        let mut wo_a_slices = Vec::with_capacity(groups_per_rank);
        for g in g_start..(g_start + groups_per_rank) {
            wo_a_slices.push(Exl3Linear::load(idx, &format!("{p}.attn.wo_a.slice.{g}"))?);
        }
        let exp_per_rank = tc.n_routed_experts / world;
        let expert_start = rank * exp_per_rank;
        let mut experts = Vec::with_capacity(exp_per_rank);
        for e in expert_start..(expert_start + exp_per_rank) {
            experts.push(ExpertWeights {
                w1: Exl3Linear::load(idx, &format!("{p}.ffn.experts.{e}.w1"))?,
                w2: Exl3Linear::load(idx, &format!("{p}.ffn.experts.{e}.w2"))?,
                w3: Exl3Linear::load(idx, &format!("{p}.ffn.experts.{e}.w3"))?,
            });
        }
        let d_w1_trel = upload_ptr_table(&experts.iter().map(|e| e.w1.trellis.ptr as u64).collect::<Vec<_>>())?;
        let d_w1_suh = upload_ptr_table(&experts.iter().map(|e| e.w1.suh.ptr as u64).collect::<Vec<_>>())?;
        let d_w1_svh = upload_ptr_table(&experts.iter().map(|e| e.w1.svh.ptr as u64).collect::<Vec<_>>())?;
        let d_w2_trel = upload_ptr_table(&experts.iter().map(|e| e.w2.trellis.ptr as u64).collect::<Vec<_>>())?;
        let d_w2_suh = upload_ptr_table(&experts.iter().map(|e| e.w2.suh.ptr as u64).collect::<Vec<_>>())?;
        let d_w2_svh = upload_ptr_table(&experts.iter().map(|e| e.w2.svh.ptr as u64).collect::<Vec<_>>())?;
        let d_w3_trel = upload_ptr_table(&experts.iter().map(|e| e.w3.trellis.ptr as u64).collect::<Vec<_>>())?;
        let d_w3_suh = upload_ptr_table(&experts.iter().map(|e| e.w3.suh.ptr as u64).collect::<Vec<_>>())?;
        let d_w3_svh = upload_ptr_table(&experts.iter().map(|e| e.w3.svh.ptr as u64).collect::<Vec<_>>())?;
        Ok(Self {
            attn_norm: load_raw_buf(idx, &format!("{p}.attn_norm.weight"))?,
            ffn_norm: load_raw_buf(idx, &format!("{p}.ffn_norm.weight"))?,
            q_norm: load_raw_buf(idx, &format!("{p}.attn.q_norm.weight"))?,
            kv_norm: load_raw_buf(idx, &format!("{p}.attn.kv_norm.weight"))?,
            attn_sink: load_raw_buf(idx, &format!("{p}.attn.attn_sink"))?,
            wq_a: Exl3Linear::load(idx, &format!("{p}.attn.wq_a"))?,
            wq_b: Exl3Linear::load_col_shard(idx, &format!("{p}.attn.wq_b"), rank, world)?,
            wkv: Exl3Linear::load(idx, &format!("{p}.attn.wkv"))?,
            wo_a_slices,
            wo_b: Exl3Linear::load_row_shard(idx, &format!("{p}.attn.wo_b"), rank, world)?,
            gate_w: load_raw_buf(idx, &format!("{p}.ffn.gate.weight"))?,
            gate_bias: load_raw_buf(idx, &format!("{p}.ffn.gate.bias"))?,
            shared_w1: Exl3Linear::load_col_shard(idx, &format!("{p}.ffn.shared_experts.w1"), rank, world)?,
            shared_w2: Exl3Linear::load_row_shard(idx, &format!("{p}.ffn.shared_experts.w2"), rank, world)?,
            shared_w3: Exl3Linear::load_col_shard(idx, &format!("{p}.ffn.shared_experts.w3"), rank, world)?,
            hc_attn_fn: load_raw_buf(idx, &format!("{p}.hc_attn_fn"))?,
            hc_attn_base: load_raw_buf(idx, &format!("{p}.hc_attn_base"))?,
            hc_attn_scale: load_raw_buf(idx, &format!("{p}.hc_attn_scale"))?,
            hc_ffn_fn: load_raw_buf(idx, &format!("{p}.hc_ffn_fn"))?,
            hc_ffn_base: load_raw_buf(idx, &format!("{p}.hc_ffn_base"))?,
            hc_ffn_scale: load_raw_buf(idx, &format!("{p}.hc_ffn_scale"))?,
            experts,
            expert_start,
            d_w1_trel,
            d_w1_suh,
            d_w1_svh,
            d_w2_trel,
            d_w2_suh,
            d_w2_svh,
            d_w3_trel,
            d_w3_suh,
            d_w3_svh,
        })
    }
}
