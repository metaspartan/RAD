use anyhow::{anyhow, Result};
use rad_cuda::comm::Tp2Comm;
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

fn buf_from_slice(bytes: &[u8]) -> Result<CudaBuf> {
    let mut b = CudaBuf::new(bytes.len()).map_err(CudaFail)?;
    b.copy_from(bytes, None).map_err(CudaFail)?;
    Ok(b)
}

fn slice_col_shard(
    trel_all: &[u8],
    suh_all: &[u8],
    svh_all: &[u8],
    k: usize,
    n: usize,
    bits: usize,
    rank: usize,
    world: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, usize)> {
    let n_local = n / world;
    if n_local % 128 != 0 {
        return Err(anyhow!("col shard n_local={n_local} not multiple of 128"));
    }
    let k_tiles = k / 16;
    let n_tiles = n / 16;
    let n_tiles_local = n_local / 16;
    let n_tile_start = rank * n_tiles_local;
    let tile_bytes = 16 * bits * 2;
    let mut trel_host = vec![0u8; k_tiles * n_tiles_local * tile_bytes];
    for kt in 0..k_tiles {
        let src_off = (kt * n_tiles + n_tile_start) * tile_bytes;
        let dst_off = kt * n_tiles_local * tile_bytes;
        let row_len = n_tiles_local * tile_bytes;
        trel_host[dst_off..dst_off + row_len].copy_from_slice(&trel_all[src_off..src_off + row_len]);
    }
    let svh_off = rank * n_local * 2;
    Ok((
        trel_host,
        suh_all.to_vec(),
        svh_all[svh_off..svh_off + n_local * 2].to_vec(),
        n_local,
    ))
}

fn slice_row_shard(
    trel_all: &[u8],
    suh_all: &[u8],
    svh_all: &[u8],
    k: usize,
    n: usize,
    bits: usize,
    rank: usize,
    world: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, usize)> {
    let k_local = k / world;
    if k_local % 128 != 0 {
        return Err(anyhow!("row shard k_local={k_local} not multiple of 128"));
    }
    let k_tiles_local = k_local / 16;
    let n_tiles = n / 16;
    let tile_bytes = 16 * bits * 2;
    let shard_bytes = k_tiles_local * n_tiles * tile_bytes;
    let src_off = rank * shard_bytes;
    let suh_off = rank * k_local * 2;
    Ok((
        trel_all[src_off..src_off + shard_bytes].to_vec(),
        suh_all[suh_off..suh_off + k_local * 2].to_vec(),
        svh_all.to_vec(),
        k_local,
    ))
}

impl Exl3Linear {
    pub fn send_raw(comm: &mut Tp2Comm, trel: &[u8], suh: &[u8], svh: &[u8], k: usize, n: usize, bits: usize) -> Result<()> {
        comm.send_u64_slice(&[k as u64, n as u64, bits as u64])?;
        comm.send_bytes(trel)?;
        comm.send_bytes(suh)?;
        comm.send_bytes(svh)?;
        Ok(())
    }

    pub fn recv(comm: &mut Tp2Comm) -> Result<Self> {
        let meta = comm.recv_u64_slice()?;
        let (k, n, bits) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
        let trel = comm.recv_bytes()?;
        let suh = comm.recv_bytes()?;
        let svh = comm.recv_bytes()?;
        Ok(Self {
            trellis: buf_from_slice(&trel)?,
            suh: buf_from_slice(&suh)?,
            svh: buf_from_slice(&svh)?,
            k,
            n,
            bits,
        })
    }

    pub fn load_replicated(
        idx: Option<&mut SafetensorsIndex>,
        prefix: &str,
        comm: &mut Tp2Comm,
    ) -> Result<Self> {
        let world = comm.world_size();
        if world > 1 && comm.rank != 0 {
            return Self::recv(comm);
        }
        let idx = idx.ok_or_else(|| anyhow!("rank 0 missing safetensors index"))?;
        let (k, n, bits) = idx.exl3_shape_and_bits(prefix)?;
        let trel = idx.get_tensor(&format!("{prefix}.trellis"))?.1.to_vec();
        let suh = idx.get_tensor(&format!("{prefix}.suh"))?.1.to_vec();
        let svh = idx.get_tensor(&format!("{prefix}.svh"))?.1.to_vec();
        if world > 1 {
            Self::send_raw(comm, &trel, &suh, &svh, k, n, bits)?;
        }
        Ok(Self {
            trellis: buf_from_slice(&trel)?,
            suh: buf_from_slice(&suh)?,
            svh: buf_from_slice(&svh)?,
            k,
            n,
            bits,
        })
    }

    pub fn load_col_shard(
        idx: Option<&mut SafetensorsIndex>,
        prefix: &str,
        comm: &mut Tp2Comm,
    ) -> Result<Self> {
        let world = comm.world_size();
        if world <= 1 {
            return Self::load_replicated(idx, prefix, comm);
        }
        if comm.rank != 0 {
            return Self::recv(comm);
        }
        let idx = idx.ok_or_else(|| anyhow!("rank 0 missing safetensors index"))?;
        let (k, n, bits) = idx.exl3_shape_and_bits(prefix)?;
        let trel_all = idx.get_tensor(&format!("{prefix}.trellis"))?.1.to_vec();
        let suh_all = idx.get_tensor(&format!("{prefix}.suh"))?.1.to_vec();
        let svh_all = idx.get_tensor(&format!("{prefix}.svh"))?.1.to_vec();
        let (t0, su0, sv0, nl) = slice_col_shard(&trel_all, &suh_all, &svh_all, k, n, bits, 0, world)?;
        let (t1, su1, sv1, _) = slice_col_shard(&trel_all, &suh_all, &svh_all, k, n, bits, 1, world)?;
        Self::send_raw(comm, &t1, &su1, &sv1, k, nl, bits)?;
        Ok(Self {
            trellis: buf_from_slice(&t0)?,
            suh: buf_from_slice(&su0)?,
            svh: buf_from_slice(&sv0)?,
            k,
            n: nl,
            bits,
        })
    }

    pub fn load_row_shard(
        idx: Option<&mut SafetensorsIndex>,
        prefix: &str,
        comm: &mut Tp2Comm,
    ) -> Result<Self> {
        let world = comm.world_size();
        if world <= 1 {
            return Self::load_replicated(idx, prefix, comm);
        }
        if comm.rank != 0 {
            return Self::recv(comm);
        }
        let idx = idx.ok_or_else(|| anyhow!("rank 0 missing safetensors index"))?;
        let (k, n, bits) = idx.exl3_shape_and_bits(prefix)?;
        let trel_all = idx.get_tensor(&format!("{prefix}.trellis"))?.1.to_vec();
        let suh_all = idx.get_tensor(&format!("{prefix}.suh"))?.1.to_vec();
        let svh_all = idx.get_tensor(&format!("{prefix}.svh"))?.1.to_vec();
        let (t0, su0, sv0, kl) = slice_row_shard(&trel_all, &suh_all, &svh_all, k, n, bits, 0, world)?;
        let (t1, su1, sv1, _) = slice_row_shard(&trel_all, &suh_all, &svh_all, k, n, bits, 1, world)?;
        Self::send_raw(comm, &t1, &su1, &sv1, kl, n, bits)?;
        Ok(Self {
            trellis: buf_from_slice(&t0)?,
            suh: buf_from_slice(&su0)?,
            svh: buf_from_slice(&sv0)?,
            k: kl,
            n,
            bits,
        })
    }

    pub fn load_rank_only(
        idx: Option<&mut SafetensorsIndex>,
        prefix_r0: &str,
        prefix_r1: &str,
        comm: &mut Tp2Comm,
    ) -> Result<Self> {
        let world = comm.world_size();
        if world <= 1 {
            return Self::load_replicated(idx, prefix_r0, comm);
        }
        if comm.rank != 0 {
            return Self::recv(comm);
        }
        let idx = idx.ok_or_else(|| anyhow!("rank 0 missing safetensors index"))?;
        let (k1, n1, b1) = idx.exl3_shape_and_bits(prefix_r1)?;
        let t1 = idx.get_tensor(&format!("{prefix_r1}.trellis"))?.1.to_vec();
        let su1 = idx.get_tensor(&format!("{prefix_r1}.suh"))?.1.to_vec();
        let sv1 = idx.get_tensor(&format!("{prefix_r1}.svh"))?.1.to_vec();
        Self::send_raw(comm, &t1, &su1, &sv1, k1, n1, b1)?;

        let (k0, n0, b0) = idx.exl3_shape_and_bits(prefix_r0)?;
        let t0 = idx.get_tensor(&format!("{prefix_r0}.trellis"))?.1.to_vec();
        let su0 = idx.get_tensor(&format!("{prefix_r0}.suh"))?.1.to_vec();
        let sv0 = idx.get_tensor(&format!("{prefix_r0}.svh"))?.1.to_vec();
        Ok(Self {
            trellis: buf_from_slice(&t0)?,
            suh: buf_from_slice(&su0)?,
            svh: buf_from_slice(&sv0)?,
            k: k0,
            n: n0,
            bits: b0,
        })
    }
}

pub fn load_raw_buf_replicated(
    idx: Option<&mut SafetensorsIndex>,
    name: &str,
    comm: &mut Tp2Comm,
) -> Result<CudaBuf> {
    let world = comm.world_size();
    if world > 1 && comm.rank != 0 {
        let bytes = comm.recv_bytes()?;
        return buf_from_slice(&bytes);
    }
    let idx = idx.ok_or_else(|| anyhow!("rank 0 missing safetensors index"))?;
    let bytes = idx.get_tensor(name)?.1.to_vec();
    if world > 1 {
        comm.send_bytes(&bytes)?;
    }
    buf_from_slice(&bytes)
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
    buf_from_slice(bytes)
}

impl LayerWeights {
    pub fn load(
        mut idx: Option<&mut SafetensorsIndex>,
        cfg: &DeepseekV41Config,
        layer_idx: usize,
        comm: &mut Tp2Comm,
    ) -> Result<Self> {
        let p = format!("layers.{layer_idx}");
        let tc = &cfg.text_config;
        let world = comm.world_size();
        let rank = comm.rank;

        let groups_per_rank = tc.o_groups / world;
        let mut wo_a_slices = Vec::with_capacity(groups_per_rank);
        for g_local in 0..groups_per_rank {
            let p0 = format!("{p}.attn.wo_a.slice.{g_local}");
            let p1 = format!("{p}.attn.wo_a.slice.{}", g_local + groups_per_rank);
            wo_a_slices.push(Exl3Linear::load_rank_only(idx.as_deref_mut(), &p0, &p1, comm)?);
        }

        let exp_per_rank = tc.n_routed_experts / world;
        let expert_start = rank * exp_per_rank;
        let mut experts = Vec::with_capacity(exp_per_rank);
        for e_local in 0..exp_per_rank {
            let e0 = e_local;
            let e1 = e_local + exp_per_rank;
            experts.push(ExpertWeights {
                w1: Exl3Linear::load_rank_only(
                    idx.as_deref_mut(),
                    &format!("{p}.ffn.experts.{e0}.w1"),
                    &format!("{p}.ffn.experts.{e1}.w1"),
                    comm,
                )?,
                w2: Exl3Linear::load_rank_only(
                    idx.as_deref_mut(),
                    &format!("{p}.ffn.experts.{e0}.w2"),
                    &format!("{p}.ffn.experts.{e1}.w2"),
                    comm,
                )?,
                w3: Exl3Linear::load_rank_only(
                    idx.as_deref_mut(),
                    &format!("{p}.ffn.experts.{e0}.w3"),
                    &format!("{p}.ffn.experts.{e1}.w3"),
                    comm,
                )?,
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
            attn_norm: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.attn_norm.weight"), comm)?,
            ffn_norm: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.ffn_norm.weight"), comm)?,
            q_norm: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.attn.q_norm.weight"), comm)?,
            kv_norm: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.attn.kv_norm.weight"), comm)?,
            attn_sink: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.attn.attn_sink"), comm)?,
            wq_a: Exl3Linear::load_replicated(idx.as_deref_mut(), &format!("{p}.attn.wq_a"), comm)?,
            wq_b: Exl3Linear::load_col_shard(idx.as_deref_mut(), &format!("{p}.attn.wq_b"), comm)?,
            wkv: Exl3Linear::load_replicated(idx.as_deref_mut(), &format!("{p}.attn.wkv"), comm)?,
            wo_a_slices,
            wo_b: Exl3Linear::load_row_shard(idx.as_deref_mut(), &format!("{p}.attn.wo_b"), comm)?,
            gate_w: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.ffn.gate.weight"), comm)?,
            gate_bias: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.ffn.gate.bias"), comm)?,
            shared_w1: Exl3Linear::load_col_shard(idx.as_deref_mut(), &format!("{p}.ffn.shared_experts.w1"), comm)?,
            shared_w2: Exl3Linear::load_row_shard(idx.as_deref_mut(), &format!("{p}.ffn.shared_experts.w2"), comm)?,
            shared_w3: Exl3Linear::load_col_shard(idx.as_deref_mut(), &format!("{p}.ffn.shared_experts.w3"), comm)?,
            hc_attn_fn: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.hc_attn_fn"), comm)?,
            hc_attn_base: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.hc_attn_base"), comm)?,
            hc_attn_scale: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.hc_attn_scale"), comm)?,
            hc_ffn_fn: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.hc_ffn_fn"), comm)?,
            hc_ffn_base: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.hc_ffn_base"), comm)?,
            hc_ffn_scale: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.hc_ffn_scale"), comm)?,
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
