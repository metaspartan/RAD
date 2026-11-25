use anyhow::Result;
use rad_cuda::CudaBuf;
use crate::config::DeepseekV41Config;
use crate::safetensors::SafetensorsIndex;
use crate::weights::{load_raw_buf, Exl3Linear, ExpertWeights};

pub struct MtpLayerWeights {
    pub main_norm: Option<CudaBuf>,
    pub main_proj: Option<Exl3Linear>,
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
    pub experts: Vec<ExpertWeights>,
    pub expert_start: usize,
}

impl MtpLayerWeights {
    pub fn load(
        idx: &mut SafetensorsIndex,
        cfg: &DeepseekV41Config,
        mtp_idx: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        let p = format!("mtp.{mtp_idx}");
        let tc = &cfg.text_config;
        let groups_per_rank = tc.o_groups / world;
        let g_start = rank * groups_per_rank;
        let mut wo_a_slices = Vec::with_capacity(groups_per_rank);
        for g in g_start..(g_start + groups_per_rank) {
            wo_a_slices.push(Exl3Linear::load(idx, &format!("{p}.attn.wo_a.slice.{g}"))?);
        }
        let n_exp = tc.dspark_n_routed_experts;
        let exp_per_rank = n_exp / world;
        let expert_start = rank * exp_per_rank;
        let mut experts = Vec::with_capacity(exp_per_rank);
        for e in expert_start..(expert_start + exp_per_rank) {
            experts.push(ExpertWeights {
                w1: Exl3Linear::load(idx, &format!("{p}.ffn.experts.{e}.w1"))?,
                w2: Exl3Linear::load(idx, &format!("{p}.ffn.experts.{e}.w2"))?,
                w3: Exl3Linear::load(idx, &format!("{p}.ffn.experts.{e}.w3"))?,
            });
        }
        let main_norm = if idx.contains(&format!("{p}.main_norm.weight")) {
            Some(load_raw_buf(idx, &format!("{p}.main_norm.weight"))?)
        } else {
            None
        };
        let main_proj = if idx.contains(&format!("{p}.main_proj.trellis")) {
            Some(Exl3Linear::load(idx, &format!("{p}.main_proj"))?)
        } else {
            None
        };
        Ok(Self {
            main_norm,
            main_proj,
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
            experts,
            expert_start,
        })
    }
}

pub fn ngram_draft(history: &[i32], anchor: i32, max_k: usize) -> Option<Vec<i32>> {
    let n = history.len();
    if n < 12 || max_k < 5 {
        return None;
    }
    let c = history[n - 1];
    let b = history[n - 2];
    let mut best_pos: Option<usize> = None;
    let mut best_len = 0usize;
    let mut i = n.saturating_sub(6);
    while i >= 2 {
        if history[i] == anchor && history[i - 1] == c && history[i - 2] == b {
            let mut mlen = 3usize;
            let lim = 32usize.min(i).min(n);
            while mlen < lim && history[i - mlen] == history[n - mlen] {
                mlen += 1;
            }
            if mlen > best_len {
                best_len = mlen;
                best_pos = Some(i);
                if mlen >= 16 {
                    break;
                }
            }
        }
        i -= 1;
    }
    let pos = best_pos?;
    let avail = (n - pos).min(max_k);
    if avail < 5 {
        return None;
    }
    let mut out = Vec::with_capacity(avail);
    out.push(anchor);
    out.extend_from_slice(&history[(pos + 1)..(pos + avail)]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_draft_matches_repeated_suffix() {
        let mut h: Vec<i32> = (1..20).collect();
        h.extend_from_slice(&[5, 6, 7, 8]);
        let d = ngram_draft(&h, 9, 8).expect("should match");
        assert_eq!(&d[..5], &[9, 10, 11, 12, 13]);
    }
}
