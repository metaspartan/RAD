use anyhow::Result;
use rad_cuda::comm::Tp2Comm;
use rad_cuda::{CudaBuf, CudaFail};
use crate::config::DeepseekV41Config;
use crate::engine::MAX_BATCH;
use crate::safetensors::SafetensorsIndex;
use crate::weights::{load_raw_buf_replicated, load_raw_buf_row_shard, Exl3Linear, LayerWeights};

pub struct MtpKvCache {
    pub k_cache: CudaBuf,
}

pub struct DsparkMtp {
    pub main_norm: CudaBuf,
    pub main_proj: Exl3Linear,
    pub final_norm: CudaBuf,
    pub markov_embed: CudaBuf,
    pub markov_head: CudaBuf,
    pub layers: Vec<LayerWeights>,
    pub kv_caches: Vec<MtpKvCache>,
    pub d_taps_f16: CudaBuf,
    pub d_main_proj: CudaBuf,
    pub d_main_x: CudaBuf,
    pub d_logits_row: CudaBuf,
    pub last_ctx_row: usize,
}

impl DsparkMtp {
    pub fn load(
        mut idx: Option<&mut SafetensorsIndex>,
        cfg: &DeepseekV41Config,
        max_ctx: usize,
        comm: &mut Tp2Comm,
    ) -> Result<Self> {
        let tc = &cfg.text_config;
        let n_mtp = tc.num_nextn_predict_layers;
        let main_norm = load_raw_buf_replicated(idx.as_deref_mut(), "mtp.0.main_norm.weight", comm)?;
        let main_proj = Exl3Linear::load_replicated(idx.as_deref_mut(), "mtp.0.main_proj", comm)?;
        let final_norm = load_raw_buf_replicated(idx.as_deref_mut(), &format!("mtp.{}.norm.weight", n_mtp - 1), comm)?;
        let markov_embed = load_raw_buf_replicated(idx.as_deref_mut(), &format!("mtp.{}.markov_head.embed.weight", n_mtp - 1), comm)?;
        let markov_head = load_raw_buf_row_shard(idx.as_deref_mut(), &format!("mtp.{}.markov_head.head.weight", n_mtp - 1), comm)?;
        let mut layers = Vec::with_capacity(n_mtp);
        let mut kv_caches = Vec::with_capacity(n_mtp);
        for i in 0..n_mtp {
            layers.push(LayerWeights::load_prefix(
                idx.as_deref_mut(),
                cfg,
                &format!("mtp.{i}"),
                tc.dspark_n_routed_experts,
                comm,
            )?);
            kv_caches.push(MtpKvCache {
                k_cache: CudaBuf::new(max_ctx * tc.head_dim * 2).map_err(CudaFail)?,
            });
        }
        let d = tc.hidden_size;
        let vocab_local = tc.vocab_size / comm.world_size();
        Ok(Self {
            main_norm,
            main_proj,
            final_norm,
            markov_embed,
            markov_head,
            layers,
            kv_caches,
            d_taps_f16: CudaBuf::new(MAX_BATCH * 3 * d * 2).map_err(CudaFail)?,
            d_main_proj: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_main_x: CudaBuf::new(MAX_BATCH * d * 2).map_err(CudaFail)?,
            d_logits_row: CudaBuf::new(vocab_local * 2).map_err(CudaFail)?,
            last_ctx_row: 0,
        })
    }
}

pub fn ngram_draft(history: &[i32], anchor: i32, max_k: usize) -> Option<Vec<i32>> {
    let n = history.len();
    if n < 8 || max_k < 4 {
        return None;
    }
    let c = history[n - 1];
    let b = history[n - 2];
    let mut best_pos: Option<usize> = None;
    let mut best_len = 0usize;
    let mut i = n.saturating_sub(4);
    while i >= 3 {
        if history[i] == anchor && history[i - 1] == c && history[i - 2] == b {
            let mut mlen = 3usize;
            let lim = 64usize.min(i).min(n);
            while mlen < lim && history[i - mlen] == history[n - mlen] {
                mlen += 1;
            }
            if mlen >= 4 && mlen > best_len {
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
    if avail < 6 {
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
