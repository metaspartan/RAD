use crate::model::InklingConfig;
use anyhow::Result;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpConfig {
    pub rank: usize,
    pub world_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TpShardPlan {
    pub rank: usize,
    pub world_size: usize,
    pub local_n_head: usize,
    pub local_n_head_kv: usize,
    pub local_n_expert: usize,
    pub head_range: Range<usize>,
    pub kv_head_range: Range<usize>,
    pub q_row_range: Range<usize>,
    pub kv_row_range: Range<usize>,
    pub r_row_range: Range<usize>,
    pub wo_col_range: Range<usize>,
    pub expert_range: Range<usize>,
}

impl TpConfig {
    pub fn single() -> Self {
        Self { rank: 0, world_size: 1 }
    }

    pub fn new(rank: usize, world_size: usize) -> Result<Self> {
        if world_size == 0 {
            anyhow::bail!("tp world_size must be >= 1");
        }
        if rank >= world_size {
            anyhow::bail!("tp rank {rank} out of range for world_size {world_size}");
        }
        Ok(Self { rank, world_size })
    }

    pub fn validate(&self, cfg: &InklingConfig) -> Result<()> {
        if cfg.n_head % self.world_size != 0 {
            anyhow::bail!("n_head={} not divisible by tp={}", cfg.n_head, self.world_size);
        }
        if cfg.n_head_kv % self.world_size != 0 {
            anyhow::bail!("n_head_kv={} not divisible by tp={}", cfg.n_head_kv, self.world_size);
        }
        if cfg.n_expert % self.world_size != 0 {
            anyhow::bail!("n_expert={} not divisible by tp={}", cfg.n_expert, self.world_size);
        }
        Ok(())
    }

    pub fn plan(&self, cfg: &InklingConfig) -> Result<TpShardPlan> {
        self.validate(cfg)?;
        let lh = cfg.n_head / self.world_size;
        let lkv = cfg.n_head_kv / self.world_size;
        let le = cfg.n_expert / self.world_size;
        let h0 = self.rank * lh;
        let kv0 = self.rank * lkv;
        let e0 = self.rank * le;
        Ok(TpShardPlan {
            rank: self.rank,
            world_size: self.world_size,
            local_n_head: lh,
            local_n_head_kv: lkv,
            local_n_expert: le,
            head_range: h0..(h0 + lh),
            kv_head_range: kv0..(kv0 + lkv),
            q_row_range: (h0 * cfg.dh)..((h0 + lh) * cfg.dh),
            kv_row_range: (kv0 * cfg.dh)..((kv0 + lkv) * cfg.dh),
            r_row_range: (h0 * cfg.d_rel)..((h0 + lh) * cfg.d_rel),
            wo_col_range: (h0 * cfg.dh)..((h0 + lh) * cfg.dh),
            expert_range: e0..(e0 + le),
        })
    }

    pub fn owns_expert(&self, cfg: &InklingConfig, exp_id: usize) -> Option<usize> {
        let le = cfg.n_expert / self.world_size;
        let e0 = self.rank * le;
        if exp_id >= e0 && exp_id < e0 + le {
            Some(exp_id - e0)
        } else {
            None
        }
    }
}

pub fn slice_row_shard(data: &[u8], row_bytes: usize, rows: Range<usize>) -> &[u8] {
    let start = rows.start * row_bytes;
    let end = rows.end * row_bytes;
    &data[start..end]
}

pub fn slice_col_shard(data: &[u8], n_rows: usize, row_bytes: usize, col_byte_start: usize, col_byte_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_rows * col_byte_len);
    for r in 0..n_rows {
        let off = r * row_bytes + col_byte_start;
        out.extend_from_slice(&data[off..(off + col_byte_len)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inkling_cfg() -> InklingConfig {
        InklingConfig {
            n_embd: 2304,
            n_head: 24,
            n_head_kv: 8,
            dh: 128,
            d_rel: 16,
            rel_extent: 64,
            n_phys: 24,
            n_passes: 3,
            core_start: 6,
            core_end: 18,
            n_expert: 128,
            n_expert_used: 8,
            n_ff_exp: 768,
            vocab_padded: 200704,
            vocab_unpadded: 200064,
            swa_window: 2048,
            is_swa: vec![false; 24],
            attn_res_block: 4,
            gate_rank: 64,
            eps: 1e-6,
            qk_clip: 12.0,
            rel_clip: 12.0,
            log_floor: 512.0,
            log_alpha: 0.1,
            route_scale: 1.0,
            final_softcap: 30.0,
            situ_beta: 1.0,
            situ_lb: 0.0,
        }
    }

    #[test]
    fn tp2_sharding_plan() {
        let cfg = inkling_cfg();
        let p0 = TpConfig::new(0, 2).unwrap().plan(&cfg).unwrap();
        let p1 = TpConfig::new(1, 2).unwrap().plan(&cfg).unwrap();
        assert_eq!(p0.local_n_head, 12);
        assert_eq!(p0.local_n_head_kv, 4);
        assert_eq!(p0.local_n_expert, 64);
        assert_eq!(p0.head_range, 0..12);
        assert_eq!(p1.head_range, 12..24);
        assert_eq!(p0.expert_range, 0..64);
        assert_eq!(p1.expert_range, 64..128);
        assert_eq!(p0.q_row_range, 0..1536);
        assert_eq!(p1.q_row_range, 1536..3072);
    }

    #[test]
    fn tp_expert_ownership() {
        let cfg = inkling_cfg();
        let tp0 = TpConfig::new(0, 2).unwrap();
        let tp1 = TpConfig::new(1, 2).unwrap();
        assert_eq!(tp0.owns_expert(&cfg, 0), Some(0));
        assert_eq!(tp0.owns_expert(&cfg, 63), Some(63));
        assert_eq!(tp0.owns_expert(&cfg, 64), None);
        assert_eq!(tp1.owns_expert(&cfg, 64), Some(0));
        assert_eq!(tp1.owns_expert(&cfg, 127), Some(63));
    }

    #[test]
    fn col_and_row_sharding() {
        let src: Vec<u8> = (0..16u8).collect();
        assert_eq!(slice_row_shard(&src, 4, 1..3), &[4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(slice_col_shard(&src, 4, 4, 2, 2), vec![2, 3, 6, 7, 10, 11, 14, 15]);
    }
}
