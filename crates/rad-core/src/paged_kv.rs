use std::collections::HashMap;

pub struct PagedKvStats {
    pub block_size: usize,
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub shared_blocks: usize,
}

impl PagedKvStats {
    pub fn usage_ratio(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks as f64 / self.total_blocks as f64
        }
    }
}

pub struct PagedKvManager {
    pub block_size: usize,
    pub total_blocks: usize,
    free_blocks: Vec<i32>,
    ref_counts: Vec<u32>,
    seq_tables: Vec<Vec<i32>>,
    prefix_index: HashMap<u64, i32>,
    block_hashes: Vec<Option<u64>>,
}

impl PagedKvManager {
    pub fn new(max_seqs: usize, max_ctx: usize, block_size: usize, override_blocks: Option<usize>) -> Self {
        let bs = block_size.max(16);
        let default_blocks = ((max_ctx + bs - 1) / bs) * max_seqs;
        let total_blocks = override_blocks.unwrap_or(default_blocks).max(16);
        let free_blocks: Vec<i32> = (0..total_blocks as i32).rev().collect();
        Self {
            block_size: bs,
            total_blocks,
            free_blocks,
            ref_counts: vec![0; total_blocks],
            seq_tables: vec![Vec::new(); max_seqs],
            prefix_index: HashMap::new(),
            block_hashes: vec![None; total_blocks],
        }
    }

    fn hash_block(prev_hash: u64, tokens: &[i32]) -> u64 {
        let mut h = prev_hash ^ 0xcbf29ce484222325u64;
        for &t in tokens {
            h ^= (t as u32) as u64;
            h = h.wrapping_mul(0x100000001b3u64);
        }
        h
    }

    pub fn free_seq(&mut self, seq: usize) {
        if seq >= self.seq_tables.len() {
            return;
        }
        let blks = std::mem::take(&mut self.seq_tables[seq]);
        for b in blks {
            let bi = b as usize;
            if bi < self.ref_counts.len() && self.ref_counts[bi] > 0 {
                self.ref_counts[bi] -= 1;
                if self.ref_counts[bi] == 0 {
                    if let Some(h) = self.block_hashes[bi].take() {
                        if self.prefix_index.get(&h) == Some(&b) {
                            self.prefix_index.remove(&h);
                        }
                    }
                    self.free_blocks.push(b);
                }
            }
        }
    }

    pub fn ensure_tokens(&mut self, seq: usize, total_tokens: usize, token_ids: &[i32]) -> bool {
        if seq >= self.seq_tables.len() {
            return false;
        }
        let needed = (total_tokens + self.block_size - 1) / self.block_size;
        let mut prev_h = 0u64;
        for blk_idx in 0..self.seq_tables[seq].len() {
            let start = blk_idx * self.block_size;
            let end = (start + self.block_size).min(token_ids.len());
            if end - start == self.block_size {
                prev_h = Self::hash_block(prev_h, &token_ids[start..end]);
            }
        }
        while self.seq_tables[seq].len() < needed {
            let blk_idx = self.seq_tables[seq].len();
            let start = blk_idx * self.block_size;
            let end = (start + self.block_size).min(token_ids.len());
            if end - start == self.block_size {
                let h = Self::hash_block(prev_h, &token_ids[start..end]);
                prev_h = h;
                if let Some(&existing) = self.prefix_index.get(&h) {
                    let ei = existing as usize;
                    if self.ref_counts[ei] > 0 {
                        self.ref_counts[ei] += 1;
                        self.seq_tables[seq].push(existing);
                        continue;
                    }
                }
                if let Some(nb) = self.free_blocks.pop() {
                    let ni = nb as usize;
                    self.ref_counts[ni] = 1;
                    self.block_hashes[ni] = Some(h);
                    self.prefix_index.insert(h, nb);
                    self.seq_tables[seq].push(nb);
                    continue;
                }
                return false;
            }
            if let Some(nb) = self.free_blocks.pop() {
                self.ref_counts[nb as usize] = 1;
                self.seq_tables[seq].push(nb);
            } else {
                return false;
            }
        }
        true
    }

    pub fn seq_block_table(&self, seq: usize) -> &[i32] {
        self.seq_tables.get(seq).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn stats(&self) -> PagedKvStats {
        let used = self.total_blocks.saturating_sub(self.free_blocks.len());
        let shared = self.ref_counts.iter().filter(|&&c| c > 1).count();
        PagedKvStats {
            block_size: self.block_size,
            total_blocks: self.total_blocks,
            used_blocks: used,
            shared_blocks: shared,
        }
    }
}
