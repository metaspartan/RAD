use anyhow::{anyhow, Result};
use rad_cuda::comm::Tp2Comm;
use rad_cuda::{CudaBuf, CudaFail};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;
use tokenizers::normalizers::{Lowercase, NormalizerWrapper, Replace, Sequence, Strip, StripAccents, NFD, NFKC};
use tokenizers::{Normalizer, NormalizedString, Tokenizer};
use crate::safetensors::SafetensorsIndex;
use crate::weights::{load_raw_buf_replicated, Exl3Linear};

const LAYER_MULTS: [[i64; 4]; 2] = [
    [76632096046245, 4839876093313, 35959672319349, 73987337458391],
    [67716810739261, 51510806800915, 30921347202721, 82619226485591],
];

pub struct EngramLayerWeights {
    pub layer_id: usize,
    pub table_idx: usize,
    pub k_weight: CudaBuf,
    pub q_weight: CudaBuf,
    pub wkv: Exl3Linear,
}

struct EngramShardFile {
    file: File,
    weight_base: u64,
    scale_base: u64,
    primes: [i64; 24],
    offsets: [i64; 24],
    mults: [i64; 4],
}

pub struct EngramHost {
    token_map: Vec<i64>,
    pad_comp: i64,
    shards: Vec<EngramShardFile>,
    e4m3_lut: [f32; 256],
}

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3i64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

pub fn generate_engram_primes() -> ([i64; 24], [i64; 24]) {
    let mut p = 16_000_001i64;
    let mut p1 = [0i64; 24];
    let mut p2 = [0i64; 24];
    let mut i = 0usize;
    while i < 24 {
        if is_prime(p) {
            p1[i] = p;
            i += 1;
        }
        p += 2;
    }
    let mut j = 0usize;
    while j < 24 {
        if is_prime(p) {
            p2[j] = p;
            j += 1;
        }
        p += 2;
    }
    (p1, p2)
}

fn build_e4m3_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for b in 0..256usize {
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (b >> 3) & 0x0f;
        let man = b & 0x07;
        let val = if exp == 15 && man == 7 {
            0.0f32
        } else if exp == 0 {
            sign * (man as f32 / 8.0f32) * 2.0f32.powi(-6)
        } else {
            sign * (1.0f32 + man as f32 / 8.0f32) * 2.0f32.powi(exp as i32 - 7)
        };
        lut[b] = val;
    }
    lut
}

fn f32_to_f16(val: f32) -> u16 {
    let b = val.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let f = b & 0x7fffff;
    if e <= 0 {
        s
    } else if e >= 31 {
        s | 0x7c00
    } else {
        s | ((e as u16) << 10) | ((f >> 13) as u16)
    }
}

fn open_engram_shard(path: &Path, layer_id: usize, primes: [i64; 24], mults: [i64; 4]) -> Result<EngramShardFile> {
    let mut f = File::open(path)?;
    let mut hlen_buf = [0u8; 8];
    f.read_exact(&mut hlen_buf)?;
    let hlen = u64::from_le_bytes(hlen_buf);
    let mut hbytes = vec![0u8; hlen as usize];
    f.read_exact(&mut hbytes)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hbytes)?;
    let data_base = 8u64 + hlen;
    let wk = format!("layers.{layer_id}.engram.embed.weight");
    let sk = format!("layers.{layer_id}.engram.embed.scale");
    let w_off = hdr
        .get(&wk)
        .and_then(|v| v.get("data_offsets"))
        .and_then(|v| v.get(0))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing {wk}"))?;
    let s_off = hdr
        .get(&sk)
        .and_then(|v| v.get("data_offsets"))
        .and_then(|v| v.get(0))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing {sk}"))?;
    let mut offsets = [0i64; 24];
    let mut acc = 0i64;
    for i in 0..24 {
        offsets[i] = acc;
        acc += primes[i];
    }
    Ok(EngramShardFile {
        file: f,
        weight_base: data_base + w_off,
        scale_base: data_base + s_off,
        primes,
        offsets,
        mults,
    })
}

pub fn build_compressed_token_map(tok: &Tokenizer, vocab_size: usize) -> Result<(Vec<i64>, usize)> {
    let sentinel = "\u{e000}";
    let norm = Sequence::new(vec![
        NormalizerWrapper::NFKC(NFKC),
        NormalizerWrapper::NFD(NFD),
        NormalizerWrapper::StripAccents(StripAccents),
        NormalizerWrapper::Lowercase(Lowercase),
        NormalizerWrapper::Replace(Replace::new(r"[ \t\r\n]+", " ").map_err(|e| anyhow!("{e}"))?),
        NormalizerWrapper::Replace(Replace::new(r"^ $", sentinel).map_err(|e| anyhow!("{e}"))?),
        NormalizerWrapper::Strip(Strip::new(true, true)),
        NormalizerWrapper::Replace(Replace::new(sentinel, " ").map_err(|e| anyhow!("{e}"))?),
    ]);
    let mut map = vec![0i64; vocab_size];
    let mut key_to_id: HashMap<String, i64> = HashMap::with_capacity(100_000);
    let mut next_id = 0i64;
    for tid in 0..vocab_size {
        let decoded = tok.decode(&[tid as u32], false).unwrap_or_default();
        let mut ns = NormalizedString::from(decoded.as_str());
        let _ = norm.normalize(&mut ns);
        let key = ns.get().to_string();
        let cid = if let Some(&existing) = key_to_id.get(&key) {
            existing
        } else {
            let id = next_id;
            next_id += 1;
            key_to_id.insert(key, id);
            id
        };
        map[tid] = cid;
    }
    Ok((map, next_id as usize))
}

impl EngramHost {
    pub fn open(engram_dir: &Path, tok: &Tokenizer, vocab_size: usize, pad_token_id: usize) -> Result<Self> {
        let (token_map, comp_vocab) = build_compressed_token_map(tok, vocab_size)?;
        let pad_comp = token_map.get(pad_token_id).copied().unwrap_or(0);
        let (p1, p2) = generate_engram_primes();
        let s1 = open_engram_shard(&engram_dir.join("model-00047-of-00048.safetensors"), 1, p1, LAYER_MULTS[0])?;
        let s2 = open_engram_shard(&engram_dir.join("model-00048-of-00048.safetensors"), 14, p2, LAYER_MULTS[1])?;
        println!("[engram] opened tables (compressed_vocab={comp_vocab})");
        Ok(Self {
            token_map,
            pad_comp,
            shards: vec![s1, s2],
            e4m3_lut: build_e4m3_lut(),
        })
    }

    pub fn lookup_batch_f16(&self, table_idx: usize, history: &[i32], start_pos: usize, m: usize) -> Result<Vec<u16>> {
        let shard = &self.shards[table_idx];
        let mut out = vec![0u16; m * 24 * 256];
        let mut wbuf = [0u8; 256];
        let mut sbuf = [0u8; 8];
        for mr in 0..m {
            let pos = start_pos + mr;
            let mut c = [self.pad_comp; 4];
            for k in 0..4 {
                if pos >= k {
                    let tid = history.get(pos - k).copied().unwrap_or(2) as usize;
                    c[k] = self.token_map.get(tid).copied().unwrap_or(self.pad_comp);
                }
            }
            let m0 = c[0].wrapping_mul(shard.mults[0]);
            let m1 = m0 ^ c[1].wrapping_mul(shard.mults[1]);
            let m2 = m1 ^ c[2].wrapping_mul(shard.mults[2]);
            let m3 = m2 ^ c[3].wrapping_mul(shard.mults[3]);
            let mixes = [m1, m2, m3];
            for ord in 0..3 {
                let mix = mixes[ord];
                for h in 0..8 {
                    let col = ord * 8 + h;
                    let row_id = (shard.offsets[col] + mix.rem_euclid(shard.primes[col])) as u64;
                    shard.file.read_exact_at(&mut wbuf, shard.weight_base + row_id * 256)?;
                    shard.file.read_exact_at(&mut sbuf, shard.scale_base + row_id * 8)?;
                    let dst = &mut out[(mr * 24 + col) * 256..(mr * 24 + col + 1) * 256];
                    for blk in 0..8 {
                        let e = sbuf[blk] as i32;
                        let scale = if e == 0 { 0.0f32 } else { f32::from_bits(((e as u32) << 23)) };
                        let base = blk * 32;
                        for j in 0..32 {
                            dst[base + j] = f32_to_f16(self.e4m3_lut[wbuf[base + j] as usize] * scale);
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

impl EngramLayerWeights {
    pub fn load(
        mut idx: Option<&mut SafetensorsIndex>,
        layer_id: usize,
        table_idx: usize,
        comm: &mut Tp2Comm,
    ) -> Result<Self> {
        let p = format!("layers.{layer_id}.engram");
        Ok(Self {
            layer_id,
            table_idx,
            k_weight: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.k_weight"), comm)?,
            q_weight: load_raw_buf_replicated(idx.as_deref_mut(), &format!("{p}.q_weight"), comm)?,
            wkv: Exl3Linear::load_replicated(idx.as_deref_mut(), &format!("{p}.wkv"), comm)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engram_primes_match_config() {
        let (p1, p2) = generate_engram_primes();
        assert_eq!(p1.iter().sum::<i64>(), 384_006_168);
        assert_eq!(p2.iter().sum::<i64>(), 384_016_682);
    }

    #[test]
    fn e4m3_lut_exact_values() {
        let lut = build_e4m3_lut();
        assert_eq!(lut[0x38], 1.0);
        assert_eq!(lut[0xb8], -1.0);
        assert_eq!(lut[0x40], 2.0);
    }
}
