use anyhow::{anyhow, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

struct ShardFile {
    mmap: Mmap,
    data_base: usize,
    tensors: HashMap<String, TensorMeta>,
}

pub struct SafetensorsIndex {
    model_dir: PathBuf,
    weight_map: HashMap<String, String>,
    shards: HashMap<String, ShardFile>,
}

#[derive(Deserialize)]
struct IndexJson {
    weight_map: HashMap<String, String>,
}

impl SafetensorsIndex {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(model_dir.join("model.safetensors.index.json"))?;
        let idx: IndexJson = serde_json::from_str(&raw)?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            weight_map: idx.weight_map,
            shards: HashMap::new(),
        })
    }

    #[allow(dead_code)]
    pub fn contains(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    fn ensure_shard(&mut self, shard_name: &str) -> Result<&ShardFile> {
        if !self.shards.contains_key(shard_name) {
            let file = File::open(self.model_dir.join(shard_name))?;
            let mmap = unsafe { Mmap::map(&file)? };
            if mmap.len() < 8 {
                return Err(anyhow!("corrupt safetensors shard {shard_name}"));
            }
            let hdr_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let hdr_end = 8 + hdr_len;
            if hdr_end > mmap.len() {
                return Err(anyhow!("safetensors header out of bounds in {shard_name}"));
            }
            let raw_map: HashMap<String, serde_json::Value> = serde_json::from_slice(&mmap[8..hdr_end])?;
            let mut tensors = HashMap::with_capacity(raw_map.len());
            for (k, v) in raw_map {
                if k == "__metadata__" {
                    continue;
                }
                if let Ok(meta) = serde_json::from_value::<TensorMeta>(v) {
                    tensors.insert(k, meta);
                }
            }
            self.shards.insert(
                shard_name.to_string(),
                ShardFile {
                    mmap,
                    data_base: hdr_end,
                    tensors,
                },
            );
        }
        Ok(self.shards.get(shard_name).unwrap())
    }

    pub fn get_tensor(&mut self, name: &str) -> Result<(TensorMeta, &[u8])> {
        let shard_name = self
            .weight_map
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?
            .clone();
        let shard = self.ensure_shard(&shard_name)?;
        let meta = shard
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} missing in shard {shard_name}"))?
            .clone();
        let start = shard.data_base + meta.data_offsets[0];
        let end = shard.data_base + meta.data_offsets[1];
        Ok((meta, &shard.mmap[start..end]))
    }

    pub fn exl3_shape_and_bits(&mut self, prefix: &str) -> Result<(usize, usize, usize)> {
        let (meta, _) = self.get_tensor(&format!("{prefix}.trellis"))?;
        if meta.shape.len() != 3 {
            return Err(anyhow!("unexpected trellis shape {:?} for {prefix}", meta.shape));
        }
        let k = meta.shape[0] * 16;
        let n = meta.shape[1] * 16;
        let bits = meta.shape[2] / 16;
        Ok((k, n, bits))
    }
}
