use anyhow::{anyhow, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct StTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct SafetensorsFile {
    mmap: Mmap,
    data_offset: usize,
    pub tensors: HashMap<String, StTensor>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < 8 {
            return Err(anyhow!("safetensors too short"));
        }
        let hdr_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let data_offset = 8 + hdr_len;
        if data_offset > mmap.len() {
            return Err(anyhow!("invalid safetensors header length"));
        }
        let val: serde_json::Value = serde_json::from_slice(&mmap[8..data_offset])?;
        let obj = val.as_object().ok_or_else(|| anyhow!("header not object"))?;
        let mut tensors = HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("missing shape for {k}"))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let offs = v["data_offsets"].as_array().ok_or_else(|| anyhow!("missing offsets for {k}"))?;
            let start = offs[0].as_u64().unwrap_or(0) as usize;
            let end = offs[1].as_u64().unwrap_or(0) as usize;
            tensors.insert(k.clone(), StTensor { dtype, shape, start, end });
        }
        Ok(Self { mmap, data_offset, tensors })
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<(&StTensor, &[u8])> {
        let t = self.tensors.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let s = self.data_offset + t.start;
        let e = self.data_offset + t.end;
        Ok((t, &self.mmap[s..e]))
    }
}
