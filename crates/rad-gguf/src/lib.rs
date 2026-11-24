use memmap2::Mmap;
use std::{collections::HashMap, fs::File, path::Path};

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub ggml_type: u32,
    pub offset: u64,
    pub size: u64,
}

pub struct GgufFile {
    mmap: Mmap,
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub version: u32,
}

impl GgufFile {
    pub fn open(p: &Path) -> std::io::Result<Self> {
        let f = File::open(p)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let mut c = Cursor { b: &mmap[..], pos: 0 };
        let mut magic = [0u8; 4];
        c.read_into(&mut magic);
        assert_eq!(magic, GGUF_MAGIC, "not a gguf file");
        let version = c.u32();
        assert!(version >= 2 && version <= 3, "unsupported gguf v{version}");
        let n_tensors = c.u64();
        let n_kv = c.u64();
        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let key = c.string();
            let val = c.value();
            kv.insert(key, val);
        }
        let alignment = match kv.get("general.alignment") {
            Some(Value::U32(v)) => *v as usize,
            Some(Value::U64(v)) => *v as usize,
            Some(Value::I32(v)) => *v as usize,
            _ => 32,
        };
        let mut raw_tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string();
            let n_dims = c.u32();
            let mut dims = [1u64; 4];
            for d in dims.iter_mut().take(n_dims as usize) {
                *d = c.u64();
            }
            let ggml_type = c.u32();
            let rel_offset = c.u64();
            let tsize = tensor_byte_size(ggml_type, &dims[..n_dims as usize]);
            raw_tensors.push((name, n_dims, dims, ggml_type, rel_offset, tsize));
        }
        let data_start = c.align(alignment) as u64;
        let mut tensors = HashMap::with_capacity(n_tensors as usize);
        for (name, n_dims, dims, ggml_type, rel_offset, size) in raw_tensors {
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    n_dims,
                    dims,
                    ggml_type,
                    offset: data_start + rel_offset,
                    size,
                },
            );
        }
        Ok(Self { mmap, kv, tensors, version })
    }

    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensors.get(name)?;
        Some(&self.mmap[t.offset as usize..(t.offset + t.size) as usize])
    }

    pub fn kv_u64(&self, k: &str) -> Option<u64> {
        match self.kv.get(k)? {
            Value::U64(v) => Some(*v),
            Value::U32(v) => Some(*v as u64),
            Value::I32(v) => Some(*v as u64),
            Value::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn kv_f32(&self, k: &str) -> Option<f32> {
        match self.kv.get(k)? {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn kv_str(&self, k: &str) -> Option<&str> {
        match self.kv.get(k)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn kv_bool_arr(&self, k: &str) -> Option<Vec<bool>> {
        match self.kv.get(k)? {
            Value::Arr(a) => Some(
                a.iter()
                    .map(|v| match v {
                        Value::Bool(b) => *b,
                        Value::U8(x) => *x != 0,
                        Value::I32(x) => *x != 0,
                        Value::U32(x) => *x != 0,
                        _ => false,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        43 => "F8_E4M3",
        _ => "UNKNOWN",
    }
}

pub fn ggml_type_block(t: u32) -> (u64, u64) {
    match t {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        39 => (32, 17),
        40 => (64, 36),
        43 => (1, 1),
        _ => panic!("unsupported ggml_type {t}"),
    }
}

fn tensor_byte_size(t: u32, dims: &[u64]) -> u64 {
    let (block, bsize) = ggml_type_block(t);
    if dims.is_empty() {
        return 0;
    }
    let row_blocks = dims[0].div_ceil(block);
    let rows: u64 = dims[1..].iter().product();
    row_blocks * bsize * rows
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn read_into(&mut self, out: &mut [u8]) {
        out.copy_from_slice(&self.b[self.pos..self.pos + out.len()]);
        self.pos += out.len();
    }
    fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.read_into(&mut b);
        u32::from_le_bytes(b)
    }
    fn u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.read_into(&mut b);
        u64::from_le_bytes(b)
    }
    fn string(&mut self) -> String {
        let n = self.u64() as usize;
        let s = String::from_utf8_lossy(&self.b[self.pos..self.pos + n]).into_owned();
        self.pos += n;
        s
    }
    fn align(&mut self, a: usize) -> usize {
        self.pos = (self.pos + a - 1) & !(a - 1);
        self.pos
    }
    fn value(&mut self) -> Value {
        let t = self.u32();
        self.value_of_type(t)
    }
    fn value_of_type(&mut self, t: u32) -> Value {
        match t {
            0 => Value::U8({ let mut b = [0; 1]; self.read_into(&mut b); b[0] }),
            1 => Value::I8({ let mut b = [0; 1]; self.read_into(&mut b); b[0] as i8 }),
            2 => Value::U16({ let mut b = [0; 2]; self.read_into(&mut b); u16::from_le_bytes(b) }),
            3 => Value::I16({ let mut b = [0; 2]; self.read_into(&mut b); i16::from_le_bytes(b) }),
            4 => Value::U32(self.u32()),
            5 => Value::I32(self.u32() as i32),
            6 => Value::F32(f32::from_bits(self.u32())),
            7 => Value::Bool({ let mut b = [0; 1]; self.read_into(&mut b); b[0] != 0 }),
            8 => Value::Str(self.string()),
            9 => {
                let et = self.u32();
                let n = self.u64() as usize;
                let mut v = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    v.push(self.value_of_type(et));
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()),
            11 => Value::I64(self.u64() as i64),
            12 => Value::F64(f64::from_bits(self.u64())),
            _ => panic!("unknown gguf kv type {t} pos={}", self.pos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip_minimal() {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let key = b"inkling.test";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&8u32.to_le_bytes());
        let val = b"inkling2_loop";
        b.extend_from_slice(&(val.len() as u64).to_le_bytes());
        b.extend_from_slice(val);
        while b.len() % 32 != 0 {
            b.push(0);
        }
        let dir = std::env::temp_dir().join("rad_gguf_test.gguf");
        std::fs::write(&dir, &b).unwrap();
        let g = GgufFile::open(&dir).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.kv_str("inkling.test"), Some("inkling2_loop"));
        std::fs::remove_file(&dir).ok();
    }
}
