use rad_core::Tokenizer;
use serde_json::Value;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Expect {
    Start,
    ObjKeyOrEnd,
    ObjKey,
    Colon,
    Value,
    CommaOrEnd,
    ArrayValOrEnd,
    InString { is_key: bool, escaped: bool },
    InNumber,
    InLiteral(&'static [u8], usize),
    Done,
}

#[derive(Clone)]
pub struct JsonGrammarState {
    stack: Vec<Container>,
    expect: Expect,
    max_depth: usize,
}

impl JsonGrammarState {
    pub fn new_json() -> Self {
        Self {
            stack: Vec::new(),
            expect: Expect::Start,
            max_depth: 16,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.expect == Expect::Done
    }

    pub fn step_byte(&mut self, b: u8) -> bool {
        match self.expect {
            Expect::InString { is_key, escaped } => {
                if escaped {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u' => {
                            self.expect = Expect::InString { is_key, escaped: false };
                            true
                        }
                        _ => false,
                    }
                } else if b == b'\\' {
                    self.expect = Expect::InString { is_key, escaped: true };
                    true
                } else if b == b'"' {
                    if is_key {
                        self.expect = Expect::Colon;
                    } else if self.stack.is_empty() {
                        self.expect = Expect::Done;
                    } else {
                        self.expect = Expect::CommaOrEnd;
                    }
                    true
                } else if b >= 0x20 {
                    true
                } else {
                    false
                }
            }
            Expect::InNumber => {
                if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-') {
                    true
                } else {
                    self.expect = if self.stack.is_empty() { Expect::Done } else { Expect::CommaOrEnd };
                    self.step_byte(b)
                }
            }
            Expect::InLiteral(lit, idx) => {
                if idx < lit.len() && b == lit[idx] {
                    if idx + 1 == lit.len() {
                        self.expect = if self.stack.is_empty() { Expect::Done } else { Expect::CommaOrEnd };
                    } else {
                        self.expect = Expect::InLiteral(lit, idx + 1);
                    }
                    true
                } else {
                    false
                }
            }
            Expect::Start => {
                if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                    true
                } else if b == b'{' {
                    self.stack.push(Container::Object);
                    self.expect = Expect::ObjKeyOrEnd;
                    true
                } else if b == b'[' {
                    self.stack.push(Container::Array);
                    self.expect = Expect::ArrayValOrEnd;
                    true
                } else {
                    false
                }
            }
            Expect::ObjKeyOrEnd => {
                if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                    true
                } else if b == b'}' {
                    self.stack.pop();
                    self.expect = if self.stack.is_empty() { Expect::Done } else { Expect::CommaOrEnd };
                    true
                } else if b == b'"' {
                    self.expect = Expect::InString { is_key: true, escaped: false };
                    true
                } else {
                    false
                }
            }
            Expect::ObjKey => {
                if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                    true
                } else if b == b'"' {
                    self.expect = Expect::InString { is_key: true, escaped: false };
                    true
                } else {
                    false
                }
            }
            Expect::Colon => {
                if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                    true
                } else if b == b':' {
                    self.expect = Expect::Value;
                    true
                } else {
                    false
                }
            }
            Expect::ArrayValOrEnd => {
                if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                    true
                } else if b == b']' {
                    self.stack.pop();
                    self.expect = if self.stack.is_empty() { Expect::Done } else { Expect::CommaOrEnd };
                    true
                } else {
                    self.expect = Expect::Value;
                    self.step_byte(b)
                }
            }
            Expect::Value => {
                if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                    true
                } else if b == b'"' {
                    self.expect = Expect::InString { is_key: false, escaped: false };
                    true
                } else if b == b'{' {
                    if self.stack.len() >= self.max_depth {
                        return false;
                    }
                    self.stack.push(Container::Object);
                    self.expect = Expect::ObjKeyOrEnd;
                    true
                } else if b == b'[' {
                    if self.stack.len() >= self.max_depth {
                        return false;
                    }
                    self.stack.push(Container::Array);
                    self.expect = Expect::ArrayValOrEnd;
                    true
                } else if b == b'-' || b.is_ascii_digit() {
                    self.expect = Expect::InNumber;
                    true
                } else if b == b't' {
                    self.expect = Expect::InLiteral(b"true", 1);
                    true
                } else if b == b'f' {
                    self.expect = Expect::InLiteral(b"false", 1);
                    true
                } else if b == b'n' {
                    self.expect = Expect::InLiteral(b"null", 1);
                    true
                } else {
                    false
                }
            }
            Expect::CommaOrEnd => {
                if matches!(b, b' ' | b'\n' | b'\r' | b'\t') {
                    true
                } else if b == b',' {
                    match self.stack.last() {
                        Some(Container::Object) => {
                            self.expect = Expect::ObjKey;
                            true
                        }
                        Some(Container::Array) => {
                            self.expect = Expect::Value;
                            true
                        }
                        None => false,
                    }
                } else if b == b'}' && self.stack.last() == Some(&Container::Object) {
                    self.stack.pop();
                    self.expect = if self.stack.is_empty() { Expect::Done } else { Expect::CommaOrEnd };
                    true
                } else if b == b']' && self.stack.last() == Some(&Container::Array) {
                    self.stack.pop();
                    self.expect = if self.stack.is_empty() { Expect::Done } else { Expect::CommaOrEnd };
                    true
                } else {
                    false
                }
            }
            Expect::Done => matches!(b, b' ' | b'\n' | b'\r' | b'\t'),
        }
    }

    pub fn accepts_bytes(&self, bytes: &[u8]) -> Option<Self> {
        let mut st = self.clone();
        for &b in bytes {
            if !st.step_byte(b) {
                return None;
            }
        }
        Some(st)
    }
}

pub struct VocabGrammarTable {
    pub token_bytes: Vec<Vec<u8>>,
    pub stop_ids: Vec<i32>,
}

impl VocabGrammarTable {
    pub fn build(tokenizer: &Tokenizer, vocab_size: usize) -> Self {
        let mut token_bytes = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size {
            let s = tokenizer.decode(&[id as u32], false).unwrap_or_default();
            token_bytes.push(s.into_bytes());
        }
        Self {
            token_bytes,
            stop_ids: vec![200006, 200008],
        }
    }

    pub fn select_best_token(&self, logits: &[f32], state: &mut JsonGrammarState) -> i32 {
        let n = logits.len().min(self.token_bytes.len());
        let mut top: Vec<(f32, usize)> = Vec::with_capacity(64);
        let mut min_top = f32::NEG_INFINITY;
        for (i, &v) in logits[..n].iter().enumerate() {
            if top.len() < 64 {
                top.push((v, i));
                if top.len() == 64 {
                    top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    min_top = top[63].0;
                }
            } else if v > min_top {
                top[63] = (v, i);
                let mut k = 63;
                while k > 0 && top[k].0 > top[k - 1].0 {
                    top.swap(k, k - 1);
                    k -= 1;
                }
                min_top = top[63].0;
            }
        }
        top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        for &(_, id) in &top {
            if self.stop_ids.contains(&(id as i32)) {
                if state.is_complete() {
                    return id as i32;
                }
                continue;
            }
            let bytes = &self.token_bytes[id];
            if bytes.is_empty() {
                continue;
            }
            if let Some(next_st) = state.accepts_bytes(bytes) {
                *state = next_st;
                return id as i32;
            }
        }

        let mut best_id = if state.is_complete() { 200006i32 } else { 0i32 };
        let mut best_val = f32::NEG_INFINITY;
        let mut best_state: Option<JsonGrammarState> = None;
        for (i, &v) in logits[..n].iter().enumerate() {
            if v <= best_val {
                continue;
            }
            if self.stop_ids.contains(&(i as i32)) {
                if state.is_complete() {
                    best_val = v;
                    best_id = i as i32;
                    best_state = None;
                }
                continue;
            }
            let bytes = &self.token_bytes[i];
            if bytes.is_empty() {
                continue;
            }
            if let Some(next_st) = state.accepts_bytes(bytes) {
                best_val = v;
                best_id = i as i32;
                best_state = Some(next_st);
            }
        }
        if let Some(ns) = best_state {
            *state = ns;
        }
        best_id
    }
}

pub fn parse_tool_calls(text: &str, allowed_tools: &[String]) -> Option<Vec<Value>> {
    let trimmed = text.trim();
    let candidate = if let Some(s) = trimmed.strip_prefix("```json") {
        s.strip_suffix("```").unwrap_or(s).trim()
    } else if let Some(s) = trimmed.strip_prefix("```") {
        s.strip_suffix("```").unwrap_or(s).trim()
    } else {
        trimmed
    };
    let val: Value = serde_json::from_str(candidate).ok()?;
    let items = if let Some(arr) = val.as_array() {
        arr.clone()
    } else if let Some(arr) = val.get("tool_calls").and_then(|v| v.as_array()) {
        arr.clone()
    } else if val.get("name").is_some() {
        vec![val]
    } else {
        return None;
    };
    let mut out = Vec::new();
    for (idx, item) in items.iter().enumerate() {
        let name = item.get("name").and_then(|v| v.as_str())?;
        if !allowed_tools.is_empty() && !allowed_tools.iter().any(|a| a == name) {
            continue;
        }
        let args = item.get("arguments").cloned().unwrap_or_else(|| serde_json::json!({}));
        let args_str = if let Some(s) = args.as_str() {
            s.to_string()
        } else {
            serde_json::to_string(&args).unwrap_or_else(|_| "{}".into())
        };
        out.push(serde_json::json!({
            "id": format!("call_rad_{idx}"),
            "type": "function",
            "function": {
                "name": name,
                "arguments": args_str
            }
        }));
    }
    if out.is_empty() { None } else { Some(out) }
}
