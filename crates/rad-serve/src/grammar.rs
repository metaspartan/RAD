use rad_core::Tokenizer;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawToolCall {
    pub inline_after_text: bool,
    pub raw_prefix_text: String,
    pub raw_payload: String,
}

static RAW_TOOL_CALL_CACHE: OnceLock<Mutex<VecDeque<(String, RawToolCall)>>> = OnceLock::new();

pub fn remember_raw_tool_call(id: &str, entry: RawToolCall) {
    if id.is_empty() || entry.raw_payload.is_empty() {
        return;
    }
    let cache = RAW_TOOL_CALL_CACHE.get_or_init(|| Mutex::new(VecDeque::with_capacity(512)));
    if let Ok(mut guard) = cache.lock() {
        if let Some(pos) = guard.iter().position(|(k, _)| k == id) {
            guard.remove(pos);
        }
        if guard.len() >= 512 {
            guard.pop_front();
        }
        guard.push_back((id.to_string(), entry));
    }
}

pub fn lookup_raw_tool_call(id: &str) -> Option<RawToolCall> {
    let cache = RAW_TOOL_CALL_CACHE.get()?;
    let guard = cache.lock().ok()?;
    guard.iter().rev().find(|(k, _)| k == id).map(|(_, v)| v.clone())
}

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

pub const TOK_ENDOFTEXT: u32 = 199999;
pub const TOK_MSG_MODEL: u32 = 200001;
pub const TOK_CONTENT_TEXT: u32 = 200004;
pub const TOK_END_SAMPLING: u32 = 200006;
pub const TOK_CONTENT_THINKING: u32 = 200008;
pub const TOK_END_MESSAGE: u32 = 200010;
pub const TOK_INVOKE_TOOL_JSON: u32 = 200049;
pub const TOK_INVOKE_TOOL_TEXT: u32 = 200057;

#[derive(Clone, Debug)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ParsedToolCall {
    pub fn to_chat_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments
            }
        })
    }
}

pub struct ParsedTurn {
    pub reasoning: Option<String>,
    pub content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

fn extract_args_str(item: &Value) -> String {
    let args = item
        .get("arguments")
        .or_else(|| item.get("args"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    if let Some(s) = args.as_str() {
        s.to_string()
    } else {
        serde_json::to_string(&args).unwrap_or_else(|_| "{}".into())
    }
}

fn extract_balanced_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for i in start..bytes.len() {
        let b = bytes[i];
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else {
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&s[start..=i]);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

pub fn parse_single_tool_call(
    payload: &str,
    fallback_name: &str,
    allowed_tools: &[String],
    call_id: String,
) -> Option<ParsedToolCall> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return None;
    }
    let fb = fallback_name.trim();
    let try_parse = |json_str: &str, cand_name: &str| -> Option<ParsedToolCall> {
        let parsed = serde_json::from_str::<Value>(json_str).ok()?;
        let val = if let Some(arr) = parsed.as_array() {
            arr.first()?.clone()
        } else {
            parsed
        };
        if !val.is_object() {
            return None;
        }
        let mut name = val
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(cand_name);
        if name.is_empty() && allowed_tools.len() == 1 {
            name = allowed_tools[0].as_str();
        }
        if name.is_empty() || (!allowed_tools.is_empty() && !allowed_tools.iter().any(|a| a == name)) {
            return None;
        }
        let arguments = if val.get("arguments").is_some() || val.get("args").is_some() {
            extract_args_str(&val)
        } else if let Some(obj) = val.as_object() {
            let mut cleaned = obj.clone();
            cleaned.remove("name");
            serde_json::to_string(&Value::Object(cleaned)).unwrap_or_else(|_| "{}".into())
        } else {
            serde_json::to_string(&val).unwrap_or_else(|_| "{}".into())
        };
        Some(ParsedToolCall {
            id: call_id.clone(),
            name: name.to_string(),
            arguments,
        })
    };
    if let Some(tc) = try_parse(trimmed, fb) {
        return Some(tc);
    }
    if let Some(brace) = trimmed.find('{') {
        let prefix = trimmed[..brace].trim();
        let cand_name = if !prefix.is_empty() { prefix } else { fb };
        if let Some(tc) = try_parse(trimmed[brace..].trim(), cand_name) {
            return Some(tc);
        }
        if let Some(bal) = extract_balanced_json_object(trimmed) {
            if let Some(tc) = try_parse(bal, cand_name) {
                return Some(tc);
            }
        }
    }
    None
}

#[derive(Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-5 && (self.repetition_penalty - 1.0).abs() <= 1e-4
    }
}

pub fn sample_from_logits(logits: &[f32], recent_tokens: &[i32], params: &mut SamplingParams) -> i32 {
    let n = logits.len();
    if n == 0 {
        return 0;
    }
    let rep = params.repetition_penalty;
    let temp = params.temperature.max(1e-4);
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(64);
    let mut min_top = f32::NEG_INFINITY;
    let mut recent_set: Vec<usize> = Vec::new();
    if (rep - 1.0).abs() > 1e-4 {
        let start = recent_tokens.len().saturating_sub(128);
        for &t in &recent_tokens[start..] {
            if t >= 0 && (t as usize) < n {
                recent_set.push(t as usize);
            }
        }
        recent_set.sort_unstable();
        recent_set.dedup();
    }
    for (i, &raw) in logits.iter().enumerate() {
        let mut v = raw;
        if !recent_set.is_empty() && recent_set.binary_search(&i).is_ok() {
            v = if v > 0.0 { v / rep } else { v * rep };
        }
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
    if params.temperature <= 1e-5 {
        return top[0].1 as i32;
    }
    if params.top_k > 0 && params.top_k < top.len() {
        top.truncate(params.top_k);
    }
    let max_l = top[0].0;
    let mut probs: Vec<(f32, usize)> = Vec::with_capacity(top.len());
    let mut sum = 0.0f32;
    for &(v, idx) in &top {
        let p = ((v - max_l) / temp).exp();
        probs.push((p, idx));
        sum += p;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return top[0].1 as i32;
    }
    let min_p = params.min_p.clamp(0.0, 1.0);
    if min_p > 0.0 {
        let thresh = probs[0].0 * min_p;
        probs.retain(|x| x.0 >= thresh);
        if probs.is_empty() {
            return top[0].1 as i32;
        }
        sum = probs.iter().map(|x| x.0).sum();
    }
    let top_p = params.top_p.clamp(0.05, 1.0);
    let mut cum = 0.0f32;
    let mut keep = probs.len();
    for (k, &(p, _)) in probs.iter().enumerate() {
        cum += p / sum;
        if cum >= top_p {
            keep = (k + 1).max(1);
            break;
        }
    }
    probs.truncate(keep);
    let kept_sum: f32 = probs.iter().map(|x| x.0).sum();
    params.seed = params.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (((params.seed >> 33) as u32) as f32 / (u32::MAX as f32)) * kept_sum;
    let mut acc = 0.0f32;
    for &(p, idx) in &probs {
        acc += p;
        if acc >= r {
            return idx as i32;
        }
    }
    probs[0].1 as i32
}

pub fn sample_from_topk(vals: &[f32], ids: &[i32], recent_tokens: &[i32], params: &mut SamplingParams) -> i32 {
    let n = vals.len().min(ids.len());
    if n == 0 {
        return 0;
    }
    let rep = params.repetition_penalty;
    let temp = params.temperature.max(1e-4);
    let mut top: Vec<(f32, i32)> = Vec::with_capacity(n);
    let has_rep = (rep - 1.0).abs() > 1e-4;
    let recent = if has_rep {
        &recent_tokens[recent_tokens.len().saturating_sub(128)..]
    } else {
        &[]
    };
    for i in 0..n {
        let id = ids[i];
        let mut v = vals[i];
        if has_rep && recent.contains(&id) {
            v = if v > 0.0 { v / rep } else { v * rep };
        }
        top.push((v, id));
    }
    top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    if params.temperature <= 1e-5 {
        return top[0].1;
    }
    if params.top_k > 0 && params.top_k < top.len() {
        top.truncate(params.top_k);
    }
    let max_l = top[0].0;
    let mut probs: Vec<(f32, i32)> = Vec::with_capacity(n);
    let mut sum = 0.0f32;
    for &(v, id) in &top {
        let p = ((v - max_l) / temp).exp();
        probs.push((p, id));
        sum += p;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return top[0].1;
    }
    let min_p = params.min_p.clamp(0.0, 1.0);
    if min_p > 0.0 {
        let thresh = probs[0].0 * min_p;
        probs.retain(|x| x.0 >= thresh);
        if probs.is_empty() {
            return top[0].1;
        }
        sum = probs.iter().map(|x| x.0).sum();
    }
    let top_p = params.top_p.clamp(0.05, 1.0);
    let mut cum = 0.0f32;
    let mut keep = probs.len();
    for (k, &(p, _)) in probs.iter().enumerate() {
        cum += p / sum;
        if cum >= top_p {
            keep = (k + 1).max(1);
            break;
        }
    }
    probs.truncate(keep);
    let kept_sum: f32 = probs.iter().map(|x| x.0).sum();
    params.seed = params.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (((params.seed >> 33) as u32) as f32 / (u32::MAX as f32)) * kept_sum;
    let mut acc = 0.0f32;
    for &(p, id) in &probs {
        acc += p;
        if acc >= r {
            return id;
        }
    }
    probs[0].1
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
        let name = match item.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None if allowed_tools.len() == 1 && item.is_object() => allowed_tools[0].as_str(),
            None => return None,
        };
        if !allowed_tools.is_empty() && !allowed_tools.iter().any(|a| a == name) {
            continue;
        }
        let args_str = if item.get("arguments").is_some() || item.get("args").is_some() {
            extract_args_str(item)
        } else if let Some(obj) = item.as_object() {
            let mut cleaned = obj.clone();
            cleaned.remove("name");
            serde_json::to_string(&Value::Object(cleaned)).unwrap_or_else(|_| "{}".into())
        } else {
            extract_args_str(item)
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    Thinking,
    Text,
    PendingHeader,
    ToolJson,
    ToolText,
}

pub struct InklingStreamParser {
    pub mode: StreamMode,
    thinking_ids: Vec<u32>,
    thinking_win: usize,
    thinking_prefix_len: usize,
    text_ids: Vec<u32>,
    text_win: usize,
    text_prefix_len: usize,
    header_ids: Vec<u32>,
    tool_ids: Vec<u32>,
    tool_name_hint: String,
    tool_inline_after_text: bool,
    tool_inline_prefix: String,
    call_prefix: String,
    allowed_tools: Vec<String>,
    pub emitted_calls: Vec<ParsedToolCall>,
}

pub enum StreamDelta {
    Reasoning(String),
    Text(String),
    ToolCall(ParsedToolCall),
}

fn decode_incremental_delta(ids: &[u32], win_start: &mut usize, prefix_len: &mut usize, tok: &Tokenizer) -> Option<String> {
    let slice = &ids[*win_start..];
    let decoded = tok.decode(slice, true).ok()?;
    if decoded.ends_with('\u{FFFD}') {
        return None;
    }
    if decoded.len() <= *prefix_len {
        return None;
    }
    let delta = decoded[*prefix_len..].to_string();
    if slice.len() >= 4 {
        let keep = ids.len() - 1;
        *win_start = keep;
        *prefix_len = tok.decode(&ids[keep..], true).map(|s| s.len()).unwrap_or(0);
    } else {
        *prefix_len = decoded.len();
    }
    Some(delta)
}

impl InklingStreamParser {
    pub fn new(initial_thinking: bool, call_prefix: String, allowed_tools: Vec<String>) -> Self {
        Self {
            mode: if initial_thinking { StreamMode::Thinking } else { StreamMode::Text },
            thinking_ids: Vec::new(),
            thinking_win: 0,
            thinking_prefix_len: 0,
            text_ids: Vec::new(),
            text_win: 0,
            text_prefix_len: 0,
            header_ids: Vec::new(),
            tool_ids: Vec::new(),
            tool_name_hint: String::new(),
            tool_inline_after_text: false,
            tool_inline_prefix: String::new(),
            call_prefix,
            allowed_tools,
            emitted_calls: Vec::new(),
        }
    }

    fn flush_tool(&mut self, tok: &Tokenizer) -> Option<ParsedToolCall> {
        if self.tool_ids.is_empty() {
            return None;
        }
        let payload = tok.decode(&self.tool_ids, true).unwrap_or_default();
        self.tool_ids.clear();
        let idx = self.emitted_calls.len();
        let cid = format!("{}_{}", self.call_prefix, idx);
        let hint = std::mem::take(&mut self.tool_name_hint);
        let inline_after_text = self.tool_inline_after_text;
        let raw_prefix_text = std::mem::take(&mut self.tool_inline_prefix);
        self.tool_inline_after_text = false;
        if let Some(tc) = parse_single_tool_call(&payload, &hint, &self.allowed_tools, cid) {
            remember_raw_tool_call(
                &tc.id,
                RawToolCall {
                    inline_after_text,
                    raw_prefix_text,
                    raw_payload: payload,
                },
            );
            self.emitted_calls.push(tc.clone());
            return Some(tc);
        }
        None
    }

    pub fn push_token(&mut self, t: u32, tok: &Tokenizer) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        match t {
            TOK_END_SAMPLING | TOK_ENDOFTEXT => {
                if matches!(self.mode, StreamMode::ToolJson | StreamMode::ToolText) {
                    if let Some(tc) = self.flush_tool(tok) {
                        out.push(StreamDelta::ToolCall(tc));
                    }
                }
            }
            TOK_END_MESSAGE => {
                if matches!(self.mode, StreamMode::ToolJson | StreamMode::ToolText) {
                    if let Some(tc) = self.flush_tool(tok) {
                        out.push(StreamDelta::ToolCall(tc));
                    }
                }
                self.header_ids.clear();
                self.mode = StreamMode::PendingHeader;
            }
            TOK_MSG_MODEL => {
                if matches!(self.mode, StreamMode::ToolJson | StreamMode::ToolText) {
                    if let Some(tc) = self.flush_tool(tok) {
                        out.push(StreamDelta::ToolCall(tc));
                    }
                }
                self.header_ids.clear();
                self.mode = StreamMode::PendingHeader;
            }
            TOK_CONTENT_THINKING => {
                self.header_ids.clear();
                self.mode = StreamMode::Thinking;
            }
            TOK_CONTENT_TEXT => {
                self.header_ids.clear();
                self.mode = StreamMode::Text;
            }
            TOK_INVOKE_TOOL_JSON => {
                if self.mode == StreamMode::Text {
                    self.tool_inline_after_text = true;
                    self.tool_inline_prefix = tok.decode(&self.text_ids, true).unwrap_or_default();
                    self.tool_name_hint = self.tool_inline_prefix.trim().to_string();
                } else {
                    self.tool_inline_after_text = false;
                    self.tool_inline_prefix = tok.decode(&self.text_ids, true).unwrap_or_default();
                    self.tool_name_hint = tok.decode(&self.header_ids, true).unwrap_or_default().trim().to_string();
                }
                self.header_ids.clear();
                self.tool_ids.clear();
                self.mode = StreamMode::ToolJson;
            }
            TOK_INVOKE_TOOL_TEXT => {
                if self.mode == StreamMode::Text {
                    self.tool_inline_after_text = true;
                    self.tool_inline_prefix = tok.decode(&self.text_ids, true).unwrap_or_default();
                    self.tool_name_hint = self.tool_inline_prefix.trim().to_string();
                } else {
                    self.tool_inline_after_text = false;
                    self.tool_inline_prefix = tok.decode(&self.text_ids, true).unwrap_or_default();
                    self.tool_name_hint = tok.decode(&self.header_ids, true).unwrap_or_default().trim().to_string();
                }
                self.header_ids.clear();
                self.tool_ids.clear();
                self.mode = StreamMode::ToolText;
            }
            _ => match self.mode {
                StreamMode::Thinking => {
                    self.thinking_ids.push(t);
                    if let Some(s) = decode_incremental_delta(&self.thinking_ids, &mut self.thinking_win, &mut self.thinking_prefix_len, tok) {
                        out.push(StreamDelta::Reasoning(s));
                    }
                }
                StreamMode::Text => {
                    self.text_ids.push(t);
                    if let Some(s) = decode_incremental_delta(&self.text_ids, &mut self.text_win, &mut self.text_prefix_len, tok) {
                        out.push(StreamDelta::Text(s));
                    }
                }
                StreamMode::PendingHeader => {
                    self.header_ids.push(t);
                }
                StreamMode::ToolJson | StreamMode::ToolText => {
                    self.tool_ids.push(t);
                }
            },
        }
        out
    }

    pub fn finish(&mut self, tok: &Tokenizer) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        if matches!(self.mode, StreamMode::ToolJson | StreamMode::ToolText) {
            if let Some(tc) = self.flush_tool(tok) {
                out.push(StreamDelta::ToolCall(tc));
            }
        } else if self.mode == StreamMode::PendingHeader && !self.header_ids.is_empty() {
            let ids = std::mem::take(&mut self.header_ids);
            self.text_ids.extend_from_slice(&ids);
            if let Ok(s) = tok.decode(&ids, true) {
                if !s.is_empty() {
                    out.push(StreamDelta::Text(s));
                }
            }
        }
        out
    }

    pub fn full_text(&self, tok: &Tokenizer) -> String {
        tok.decode(&self.text_ids, true).unwrap_or_default()
    }

    pub fn full_reasoning(&self, tok: &Tokenizer) -> Option<String> {
        if self.thinking_ids.is_empty() {
            None
        } else {
            let s = tok.decode(&self.thinking_ids, true).unwrap_or_default().trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        }
    }
}

pub fn parse_inkling_tokens(
    tokens: &[u32],
    tok: &Tokenizer,
    initial_thinking: bool,
    allowed_tools: &[String],
    call_prefix: &str,
) -> ParsedTurn {
    let mut p = InklingStreamParser::new(initial_thinking, call_prefix.to_string(), allowed_tools.to_vec());
    for &t in tokens {
        let _ = p.push_token(t, tok);
    }
    let _ = p.finish(tok);
    let mut content = p.full_text(tok).trim().to_string();
    let reasoning = p.full_reasoning(tok);
    let mut tool_calls = p.emitted_calls;
    if tool_calls.is_empty() && !allowed_tools.is_empty() {
        if let Some(vals) = parse_tool_calls(&content, allowed_tools) {
            for (idx, v) in vals.into_iter().enumerate() {
                if let (Some(name), Some(args)) = (
                    v.get("function").and_then(|f| f.get("name")).and_then(|x| x.as_str()),
                    v.get("function").and_then(|f| f.get("arguments")).and_then(|x| x.as_str()),
                ) {
                    tool_calls.push(ParsedToolCall {
                        id: format!("{call_prefix}_{idx}"),
                        name: name.to_string(),
                        arguments: args.to_string(),
                    });
                }
            }
            if !tool_calls.is_empty() {
                content.clear();
            }
        }
    }
    ParsedTurn {
        reasoning,
        content,
        tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_native_args_and_balanced_json() {
        let allowed = vec!["exec".to_string()];
        let tc = parse_single_tool_call(
            r#"{"name":"exec","args":{"command":"git status"}}"#,
            "exec",
            &allowed,
            "c0".into(),
        )
        .unwrap();
        assert_eq!(tc.name, "exec");
        assert_eq!(tc.arguments, r#"{"command":"git status"}"#);

        let tc2 = parse_single_tool_call(
            r#"exec{"name":"exec","args":{"command":"ls"}} trailing junk"#,
            "exec",
            &allowed,
            "c1".into(),
        )
        .unwrap();
        assert_eq!(tc2.arguments, r#"{"command":"ls"}"#);

        let tc3 = parse_single_tool_call(
            r#"{"name":"exec","command":"git log -1"}"#,
            "exec",
            &allowed,
            "c2".into(),
        )
        .unwrap();
        assert_eq!(tc3.arguments, r#"{"command":"git log -1"}"#);

        let tc4 = parse_single_tool_call(
            r#"[{"command":"ls -la"}]"#,
            "",
            &allowed,
            "c4".into(),
        )
        .unwrap();
        assert_eq!(tc4.name, "exec");
        assert_eq!(tc4.arguments, r#"{"command":"ls -la"}"#);

        assert!(parse_single_tool_call("not valid json at all", "exec", &allowed, "c3".into()).is_none());
    }

    #[test]
    fn raw_tool_call_cache_roundtrip() {
        let entry = RawToolCall {
            inline_after_text: true,
            raw_prefix_text: "read_file".into(),
            raw_payload: r#"{"name": "read_file", "args": {"path": "src/main.rs", "limit": 50}}"#.into(),
        };
        remember_raw_tool_call("call_test_roundtrip_0", entry.clone());
        assert_eq!(lookup_raw_tool_call("call_test_roundtrip_0"), Some(entry));
        assert_eq!(lookup_raw_tool_call("call_nonexistent_xyz"), None);
    }

    #[test]
    fn sampling_repetition_penalty_shifts_top_token() {
        let logits = vec![5.0f32, 4.5f32, 1.0f32];
        let mut p = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.2,
            seed: 42,
        };
        assert_eq!(sample_from_logits(&logits, &[0], &mut p), 1);
    }

    #[test]
    fn sampling_top_k_and_min_p_filters() {
        let logits = vec![10.0f32, 9.9f32, 2.0f32, 1.0f32];
        let mut p_k1 = SamplingParams {
            temperature: 0.8,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 12345,
        };
        for _ in 0..16 {
            assert_eq!(sample_from_logits(&logits, &[], &mut p_k1), 0);
        }
        let mut p_minp = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.5,
            repetition_penalty: 1.0,
            seed: 99,
        };
        for _ in 0..32 {
            let tok = sample_from_logits(&logits, &[], &mut p_minp);
            assert!(tok == 0 || tok == 1);
        }
        let vals = [10.0f32, 9.9f32, 2.0f32, 1.0f32];
        let ids = [100i32, 200, 300, 400];
        let mut p_tk = SamplingParams {
            temperature: 0.8,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 7,
        };
        assert_eq!(sample_from_topk(&vals, &ids, &[], &mut p_tk), 100);
    }
}
