mod grammar;
mod metrics;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path as AxumPath, Query, Request, State,
    },
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use rad_core::{DFlash2Drafter, DsparkDrafter, GpuTopology, InklingEngine, Tokenizer, VisionEncoder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::VecDeque,
    convert::Infallible,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

enum GenEvent {
    Prefill { prompt_n: usize, cached_n: usize, pp_ms: f64 },
    Token(u32),
    Done { finish_reason: &'static str, pred_n: usize, dec_ms: f64 },
    Error(String),
}

struct GenJob {
    prompt_ids: Vec<i32>,
    images: Vec<String>,
    max_tokens: usize,
    #[allow(dead_code)]
    enable_thinking: bool,
    json_grammar: bool,
    sampling: grammar::SamplingParams,
    tx: mpsc::UnboundedSender<GenEvent>,
}

enum Drafter {
    DFlash2(DFlash2Drafter),
    Dspark(DsparkDrafter),
}

impl Drafter {
    fn spec_k(&self) -> usize {
        match self {
            Drafter::DFlash2(_) => 8,
            Drafter::Dspark(_) => 6,
        }
    }

    fn mode_name(&self) -> &'static str {
        match self {
            Drafter::DFlash2(_) => "dflash2",
            Drafter::Dspark(_) => "dspark",
        }
    }

    #[allow(dead_code)]
    fn ctx_len(&self) -> i32 {
        match self {
            Drafter::DFlash2(d) => d.ctx_len,
            Drafter::Dspark(d) => d.ctx_len,
        }
    }

    fn seq_ctx_len(&self, seq: usize) -> i32 {
        match self {
            Drafter::DFlash2(d) => d.seq_ctx_len.get(seq).copied().unwrap_or(-1),
            Drafter::Dspark(d) => d.seq_ctx_len.get(seq).copied().unwrap_or(-1),
        }
    }

    fn max_seqs(&self) -> usize {
        match self {
            Drafter::DFlash2(d) => d.max_seqs,
            Drafter::Dspark(d) => d.max_seqs,
        }
    }

    fn sync(&self) {
        if let Drafter::DFlash2(d) = self {
            d.sync();
        }
    }

    fn reset_seq(&mut self, seq: usize) {
        match self {
            Drafter::DFlash2(d) => d.reset_seq(seq),
            Drafter::Dspark(d) => d.reset_seq(seq),
        }
    }

    #[allow(dead_code)]
    fn inject_target_feats_seq(&mut self, seq: usize, feats_ptr: *const f32, pos: i32) -> anyhow::Result<()> {
        match self {
            Drafter::DFlash2(d) => d.inject_target_feats_seq(seq, feats_ptr, pos),
            Drafter::Dspark(d) => d.inject_target_feats_seq(seq, feats_ptr, pos),
        }
    }

    fn inject_batch_seq(&mut self, seq: usize, engine: &InklingEngine, start_tok: usize, pos0: i32, n_tok: usize) -> anyhow::Result<()> {
        match self {
            Drafter::DFlash2(d) => {
                let (fptr, mb) = engine.batch_dspark_feats_ptr();
                d.inject_target_feats_batch_seq(seq, fptr, mb, start_tok, pos0, n_tok)
            }
            Drafter::Dspark(d) => {
                for j in 0..n_tok {
                    let fptr = engine.gather_batch_dspark_feats(start_tok + j);
                    d.inject_target_feats_seq(seq, fptr, pos0 + j as i32)?;
                }
                Ok(())
            }
        }
    }

    fn inject_batch_seq_async(&mut self, seq: usize, engine: &InklingEngine, start_tok: usize, pos0: i32, n_tok: usize) -> anyhow::Result<()> {
        match self {
            Drafter::DFlash2(d) => {
                let (fptr, mb) = engine.batch_dspark_feats_ptr();
                d.inject_target_feats_batch_seq_async(seq, fptr, mb, start_tok, pos0, n_tok)
            }
            Drafter::Dspark(d) => {
                for j in 0..n_tok {
                    let fptr = engine.gather_batch_dspark_feats(start_tok + j);
                    d.inject_target_feats_seq(seq, fptr, pos0 + j as i32)?;
                }
                Ok(())
            }
        }
    }

    fn draft(&mut self, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> anyhow::Result<Vec<i32>> {
        match self {
            Drafter::DFlash2(d) => d.draft(anchor_tok, target, k_len),
            Drafter::Dspark(d) => d.draft(anchor_tok, target, k_len),
        }
    }

    fn draft_seq(&mut self, seq: usize, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> anyhow::Result<Vec<i32>> {
        match self {
            Drafter::DFlash2(d) => d.draft_seq(seq, anchor_tok, target, k_len),
            Drafter::Dspark(d) => Ok(d.draft_multi(&[seq as i32], &[anchor_tok], target, k_len)?.pop().unwrap()),
        }
    }

    fn draft_multi(&mut self, seq_ids: &[i32], anchor_toks: &[i32], target: &InklingEngine, k_len: usize) -> anyhow::Result<Vec<Vec<i32>>> {
        match self {
            Drafter::DFlash2(d) => d.draft_multi(seq_ids, anchor_toks, target, k_len),
            Drafter::Dspark(d) => d.draft_multi(seq_ids, anchor_toks, target, k_len),
        }
    }
}

struct ActiveSlot {
    seq: usize,
    next_tok: i32,
    max_tokens: usize,
    emitted: usize,
    #[allow(dead_code)]
    enable_thinking: bool,
    grammar: Option<grammar::JsonGrammarState>,
    sampling: grammar::SamplingParams,
    started_at: Instant,
    last_tok_at: Instant,
    tx: mpsc::UnboundedSender<GenEvent>,
}

struct PrefillSlot {
    seq: usize,
    job: GenJob,
    offset: usize,
    cached_n: usize,
    started_at: Instant,
}

struct InFlightGuard(Arc<AtomicU64>);
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

struct WorkerHandle {
    #[allow(dead_code)]
    dev_id: i32,
    job_tx: mpsc::UnboundedSender<GenJob>,
    in_flight: Arc<AtomicU64>,
    last_prefix_hash: Arc<AtomicU64>,
}

fn compute_prefix_hash(tokens: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in tokens.iter().take(256) {
        h ^= t as u32 as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct AppState {
    workers: Vec<WorkerHandle>,
    gpu_ids: Vec<i32>,
    parallel_mode: &'static str,
    p2p_links: usize,
    tokenizer: Arc<Tokenizer>,
    metrics: Arc<metrics::Metrics>,
    api_keys: Vec<String>,
    metrics_auth: bool,
    spec_enabled: bool,
    spec_mode: &'static str,
    vision_enabled: bool,
    mmproj_path: Option<String>,
    prefill_chunk_size: usize,
    kv_block_size: usize,
    max_ctx: usize,
    port: u16,
}

impl AppState {
    fn dispatch(&self, job: GenJob) -> Result<Arc<AtomicU64>, ()> {
        if self.workers.is_empty() {
            return Err(());
        }
        let ph = compute_prefix_hash(&job.prompt_ids);
        let mut best_idx = 0usize;
        if self.workers.len() > 1 {
            let mut min_load = u64::MAX;
            for (i, w) in self.workers.iter().enumerate() {
                let load = w.in_flight.load(Ordering::Relaxed);
                if load < min_load {
                    min_load = load;
                    best_idx = i;
                }
            }
            for (i, w) in self.workers.iter().enumerate() {
                if w.last_prefix_hash.load(Ordering::Relaxed) == ph {
                    let load = w.in_flight.load(Ordering::Relaxed);
                    if load <= min_load + 1 {
                        best_idx = i;
                        break;
                    }
                }
            }
        }
        let w = &self.workers[best_idx];
        w.last_prefix_hash.store(ph, Ordering::Relaxed);
        w.in_flight.fetch_add(1, Ordering::Relaxed);
        if w.job_tx.send(job).is_err() {
            w.in_flight.fetch_sub(1, Ordering::Relaxed);
            return Err(());
        }
        Ok(w.in_flight.clone())
    }
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    engine: &'static str,
    gpu_count: usize,
    gpu_ids: Vec<i32>,
    parallel_mode: &'static str,
    p2p_links: usize,
    speculative_decoding: bool,
    spec_mode: &'static str,
    vision_enabled: bool,
    mmproj_detected: bool,
    paged_kv: bool,
    kv_block_size: usize,
    kv_blocks_used: u64,
    kv_blocks_total: u64,
    chunked_prefill: bool,
    prefill_chunk_size: usize,
    grammar_constraints: bool,
    metrics_enabled: bool,
    max_ctx: usize,
    port: u16,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
    #[serde(default)]
    tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopInput {
    Single(String),
    Multi(Vec<String>),
}

impl StopInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopInput::Single(s) => if s.is_empty() { Vec::new() } else { vec![s] },
            StopInput::Multi(v) => v.into_iter().filter(|s| !s.is_empty()).collect(),
        }
    }
}

#[derive(Deserialize)]
struct TokenizeRequest {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct DetokenizeRequest {
    tokens: Vec<u32>,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PromptInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: PromptInput,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct ResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input: serde_json::Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    max_output_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
}

#[derive(Deserialize, Default)]
struct WsAuthQuery {
    #[serde(default)]
    api_key: Option<String>,
}

fn const_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn check_auth_keys(headers: &HeaderMap, keys: &[String]) -> bool {
    if keys.is_empty() {
        return true;
    }
    let provided = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
    match provided {
        Some(tok) => keys.iter().any(|k| const_eq(tok.as_bytes(), k.as_bytes())),
        None => false,
    }
}

fn check_key_str(provided: Option<&str>, keys: &[String]) -> bool {
    if keys.is_empty() {
        return true;
    }
    match provided {
        Some(tok) => keys.iter().any(|k| const_eq(tok.as_bytes(), k.as_bytes())),
        None => false,
    }
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn next_id(prefix: &str) -> String {
    let n = REQ_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-rad-{}-{n}", now_unix())
}

fn msg_text_and_images(val: &serde_json::Value, images: &mut Vec<String>) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            let mut out = String::new();
            for item in arr {
                let ty = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if ty == "image_url" || ty == "input_image" {
                    if let Some(url) = item.get("image_url").and_then(|u| u.get("url").or(Some(u))).and_then(|v| v.as_str()) {
                        images.push(url.to_string());
                        out.push_str("<|content_image|>");
                    }
                } else if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                    out.push_str(t);
                } else if let Some(s) = item.as_str() {
                    out.push_str(s);
                }
            }
            out
        }
        _ => String::new(),
    }
}

fn normalize_tool_specs(tools: &[serde_json::Value]) -> (Vec<serde_json::Value>, Vec<String>) {
    let mut specs = Vec::new();
    let mut names = Vec::new();
    for t in tools {
        let fn_obj = t.get("function").unwrap_or(t);
        if let Some(name) = fn_obj.get("name").and_then(|v| v.as_str()) {
            names.push(name.to_string());
            let desc = fn_obj.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let params = fn_obj.get("parameters").cloned().unwrap_or_else(|| json!({}));
            let ty = t.get("type").and_then(|v| v.as_str()).unwrap_or("function");
            specs.push(json!({
                "description": desc,
                "name": name,
                "parameters": params,
                "type": ty
            }));
        }
    }
    (specs, names)
}

fn detect_repeated_tool_loop(messages: &[ChatMessage]) -> bool {
    let mut recent_sigs: Vec<String> = Vec::new();
    for m in messages.iter().rev() {
        if m.role == "assistant" {
            if let Some(ref tcs) = m.tool_calls {
                if !tcs.is_empty() {
                    let mut parts = Vec::new();
                    for tc in tcs {
                        let fn_obj = tc.get("function").unwrap_or(tc);
                        let name = fn_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let args = fn_obj
                            .get("arguments")
                            .or_else(|| fn_obj.get("args"))
                            .map(|v| match v {
                                serde_json::Value::String(s) => s.trim().to_string(),
                                other => serde_json::to_string(other).unwrap_or_default(),
                            })
                            .unwrap_or_default();
                        parts.push(format!("{name}:{args}"));
                    }
                    recent_sigs.push(parts.join("|"));
                    if recent_sigs.len() >= 4 {
                        break;
                    }
                }
            }
        }
    }
    if recent_sigs.len() < 2 {
        return false;
    }
    let last = &recent_sigs[0];
    recent_sigs[1..].iter().any(|prev| prev == last)
}

fn format_prompt_with_tools(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    #[allow(dead_code)]
    enable_thinking: bool,
) -> (String, Vec<String>, Vec<String>) {
    let mut s = String::from("<|begin_of_text|>");
    let mut images = Vec::new();
    let mut allowed_tools = Vec::new();
    if let Some(ts) = tools {
        if !ts.is_empty() {
            let (specs, names) = normalize_tool_specs(ts);
            allowed_tools = names;
            if !specs.is_empty() {
                let spec_json = serde_json::to_string(&specs).unwrap_or_else(|_| "[]".into());
                s.push_str("<|message_system|>tool_declare<|content_xml|>");
                s.push_str(&spec_json);
                s.push_str("<|end_message|>");
            }
        }
    }
    let mut effort_emitted = false;
    let effort_str = if enable_thinking { "0.7" } else { "0" };
    for m in messages {
        if !effort_emitted && !matches!(m.role.as_str(), "system" | "developer") {
            s.push_str("<|message_system|><|content_text|>Thinking effort level: ");
            s.push_str(effort_str);
            s.push_str("<|end_message|>");
            effort_emitted = true;
        }
        let txt = msg_text_and_images(&m.content, &mut images);
        match m.role.as_str() {
            "system" | "developer" => {
                s.push_str("<|message_system|><|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            "user" => {
                s.push_str("<|message_user|><|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            "assistant" => {
                if let Some(ref rc) = m.reasoning_content {
                    if !rc.is_empty() {
                        s.push_str("<|message_model|><|content_thinking|>");
                        s.push_str(rc);
                        s.push_str("<|end_message|>");
                    }
                }
                let has_tcs = m.tool_calls.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
                if !txt.is_empty() || !has_tcs {
                    s.push_str("<|message_model|><|content_text|>");
                    s.push_str(&txt);
                    s.push_str("<|end_message|>");
                }
                if let Some(ref tcs) = m.tool_calls {
                    for tc in tcs {
                        let fn_obj = tc.get("function").unwrap_or(tc);
                        if let Some(name) = fn_obj.get("name").and_then(|v| v.as_str()) {
                            let args_val = match fn_obj.get("arguments").or_else(|| fn_obj.get("args")) {
                                Some(serde_json::Value::String(st)) => {
                                    serde_json::from_str::<serde_json::Value>(st).unwrap_or_else(|_| json!({}))
                                }
                                Some(v) if v.is_object() => v.clone(),
                                _ => json!({}),
                            };
                            let name_json = serde_json::to_string(name).unwrap_or_else(|_| format!("\"{name}\""));
                            let args_json = serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".into());
                            let call_json = format!("{{\"name\":{name_json},\"args\":{args_json}}}");
                            s.push_str("<|message_model|>");
                            s.push_str(name);
                            s.push_str("<|content_invoke_tool_json|>");
                            s.push_str(&call_json);
                            s.push_str("<|end_message|>");
                        }
                    }
                }
                s.push_str("<|content_model_end_sampling|>");
            }
            "tool" => {
                let mut tname = m.name.clone().unwrap_or_default();
                if tname.is_empty() {
                    if let Some(ref cid) = m.tool_call_id {
                        'outer: for prev in messages {
                            if prev.role == "assistant" {
                                if let Some(ref tcs) = prev.tool_calls {
                                    for tc in tcs {
                                        if tc.get("id").and_then(|v| v.as_str()) == Some(cid.as_str()) {
                                            if let Some(n) = tc.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()) {
                                                tname = n.to_string();
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                s.push_str("<|message_tool|>");
                s.push_str(&tname);
                s.push_str("<|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            _ => {}
        }
    }
    if !effort_emitted {
        s.push_str("<|message_system|><|content_text|>Thinking effort level: ");
        s.push_str(effort_str);
        s.push_str("<|end_message|>");
    }
    if enable_thinking {
        s.push_str("<|message_model|><|content_thinking|>");
    } else {
        s.push_str("<|message_model|><|content_text|>");
    }
    (s, images, allowed_tools)
}

fn format_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    format_prompt_with_tools(messages, None, enable_thinking).0
}

fn format_responses_input_with_tools(
    input: &serde_json::Value,
    instructions: Option<&str>,
    tools: Option<&[serde_json::Value]>,
    #[allow(dead_code)]
    enable_thinking: bool,
) -> (String, Vec<String>, Vec<String>) {
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(inst) = instructions {
        if !inst.is_empty() {
            msgs.push(ChatMessage {
                role: "system".into(),
                content: json!(inst),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
    }
    match input {
        serde_json::Value::String(s) => {
            msgs.push(ChatMessage {
                role: "user".into(),
                content: json!(s),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if item_type == "function_call" {
                    let call_id = item.get("call_id").or_else(|| item.get("id")).and_then(|v| v.as_str()).unwrap_or("call_0").to_string();
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let arguments = item.get("arguments").cloned().unwrap_or_else(|| json!("{}"));
                    let tc_json = json!({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments}
                    });
                    if let Some(last) = msgs.last_mut() {
                        if last.role == "assistant" {
                            last.tool_calls.get_or_insert_with(Vec::new).push(tc_json);
                            continue;
                        }
                    }
                    msgs.push(ChatMessage {
                        role: "assistant".into(),
                        content: json!(""),
                        tool_calls: Some(vec![tc_json]),
                        tool_call_id: None,
                        name: None,
                        reasoning_content: None,
                    });
                } else if item_type == "function_call_output" {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).map(|s| s.to_string());
                    let output = item.get("output").cloned().unwrap_or_else(|| json!(""));
                    msgs.push(ChatMessage {
                        role: "tool".into(),
                        content: output,
                        tool_calls: None,
                        tool_call_id: call_id,
                        name: None,
                        reasoning_content: None,
                    });
                } else {
                    let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
                    let content = item.get("content").cloned().unwrap_or_else(|| json!(""));
                    msgs.push(ChatMessage {
                        role,
                        content,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                        reasoning_content: None,
                    });
                }
            }
        }
        _ => {
            msgs.push(ChatMessage {
                role: "user".into(),
                content: json!(""),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
    }
    format_prompt_with_tools(&msgs, tools, enable_thinking)
}

#[allow(dead_code)]
fn split_thinking(raw: &str, enable_thinking: bool) -> (Option<String>, String) {
    let cleaned = raw
        .replace("<|content_model_end_sampling|>", "")
        .replace("<|end_message|>", "");
    if !enable_thinking {
        if let Some((_, rest)) = cleaned.rsplit_once("<|content_text|>") {
            return (None, rest.trim().to_string());
        }
        return (None, cleaned.trim().to_string());
    }
    if let Some((think, ans)) = cleaned.split_once("<|content_text|>") {
        let t = think.replace("<|message_model|>", "").replace("<|content_thinking|>", "").trim().to_string();
        let a = ans.trim().to_string();
        (if t.is_empty() { None } else { Some(t) }, a)
    } else {
        (None, cleaned.trim().to_string())
    }
}

fn pick_best_slot(eng: &InklingEngine, free: &[bool], prompt: &[i32]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (s, &is_free) in free.iter().enumerate() {
        if !is_free {
            continue;
        }
        let cached = &eng.seq_tokens[s];
        let score = if !cached.is_empty() && prompt.len() > cached.len() && prompt.starts_with(cached) {
            cached.len()
        } else {
            0
        };
        if best.map(|(_, b)| score > b).unwrap_or(true) {
            best = Some((s, score));
        }
    }
    best.map(|(s, _)| s)
}

fn prefill_with_drafter(
    engine: &mut InklingEngine,
    seq: usize,
    prompt_ids: &[i32],
    images: &[String],
    image_tok_id: Option<i32>,
    mut drafter: Option<&mut Drafter>,
    mut vision: Option<&mut VisionEncoder>,
) -> anyhow::Result<(i32, usize)> {
    if !images.is_empty() {
        if let (Some(vis), Some(img_id)) = (vision.as_deref_mut(), image_tok_id) {
            engine.reset_seq(seq)?;
            if let Some(ref mut d) = drafter {
                d.reset_seq(seq);
            }
            let mut img_idx = 0usize;
            let mut seg_start = 0usize;
            let mut last_tok = 0i32;
            for idx in 0..prompt_ids.len() {
                if prompt_ids[idx] == img_id && img_idx < images.len() {
                    let seg = &prompt_ids[seg_start..idx];
                    if !seg.is_empty() {
                        let pos0 = engine.seq_pos[seq];
                        let _ = engine.prefill_tokens_seq(seq, seg)?;
                        if let Some(ref mut d) = drafter {
                            if seq < d.max_seqs() {
                                d.inject_batch_seq(seq, engine, 0, pos0, seg.len())?;
                            }
                        }
                    }
                    let embeds = vis.encode_data_url(&images[img_idx])?;
                    let n_vis = vis.out_tokens;
                    let pos0 = engine.seq_pos[seq];
                    last_tok = engine.prefill_embeds_seq(seq, &embeds, n_vis)?;
                    if let Some(ref mut d) = drafter {
                        if seq < d.max_seqs() {
                            d.inject_batch_seq(seq, engine, 0, pos0, n_vis)?;
                        }
                    }
                    img_idx += 1;
                    seg_start = idx + 1;
                }
            }
            if seg_start < prompt_ids.len() {
                let seg = &prompt_ids[seg_start..];
                let pos0 = engine.seq_pos[seq];
                last_tok = engine.prefill_tokens_seq(seq, seg)?;
                if let Some(ref mut d) = drafter {
                    if seq < d.max_seqs() {
                        d.inject_batch_seq(seq, engine, 0, pos0, seg.len())?;
                    }
                }
            }
            return Ok((last_tok, 0));
        }
    }
    match drafter {
        Some(Drafter::Dspark(d)) => engine.prefill_seq_dspark(seq, prompt_ids, Some(d)),
        Some(Drafter::DFlash2(d)) => {
            if seq < d.max_seqs {
                let cached = &engine.seq_tokens[seq];
                let mut common = 0usize;
                let lim = cached.len().min(prompt_ids.len().saturating_sub(1));
                while common < lim && cached[common] == prompt_ids[common] {
                    common += 1;
                }
                if common == 0 || common != cached.len() || d.seq_ctx_len[seq] as usize != common {
                    engine.reset_seq(seq)?;
                    d.reset_seq(seq);
                    common = 0;
                }
                let rem = &prompt_ids[common..];
                let total_len = prompt_ids.len();
                let inj_start = total_len.saturating_sub(d.swa_window);
                let mut last = 0i32;
                let mut i = 0usize;
                while i < rem.len() {
                    let chunk = &rem[i..(i + 1024).min(rem.len())];
                    let pos0 = engine.seq_pos[seq];
                    last = engine.prefill_tokens_seq(seq, chunk)?;
                    let chunk_start = common + i;
                    let chunk_end = chunk_start + chunk.len();
                    if chunk_end > inj_start {
                        let s0 = inj_start.saturating_sub(chunk_start);
                        let cnt = chunk.len() - s0;
                        let (fptr, mb) = engine.batch_dspark_feats_ptr();
                        d.inject_target_feats_batch_seq(seq, fptr, mb, s0, pos0 + s0 as i32, cnt)?;
                    } else {
                        d.advance_ctx_len_seq(seq, pos0 + chunk.len() as i32);
                    }
                    i += chunk.len();
                }
                Ok((last, common))
            } else {
                engine.prefill_seq_dspark(seq, prompt_ids, None)
            }
        }
        None => engine.prefill_seq_dspark(seq, prompt_ids, None),
    }
}

fn ngram_bucket_len(avail: usize) -> usize {
    if avail >= 128 { 128 }
    else if avail >= 96 { 96 }
    else if avail >= 64 { 64 }
    else if avail >= 48 { 48 }
    else if avail >= 32 { 32 }
    else if avail >= 24 { 24 }
    else if avail >= 16 { 16 }
    else if avail >= 12 { 12 }
    else if avail >= 8 { 8 }
    else { 0 }
}

fn ngram_lookup_draft(history: &[i32], anchor: i32, max_k: usize) -> Option<Vec<i32>> {
    let n = history.len();
    if n < 16 || max_k < 8 {
        return None;
    }
    let min_follow = 8usize;
    let c = history[n - 1];
    let b = history[n - 2];
    let mut best_pos: Option<usize> = None;
    let mut best_len: usize = 0;
    let mut i = n.saturating_sub(min_follow + 1);
    while i >= 2 {
        if history[i] == anchor && history[i - 1] == c && history[i - 2] == b {
            let mut mlen = 3usize;
            let max_m = 16usize.min(i).min(n);
            while mlen < max_m && history[i - mlen] == history[n - mlen] {
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
    if let Some(pos) = best_pos {
        let cap_k = if best_len >= 6 {
            max_k
        } else if best_len >= 4 {
            max_k.min(32)
        } else {
            max_k.min(16)
        };
        let take = ngram_bucket_len((n - pos).min(cap_k));
        if take >= min_follow {
            let mut out = Vec::with_capacity(take);
            out.push(anchor);
            out.extend_from_slice(&history[(pos + 1)..(pos + take)]);
            return Some(out);
        }
    }
    None
}

fn sync_kv_metrics(engine: &InklingEngine, metrics: &metrics::Metrics) {
    let st = engine.paged_kv.stats();
    metrics.kv_blocks_total.store(st.total_blocks as u64, Ordering::Relaxed);
    metrics.kv_blocks_used.store(st.used_blocks as u64, Ordering::Relaxed);
    metrics.kv_blocks_shared.store(st.shared_blocks as u64, Ordering::Relaxed);
}

fn run_scheduler(
    mut engine: InklingEngine,
    mut drafter: Option<Drafter>,
    mut vision: Option<VisionEncoder>,
    image_tok_id: Option<i32>,
    vocab_grammar: Arc<grammar::VocabGrammarTable>,
    metrics: Arc<metrics::Metrics>,
    prefill_chunk_size: usize,
    mut rx: mpsc::UnboundedReceiver<GenJob>,
) {
    let max_seqs = engine.max_seqs;
    let mut free = vec![true; max_seqs];
    let mut active: Vec<ActiveSlot> = Vec::with_capacity(max_seqs);
    let mut prefilling: VecDeque<PrefillSlot> = VecDeque::new();
    let mut pending: VecDeque<GenJob> = VecDeque::new();
    let spec_k = drafter.as_ref().map(|d| d.spec_k()).unwrap_or(4);
    sync_kv_metrics(&engine, &metrics);

    loop {
        if active.is_empty() && prefilling.is_empty() && pending.is_empty() {
            metrics.requests_active.store(0, Ordering::Relaxed);
            metrics.requests_queued.store(0, Ordering::Relaxed);
            metrics.requests_prefilling.store(0, Ordering::Relaxed);
            match rx.blocking_recv() {
                Some(job) => pending.push_back(job),
                None => break,
            }
        }
        while let Ok(job) = rx.try_recv() {
            pending.push_back(job);
        }

        while !pending.is_empty() {
            let slot_opt = {
                let front = pending.front().unwrap();
                pick_best_slot(&engine, &free, &front.prompt_ids)
            };
            let Some(seq) = slot_opt else { break; };
            let job = pending.pop_front().unwrap();
            free[seq] = false;
            if !job.images.is_empty() && vision.is_some() && image_tok_id.is_some() {
                let tp = Instant::now();
                match prefill_with_drafter(&mut engine, seq, &job.prompt_ids, &job.images, image_tok_id, drafter.as_mut(), vision.as_mut()) {
                    Ok((mut first_tok, cached_n)) => {
                        let pp_s = tp.elapsed().as_secs_f64();
                        let pp_ms = pp_s * 1000.0;
                        metrics.prompt_tokens_total.fetch_add(job.prompt_ids.len() as u64, Ordering::Relaxed);
                        metrics.prefill_us_total.fetch_add((pp_s * 1e6) as u64, Ordering::Relaxed);
                        metrics.observe_ttft(pp_s);
                        let _ = engine.paged_kv.ensure_tokens(seq, engine.seq_pos[seq] as usize, &engine.seq_tokens[seq]);
                        sync_kv_metrics(&engine, &metrics);
                        let mut gram = if job.json_grammar { Some(grammar::JsonGrammarState::new_json()) } else { None };
                        let mut sampling = job.sampling;
                        if let Some(ref mut g) = gram {
                            if let Ok(logits) = engine.read_batch_logits(0) {
                                first_tok = vocab_grammar.select_best_token(&logits, g);
                            }
                        } else if !sampling.is_greedy() {
                            if let Ok(logits) = engine.read_batch_logits(0) {
                                first_tok = grammar::sample_from_logits(&logits, &engine.seq_tokens[seq], &mut sampling);
                            }
                        }
                        let _ = job.tx.send(GenEvent::Prefill { prompt_n: job.prompt_ids.len(), cached_n, pp_ms });
                        if job.max_tokens == 0 || first_tok == 200006 || first_tok == 199999 {
                            let _ = job.tx.send(GenEvent::Done { finish_reason: if job.max_tokens == 0 { "length" } else { "stop" }, pred_n: 0, dec_ms: 0.0 });
                            free[seq] = true;
                        } else {
                            let _ = job.tx.send(GenEvent::Token(first_tok as u32));
                            metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                            if job.max_tokens == 1 {
                                let _ = job.tx.send(GenEvent::Done { finish_reason: "length", pred_n: 1, dec_ms: 0.0 });
                                free[seq] = true;
                            } else {
                                let now = Instant::now();
                                active.push(ActiveSlot {
                                    seq, next_tok: first_tok, max_tokens: job.max_tokens, emitted: 1,
                                    enable_thinking: job.enable_thinking, grammar: gram, sampling,
                                    started_at: now, last_tok_at: now, tx: job.tx,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        let _ = job.tx.send(GenEvent::Error(format!("{e}")));
                        free[seq] = true;
                    }
                }
            } else {
                let mut common = 0usize;
                {
                    let cached = &engine.seq_tokens[seq];
                    let lim = cached.len().min(job.prompt_ids.len().saturating_sub(1));
                    while common < lim && cached[common] == job.prompt_ids[common] {
                        common += 1;
                    }
                }
                let drafter_matches = match drafter.as_ref() {
                    Some(d) if seq < d.max_seqs() => d.seq_ctx_len(seq) as usize == common,
                    _ => true,
                };
                if common == 0 || common != engine.seq_tokens[seq].len() || !drafter_matches {
                    let _ = engine.reset_seq(seq);
                    if let Some(ref mut d) = drafter {
                        if seq < d.max_seqs() {
                            d.reset_seq(seq);
                        }
                    }
                    common = 0;
                } else if common > 0 {
                    metrics.prefix_cache_hits_total.fetch_add(1, Ordering::Relaxed);
                    metrics.prefix_cache_tokens_saved_total.fetch_add(common as u64, Ordering::Relaxed);
                }
                prefilling.push_back(PrefillSlot {
                    seq,
                    job,
                    offset: common,
                    cached_n: common,
                    started_at: Instant::now(),
                });
            }
        }

        let mut pf_budget = if active.is_empty() { 1024 } else { prefill_chunk_size.max(64).min(1024) };
        while pf_budget > 0 {
            let Some(mut pf) = prefilling.pop_front() else { break; };
            let end = (pf.offset + pf_budget).min(pf.job.prompt_ids.len());
            let chunk = &pf.job.prompt_ids[pf.offset..end];
            pf_budget = pf_budget.saturating_sub(chunk.len());
            let pos0 = engine.seq_pos[pf.seq];
            match engine.prefill_tokens_seq(pf.seq, chunk) {
                Ok(mut last_tok) => {
                    if let Some(ref mut d) = drafter {
                        if pf.seq < d.max_seqs() {
                            match d {
                                Drafter::DFlash2(df) => {
                                    let inj_start = pf.job.prompt_ids.len().saturating_sub(df.swa_window);
                                    if end > inj_start {
                                        let s0 = inj_start.saturating_sub(pf.offset);
                                        let cnt = chunk.len() - s0;
                                        let (fptr, mb) = engine.batch_dspark_feats_ptr();
                                        let _ = df.inject_target_feats_batch_seq(pf.seq, fptr, mb, s0, pos0 + s0 as i32, cnt);
                                    } else {
                                        df.advance_ctx_len_seq(pf.seq, pos0 + chunk.len() as i32);
                                    }
                                }
                                Drafter::Dspark(_) => {
                                    let _ = d.inject_batch_seq(pf.seq, &engine, 0, pos0, chunk.len());
                                }
                            }
                        }
                    }
                    let _ = engine.paged_kv.ensure_tokens(pf.seq, engine.seq_pos[pf.seq] as usize, &engine.seq_tokens[pf.seq]);
                    sync_kv_metrics(&engine, &metrics);
                    pf.offset = end;
                    if pf.offset < pf.job.prompt_ids.len() {
                        prefilling.push_back(pf);
                        break;
                    } else {
                        let pp_s = pf.started_at.elapsed().as_secs_f64();
                        let pp_ms = pp_s * 1000.0;
                        metrics.prompt_tokens_total.fetch_add(pf.job.prompt_ids.len() as u64, Ordering::Relaxed);
                        metrics.prefill_us_total.fetch_add((pp_s * 1e6) as u64, Ordering::Relaxed);
                        metrics.observe_ttft(pp_s);
                        let mut gram = if pf.job.json_grammar { Some(grammar::JsonGrammarState::new_json()) } else { None };
                        let mut sampling = pf.job.sampling;
                        if let Some(ref mut g) = gram {
                            if let Ok(logits) = engine.read_batch_logits(0) {
                                last_tok = vocab_grammar.select_best_token(&logits, g);
                            }
                        } else if !sampling.is_greedy() {
                            if let Ok(logits) = engine.read_batch_logits(0) {
                                last_tok = grammar::sample_from_logits(&logits, &engine.seq_tokens[pf.seq], &mut sampling);
                            }
                        }
                        let _ = pf.job.tx.send(GenEvent::Prefill {
                            prompt_n: pf.job.prompt_ids.len(),
                            cached_n: pf.cached_n,
                            pp_ms,
                        });
                        if pf.job.max_tokens == 0 || last_tok == 200006 || last_tok == 199999 {
                            let _ = pf.job.tx.send(GenEvent::Done {
                                finish_reason: if pf.job.max_tokens == 0 { "length" } else { "stop" },
                                pred_n: 0,
                                dec_ms: 0.0,
                            });
                            free[pf.seq] = true;
                        } else {
                            let _ = pf.job.tx.send(GenEvent::Token(last_tok as u32));
                            metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                            if pf.job.max_tokens == 1 {
                                let _ = pf.job.tx.send(GenEvent::Done { finish_reason: "length", pred_n: 1, dec_ms: 0.0 });
                                free[pf.seq] = true;
                            } else {
                                let now = Instant::now();
                                active.push(ActiveSlot {
                                    seq: pf.seq,
                                    next_tok: last_tok,
                                    max_tokens: pf.job.max_tokens,
                                    emitted: 1,
                                    enable_thinking: pf.job.enable_thinking,
                                    grammar: gram,
                                    sampling,
                                    started_at: now,
                                    last_tok_at: now,
                                    tx: pf.job.tx,
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = pf.job.tx.send(GenEvent::Error(format!("{e}")));
                    free[pf.seq] = true;
                }
            }
        }

        metrics.requests_active.store(active.len() as i64, Ordering::Relaxed);
        metrics.requests_queued.store(pending.len() as i64, Ordering::Relaxed);
        metrics.requests_prefilling.store(prefilling.len() as i64, Ordering::Relaxed);

        if active.is_empty() {
            continue;
        }

        let any_grammar = active.iter().any(|a| a.grammar.is_some() || !a.sampling.is_greedy());

        if active.len() == 1 {
            if let Some(ref mut d) = drafter {
                let seq = active[0].seq;
                if seq < d.max_seqs() && d.seq_ctx_len(seq) == engine.seq_pos[seq] {
                    let mut slot = active.pop().unwrap();
                    let pos0 = engine.seq_pos[seq];
                    let t_step = Instant::now();
                    let needs_logits = slot.grammar.is_some() || !slot.sampling.is_greedy();
                    let res = (|| -> anyhow::Result<Vec<i32>> {
                        let rem_kv = engine.max_ctx.saturating_sub(pos0 as usize);
                        let max_ng_k = 128usize.min(rem_kv);
                        let draft = if let Some(ng) = ngram_lookup_draft(&engine.seq_tokens[seq], slot.next_tok, max_ng_k) {
                            ng
                        } else if seq == 0 {
                            d.draft(slot.next_tok, &engine, spec_k)?
                        } else {
                            d.draft_seq(seq, slot.next_tok, &engine, spec_k)?
                        };
                        let dk = draft.len();
                        let verified = engine.verify_batch_seq(seq, &draft)?;
                        let mut n_acc = 1usize;
                        let mut last_pred = verified[0];
                        if needs_logits {
                            if let Some(ref mut g) = slot.grammar {
                                for j in 0..dk {
                                    let logits = engine.read_batch_logits(j)?;
                                    let pred = vocab_grammar.select_best_token(&logits, g);
                                    last_pred = pred;
                                    if j + 1 < dk && pred == draft[j + 1] {
                                        n_acc += 1;
                                    } else {
                                        break;
                                    }
                                }
                            } else {
                                let (tk_vals, tk_ids) = engine.read_batch_topk64(dk)?;
                                let seq_len = engine.seq_tokens[seq].len();
                                let mut recent: Vec<i32> = engine.seq_tokens[seq][seq_len.saturating_sub(128)..].to_vec();
                                for j in 0..dk {
                                    recent.push(draft[j]);
                                    let off = j * 64;
                                    let pred = grammar::sample_from_topk(
                                        &tk_vals[off..off + 64],
                                        &tk_ids[off..off + 64],
                                        &recent,
                                        &mut slot.sampling,
                                    );
                                    last_pred = pred;
                                    if j + 1 < dk && pred == draft[j + 1] {
                                        n_acc += 1;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        } else {
                            for j in 0..(dk - 1) {
                                if verified[j] == draft[j + 1] {
                                    n_acc += 1;
                                } else {
                                    break;
                                }
                            }
                            last_pred = verified[n_acc - 1];
                        }
                        engine.rewind_verify_seq_to(seq, n_acc, dk)?;
                        d.inject_batch_seq_async(seq, &engine, 0, pos0, n_acc)?;
                        for j in 0..n_acc {
                            engine.seq_tokens[seq].push(draft[j]);
                        }
                        let _ = engine.paged_kv.ensure_tokens(seq, engine.seq_pos[seq] as usize, &engine.seq_tokens[seq]);
                        let mut emitted_toks = Vec::with_capacity(n_acc);
                        for j in 1..n_acc {
                            emitted_toks.push(draft[j]);
                        }
                        emitted_toks.push(last_pred);
                        Ok(emitted_toks)
                    })();
                    let step_s = t_step.elapsed().as_secs_f64();
                    sync_kv_metrics(&engine, &metrics);
                    match res {
                        Ok(toks_out) => {
                            let n_out = toks_out.len();
                            metrics.spec_rounds_total.fetch_add(1, Ordering::Relaxed);
                            metrics.spec_accepted_tokens_total.fetch_add(n_out as u64, Ordering::Relaxed);
                            metrics.decode_us_total.fetch_add((step_s * 1e6) as u64, Ordering::Relaxed);
                            if n_out > 0 {
                                metrics.observe_itl(step_s / n_out as f64);
                            }
                            let mut finished = false;
                            let mut last_tok = slot.next_tok;
                            for nxt in toks_out {
                                let is_stop = nxt == 200006 || nxt == 199999;
                                if is_stop || slot.tx.is_closed() {
                                    let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                    let _ = slot.tx.send(GenEvent::Done {
                                        finish_reason: "stop",
                                        pred_n: slot.emitted,
                                        dec_ms,
                                    });
                                    free[slot.seq] = true;
                                    finished = true;
                                    break;
                                }
                                slot.emitted += 1;
                                metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                                let _ = slot.tx.send(GenEvent::Token(nxt as u32));
                                last_tok = nxt;
                                if slot.emitted >= slot.max_tokens {
                                    let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                    let _ = slot.tx.send(GenEvent::Done {
                                        finish_reason: "length",
                                        pred_n: slot.emitted,
                                        dec_ms,
                                    });
                                    free[slot.seq] = true;
                                    finished = true;
                                    break;
                                }
                            }
                            if !finished {
                                slot.next_tok = last_tok;
                                slot.last_tok_at = Instant::now();
                                active.push(slot);
                            }
                        }
                        Err(e) => {
                            let _ = slot.tx.send(GenEvent::Error(format!("{e}")));
                            free[slot.seq] = true;
                        }
                    }
                    continue;
                }
            }
        }

        if active.len() >= 1 && active.len() <= 16 && !any_grammar {
            if let Some(ref mut d) = drafter {
                let all_ready = active.iter().all(|a| a.seq < d.max_seqs() && d.seq_ctx_len(a.seq) == engine.seq_pos[a.seq]);
                if all_ready {
                    let multi_k = if active.len() == 1 { spec_k } else { 4usize };
                    let seq_ids: Vec<i32> = active.iter().map(|a| a.seq as i32).collect();
                    let anchors: Vec<i32> = active.iter().map(|a| a.next_tok).collect();
                    let pos_before: Vec<i32> = active.iter().map(|a| engine.seq_pos[a.seq]).collect();
                    let t_step = Instant::now();
                    let res = (|| -> anyhow::Result<Vec<Vec<i32>>> {
                        let drafts = d.draft_multi(&seq_ids, &anchors, &engine, multi_k)?;
                        let preds = engine.verify_multi(&seq_ids, &drafts)?;
                        let mut emits = vec![0usize; active.len()];
                        let mut outs = Vec::with_capacity(active.len());
                        for u in 0..active.len() {
                            let mut n_acc = 1usize;
                            for j in 0..(multi_k - 1) {
                                if preds[u][j] == drafts[u][j + 1] {
                                    n_acc += 1;
                                } else {
                                    break;
                                }
                            }
                            emits[u] = n_acc;
                            let mut tok_out = Vec::with_capacity(n_acc);
                            for j in 1..n_acc {
                                tok_out.push(drafts[u][j]);
                            }
                            tok_out.push(preds[u][n_acc - 1]);
                            outs.push(tok_out);
                        }
                        engine.rewind_verify_multi(&seq_ids, &emits, multi_k)?;
                        for u in 0..active.len() {
                            let su = seq_ids[u] as usize;
                            d.inject_batch_seq_async(su, &engine, u * multi_k, pos_before[u], emits[u])?;
                            for j in 0..emits[u] {
                                engine.seq_tokens[su].push(drafts[u][j]);
                            }
                            let _ = engine.paged_kv.ensure_tokens(su, engine.seq_pos[su] as usize, &engine.seq_tokens[su]);
                        }
                        d.sync();
                        Ok(outs)
                    })();
                    let step_s = t_step.elapsed().as_secs_f64();
                    sync_kv_metrics(&engine, &metrics);
                    match res {
                        Ok(all_outs) => {
                            let tot_out: usize = all_outs.iter().map(|v| v.len()).sum();
                            metrics.spec_rounds_total.fetch_add(1, Ordering::Relaxed);
                            metrics.spec_accepted_tokens_total.fetch_add(tot_out as u64, Ordering::Relaxed);
                            metrics.decode_us_total.fetch_add((step_s * 1e6) as u64, Ordering::Relaxed);
                            if tot_out > 0 {
                                metrics.observe_itl(step_s / (tot_out as f64 / active.len().max(1) as f64));
                            }
                            let mut keep = Vec::with_capacity(active.len());
                            for (mut slot, toks_out) in active.into_iter().zip(all_outs.into_iter()) {
                                let mut finished = false;
                                let mut last_tok = slot.next_tok;
                                for nxt in toks_out {
                                    let is_stop = nxt == 200006 || nxt == 199999;
                                    if is_stop || slot.tx.is_closed() {
                                        let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                        let _ = slot.tx.send(GenEvent::Done {
                                            finish_reason: "stop",
                                            pred_n: slot.emitted,
                                            dec_ms,
                                        });
                                        free[slot.seq] = true;
                                        finished = true;
                                        break;
                                    }
                                    slot.emitted += 1;
                                    metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                                    let _ = slot.tx.send(GenEvent::Token(nxt as u32));
                                    last_tok = nxt;
                                    if slot.emitted >= slot.max_tokens {
                                        let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                        let _ = slot.tx.send(GenEvent::Done {
                                            finish_reason: "length",
                                            pred_n: slot.emitted,
                                            dec_ms,
                                        });
                                        free[slot.seq] = true;
                                        finished = true;
                                        break;
                                    }
                                }
                                if !finished {
                                    slot.next_tok = last_tok;
                                    keep.push(slot);
                                }
                            }
                            active = keep;
                        }
                        Err(e) => {
                            let msg = format!("{e}");
                            for slot in active.drain(..) {
                                let _ = slot.tx.send(GenEvent::Error(msg.clone()));
                                free[slot.seq] = true;
                            }
                        }
                    }
                    continue;
                }
            }
        }

        let seq_ids: Vec<i32> = active.iter().map(|a| a.seq as i32).collect();
        let toks: Vec<i32> = active.iter().map(|a| a.next_tok).collect();
        let t_step = Instant::now();
        match engine.step_multi(&seq_ids, &toks) {
            Ok(nexts) => {
                let step_s = t_step.elapsed().as_secs_f64();
                metrics.decode_us_total.fetch_add((step_s * 1e6) as u64, Ordering::Relaxed);
                metrics.observe_itl(step_s);
                let mut keep = Vec::with_capacity(active.len());
                for (u, (mut slot, mut nxt)) in active.into_iter().zip(nexts.into_iter()).enumerate() {
                    if let Some(ref mut g) = slot.grammar {
                        if let Ok(logits) = engine.read_batch_logits(u) {
                            nxt = vocab_grammar.select_best_token(&logits, g);
                        }
                    } else if !slot.sampling.is_greedy() {
                        if let Ok(logits) = engine.read_batch_logits(u) {
                            nxt = grammar::sample_from_logits(&logits, &engine.seq_tokens[slot.seq], &mut slot.sampling);
                        }
                    }
                    let _ = engine.paged_kv.ensure_tokens(slot.seq, engine.seq_pos[slot.seq] as usize, &engine.seq_tokens[slot.seq]);
                    let is_stop = nxt == 200006 || nxt == 199999;
                    if is_stop || slot.tx.is_closed() {
                        let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                        let _ = slot.tx.send(GenEvent::Done {
                            finish_reason: "stop",
                            pred_n: slot.emitted,
                            dec_ms,
                        });
                        free[slot.seq] = true;
                    } else {
                        slot.emitted += 1;
                        metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                        let _ = slot.tx.send(GenEvent::Token(nxt as u32));
                        if slot.emitted >= slot.max_tokens {
                            let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                            let _ = slot.tx.send(GenEvent::Done {
                                finish_reason: "length",
                                pred_n: slot.emitted,
                                dec_ms,
                            });
                            free[slot.seq] = true;
                        } else {
                            slot.next_tok = nxt;
                            slot.last_tok_at = Instant::now();
                            keep.push(slot);
                        }
                    }
                }
                sync_kv_metrics(&engine, &metrics);
                active = keep;
            }
            Err(e) => {
                let msg = format!("{e}");
                for slot in active.drain(..) {
                    let _ = slot.tx.send(GenEvent::Error(msg.clone()));
                    free[slot.seq] = true;
                }
            }
        }
    }
}

async fn collect_job(
    st: &AppState,
    prompt_ids: Vec<i32>,
    images: Vec<String>,
    max_tokens: usize,
    #[allow(dead_code)]
    enable_thinking: bool,
    json_grammar: bool,
    sampling: grammar::SamplingParams,
) -> Result<(Vec<u32>, usize, usize, f64, f64, &'static str), String> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let in_flight = st
        .dispatch(GenJob { prompt_ids, images, max_tokens, enable_thinking, json_grammar, sampling, tx })
        .map_err(|_| "scheduler closed".to_string())?;
    let mut out_ids = Vec::with_capacity(max_tokens.min(4096));
    let mut prompt_n = 0usize;
    let mut cached_n = 0usize;
    let mut pp_ms = 0.0f64;
    let mut dec_ms = 0.0f64;
    let mut finish_reason = "stop";
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Prefill { prompt_n: pn, cached_n: cn, pp_ms: pm } => {
                prompt_n = pn;
                cached_n = cn;
                pp_ms = pm;
            }
            GenEvent::Token(t) => out_ids.push(t),
            GenEvent::Done { finish_reason: fr, dec_ms: dm, .. } => {
                finish_reason = fr;
                dec_ms = dm;
                break;
            }
            GenEvent::Error(e) => {
                in_flight.fetch_sub(1, Ordering::Relaxed);
                return Err(e);
            }
        }
    }
    in_flight.fetch_sub(1, Ordering::Relaxed);
    Ok((out_ids, prompt_n, cached_n, pp_ms, dec_ms, finish_reason))
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Health> {
    Json(Health {
        status: "ok",
        engine: "rad",
        gpu_count: st.gpu_ids.len(),
        gpu_ids: st.gpu_ids.clone(),
        parallel_mode: st.parallel_mode,
        p2p_links: st.p2p_links,
        speculative_decoding: st.spec_enabled,
        spec_mode: st.spec_mode,
        vision_enabled: st.vision_enabled,
        mmproj_detected: st.mmproj_path.is_some(),
        paged_kv: true,
        kv_block_size: st.kv_block_size,
        kv_blocks_used: st.metrics.kv_blocks_used.load(Ordering::Relaxed),
        kv_blocks_total: st.metrics.kv_blocks_total.load(Ordering::Relaxed),
        chunked_prefill: true,
        prefill_chunk_size: st.prefill_chunk_size,
        grammar_constraints: true,
        metrics_enabled: true,
        max_ctx: st.max_ctx,
        port: st.port,
    })
}

async fn tokenize_endpoint(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<TokenizeRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let text = req.prompt.or(req.text).unwrap_or_default();
    match st.tokenizer.encode(text, false) {
        Ok(enc) => {
            let ids = enc.get_ids();
            Json(json!({"tokens": ids, "count": ids.len(), "max_model_len": st.max_ctx})).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    }
}

async fn detokenize_endpoint(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<DetokenizeRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    match st.tokenizer.decode(&req.tokens, false) {
        Ok(text) => Json(json!({"text": text})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    }
}

async fn metrics_endpoint(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if st.metrics_auth && !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let body = st.metrics.render();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"))],
        body,
    )
        .into_response()
}

async fn cors_and_req_id(req: Request, next: Next) -> Response {
    let rid = next_id("req");
    if req.method() == Method::OPTIONS {
        let mut resp = StatusCode::NO_CONTENT.into_response();
        let h = resp.headers_mut();
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
        h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("Authorization, Content-Type, X-API-Key"));
        if let Ok(v) = HeaderValue::from_str(&rid) {
            h.insert("x-request-id", v);
        }
        return resp;
    }
    let mut resp = next.run(req).await;
    let h = resp.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    if let Ok(v) = HeaderValue::from_str(&rid) {
        h.insert("x-request-id", v);
    }
    resp
}

async fn model_by_id(State(st): State<Arc<AppState>>, headers: HeaderMap, AxumPath(id): AxumPath<String>) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let now = now_unix();
    Json(json!({
        "id": id,
        "object": "model",
        "created": now,
        "owned_by": "rad",
        "capabilities": {
            "speculative_decoding": st.spec_enabled,
            "spec_mode": st.spec_mode,
            "vision": st.vision_enabled,
            "mmproj_detected": st.mmproj_path.is_some()
        }
    }))
    .into_response()
}

async fn models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let now = now_unix();
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": "inkling-2-mini",
                "object": "model",
                "created": now,
                "owned_by": "rad",
                "capabilities": {
                    "speculative_decoding": st.spec_enabled,
                    "spec_mode": st.spec_mode,
                    "vision": st.vision_enabled,
                    "mmproj_detected": st.mmproj_path.is_some()
                }
            }
        ]
    }))
    .into_response()
}

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let model_id = req.model.unwrap_or_else(|| "inkling-2-mini".into());
    let max_tokens = req.max_tokens.or(req.max_completion_tokens).unwrap_or(4096).min(16384);
    let enable_thinking = req.chat_template_kwargs.and_then(|k| k.enable_thinking).unwrap_or(false);
    let stream = req.stream.unwrap_or(false);
    let mut force_tool_json = false;
    let mut forced_tool_name: Option<String> = None;
    if let Some(ref tc) = req.tool_choice {
        if tc.as_str() == Some("required") {
            force_tool_json = true;
        } else if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) {
            force_tool_json = true;
            forced_tool_name = Some(name.to_string());
        }
    }
    let response_json = req.response_format.as_ref().and_then(|rf| rf.get("type")).and_then(|t| t.as_str()).map(|t| t == "json_object" || t == "json_schema").unwrap_or(false);
    let json_grammar = force_tool_json || response_json;
    let initial_thinking = enable_thinking && !json_grammar;
    let include_usage = req.stream_options.as_ref().and_then(|o| o.include_usage).unwrap_or(false);
    let stop_seqs: Vec<String> = req.stop.map(|s| s.into_vec()).unwrap_or_default();
    let base_seed = req.seed.unwrap_or_else(|| now_unix() ^ REQ_SEQ.load(Ordering::Relaxed));
    let extra_pen = req.presence_penalty.unwrap_or(0.0).max(req.frequency_penalty.unwrap_or(0.0)).clamp(0.0, 2.0);
    let eff_rep = req.repetition_penalty.unwrap_or_else(|| if extra_pen > 1e-4 { 1.0 + 0.15 * extra_pen } else { 1.0 });
    let loop_detected = detect_repeated_tool_loop(&req.messages);
    let mut sampling = grammar::SamplingParams::greedy();
    if loop_detected {
        sampling = grammar::SamplingParams {
            temperature: req.temperature.unwrap_or(0.7).max(0.5),
            top_p: req.top_p.unwrap_or(0.95),
            repetition_penalty: eff_rep.max(1.15),
            seed: base_seed,
        };
    } else if (eff_rep - 1.0).abs() > 1e-4 {
        sampling = grammar::SamplingParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(0.95),
            repetition_penalty: eff_rep,
            seed: base_seed,
        };
    }

    let (mut prompt, images, allowed_tools) = format_prompt_with_tools(&req.messages, req.tools.as_deref(), initial_thinking);
    if loop_detected {
        let nudge = "<|message_system|><|content_text|>You already ran that exact tool call and received its output above. Do NOT repeat the same command again — inspect something new or summarize your findings now.<|end_message|>";
        if let Some(pos) = prompt.rfind("<|message_model|>") {
            prompt.insert_str(pos, nudge);
        }
    }
    if force_tool_json {
        let extra = if let Some(ref fn_name) = forced_tool_name {
            format!("<|message_system|><|content_text|>You MUST call tool `{fn_name}` by responding ONLY with valid JSON: {{\"name\": \"{fn_name}\", \"arguments\": {{...}}}}.<|end_message|>")
        } else {
            "<|message_system|><|content_text|>You MUST respond ONLY with a valid JSON tool call: {\"name\": \"<tool_name>\", \"arguments\": {...}}.<|end_message|>".to_string()
        };
        if let Some(rest) = prompt.strip_prefix("<|begin_of_text|>") {
            prompt = format!("<|begin_of_text|>{extra}{rest}");
        }
    } else if response_json {
        let json_sys = "<|message_system|><|content_text|>Respond ONLY with valid JSON.<|end_message|>";
        if let Some(rest) = prompt.strip_prefix("<|begin_of_text|>") {
            prompt = format!("<|begin_of_text|>{json_sys}{rest}");
        }
    }

    let enc = match st.tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => {
            st.metrics.requests_err.fetch_add(1, Ordering::Relaxed);
            return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response();
        }
    };
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if prompt_ids.len() >= st.max_ctx {
        st.metrics.requests_err.fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("prompt has {} tokens which exceeds max_ctx {}", prompt_ids.len(), st.max_ctx),
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded"
                }
            })),
        )
            .into_response();
    }
    let max_tokens = max_tokens.min(st.max_ctx.saturating_sub(prompt_ids.len() + 1).max(1));
    let created = now_unix();
    let req_id = next_id("chatcmpl");
    let call_prefix = next_id("call");

    if stream {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let in_flight = match st.dispatch(GenJob { prompt_ids, images, max_tokens, enable_thinking: initial_thinking, json_grammar, sampling, tx }) {
            Ok(g) => InFlightGuard(g),
            Err(_) => {
                st.metrics.requests_err.fetch_add(1, Ordering::Relaxed);
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":"scheduler closed"}))).into_response();
            }
        };
        st.metrics.requests_ok.fetch_add(1, Ordering::Relaxed);
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let tok = st.tokenizer.clone();
        tokio::spawn(async move {
            let _guard = in_flight;
            let mut parser = grammar::InklingStreamParser::new(initial_thinking, call_prefix, allowed_tools);
            let mut sent_role = false;
            let mut tc_index = 0usize;
            let mut prompt_n_sse = 0usize;
            let mut cached_n_sse = 0usize;
            while let Some(ev) = rx.recv().await {
                match ev {
                    GenEvent::Prefill { prompt_n, cached_n, .. } => {
                        prompt_n_sse = prompt_n;
                        cached_n_sse = cached_n;
                    }
                    GenEvent::Token(t) => {
                        let deltas = parser.push_token(t, &tok);
                        for d in deltas {
                            let d_obj = match d {
                                grammar::StreamDelta::Text(s) => {
                                    if !sent_role {
                                        sent_role = true;
                                        json!({"role": "assistant", "content": s})
                                    } else {
                                        json!({"content": s})
                                    }
                                }
                                grammar::StreamDelta::Reasoning(s) => {
                                    if !sent_role {
                                        sent_role = true;
                                        json!({"role": "assistant", "reasoning_content": s})
                                    } else {
                                        json!({"reasoning_content": s})
                                    }
                                }
                                grammar::StreamDelta::ToolCall(tc) => {
                                    let idx = tc_index;
                                    tc_index += 1;
                                    let tc_delta = json!([{
                                        "index": idx,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.name,
                                            "arguments": tc.arguments
                                        }
                                    }]);
                                    if !sent_role {
                                        sent_role = true;
                                        json!({"role": "assistant", "content": serde_json::Value::Null, "tool_calls": tc_delta})
                                    } else {
                                        json!({"tool_calls": tc_delta})
                                    }
                                }
                            };
                            let chunk = json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
                            });
                            let _ = sse_tx.send(Ok(Event::default().data(chunk.to_string())));
                        }
                    }
                    GenEvent::Done { finish_reason, pred_n, .. } => {
                        for d in parser.finish(&tok) {
                            if let grammar::StreamDelta::ToolCall(tc) = d {
                                let idx = tc_index;
                                tc_index += 1;
                                let tc_delta = json!([{
                                    "index": idx,
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments
                                    }
                                }]);
                                let d_obj = if !sent_role {
                                    sent_role = true;
                                    json!({"role": "assistant", "content": serde_json::Value::Null, "tool_calls": tc_delta})
                                } else {
                                    json!({"tool_calls": tc_delta})
                                };
                                let chunk = json!({
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
                                });
                                let _ = sse_tx.send(Ok(Event::default().data(chunk.to_string())));
                            }
                        }
                        let final_fr = if tc_index > 0 { "tool_calls" } else { finish_reason };
                        let done_chunk = json!({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": final_fr}]
                        });
                        let _ = sse_tx.send(Ok(Event::default().data(done_chunk.to_string())));
                        if include_usage {
                            let usage_chunk = json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": prompt_n_sse,
                                    "completion_tokens": pred_n,
                                    "total_tokens": prompt_n_sse + pred_n,
                                    "prompt_tokens_details": {"cached_tokens": cached_n_sse}
                                }
                            });
                            let _ = sse_tx.send(Ok(Event::default().data(usage_chunk.to_string())));
                        }
                        let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                        break;
                    }
                    GenEvent::Error(_) => {
                        let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                        break;
                    }
                }
            }
        });
        return Sse::new(UnboundedReceiverStream::new(sse_rx))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let (out_ids, prompt_n, cached_n, pp_ms, dec_ms, mut finish_reason) =
        match collect_job(&st, prompt_ids, images, max_tokens, initial_thinking, json_grammar, sampling).await {
            Ok(v) => v,
            Err(e) => {
                st.metrics.requests_err.fetch_add(1, Ordering::Relaxed);
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response();
            }
        };
    st.metrics.requests_ok.fetch_add(1, Ordering::Relaxed);
    let pred_n = out_ids.len();
    let mut parsed = grammar::parse_inkling_tokens(&out_ids, &st.tokenizer, initial_thinking, &allowed_tools, &call_prefix);
    if !stop_seqs.is_empty() && parsed.tool_calls.is_empty() {
        let mut earliest: Option<usize> = None;
        for ss in &stop_seqs {
            if let Some(pos) = parsed.content.find(ss.as_str()) {
                earliest = Some(match earliest {
                    Some(cur) => cur.min(pos),
                    None => pos,
                });
            }
        }
        if let Some(cut) = earliest {
            parsed.content.truncate(cut);
            finish_reason = "stop";
        }
    }
    let mut msg_obj = if !parsed.tool_calls.is_empty() {
        finish_reason = "tool_calls";
        let tcs: Vec<serde_json::Value> = parsed.tool_calls.iter().map(|tc| tc.to_chat_json()).collect();
        let content_val = if parsed.content.is_empty() {
            serde_json::Value::Null
        } else {
            json!(parsed.content)
        };
        json!({"role": "assistant", "content": content_val, "tool_calls": tcs})
    } else {
        json!({"role": "assistant", "content": parsed.content})
    };
    if let Some(r) = parsed.reasoning {
        msg_obj["reasoning_content"] = json!(r);
    }
    Json(json!({
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": msg_obj,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_n,
            "completion_tokens": pred_n,
            "total_tokens": prompt_n + pred_n,
            "prompt_tokens_details": {"cached_tokens": cached_n}
        },
        "timings": {
            "prompt_n": prompt_n,
            "cached_n": cached_n,
            "prompt_ms": pp_ms,
            "prompt_per_second": if pp_ms > 0.0 { prompt_n as f64 / (pp_ms / 1000.0) } else { 0.0 },
            "predicted_n": pred_n,
            "predicted_ms": dec_ms,
            "predicted_per_second": if dec_ms > 0.0 { pred_n as f64 / (dec_ms / 1000.0) } else { 0.0 }
        }
    }))
    .into_response()
}

async fn completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let model_id = req.model.unwrap_or_else(|| "inkling-2-mini".into());
    let max_tokens = req.max_tokens.unwrap_or(256).min(8192);
    let _stream = req.stream.unwrap_or(false);
    let prompts: Vec<String> = match req.prompt {
        PromptInput::Single(s) => vec![s],
        PromptInput::Batch(v) => v,
    };
    let created = now_unix();
    let req_id = next_id("cmpl");

    let mut handles = Vec::with_capacity(prompts.len());
    for pr in prompts {
        let enc = match st.tokenizer.encode(pr, false) {
            Ok(e) => e,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
        };
        let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
        let st_c = st.clone();
        handles.push(tokio::spawn(async move {
            collect_job(&st_c, prompt_ids, Vec::new(), max_tokens, false, false, grammar::SamplingParams::greedy()).await
        }));
    }

    let mut choices = Vec::with_capacity(handles.len());
    let mut total_prompt = 0usize;
    let mut total_pred = 0usize;
    let mut max_dec_ms = 0.0f64;
    for (idx, h) in handles.into_iter().enumerate() {
        let res = match h.await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response(),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("{e}")}))).into_response(),
        };
        let (out_ids, pn, _, _, dm, fr) = res;
        total_prompt += pn;
        total_pred += out_ids.len();
        if dm > max_dec_ms {
            max_dec_ms = dm;
        }
        let txt = st.tokenizer.decode(&out_ids, true).unwrap_or_default();
        choices.push(json!({
            "index": idx,
            "text": txt,
            "finish_reason": fr
        }));
    }

    Json(json!({
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_pred,
            "total_tokens": total_prompt + total_pred
        },
        "timings": {
            "batch_size": choices.len(),
            "total_predicted_n": total_pred,
            "wall_decode_ms": max_dec_ms,
            "aggregate_tokens_per_second": if max_dec_ms > 0.0 { total_pred as f64 / (max_dec_ms / 1000.0) } else { 0.0 }
        }
    }))
    .into_response()
}

async fn responses_http(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let model_id = req.model.unwrap_or_else(|| "inkling-2-mini".into());
    let max_tokens = req.max_output_tokens.unwrap_or(4096).min(16384);
    let stream = req.stream.unwrap_or(false);
    let enable_thinking = req
        .reasoning
        .as_ref()
        .and_then(|r| r.get("effort"))
        .and_then(|e| e.as_str())
        .map(|s| s != "none")
        .unwrap_or(false);
    let (prompt, images, allowed_tools) = format_responses_input_with_tools(
        &req.input,
        req.instructions.as_deref(),
        req.tools.as_deref(),
        enable_thinking,
    );
    let enc = match st.tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    };
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    let resp_id = next_id("resp");
    let call_prefix = next_id("call");
    let msg_id = next_id("msg");
    let created = now_unix();

    if stream {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let in_flight = match st.dispatch(GenJob { prompt_ids, images, max_tokens, enable_thinking, json_grammar: false, sampling: grammar::SamplingParams::greedy(), tx }) {
            Ok(g) => InFlightGuard(g),
            Err(_) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":"scheduler closed"}))).into_response();
            }
        };
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let tok = st.tokenizer.clone();
        tokio::spawn(async move {
            let _guard = in_flight;
            let mut parser = grammar::InklingStreamParser::new(enable_thinking, call_prefix, allowed_tools);
            let mut msg_started = false;
            let mut output_items: Vec<serde_json::Value> = Vec::new();
            let mut out_index = 0usize;
            let mut prompt_n = 0usize;
            let mut cached_n = 0usize;
            let _ = sse_tx.send(Ok(Event::default().event("response.created").data(
                json!({"type":"response.created","response":{"id":resp_id,"object":"response","status":"in_progress","model":model_id,"created_at":created,"output":[]}}).to_string()
            )));
            while let Some(ev) = rx.recv().await {
                match ev {
                    GenEvent::Prefill { prompt_n: pn, cached_n: cn, .. } => {
                        prompt_n = pn;
                        cached_n = cn;
                    }
                    GenEvent::Token(t) => {
                        for d in parser.push_token(t, &tok) {
                            match d {
                                grammar::StreamDelta::Text(delta) => {
                                    if !msg_started {
                                        msg_started = true;
                                        let _ = sse_tx.send(Ok(Event::default().event("response.output_item.added").data(
                                            json!({
                                                "type": "response.output_item.added",
                                                "output_index": out_index,
                                                "item": {"id": msg_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
                                            }).to_string()
                                        )));
                                    }
                                    let _ = sse_tx.send(Ok(Event::default().event("response.output_text.delta").data(
                                        json!({"type":"response.output_text.delta","response_id":resp_id,"item_id":msg_id,"output_index":out_index,"content_index":0,"delta":delta}).to_string()
                                    )));
                                }
                                grammar::StreamDelta::Reasoning(_) => {}
                                grammar::StreamDelta::ToolCall(tc) => {
                                    if msg_started {
                                        let full_txt = parser.full_text(&tok);
                                        let msg_item = json!({
                                            "id": msg_id,
                                            "type": "message",
                                            "status": "completed",
                                            "role": "assistant",
                                            "content": [{"type": "output_text", "text": full_txt}]
                                        });
                                        let _ = sse_tx.send(Ok(Event::default().event("response.output_text.done").data(
                                            json!({"type":"response.output_text.done","item_id":msg_id,"output_index":out_index,"content_index":0,"text":full_txt}).to_string()
                                        )));
                                        let _ = sse_tx.send(Ok(Event::default().event("response.output_item.done").data(
                                            json!({"type":"response.output_item.done","output_index":out_index,"item":msg_item}).to_string()
                                        )));
                                        output_items.push(msg_item);
                                        out_index += 1;
                                        msg_started = false;
                                    }
                                    let fc_id = next_id("fc");
                                    let fc_added = json!({
                                        "id": fc_id,
                                        "type": "function_call",
                                        "status": "in_progress",
                                        "call_id": tc.id,
                                        "name": tc.name,
                                        "arguments": ""
                                    });
                                    let _ = sse_tx.send(Ok(Event::default().event("response.output_item.added").data(
                                        json!({"type":"response.output_item.added","output_index":out_index,"item":fc_added}).to_string()
                                    )));
                                    let _ = sse_tx.send(Ok(Event::default().event("response.function_call_arguments.delta").data(
                                        json!({"type":"response.function_call_arguments.delta","item_id":fc_id,"output_index":out_index,"delta":tc.arguments}).to_string()
                                    )));
                                    let _ = sse_tx.send(Ok(Event::default().event("response.function_call_arguments.done").data(
                                        json!({"type":"response.function_call_arguments.done","item_id":fc_id,"output_index":out_index,"arguments":tc.arguments}).to_string()
                                    )));
                                    let fc_done = json!({
                                        "id": fc_id,
                                        "type": "function_call",
                                        "status": "completed",
                                        "call_id": tc.id,
                                        "name": tc.name,
                                        "arguments": tc.arguments
                                    });
                                    let _ = sse_tx.send(Ok(Event::default().event("response.output_item.done").data(
                                        json!({"type":"response.output_item.done","output_index":out_index,"item":fc_done}).to_string()
                                    )));
                                    output_items.push(fc_done);
                                    out_index += 1;
                                }
                            }
                        }
                    }
                    GenEvent::Done { pred_n, dec_ms, .. } => {
                        for d in parser.finish(&tok) {
                            if let grammar::StreamDelta::ToolCall(tc) = d {
                                if msg_started {
                                    let full_txt = parser.full_text(&tok);
                                    let msg_item = json!({
                                        "id": msg_id,
                                        "type": "message",
                                        "status": "completed",
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": full_txt}]
                                    });
                                    let _ = sse_tx.send(Ok(Event::default().event("response.output_item.done").data(
                                        json!({"type":"response.output_item.done","output_index":out_index,"item":msg_item}).to_string()
                                    )));
                                    output_items.push(msg_item);
                                    out_index += 1;
                                    msg_started = false;
                                }
                                let fc_id = next_id("fc");
                                let fc_done = json!({
                                    "id": fc_id,
                                    "type": "function_call",
                                    "status": "completed",
                                    "call_id": tc.id,
                                    "name": tc.name,
                                    "arguments": tc.arguments
                                });
                                let _ = sse_tx.send(Ok(Event::default().event("response.output_item.added").data(
                                    json!({"type":"response.output_item.added","output_index":out_index,"item":fc_done}).to_string()
                                )));
                                let _ = sse_tx.send(Ok(Event::default().event("response.function_call_arguments.delta").data(
                                    json!({"type":"response.function_call_arguments.delta","item_id":fc_id,"output_index":out_index,"delta":tc.arguments}).to_string()
                                )));
                                let _ = sse_tx.send(Ok(Event::default().event("response.function_call_arguments.done").data(
                                    json!({"type":"response.function_call_arguments.done","item_id":fc_id,"output_index":out_index,"arguments":tc.arguments}).to_string()
                                )));
                                let _ = sse_tx.send(Ok(Event::default().event("response.output_item.done").data(
                                    json!({"type":"response.output_item.done","output_index":out_index,"item":fc_done}).to_string()
                                )));
                                output_items.push(fc_done);
                                out_index += 1;
                            }
                        }
                        let full = parser.full_text(&tok);
                        if msg_started {
                            let msg_item = json!({
                                "id": msg_id,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": full}]
                            });
                            let _ = sse_tx.send(Ok(Event::default().event("response.output_text.done").data(
                                json!({"type":"response.output_text.done","item_id":msg_id,"output_index":out_index,"content_index":0,"text":full}).to_string()
                            )));
                            let _ = sse_tx.send(Ok(Event::default().event("response.output_item.done").data(
                                json!({"type":"response.output_item.done","output_index":out_index,"item":msg_item}).to_string()
                            )));
                            output_items.push(msg_item);
                        }
                        let _ = sse_tx.send(Ok(Event::default().event("response.completed").data(
                            json!({
                                "type":"response.completed",
                                "response":{
                                    "id":resp_id,
                                    "object":"response",
                                    "status":"completed",
                                    "model":model_id,
                                    "output":output_items,
                                    "output_text":full,
                                    "usage":{
                                        "input_tokens":prompt_n,
                                        "input_tokens_details":{"cached_tokens":cached_n},
                                        "output_tokens":pred_n,
                                        "total_tokens":prompt_n + pred_n
                                    },
                                    "timings":{"predicted_n":pred_n,"predicted_ms":dec_ms}
                                }
                            }).to_string()
                        )));
                        break;
                    }
                    GenEvent::Error(e) => {
                        let _ = sse_tx.send(Ok(Event::default().event("error").data(json!({"type":"error","error":e}).to_string())));
                        break;
                    }
                }
            }
        });
        return Sse::new(UnboundedReceiverStream::new(sse_rx))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let (out_ids, prompt_n, cached_n, pp_ms, dec_ms, _) =
        match collect_job(&st, prompt_ids, images, max_tokens, enable_thinking, false, grammar::SamplingParams::greedy()).await {
            Ok(v) => v,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response(),
        };
    let pred_n = out_ids.len();
    let parsed = grammar::parse_inkling_tokens(&out_ids, &st.tokenizer, enable_thinking, &allowed_tools, &call_prefix);
    let mut output_items: Vec<serde_json::Value> = Vec::new();
    if !parsed.content.is_empty() || parsed.tool_calls.is_empty() {
        output_items.push(json!({
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": parsed.content}]
        }));
    }
    for tc in &parsed.tool_calls {
        output_items.push(json!({
            "id": next_id("fc"),
            "type": "function_call",
            "status": "completed",
            "call_id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments
        }));
    }
    Json(json!({
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model_id,
        "output": output_items,
        "output_text": parsed.content,
        "usage": {
            "input_tokens": prompt_n,
            "output_tokens": pred_n,
            "total_tokens": prompt_n + pred_n,
            "input_tokens_details": {"cached_tokens": cached_n}
        },
        "timings": {
            "prompt_ms": pp_ms,
            "predicted_ms": dec_ms,
            "predicted_per_second": if dec_ms > 0.0 { pred_n as f64 / (dec_ms / 1000.0) } else { 0.0 }
        }
    }))
    .into_response()
}

async fn responses_ws(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<WsAuthQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    let ok = check_auth_keys(&headers, &st.api_keys)
        || check_key_str(q.api_key.as_deref(), &st.api_keys);
    if !ok {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    ws.on_upgrade(move |socket| handle_ws(socket, st))
}

async fn handle_ws(mut socket: WebSocket, st: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.next().await {
        let txt = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            Message::Ping(p) => {
                let _ = socket.send(Message::Pong(p)).await;
                continue;
            }
            _ => continue,
        };
        let val: serde_json::Value = match serde_json::from_str(&txt) {
            Ok(v) => v,
            Err(e) => {
                let _ = socket.send(Message::Text(json!({"type":"error","error":format!("{e}")}).to_string())).await;
                continue;
            }
        };
        let body = val.get("response").unwrap_or(&val);
        let model_id = body.get("model").and_then(|v| v.as_str()).unwrap_or("inkling-2-mini").to_string();
        let max_tokens = body.get("max_output_tokens").and_then(|v| v.as_u64()).unwrap_or(4096).min(16384) as usize;
        let instructions = body.get("instructions").and_then(|v| v.as_str());
        let input = body.get("input").cloned().unwrap_or_else(|| json!(""));
        let tools_vec = body.get("tools").and_then(|v| v.as_array()).cloned();
        let (prompt, images, allowed_tools) = format_responses_input_with_tools(&input, instructions, tools_vec.as_deref(), false);
        let enc = match st.tokenizer.encode(prompt, false) {
            Ok(e) => e,
            Err(e) => {
                let _ = socket.send(Message::Text(json!({"type":"error","error":format!("{e}")}).to_string())).await;
                continue;
            }
        };
        let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
        let resp_id = next_id("resp");
        let call_prefix = next_id("call");
        let msg_id = next_id("msg");
        let (tx, mut rx) = mpsc::unbounded_channel();
        let _guard = match st.dispatch(GenJob { prompt_ids, images, max_tokens, enable_thinking: false, json_grammar: false, sampling: grammar::SamplingParams::greedy(), tx }) {
            Ok(g) => InFlightGuard(g),
            Err(_) => {
                let _ = socket.send(Message::Text(json!({"type":"error","error":"scheduler closed"}).to_string())).await;
                break;
            }
        };
        let _ = socket.send(Message::Text(json!({
            "type": "response.created",
            "response": {"id": resp_id, "object": "response", "status": "in_progress", "model": model_id, "output": []}
        }).to_string())).await;

        let mut parser = grammar::InklingStreamParser::new(false, call_prefix, allowed_tools);
        let mut output_items: Vec<serde_json::Value> = Vec::new();
        let mut msg_started = false;
        let mut out_index = 0usize;
        let mut prompt_n = 0usize;
        let mut cached_n = 0usize;
        let mut pp_ms = 0.0f64;
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Prefill { prompt_n: pn, cached_n: cn, pp_ms: pm } => {
                    prompt_n = pn;
                    cached_n = cn;
                    pp_ms = pm;
                }
                GenEvent::Token(t) => {
                    for d in parser.push_token(t, &st.tokenizer) {
                        match d {
                            grammar::StreamDelta::Text(delta) => {
                                msg_started = true;
                                if socket.send(Message::Text(json!({
                                    "type": "response.output_text.delta",
                                    "response_id": resp_id,
                                    "item_id": msg_id,
                                    "output_index": out_index,
                                    "delta": delta
                                }).to_string())).await.is_err() {
                                    return;
                                }
                            }
                            grammar::StreamDelta::Reasoning(_) => {}
                            grammar::StreamDelta::ToolCall(tc) => {
                                if msg_started {
                                    let full_txt = parser.full_text(&st.tokenizer);
                                    output_items.push(json!({
                                        "id": msg_id,
                                        "type": "message",
                                        "status": "completed",
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": full_txt}]
                                    }));
                                    out_index += 1;
                                    msg_started = false;
                                }
                                let fc_id = next_id("fc");
                                let fc_done = json!({
                                    "id": fc_id,
                                    "type": "function_call",
                                    "status": "completed",
                                    "call_id": tc.id,
                                    "name": tc.name,
                                    "arguments": tc.arguments
                                });
                                let _ = socket.send(Message::Text(json!({
                                    "type": "response.output_item.done",
                                    "output_index": out_index,
                                    "item": fc_done
                                }).to_string())).await;
                                output_items.push(fc_done);
                                out_index += 1;
                            }
                        }
                    }
                }
                GenEvent::Done { pred_n, dec_ms, .. } => {
                    for d in parser.finish(&st.tokenizer) {
                        if let grammar::StreamDelta::ToolCall(tc) = d {
                            let fc_id = next_id("fc");
                            let fc_done = json!({
                                "id": fc_id,
                                "type": "function_call",
                                "status": "completed",
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments
                            });
                            let _ = socket.send(Message::Text(json!({
                                "type": "response.output_item.done",
                                "output_index": out_index,
                                "item": fc_done
                            }).to_string())).await;
                            output_items.push(fc_done);
                            out_index += 1;
                        }
                    }
                    let full = parser.full_text(&st.tokenizer).trim().to_string();
                    if msg_started || output_items.is_empty() {
                        output_items.push(json!({
                            "id": msg_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": full}]
                        }));
                    }
                    let _ = socket.send(Message::Text(json!({
                        "type": "response.completed",
                        "response": {
                            "id": resp_id,
                            "object": "response",
                            "status": "completed",
                            "model": model_id,
                            "output": output_items,
                            "output_text": full,
                            "usage": {
                                "input_tokens": prompt_n,
                                "output_tokens": pred_n,
                                "total_tokens": prompt_n + pred_n,
                                "input_tokens_details": {"cached_tokens": cached_n}
                            },
                            "timings": {
                                "prompt_ms": pp_ms,
                                "predicted_ms": dec_ms,
                                "predicted_per_second": if dec_ms > 0.0 { pred_n as f64 / (dec_ms / 1000.0) } else { 0.0 }
                            }
                        }
                    }).to_string())).await;
                    break;
                }
                GenEvent::Error(e) => {
                    let _ = socket.send(Message::Text(json!({"type":"error","error":e}).to_string())).await;
                    break;
                }
            }
        }
    }
}

struct CliOpts {
    cmd: String,
    prompt: Option<String>,
    image: Option<String>,
    model: String,
    tokenizer: Option<String>,
    dflash2: Option<String>,
    dspark: Option<String>,
    mmproj: Option<String>,
    api_key: String,
    metrics_auth: bool,
    prefill_chunk_size: usize,
    kv_block_size: usize,
    ctx: usize,
    max_seqs: usize,
    max_tokens: usize,
    port: u16,
    gpus: String,
    tp: usize,
    use_spec: bool,
    use_vision: bool,
    thinking: bool,
}

fn print_usage() {
    println!("RAD - High-performance Rust + ROCm/HIP LLM inference engine");
    println!();
    println!("USAGE:");
    println!("  rad serve [OPTIONS]            Start OpenAI HTTP/SSE/WebSocket server");
    println!("  rad run <PROMPT> [OPTIONS]     Run one-shot local generation");
    println!("  rad bench [OPTIONS]            Run built-in decode, prefill, and multi-user benchmarks");
    println!("  rad info [OPTIONS]             Print engine configuration and detected files");
    println!();
    println!("OPTIONS:");
    println!("  -m, --model <PATH>             Target GGUF model path (env: RAD_MODEL)");
    println!("  -t, --tokenizer <PATH>         HuggingFace tokenizer.json (auto-detected if next to model)");
    println!("      --dflash2 <PATH>           DFlash 2 safetensors dir/file (auto-detected, preferred over DSpark)");
    println!("  -d, --dspark <PATH>            DSpark draft GGUF (auto-detected if next to model)");
    println!("      --mmproj <PATH>            Vision mmproj GGUF (auto-detected, enabled by default)");
    println!("  -i, --image <PATH>             Image file for `rad run` multimodal inference");
    println!("  -p, --port <PORT>              Server port [default: 6733] (env: RAD_PORT)");
    println!("      --ctx <TOKENS>             Context length per sequence [default: 8192] (env: RAD_CTX)");
    println!("      --max-seqs <N>             Max concurrent sequences [default: 32] (env: RAD_MAX_SEQS)");
    println!("      --prefill-chunk-size <N>   Chunked prefill tokens per tick [default: 512] (env: RAD_PREFILL_CHUNK_SIZE)");
    println!("      --kv-block-size <N>        Paged KV cache block size [default: 16] (env: RAD_KV_BLOCK_SIZE)");
    println!("  -n, --max-tokens <N>           Max tokens for `rad run` [default: 256]");
    println!("      --api-key <KEY[,KEY2]>     Optional Bearer/X-API-Key auth (env: RAD_API_KEY)");
    println!("      --metrics-auth             Require API key for GET /metrics");
    println!("      --gpus <LIST|auto>         HIP GPU IDs for multi-GPU data-parallel serving (e.g. 0,1 or auto) [env: RAD_GPUS]");
    println!("      --tp <N>                   Tensor-parallel degree (default: 1) [env: RAD_TP]");
    println!("      --no-spec                  Disable speculative decoding (DFlash 2 / DSpark)");
    println!("      --no-vision                Disable vision encoder loading");
    println!("      --thinking                 Enable thinking tokens for `rad run`");
    println!("  -h, --help                     Show this help message");
}

fn resolve_sibling(model_path: &str, candidates: &[&str]) -> Option<String> {
    let parent = Path::new(model_path).parent()?;
    for c in candidates {
        let p = parent.join(c);
        if p.exists() {
            return Some(p.to_string_lossy().into_owned());
        }
    }
    None
}

fn parse_cli() -> anyhow::Result<CliOpts> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cmd = "serve".to_string();
    let mut idx = 0usize;
    if let Some(first) = args.first() {
        match first.as_str() {
            "serve" | "run" | "bench" | "info" => {
                cmd = first.clone();
                idx = 1;
            }
            "-h" | "--help" | "help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {}
        }
    }
    let mut prompt: Option<String> = None;
    let mut image: Option<String> = None;
    let mut model = std::env::var("RAD_MODEL")
        .unwrap_or_else(|_| "./models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf".into());
    let mut tokenizer = std::env::var("RAD_TOKENIZER").ok();
    let mut dflash2 = std::env::var("RAD_DFLASH2").ok();
    let mut dspark = std::env::var("RAD_DSPARK").ok();
    let mut mmproj = std::env::var("RAD_MMPROJ").ok();
    let mut api_key = std::env::var("RAD_API_KEY").unwrap_or_default();
    let mut metrics_auth = std::env::var("RAD_METRICS_AUTH").map(|v| v == "1").unwrap_or(false);
    let mut prefill_chunk_size: usize = std::env::var("RAD_PREFILL_CHUNK_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let mut kv_block_size: usize = std::env::var("RAD_KV_BLOCK_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let mut ctx: usize = std::env::var("RAD_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let mut max_seqs: usize = std::env::var("RAD_MAX_SEQS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let mut max_tokens: usize = 256;
    let mut port: u16 = std::env::var("RAD_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(6733);
    let mut gpus = std::env::var("RAD_GPUS").unwrap_or_default();
    let mut tp: usize = std::env::var("RAD_TP").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut use_spec = std::env::var("RAD_SPEC").map(|v| v != "0").unwrap_or(true);
    let mut use_vision = std::env::var("RAD_VISION").map(|v| v != "0").unwrap_or(true);
    let mut thinking = false;

    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "-m" | "--model" => {
                idx += 1;
                model = args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --model value"))?;
            }
            "-t" | "--tokenizer" => {
                idx += 1;
                tokenizer = Some(args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --tokenizer value"))?);
            }
            "--dflash2" => {
                idx += 1;
                dflash2 = Some(args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --dflash2 value"))?);
            }
            "-d" | "--dspark" => {
                idx += 1;
                dspark = Some(args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --dspark value"))?);
            }
            "--mmproj" => {
                idx += 1;
                mmproj = Some(args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --mmproj value"))?);
            }
            "-i" | "--image" => {
                idx += 1;
                image = Some(args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --image value"))?);
            }
            "-p" | "--port" => {
                idx += 1;
                port = args.get(idx).ok_or_else(|| anyhow::anyhow!("missing --port value"))?.parse()?;
            }
            "--ctx" => {
                idx += 1;
                ctx = args.get(idx).ok_or_else(|| anyhow::anyhow!("missing --ctx value"))?.parse()?;
            }
            "--max-seqs" => {
                idx += 1;
                max_seqs = args.get(idx).ok_or_else(|| anyhow::anyhow!("missing --max-seqs value"))?.parse()?;
            }
            "-n" | "--max-tokens" => {
                idx += 1;
                max_tokens = args.get(idx).ok_or_else(|| anyhow::anyhow!("missing --max-tokens value"))?.parse()?;
            }
            "--api-key" => {
                idx += 1;
                api_key = args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --api-key value"))?;
            }
            "--metrics-auth" => {
                metrics_auth = true;
            }
            "--gpus" => {
                idx += 1;
                gpus = args.get(idx).cloned().ok_or_else(|| anyhow::anyhow!("missing --gpus value"))?;
            }
            "--tp" => {
                idx += 1;
                tp = args.get(idx).ok_or_else(|| anyhow::anyhow!("missing --tp value"))?.parse()?;
            }
            "--prefill-chunk-size" => {
                idx += 1;
                prefill_chunk_size = args.get(idx).ok_or_else(|| anyhow::anyhow!("missing --prefill-chunk-size value"))?.parse()?;
            }
            "--kv-block-size" => {
                idx += 1;
                kv_block_size = args.get(idx).ok_or_else(|| anyhow::anyhow!("missing --kv-block-size value"))?.parse()?;
            }
            "--no-spec" => {
                use_spec = false;
            }
            "--no-vision" => {
                use_vision = false;
            }
            "--thinking" => {
                thinking = true;
            }
            other if !other.starts_with('-') && cmd == "run" && prompt.is_none() => {
                prompt = Some(other.to_string());
            }
            other => {
                anyhow::bail!("unknown argument: {other} (run `rad --help`)");
            }
        }
        idx += 1;
    }

    if tokenizer.is_none() {
        tokenizer = resolve_sibling(&model, &["tokenizer.json", "../inkling2-src/tokenizer.json"]);
    }
    if dflash2.is_none() {
        dflash2 = resolve_sibling(&model, &["../inkling2-dflash2/model.safetensors", "dflash2/model.safetensors", "model.safetensors"]);
    }
    if dspark.is_none() {
        dspark = resolve_sibling(&model, &["dspark-Inkling-2-Mini-Q8_0.gguf", "dspark.gguf"]);
    }
    if mmproj.is_none() {
        mmproj = resolve_sibling(&model, &["mmproj-F16.gguf", "mmproj.gguf"]);
    }

    Ok(CliOpts {
        cmd,
        prompt,
        image,
        model,
        tokenizer,
        dflash2,
        dspark,
        mmproj,
        api_key,
        metrics_auth,
        prefill_chunk_size,
        kv_block_size,
        ctx,
        max_seqs,
        max_tokens,
        port,
        gpus,
        tp,
        use_spec,
        use_vision,
        thinking,
    })
}

fn load_drafter(opts: &CliOpts, engine: &mut InklingEngine, max_seqs: usize) -> anyhow::Result<Option<Drafter>> {
    if !opts.use_spec {
        return Ok(None);
    }
    if let Some(ref p) = opts.dflash2 {
        if Path::new(p).exists() {
            for &bk in &[8usize, 12, 16, 24, 32, 48, 64, 96, 128] {
                engine.capture_verify_graph(bk)?;
            }
            let mut d = DFlash2Drafter::load_multi(Path::new(p), opts.ctx, max_seqs)?;
            d.capture_draft_graph(engine, 8)?;
            return Ok(Some(Drafter::DFlash2(d)));
        }
    }
    if let Some(ref p) = opts.dspark {
        if Path::new(p).exists() {
            engine.capture_verify_graph(6)?;
            let mut d = DsparkDrafter::load_multi(Path::new(p), opts.ctx, max_seqs)?;
            d.capture_draft_graph(engine, 6)?;
            return Ok(Some(Drafter::Dspark(d)));
        }
    }
    Ok(None)
}

fn load_vision(opts: &CliOpts) -> anyhow::Result<Option<VisionEncoder>> {
    if !opts.use_vision {
        return Ok(None);
    }
    if let Some(ref p) = opts.mmproj {
        if Path::new(p).exists() {
            return Ok(Some(VisionEncoder::load(Path::new(p))?));
        }
    }
    Ok(None)
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len().max(1) as f64;
    let m = vals.iter().sum::<f64>() / n;
    let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, v.sqrt())
}

fn run_bench(opts: &CliOpts, tok_path: &str) -> anyhow::Result<()> {
    println!("RAD Benchmark Suite (v{})", env!("CARGO_PKG_VERSION"));
    let t0 = Instant::now();
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let max_seqs = if opts.ctx > 32768 {
        opts.max_seqs.max(1)
    } else if opts.ctx > 8192 {
        opts.max_seqs.max(4)
    } else {
        opts.max_seqs.max(16)
    };
    let mut engine = InklingEngine::load_multi(Path::new(&opts.model), opts.ctx.max(4096), max_seqs)?;
    engine.capture_decode_graph()?;
    let mut drafter = load_drafter(opts, &mut engine, max_seqs.min(8))?;
    let load_s = t0.elapsed().as_secs_f64();
    let spec_mode = drafter.as_ref().map(|d| d.mode_name()).unwrap_or("none");
    println!("Loaded model in {:.2}s (spec={spec_mode}, ctx={}, max_seqs={max_seqs})", load_s, opts.ctx);
    println!();

    let short_p = format_prompt(
        &[ChatMessage { role: "user".into(), content: json!("Explain GPU memory bandwidth in one paragraph."), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
        false,
    );
    let short_ids: Vec<i32> = tokenizer.encode(short_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
    engine.reset_seq(0)?;
    let _ = engine.prefill_tokens_seq(0, &short_ids)?;
    let mut ttft_runs = Vec::new();
    let mut cur_tok = 0i32;
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let ts = Instant::now();
        cur_tok = engine.prefill_tokens_seq(0, &short_ids)?;
        ttft_runs.push(ts.elapsed().as_secs_f64() * 1000.0);
    }
    let (ttft_ms, ttft_sd) = mean_std(&ttft_runs);

    let dec_n = 128usize;
    let mut tg128_runs = Vec::new();
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let mut ct = engine.prefill_tokens_seq(0, &short_ids)?;
        let td = Instant::now();
        for _ in 0..dec_n {
            ct = engine.step(ct)?;
        }
        let _ = ct;
        tg128_runs.push(dec_n as f64 / td.elapsed().as_secs_f64());
    }
    let (tg128_tps, tg128_sd) = mean_std(&tg128_runs);

    let long_chunk = "Modern GPU architectures execute thousands of lightweight threads in parallel across SIMD wavefronts while hiding global HBM latency with large register files, shared local data store memory, and multi-level caches. ";
    let mut long_text = String::from("Summarize the following technical notes:\n");
    for _ in 0..80 {
        long_text.push_str(long_chunk);
    }
    let long_p = format_prompt(
        &[ChatMessage { role: "user".into(), content: json!(long_text), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
        false,
    );
    let base_ids: Vec<i32> = tokenizer.encode(long_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
    let pp512_ids = &base_ids[..512.min(base_ids.len())];
    let pp2048_ids = &base_ids[..2048.min(base_ids.len())];

    engine.reset_seq(0)?;
    let _ = engine.prefill_tokens_seq(0, pp512_ids)?;
    let mut pp512_runs = Vec::new();
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let tpp = Instant::now();
        let _ = engine.prefill_tokens_seq(0, pp512_ids)?;
        pp512_runs.push(pp512_ids.len() as f64 / tpp.elapsed().as_secs_f64());
    }
    let (pp512_tps, pp512_sd) = mean_std(&pp512_runs);

    engine.reset_seq(0)?;
    let _ = engine.prefill_tokens_seq(0, pp2048_ids)?;
    let mut pp2048_runs = Vec::new();
    let mut last_d2048 = 0i32;
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let tpp = Instant::now();
        last_d2048 = engine.prefill_tokens_seq(0, pp2048_ids)?;
        pp2048_runs.push(pp2048_ids.len() as f64 / tpp.elapsed().as_secs_f64());
    }
    let (pp2048_tps, pp2048_sd) = mean_std(&pp2048_runs);

    let td2k = Instant::now();
    let mut ct2k = last_d2048;
    for _ in 0..dec_n {
        ct2k = engine.step(ct2k)?;
    }
    let _ = ct2k;
    let tg128_d2048_tps = dec_n as f64 / td2k.elapsed().as_secs_f64();

    let mut pp2048_spec_tps = 0.0f64;
    let mut pp2048_spec_sd = 0.0f64;
    if let Some(ref mut d) = drafter {
        engine.reset_seq(0)?;
        d.reset_seq(0);
        let _ = prefill_with_drafter(&mut engine, 0, pp2048_ids, &[], None, Some(d), None)?;
        let mut s_runs = Vec::new();
        for _ in 0..3 {
            engine.reset_seq(0)?;
            d.reset_seq(0);
            let tps = Instant::now();
            let _ = prefill_with_drafter(&mut engine, 0, pp2048_ids, &[], None, Some(d), None)?;
            s_runs.push(pp2048_ids.len() as f64 / tps.elapsed().as_secs_f64());
        }
        let (m, s) = mean_std(&s_runs);
        pp2048_spec_tps = m;
        pp2048_spec_sd = s;
    }

    let _ = cur_tok;
    println!("| Benchmark | Result | Details |");
    println!("| --- | --- | --- |");
    println!("| Cold Load | {:.2} s | spec={spec_mode}, hipGraph captured |", load_s);
    println!("| Short Prompt TTFT | {:.1} ± {:.1} ms | {} tokens (3 runs) |", ttft_ms, ttft_sd, short_ids.len());
    println!("| pp512 (No-Spec) | {:.1} ± {:.1} tok/s | {} tokens, llama-bench parity (3 runs) |", pp512_tps, pp512_sd, pp512_ids.len());
    println!("| pp2048 (No-Spec) | {:.1} ± {:.1} tok/s | {} tokens, llama-bench parity (3 runs) |", pp2048_tps, pp2048_sd, pp2048_ids.len());
    if pp2048_spec_tps > 0.0 {
        println!("| pp2048 ({spec_mode}) | {:.1} ± {:.1} tok/s | {} tokens, batched feature inject (3 runs) |", pp2048_spec_tps, pp2048_spec_sd, pp2048_ids.len());
    }
    println!("| tg128 (No-Spec) | {:.1} ± {:.1} tok/s | B=1 hipGraph, d=0 (3 runs) |", tg128_tps, tg128_sd);
    println!("| tg128 @ d=2048 (No-Spec) | {:.1} tok/s | B=1 hipGraph, 2K KV depth |", tg128_d2048_tps);

    if let Some(ref mut d) = drafter {
        let k_len = d.spec_k();
        let mut long_code_ctx = String::from("Here is an existing Rust networking library:\n```rust\n");
        for i in 0..38 {
            long_code_ctx.push_str(&format!(
                "pub struct BufferPool{i} {{ cap: usize, used: usize }}\nimpl BufferPool{i} {{ pub fn new(cap: usize) -> Self {{ Self {{ cap, used: 0 }} }} pub fn remaining(&self) -> usize {{ self.cap - self.used }} }}\n"
            ));
        }
        long_code_ctx.push_str("```\nNow write a complete Python implementation of quicksort and mergesort with a benchmark comparing them on random arrays.");
        let prompts: [(&str, &str); 4] = [
            ("Code", "Show me a Python implementation of quicksort and mergesort with a benchmark comparing them on random arrays."),
            ("Code (2K ctx)", &long_code_ctx),
            ("Math", "Derive the closed-form formula for the sum of squares 1^2 + 2^2 + ... + n^2 step by step and verify for n=5."),
            ("Prose", "Explain how modern GPU memory hierarchies work, from registers to L1/L2 caches and HBM, and why memory bandwidth limits LLM decoding."),
        ];
        for (label, text) in prompts {
            let p = format_prompt(&[ChatMessage { role: "user".into(), content: json!(text), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }], false);
            let ids: Vec<i32> = tokenizer.encode(p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            engine.reset_seq(0)?;
            d.reset_seq(0);
            let (mut ct, _) = prefill_with_drafter(&mut engine, 0, &ids, &[], None, Some(d), None)?;
            let mut gen = 0usize;
            let mut rounds = 0usize;
            let mut d_ms = 0.0f64;
            let mut v_ms = 0.0f64;
            let mut i_ms = 0.0f64;
            let t1 = Instant::now();
            while gen < 160 {
                let td0 = Instant::now();
                let draft = d.draft(ct, &engine, k_len)?;
                d_ms += td0.elapsed().as_secs_f64() * 1000.0;
                let pos0 = engine.pos;
                let tv0 = Instant::now();
                let preds = engine.verify_batch(&draft)?;
                v_ms += tv0.elapsed().as_secs_f64() * 1000.0;
                let mut n_acc = 1usize;
                for j in 0..(k_len - 1) {
                    if preds[j] == draft[j + 1] { n_acc += 1; } else { break; }
                }
                engine.rewind_verify_to(n_acc, k_len)?;
                engine.seq_pos[0] = engine.pos;
                let ti0 = Instant::now();
                d.inject_batch_seq_async(0, &engine, 0, pos0, n_acc)?;
                i_ms += ti0.elapsed().as_secs_f64() * 1000.0;
                ct = preds[n_acc - 1];
                gen += n_acc;
                rounds += 1;
                if ct == 200006 || ct == 200008 { break; }
            }
            let tps = gen as f64 / t1.elapsed().as_secs_f64();
            let avg_e = gen as f64 / rounds.max(1) as f64;
            println!(
                "| Single-User {label} ({spec_mode} K={k_len}) | {:.1} tok/s | emit={:.2}, draft={:.2}ms, verify={:.2}ms, inject={:.2}ms |",
                tps, avg_e, d_ms / rounds as f64, v_ms / rounds as f64, i_ms / rounds as f64
            );
        }

        for &b in &[4usize, 8usize, 16usize] {
            if b <= d.max_seqs() {
                let mk = 4usize;
                let code_p = format_prompt(
                    &[ChatMessage { role: "user".into(), content: json!("Show me a Python implementation of quicksort and mergesort."), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
                    false,
                );
                let ids: Vec<i32> = tokenizer.encode(code_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
                let seq_ids: Vec<i32> = (0..b as i32).collect();
                let mut cur = vec![0i32; b];
                for u in 0..b {
                    engine.reset_seq(u)?;
                    d.reset_seq(u);
                    let (t, _) = prefill_with_drafter(&mut engine, u, &ids, &[], None, Some(d), None)?;
                    cur[u] = t;
                }
                let _ = d.draft_multi(&seq_ids, &cur, &engine, mk)?;
                let mut tot = 0usize;
                let t1 = Instant::now();
                while tot < b * 80 {
                    let drafts = d.draft_multi(&seq_ids, &cur, &engine, mk)?;
                    let pos0: Vec<i32> = (0..b).map(|u| engine.seq_pos[u]).collect();
                    let preds = engine.verify_multi(&seq_ids, &drafts)?;
                    let mut emits = vec![0usize; b];
                    for u in 0..b {
                        let mut n_acc = 1usize;
                        for j in 0..(mk - 1) {
                            if preds[u][j] == drafts[u][j + 1] { n_acc += 1; } else { break; }
                        }
                        emits[u] = n_acc;
                        cur[u] = preds[u][n_acc - 1];
                        tot += n_acc;
                    }
                    engine.rewind_verify_multi(&seq_ids, &emits, mk)?;
                    for u in 0..b {
                        d.inject_batch_seq_async(u, &engine, u * mk, pos0[u], emits[u])?;
                    }
                    d.sync();
                }
                let agg = tot as f64 / t1.elapsed().as_secs_f64();
                let per_u = agg / b as f64;
                println!("| Multi-User B={b} ({spec_mode} K={mk}) | {:.1} agg tok/s | {:.1} tok/s/user (ITL {:.1}ms) |", agg, per_u, 1000.0 / per_u);
            }
        }
    }

    for &b in &[8usize, 16usize, 32usize, 64usize] {
        if b <= max_seqs {
            let seq_ids: Vec<i32> = (0..b as i32).collect();
            let mut cur = vec![0i32; b];
            for u in 0..b {
                engine.reset_seq(u)?;
                cur[u] = engine.prefill_tokens_seq(u, &short_ids)?;
            }
            let _ = engine.step_multi(&seq_ids, &cur)?;
            let steps = 48usize;
            let t1 = Instant::now();
            for _ in 0..steps {
                cur = engine.step_multi(&seq_ids, &cur)?;
            }
            let agg = (b * steps) as f64 / t1.elapsed().as_secs_f64();
            let per_u = agg / b as f64;
            println!("| Multi-User B={b} (No-Spec) | {:.1} agg tok/s | {:.1} tok/s/user (ITL {:.1}ms) |", agg, per_u, 1000.0 / per_u);
        }
    }

    if let Some(ref mut d) = drafter {
        let k_len = d.spec_k();
        let targets: &[usize] = &[2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
        let mut long_targets: Vec<usize> = targets
            .iter()
            .copied()
            .filter(|&t| t <= opts.ctx)
            .map(|t| t.min(opts.ctx.saturating_sub(768)))
            .collect();
        if long_targets.len() >= 2 {
            println!();
            println!("Long-Context Scaling (prefill + hybrid {spec_mode}/ngram K=128 decode):");
            let block_txt = "pub struct KvBlock {\n    pub block_id: usize,\n    pub ref_count: u32,\n    pub tokens: [i32; 16],\n    pub dirty: bool,\n}\nimpl KvBlock {\n    pub fn new(block_id: usize) -> Self {\n        Self { block_id, ref_count: 1, tokens: [0; 16], dirty: false }\n    }\n    pub fn retain(&mut self) {\n        self.ref_count = self.ref_count.saturating_add(1);\n    }\n    pub fn release(&mut self) -> bool {\n        self.ref_count = self.ref_count.saturating_sub(1);\n        self.ref_count == 0\n    }\n}\n";
            let target_txt = "pub struct KvCachePool {\n    pub blocks: Vec<KvBlock>,\n    pub free_list: Vec<usize>,\n    pub block_size: usize,\n    pub max_blocks: usize,\n    pub hits: u64,\n    pub misses: u64,\n}\nimpl KvCachePool {\n    pub fn with_capacity(max_blocks: usize, block_size: usize) -> Self {\n        let mut blocks = Vec::with_capacity(max_blocks);\n        let mut free_list = Vec::with_capacity(max_blocks);\n        for i in 0..max_blocks {\n            blocks.push(KvBlock::new(i));\n            free_list.push(max_blocks - 1 - i);\n        }\n        Self { blocks, free_list, block_size, max_blocks, hits: 0, misses: 0 }\n    }\n    pub fn alloc_block(&mut self) -> Option<usize> {\n        let id = self.free_list.pop()?;\n        self.blocks[id].ref_count = 1;\n        self.blocks[id].dirty = true;\n        Some(id)\n    }\n    pub fn free_block(&mut self, id: usize) {\n        if id < self.blocks.len() && self.blocks[id].release() {\n            self.blocks[id].dirty = false;\n            self.free_list.push(id);\n        }\n    }\n    pub fn retain_block(&mut self, id: usize) -> bool {\n        if id < self.blocks.len() && self.blocks[id].ref_count > 0 {\n            self.blocks[id].retain();\n            self.hits += 1;\n            true\n        } else {\n            self.misses += 1;\n            false\n        }\n    }\n    pub fn write_token(&mut self, id: usize, slot: usize, tok: i32) -> bool {\n        if id < self.blocks.len() && slot < 16 {\n            self.blocks[id].tokens[slot] = tok;\n            self.blocks[id].dirty = true;\n            true\n        } else {\n            false\n        }\n    }\n    pub fn read_token(&self, id: usize, slot: usize) -> Option<i32> {\n        if id < self.blocks.len() && slot < 16 {\n            Some(self.blocks[id].tokens[slot])\n        } else {\n            None\n        }\n    }\n    pub fn used_blocks(&self) -> usize {\n        self.max_blocks.saturating_sub(self.free_list.len())\n    }\n    pub fn free_blocks(&self) -> usize {\n        self.free_list.len()\n    }\n    pub fn utilization(&self) -> f64 {\n        if self.max_blocks == 0 { 0.0 } else { self.used_blocks() as f64 / self.max_blocks as f64 }\n    }\n    pub fn hit_rate(&self) -> f64 {\n        let total = self.hits + self.misses;\n        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }\n    }\n    pub fn reset_all(&mut self) {\n        self.free_list.clear();\n        for i in 0..self.max_blocks {\n            self.blocks[i].ref_count = 0;\n            self.blocks[i].dirty = false;\n            self.free_list.push(self.max_blocks - 1 - i);\n        }\n        self.hits = 0;\n        self.misses = 0;\n    }\n}\n";
            let block_ids: Vec<i32> = tokenizer.encode(block_txt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let target_ids: Vec<i32> = tokenizer.encode(target_txt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let prefix_p = format_prompt(
                &[ChatMessage { role: "user".into(), content: json!("Here is a large Rust codebase file. At the end is `KvCachePool`. Output ONLY the exact `KvCachePool` struct and `impl KvCachePool` code verbatim from the file (no markdown fences, no commentary):\n"), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
                false,
            );
            let mut prefix_ids: Vec<i32> = tokenizer.encode(prefix_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let tail_ids = if prefix_ids.len() >= 2 { prefix_ids.split_off(prefix_ids.len() - 2) } else { Vec::new() };
            for tgt in long_targets.drain(..) {
                let mut ids = Vec::with_capacity(tgt + 64);
                ids.extend_from_slice(&prefix_ids);
                while ids.len() + block_ids.len() + target_ids.len() + tail_ids.len() <= tgt {
                    ids.extend_from_slice(&block_ids);
                }
                ids.extend_from_slice(&target_ids);
                ids.extend_from_slice(&tail_ids);
                engine.reset_seq(0)?;
                d.reset_seq(0);
                let tp = Instant::now();
                let (mut ct, _) = prefill_with_drafter(&mut engine, 0, &ids, &[], None, Some(d), None)?;
                let pp_s = tp.elapsed().as_secs_f64();
                let pp_tps = ids.len() as f64 / pp_s.max(1e-6);
                let mut gen = 0usize;
                let mut rounds = 0usize;
                let td = Instant::now();
                while gen < 512 {
                    let rem_kv = engine.max_ctx.saturating_sub(engine.pos as usize);
                    let draft = if let Some(ng) = ngram_lookup_draft(&engine.seq_tokens[0], ct, 128usize.min(rem_kv)) {
                        ng
                    } else {
                        d.draft(ct, &engine, k_len)?
                    };
                    let dk = draft.len();
                    let pos0 = engine.pos;
                    let preds = engine.verify_batch(&draft)?;
                    let mut n_acc = 1usize;
                    for j in 0..(dk - 1) {
                        if preds[j] == draft[j + 1] { n_acc += 1; } else { break; }
                    }
                    engine.rewind_verify_to(n_acc, dk)?;
                    engine.seq_pos[0] = engine.pos;
                    d.inject_batch_seq_async(0, &engine, 0, pos0, n_acc)?;
                    for j in 0..n_acc {
                        engine.seq_tokens[0].push(draft[j]);
                    }
                    ct = preds[n_acc - 1];
                    gen += n_acc;
                    rounds += 1;
                    if ct == 200006 || ct == 199999 { break; }
                }
                let dec_s = td.elapsed().as_secs_f64().max(1e-6);
                let dec_tps = gen as f64 / dec_s;
                println!(
                    "| Long-Ctx {:>6} tok | prefill {:>6.1} tok/s ({:.2}s) | decode {:>6.1} tok/s (emit={:.1}, {gen} tok) |",
                    ids.len(), pp_tps, pp_s, dec_tps, gen as f64 / rounds.max(1) as f64
                );
            }
        }
    }

    Ok(())
}

fn resolve_gpu_ids(gpus_arg: &str) -> anyhow::Result<(Vec<i32>, usize)> {
    let trimmed = gpus_arg.trim();
    if trimmed.is_empty() {
        let topo = GpuTopology::probe(Some(&[0]))?;
        return Ok((vec![0], topo.p2p_pairs()));
    }
    if trimmed.eq_ignore_ascii_case("auto") {
        let topo = GpuTopology::probe(None)?;
        let min_vram = 16usize * 1024 * 1024 * 1024;
        let mut selected: Vec<i32> = topo
            .devices
            .iter()
            .filter(|d| d.total_bytes >= min_vram)
            .map(|d| d.id)
            .collect();
        if selected.is_empty() && !topo.devices.is_empty() {
            selected.push(topo.devices[0].id);
        }
        let sel_topo = GpuTopology::probe(Some(&selected))?;
        return Ok((selected, sel_topo.p2p_pairs()));
    }
    let mut ids = Vec::new();
    for part in trimmed.split(',') {
        let p = part.trim();
        if !p.is_empty() {
            ids.push(p.parse::<i32>().map_err(|_| anyhow::anyhow!("invalid GPU id: {p}"))?);
        }
    }
    if ids.is_empty() {
        ids.push(0);
    }
    let topo = GpuTopology::probe(Some(&ids))?;
    Ok((ids, topo.p2p_pairs()))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let opts = parse_cli()?;
    let tok_path = opts.tokenizer.clone().unwrap_or_else(|| "./models/tokenizer.json".into());

    if opts.cmd == "bench" {
        return run_bench(&opts, &tok_path);
    }

    if opts.cmd == "info" {
        println!("RAD v{}", env!("CARGO_PKG_VERSION"));
        println!("  model:     {} (exists={})", opts.model, Path::new(&opts.model).exists());
        println!("  tokenizer: {} (exists={})", tok_path, Path::new(&tok_path).exists());
        println!("  dflash2:   {:?}", opts.dflash2);
        println!("  dspark:    {:?}", opts.dspark);
        println!("  mmproj:    {:?} (vision={})", opts.mmproj, opts.use_vision && opts.mmproj.is_some());
        println!("  ctx:       {}", opts.ctx);
        println!("  max_seqs:  {}", opts.max_seqs);
        println!("  port:      {}", opts.port);
        if let Ok(topo) = GpuTopology::probe(None) {
            println!("  gpus:      {} detected (p2p_pairs={})", topo.devices.len(), topo.p2p_pairs());
            for d in &topo.devices {
                let free_gb = d.free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                let tot_gb = d.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                println!("    GPU {}: {:.2} / {:.2} GB free", d.id, free_gb, tot_gb);
            }
            for i in 0..topo.devices.len() {
                for j in (i + 1)..topo.devices.len() {
                    let bi = topo.p2p[i][j] && topo.p2p[j][i];
                    println!("    P2P GPU {} <-> GPU {}: {}", topo.devices[i].id, topo.devices[j].id, if bi { "enabled" } else { "host-staged" });
                }
            }
        }
        println!("  endpoints: /health, /metrics, /v1/models, /v1/chat/completions, /v1/completions, /v1/responses (HTTP/SSE/WS)");
        return Ok(());
    }

    if opts.cmd == "run" {
        let user_prompt = opts.prompt.clone().ok_or_else(|| anyhow::anyhow!("rad run requires a <PROMPT>"))?;
        let t0 = Instant::now();
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let image_tok_id = tokenizer.token_to_id("<|content_image|>").map(|id| id as i32);
        let mut engine = InklingEngine::load(Path::new(&opts.model), opts.ctx)?;
        engine.capture_decode_graph()?;
        let mut drafter = load_drafter(&opts, &mut engine, 1)?;
        let mut vision = if opts.image.is_some() { load_vision(&opts)? } else { None };
        let spec_mode = drafter.as_ref().map(|d| d.mode_name()).unwrap_or("none");
        eprintln!("rad loaded in {:.2}s (spec={}, vision={})", t0.elapsed().as_secs_f64(), spec_mode, vision.is_some());
        let (images, prompt_content) = if let Some(ref img_path) = opts.image {
            (vec![img_path.clone()], format!("<|content_image|>{user_prompt}"))
        } else {
            (Vec::new(), user_prompt)
        };
        let formatted = format_prompt(
            &[ChatMessage { role: "user".into(), content: serde_json::Value::String(prompt_content), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
            opts.thinking,
        );
        let ids: Vec<i32> = tokenizer.encode(formatted, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
        let tp = Instant::now();
        let (mut cur_tok, _) = prefill_with_drafter(&mut engine, 0, &ids, &images, image_tok_id, drafter.as_mut(), vision.as_mut())?;
        let pp_ms = tp.elapsed().as_secs_f64() * 1000.0;
        let td = Instant::now();
        let mut emitted: Vec<u32> = Vec::new();
        let mut prev_len = 0usize;
        if let Some(ref mut dr) = drafter {
            let k_len = dr.spec_k();
            while emitted.len() < opts.max_tokens {
                if cur_tok == 200006 || cur_tok == 200008 { break; }
                let draft = dr.draft(cur_tok, &engine, k_len)?;
                let pos_before = engine.pos;
                let preds = engine.verify_batch(&draft)?;
                let mut n_acc = 1usize;
                for j in 0..(k_len - 1) {
                    if preds[j] == draft[j + 1] { n_acc += 1; } else { break; }
                }
                let mut stop = false;
                for j in 0..n_acc {
                    let t = draft[j];
                    if t == 200006 || t == 200008 { stop = true; break; }
                    emitted.push(t as u32);
                    if emitted.len() >= opts.max_tokens { stop = true; break; }
                }
                dr.inject_batch_seq(0, &engine, 0, pos_before, n_acc)?;
                let txt = tokenizer.decode(&emitted, true).unwrap_or_default();
                if txt.len() > prev_len {
                    print!("{}", &txt[prev_len..]);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                    prev_len = txt.len();
                }
                engine.rewind_verify_to(n_acc, k_len)?;
                cur_tok = preds[n_acc - 1];
                if stop { break; }
            }
        } else {
            while emitted.len() < opts.max_tokens {
                if cur_tok == 200006 || cur_tok == 200008 { break; }
                emitted.push(cur_tok as u32);
                let txt = tokenizer.decode(&emitted, true).unwrap_or_default();
                if txt.len() > prev_len {
                    print!("{}", &txt[prev_len..]);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                    prev_len = txt.len();
                }
                cur_tok = engine.step(cur_tok)?;
            }
        }
        let dec_s = td.elapsed().as_secs_f64();
        println!();
        eprintln!(
            "stats: prompt={} tok ({:.1}ms, {:.1} tok/s) | gen={} tok ({:.2}s, {:.1} tok/s)",
            ids.len(), pp_ms, ids.len() as f64 / (pp_ms / 1000.0).max(1e-6),
            emitted.len(), dec_s, emitted.len() as f64 / dec_s.max(1e-6)
        );
        return Ok(());
    }

    let t0 = Instant::now();
    let (gpu_ids, p2p_links) = resolve_gpu_ids(&opts.gpus)?;
    if opts.tp > 1 {
        anyhow::bail!(
            "use data-parallel multi-GPU serving via `--gpus 0,1` or `--gpus auto` (detected {} GPU(s), p2p_links={})",
            gpu_ids.len(),
            p2p_links
        );
    }
    let tokenizer = Arc::new(Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    let image_tok_id = tokenizer.token_to_id("<|content_image|>").map(|id| id as i32);
    std::env::set_var("RAD_KV_BLOCK_SIZE", opts.kv_block_size.to_string());
    let metrics = Arc::new(metrics::Metrics::new());
    let api_keys: Vec<String> = opts
        .api_key
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut workers: Vec<WorkerHandle> = Vec::with_capacity(gpu_ids.len());
    let mut spec_enabled = false;
    let mut spec_mode: &'static str = "none";
    let mut vision_enabled = false;
    let mut vocab_grammar: Option<Arc<grammar::VocabGrammarTable>> = None;

    for &dev_id in &gpu_ids {
        rad_hip::set_device(dev_id).map_err(|e| anyhow::anyhow!("hipSetDevice({dev_id}) failed: rc={e}"))?;
        let mut engine = InklingEngine::load_multi(Path::new(&opts.model), opts.ctx, opts.max_seqs)?;
        engine.capture_decode_graph()?;
        let drafter = load_drafter(&opts, &mut engine, opts.max_seqs.min(8))?;
        let vision = load_vision(&opts)?;
        spec_enabled = drafter.is_some();
        spec_mode = drafter.as_ref().map(|d| d.mode_name()).unwrap_or("none");
        vision_enabled = vision.is_some();
        let vg = vocab_grammar
            .get_or_insert_with(|| Arc::new(grammar::VocabGrammarTable::build(&tokenizer, engine.cfg.vocab_unpadded)))
            .clone();
        let (job_tx, job_rx) = mpsc::unbounded_channel();
        let sched_metrics = metrics.clone();
        let prefill_chunk = opts.prefill_chunk_size;
        std::thread::spawn(move || {
            let _ = rad_hip::set_device(dev_id);
            run_scheduler(engine, drafter, vision, image_tok_id, vg, sched_metrics, prefill_chunk, job_rx);
        });
        workers.push(WorkerHandle {
            dev_id,
            job_tx,
            in_flight: Arc::new(AtomicU64::new(0)),
            last_prefix_hash: Arc::new(AtomicU64::new(0)),
        });
    }

    let parallel_mode = if gpu_ids.len() > 1 { "data-parallel" } else { "single" };
    eprintln!(
        "rad serve ready in {:.2}s (gpus={:?}, mode={}, p2p={}, ctx={}, max_seqs={}, chunk={}, kv_blk={}, spec={}, vision={}, auth={}, port={})",
        t0.elapsed().as_secs_f64(),
        gpu_ids,
        parallel_mode,
        p2p_links,
        opts.ctx,
        opts.max_seqs,
        opts.prefill_chunk_size,
        opts.kv_block_size,
        spec_mode,
        vision_enabled,
        !api_keys.is_empty(),
        opts.port
    );

    let state = Arc::new(AppState {
        workers,
        gpu_ids,
        parallel_mode,
        p2p_links,
        tokenizer,
        metrics,
        api_keys,
        metrics_auth: opts.metrics_auth,
        spec_enabled,
        spec_mode,
        vision_enabled,
        mmproj_path: opts.mmproj,
        prefill_chunk_size: opts.prefill_chunk_size,
        kv_block_size: opts.kv_block_size,
        max_ctx: opts.ctx,
        port: opts.port,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(models))
        .route("/v1/models/:id", get(model_by_id))
        .route("/v1/tokenize", post(tokenize_endpoint))
        .route("/v1/detokenize", post(detokenize_endpoint))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses_http).get(responses_ws))
        .layer(middleware::from_fn(cors_and_req_id))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", opts.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
