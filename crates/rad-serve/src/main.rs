use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use rad_core::{DFlash2Drafter, DsparkDrafter, InklingEngine, Tokenizer, VisionEncoder};
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
    enable_thinking: bool,
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

    fn reset_seq(&mut self, seq: usize) {
        match self {
            Drafter::DFlash2(d) => d.reset_seq(seq),
            Drafter::Dspark(d) => d.reset_seq(seq),
        }
    }

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

    fn draft(&mut self, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> anyhow::Result<Vec<i32>> {
        match self {
            Drafter::DFlash2(d) => d.draft(anchor_tok, target, k_len),
            Drafter::Dspark(d) => d.draft(anchor_tok, target, k_len),
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
    enable_thinking: bool,
    started_at: Instant,
    tx: mpsc::UnboundedSender<GenEvent>,
}

struct AppState {
    job_tx: mpsc::UnboundedSender<GenJob>,
    tokenizer: Arc<Tokenizer>,
    api_key: String,
    spec_enabled: bool,
    spec_mode: &'static str,
    vision_enabled: bool,
    mmproj_path: Option<String>,
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    engine: &'static str,
    speculative_decoding: bool,
    spec_mode: &'static str,
    vision_enabled: bool,
    mmproj_detected: bool,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
    #[serde(default)]
    tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
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
}

#[derive(Deserialize, Default)]
struct WsAuthQuery {
    #[serde(default)]
    api_key: Option<String>,
}

fn check_auth(headers: &HeaderMap, expected: &str) -> bool {
    if expected.is_empty() {
        return true;
    }
    match headers.get("authorization").and_then(|v| v.to_str().ok()) {
        Some(h) => h.strip_prefix("Bearer ").map(|t| t == expected).unwrap_or(false),
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

fn format_prompt_with_images(messages: &[ChatMessage], enable_thinking: bool) -> (String, Vec<String>) {
    let mut s = String::from("<|begin_of_text|>");
    let mut images = Vec::new();
    for m in messages {
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
                s.push_str("<|message_model|><|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            "tool" => {
                s.push_str("<|message_tool|><|content_text|>");
                s.push_str(&txt);
                s.push_str("<|end_message|>");
            }
            _ => {}
        }
    }
    if enable_thinking {
        s.push_str("<|message_model|><|content_thinking|>");
    } else {
        s.push_str("<|message_model|><|content_thinking|><|content_text|>");
    }
    (s, images)
}

fn format_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    format_prompt_with_images(messages, enable_thinking).0
}

fn format_responses_input(input: &serde_json::Value, instructions: Option<&str>) -> String {
    let mut msgs = Vec::new();
    if let Some(inst) = instructions {
        if !inst.is_empty() {
            msgs.push(ChatMessage { role: "system".into(), content: json!(inst), tool_calls: None, name: None });
        }
    }
    match input {
        serde_json::Value::String(s) => {
            msgs.push(ChatMessage { role: "user".into(), content: json!(s), tool_calls: None, name: None });
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
                let content = item.get("content").cloned().unwrap_or_else(|| json!(""));
                msgs.push(ChatMessage { role, content, tool_calls: None, name: None });
            }
        }
        _ => {
            msgs.push(ChatMessage { role: "user".into(), content: json!(""), tool_calls: None, name: None });
        }
    }
    format_prompt(&msgs, false)
}

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
                let mut last = 0i32;
                let mut i = 0usize;
                while i < rem.len() {
                    let chunk = &rem[i..(i + 1024).min(rem.len())];
                    let pos0 = engine.seq_pos[seq];
                    last = engine.prefill_tokens_seq(seq, chunk)?;
                    let (fptr, mb) = engine.batch_dspark_feats_ptr();
                    d.inject_target_feats_batch_seq(seq, fptr, mb, 0, pos0, chunk.len())?;
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

fn run_scheduler(
    mut engine: InklingEngine,
    mut drafter: Option<Drafter>,
    mut vision: Option<VisionEncoder>,
    image_tok_id: Option<i32>,
    mut rx: mpsc::UnboundedReceiver<GenJob>,
) {
    let max_seqs = engine.max_seqs;
    let mut free = vec![true; max_seqs];
    let mut active: Vec<ActiveSlot> = Vec::with_capacity(max_seqs);
    let mut pending: VecDeque<GenJob> = VecDeque::new();
    let spec_k = drafter.as_ref().map(|d| d.spec_k()).unwrap_or(4);

    loop {
        if active.is_empty() && pending.is_empty() {
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
            let tp = Instant::now();
            match prefill_with_drafter(&mut engine, seq, &job.prompt_ids, &job.images, image_tok_id, drafter.as_mut(), vision.as_mut()) {
                Ok((first_tok, cached_n)) => {
                    let pp_ms = tp.elapsed().as_secs_f64() * 1000.0;
                    let _ = job.tx.send(GenEvent::Prefill {
                        prompt_n: job.prompt_ids.len(),
                        cached_n,
                        pp_ms,
                    });
                    if job.max_tokens == 0 || first_tok == 200006 || (first_tok == 200008 && !job.enable_thinking) {
                        let _ = job.tx.send(GenEvent::Done {
                            finish_reason: if job.max_tokens == 0 { "length" } else { "stop" },
                            pred_n: 0,
                            dec_ms: 0.0,
                        });
                        free[seq] = true;
                    } else {
                        let _ = job.tx.send(GenEvent::Token(first_tok as u32));
                        if job.max_tokens == 1 {
                            let _ = job.tx.send(GenEvent::Done {
                                finish_reason: "length",
                                pred_n: 1,
                                dec_ms: 0.0,
                            });
                            free[seq] = true;
                        } else {
                            active.push(ActiveSlot {
                                seq,
                                next_tok: first_tok,
                                max_tokens: job.max_tokens,
                                emitted: 1,
                                enable_thinking: job.enable_thinking,
                                started_at: Instant::now(),
                                tx: job.tx,
                            });
                        }
                    }
                }
                Err(e) => {
                    let _ = job.tx.send(GenEvent::Error(format!("{e}")));
                    free[seq] = true;
                }
            }
        }

        if active.is_empty() {
            continue;
        }

        if active.len() == 1 && active[0].seq == 0 {
            if let Some(ref mut d) = drafter {
                if d.ctx_len() == engine.seq_pos[0] {
                    let mut slot = active.pop().unwrap();
                    engine.pos = engine.seq_pos[0];
                    let pos0 = engine.pos;
                    let res = (|| -> anyhow::Result<Vec<i32>> {
                        let draft = d.draft(slot.next_tok, &engine, spec_k)?;
                        let verified = engine.verify_batch(&draft)?;
                        let mut n_acc = 1usize;
                        for j in 0..(spec_k - 1) {
                            if verified[j] == draft[j + 1] {
                                n_acc += 1;
                            } else {
                                break;
                            }
                        }
                        engine.rewind_verify_to(n_acc, spec_k)?;
                        engine.seq_pos[0] = engine.pos;
                        d.inject_batch_seq(0, &engine, 0, pos0, n_acc)?;
                        for j in 0..n_acc {
                            engine.seq_tokens[0].push(draft[j]);
                        }
                        let mut emitted_toks = Vec::with_capacity(n_acc);
                        for j in 1..n_acc {
                            emitted_toks.push(draft[j]);
                        }
                        emitted_toks.push(verified[n_acc - 1]);
                        Ok(emitted_toks)
                    })();
                    match res {
                        Ok(toks_out) => {
                            let mut finished = false;
                            let mut last_tok = slot.next_tok;
                            for nxt in toks_out {
                                let is_stop = nxt == 200006 || (nxt == 200008 && !slot.enable_thinking);
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

        if active.len() >= 2 && active.len() <= 8 {
            if let Some(ref mut d) = drafter {
                let all_ready = active.iter().all(|a| a.seq < d.max_seqs() && d.seq_ctx_len(a.seq) == engine.seq_pos[a.seq]);
                if all_ready {
                    let multi_k = 4usize;
                    let seq_ids: Vec<i32> = active.iter().map(|a| a.seq as i32).collect();
                    let anchors: Vec<i32> = active.iter().map(|a| a.next_tok).collect();
                    let pos_before: Vec<i32> = active.iter().map(|a| engine.seq_pos[a.seq]).collect();
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
                            d.inject_batch_seq(su, &engine, u * multi_k, pos_before[u], emits[u])?;
                            for j in 0..emits[u] {
                                engine.seq_tokens[su].push(drafts[u][j]);
                            }
                        }
                        Ok(outs)
                    })();
                    match res {
                        Ok(all_outs) => {
                            let mut keep = Vec::with_capacity(active.len());
                            for (mut slot, toks_out) in active.into_iter().zip(all_outs.into_iter()) {
                                let mut finished = false;
                                let mut last_tok = slot.next_tok;
                                for nxt in toks_out {
                                    let is_stop = nxt == 200006 || (nxt == 200008 && !slot.enable_thinking);
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
        match engine.step_multi(&seq_ids, &toks) {
            Ok(nexts) => {
                let mut keep = Vec::with_capacity(active.len());
                for (mut slot, nxt) in active.into_iter().zip(nexts.into_iter()) {
                    let is_stop = nxt == 200006 || (nxt == 200008 && !slot.enable_thinking);
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
                            keep.push(slot);
                        }
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
    }
}

async fn collect_job(
    st: &AppState,
    prompt_ids: Vec<i32>,
    images: Vec<String>,
    max_tokens: usize,
    enable_thinking: bool,
) -> Result<(Vec<u32>, usize, usize, f64, f64, &'static str), String> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    st.job_tx
        .send(GenJob { prompt_ids, images, max_tokens, enable_thinking, tx })
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
            GenEvent::Error(e) => return Err(e),
        }
    }
    Ok((out_ids, prompt_n, cached_n, pp_ms, dec_ms, finish_reason))
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Health> {
    Json(Health {
        status: "ok",
        engine: "rad",
        speculative_decoding: st.spec_enabled,
        spec_mode: st.spec_mode,
        vision_enabled: st.vision_enabled,
        mmproj_detected: st.mmproj_path.is_some(),
    })
}

async fn models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&headers, &st.api_key) {
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
    if !check_auth(&headers, &st.api_key) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let model_id = req.model.unwrap_or_else(|| "inkling-2-mini".into());
    let max_tokens = req.max_tokens.unwrap_or(512).min(8192);
    let enable_thinking = req.chat_template_kwargs.and_then(|k| k.enable_thinking).unwrap_or(false);
    let stream = req.stream.unwrap_or(false);
    let (mut prompt, images) = format_prompt_with_images(&req.messages, enable_thinking);
    if let Some(ref tools) = req.tools {
        if !tools.is_empty() {
            let tools_json = serde_json::to_string(tools).unwrap_or_default();
            let tool_sys = format!("<|message_system|><|content_text|>Available tools: {tools_json}<|end_message|>");
            if let Some(rest) = prompt.strip_prefix("<|begin_of_text|>") {
                prompt = format!("<|begin_of_text|>{tool_sys}{rest}");
            }
        }
    }

    let enc = match st.tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    };
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    let created = now_unix();
    let req_id = next_id("chatcmpl");

    if stream {
        let (tx, mut rx) = mpsc::unbounded_channel();
        if st.job_tx.send(GenJob { prompt_ids, images, max_tokens, enable_thinking, tx }).is_err() {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":"scheduler closed"}))).into_response();
        }
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let tok = st.tokenizer.clone();
        tokio::spawn(async move {
            let mut acc_ids: Vec<u32> = Vec::new();
            let mut prev_len = 0usize;
            let mut sent_role = false;
            while let Some(ev) = rx.recv().await {
                match ev {
                    GenEvent::Prefill { .. } => {}
                    GenEvent::Token(t) => {
                        acc_ids.push(t);
                        let decoded = tok.decode(&acc_ids, true).unwrap_or_default();
                        if decoded.len() > prev_len {
                            let delta = decoded[prev_len..].to_string();
                            prev_len = decoded.len();
                            let d_obj = if !sent_role {
                                sent_role = true;
                                json!({"role": "assistant", "content": delta})
                            } else {
                                json!({"content": delta})
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
                    GenEvent::Done { finish_reason, .. } => {
                        let done_chunk = json!({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                        });
                        let _ = sse_tx.send(Ok(Event::default().data(done_chunk.to_string())));
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

    let (out_ids, prompt_n, cached_n, pp_ms, dec_ms, finish_reason) =
        match collect_job(&st, prompt_ids, images, max_tokens, enable_thinking).await {
            Ok(v) => v,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response(),
        };
    let pred_n = out_ids.len();
    let raw_text = st.tokenizer.decode(&out_ids, false).unwrap_or_default();
    let (reasoning, content) = split_thinking(&raw_text, enable_thinking);
    let mut msg_obj = json!({"role": "assistant", "content": content});
    if let Some(r) = reasoning {
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
    if !check_auth(&headers, &st.api_key) {
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
            collect_job(&st_c, prompt_ids, Vec::new(), max_tokens, false).await
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
    if !check_auth(&headers, &st.api_key) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let model_id = req.model.unwrap_or_else(|| "inkling-2-mini".into());
    let max_tokens = req.max_output_tokens.unwrap_or(512).min(8192);
    let stream = req.stream.unwrap_or(false);
    let prompt = format_responses_input(&req.input, req.instructions.as_deref());
    let enc = match st.tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    };
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    let resp_id = next_id("resp");
    let created = now_unix();

    if stream {
        let (tx, mut rx) = mpsc::unbounded_channel();
        if st.job_tx.send(GenJob { prompt_ids, images: Vec::new(), max_tokens, enable_thinking: false, tx }).is_err() {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":"scheduler closed"}))).into_response();
        }
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let tok = st.tokenizer.clone();
        tokio::spawn(async move {
            let mut acc_ids: Vec<u32> = Vec::new();
            let mut prev_len = 0usize;
            let _ = sse_tx.send(Ok(Event::default().event("response.created").data(
                json!({"type":"response.created","response":{"id":resp_id,"status":"in_progress","model":model_id,"created_at":created}}).to_string()
            )));
            while let Some(ev) = rx.recv().await {
                match ev {
                    GenEvent::Prefill { .. } => {}
                    GenEvent::Token(t) => {
                        acc_ids.push(t);
                        let decoded = tok.decode(&acc_ids, true).unwrap_or_default();
                        if decoded.len() > prev_len {
                            let delta = decoded[prev_len..].to_string();
                            prev_len = decoded.len();
                            let _ = sse_tx.send(Ok(Event::default().event("response.output_text.delta").data(
                                json!({"type":"response.output_text.delta","response_id":resp_id,"delta":delta}).to_string()
                            )));
                        }
                    }
                    GenEvent::Done { pred_n, dec_ms, .. } => {
                        let full = tok.decode(&acc_ids, true).unwrap_or_default();
                        let _ = sse_tx.send(Ok(Event::default().event("response.completed").data(
                            json!({
                                "type":"response.completed",
                                "response":{
                                    "id":resp_id,
                                    "status":"completed",
                                    "model":model_id,
                                    "output_text":full,
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
        match collect_job(&st, prompt_ids, Vec::new(), max_tokens, false).await {
            Ok(v) => v,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response(),
        };
    let pred_n = out_ids.len();
    let text = st.tokenizer.decode(&out_ids, true).unwrap_or_default().trim().to_string();
    Json(json!({
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model_id,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}]
        }],
        "output_text": text,
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
    let ok = check_auth(&headers, &st.api_key)
        || st.api_key.is_empty()
        || q.api_key.as_deref() == Some(st.api_key.as_str());
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
        let max_tokens = body.get("max_output_tokens").and_then(|v| v.as_u64()).unwrap_or(512).min(8192) as usize;
        let instructions = body.get("instructions").and_then(|v| v.as_str());
        let input = body.get("input").cloned().unwrap_or_else(|| json!(""));
        let prompt = format_responses_input(&input, instructions);
        let enc = match st.tokenizer.encode(prompt, false) {
            Ok(e) => e,
            Err(e) => {
                let _ = socket.send(Message::Text(json!({"type":"error","error":format!("{e}")}).to_string())).await;
                continue;
            }
        };
        let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
        let resp_id = next_id("resp");
        let (tx, mut rx) = mpsc::unbounded_channel();
        if st.job_tx.send(GenJob { prompt_ids, images: Vec::new(), max_tokens, enable_thinking: false, tx }).is_err() {
            let _ = socket.send(Message::Text(json!({"type":"error","error":"scheduler closed"}).to_string())).await;
            break;
        }
        let _ = socket.send(Message::Text(json!({
            "type": "response.created",
            "response": {"id": resp_id, "status": "in_progress", "model": model_id}
        }).to_string())).await;

        let mut acc_ids: Vec<u32> = Vec::new();
        let mut prev_len = 0usize;
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
                    acc_ids.push(t);
                    let decoded = st.tokenizer.decode(&acc_ids, true).unwrap_or_default();
                    if decoded.len() > prev_len {
                        let delta = decoded[prev_len..].to_string();
                        prev_len = decoded.len();
                        if socket.send(Message::Text(json!({
                            "type": "response.output_text.delta",
                            "response_id": resp_id,
                            "delta": delta
                        }).to_string())).await.is_err() {
                            return;
                        }
                    }
                }
                GenEvent::Done { pred_n, dec_ms, .. } => {
                    let full = st.tokenizer.decode(&acc_ids, true).unwrap_or_default().trim().to_string();
                    let _ = socket.send(Message::Text(json!({
                        "type": "response.completed",
                        "response": {
                            "id": resp_id,
                            "status": "completed",
                            "model": model_id,
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
    ctx: usize,
    max_seqs: usize,
    max_tokens: usize,
    port: u16,
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
    println!("  -p, --port <PORT>              Server port [default: 8088] (env: RAD_PORT)");
    println!("      --ctx <TOKENS>             Context length per sequence [default: 8192] (env: RAD_CTX)");
    println!("      --max-seqs <N>             Max concurrent sequences [default: 32] (env: RAD_MAX_SEQS)");
    println!("  -n, --max-tokens <N>           Max tokens for `rad run` [default: 256]");
    println!("      --api-key <KEY>            Optional Bearer API key (env: RAD_API_KEY)");
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
    let mut ctx: usize = std::env::var("RAD_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let mut max_seqs: usize = std::env::var("RAD_MAX_SEQS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let mut max_tokens: usize = 256;
    let mut port: u16 = std::env::var("RAD_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8088);
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
        ctx,
        max_seqs,
        max_tokens,
        port,
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
            engine.capture_verify_graph(8)?;
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

fn run_bench(opts: &CliOpts, tok_path: &str) -> anyhow::Result<()> {
    println!("RAD Benchmark Suite (v{})", env!("CARGO_PKG_VERSION"));
    let t0 = Instant::now();
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let max_seqs = opts.max_seqs.max(16);
    let mut engine = InklingEngine::load_multi(Path::new(&opts.model), opts.ctx.max(4096), max_seqs)?;
    engine.capture_decode_graph()?;
    let mut drafter = load_drafter(opts, &mut engine, max_seqs.min(8))?;
    let load_s = t0.elapsed().as_secs_f64();
    let spec_mode = drafter.as_ref().map(|d| d.mode_name()).unwrap_or("none");
    println!("Loaded model in {:.2}s (spec={spec_mode}, ctx={}, max_seqs={max_seqs})", load_s, opts.ctx);
    println!();

    let short_p = format_prompt(
        &[ChatMessage { role: "user".into(), content: json!("Explain GPU memory bandwidth in one paragraph."), tool_calls: None, name: None }],
        false,
    );
    let short_ids: Vec<i32> = tokenizer.encode(short_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
    engine.reset_seq(0)?;
    let _ = engine.prefill_tokens_seq(0, &short_ids)?;
    engine.reset_seq(0)?;
    let ts = Instant::now();
    let mut cur_tok = engine.prefill_tokens_seq(0, &short_ids)?;
    let ttft_ms = ts.elapsed().as_secs_f64() * 1000.0;

    let td = Instant::now();
    let dec_n = 128usize;
    for _ in 0..dec_n {
        cur_tok = engine.step(cur_tok)?;
    }
    let b1_nospec_tps = dec_n as f64 / td.elapsed().as_secs_f64();

    let long_chunk = "Modern GPU architectures execute thousands of lightweight threads in parallel across SIMD wavefronts while hiding global HBM latency with large register files, shared local data store memory, and multi-level caches. ";
    let mut long_text = String::from("Summarize the following technical notes:\n");
    for _ in 0..45 {
        long_text.push_str(long_chunk);
    }
    let long_p = format_prompt(
        &[ChatMessage { role: "user".into(), content: json!(long_text), tool_calls: None, name: None }],
        false,
    );
    let long_ids: Vec<i32> = tokenizer.encode(long_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();

    engine.reset_seq(0)?;
    let tpp = Instant::now();
    let _ = engine.prefill_tokens_seq(0, &long_ids)?;
    let pp_nospec_tps = long_ids.len() as f64 / tpp.elapsed().as_secs_f64();

    let mut pp_spec_tps = 0.0f64;
    if let Some(ref mut d) = drafter {
        engine.reset_seq(0)?;
        d.reset_seq(0);
        let tps = Instant::now();
        let _ = prefill_with_drafter(&mut engine, 0, &long_ids, &[], None, Some(d), None)?;
        pp_spec_tps = long_ids.len() as f64 / tps.elapsed().as_secs_f64();
    }

    println!("| Benchmark | Result | Details |");
    println!("| --- | --- | --- |");
    println!("| Cold Load | {:.2} s | spec={spec_mode}, hipGraph captured |", load_s);
    println!("| Short Prompt TTFT | {:.1} ms | {} tokens |", ttft_ms, short_ids.len());
    println!("| Long Prefill (No-Spec) | {:.1} tok/s | {} tokens |", pp_nospec_tps, long_ids.len());
    if pp_spec_tps > 0.0 {
        println!("| Long Prefill ({spec_mode}) | {:.1} tok/s | {} tokens (batched feature inject) |", pp_spec_tps, long_ids.len());
    }
    println!("| Single-User Decode (No-Spec) | {:.1} tok/s | B=1 hipGraph ({dec_n} tok) |", b1_nospec_tps);

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
            let p = format_prompt(&[ChatMessage { role: "user".into(), content: json!(text), tool_calls: None, name: None }], false);
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
                d.inject_batch_seq(0, &engine, 0, pos0, n_acc)?;
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

        for &b in &[4usize, 8usize] {
            if b <= d.max_seqs() {
                let mk = 4usize;
                let code_p = format_prompt(
                    &[ChatMessage { role: "user".into(), content: json!("Show me a Python implementation of quicksort and mergesort."), tool_calls: None, name: None }],
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
                        d.inject_batch_seq(u, &engine, u * mk, pos0[u], emits[u])?;
                    }
                }
                let agg = tot as f64 / t1.elapsed().as_secs_f64();
                println!("| Multi-User B={b} ({spec_mode} K={mk}) | {:.1} agg tok/s | {:.1} tok/s/user |", agg, agg / b as f64);
            }
        }
    }

    for &b in &[8usize, 16usize] {
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
            println!("| Multi-User B={b} (No-Spec) | {:.1} agg tok/s | {:.1} tok/s/user |", agg, agg / b as f64);
        }
    }

    Ok(())
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
        println!("  endpoints: /health, /v1/models, /v1/chat/completions, /v1/completions, /v1/responses (HTTP/SSE/WS)");
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
            &[ChatMessage { role: "user".into(), content: serde_json::Value::String(prompt_content), tool_calls: None, name: None }],
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
    let tokenizer = Arc::new(Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    let image_tok_id = tokenizer.token_to_id("<|content_image|>").map(|id| id as i32);
    let mut engine = InklingEngine::load_multi(Path::new(&opts.model), opts.ctx, opts.max_seqs)?;
    engine.capture_decode_graph()?;
    let drafter = load_drafter(&opts, &mut engine, opts.max_seqs.min(8))?;
    let vision = load_vision(&opts)?;
    let spec_enabled = drafter.is_some();
    let spec_mode = drafter.as_ref().map(|d| d.mode_name()).unwrap_or("none");
    let vision_enabled = vision.is_some();
    eprintln!(
        "rad serve ready in {:.2}s (ctx={}, max_seqs={}, spec={}, vision={}, port={})",
        t0.elapsed().as_secs_f64(),
        opts.ctx,
        opts.max_seqs,
        spec_mode,
        vision_enabled,
        opts.port
    );

    let (job_tx, job_rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || run_scheduler(engine, drafter, vision, image_tok_id, job_rx));

    let state = Arc::new(AppState {
        job_tx,
        tokenizer,
        api_key: opts.api_key,
        spec_enabled,
        spec_mode,
        vision_enabled,
        mmproj_path: opts.mmproj,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses_http).get(responses_ws))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", opts.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
