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
use rad_core::{DsparkDrafter, InklingEngine, Tokenizer};
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
    max_tokens: usize,
    enable_thinking: bool,
    tx: mpsc::UnboundedSender<GenEvent>,
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
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
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

fn msg_text(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            let mut out = String::new();
            for item in arr {
                if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
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

fn format_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    let mut s = String::from("<|begin_of_text|>");
    for m in messages {
        let txt = msg_text(&m.content);
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
    s
}

fn format_responses_input(input: &serde_json::Value, instructions: Option<&str>) -> String {
    let mut msgs = Vec::new();
    if let Some(inst) = instructions {
        if !inst.is_empty() {
            msgs.push(ChatMessage { role: "system".into(), content: json!(inst) });
        }
    }
    match input {
        serde_json::Value::String(s) => {
            msgs.push(ChatMessage { role: "user".into(), content: json!(s) });
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
                let content = item.get("content").cloned().unwrap_or_else(|| json!(""));
                msgs.push(ChatMessage { role, content });
            }
        }
        _ => {
            msgs.push(ChatMessage { role: "user".into(), content: json!("") });
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

fn run_scheduler(mut engine: InklingEngine, mut drafter: Option<DsparkDrafter>, mut rx: mpsc::UnboundedReceiver<GenJob>) {
    let max_seqs = engine.max_seqs;
    let mut free = vec![true; max_seqs];
    let mut active: Vec<ActiveSlot> = Vec::with_capacity(max_seqs);
    let mut pending: VecDeque<GenJob> = VecDeque::new();
    let spec_k = 4usize;

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
            match engine.prefill_seq_dspark(seq, &job.prompt_ids, drafter.as_mut()) {
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
                if d.ctx_len == engine.seq_pos[0] {
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
                        for j in 0..n_acc {
                            let fptr = engine.gather_batch_dspark_feats(j);
                            d.inject_target_feats(fptr, pos0 + j as i32)?;
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
    max_tokens: usize,
    enable_thinking: bool,
) -> Result<(Vec<u32>, usize, usize, f64, f64, &'static str), String> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    st.job_tx
        .send(GenJob { prompt_ids, max_tokens, enable_thinking, tx })
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

async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}

async fn models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&headers, &st.api_key) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let now = now_unix();
    Json(json!({
        "object": "list",
        "data": [
            {"id": "inkling-2-mini", "object": "model", "created": now, "owned_by": "rad"},
            {"id": "ornith-1.5-35b-a3b", "object": "model", "created": now, "owned_by": "rad"}
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
    let prompt = format_prompt(&req.messages, enable_thinking);

    let enc = match st.tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    };
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    let created = now_unix();
    let req_id = next_id("chatcmpl");

    if stream {
        let (tx, mut rx) = mpsc::unbounded_channel();
        if st.job_tx.send(GenJob { prompt_ids, max_tokens, enable_thinking, tx }).is_err() {
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
        match collect_job(&st, prompt_ids, max_tokens, enable_thinking).await {
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
            collect_job(&st_c, prompt_ids, max_tokens, false).await
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
        if st.job_tx.send(GenJob { prompt_ids, max_tokens, enable_thinking: false, tx }).is_err() {
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
        match collect_job(&st, prompt_ids, max_tokens, false).await {
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
        if st.job_tx.send(GenJob { prompt_ids, max_tokens, enable_thinking: false, tx }).is_err() {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("RAD_MODEL")
        .unwrap_or_else(|_| "./models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf".into());
    let tok_path = std::env::var("RAD_TOKENIZER")
        .unwrap_or_else(|_| "./models/tokenizer.json".into());
    let api_key = std::env::var("RAD_API_KEY").unwrap_or_default();
    let max_ctx: usize = std::env::var("RAD_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);
    let max_seqs: usize = std::env::var("RAD_MAX_SEQS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let port: u16 = std::env::var("RAD_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8088);

    let t0 = Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    let mut engine = InklingEngine::load_multi(Path::new(&model_path), max_ctx, max_seqs)?;
    engine.capture_decode_graph()?;
    let use_spec = std::env::var("RAD_SPEC").map(|v| v != "0").unwrap_or(true);
    let drafter = if use_spec {
        let dpath = std::env::var("RAD_DSPARK")
            .unwrap_or_else(|_| "./models/dspark-Inkling-2-Mini-Q8_0.gguf".into());
        if Path::new(&dpath).exists() {
            engine.capture_verify_graph(4)?;
            let mut d = DsparkDrafter::load(Path::new(&dpath), max_ctx)?;
            d.capture_draft_graph(&engine, 4)?;
            Some(d)
        } else {
            None
        }
    } else {
        None
    };
    eprintln!(
        "rad-serve ready in {:.2}s (ctx={max_ctx}, max_seqs={max_seqs}, spec={}, port={port})",
        t0.elapsed().as_secs_f64(),
        drafter.is_some()
    );

    let (job_tx, job_rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || run_scheduler(engine, drafter, job_rx));

    let state = Arc::new(AppState { job_tx, tokenizer, api_key });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses_http).get(responses_ws))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
