use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use crate::grammar;
use crate::prompt::{convert_anthropic_request, detect_repeated_tool_loop, format_prompt_with_tools, format_responses_input_with_tools};
use crate::scheduler::collect_job;
use crate::types::*;

pub(crate) async fn health(State(st): State<Arc<AppState>>) -> Json<Health> {
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

pub(crate) async fn version_endpoint() -> Json<serde_json::Value> {
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "rad"
    }))
}

pub(crate) async fn ping_endpoint() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

pub(crate) async fn props_endpoint(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    Json(json!({
        "default_generation_settings": {
            "n_ctx": st.max_ctx,
            "model": "inkling-2-mini"
        },
        "total_slots": st.workers.len(),
        "speculative_decoding": st.spec_enabled,
        "spec_mode": st.spec_mode,
        "vision_enabled": st.vision_enabled,
        "kv_block_size": st.kv_block_size,
        "prefill_chunk_size": st.prefill_chunk_size,
        "version": env!("CARGO_PKG_VERSION")
    }))
    .into_response()
}

pub(crate) async fn slots_endpoint(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    Json(json!({
        "active": st.metrics.requests_active.load(Ordering::Relaxed),
        "prefilling": st.metrics.requests_prefilling.load(Ordering::Relaxed),
        "queued": st.metrics.requests_queued.load(Ordering::Relaxed),
        "kv_blocks_used": st.metrics.kv_blocks_used.load(Ordering::Relaxed),
        "kv_blocks_total": st.metrics.kv_blocks_total.load(Ordering::Relaxed),
        "kv_blocks_shared": st.metrics.kv_blocks_shared.load(Ordering::Relaxed)
    }))
    .into_response()
}

pub(crate) async fn tokenize_endpoint(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<TokenizeRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let text = req.prompt.or(req.text).or(req.content).unwrap_or_default();
    match st.tokenizer.encode(text, false) {
        Ok(enc) => {
            let ids = enc.get_ids();
            Json(json!({"tokens": ids, "count": ids.len(), "max_model_len": st.max_ctx})).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    }
}

pub(crate) async fn detokenize_endpoint(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<DetokenizeRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    match st.tokenizer.decode(&req.tokens, false) {
        Ok(text) => Json(json!({"text": text, "content": text})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    }
}

pub(crate) async fn metrics_endpoint(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
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

pub(crate) async fn cors_and_req_id(req: Request, next: Next) -> Response {
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

pub(crate) async fn model_by_id(State(st): State<Arc<AppState>>, headers: HeaderMap, AxumPath(id): AxumPath<String>) -> Response {
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

pub(crate) async fn models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
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

pub(crate) async fn chat_completions(
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
    let mut sampling = grammar::SamplingParams {
        temperature: req.temperature.unwrap_or(0.0).max(0.0),
        top_p: req.top_p.unwrap_or(0.95).clamp(0.0, 1.0),
        top_k: req.top_k.unwrap_or(0),
        min_p: req.min_p.unwrap_or(0.0).clamp(0.0, 1.0),
        repetition_penalty: eff_rep,
        seed: base_seed,
    };
    if loop_detected {
        sampling.temperature = sampling.temperature.max(0.5);
        sampling.repetition_penalty = sampling.repetition_penalty.max(1.15);
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
            let mut text_acc = String::new();
            let mut pred_n_sse = 0usize;
            while let Some(ev) = rx.recv().await {
                if sse_tx.is_closed() {
                    break;
                }
                match ev {
                    GenEvent::Prefill { prompt_n, cached_n, .. } => {
                        prompt_n_sse = prompt_n;
                        cached_n_sse = cached_n;
                    }
                    GenEvent::Token(t) => {
                        pred_n_sse += 1;
                        let deltas = parser.push_token(t, &tok);
                        let mut hit_stop = false;
                        for d in deltas {
                            let d_obj = match d {
                                grammar::StreamDelta::Text(mut s) => {
                                    if !stop_seqs.is_empty() {
                                        let prev_len = text_acc.len();
                                        text_acc.push_str(&s);
                                        if let Some(pos) = stop_seqs.iter().filter(|st| !st.is_empty()).filter_map(|st| text_acc.find(st.as_str())).min() {
                                            hit_stop = true;
                                            if pos > prev_len {
                                                s = text_acc[prev_len..pos].to_string();
                                            } else {
                                                s.clear();
                                            }
                                        }
                                    }
                                    if s.is_empty() && hit_stop {
                                        break;
                                    }
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
                            if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).is_err() {
                                return;
                            }
                            if hit_stop {
                                break;
                            }
                        }
                        if hit_stop {
                            let done_chunk = json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
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
                                        "completion_tokens": pred_n_sse,
                                        "total_tokens": prompt_n_sse + pred_n_sse,
                                        "prompt_tokens_details": {"cached_tokens": cached_n_sse}
                                    }
                                });
                                let _ = sse_tx.send(Ok(Event::default().data(usage_chunk.to_string())));
                            }
                            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                            break;
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

pub(crate) async fn completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"unauthorized"}))).into_response();
    }
    let model_id = req.model.unwrap_or_else(|| "inkling-2-mini".into());
    let max_tokens = req.max_tokens.unwrap_or(256).min(16384);
    let stream = req.stream.unwrap_or(false);
    let include_usage = req.stream_options.and_then(|s| s.include_usage).unwrap_or(false);
    let stop_seqs = req.stop.map(|s| s.into_vec()).unwrap_or_default();
    let mut sampling = grammar::SamplingParams::greedy();
    if let Some(t) = req.temperature { sampling.temperature = t.max(0.0); }
    if let Some(p) = req.top_p { sampling.top_p = p.clamp(0.0, 1.0); }
    if let Some(k) = req.top_k { sampling.top_k = k; }
    if let Some(mp) = req.min_p { sampling.min_p = mp.clamp(0.0, 1.0); }
    let prompts: Vec<String> = match req.prompt {
        PromptInput::Single(s) => vec![s],
        PromptInput::Batch(v) => v,
    };
    let created = now_unix();
    let req_id = next_id("cmpl");

    if stream && prompts.len() == 1 {
        let enc = match st.tokenizer.encode(prompts[0].clone(), false) {
            Ok(e) => e,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
        };
        let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
        if prompt_ids.len() >= st.max_ctx {
            return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("prompt has {} tokens which exceeds max_ctx {}", prompt_ids.len(), st.max_ctx)}))).into_response();
        }
        let max_tokens = max_tokens.min(st.max_ctx.saturating_sub(prompt_ids.len() + 1).max(1));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let in_flight = match st.dispatch(GenJob { prompt_ids, images: Vec::new(), max_tokens, enable_thinking: false, json_grammar: false, sampling, tx }) {
            Ok(g) => InFlightGuard(g),
            Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":"scheduler closed"}))).into_response(),
        };
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let tok = st.tokenizer.clone();
        tokio::spawn(async move {
            let _guard = in_flight;
            let mut prompt_n = 0usize;
            let mut cached_n = 0usize;
            let mut out_ids: Vec<u32> = Vec::new();
            let mut prev_str = String::new();
            while let Some(ev) = rx.recv().await {
                if sse_tx.is_closed() {
                    break;
                }
                match ev {
                    GenEvent::Prefill { prompt_n: pn, cached_n: cn, .. } => {
                        prompt_n = pn;
                        cached_n = cn;
                    }
                    GenEvent::Token(t) => {
                        out_ids.push(t);
                        let cur = tok.decode(&out_ids, true).unwrap_or_default();
                        if cur.len() > prev_str.len() && cur.is_char_boundary(prev_str.len()) {
                            let delta = &cur[prev_str.len()..];
                            if !stop_seqs.is_empty() && stop_seqs.iter().any(|s| cur.contains(s.as_str())) {
                                let _ = sse_tx.send(Ok(Event::default().data(json!({
                                    "id": req_id, "object": "text_completion", "created": created, "model": model_id,
                                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]
                                }).to_string())));
                                let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                                break;
                            }
                            if sse_tx.send(Ok(Event::default().data(json!({
                                "id": req_id, "object": "text_completion", "created": created, "model": model_id,
                                "choices": [{"index": 0, "text": delta, "finish_reason": serde_json::Value::Null}]
                            }).to_string()))).is_err() {
                                break;
                            }
                            prev_str = cur;
                        }
                    }
                    GenEvent::Done { finish_reason, pred_n, .. } => {
                        let mut chunk = json!({
                            "id": req_id, "object": "text_completion", "created": created, "model": model_id,
                            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}]
                        });
                        if include_usage {
                            chunk["usage"] = json!({
                                "prompt_tokens": prompt_n,
                                "completion_tokens": pred_n,
                                "total_tokens": prompt_n + pred_n,
                                "prompt_tokens_details": {"cached_tokens": cached_n}
                            });
                        }
                        let _ = sse_tx.send(Ok(Event::default().data(chunk.to_string())));
                        let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                        break;
                    }
                    GenEvent::Error(e) => {
                        let _ = sse_tx.send(Ok(Event::default().data(json!({"error": e}).to_string())));
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

    let mut handles = Vec::with_capacity(prompts.len());
    for pr in prompts {
        let enc = match st.tokenizer.encode(pr, false) {
            Ok(e) => e,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
        };
        let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
        let st_c = st.clone();
        handles.push(tokio::spawn(async move {
            collect_job(&st_c, prompt_ids, Vec::new(), max_tokens, false, false, sampling).await
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


pub(crate) async fn anthropic_count_tokens(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"type":"error","error":{"type":"authentication_error","message":"unauthorized"}}))).into_response();
    }
    let (msgs, oai_tools, enable_thinking) = convert_anthropic_request(&req);
    let (prompt, _, _) = format_prompt_with_tools(&msgs, oai_tools.as_deref(), enable_thinking);
    match st.tokenizer.encode(prompt, false) {
        Ok(enc) => Json(json!({"input_tokens": enc.get_ids().len()})).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"type":"error","error":{"type":"invalid_request_error","message":format!("{e}")}}))).into_response(),
    }
}

pub(crate) async fn anthropic_messages(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Response {
    if !check_auth_keys(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"type":"error","error":{"type":"authentication_error","message":"unauthorized"}}))).into_response();
    }
    let model_id = req.model.clone().unwrap_or_else(|| "inkling-2-mini".into());
    let max_tokens = req.max_tokens.unwrap_or(4096).min(16384);
    let stream = req.stream.unwrap_or(false);
    let stop_seqs = req.stop_sequences.clone().unwrap_or_default();
    let mut sampling = grammar::SamplingParams::greedy();
    if let Some(t) = req.temperature { sampling.temperature = t.max(0.0); }
    if let Some(p) = req.top_p { sampling.top_p = p.clamp(0.0, 1.0); }
    if let Some(k) = req.top_k { sampling.top_k = k; }
    if let Some(mp) = req.min_p { sampling.min_p = mp.clamp(0.0, 1.0); }
    let (msgs, oai_tools, enable_thinking) = convert_anthropic_request(&req);
    let (prompt, images, allowed_tools) = format_prompt_with_tools(&msgs, oai_tools.as_deref(), enable_thinking);
    let enc = match st.tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"type":"error","error":{"type":"invalid_request_error","message":format!("{e}")}}))).into_response(),
    };
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if prompt_ids.len() >= st.max_ctx {
        return (StatusCode::BAD_REQUEST, Json(json!({"type":"error","error":{"type":"invalid_request_error","message":format!("prompt has {} tokens which exceeds max_ctx {}", prompt_ids.len(), st.max_ctx)}}))).into_response();
    }
    let max_tokens = max_tokens.min(st.max_ctx.saturating_sub(prompt_ids.len() + 1).max(1));
    let msg_id = next_id("msg");
    let call_prefix = next_id("toolu");

    if stream {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let in_flight = match st.dispatch(GenJob { prompt_ids, images, max_tokens, enable_thinking, json_grammar: false, sampling, tx }) {
            Ok(g) => InFlightGuard(g),
            Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"type":"error","error":{"type":"api_error","message":"scheduler closed"}}))).into_response(),
        };
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let tok = st.tokenizer.clone();
        tokio::spawn(async move {
            let _guard = in_flight;
            let mut parser = grammar::InklingStreamParser::new(enable_thinking, call_prefix, allowed_tools);
            let mut block_idx = 0usize;
            let mut text_open = false;
            let mut think_open = false;
            let mut saw_tool = false;
            let mut started_msg = false;
            let mut text_acc = String::new();
            let mut pred_n_anth = 0usize;
            while let Some(ev) = rx.recv().await {
                if sse_tx.is_closed() {
                    break;
                }
                match ev {
                    GenEvent::Prefill { prompt_n, cached_n, .. } => {
                        started_msg = true;
                        if sse_tx.send(Ok(Event::default().event("message_start").data(json!({
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model_id,
                                "content": [],
                                "stop_reason": serde_json::Value::Null,
                                "stop_sequence": serde_json::Value::Null,
                                "usage": {
                                    "input_tokens": prompt_n.saturating_sub(cached_n),
                                    "cache_read_input_tokens": cached_n,
                                    "output_tokens": 0
                                }
                            }
                        }).to_string()))).is_err() {
                            break;
                        }
                    }
                    GenEvent::Token(t) => {
                        pred_n_anth += 1;
                        if !started_msg {
                            started_msg = true;
                            if sse_tx.send(Ok(Event::default().event("message_start").data(json!({
                                "type": "message_start",
                                "message": {"id": msg_id, "type": "message", "role": "assistant", "model": model_id, "content": [], "stop_reason": serde_json::Value::Null, "usage": {"input_tokens": 0, "output_tokens": 0}}
                            }).to_string()))).is_err() {
                                break;
                            }
                        }
                        let mut hit_stop_seq: Option<String> = None;
                        for d in parser.push_token(t, &tok) {
                            match d {
                                grammar::StreamDelta::Reasoning(delta) => {
                                    if !think_open {
                                        think_open = true;
                                        if sse_tx.send(Ok(Event::default().event("content_block_start").data(json!({
                                            "type": "content_block_start", "index": block_idx, "content_block": {"type": "thinking", "thinking": ""}
                                        }).to_string()))).is_err() { return; }
                                    }
                                    if sse_tx.send(Ok(Event::default().event("content_block_delta").data(json!({
                                        "type": "content_block_delta", "index": block_idx, "delta": {"type": "thinking_delta", "thinking": delta}
                                    }).to_string()))).is_err() { return; }
                                }
                                grammar::StreamDelta::Text(mut delta) => {
                                    if !stop_seqs.is_empty() {
                                        let prev_len = text_acc.len();
                                        text_acc.push_str(&delta);
                                        if let Some((pos, matched)) = stop_seqs.iter().filter(|s| !s.is_empty()).filter_map(|s| text_acc.find(s.as_str()).map(|p| (p, s.clone()))).min_by_key(|x| x.0) {
                                            hit_stop_seq = Some(matched);
                                            if pos > prev_len {
                                                delta = text_acc[prev_len..pos].to_string();
                                            } else {
                                                delta.clear();
                                            }
                                        }
                                    }
                                    if !delta.is_empty() {
                                        if think_open {
                                            let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                                            block_idx += 1;
                                            think_open = false;
                                        }
                                        if !text_open {
                                            text_open = true;
                                            if sse_tx.send(Ok(Event::default().event("content_block_start").data(json!({
                                                "type": "content_block_start", "index": block_idx, "content_block": {"type": "text", "text": ""}
                                            }).to_string()))).is_err() { return; }
                                        }
                                        if sse_tx.send(Ok(Event::default().event("content_block_delta").data(json!({
                                            "type": "content_block_delta", "index": block_idx, "delta": {"type": "text_delta", "text": delta}
                                        }).to_string()))).is_err() { return; }
                                    }
                                    if hit_stop_seq.is_some() {
                                        break;
                                    }
                                }
                                grammar::StreamDelta::ToolCall(tc) => {
                                    if think_open {
                                        let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                                        block_idx += 1;
                                        think_open = false;
                                    }
                                    if text_open {
                                        let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                                        block_idx += 1;
                                        text_open = false;
                                    }
                                    saw_tool = true;
                                    let _ = sse_tx.send(Ok(Event::default().event("content_block_start").data(json!({
                                        "type": "content_block_start", "index": block_idx,
                                        "content_block": {"type": "tool_use", "id": tc.id, "name": tc.name, "input": {}}
                                    }).to_string())));
                                    let _ = sse_tx.send(Ok(Event::default().event("content_block_delta").data(json!({
                                        "type": "content_block_delta", "index": block_idx,
                                        "delta": {"type": "input_json_delta", "partial_json": tc.arguments}
                                    }).to_string())));
                                    let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                                    block_idx += 1;
                                }
                            }
                        }
                        if let Some(matched) = hit_stop_seq {
                            if think_open || text_open {
                                let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                            }
                            let _ = sse_tx.send(Ok(Event::default().event("message_delta").data(json!({
                                "type": "message_delta",
                                "delta": {"stop_reason": "stop_sequence", "stop_sequence": matched},
                                "usage": {"output_tokens": pred_n_anth}
                            }).to_string())));
                            let _ = sse_tx.send(Ok(Event::default().event("message_stop").data(json!({"type": "message_stop"}).to_string())));
                            break;
                        }
                    }
                    GenEvent::Done { finish_reason, pred_n, .. } => {
                        for d in parser.finish(&tok) {
                            if let grammar::StreamDelta::ToolCall(tc) = d {
                                if think_open {
                                    let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                                    block_idx += 1;
                                    think_open = false;
                                }
                                if text_open {
                                    let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                                    block_idx += 1;
                                    text_open = false;
                                }
                                saw_tool = true;
                                let _ = sse_tx.send(Ok(Event::default().event("content_block_start").data(json!({
                                    "type": "content_block_start", "index": block_idx,
                                    "content_block": {"type": "tool_use", "id": tc.id, "name": tc.name, "input": {}}
                                }).to_string())));
                                let _ = sse_tx.send(Ok(Event::default().event("content_block_delta").data(json!({
                                    "type": "content_block_delta", "index": block_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": tc.arguments}
                                }).to_string())));
                                let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                                block_idx += 1;
                            }
                        }
                        if think_open || text_open {
                            let _ = sse_tx.send(Ok(Event::default().event("content_block_stop").data(json!({"type": "content_block_stop", "index": block_idx}).to_string())));
                        }
                        let stop_reason = if saw_tool { "tool_use" } else if finish_reason == "length" { "max_tokens" } else { "end_turn" };
                        let _ = sse_tx.send(Ok(Event::default().event("message_delta").data(json!({
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": serde_json::Value::Null},
                            "usage": {"output_tokens": pred_n}
                        }).to_string())));
                        let _ = sse_tx.send(Ok(Event::default().event("message_stop").data(json!({"type": "message_stop"}).to_string())));
                        break;
                    }
                    GenEvent::Error(e) => {
                        let _ = sse_tx.send(Ok(Event::default().event("error").data(json!({"type":"error","error":{"type":"api_error","message":e}}).to_string())));
                        break;
                    }
                }
            }
        });
        return Sse::new(UnboundedReceiverStream::new(sse_rx))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let (out_ids, prompt_n, cached_n, _pp_ms, _dec_ms, finish_reason) =
        match collect_job(&st, prompt_ids, images, max_tokens, enable_thinking, false, sampling).await {
            Ok(v) => v,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"type":"error","error":{"type":"api_error","message":e}}))).into_response(),
        };
    let pred_n = out_ids.len();
    let mut parsed = grammar::parse_inkling_tokens(&out_ids, &st.tokenizer, enable_thinking, &allowed_tools, &call_prefix);
    if !stop_seqs.is_empty() && parsed.tool_calls.is_empty() {
        for ss in &stop_seqs {
            if let Some(pos) = parsed.content.find(ss.as_str()) {
                parsed.content.truncate(pos);
            }
        }
    }
    let mut blocks: Vec<serde_json::Value> = Vec::new();
    if let Some(r) = parsed.reasoning {
        blocks.push(json!({"type": "thinking", "thinking": r}));
    }
    if !parsed.content.is_empty() || parsed.tool_calls.is_empty() {
        blocks.push(json!({"type": "text", "text": parsed.content}));
    }
    for tc in &parsed.tool_calls {
        let input_val: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({}));
        blocks.push(json!({"type": "tool_use", "id": tc.id, "name": tc.name, "input": input_val}));
    }
    let stop_reason = if !parsed.tool_calls.is_empty() {
        "tool_use"
    } else if finish_reason == "length" {
        "max_tokens"
    } else {
        "end_turn"
    };
    Json(json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": serde_json::Value::Null,
        "usage": {
            "input_tokens": prompt_n.saturating_sub(cached_n),
            "cache_read_input_tokens": cached_n,
            "output_tokens": pred_n
        }
    }))
    .into_response()
}

pub(crate) async fn responses_http(
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
                if sse_tx.is_closed() {
                    break;
                }
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
                                        if sse_tx.send(Ok(Event::default().event("response.output_item.added").data(
                                            json!({
                                                "type": "response.output_item.added",
                                                "output_index": out_index,
                                                "item": {"id": msg_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
                                            }).to_string()
                                        ))).is_err() { return; }
                                    }
                                    if sse_tx.send(Ok(Event::default().event("response.output_text.delta").data(
                                        json!({"type":"response.output_text.delta","response_id":resp_id,"item_id":msg_id,"output_index":out_index,"content_index":0,"delta":delta}).to_string()
                                    ))).is_err() { return; }
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

pub(crate) async fn responses_ws(
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

pub(crate) async fn handle_ws(mut socket: WebSocket, st: Arc<AppState>) {
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
