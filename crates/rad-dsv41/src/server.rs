use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
pub use rad_http::RadMetrics;
use rad_http::{
    apply_common_layers, check_api_key, extract_content_str, now_unix, parse_reasoning_effort,
    ping_handler, prometheus_response, unauthorized_response, version_handler, ChatTemplateKwargs,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

pub struct ServerState {
    pub model_name: String,
    pub api_keys: Vec<String>,
    pub max_ctx: usize,
    pub world_size: usize,
    pub metrics: Arc<RadMetrics>,
    pub job_tx: mpsc::UnboundedSender<ServeJob>,
}

pub struct ServeJob {
    pub prompt: String,
    pub max_tokens: usize,
    pub tx: mpsc::UnboundedSender<ServeEvent>,
}

pub enum ServeEvent {
    Token(String),
    Done { prompt_n: usize, pred_n: usize, finish_reason: &'static str },
    Error(String),
}

#[derive(Deserialize)]
pub struct ChatReq {
    pub model: Option<String>,
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<Value>,
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

async fn health(State(st): State<Arc<ServerState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "engine": "rad-cuda",
        "model": st.model_name,
        "world_size": st.world_size,
        "max_ctx": st.max_ctx
    }))
}

async fn metrics(State(st): State<Arc<ServerState>>) -> Response {
    prometheus_response(st.metrics.render_prometheus(&st.model_name, st.max_ctx))
}

async fn models(State(st): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    if !check_api_key(&headers, &st.api_keys) {
        return unauthorized_response();
    }
    Json(json!({
        "object": "list",
        "data": [{
            "id": st.model_name,
            "object": "model",
            "created": now_unix(),
            "owned_by": "rad-cuda"
        }]
    }))
    .into_response()
}

pub fn format_dsv41_prompt(
    messages: &[Value],
    tools: Option<&[Value]>,
    enable_thinking: bool,
    reasoning_effort: u32,
) -> String {
    let effort = reasoning_effort.clamp(1, 100);
    let mut out = format!(
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    );
    let mut sys_text = String::new();
    for m in messages {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role == "system" || role == "developer" {
            let c = extract_content_str(m);
            if !c.is_empty() {
                if !sys_text.is_empty() {
                    sys_text.push('\n');
                }
                sys_text.push_str(&c);
            }
        }
    }
    if let Some(ts) = tools {
        if !ts.is_empty() {
            if !sys_text.is_empty() {
                sys_text.push_str("\n\n");
            }
            sys_text.push_str("## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\n### Available Tool Schemas\n\n");
            for t in ts {
                let schema = t.get("function").unwrap_or(t);
                sys_text.push_str(&schema.to_string());
                sys_text.push('\n');
            }
        }
    }
    if !sys_text.is_empty() {
        out.push_str(&sys_text);
    }
    let mut prev_user = false;
    for m in messages {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        match role {
            "system" | "developer" => {}
            "user" => {
                if !prev_user {
                    out.push_str("<｜User｜>");
                    prev_user = true;
                }
                out.push_str(&extract_content_str(m));
            }
            "tool" => {
                if !prev_user {
                    out.push_str("<｜User｜>");
                    prev_user = true;
                }
                out.push_str("<tool_result>");
                out.push_str(&extract_content_str(m));
                out.push_str("</tool_result>");
            }
            "assistant" => {
                prev_user = false;
                out.push_str("<｜Assistant｜>");
                if let Some(rc) = m.get("reasoning_content").and_then(|v| v.as_str()) {
                    if !rc.is_empty() {
                        out.push_str("<think>");
                        out.push_str(rc);
                        out.push_str("</think>");
                    }
                }
                out.push_str(&extract_content_str(m));
                if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                    if !tcs.is_empty() {
                        out.push_str("\n\n<｜DSML｜ calls>\n");
                        for tc in tcs {
                            let fn_obj = tc.get("function").unwrap_or(tc);
                            let name = fn_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            out.push_str(&format!("<｜DSML｜ invoke name=\"{name}\">\n"));
                            let args_val = match fn_obj.get("arguments") {
                                Some(Value::String(s)) => serde_json::from_str::<Value>(s).unwrap_or(Value::String(s.clone())),
                                Some(v) => v.clone(),
                                None => json!({}),
                            };
                            if let Some(obj) = args_val.as_object() {
                                for (k, v) in obj {
                                    let (is_str, val_str) = match v {
                                        Value::String(s) => ("true", s.clone()),
                                        other => ("false", other.to_string()),
                                    };
                                    out.push_str(&format!("<｜DSML｜ parameter name=\"{k}\" string=\"{is_str}\">{val_str}</｜DSML｜ parameter>\n"));
                                }
                            }
                            out.push_str("</｜DSML｜ invoke>\n");
                        }
                        out.push_str("</｜DSML｜ calls>");
                    }
                }
                out.push_str("<｜end▁of▁sentence｜>");
            }
            _ => {}
        }
    }
    if enable_thinking {
        out.push_str("<｜Assistant｜><think>");
    } else {
        out.push_str("<｜Assistant｜></think>");
    }
    out
}

pub fn split_think_and_dsml(raw: &str) -> (Option<String>, String, Vec<Value>) {
    let (reasoning, rest) = if let Some(idx) = raw.find("</think>") {
        let r = raw[..idx].trim().to_string();
        let after = raw[idx + "</think>".len()..].trim_start();
        (if r.is_empty() { None } else { Some(r) }, after)
    } else {
        (None, raw)
    };
    let dsml_open = "<｜DSML｜ calls>";
    let dsml_close = "</｜DSML｜ calls>";
    let mut tool_calls = Vec::new();
    let content = if let Some(s) = rest.find(dsml_open) {
        let before = rest[..s].trim().to_string();
        let body = &rest[s + dsml_open.len()..];
        let block = body.find(dsml_close).map(|e| &body[..e]).unwrap_or(body);
        let inv_open = "<｜DSML｜ invoke name=\"";
        let inv_close = "</｜DSML｜ invoke>";
        let mut cursor = block;
        let mut idx = 0usize;
        while let Some(i_start) = cursor.find(inv_open) {
            let after_name = &cursor[i_start + inv_open.len()..];
            let Some(q_end) = after_name.find("\">") else { break };
            let fn_name = after_name[..q_end].to_string();
            let inv_body_start = &after_name[q_end + 2..];
            let (inv_body, next_cur) = if let Some(ic) = inv_body_start.find(inv_close) {
                (&inv_body_start[..ic], &inv_body_start[ic + inv_close.len()..])
            } else {
                (inv_body_start, "")
            };
            let mut args_map = serde_json::Map::new();
            let p_open = "<｜DSML｜ parameter name=\"";
            let p_close = "</｜DSML｜ parameter>";
            let mut pcur = inv_body;
            while let Some(ps) = pcur.find(p_open) {
                let pa = &pcur[ps + p_open.len()..];
                let Some(pe) = pa.find('"') else { break };
                let pname = pa[..pe].to_string();
                let Some(gt) = pa.find('>') else { break };
                let attr = &pa[..gt];
                let is_str = !attr.contains("string=\"false\"");
                let val_start = &pa[gt + 1..];
                let (vraw, np) = if let Some(pc) = val_start.find(p_close) {
                    (&val_start[..pc], &val_start[pc + p_close.len()..])
                } else {
                    (val_start, "")
                };
                let jval = if is_str {
                    Value::String(vraw.to_string())
                } else {
                    serde_json::from_str::<Value>(vraw.trim()).unwrap_or_else(|_| Value::String(vraw.to_string()))
                };
                args_map.insert(pname, jval);
                pcur = np;
            }
            tool_calls.push(json!({
                "id": format!("call_rad_{idx}"),
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": Value::Object(args_map).to_string()
                }
            }));
            idx += 1;
            cursor = next_cur;
        }
        before
    } else {
        rest.to_string()
    };
    (reasoning, content, tool_calls)
}

async fn chat_completions(
    State(st): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<ChatReq>,
) -> Response {
    if !check_api_key(&headers, &st.api_keys) {
        return unauthorized_response();
    }
    let model_id = req.model.unwrap_or_else(|| st.model_name.clone());
    let max_tokens = req.max_tokens.or(req.max_completion_tokens).unwrap_or(2048).min(32768);
    let stream = req.stream.unwrap_or(false);
    let enable_thinking = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.enable_thinking)
        .or(req.enable_thinking)
        .unwrap_or(true);
    let reasoning_effort = parse_reasoning_effort(
        req.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.reasoning_effort.as_ref())
            .or(req.reasoning_effort.as_ref()),
    );
    let prompt = format_dsv41_prompt(&req.messages, req.tools.as_deref(), enable_thinking, reasoning_effort);
    let req_id = format!("chatcmpl-rad-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let created = now_unix();

    let (tx, mut rx) = mpsc::unbounded_channel();
    st.metrics.requests_queued.fetch_add(1, Ordering::Relaxed);
    if st.job_tx.send(ServeJob { prompt, max_tokens, tx }).is_err() {
        st.metrics.requests_queued.fetch_sub(1, Ordering::Relaxed);
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "engine closed"}))).into_response();
    }

    if stream {
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        tokio::spawn(async move {
            let mut sent_role = false;
            let mut in_think = enable_thinking;
            let mut raw_acc = String::new();
            let mut in_dsml = false;
            while let Some(ev) = rx.recv().await {
                if sse_tx.is_closed() {
                    break;
                }
                match ev {
                    ServeEvent::Token(delta) => {
                        raw_acc.push_str(&delta);
                        if in_dsml {
                            continue;
                        }
                        if in_think {
                            if let Some(pos) = delta.find("</think>") {
                                let r_part = &delta[..pos];
                                let c_part = &delta[pos + "</think>".len()..];
                                in_think = false;
                                if !r_part.is_empty() {
                                    let d_obj = if !sent_role {
                                        sent_role = true;
                                        json!({"role": "assistant", "reasoning_content": r_part})
                                    } else {
                                        json!({"reasoning_content": r_part})
                                    };
                                    let chunk = json!({
                                        "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                        "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
                                    });
                                    if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).is_err() { return; }
                                }
                                if !c_part.is_empty() {
                                    let d_obj = if !sent_role {
                                        sent_role = true;
                                        json!({"role": "assistant", "content": c_part})
                                    } else {
                                        json!({"content": c_part})
                                    };
                                    let chunk = json!({
                                        "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                        "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
                                    });
                                    if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).is_err() { return; }
                                }
                            } else {
                                let d_obj = if !sent_role {
                                    sent_role = true;
                                    json!({"role": "assistant", "reasoning_content": delta})
                                } else {
                                    json!({"reasoning_content": delta})
                                };
                                let chunk = json!({
                                    "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                    "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
                                });
                                if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).is_err() { return; }
                            }
                        } else if raw_acc.contains("<｜DSML｜ calls>") {
                            in_dsml = true;
                        } else {
                            let d_obj = if !sent_role {
                                sent_role = true;
                                json!({"role": "assistant", "content": delta})
                            } else {
                                json!({"content": delta})
                            };
                            let chunk = json!({
                                "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
                            });
                            if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).is_err() { return; }
                        }
                    }
                    ServeEvent::Done { finish_reason, .. } => {
                        let (_, _, tool_calls) = split_think_and_dsml(&raw_acc);
                        let mut fr = finish_reason;
                        if !tool_calls.is_empty() {
                            fr = "tool_calls";
                            let tc_deltas: Vec<Value> = tool_calls
                                .into_iter()
                                .enumerate()
                                .map(|(idx, mut tc)| {
                                    if let Some(obj) = tc.as_object_mut() {
                                        obj.insert("index".to_string(), json!(idx));
                                    }
                                    tc
                                })
                                .collect();
                            let chunk = json!({
                                "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": null}]
                            });
                            let _ = sse_tx.send(Ok(Event::default().data(chunk.to_string())));
                        }
                        let done = json!({
                            "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": fr}]
                        });
                        let _ = sse_tx.send(Ok(Event::default().data(done.to_string())));
                        let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                        break;
                    }
                    ServeEvent::Error(e) => {
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

    let mut raw = String::new();
    let mut prompt_n = 0usize;
    let mut pred_n = 0usize;
    let mut finish_reason = "stop";
    while let Some(ev) = rx.recv().await {
        match ev {
            ServeEvent::Token(s) => raw.push_str(&s),
            ServeEvent::Done { prompt_n: pn, pred_n: gn, finish_reason: fr } => {
                prompt_n = pn;
                pred_n = gn;
                finish_reason = fr;
                break;
            }
            ServeEvent::Error(e) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response();
            }
        }
    }
    let (reasoning, content, tool_calls) = split_think_and_dsml(&raw);
    if !tool_calls.is_empty() {
        finish_reason = "tool_calls";
    }
    let mut msg = json!({"role": "assistant", "content": content});
    if let Some(rc) = reasoning {
        msg["reasoning_content"] = json!(rc);
    }
    if !tool_calls.is_empty() {
        msg["tool_calls"] = Value::Array(tool_calls);
    }
    Json(json!({
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_n,
            "completion_tokens": pred_n,
            "total_tokens": prompt_n + pred_n
        }
    }))
    .into_response()
}

pub fn build_router(state: Arc<ServerState>) -> Router {
    apply_common_layers(
        Router::new()
            .route("/health", get(health))
            .route("/ping", get(ping_handler))
            .route("/ready", get(ping_handler))
            .route("/live", get(ping_handler))
            .route("/version", get(version_handler))
            .route("/metrics", get(metrics))
            .route("/models", get(models))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat_completions)),
    )
    .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_think_and_dsml_tool_calls() {
        let raw = "Let me check.\n</think>I will run it.\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"exec\">\n<｜DSML｜ parameter name=\"cmd\" string=\"true\">ls -la</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"timeout\" string=\"false\">30</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let (rc, content, tcs) = split_think_and_dsml(raw);
        assert_eq!(rc.as_deref(), Some("Let me check."));
        assert_eq!(content, "I will run it.");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["function"]["name"], "exec");
        let args: Value = serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["cmd"], "ls -la");
        assert_eq!(args["timeout"], 30);
    }
}
