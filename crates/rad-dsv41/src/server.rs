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
static TC_COUNTER: AtomicU64 = AtomicU64::new(1);

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
    pub enable_thinking: bool,
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

fn render_tools_block(ts: &[Value]) -> String {
    let mut s = String::from(
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n",
    );
    for t in ts {
        let schema = t.get("function").unwrap_or(t);
        s.push_str(&schema.to_string());
        s.push('\n');
    }
    s.push_str("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
    s
}

pub fn format_dsv41_prompt(
    messages: &[Value],
    tools: Option<&[Value]>,
    enable_thinking: bool,
    reasoning_effort: u32,
) -> String {
    let effort = reasoning_effort.clamp(1, 100);
    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    let drop_thinking = !has_tools;

    struct NormMsg {
        role: String,
        parts: Vec<String>,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<Value>>,
    }

    let mut norm: Vec<NormMsg> = Vec::new();
    for m in messages {
        let role_raw = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let role = if role_raw == "developer" { "system" } else { role_raw };
        if role == "user" || role == "tool" {
            let part = if role == "tool" {
                format!("<tool_result>{}</tool_result>", extract_content_str(m))
            } else {
                extract_content_str(m)
            };
            if let Some(last) = norm.last_mut() {
                if last.role == "user" {
                    last.parts.push(part);
                    continue;
                }
            }
            norm.push(NormMsg {
                role: "user".to_string(),
                parts: vec![part],
                reasoning_content: None,
                tool_calls: None,
            });
        } else {
            let rc = m
                .get("reasoning_content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let tcs = m
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .filter(|a| !a.is_empty())
                .cloned();
            norm.push(NormMsg {
                role: role.to_string(),
                parts: vec![extract_content_str(m)],
                reasoning_content: rc,
                tool_calls: tcs,
            });
        }
    }

    let mut last_user: isize = -1;
    for (idx, m) in norm.iter().enumerate() {
        if m.role == "user" || (m.role == "system" && idx > 0) {
            last_user = idx as isize;
        }
    }

    let tools_tmpl = tools.filter(|t| !t.is_empty()).map(render_tools_block);
    let mut out = String::from("<｜begin▁of▁sentence｜>");
    let total = norm.len();

    for (idx, m) in norm.iter().enumerate() {
        if idx == 0 && (enable_thinking || m.role == "system") {
            out.push_str("<｜System｜>");
        }
        if idx == 0 && enable_thinking {
            out.push_str(&format!(
                "Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
            ));
        }

        match m.role.as_str() {
            "system" => {
                if idx > 0 {
                    out.push_str("<｜System｜>");
                }
                if let Some(t) = m.parts.first() {
                    out.push_str(t);
                }
                if idx == 0 {
                    if let Some(ref tt) = tools_tmpl {
                        out.push_str("\n\n");
                        out.push_str(tt);
                    }
                }
            }
            "user" => {
                if idx == 0 {
                    if let Some(ref tt) = tools_tmpl {
                        if !enable_thinking {
                            out.push_str("<｜System｜>");
                        }
                        out.push_str(tt);
                    }
                }
                out.push_str("<｜User｜>");
                out.push_str(&m.parts.join("\n\n"));
            }
            "latest_reminder" => {
                out.push_str("<｜latest_reminder｜>");
                if let Some(t) = m.parts.first() {
                    out.push_str(t);
                }
            }
            "assistant" => {
                if enable_thinking && (!drop_thinking || (idx as isize) > last_user) {
                    if let Some(ref rc) = m.reasoning_content {
                        out.push_str(rc);
                    }
                    out.push_str("</think>");
                }
                if let Some(t) = m.parts.first() {
                    out.push_str(t);
                }
                if let Some(ref tcs) = m.tool_calls {
                    out.push_str("\n\n<｜DSML｜ calls>\n");
                    for (ti, tc) in tcs.iter().enumerate() {
                        let fn_obj = tc.get("function").unwrap_or(tc);
                        let name = fn_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        out.push_str(&format!("<｜DSML｜ invoke name=\"{name}\">\n"));
                        let args_val = match fn_obj.get("arguments") {
                            Some(Value::String(s)) => serde_json::from_str::<Value>(s).unwrap_or(Value::String(s.clone())),
                            Some(v) => v.clone(),
                            None => json!({}),
                        };
                        if let Some(obj) = args_val.as_object() {
                            let n_k = obj.len();
                            for (ki, (k, v)) in obj.iter().enumerate() {
                                let (is_str, val_str) = match v {
                                    Value::String(s) => ("true", s.clone()),
                                    other => ("false", other.to_string()),
                                };
                                out.push_str(&format!("<｜DSML｜ parameter name=\"{k}\" string=\"{is_str}\">{val_str}</｜DSML｜ parameter>"));
                                if ki + 1 < n_k {
                                    out.push('\n');
                                }
                            }
                        } else if let Some(s) = args_val.as_str() {
                            out.push_str(&format!("<｜DSML｜ parameter name=\"arguments\" string=\"true\">{s}</｜DSML｜ parameter>"));
                        }
                        out.push_str("\n</｜DSML｜ invoke>");
                        if ti + 1 < tcs.len() {
                            out.push('\n');
                        }
                    }
                    out.push_str("\n</｜DSML｜ calls>");
                }
                out.push_str("<｜end▁of▁sentence｜>");
            }
            _ => {}
        }

        let is_last = idx + 1 == total;
        let next_is_asst = !is_last && matches!(norm[idx + 1].role.as_str(), "assistant" | "latest_reminder");
        if is_last || next_is_asst {
            if m.role == "user" || (m.role == "system" && idx > 0) {
                out.push_str("<｜Assistant｜>");
                if enable_thinking && (!drop_thinking || (idx as isize) >= last_user) {
                    out.push_str("<think>");
                } else {
                    out.push_str("</think>");
                }
            }
        }
    }

    if total == 0 {
        out.push_str("<｜Assistant｜>");
        if enable_thinking {
            out.push_str("<think>");
        } else {
            out.push_str("</think>");
        }
    }
    out
}

fn extract_xml_attr(attrs: &str, key: &str) -> Option<String> {
    let bytes = attrs.as_bytes();
    let kbytes = key.as_bytes();
    let mut i = 0usize;
    while i + kbytes.len() + 2 <= bytes.len() {
        if &bytes[i..i + kbytes.len()] == kbytes {
            let prev_ok = i == 0 || bytes[i - 1].is_ascii_whitespace();
            let mut j = i + kbytes.len();
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if prev_ok && j < bytes.len() && bytes[j] == b'=' {
                j += 1;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && (bytes[j] == b'"' || bytes[j] == b'\'') {
                    let quote = bytes[j];
                    let start = j + 1;
                    if let Some(rel_end) = attrs[start..].find(quote as char) {
                        return Some(attrs[start..start + rel_end].to_string());
                    }
                }
            }
        }
        i += 1;
    }
    None
}

fn find_dsml_tag(haystack: &str, tag_names: &[&str]) -> Option<(usize, usize, String)> {
    let prefix = "<｜DSML｜";
    let mut off = 0usize;
    while let Some(rel) = haystack[off..].find(prefix) {
        let pos = off + rel;
        let after_p = &haystack[pos + prefix.len()..];
        let trimmed = after_p.trim_start();
        let ws_len = after_p.len() - trimmed.len();
        for &tn in tag_names {
            if let Some(rem) = trimmed.strip_prefix(tn) {
                let next_ok = rem.is_empty()
                    || rem.starts_with('>')
                    || rem.starts_with('/')
                    || rem.chars().next().map(|c| c.is_whitespace()).unwrap_or(false);
                if next_ok {
                    let head_end = pos + prefix.len() + ws_len + tn.len();
                    return Some((pos, head_end, tn.to_string()));
                }
            }
        }
        off = pos + prefix.len();
    }
    None
}

fn find_dsml_close(haystack: &str, tag_names: &[&str]) -> Option<(usize, usize)> {
    let prefix = "</｜DSML｜";
    let mut off = 0usize;
    while let Some(rel) = haystack[off..].find(prefix) {
        let pos = off + rel;
        let after_p = &haystack[pos + prefix.len()..];
        let trimmed = after_p.trim_start();
        for &tn in tag_names {
            if let Some(rem) = trimmed.strip_prefix(tn) {
                if let Some(gt) = rem.find('>') {
                    let end = pos + prefix.len() + (after_p.len() - trimmed.len()) + tn.len() + gt + 1;
                    return Some((pos, end));
                }
            }
        }
        off = pos + prefix.len();
    }
    None
}

pub fn split_think_and_dsml(raw: &str) -> (Option<String>, String, Vec<Value>) {
    let (reasoning, rest) = if let Some(idx) = raw.find("</think>") {
        let r = raw[..idx].trim().to_string();
        let after = raw[idx + "</think>".len()..].trim_start();
        (if r.is_empty() { None } else { Some(r) }, after)
    } else {
        (None, raw)
    };
    let outer = find_dsml_tag(rest, &["calls", "function_calls"]);
    let direct_inv = if outer.is_none() {
        find_dsml_tag(rest, &["invoke", "call"])
    } else {
        None
    };
    let mut tool_calls = Vec::new();
    let content = if outer.is_some() || direct_inv.is_some() {
        let (cut_pos, block) = if let Some((s, head_end, _)) = outer {
            let after_head = &rest[head_end..];
            let body = if let Some(gt) = after_head.find('>') {
                &after_head[gt + 1..]
            } else {
                after_head
            };
            let blk = find_dsml_close(body, &["calls", "function_calls"])
                .map(|(e, _)| &body[..e])
                .unwrap_or(body);
            (s, blk)
        } else {
            let (s, _, _) = direct_inv.unwrap();
            (s, &rest[s..])
        };
        let before = rest[..cut_pos].trim().to_string();
        let mut cursor = block;
        while let Some((_, head_end, _)) = find_dsml_tag(cursor, &["invoke", "call"]) {
            let after_head = &cursor[head_end..];
            let Some(gt) = after_head.find('>') else { break };
            let attrs = &after_head[..gt];
            let Some(fn_name) = extract_xml_attr(attrs, "name") else {
                cursor = &after_head[gt + 1..];
                continue;
            };
            let inv_body_start = &after_head[gt + 1..];
            let (inv_body, next_cur) = if let Some((ic, ie)) = find_dsml_close(inv_body_start, &["invoke", "call"]) {
                (&inv_body_start[..ic], &inv_body_start[ie..])
            } else {
                (inv_body_start, "")
            };
            let mut args_map = serde_json::Map::new();
            let mut pcur = inv_body;
            while let Some((_, phead_end, _)) = find_dsml_tag(pcur, &["parameter", "param"]) {
                let pa = &pcur[phead_end..];
                let Some(pgt) = pa.find('>') else { break };
                let pattrs = &pa[..pgt];
                let Some(pname) = extract_xml_attr(pattrs, "name") else {
                    pcur = &pa[pgt + 1..];
                    continue;
                };
                let is_str = extract_xml_attr(pattrs, "string")
                    .map(|v| v != "false")
                    .unwrap_or(true);
                let val_start = &pa[pgt + 1..];
                let (vraw, np) = if let Some((pc, pe)) = find_dsml_close(val_start, &["parameter", "param"]) {
                    (&val_start[..pc], &val_start[pe..])
                } else {
                    (val_start, "")
                };
                let jval = if is_str {
                    let trimmed = vraw.trim();
                    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
                        serde_json::from_str::<String>(trimmed)
                            .map(Value::String)
                            .unwrap_or_else(|_| Value::String(vraw.to_string()))
                    } else {
                        Value::String(vraw.to_string())
                    }
                } else {
                    serde_json::from_str::<Value>(vraw.trim()).unwrap_or_else(|_| Value::String(vraw.to_string()))
                };
                args_map.insert(pname, jval);
                pcur = np;
            }
            let tc_id = TC_COUNTER.fetch_add(1, Ordering::Relaxed);
            tool_calls.push(json!({
                "id": format!("call_rad_{tc_id}"),
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": Value::Object(args_map).to_string()
                }
            }));
            cursor = next_cur;
        }
        if tool_calls.is_empty() {
            eprintln!("[rad-dsv41] unparsed DSML block: {:?}", &block[..block.len().min(300)]);
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
    if st.job_tx.send(ServeJob { prompt, max_tokens, enable_thinking, tx }).is_err() {
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
            let mut pending = String::new();
            let emit_content = |txt: &str, sent_role: &mut bool, sse_tx: &mpsc::UnboundedSender<Result<Event, Infallible>>, req_id: &str, created: u64, model_id: &str| -> bool {
                if txt.is_empty() {
                    return true;
                }
                let d_obj = if !*sent_role {
                    *sent_role = true;
                    json!({"role": "assistant", "content": txt})
                } else {
                    json!({"content": txt})
                };
                let chunk = json!({
                    "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
                });
                sse_tx.send(Ok(Event::default().data(chunk.to_string()))).is_ok()
            };
            let push_content = |s: &str, pending: &mut String, in_dsml: &mut bool, sent_role: &mut bool, sse_tx: &mpsc::UnboundedSender<Result<Event, Infallible>>, req_id: &str, created: u64, model_id: &str| -> bool {
                pending.push_str(s);
                if let Some(pos) = pending.find("<｜DSML｜") {
                    let safe = pending[..pos].to_string();
                    pending.clear();
                    *in_dsml = true;
                    return emit_content(&safe, sent_role, sse_tx, req_id, created, model_id);
                }
                let marker = "<｜DSML｜";
                let mut keep = 0usize;
                for (bi, _) in pending.char_indices() {
                    let tail = &pending[bi..];
                    if tail.len() < marker.len() && marker.starts_with(tail) {
                        keep = tail.len();
                        break;
                    }
                }
                let flush_len = pending.len() - keep;
                if flush_len > 0 {
                    let safe = pending[..flush_len].to_string();
                    pending.drain(..flush_len);
                    return emit_content(&safe, sent_role, sse_tx, req_id, created, model_id);
                }
                true
            };
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
                                    if !push_content(c_part, &mut pending, &mut in_dsml, &mut sent_role, &sse_tx, &req_id, created, &model_id) {
                                        return;
                                    }
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
                        } else {
                            if !push_content(&delta, &mut pending, &mut in_dsml, &mut sent_role, &sse_tx, &req_id, created, &model_id) {
                                return;
                            }
                        }
                    }
                    ServeEvent::Done { finish_reason, .. } => {
                        if !in_dsml && !pending.is_empty() {
                            let rem = std::mem::take(&mut pending);
                            let _ = emit_content(&rem, &mut sent_role, &sse_tx, &req_id, created, &model_id);
                        }
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
                            let d_obj = if !sent_role {
                                json!({"role": "assistant", "tool_calls": tc_deltas})
                            } else {
                                json!({"tool_calls": tc_deltas})
                            };
                            let chunk = json!({
                                "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                "choices": [{"index": 0, "delta": d_obj, "finish_reason": null}]
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
        let raw = "Let me check.\n</think>I will run it.\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"exec\">\n<｜DSML｜ parameter name=\"cmd\" string=\"true\">\"ls -la\"</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"timeout\" string=\"false\">30</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
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
