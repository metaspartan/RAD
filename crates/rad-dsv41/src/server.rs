use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
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
    pub messages: Vec<serde_json::Value>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub stream: Option<bool>,
}

fn check_auth(headers: &HeaderMap, keys: &[String]) -> bool {
    if keys.is_empty() {
        return true;
    }
    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(tok) = auth.strip_prefix("Bearer ").or_else(|| auth.strip_prefix("bearer ")) {
            if keys.iter().any(|k| k == tok.trim()) {
                return true;
            }
        }
    }
    if let Some(k) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        if keys.iter().any(|ak| ak == k.trim()) {
            return true;
        }
    }
    false
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

async fn health(State(st): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "engine": "rad-cuda",
        "model": st.model_name,
        "world_size": st.world_size,
        "max_ctx": st.max_ctx
    }))
}

async fn models(State(st): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    if !check_auth(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
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

fn format_dsv41_prompt(messages: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
        match role {
            "system" | "developer" => {
                out.push_str(content);
                out.push('\n');
            }
            "user" => {
                out.push_str("<｜User｜>");
                out.push_str(content);
            }
            "assistant" => {
                out.push_str("<｜Assistant｜>");
                out.push_str(content);
                out.push_str("<｜end▁of▁sentence｜>");
            }
            _ => {}
        }
    }
    out.push_str("<｜Assistant｜>");
    out
}

async fn chat_completions(
    State(st): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<ChatReq>,
) -> Response {
    if !check_auth(&headers, &st.api_keys) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let model_id = req.model.unwrap_or_else(|| st.model_name.clone());
    let max_tokens = req.max_tokens.or(req.max_completion_tokens).unwrap_or(2048).min(32768);
    let stream = req.stream.unwrap_or(false);
    let prompt = format_dsv41_prompt(&req.messages);
    let req_id = format!("chatcmpl-rad-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
    let created = now_unix();

    let (tx, mut rx) = mpsc::unbounded_channel();
    if st.job_tx.send(ServeJob { prompt, max_tokens, tx }).is_err() {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "engine closed"}))).into_response();
    }

    if stream {
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        tokio::spawn(async move {
            let mut sent_role = false;
            while let Some(ev) = rx.recv().await {
                if sse_tx.is_closed() {
                    break;
                }
                match ev {
                    ServeEvent::Token(delta) => {
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
                        if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).is_err() {
                            return;
                        }
                    }
                    ServeEvent::Done { finish_reason, .. } => {
                        let done = json!({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
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

    let mut content = String::new();
    let mut prompt_n = 0usize;
    let mut pred_n = 0usize;
    let mut finish_reason = "stop";
    while let Some(ev) = rx.recv().await {
        match ev {
            ServeEvent::Token(s) => content.push_str(&s),
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
    Json(json!({
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
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
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}
