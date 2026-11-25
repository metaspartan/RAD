use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

static REQ_ID_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Deserialize, Default)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<Value>,
}

pub fn check_api_key(headers: &HeaderMap, keys: &[String]) -> bool {
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

pub fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": {"message": "unauthorized", "type": "authentication_error"}})),
    )
        .into_response()
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn parse_reasoning_effort(v: Option<&Value>) -> u32 {
    match v {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(50).clamp(1, 100) as u32,
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "none" | "minimal" => 10,
            "low" => 25,
            "medium" => 50,
            "high" => 85,
            "max" => 100,
            other => other.parse::<u32>().unwrap_or(50).clamp(1, 100),
        },
        _ => 50,
    }
}

pub fn extract_content_str(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            let mut s = String::new();
            for item in arr {
                if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                    s.push_str(t);
                }
            }
            s
        }
        Some(v) if !v.is_null() => v.to_string(),
        _ => String::new(),
    }
}

pub fn detect_degenerate_cycle<T: Eq>(tokens: &[T]) -> bool {
    let n = tokens.len();
    if n < 48 {
        return false;
    }
    for p in 1..=64usize {
        let min_reps = if p == 1 {
            32
        } else if p <= 4 {
            16
        } else if p <= 8 {
            10
        } else if p <= 16 {
            6
        } else {
            4
        };
        let need = p * min_reps;
        if n < need {
            continue;
        }
        let win = &tokens[n - need..];
        let pat = &win[need - p..];
        if win.chunks_exact(p).all(|c| c == pat) {
            return true;
        }
    }
    false
}

pub async fn ping_handler() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

pub async fn version_handler() -> Json<Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION"), "engine": "rad"}))
}

pub fn prometheus_response(body: String) -> Response {
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
        .into_response()
}

pub async fn cors_and_req_id(req: axum::extract::Request, next: Next) -> Response {
    let req_id = format!("rad-{}", REQ_ID_SEQ.fetch_add(1, Ordering::Relaxed));
    let mut resp = if req.method() == axum::http::Method::OPTIONS {
        StatusCode::NO_CONTENT.into_response()
    } else {
        next.run(req).await
    };
    let h = resp.headers_mut();
    h.insert("access-control-allow-origin", HeaderValue::from_static("*"));
    h.insert(
        "access-control-allow-methods",
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    h.insert(
        "access-control-allow-headers",
        HeaderValue::from_static("authorization, content-type, x-api-key, anthropic-version"),
    );
    if let Ok(v) = HeaderValue::from_str(&req_id) {
        h.insert("x-request-id", v);
    }
    resp
}

pub fn apply_common_layers<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .layer(axum::middleware::from_fn(cors_and_req_id))
}

#[derive(Default)]
pub struct RadMetrics {
    pub requests_active: AtomicI64,
    pub requests_queued: AtomicI64,
    pub requests_completed: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub generated_tokens: AtomicU64,
    pub cached_prompt_tokens: AtomicU64,
    pub prefix_cache_hits: AtomicU64,
    pub prefix_cache_queries: AtomicU64,
    pub kv_pos: AtomicU64,
    pub spec_accepted: AtomicU64,
    pub spec_drafted: AtomicU64,
    pub spec_rounds: AtomicU64,
    pub ttft_us_sum: AtomicU64,
    pub ttft_count: AtomicU64,
    pub decode_us_sum: AtomicU64,
    pub decode_count: AtomicU64,
    pub req_gen_tokens_sum: AtomicU64,
    pub req_gen_tokens_count: AtomicU64,
}

impl RadMetrics {
    pub fn render_prometheus(&self, model: &str, max_ctx: usize) -> String {
        let kv_ratio = (self.kv_pos.load(Ordering::Relaxed) as f64 / max_ctx.max(1) as f64).min(1.0);
        let ttft_s = self.ttft_us_sum.load(Ordering::Relaxed) as f64 / 1e6;
        let dec_s = self.decode_us_sum.load(Ordering::Relaxed) as f64 / 1e6;
        format!(
            "rad_requests_active{{model=\"{model}\"}} {}\n\
rad_requests_queued{{model=\"{model}\"}} {}\n\
rad_requests_completed_total{{model=\"{model}\"}} {}\n\
rad_prompt_tokens_total{{model=\"{model}\"}} {}\n\
rad_generated_tokens_total{{model=\"{model}\"}} {}\n\
rad_prefix_cache_tokens_saved_total{{model=\"{model}\"}} {}\n\
rad_prefix_cache_hits_total{{model=\"{model}\"}} {}\n\
rad_prefix_cache_queries_total{{model=\"{model}\"}} {}\n\
rad_kv_cache_usage_ratio{{model=\"{model}\"}} {:.6}\n\
rad_spec_accepted_tokens_total{{model=\"{model}\"}} {}\n\
rad_spec_drafted_tokens_total{{model=\"{model}\"}} {}\n\
rad_spec_rounds_total{{model=\"{model}\"}} {}\n\
rad_ttft_seconds_sum{{model=\"{model}\"}} {:.6}\n\
rad_ttft_seconds_count{{model=\"{model}\"}} {}\n\
rad_decode_seconds_sum{{model=\"{model}\"}} {:.6}\n\
rad_decode_seconds_count{{model=\"{model}\"}} {}\n\
rad_request_generation_tokens_sum{{model=\"{model}\"}} {}\n\
rad_request_generation_tokens_count{{model=\"{model}\"}} {}\n",
            self.requests_active.load(Ordering::Relaxed),
            self.requests_queued.load(Ordering::Relaxed),
            self.requests_completed.load(Ordering::Relaxed),
            self.prompt_tokens.load(Ordering::Relaxed),
            self.generated_tokens.load(Ordering::Relaxed),
            self.cached_prompt_tokens.load(Ordering::Relaxed),
            self.prefix_cache_hits.load(Ordering::Relaxed),
            self.prefix_cache_queries.load(Ordering::Relaxed),
            kv_ratio,
            self.spec_accepted.load(Ordering::Relaxed),
            self.spec_drafted.load(Ordering::Relaxed),
            self.spec_rounds.load(Ordering::Relaxed),
            ttft_s,
            self.ttft_count.load(Ordering::Relaxed),
            dec_s,
            self.decode_count.load(Ordering::Relaxed),
            self.req_gen_tokens_sum.load(Ordering::Relaxed),
            self.req_gen_tokens_count.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_reasoning_effort_variants() {
        assert_eq!(parse_reasoning_effort(Some(&json!("low"))), 25);
        assert_eq!(parse_reasoning_effort(Some(&json!("high"))), 85);
        assert_eq!(parse_reasoning_effort(Some(&json!(42))), 42);
        assert_eq!(parse_reasoning_effort(None), 50);
    }

    #[test]
    fn detects_short_and_long_cycles() {
        let mut v = vec![1u32, 2, 3, 4];
        for _ in 0..16 {
            v.extend_from_slice(&[10, 20, 30, 40]);
        }
        assert!(detect_degenerate_cycle(&v));
        let mut long_v = Vec::new();
        for _ in 0..4 {
            long_v.extend(100u32..149u32);
        }
        assert!(detect_degenerate_cycle(&long_v));
    }
}
