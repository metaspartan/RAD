use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Default)]
pub struct ServerHealth {
    pub ok: bool,
    pub engine: String,
    pub model: String,
    pub max_ctx: usize,
    pub world_size: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ServerMetrics {
    pub requests_active: u64,
    pub prompt_tokens_total: u64,
    pub generated_tokens_total: u64,
    pub prefix_cache_hits: u64,
    pub prefix_cache_queries: u64,
    pub prefix_cache_tokens_saved: u64,
    pub spec_avg_emit: f64,
    pub spec_accepted_total: u64,
    pub spec_rounds_total: u64,
    pub kv_cache_ratio: f64,
}

impl ServerMetrics {
    pub fn cache_hit_pct(&self) -> f64 {
        if self.prefix_cache_queries == 0 {
            0.0
        } else {
            (self.prefix_cache_hits as f64 / self.prefix_cache_queries as f64) * 100.0
        }
    }

    pub fn avg_emit(&self) -> f64 {
        if self.spec_avg_emit > 0.0 {
            self.spec_avg_emit
        } else if self.spec_rounds_total > 0 {
            self.spec_accepted_total as f64 / self.spec_rounds_total as f64
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
pub enum StreamEvent {
    Reasoning(String),
    Content(String),
    Done(TurnStats),
    Error(String),
}

#[derive(Clone, Debug, Default)]
pub struct TurnStats {
    pub ttft_ms: f64,
    pub total_s: f64,
    pub decode_s: f64,
    pub gen_tokens: usize,
    pub tok_per_sec: f64,
    pub cache_hit_pct: f64,
    pub avg_emit: f64,
}

pub fn parse_prometheus_metrics(text: &str) -> ServerMetrics {
    let mut m = ServerMetrics::default();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(raw_key) = parts.next() else { continue };
        let Some(raw_val) = parts.next() else { continue };
        let Ok(val) = raw_val.parse::<f64>() else { continue };
        let key = raw_key.split('{').next().unwrap_or(raw_key);
        match key {
            "rad_requests_active" => m.requests_active = val as u64,
            "rad_prompt_tokens_total" => m.prompt_tokens_total = val as u64,
            "rad_generated_tokens_total" => m.generated_tokens_total = val as u64,
            "rad_prefix_cache_hits_total" => m.prefix_cache_hits = val as u64,
            "rad_prefix_cache_queries_total" => m.prefix_cache_queries = val as u64,
            "rad_prefix_cache_tokens_saved_total" => m.prefix_cache_tokens_saved = val as u64,
            "rad_spec_avg_emit" => m.spec_avg_emit = val,
            "rad_spec_accepted_tokens_total" => m.spec_accepted_total = val as u64,
            "rad_spec_rounds_total" => m.spec_rounds_total = val as u64,
            "rad_kv_cache_usage_ratio" => m.kv_cache_ratio = val,
            _ => {}
        }
    }
    m
}

pub async fn fetch_health(client: &Client, base_url: &str, api_key: Option<&str>) -> ServerHealth {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let mut req = client.get(&url).timeout(Duration::from_secs(3));
    if let Some(k) = api_key {
        if !k.is_empty() {
            req = req.bearer_auth(k);
        }
    }
    let Ok(resp) = req.send().await else {
        return ServerHealth::default();
    };
    if !resp.status().is_success() {
        return ServerHealth::default();
    }
    let Ok(v) = resp.json::<Value>().await else {
        return ServerHealth { ok: true, ..Default::default() };
    };
    ServerHealth {
        ok: true,
        engine: v.get("engine").and_then(|x| x.as_str()).unwrap_or("rad").to_string(),
        model: v.get("model").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        max_ctx: v.get("max_ctx").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
        world_size: v.get("world_size").and_then(|x| x.as_u64()).unwrap_or(1) as usize,
    }
}

pub async fn fetch_models(client: &Client, base_url: &str, api_key: Option<&str>) -> Vec<String> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let mut req = client.get(&url).timeout(Duration::from_secs(3));
    if let Some(k) = api_key {
        if !k.is_empty() {
            req = req.bearer_auth(k);
        }
    }
    let Ok(resp) = req.send().await else { return Vec::new() };
    let Ok(v) = resp.json::<Value>().await else { return Vec::new() };
    v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

pub async fn fetch_metrics(client: &Client, base_url: &str, api_key: Option<&str>) -> Option<ServerMetrics> {
    let url = format!("{}/metrics", base_url.trim_end_matches('/'));
    let mut req = client.get(&url).timeout(Duration::from_secs(2));
    if let Some(k) = api_key {
        if !k.is_empty() {
            req = req.bearer_auth(k);
        }
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let text = resp.text().await.ok()?;
    Some(parse_prometheus_metrics(&text))
}

#[derive(Clone)]
pub struct StreamParams {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub enable_thinking: bool,
    pub reasoning_effort: String,
}

pub async fn stream_chat(client: Client, params: StreamParams, tx: UnboundedSender<StreamEvent>) {
    if let Err(e) = stream_chat_inner(client, params, &tx).await {
        let _ = tx.send(StreamEvent::Error(e.to_string()));
    }
}

async fn stream_chat_inner(client: Client, params: StreamParams, tx: &UnboundedSender<StreamEvent>) -> Result<()> {
    let url = format!("{}/v1/chat/completions", params.base_url.trim_end_matches('/'));
    let m_before = fetch_metrics(&client, &params.base_url, params.api_key.as_deref()).await;
    let body = json!({
        "model": params.model,
        "messages": params.messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "stream": true,
        "chat_template_kwargs": {
            "enable_thinking": params.enable_thinking,
            "reasoning_effort": params.reasoning_effort,
        }
    });
    let mut req = client.post(&url).json(&body);
    if let Some(ref k) = params.api_key {
        if !k.is_empty() {
            req = req.bearer_auth(k);
        }
    }
    let t0 = Instant::now();
    let resp = req.send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let err_body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("HTTP {status}: {err_body}"));
    }

    let mut ttft_s: Option<f64> = None;
    let mut chunks = 0usize;
    let mut usage_completion: Option<usize> = None;
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();

    while let Some(item) = stream.next().await {
        let bytes = item?;
        buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_string();
            buf.drain(..=nl);
            if !line.starts_with("data:") {
                continue;
            }
            let payload = line[5..].trim();
            if payload == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(payload) else {
                continue;
            };
            if let Some(ct) = v.pointer("/usage/completion_tokens").and_then(|x| x.as_u64()) {
                usage_completion = Some(ct as usize);
            }
            let Some(delta) = v.pointer("/choices/0/delta") else {
                continue;
            };
            if let Some(rc) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
                if !rc.is_empty() {
                    if ttft_s.is_none() {
                        ttft_s = Some(t0.elapsed().as_secs_f64());
                    }
                    chunks += 1;
                    let _ = tx.send(StreamEvent::Reasoning(rc.to_string()));
                }
            }
            if let Some(cc) = delta.get("content").and_then(|x| x.as_str()) {
                if !cc.is_empty() {
                    if ttft_s.is_none() {
                        ttft_s = Some(t0.elapsed().as_secs_f64());
                    }
                    chunks += 1;
                    let _ = tx.send(StreamEvent::Content(cc.to_string()));
                }
            }
        }
    }

    let total_s = t0.elapsed().as_secs_f64();
    let ttft = ttft_s.unwrap_or(total_s);
    let decode_s = (total_s - ttft).max(1e-4);
    let m_after = fetch_metrics(&client, &params.base_url, params.api_key.as_deref()).await;
    let server_gen_delta = match (&m_before, &m_after) {
        (Some(b), Some(a)) if a.generated_tokens_total > b.generated_tokens_total => {
            Some((a.generated_tokens_total - b.generated_tokens_total) as usize)
        }
        _ => None,
    };
    let gen_tokens = usage_completion.or(server_gen_delta).unwrap_or(chunks).max(chunks);
    let tok_per_sec = if gen_tokens > 0 { gen_tokens as f64 / decode_s } else { 0.0 };
    let (cache_hit_pct, avg_emit) = m_after
        .as_ref()
        .map(|m| (m.cache_hit_pct(), m.avg_emit()))
        .unwrap_or((0.0, 0.0));

    let _ = tx.send(StreamEvent::Done(TurnStats {
        ttft_ms: ttft * 1000.0,
        total_s,
        decode_s,
        gen_tokens,
        tok_per_sec,
        cache_hit_pct,
        avg_emit,
    }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rad_prometheus_metrics() {
        let sample = "\
# HELP rad_requests_active Active requests
rad_requests_active{model=\"deepseek-v4-flash-vision-exp\"} 1
rad_generated_tokens_total{model=\"deepseek-v4-flash-vision-exp\"} 21898
rad_prefix_cache_hits_total{model=\"deepseek-v4-flash-vision-exp\"} 13
rad_prefix_cache_queries_total{model=\"deepseek-v4-flash-vision-exp\"} 14
rad_spec_accepted_tokens_total{model=\"deepseek-v4-flash-vision-exp\"} 20275
rad_spec_rounds_total{model=\"deepseek-v4-flash-vision-exp\"} 1818
rad_kv_cache_usage_ratio{model=\"deepseek-v4-flash-vision-exp\"} 0.0845
";
        let m = parse_prometheus_metrics(sample);
        assert_eq!(m.requests_active, 1);
        assert_eq!(m.generated_tokens_total, 21898);
        assert!((m.cache_hit_pct() - 92.857).abs() < 0.01);
        assert!((m.avg_emit() - 11.152).abs() < 0.01);
        assert!((m.kv_cache_ratio - 0.0845).abs() < 1e-4);
    }
}
