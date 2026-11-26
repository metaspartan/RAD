use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use axum::http::HeaderMap;
use rad_core::{DFlash2Drafter, DsparkDrafter, InklingEngine, Tokenizer};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use crate::{grammar, metrics};

pub(crate) static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

pub(crate) enum GenEvent {
    Prefill { prompt_n: usize, cached_n: usize, pp_ms: f64 },
    Token(u32),
    Done { finish_reason: &'static str, pred_n: usize, dec_ms: f64 },
    Error(String),
}

pub(crate) struct GenJob {
    pub(crate) prompt_ids: Vec<i32>,
    pub(crate) images: Vec<String>,
    pub(crate) max_tokens: usize,
    #[allow(dead_code)]
    pub(crate) enable_thinking: bool,
    pub(crate) json_grammar: bool,
    pub(crate) sampling: grammar::SamplingParams,
    pub(crate) tx: mpsc::UnboundedSender<GenEvent>,
}

pub(crate) enum Drafter {
    DFlash2(DFlash2Drafter),
    Dspark(DsparkDrafter),
}

impl Drafter {
    pub(crate) fn spec_k(&self) -> usize {
        match self {
            Drafter::DFlash2(_) => 8,
            Drafter::Dspark(_) => 6,
        }
    }

    pub(crate) fn mode_name(&self) -> &'static str {
        match self {
            Drafter::DFlash2(_) => "dflash2",
            Drafter::Dspark(_) => "dspark",
        }
    }

    #[allow(dead_code)]
    pub(crate) fn ctx_len(&self) -> i32 {
        match self {
            Drafter::DFlash2(d) => d.ctx_len,
            Drafter::Dspark(d) => d.ctx_len,
        }
    }

    pub(crate) fn seq_ctx_len(&self, seq: usize) -> i32 {
        match self {
            Drafter::DFlash2(d) => d.seq_ctx_len.get(seq).copied().unwrap_or(-1),
            Drafter::Dspark(d) => d.seq_ctx_len.get(seq).copied().unwrap_or(-1),
        }
    }

    pub(crate) fn max_seqs(&self) -> usize {
        match self {
            Drafter::DFlash2(d) => d.max_seqs,
            Drafter::Dspark(d) => d.max_seqs,
        }
    }

    pub(crate) fn sync(&self) {
        if let Drafter::DFlash2(d) = self {
            d.sync();
        }
    }

    pub(crate) fn reset_seq(&mut self, seq: usize) {
        match self {
            Drafter::DFlash2(d) => d.reset_seq(seq),
            Drafter::Dspark(d) => d.reset_seq(seq),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn inject_target_feats_seq(&mut self, seq: usize, feats_ptr: *const f32, pos: i32) -> anyhow::Result<()> {
        match self {
            Drafter::DFlash2(d) => d.inject_target_feats_seq(seq, feats_ptr, pos),
            Drafter::Dspark(d) => d.inject_target_feats_seq(seq, feats_ptr, pos),
        }
    }

    pub(crate) fn inject_batch_seq(&mut self, seq: usize, engine: &InklingEngine, start_tok: usize, pos0: i32, n_tok: usize) -> anyhow::Result<()> {
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

    pub(crate) fn inject_batch_seq_async(&mut self, seq: usize, engine: &InklingEngine, start_tok: usize, pos0: i32, n_tok: usize) -> anyhow::Result<()> {
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

    pub(crate) fn stream_opt(&self) -> Option<rad_hip::HipStream> {
        match self {
            Drafter::DFlash2(d) => Some(d.stream()),
            Drafter::Dspark(_) => None,
        }
    }

    pub(crate) fn draft_and_verify_seq(
        &mut self,
        seq: usize,
        anchor_tok: i32,
        engine: &mut InklingEngine,
        k_len: usize,
    ) -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
        match self {
            Drafter::DFlash2(d) => {
                let k = k_len.min(d.block_size);
                let s = d.stream();
                let mut draft = vec![0i32; k];
                let d_ptr = d.draft_enqueue_on(seq, anchor_tok, engine, k, s)?;
                let verified = engine.verify_batch_seq_from_dev(seq, d_ptr, k, s, &mut draft)?;
                Ok((draft, verified))
            }
            Drafter::Dspark(d) => {
                let draft = if seq == 0 {
                    d.draft(anchor_tok, engine, k_len)?
                } else {
                    d.draft_seq(seq, anchor_tok, engine, k_len)?
                };
                let verified = engine.verify_batch_seq(seq, &draft)?;
                Ok((draft, verified))
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn draft(&mut self, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> anyhow::Result<Vec<i32>> {
        match self {
            Drafter::DFlash2(d) => d.draft(anchor_tok, target, k_len),
            Drafter::Dspark(d) => d.draft(anchor_tok, target, k_len),
        }
    }

    pub(crate) fn draft_seq(&mut self, seq: usize, anchor_tok: i32, target: &InklingEngine, k_len: usize) -> anyhow::Result<Vec<i32>> {
        match self {
            Drafter::DFlash2(d) => d.draft_seq(seq, anchor_tok, target, k_len),
            Drafter::Dspark(d) => Ok(d.draft_multi(&[seq as i32], &[anchor_tok], target, k_len)?.pop().unwrap()),
        }
    }

    pub(crate) fn draft_multi(&mut self, seq_ids: &[i32], anchor_toks: &[i32], target: &InklingEngine, k_len: usize) -> anyhow::Result<Vec<Vec<i32>>> {
        match self {
            Drafter::DFlash2(d) => d.draft_multi(seq_ids, anchor_toks, target, k_len),
            Drafter::Dspark(d) => d.draft_multi(seq_ids, anchor_toks, target, k_len),
        }
    }
}

pub(crate) struct ActiveSlot {
    pub(crate) seq: usize,
    pub(crate) next_tok: i32,
    pub(crate) max_tokens: usize,
    pub(crate) emitted: usize,
    pub(crate) last_emits: usize,
    #[allow(dead_code)]
    pub(crate) enable_thinking: bool,
    pub(crate) grammar: Option<grammar::JsonGrammarState>,
    pub(crate) sampling: grammar::SamplingParams,
    pub(crate) started_at: Instant,
    pub(crate) last_tok_at: Instant,
    pub(crate) tx: mpsc::UnboundedSender<GenEvent>,
}

pub(crate) struct PrefillSlot {
    pub(crate) seq: usize,
    pub(crate) job: GenJob,
    pub(crate) offset: usize,
    pub(crate) cached_n: usize,
    pub(crate) started_at: Instant,
}

pub(crate) struct InFlightGuard(pub(crate) Arc<AtomicU64>);
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(crate) struct WorkerHandle {
    #[allow(dead_code)]
    pub(crate) dev_id: i32,
    pub(crate) job_tx: mpsc::UnboundedSender<GenJob>,
    pub(crate) in_flight: Arc<AtomicU64>,
    pub(crate) last_prefix_hash: Arc<AtomicU64>,
}

pub(crate) fn compute_prefix_hash(tokens: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in tokens.iter().take(256) {
        h ^= t as u32 as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub(crate) struct AppState {
    pub(crate) workers: Vec<WorkerHandle>,
    pub(crate) gpu_ids: Vec<i32>,
    pub(crate) parallel_mode: &'static str,
    pub(crate) p2p_links: usize,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) metrics: Arc<metrics::Metrics>,
    pub(crate) api_keys: Vec<String>,
    pub(crate) metrics_auth: bool,
    pub(crate) spec_enabled: bool,
    pub(crate) spec_mode: &'static str,
    pub(crate) vision_enabled: bool,
    pub(crate) mmproj_path: Option<String>,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) kv_block_size: usize,
    pub(crate) max_ctx: usize,
    pub(crate) port: u16,
}

impl AppState {
    pub(crate) fn dispatch(&self, job: GenJob) -> Result<Arc<AtomicU64>, ()> {
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
pub(crate) struct Health {
    pub(crate) status: &'static str,
    pub(crate) engine: &'static str,
    pub(crate) gpu_count: usize,
    pub(crate) gpu_ids: Vec<i32>,
    pub(crate) parallel_mode: &'static str,
    pub(crate) p2p_links: usize,
    pub(crate) speculative_decoding: bool,
    pub(crate) spec_mode: &'static str,
    pub(crate) vision_enabled: bool,
    pub(crate) mmproj_detected: bool,
    pub(crate) paged_kv: bool,
    pub(crate) kv_block_size: usize,
    pub(crate) kv_blocks_used: u64,
    pub(crate) kv_blocks_total: u64,
    pub(crate) chunked_prefill: bool,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) grammar_constraints: bool,
    pub(crate) metrics_enabled: bool,
    pub(crate) max_ctx: usize,
    pub(crate) port: u16,
}

#[derive(Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: serde_json::Value,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
}

#[derive(Deserialize, Default)]
pub(crate) struct ChatTemplateKwargs {
    #[serde(default)]
    pub(crate) enable_thinking: Option<bool>,
}

#[derive(Deserialize, Default)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum StopInput {
    Single(String),
    Multi(Vec<String>),
}

impl StopInput {
    pub(crate) fn into_vec(self) -> Vec<String> {
        match self {
            StopInput::Single(s) => if s.is_empty() { Vec::new() } else { vec![s] },
            StopInput::Multi(v) => v.into_iter().filter(|s| !s.is_empty()).collect(),
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct TokenizeRequest {
    #[serde(default)]
    pub(crate) prompt: Option<String>,
    #[serde(default)]
    pub(crate) text: Option<String>,
    #[serde(default)]
    pub(crate) content: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct DetokenizeRequest {
    pub(crate) tokens: Vec<u32>,
}

#[derive(Deserialize)]
pub(crate) struct ChatRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(crate) max_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stream: Option<bool>,
    #[serde(default)]
    pub(crate) stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) top_p: Option<f32>,
    #[serde(default)]
    pub(crate) top_k: Option<usize>,
    #[serde(default)]
    pub(crate) min_p: Option<f32>,
    #[serde(default)]
    pub(crate) repetition_penalty: Option<f32>,
    #[serde(default)]
    pub(crate) presence_penalty: Option<f32>,
    #[serde(default)]
    pub(crate) frequency_penalty: Option<f32>,
    #[serde(default)]
    pub(crate) seed: Option<u64>,
    #[serde(default)]
    pub(crate) stop: Option<StopInput>,
    #[serde(default)]
    pub(crate) tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum PromptInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
pub(crate) struct CompletionRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) prompt: PromptInput,
    #[serde(default)]
    pub(crate) max_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stream: Option<bool>,
    #[serde(default)]
    pub(crate) stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) top_p: Option<f32>,
    #[serde(default)]
    pub(crate) top_k: Option<usize>,
    #[serde(default)]
    pub(crate) min_p: Option<f32>,
    #[serde(default)]
    pub(crate) stop: Option<StopInput>,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicMessagesRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) max_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) system: Option<serde_json::Value>,
    pub(crate) messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub(crate) tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub(crate) stream: Option<bool>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) top_p: Option<f32>,
    #[serde(default)]
    pub(crate) top_k: Option<usize>,
    #[serde(default)]
    pub(crate) min_p: Option<f32>,
    #[serde(default)]
    pub(crate) stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) thinking: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub(crate) struct ResponsesRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) input: serde_json::Value,
    #[serde(default)]
    pub(crate) instructions: Option<String>,
    #[serde(default)]
    pub(crate) max_output_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stream: Option<bool>,
    #[serde(default)]
    pub(crate) tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) reasoning: Option<serde_json::Value>,
}

#[derive(Deserialize, Default)]
pub(crate) struct WsAuthQuery {
    #[serde(default)]
    pub(crate) api_key: Option<String>,
}

pub(crate) fn const_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

pub(crate) fn check_auth_keys(headers: &HeaderMap, keys: &[String]) -> bool {
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

pub(crate) fn check_key_str(provided: Option<&str>, keys: &[String]) -> bool {
    if keys.is_empty() {
        return true;
    }
    match provided {
        Some(tok) => keys.iter().any(|k| const_eq(tok.as_bytes(), k.as_bytes())),
        None => false,
    }
}

pub(crate) fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub(crate) fn next_id(prefix: &str) -> String {
    let n = REQ_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-rad-{}-{n}", now_unix())
}
