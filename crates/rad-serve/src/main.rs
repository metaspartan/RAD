mod bench;
mod grammar;
mod handlers;
mod metrics;
mod prompt;
mod scheduler;
mod types;

use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use rad_core::{DFlash2Drafter, DsparkDrafter, GpuTopology, InklingEngine, Tokenizer, VisionEncoder};
use tokio::sync::mpsc;
use crate::bench::run_bench;
use crate::handlers::*;
use crate::prompt::format_prompt;
use crate::scheduler::{prefill_with_drafter, run_scheduler};
use crate::types::{AppState, ChatMessage, Drafter, WorkerHandle};

pub(crate) struct CliOpts {
    pub(crate) cmd: String,
    pub(crate) prompt: Option<String>,
    pub(crate) image: Option<String>,
    pub(crate) model: String,
    pub(crate) tokenizer: Option<String>,
    pub(crate) dflash2: Option<String>,
    pub(crate) dspark: Option<String>,
    pub(crate) mmproj: Option<String>,
    pub(crate) api_key: String,
    pub(crate) metrics_auth: bool,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) kv_block_size: usize,
    pub(crate) ctx: usize,
    pub(crate) max_seqs: usize,
    pub(crate) max_tokens: usize,
    pub(crate) port: u16,
    pub(crate) gpus: String,
    pub(crate) tp: usize,
    pub(crate) use_spec: bool,
    pub(crate) use_vision: bool,
    pub(crate) thinking: bool,
}

pub(crate) fn print_usage() {
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

pub(crate) fn resolve_sibling(model_path: &str, candidates: &[&str]) -> Option<String> {
    let parent = Path::new(model_path).parent()?;
    for c in candidates {
        let p = parent.join(c);
        if p.exists() {
            return Some(p.to_string_lossy().into_owned());
        }
    }
    None
}

pub(crate) fn parse_cli() -> anyhow::Result<CliOpts> {
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

pub(crate) fn load_drafter(opts: &CliOpts, engine: &mut InklingEngine, max_seqs: usize) -> anyhow::Result<Option<Drafter>> {
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

pub(crate) fn load_vision(opts: &CliOpts) -> anyhow::Result<Option<VisionEncoder>> {
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
        .route("/ping", get(ping_endpoint))
        .route("/ready", get(ping_endpoint))
        .route("/live", get(ping_endpoint))
        .route("/version", get(version_endpoint))
        .route("/props", get(props_endpoint))
        .route("/slots", get(slots_endpoint))
        .route("/metrics", get(metrics_endpoint))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/v1/models/:id", get(model_by_id))
        .route("/tokenize", post(tokenize_endpoint))
        .route("/detokenize", post(detokenize_endpoint))
        .route("/v1/tokenize", post(tokenize_endpoint))
        .route("/v1/detokenize", post(detokenize_endpoint))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses_http).get(responses_ws))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .layer(middleware::from_fn(cors_and_req_id))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", opts.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
