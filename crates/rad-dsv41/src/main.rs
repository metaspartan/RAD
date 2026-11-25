mod config;
mod cuda_ops;
mod engram;
mod engine;
mod mtp;
mod safetensors;
mod server;
mod weights;

use anyhow::Result;
use clap::Parser;
use config::DeepseekV41Config;
use engine::Dsv41Engine;
use rad_cuda::comm::Tp2Comm;
use rad_cuda::{CudaBuf, CudaFail};
use safetensors::SafetensorsIndex;
use server::{build_router, format_dsv41_prompt, RadMetrics, ServeEvent, ServeJob, ServerState};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(name = "rad-dsv41", about = "RAD CUDA DeepSeek-V4.1-Flash EXL3 Engine")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    engram_dir: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    enable_mtp: bool,
    #[arg(long, default_value_t = false)]
    bench_exl3: bool,
    #[arg(long, default_value_t = false)]
    bench_layer: bool,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 64)]
    gen_tokens: usize,
    #[arg(long, default_value_t = false)]
    serve: bool,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8888)]
    port: u16,
    #[arg(long, default_value = "deepseek-v4-flash-vision-exp")]
    name: String,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    api_key_file: Option<PathBuf>,
    #[arg(long, default_value_t = 40)]
    max_layers: usize,
    #[arg(long, default_value_t = 8192)]
    max_ctx: usize,
    #[arg(long, default_value_t = 0)]
    rank: usize,
    #[arg(long)]
    tp_addr: Option<String>,
}

fn run_bench_exl3(model_dir: &PathBuf, comm: &mut Tp2Comm) -> Result<()> {
    let cfg = DeepseekV41Config::load(model_dir)?;
    let mut idx = SafetensorsIndex::open(model_dir)?;
    let (free_b, tot_b) = rad_cuda::mem_info().map_err(CudaFail)?;
    println!(
        "rad-dsv41 rank={} world={} layers={} hidden={} experts={}x{} mtp={} dspark_blk={} vram_free={:.1}/{:.1}GB tensors={}",
        comm.rank,
        comm.world_size(),
        cfg.text_config.num_hidden_layers,
        cfg.text_config.hidden_size,
        cfg.text_config.n_routed_experts,
        cfg.text_config.num_experts_per_tok,
        cfg.text_config.num_nextn_predict_layers,
        cfg.text_config.dspark_block_size,
        free_b as f64 / 1e9,
        tot_b as f64 / 1e9,
        idx.tensor_count()
    );

    let scale = 1.0f32 / 128.0f32.sqrt();
    for prefix in ["layers.0.attn.wkv", "layers.0.attn.wq_b", "layers.0.ffn.experts.0.w1", "head"] {
        let (k, n, bits) = idx.exl3_shape_and_bits(prefix)?;
        let (_, trel) = idx.get_tensor(&format!("{prefix}.trellis"))?;
        let trel_len = trel.len();
        let mut d_trel = CudaBuf::new(trel_len).map_err(CudaFail)?;
        d_trel.copy_from(trel, None).map_err(CudaFail)?;
        let (_, suh) = idx.get_tensor(&format!("{prefix}.suh"))?;
        let mut d_suh = CudaBuf::new(suh.len()).map_err(CudaFail)?;
        d_suh.copy_from(suh, None).map_err(CudaFail)?;
        let (_, svh) = idx.get_tensor(&format!("{prefix}.svh"))?;
        let mut d_svh = CudaBuf::new(svh.len()).map_err(CudaFail)?;
        d_svh.copy_from(svh, None).map_err(CudaFail)?;

        let d_x = CudaBuf::new(k * 2 * 16).map_err(CudaFail)?;
        let d_xrot = CudaBuf::new(k * 2 * 16).map_err(CudaFail)?;
        let d_z = CudaBuf::new(n * 4 * 16).map_err(CudaFail)?;
        let d_y = CudaBuf::new(n * 2 * 16).map_err(CudaFail)?;

        for m in [1i32, 5i32, 16i32] {
            for _ in 0..10 {
                unsafe {
                    cuda_ops::rad_cuda_wht128_pre_mul_f16(d_x.ptr, d_suh.ptr, d_xrot.ptr, m, k as i32, scale, std::ptr::null_mut());
                    cuda_ops::rad_cuda_exl3_mma_f16(d_xrot.ptr, d_trel.ptr, d_svh.ptr, d_z.ptr, d_y.ptr, m, k as i32, n as i32, bits as i32, scale, 0, std::ptr::null_mut());
                }
            }
            rad_cuda::sync_device().map_err(CudaFail)?;
            let t0 = Instant::now();
            let iters = 100;
            for _ in 0..iters {
                unsafe {
                    cuda_ops::rad_cuda_wht128_pre_mul_f16(d_x.ptr, d_suh.ptr, d_xrot.ptr, m, k as i32, scale, std::ptr::null_mut());
                    cuda_ops::rad_cuda_exl3_mma_f16(d_xrot.ptr, d_trel.ptr, d_svh.ptr, d_z.ptr, d_y.ptr, m, k as i32, n as i32, bits as i32, scale, 0, std::ptr::null_mut());
                }
            }
            rad_cuda::sync_device().map_err(CudaFail)?;
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            let gb_s = (trel_len as f64) / dt / 1e9;
            println!(
                "  [{prefix:24} {bits}b K={k:5} N={n:6} M={m:2}] {:7.1} us  {:5.1} GB/s",
                dt * 1e6,
                gb_s
            );
        }
    }
    Ok(())
}

fn worker_loop(eng: &mut Dsv41Engine, comm: &mut Tp2Comm) -> Result<()> {
    let mut toks = Vec::new();
    loop {
        let mut cmd = 0u32;
        if comm.broadcast_u32(&mut cmd).is_err() {
            break;
        }
        match cmd {
            0 => eng.pos = 0,
            1 => {
                comm.broadcast_i32_vec(&mut toks)?;
                eng.step_batch_no_head(&toks, comm)?;
            }
            2 => {
                comm.broadcast_i32_vec(&mut toks)?;
                let _ = eng.step_batch(&toks, comm)?;
            }
            3 => {
                let mut pval = 0u32;
                comm.broadcast_u32(&mut pval)?;
                eng.pos = pval as usize;
                let mut crow = 0u32;
                comm.broadcast_u32(&mut crow)?;
                if let Some(ref mut mtp) = eng.mtp {
                    mtp.last_ctx_row = crow as usize;
                }
            }
            4 => {
                eng.profile_step(eng.cfg.bos_token_id, comm)?;
            }
            5 => {
                let mut a_val = 0u32;
                comm.broadcast_u32(&mut a_val)?;
                let _ = eng.draft_dspark(a_val as i32, comm)?;
            }
            99 => break,
            _ => {}
        }
    }
    Ok(())
}

fn run_bench_layer(cli: &Cli, comm: &mut Tp2Comm) -> Result<()> {
    let t_load = Instant::now();
    let mut eng = Dsv41Engine::load(
        &cli.model,
        cli.engram_dir.as_deref(),
        cli.max_ctx,
        comm,
        Some(cli.max_layers),
        cli.enable_mtp,
    )?;
    println!(
        "loaded {} layer(s) (rank={}/{}) in {:.2}s",
        eng.layers.len(),
        eng.rank,
        eng.world,
        t_load.elapsed().as_secs_f64()
    );
    if comm.rank != 0 {
        return worker_loop(&mut eng, comm);
    }
    let mut tok = eng.cfg.bos_token_id;
    for _ in 0..5 {
        let mut one = vec![tok];
        let mut cmd = 2u32;
        comm.broadcast_u32(&mut cmd)?;
        comm.broadcast_i32_vec(&mut one)?;
        tok = eng.step_decode(tok, comm)?;
    }
    let t0 = Instant::now();
    let steps = 20;
    for _ in 0..steps {
        let mut one = vec![tok];
        let mut cmd = 2u32;
        comm.broadcast_u32(&mut cmd)?;
        comm.broadcast_i32_vec(&mut one)?;
        tok = eng.step_decode(tok, comm)?;
    }
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!(
        "step_decode ({} layer + head): {:.2} ms/step ({:.1} tok/s), last_tok={}",
        eng.layers.len(),
        dt_ms,
        1000.0 / dt_ms,
        tok
    );
    let mut pcmd = 4u32;
    comm.broadcast_u32(&mut pcmd)?;
    eng.profile_step(eng.cfg.bos_token_id, comm)?;

    let test_seq = vec![0i32, 1000, 2000, 3000];
    let mut r0 = 0u32;
    comm.broadcast_u32(&mut r0)?;
    eng.pos = 0;
    let mut seq_preds = Vec::new();
    for &t in &test_seq {
        let mut one = vec![t];
        let mut c = 2u32;
        comm.broadcast_u32(&mut c)?;
        comm.broadcast_i32_vec(&mut one)?;
        seq_preds.push(eng.step_decode(t, comm)?);
    }
    let mut r1 = 0u32;
    comm.broadcast_u32(&mut r1)?;
    eng.pos = 0;
    let mut b_in = test_seq.clone();
    let mut c2 = 2u32;
    comm.broadcast_u32(&mut c2)?;
    comm.broadcast_i32_vec(&mut b_in)?;
    let batch_preds = eng.step_batch(&test_seq, comm)?;
    println!("PARITY M=1 vs M=4: seq={seq_preds:?} batch={batch_preds:?} match={}", seq_preds == batch_preds);

    let mut quit = 99u32;
    let _ = comm.broadcast_u32(&mut quit);
    Ok(())
}

fn run_prompt(cli: &Cli, prompt_text: &str, comm: &mut Tp2Comm) -> Result<()> {
    let t_load = Instant::now();
    let mut eng = Dsv41Engine::load(
        &cli.model,
        cli.engram_dir.as_deref(),
        cli.max_ctx,
        comm,
        Some(cli.max_layers),
        cli.enable_mtp,
    )?;
    println!(
        "loaded {} layer(s) (rank={}/{}) in {:.2}s",
        eng.layers.len(),
        eng.rank,
        eng.world,
        t_load.elapsed().as_secs_f64()
    );
    if comm.rank != 0 {
        return worker_loop(&mut eng, comm);
    }
    let tok = Tokenizer::from_file(cli.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
    let formatted = format_dsv41_prompt(
        &[serde_json::json!({"role": "user", "content": prompt_text})],
        None,
        true,
        50,
    );
    let enc = tok.encode(formatted, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let p_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    let t_pre = Instant::now();
    let mut next_tok = eng.prefill_tokens(&p_ids, comm)?;
    let ttft_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    let mut history = p_ids.clone();
    let mut gen: Vec<u32> = Vec::new();
    let t_dec = Instant::now();
    while gen.len() < cli.gen_tokens && next_tok != eng.cfg.eos_token_id {
        gen.push(next_tok as u32);
        history.push(next_tok);
        let accepted = eng.step_speculative(&history, next_tok, comm)?;
        let last = *accepted.last().unwrap_or(&eng.cfg.eos_token_id);
        for &a in &accepted[..accepted.len().saturating_sub(1)] {
            if a == eng.cfg.eos_token_id || gen.len() >= cli.gen_tokens {
                break;
            }
            gen.push(a as u32);
            history.push(a);
        }
        next_tok = last;
    }
    let dec_s = t_dec.elapsed().as_secs_f64();
    let text = tok.decode(&gen, false).unwrap_or_default();
    println!(
        "prefill={} toks in {:.1} ms ({:.1} tok/s) | decode={} toks in {:.2}s ({:.1} tok/s)",
        p_ids.len(),
        ttft_ms,
        p_ids.len() as f64 / (ttft_ms / 1000.0),
        gen.len(),
        dec_s,
        gen.len() as f64 / dec_s.max(1e-6)
    );
    println!("--- OUTPUT ---\n{text}\n--------------");
    let mut quit = 99u32;
    let _ = comm.broadcast_u32(&mut quit);
    Ok(())
}

fn run_serve(cli: &Cli, mut comm: Tp2Comm) -> Result<()> {
    let t_load = Instant::now();
    let world_size = comm.world_size();
    let mut eng = Dsv41Engine::load(
        &cli.model,
        cli.engram_dir.as_deref(),
        cli.max_ctx,
        &mut comm,
        Some(cli.max_layers),
        cli.enable_mtp,
    )?;
    println!(
        "rad-dsv41 serve ready in {:.2}s (rank={}/{}, layers={}, ctx={}, port={})",
        t_load.elapsed().as_secs_f64(),
        eng.rank,
        eng.world,
        eng.layers.len(),
        cli.max_ctx,
        cli.port
    );

    if comm.rank != 0 {
        return worker_loop(&mut eng, &mut comm);
    }

    let tok_path = cli.model.join("tokenizer.json");
    let tok = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
    let eos = eng.cfg.eos_token_id;

    let mut api_keys = Vec::new();
    if let Some(ref k) = cli.api_key {
        if !k.trim().is_empty() {
            api_keys.push(k.trim().to_string());
        }
    }
    if let Some(ref kf) = cli.api_key_file {
        if let Ok(s) = std::fs::read_to_string(kf) {
            if !s.trim().is_empty() {
                api_keys.push(s.trim().to_string());
            }
        }
    }

    let metrics = Arc::new(RadMetrics::default());
    let m_worker = metrics.clone();
    let (job_tx, mut job_rx) = mpsc::unbounded_channel::<ServeJob>();
    std::thread::spawn(move || {
        let mut cached_seq: Vec<i32> = Vec::new();
        while let Some(job) = job_rx.blocking_recv() {
            m_worker.requests_queued.fetch_sub(1, Ordering::Relaxed);
            if job.tx.is_closed() {
                continue;
            }
            m_worker.requests_active.fetch_add(1, Ordering::Relaxed);
            let enc = match tok.encode(job.prompt, false) {
                Ok(e) => e,
                Err(e) => {
                    m_worker.requests_active.fetch_sub(1, Ordering::Relaxed);
                    let _ = job.tx.send(ServeEvent::Error(format!("{e}")));
                    continue;
                }
            };
            let p_ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
            let prompt_n = p_ids.len();
            if prompt_n >= eng.max_ctx {
                m_worker.requests_active.fetch_sub(1, Ordering::Relaxed);
                let _ = job.tx.send(ServeEvent::Error(format!(
                    "prompt length {prompt_n} exceeds max_ctx {}",
                    eng.max_ctx
                )));
                continue;
            }
            let mut lcp = 0usize;
            let max_lcp = cached_seq.len().min(prompt_n.saturating_sub(1));
            while lcp < max_lcp && cached_seq[lcp] == p_ids[lcp] {
                lcp += 1;
            }
            m_worker.prompt_tokens.fetch_add(prompt_n as u64, Ordering::Relaxed);
            m_worker.prefix_cache_queries.fetch_add(1, Ordering::Relaxed);
            if lcp > 0 {
                m_worker.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
                m_worker.cached_prompt_tokens.fetch_add(lcp as u64, Ordering::Relaxed);
            }
            eng.pos = lcp;
            let mut pcmd = 3u32;
            let _ = comm.broadcast_u32(&mut pcmd);
            let mut pval = lcp as u32;
            let _ = comm.broadcast_u32(&mut pval);
            let mut crow = 0u32;
            let _ = comm.broadcast_u32(&mut crow);
            let t_pre = Instant::now();
            let mut next_tok = match eng.prefill_tokens(&p_ids[lcp..], &mut comm) {
                Ok(t) => t,
                Err(e) => {
                    cached_seq.clear();
                    m_worker.requests_active.fetch_sub(1, Ordering::Relaxed);
                    let _ = job.tx.send(ServeEvent::Error(format!("{e}")));
                    continue;
                }
            };
            m_worker.ttft_us_sum.fetch_add(t_pre.elapsed().as_micros() as u64, Ordering::Relaxed);
            m_worker.ttft_count.fetch_add(1, Ordering::Relaxed);
            m_worker.kv_pos.store(eng.pos as u64, Ordering::Relaxed);
            let t_dec = Instant::now();
            let mut history = p_ids;
            let mut gen_ids: Vec<u32> = Vec::new();
            let mut prev_str = String::new();
            let mut finish_reason = "length";
            while gen_ids.len() < job.max_tokens && eng.pos + 1 < eng.max_ctx {
                if next_tok == eos || job.tx.is_closed() {
                    finish_reason = "stop";
                    break;
                }
                gen_ids.push(next_tok as u32);
                m_worker.generated_tokens.fetch_add(1, Ordering::Relaxed);
                if let Ok(cur) = tok.decode(&gen_ids, true) {
                    if cur.len() > prev_str.len() && cur.is_char_boundary(prev_str.len()) {
                        let delta = cur[prev_str.len()..].to_string();
                        prev_str = cur;
                        if job.tx.send(ServeEvent::Token(delta)).is_err() {
                            finish_reason = "stop";
                            break;
                        }
                    }
                }
                let res = eng.step_speculative(&history, next_tok, &mut comm);
                history.push(next_tok);
                m_worker.kv_pos.store(eng.pos as u64, Ordering::Relaxed);
                match res {
                    Ok(accepted) => {
                        let drafted = eng.last_draft_len.saturating_sub(1);
                        if drafted > 0 {
                            let acc_draft = accepted.len().saturating_sub(1);
                            m_worker.spec_drafted.fetch_add(drafted as u64, Ordering::Relaxed);
                            m_worker.spec_accepted.fetch_add(acc_draft as u64, Ordering::Relaxed);
                            m_worker.spec_rounds.fetch_add(1, Ordering::Relaxed);
                        }
                        let last = *accepted.last().unwrap_or(&eos);
                        let mut stopped = false;
                        for &a in &accepted[..accepted.len().saturating_sub(1)] {
                            if a == eos || gen_ids.len() >= job.max_tokens || job.tx.is_closed() {
                                finish_reason = "stop";
                                stopped = true;
                                break;
                            }
                            gen_ids.push(a as u32);
                            m_worker.generated_tokens.fetch_add(1, Ordering::Relaxed);
                            history.push(a);
                            if let Ok(cur) = tok.decode(&gen_ids, true) {
                                if cur.len() > prev_str.len() && cur.is_char_boundary(prev_str.len()) {
                                    let delta = cur[prev_str.len()..].to_string();
                                    prev_str = cur;
                                    if job.tx.send(ServeEvent::Token(delta)).is_err() {
                                        finish_reason = "stop";
                                        stopped = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if stopped || rad_http::detect_degenerate_cycle(&gen_ids) {
                            finish_reason = "stop";
                            break;
                        }
                        next_tok = last;
                    }
                    Err(e) => {
                        let _ = job.tx.send(ServeEvent::Error(format!("{e}")));
                        break;
                    }
                }
            }
            if gen_ids.len() > 1 {
                m_worker.decode_us_sum.fetch_add(t_dec.elapsed().as_micros() as u64, Ordering::Relaxed);
                m_worker.decode_count.fetch_add(1, Ordering::Relaxed);
                m_worker.req_gen_tokens_sum.fetch_add(gen_ids.len() as u64, Ordering::Relaxed);
                m_worker.req_gen_tokens_count.fetch_add(1, Ordering::Relaxed);
            }
            m_worker.requests_completed.fetch_add(1, Ordering::Relaxed);
            m_worker.requests_active.fetch_sub(1, Ordering::Relaxed);
            cached_seq = history;
            let _ = job.tx.send(ServeEvent::Done {
                prompt_n,
                pred_n: gen_ids.len(),
                finish_reason,
            });
        }
    });

    let state = Arc::new(ServerState {
        model_name: cli.name.clone(),
        api_keys,
        max_ctx: cli.max_ctx,
        world_size,
        metrics,
        job_tx,
    });
    let app = build_router(state);
    let addr = format!("{}:{}", cli.host, cli.port);
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    rad_cuda::set_device(0).map_err(CudaFail)?;
    let mut comm = match cli.tp_addr {
        Some(ref addr) => Tp2Comm::connect(cli.rank, addr)?,
        None => Tp2Comm::single(),
    };
    if cli.bench_exl3 {
        run_bench_exl3(&cli.model, &mut comm)?;
    }
    if cli.bench_layer {
        run_bench_layer(&cli, &mut comm)?;
    }
    if let Some(ref p) = cli.prompt {
        run_prompt(&cli, p, &mut comm)?;
    }
    if cli.serve {
        run_serve(&cli, comm)?;
    }
    Ok(())
}
