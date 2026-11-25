mod config;
mod cuda_ops;
mod engine;
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
use server::{build_router, ServeEvent, ServeJob, ServerState};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(name = "rad-dsv41", about = "RAD CUDA DeepSeek-V4.1-Flash EXL3 Engine")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = false)]
    bench_exl3: bool,
    #[arg(long, default_value_t = false)]
    bench_layer: bool,
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

fn run_bench_layer(model_dir: &PathBuf, max_ctx: usize, max_layers: usize, comm: &mut Tp2Comm) -> Result<()> {
    let t_load = Instant::now();
    let mut eng = Dsv41Engine::load(model_dir, max_ctx, comm.rank, comm.world_size(), Some(max_layers))?;
    println!(
        "loaded {} layer(s) (rank={}/{}) in {:.2}s",
        eng.layers.len(),
        eng.rank,
        eng.world,
        t_load.elapsed().as_secs_f64()
    );
    let mut tok = eng.cfg.bos_token_id;
    for _ in 0..5 {
        tok = eng.step_decode(tok, comm)?;
    }
    let t0 = Instant::now();
    let steps = 20;
    for _ in 0..steps {
        tok = eng.step_decode(tok, comm)?;
    }
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!(
        "step_decode ({} layer + head): {:.2} ms/step, last_tok={}",
        eng.layers.len(),
        dt_ms,
        tok
    );
    Ok(())
}

fn run_serve(cli: &Cli, mut comm: Tp2Comm) -> Result<()> {
    let t_load = Instant::now();
    let mut eng = Dsv41Engine::load(&cli.model, cli.max_ctx, comm.rank, comm.world_size(), Some(cli.max_layers))?;
    let tok_path = cli.model.join("tokenizer.json");
    let tok = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
    let eos = eng.cfg.eos_token_id;
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
        loop {
            let mut cmd = 0u32;
            if comm.broadcast_u32(&mut cmd).is_err() {
                break;
            }
            if cmd == 0 {
                eng.pos = 0;
            } else {
                let _ = eng.step_decode((cmd - 1) as i32, &mut comm);
            }
        }
        return Ok(());
    }

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

    let (job_tx, mut job_rx) = mpsc::unbounded_channel::<ServeJob>();
    std::thread::spawn(move || {
        while let Some(job) = job_rx.blocking_recv() {
            let enc = match tok.encode(job.prompt, true) {
                Ok(e) => e,
                Err(e) => {
                    let _ = job.tx.send(ServeEvent::Error(format!("{e}")));
                    continue;
                }
            };
            let ids = enc.get_ids();
            let prompt_n = ids.len();
            eng.pos = 0;
            let mut reset_cmd = 0u32;
            let _ = comm.broadcast_u32(&mut reset_cmd);
            let mut next_tok = eng.cfg.bos_token_id;
            for &id in ids {
                let mut tcmd = id + 1;
                let _ = comm.broadcast_u32(&mut tcmd);
                match eng.step_decode(id as i32, &mut comm) {
                    Ok(nt) => next_tok = nt,
                    Err(e) => {
                        let _ = job.tx.send(ServeEvent::Error(format!("{e}")));
                        next_tok = eos;
                        break;
                    }
                }
            }
            let mut gen_ids: Vec<u32> = Vec::new();
            let mut prev_str = String::new();
            let mut finish_reason = "length";
            for _ in 0..job.max_tokens {
                if next_tok == eos || job.tx.is_closed() {
                    finish_reason = "stop";
                    break;
                }
                gen_ids.push(next_tok as u32);
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
                let mut tcmd = (next_tok as u32) + 1;
                let _ = comm.broadcast_u32(&mut tcmd);
                match eng.step_decode(next_tok, &mut comm) {
                    Ok(nt) => next_tok = nt,
                    Err(e) => {
                        let _ = job.tx.send(ServeEvent::Error(format!("{e}")));
                        break;
                    }
                }
            }
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
        world_size: 1,
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
        run_bench_layer(&cli.model, cli.max_ctx, cli.max_layers, &mut comm)?;
    }
    if cli.serve {
        run_serve(&cli, comm)?;
    }
    Ok(())
}
