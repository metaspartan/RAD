mod config;
mod cuda_ops;
mod safetensors;

use anyhow::Result;
use clap::Parser;
use config::DeepseekV41Config;
use rad_cuda::comm::Tp2Comm;
use rad_cuda::{CudaBuf, CudaFail};
use safetensors::SafetensorsIndex;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "rad-dsv41", about = "RAD CUDA DeepSeek-V4.1-Flash EXL3 Engine")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = false)]
    bench_exl3: bool,
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

        let d_x = CudaBuf::new(k * 2 * 5).map_err(CudaFail)?;
        let d_xrot = CudaBuf::new(k * 2 * 5).map_err(CudaFail)?;
        let d_z = CudaBuf::new(n * 4 * 5).map_err(CudaFail)?;
        let d_y = CudaBuf::new(n * 2 * 5).map_err(CudaFail)?;

        for m in [1i32, 5i32] {
            for _ in 0..10 {
                unsafe {
                    cuda_ops::rad_cuda_wht128_pre_mul_f16(d_x.ptr, d_suh.ptr, d_xrot.ptr, m, k as i32, scale, std::ptr::null_mut());
                    cuda_ops::rad_cuda_exl3_gemv_f16(d_xrot.ptr, d_trel.ptr, d_svh.ptr, d_z.ptr, d_y.ptr, m, k as i32, n as i32, bits as i32, scale, 0, std::ptr::null_mut());
                }
            }
            rad_cuda::sync_device().map_err(CudaFail)?;
            let t0 = Instant::now();
            let iters = 200;
            for _ in 0..iters {
                unsafe {
                    cuda_ops::rad_cuda_wht128_pre_mul_f16(d_x.ptr, d_suh.ptr, d_xrot.ptr, m, k as i32, scale, std::ptr::null_mut());
                    cuda_ops::rad_cuda_exl3_gemv_f16(d_xrot.ptr, d_trel.ptr, d_svh.ptr, d_z.ptr, d_y.ptr, m, k as i32, n as i32, bits as i32, scale, 0, std::ptr::null_mut());
                }
            }
            rad_cuda::sync_device().map_err(CudaFail)?;
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            let gb_s = (trel_len as f64) / dt / 1e9;
            println!(
                "  [{prefix:24s} {bits}b K={k:5} N={n:6} M={m}] {:7.1} us  {:5.1} GB/s",
                dt * 1e6,
                gb_s
            );
        }
    }

    if comm.world_size() == 2 {
        let mut buf = vec![1.0f32; cfg.text_config.hidden_size];
        for _ in 0..20 {
            comm.all_reduce_sum_f32(&mut buf)?;
        }
        let t0 = Instant::now();
        let iters = 500;
        for _ in 0..iters {
            comm.all_reduce_sum_f32(&mut buf)?;
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("  [ConnectX-7 TP=2 AllReduce 5120xf32] {:.1} us/call", us);
    }
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
    Ok(())
}
