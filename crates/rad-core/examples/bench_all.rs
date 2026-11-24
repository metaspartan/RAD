use anyhow::Result;
use rad_core::{InklingEngine, Tokenizer};
use std::path::Path;
use std::time::Instant;

fn fmt_prompt(user: &str) -> String {
    format!(
        "<|begin_of_text|><|message_user|><|content_text|>{user}<|end_message|><|message_model|><|content_thinking|><|content_text|>"
    )
}

fn main() -> Result<()> {
    let model_path = std::env::var("RAD_MODEL")
        .unwrap_or_else(|_| "./models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf".into());
    let tok_path = std::env::var("RAD_TOKENIZER")
        .unwrap_or_else(|_| "./models/tokenizer.json".into());
    let tok = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    let t0 = Instant::now();
    let max_seqs: usize = std::env::var("RAD_BENCH_SEQS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_ctx: usize = std::env::var("RAD_BENCH_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let mut eng = InklingEngine::load_multi(Path::new(&model_path), max_ctx, max_seqs)?;
    let load_s = t0.elapsed().as_secs_f64();
    let tc = Instant::now();
    eng.capture_decode_graph()?;
    let cap_ms = tc.elapsed().as_secs_f64() * 1000.0;
    eng.capture_verify_graph(8)?;
    println!("load={load_s:.2}s capture={cap_ms:.1}ms");

    let p1 = fmt_prompt("What is 17 * 23? Answer with just the number.");
    let ids1: Vec<i32> = tok.encode(p1, false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
    eng.reset_seq(0)?;
    let (mut nxt, _) = eng.prefill_seq(0, &ids1)?;
    let mut out1 = Vec::new();
    for _ in 0..32 {
        if nxt == 200006 || nxt == 200008 { break; }
        out1.push(nxt as u32);
        nxt = eng.step_multi(&[0], &[nxt])?[0];
    }
    println!("math: {:?}", tok.decode(&out1, true).unwrap_or_default().trim());

    let p2 = fmt_prompt("Write a concise Python function is_palindrome(s) and explain it in one sentence.");
    let ids2: Vec<i32> = tok.encode(p2, false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
    eng.reset_seq(0)?;
    let tp = Instant::now();
    let (mut nxt, _) = eng.prefill_seq(0, &ids2)?;
    let pp_ms = tp.elapsed().as_secs_f64() * 1000.0;
    let mut out2 = Vec::new();
    let td = Instant::now();
    for _ in 0..128 {
        if nxt == 200006 || nxt == 200008 { break; }
        out2.push(nxt as u32);
        nxt = eng.step_multi(&[0], &[nxt])?[0];
    }
    let dec_s = td.elapsed().as_secs_f64();
    let dec_tps = out2.len() as f64 / dec_s;
    println!("b1_decode: {} tok in {:.3}s = {:.2} tok/s (pp={} tok in {:.1}ms)", out2.len(), dec_s, dec_tps, ids2.len(), pp_ms);
    println!("code: {:?}", tok.decode(&out2, true).unwrap_or_default().chars().take(160).collect::<String>());

    if max_ctx >= 2048 {
        let mut long_text = String::from("Summarize the key points of the following system specification:\n");
        for i in 0..60 {
            long_text.push_str(&format!("Requirement {i}: The distributed database must maintain linearizable consistency across replicas while keeping p99 latency under 10ms.\n"));
        }
        let plong = fmt_prompt(&long_text);
        let idsl: Vec<i32> = tok.encode(plong, false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        eng.reset_seq(0)?;
        let tpl = Instant::now();
        let _ = eng.prefill_seq(0, &idsl)?;
        let ppl_s = tpl.elapsed().as_secs_f64();
        println!("long_prefill: {} tok in {:.3}s = {:.1} tok/s", idsl.len(), ppl_s, idsl.len() as f64 / ppl_s);

        let mut idsl2 = idsl.clone();
        let extra: Vec<i32> = tok.encode(" Also list 3 risks.", false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        idsl2.extend_from_slice(&extra);
        let tpc = Instant::now();
        let (_, cached) = eng.prefill_seq(0, &idsl2)?;
        let pc_ms = tpc.elapsed().as_secs_f64() * 1000.0;
        println!("prefix_cache: total={} cached={} suffix={} ttft={:.1}ms", idsl2.len(), cached, idsl2.len() - cached, pc_ms);
    }

    eng.reset()?;
    let _ = eng.prefill_tokens(&ids2)?;
    let dummy8 = vec![100i32; 8];
    let _ = eng.verify_batch(&dummy8)?;
    let tv = Instant::now();
    let iters = 10;
    for _ in 0..iters {
        let _ = eng.verify_batch(&dummy8)?;
    }
    let v_ms = tv.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("verify_b8: {:.2} ms/step (max={:.1} tok/s @8/8, {:.1} tok/s @6/8)", v_ms, 8000.0 / v_ms, 6000.0 / v_ms);

    let prompts = [
        "What is 12 + 34?",
        "Name 3 planets.",
        "What is the capital of France?",
        "Write hello world in Rust.",
        "Explain TCP in one sentence.",
        "What is 9 * 9?",
        "Name 3 programming languages.",
        "What color is the sky?",
    ];
    let mut cur_toks = vec![0i32; 8];
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); 8];
    for (s, pr) in prompts.iter().enumerate() {
        let ids: Vec<i32> = tok.encode(fmt_prompt(pr), false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        eng.reset_seq(s)?;
        let (first, _) = eng.prefill_seq(s, &ids)?;
        cur_toks[s] = first;
    }
    let seq_ids: Vec<i32> = (0..8).collect();
    cur_toks = eng.step_multi(&seq_ids, &cur_toks)?;
    let tm = Instant::now();
    let steps = 48;
    for _ in 0..steps {
        for s in 0..8 {
            outs[s].push(cur_toks[s] as u32);
        }
        cur_toks = eng.step_multi(&seq_ids, &cur_toks)?;
    }
    let m_s = tm.elapsed().as_secs_f64();
    let total_toks = steps * 8;
    println!("multi_b8: {} tok in {:.3}s = {:.1} agg tok/s ({:.1} tok/s/user)", total_toks, m_s, total_toks as f64 / m_s, steps as f64 / m_s);
    for s in 0..3 {
        let txt = tok.decode(&outs[s], true).unwrap_or_default();
        println!("  user[{s}]: {:?}", txt.chars().take(80).collect::<String>());
    }

    let mut cur16 = vec![0i32; 16];
    let mut outs16: Vec<Vec<u32>> = vec![Vec::new(); 16];
    for s in 0..16 {
        let pr = prompts[s % 8];
        let ids: Vec<i32> = tok.encode(fmt_prompt(pr), false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        eng.reset_seq(s)?;
        let (first, _) = eng.prefill_seq(s, &ids)?;
        cur16[s] = first;
    }
    let seq16: Vec<i32> = (0..16).collect();
    cur16 = eng.step_multi(&seq16, &cur16)?;
    let tm16 = Instant::now();
    for _ in 0..steps {
        for s in 0..16 {
            outs16[s].push(cur16[s] as u32);
        }
        cur16 = eng.step_multi(&seq16, &cur16)?;
    }
    let m16_s = tm16.elapsed().as_secs_f64();
    let tot16 = steps * 16;
    println!("multi_b16: {} tok in {:.3}s = {:.1} agg tok/s ({:.1} tok/s/user)", tot16, m16_s, tot16 as f64 / m16_s, steps as f64 / m16_s);

    let mut cur32 = vec![0i32; 32];
    let mut outs32: Vec<Vec<u32>> = vec![Vec::new(); 32];
    for s in 0..32 {
        let pr = prompts[s % 8];
        let ids: Vec<i32> = tok.encode(fmt_prompt(pr), false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        eng.reset_seq(s)?;
        let (first, _) = eng.prefill_seq(s, &ids)?;
        cur32[s] = first;
    }
    let seq32: Vec<i32> = (0..32).collect();
    cur32 = eng.step_multi(&seq32, &cur32)?;
    let tm32 = Instant::now();
    for _ in 0..steps {
        for s in 0..32 {
            outs32[s].push(cur32[s] as u32);
        }
        cur32 = eng.step_multi(&seq32, &cur32)?;
    }
    let m32_s = tm32.elapsed().as_secs_f64();
    let tot32 = steps * 32;
    println!("multi_b32: {} tok in {:.3}s = {:.1} agg tok/s ({:.1} tok/s/user)", tot32, m32_s, tot32 as f64 / m32_s, steps as f64 / m32_s);

    if max_seqs >= 64 {
        let mut cur64 = vec![0i32; 64];
        for s in 0..64 {
            let pr = prompts[s % 8];
            let ids: Vec<i32> = tok.encode(fmt_prompt(pr), false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
            eng.reset_seq(s)?;
            let (first, _) = eng.prefill_seq(s, &ids)?;
            cur64[s] = first;
        }
        let seq64: Vec<i32> = (0..64).collect();
        cur64 = eng.step_multi(&seq64, &cur64)?;
        let tm64 = Instant::now();
        for _ in 0..steps {
            cur64 = eng.step_multi(&seq64, &cur64)?;
        }
        let m64_s = tm64.elapsed().as_secs_f64();
        let tot64 = steps * 64;
        println!("multi_b64: {} tok in {:.3}s = {:.1} agg tok/s ({:.1} tok/s/user)", tot64, m64_s, tot64 as f64 / m64_s, steps as f64 / m64_s);
    }
    if max_seqs >= 128 {
        let mut cur128 = vec![0i32; 128];
        for s in 0..128 {
            let pr = prompts[s % 8];
            let ids: Vec<i32> = tok.encode(fmt_prompt(pr), false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
            eng.reset_seq(s)?;
            let (first, _) = eng.prefill_seq(s, &ids)?;
            cur128[s] = first;
        }
        let seq128: Vec<i32> = (0..128).collect();
        cur128 = eng.step_multi(&seq128, &cur128)?;
        let tm128 = Instant::now();
        for _ in 0..steps {
            cur128 = eng.step_multi(&seq128, &cur128)?;
        }
        let m128_s = tm128.elapsed().as_secs_f64();
        let tot128 = steps * 128;
        println!("multi_b128: {} tok in {:.3}s = {:.1} agg tok/s ({:.1} tok/s/user)", tot128, m128_s, tot128 as f64 / m128_s, steps as f64 / m128_s);
    }
    Ok(())
}
