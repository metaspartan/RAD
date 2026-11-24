use rad_core::{DFlash2Drafter, InklingEngine, Tokenizer};
use std::path::Path;
use std::time::Instant;

fn run_single(eng: &mut InklingEngine, df: &mut DFlash2Drafter, tok: &Tokenizer, prompt: &str, k_len: usize, max_toks: usize, label: &str) {
    let _ = eng.reset();
    df.reset();
    df.capture_draft_graph(eng, k_len).unwrap();

    let enc = tok.encode(prompt, false).unwrap();
    let ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();

    let mut cur_tok = 0i32;
    for (i, &t) in ids.iter().enumerate() {
        cur_tok = eng.step(t).unwrap();
        df.inject_target_feats_seq(0, eng.single_dspark_feats_ptr(), i as i32).unwrap();
    }

    let mut out = vec![cur_tok];
    let mut rounds = 0usize;
    let mut draft_ms = 0.0f64;
    let mut verify_ms = 0.0f64;
    let mut inject_ms = 0.0f64;
    let t0 = Instant::now();

    while out.len() < max_toks {
        let td = Instant::now();
        let draft_toks = df.draft(cur_tok, eng, k_len).unwrap();
        draft_ms += td.elapsed().as_secs_f64() * 1000.0;

        let pos_before = eng.pos;
        let tv = Instant::now();
        let preds = eng.verify_batch(&draft_toks).unwrap();
        verify_ms += tv.elapsed().as_secs_f64() * 1000.0;

        let mut n_acc = 0usize;
        for j in 0..(k_len - 1) {
            if preds[j] == draft_toks[j + 1] {
                n_acc += 1;
            } else {
                break;
            }
        }
        let bonus = preds[n_acc];
        let emit = n_acc + 1;
        eng.rewind_verify_to(emit, k_len).unwrap();

        let ti = Instant::now();
        let ds = df.stream();
        for j in 0..emit {
            let fptr = eng.gather_batch_dspark_feats_async(ds, j);
            df.inject_target_feats_seq_async(0, fptr, pos_before + j as i32).unwrap();
        }
        df.sync();
        inject_ms += ti.elapsed().as_secs_f64() * 1000.0;

        for j in 0..n_acc {
            out.push(draft_toks[j + 1]);
        }
        out.push(bonus);
        cur_tok = bonus;
        rounds += 1;
        if cur_tok == 200001 || cur_tok == 200002 {
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let gen = out.len() - 1;
    let tps = gen as f64 / elapsed;
    let avg_emit = gen as f64 / rounds as f64;
    let text = tok.decode(&out.iter().map(|&x| x as u32).collect::<Vec<_>>(), true).unwrap_or_default();
    let preview: String = text.chars().take(120).collect();
    println!(
        "{label} K={k_len}: {gen} tok in {:.3}s = {:.1} tok/s | rounds={rounds} avg_emit={avg_emit:.2} | draft={:.2}ms verify={:.2}ms inject={:.2}ms",
        elapsed, tps, draft_ms / rounds as f64, verify_ms / rounds as f64, inject_ms / rounds as f64
    );
    println!("  preview: {:?}", preview);
}

fn run_multi(eng: &mut InklingEngine, df: &mut DFlash2Drafter, tok: &Tokenizer, prompts: &[&str], n_users: usize, k_len: usize, max_toks: usize) {
    let mut cur_toks = vec![0i32; n_users];
    for u in 0..n_users {
        let _ = eng.reset_seq(u);
        df.reset_seq(u);
        let enc = tok.encode(prompts[u % prompts.len()], false).unwrap();
        let ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
        let mut last = 0i32;
        for (i, &t) in ids.iter().enumerate() {
            last = eng.step_multi(&[u as i32], &[t]).unwrap()[0];
            let fptr = eng.gather_batch_dspark_feats(0);
            df.inject_target_feats_seq(u, fptr, i as i32).unwrap();
        }
        cur_toks[u] = last;
    }

    let seq_ids: Vec<i32> = (0..n_users as i32).collect();
    let _ = df.draft_multi(&seq_ids, &cur_toks, eng, k_len).unwrap();

    let mut total_emitted = 0usize;
    let mut rounds = 0usize;
    let mut draft_ms = 0.0f64;
    let mut verify_ms = 0.0f64;
    let mut inject_ms = 0.0f64;
    let t0 = Instant::now();

    while total_emitted < n_users * max_toks {
        let td = Instant::now();
        let drafts = df.draft_multi(&seq_ids, &cur_toks, eng, k_len).unwrap();
        draft_ms += td.elapsed().as_secs_f64() * 1000.0;

        let pos_before: Vec<i32> = (0..n_users).map(|u| eng.seq_pos[u]).collect();

        let tv = Instant::now();
        let preds = eng.verify_multi(&seq_ids, &drafts).unwrap();
        verify_ms += tv.elapsed().as_secs_f64() * 1000.0;

        let mut emits = vec![0usize; n_users];
        for u in 0..n_users {
            let mut n_acc = 0usize;
            for j in 0..(k_len - 1) {
                if preds[u][j] == drafts[u][j + 1] {
                    n_acc += 1;
                } else {
                    break;
                }
            }
            emits[u] = n_acc + 1;
            cur_toks[u] = preds[u][n_acc];
            total_emitted += emits[u];
        }
        eng.rewind_verify_multi(&seq_ids, &emits, k_len).unwrap();

        let ti = Instant::now();
        let ds = df.stream();
        for u in 0..n_users {
            let base = u * k_len;
            for j in 0..emits[u] {
                let fptr = eng.gather_batch_dspark_feats_async(ds, base + j);
                df.inject_target_feats_seq_async(u, fptr, pos_before[u] + j as i32).unwrap();
            }
        }
        df.sync();
        inject_ms += ti.elapsed().as_secs_f64() * 1000.0;
        rounds += 1;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let agg_tps = total_emitted as f64 / elapsed;
    let per_user = agg_tps / n_users as f64;
    let avg_emit = total_emitted as f64 / (rounds * n_users) as f64;
    println!(
        "MULTI B={n_users} K={k_len}: {total_emitted} tok in {:.3}s = {:.1} agg tok/s ({:.1}/user) | avg_emit={avg_emit:.2} | draft={:.2}ms verify={:.2}ms inject={:.2}ms",
        elapsed, agg_tps, per_user, draft_ms / rounds as f64, verify_ms / rounds as f64, inject_ms / rounds as f64
    );
}

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "/home/ghost/models/inkling2/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf".into());
    let dflash2_path = std::env::args().nth(2).unwrap_or_else(|| "/home/ghost/models/inkling2-dflash2".into());
    let tok_path = std::env::args().nth(3).unwrap_or_else(|| "/home/ghost/models/inkling2-src/tokenizer.json".into());

    let tok = Tokenizer::from_file(&tok_path).unwrap();
    let mut eng = InklingEngine::load_multi(Path::new(&model_path), 4096, 8).unwrap();
    eng.capture_decode_graph().unwrap();
    let mut df = DFlash2Drafter::load_multi(Path::new(&dflash2_path), 4096, 8).unwrap();

    let code_prompt = "<|im_start|>user\nShow me a Python implementation of quicksort and mergesort with a benchmark comparing them on random arrays.<|im_end|>\n<|im_start|>assistant\n<|thought|>\n</|thought>\n";
    let prose_prompt = "<|im_start|>user\nExplain how modern GPU memory hierarchies work, from registers to L1/L2 caches and HBM, and why memory bandwidth limits LLM decoding.<|im_end|>\n<|im_start|>assistant\n<|thought|>\n</|thought>\n";
    let math_prompt = "<|im_start|>user\nDerive the closed-form formula for the sum of squares 1^2 + 2^2 + ... + n^2 step by step and verify for n=5.<|im_end|>\n<|im_start|>assistant\n<|thought|>\n</|thought>\n";

    for &k in &[6usize, 8usize] {
        run_single(&mut eng, &mut df, &tok, code_prompt, k, 160, "CODE");
        run_single(&mut eng, &mut df, &tok, math_prompt, k, 160, "MATH");
        run_single(&mut eng, &mut df, &tok, prose_prompt, k, 160, "PROSE");
    }

    let prompts = [code_prompt, math_prompt, code_prompt, math_prompt];
    run_multi(&mut eng, &mut df, &tok, &prompts, 4, 4, 96);
    run_multi(&mut eng, &mut df, &tok, &prompts, 4, 6, 96);
    run_multi(&mut eng, &mut df, &tok, &prompts, 4, 8, 96);
    run_multi(&mut eng, &mut df, &tok, &prompts, 8, 4, 96);
    run_multi(&mut eng, &mut df, &tok, &prompts, 8, 6, 96);
    run_multi(&mut eng, &mut df, &tok, &prompts, 8, 8, 96);
}
