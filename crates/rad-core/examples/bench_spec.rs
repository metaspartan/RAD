use anyhow::Result;
use rad_core::{DsparkDrafter, InklingEngine, Tokenizer};
use std::path::Path;
use std::time::Instant;

fn fmt_prompt(user: &str) -> String {
    format!(
        "<|begin_of_text|><|message_user|><|content_text|>{user}<|end_message|><|message_model|><|content_thinking|><|content_text|>"
    )
}

fn run_nospec(eng: &mut InklingEngine, prompt_ids: &[i32], max_toks: usize) -> Result<(Vec<i32>, f64)> {
    eng.reset()?;
    let mut nxt = eng.prefill_tokens(prompt_ids)?;
    let mut out = Vec::with_capacity(max_toks);
    let t0 = Instant::now();
    for _ in 0..max_toks {
        if nxt == 200006 || nxt == 200008 { break; }
        out.push(nxt);
        nxt = eng.step(nxt)?;
    }
    let dt = t0.elapsed().as_secs_f64();
    Ok((out, dt))
}

fn run_spec(eng: &mut InklingEngine, drafter: &mut DsparkDrafter, prompt_ids: &[i32], max_toks: usize, k_len: usize) -> Result<(Vec<i32>, f64, f64, f64, usize)> {
    eng.reset()?;
    drafter.reset();
    let mut nxt = 0i32;
    for (p, &t) in prompt_ids.iter().enumerate() {
        nxt = eng.step(t)?;
        drafter.inject_target_feats(eng.single_dspark_feats_ptr(), p as i32)?;
    }
    let mut out = Vec::with_capacity(max_toks);
    let mut rounds = 0usize;
    let mut draft_ms = 0.0f64;
    let mut verify_ms = 0.0f64;
    let t0 = Instant::now();
    while out.len() < max_toks {
        if nxt == 200006 || nxt == 200008 { break; }
        let td = Instant::now();
        let draft = drafter.draft(nxt, eng, k_len)?;
        draft_ms += td.elapsed().as_secs_f64() * 1000.0;

        let pos0 = eng.pos;
        let tv = Instant::now();
        let verified = eng.verify_batch(&draft)?;
        verify_ms += tv.elapsed().as_secs_f64() * 1000.0;

        let mut n_acc = 1usize;
        for j in 0..(k_len - 1) {
            if verified[j] == draft[j + 1] {
                n_acc += 1;
            } else {
                break;
            }
        }
        eng.rewind_verify_to(n_acc, k_len)?;
        for j in 0..n_acc {
            let fptr = eng.gather_batch_dspark_feats(j);
            drafter.inject_target_feats(fptr, pos0 + j as i32)?;
        }
        let mut stop = false;
        for j in 0..n_acc {
            let tok = draft[j];
            if tok == 200006 || tok == 200008 { stop = true; break; }
            out.push(tok);
            if out.len() >= max_toks { stop = true; break; }
        }
        nxt = verified[n_acc - 1];
        rounds += 1;
        if stop { break; }
    }
    let dt = t0.elapsed().as_secs_f64();
    Ok((out, dt, draft_ms / rounds.max(1) as f64, verify_ms / rounds.max(1) as f64, rounds))
}

fn main() -> Result<()> {
    let model_path = std::env::var("RAD_MODEL")
        .unwrap_or_else(|_| "./models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf".into());
    let draft_path = std::env::var("RAD_DSPARK")
        .unwrap_or_else(|_| "./models/dspark-Inkling-2-Mini-Q8_0.gguf".into());
    let tok_path = std::env::var("RAD_TOKENIZER")
        .unwrap_or_else(|_| "./models/tokenizer.json".into());
    let tok = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut eng = InklingEngine::load_multi(Path::new(&model_path), 2048, 8)?;
    eng.capture_decode_graph()?;
    let mut drafter = DsparkDrafter::load_multi(Path::new(&draft_path), 2048, 8)?;

    let prompts = [
        ("code", "Write a complete Python implementation of an LRU cache using OrderedDict with get and put methods, plus a short usage example."),
        ("prose", "Explain in three clear sentences how a B-tree database index speeds up range queries."),
    ];

    for (name, pr) in prompts {
        let ids: Vec<i32> = tok.encode(fmt_prompt(pr), false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        let (ref_out, ref_dt) = run_nospec(&mut eng, &ids, 128)?;
        println!("[{name}] nospec: {} tok in {:.3}s = {:.2} tok/s", ref_out.len(), ref_dt, ref_out.len() as f64 / ref_dt);

        for k_len in [4usize, 6usize, 8usize] {
            eng.capture_verify_graph(k_len)?;
            drafter.capture_draft_graph(&eng, k_len)?;
            let (spec_out, spec_dt, d_ms, v_ms, rounds) = run_spec(&mut eng, &mut drafter, &ids, 128, k_len)?;
            let exact = spec_out == ref_out;
            let emit = spec_out.len() as f64 / rounds.max(1) as f64;
            println!(
                "[{name}] spec K={k_len}: {} tok in {:.3}s = {:.2} tok/s | exact={exact} emit={emit:.2} draft={d_ms:.2}ms verify={v_ms:.2}ms",
                spec_out.len(), spec_dt, spec_out.len() as f64 / spec_dt
            );
            if !exact {
                let mut diff = 0;
                while diff < ref_out.len().min(spec_out.len()) && ref_out[diff] == spec_out[diff] { diff += 1; }
                println!("  FIRST DIFF at {diff}: ref={:?} spec={:?}", ref_out.get(diff), spec_out.get(diff));
            }
        }
    }
    for (n_users, k_len) in [(4usize, 4usize), (8usize, 4usize)] {
        let code_prompt = "Write a complete Python implementation of an LRU cache using OrderedDict with get and put methods, plus a short usage example.";
        let ids: Vec<i32> = tok.encode(fmt_prompt(code_prompt), false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        let seq_ids: Vec<i32> = (0..n_users as i32).collect();
        let mut cur_toks = vec![0i32; n_users];
        for u in 0..n_users {
            eng.reset_seq(u)?;
            drafter.reset_seq(u);
            let (first, _) = eng.prefill_seq(u, &ids)?;
            for j in 0..ids.len() {
                let fptr = eng.gather_batch_dspark_feats(j);
                drafter.inject_target_feats_seq(u, fptr, j as i32)?;
            }
            cur_toks[u] = first;
        }
        let dummy_drafts = drafter.draft_multi(&seq_ids, &cur_toks, &eng, k_len)?;
        let _ = eng.verify_multi(&seq_ids, &dummy_drafts)?;
        for u in 0..n_users {
            eng.reset_seq(u)?;
            drafter.reset_seq(u);
            let (first, _) = eng.prefill_seq(u, &ids)?;
            for j in 0..ids.len() {
                let fptr = eng.gather_batch_dspark_feats(j);
                drafter.inject_target_feats_seq(u, fptr, j as i32)?;
            }
            cur_toks[u] = first;
        }
        let mut outs: Vec<Vec<u32>> = vec![Vec::new(); n_users];
        let mut total_emitted = 0usize;
        let mut rounds = 0usize;
        let mut draft_ms = 0.0f64;
        let mut verify_ms = 0.0f64;
        let t0 = Instant::now();
        for _ in 0..24 {
            let td = Instant::now();
            let drafts = drafter.draft_multi(&seq_ids, &cur_toks, &eng, k_len)?;
            draft_ms += td.elapsed().as_secs_f64() * 1000.0;

            let pos_before: Vec<i32> = (0..n_users).map(|u| eng.seq_pos[u]).collect();
            let tv = Instant::now();
            let preds = eng.verify_multi(&seq_ids, &drafts)?;
            verify_ms += tv.elapsed().as_secs_f64() * 1000.0;

            let mut n_acc = vec![1usize; n_users];
            for u in 0..n_users {
                let mut acc = 1usize;
                for j in 0..(k_len - 1) {
                    if preds[u][j] == drafts[u][j + 1] {
                        acc += 1;
                    } else {
                        break;
                    }
                }
                n_acc[u] = acc;
                for j in 0..acc {
                    outs[u].push(drafts[u][j] as u32);
                    let fptr = eng.gather_batch_dspark_feats(u * k_len + j);
                    drafter.inject_target_feats_seq(u, fptr, pos_before[u] + j as i32)?;
                }
                cur_toks[u] = preds[u][acc - 1];
                total_emitted += acc;
            }
            eng.rewind_verify_multi(&seq_ids, &n_acc, k_len)?;
            rounds += 1;
        }
        let dt = t0.elapsed().as_secs_f64();
        let agg = total_emitted as f64 / dt;
        let per_user = agg / n_users as f64;
        let avg_emit = total_emitted as f64 / (rounds * n_users) as f64;
        println!(
            "[multi_spec B={n_users} K={k_len}]: {total_emitted} tok in {dt:.3}s = {agg:.1} agg tok/s ({per_user:.1} tok/s/user) | emit={avg_emit:.2}/user draft={:.2}ms verify={:.2}ms",
            draft_ms / rounds as f64, verify_ms / rounds as f64
        );
        let sample = tok.decode(&outs[0], true).unwrap_or_default();
        println!("  user[0]: {:?}", sample.chars().take(100).collect::<String>());
    }
    Ok(())
}
