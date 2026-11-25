use std::path::Path;
use std::time::Instant;
use rad_core::{InklingEngine, Tokenizer};
use serde_json::json;
use crate::prompt::format_prompt;
use crate::scheduler::{ngram_bucket_len, ngram_lookup_draft, prefill_with_drafter};
use crate::types::ChatMessage;
use crate::{load_drafter, CliOpts};

pub(crate) fn mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len().max(1) as f64;
    let m = vals.iter().sum::<f64>() / n;
    let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, v.sqrt())
}

pub(crate) fn run_bench(opts: &CliOpts, tok_path: &str) -> anyhow::Result<()> {
    println!("RAD Benchmark Suite (v{})", env!("CARGO_PKG_VERSION"));
    let t0 = Instant::now();
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let max_seqs = if opts.ctx > 32768 {
        opts.max_seqs.max(1)
    } else if opts.ctx > 8192 {
        opts.max_seqs.max(4)
    } else {
        opts.max_seqs.max(16)
    };
    let mut engine = InklingEngine::load_multi(Path::new(&opts.model), opts.ctx.max(4096), max_seqs)?;
    engine.capture_decode_graph()?;
    let mut drafter = load_drafter(opts, &mut engine, max_seqs.min(16))?;
    let load_s = t0.elapsed().as_secs_f64();
    let spec_mode = drafter.as_ref().map(|d| d.mode_name()).unwrap_or("none");
    println!("Loaded model in {:.2}s (spec={spec_mode}, ctx={}, max_seqs={max_seqs})", load_s, opts.ctx);
    println!();

    let short_p = format_prompt(
        &[ChatMessage { role: "user".into(), content: json!("Explain GPU memory bandwidth in one paragraph."), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
        false,
    );
    let short_ids: Vec<i32> = tokenizer.encode(short_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
    engine.reset_seq(0)?;
    let _ = engine.prefill_tokens_seq(0, &short_ids)?;
    let mut ttft_runs = Vec::new();
    let mut cur_tok = 0i32;
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let ts = Instant::now();
        cur_tok = engine.prefill_tokens_seq(0, &short_ids)?;
        ttft_runs.push(ts.elapsed().as_secs_f64() * 1000.0);
    }
    let (ttft_ms, ttft_sd) = mean_std(&ttft_runs);

    let dec_n = 128usize;
    let mut tg128_runs = Vec::new();
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let mut ct = engine.prefill_tokens_seq(0, &short_ids)?;
        let td = Instant::now();
        for _ in 0..dec_n {
            ct = engine.step(ct)?;
        }
        let _ = ct;
        tg128_runs.push(dec_n as f64 / td.elapsed().as_secs_f64());
    }
    let (tg128_tps, tg128_sd) = mean_std(&tg128_runs);

    let long_chunk = "Modern GPU architectures execute thousands of lightweight threads in parallel across SIMD wavefronts while hiding global HBM latency with large register files, shared local data store memory, and multi-level caches. ";
    let mut long_text = String::from("Summarize the following technical notes:\n");
    for _ in 0..80 {
        long_text.push_str(long_chunk);
    }
    let long_p = format_prompt(
        &[ChatMessage { role: "user".into(), content: json!(long_text), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
        false,
    );
    let base_ids: Vec<i32> = tokenizer.encode(long_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
    let pp512_ids = &base_ids[..512.min(base_ids.len())];
    let pp2048_ids = &base_ids[..2048.min(base_ids.len())];

    engine.reset_seq(0)?;
    let _ = engine.prefill_tokens_seq(0, pp512_ids)?;
    let mut pp512_runs = Vec::new();
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let tpp = Instant::now();
        let _ = engine.prefill_tokens_seq(0, pp512_ids)?;
        pp512_runs.push(pp512_ids.len() as f64 / tpp.elapsed().as_secs_f64());
    }
    let (pp512_tps, pp512_sd) = mean_std(&pp512_runs);

    engine.reset_seq(0)?;
    let _ = engine.prefill_tokens_seq(0, pp2048_ids)?;
    let mut pp2048_runs = Vec::new();
    let mut last_d2048 = 0i32;
    for _ in 0..3 {
        engine.reset_seq(0)?;
        let tpp = Instant::now();
        last_d2048 = engine.prefill_tokens_seq(0, pp2048_ids)?;
        pp2048_runs.push(pp2048_ids.len() as f64 / tpp.elapsed().as_secs_f64());
    }
    let (pp2048_tps, pp2048_sd) = mean_std(&pp2048_runs);

    let td2k = Instant::now();
    let mut ct2k = last_d2048;
    for _ in 0..dec_n {
        ct2k = engine.step(ct2k)?;
    }
    let _ = ct2k;
    let tg128_d2048_tps = dec_n as f64 / td2k.elapsed().as_secs_f64();

    let mut pp2048_spec_tps = 0.0f64;
    let mut pp2048_spec_sd = 0.0f64;
    if let Some(ref mut d) = drafter {
        engine.reset_seq(0)?;
        d.reset_seq(0);
        let _ = prefill_with_drafter(&mut engine, 0, pp2048_ids, &[], None, Some(d), None)?;
        let mut s_runs = Vec::new();
        for _ in 0..3 {
            engine.reset_seq(0)?;
            d.reset_seq(0);
            let tps = Instant::now();
            let _ = prefill_with_drafter(&mut engine, 0, pp2048_ids, &[], None, Some(d), None)?;
            s_runs.push(pp2048_ids.len() as f64 / tps.elapsed().as_secs_f64());
        }
        let (m, s) = mean_std(&s_runs);
        pp2048_spec_tps = m;
        pp2048_spec_sd = s;
    }

    let _ = cur_tok;
    println!("| Benchmark | Result | Details |");
    println!("| --- | --- | --- |");
    println!("| Cold Load | {:.2} s | spec={spec_mode}, hipGraph captured |", load_s);
    println!("| Short Prompt TTFT | {:.1} ± {:.1} ms | {} tokens (3 runs) |", ttft_ms, ttft_sd, short_ids.len());
    println!("| pp512 (No-Spec) | {:.1} ± {:.1} tok/s | {} tokens, llama-bench parity (3 runs) |", pp512_tps, pp512_sd, pp512_ids.len());
    println!("| pp2048 (No-Spec) | {:.1} ± {:.1} tok/s | {} tokens, llama-bench parity (3 runs) |", pp2048_tps, pp2048_sd, pp2048_ids.len());
    if pp2048_spec_tps > 0.0 {
        println!("| pp2048 ({spec_mode}) | {:.1} ± {:.1} tok/s | {} tokens, batched feature inject (3 runs) |", pp2048_spec_tps, pp2048_spec_sd, pp2048_ids.len());
    }
    println!("| tg128 (No-Spec) | {:.1} ± {:.1} tok/s | B=1 hipGraph, d=0 (3 runs) |", tg128_tps, tg128_sd);
    println!("| tg128 @ d=2048 (No-Spec) | {:.1} tok/s | B=1 hipGraph, 2K KV depth |", tg128_d2048_tps);

    if let Some(ref mut d) = drafter {
        let k_len = d.spec_k();
        let mut long_code_ctx = String::from("Here is an existing Rust networking library:\n```rust\n");
        for i in 0..38 {
            long_code_ctx.push_str(&format!(
                "pub struct BufferPool{i} {{ cap: usize, used: usize }}\nimpl BufferPool{i} {{ pub fn new(cap: usize) -> Self {{ Self {{ cap, used: 0 }} }} pub fn remaining(&self) -> usize {{ self.cap - self.used }} }}\n"
            ));
        }
        long_code_ctx.push_str("```\nNow write a complete Python implementation of quicksort and mergesort with a benchmark comparing them on random arrays.");
        let prompts: [(&str, &str); 4] = [
            ("Code", "Show me a Python implementation of quicksort and mergesort with a benchmark comparing them on random arrays."),
            ("Code (2K ctx)", &long_code_ctx),
            ("Math", "Derive the closed-form formula for the sum of squares 1^2 + 2^2 + ... + n^2 step by step and verify for n=5."),
            ("Prose", "Explain how modern GPU memory hierarchies work, from registers to L1/L2 caches and HBM, and why memory bandwidth limits LLM decoding."),
        ];
        for (label, text) in prompts {
            let p = format_prompt(&[ChatMessage { role: "user".into(), content: json!(text), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }], false);
            let ids: Vec<i32> = tokenizer.encode(p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            engine.reset_seq(0)?;
            d.reset_seq(0);
            let (mut ct, _) = prefill_with_drafter(&mut engine, 0, &ids, &[], None, Some(d), None)?;
            let mut gen = 0usize;
            let mut rounds = 0usize;
            let mut d_ms = 0.0f64;
            let mut v_ms = 0.0f64;
            let mut i_ms = 0.0f64;
            let t1 = Instant::now();
            while gen < 160 {
                let td0 = Instant::now();
                let draft = d.draft(ct, &engine, k_len)?;
                d_ms += td0.elapsed().as_secs_f64() * 1000.0;
                let pos0 = engine.pos;
                let tv0 = Instant::now();
                let preds = engine.verify_batch(&draft)?;
                v_ms += tv0.elapsed().as_secs_f64() * 1000.0;
                let mut n_acc = 1usize;
                for j in 0..(k_len - 1) {
                    if preds[j] == draft[j + 1] { n_acc += 1; } else { break; }
                }
                engine.rewind_verify_to(n_acc, k_len)?;
                engine.seq_pos[0] = engine.pos;
                let ti0 = Instant::now();
                d.inject_batch_seq_async(0, &engine, 0, pos0, n_acc)?;
                i_ms += ti0.elapsed().as_secs_f64() * 1000.0;
                ct = preds[n_acc - 1];
                gen += n_acc;
                rounds += 1;
                if ct == 200006 || ct == 200008 { break; }
            }
            let tps = gen as f64 / t1.elapsed().as_secs_f64();
            let avg_e = gen as f64 / rounds.max(1) as f64;
            println!(
                "| Single-User {label} ({spec_mode} K={k_len}) | {:.1} tok/s | emit={:.2}, draft={:.2}ms, verify={:.2}ms, inject={:.2}ms |",
                tps, avg_e, d_ms / rounds as f64, v_ms / rounds as f64, i_ms / rounds as f64
            );
        }

        for &b in &[4usize, 8usize, 16usize] {
            if b <= d.max_seqs() {
                let mk = if b * d.spec_k() <= engine.multi_verify_cap { d.spec_k() } else { 4usize };
                let code_p = format_prompt(
                    &[ChatMessage { role: "user".into(), content: json!("Show me a Python implementation of quicksort and mergesort."), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
                    false,
                );
                let ids: Vec<i32> = tokenizer.encode(code_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
                let seq_ids: Vec<i32> = (0..b as i32).collect();
                let mut cur = vec![0i32; b];
                for u in 0..b {
                    engine.reset_seq(u)?;
                    d.reset_seq(u);
                    let (t, _) = prefill_with_drafter(&mut engine, u, &ids, &[], None, Some(d), None)?;
                    cur[u] = t;
                }
                let _ = d.draft_multi(&seq_ids, &cur, &engine, mk)?;
                let mut tot = 0usize;
                let t1 = Instant::now();
                while tot < b * 80 {
                    let drafts = d.draft_multi(&seq_ids, &cur, &engine, mk)?;
                    let pos0: Vec<i32> = (0..b).map(|u| engine.seq_pos[u]).collect();
                    let preds = engine.verify_multi(&seq_ids, &drafts)?;
                    let mut emits = vec![0usize; b];
                    for u in 0..b {
                        let mut n_acc = 1usize;
                        for j in 0..(mk - 1) {
                            if preds[u][j] == drafts[u][j + 1] { n_acc += 1; } else { break; }
                        }
                        emits[u] = n_acc;
                        cur[u] = preds[u][n_acc - 1];
                        tot += n_acc;
                    }
                    engine.rewind_verify_multi(&seq_ids, &emits, mk)?;
                    for u in 0..b {
                        d.inject_batch_seq_async(u, &engine, u * mk, pos0[u], emits[u])?;
                    }
                    d.sync();
                }
                let agg = tot as f64 / t1.elapsed().as_secs_f64();
                let per_u = agg / b as f64;
                println!("| Multi-User B={b} ({spec_mode} K={mk}) | {:.1} agg tok/s | {:.1} tok/s/user (ITL {:.1}ms) |", agg, per_u, 1000.0 / per_u);

                let hk_max = (engine.multi_verify_cap / b).min(128);
                if hk_max >= 16 {
                    let edit_txt = "pub struct KvCachePool {\n    pub blocks: Vec<KvBlock>,\n    pub free_list: Vec<usize>,\n    pub block_size: usize,\n    pub max_blocks: usize,\n    pub hits: u64,\n    pub misses: u64,\n}\nimpl KvCachePool {\n    pub fn with_capacity(max_blocks: usize, block_size: usize) -> Self {\n        let mut blocks = Vec::with_capacity(max_blocks);\n        let mut free_list = Vec::with_capacity(max_blocks);\n        for i in 0..max_blocks {\n            blocks.push(KvBlock::new(i));\n            free_list.push(max_blocks - 1 - i);\n        }\n        Self { blocks, free_list, block_size, max_blocks, hits: 0, misses: 0 }\n    }\n    pub fn alloc_block(&mut self) -> Option<usize> {\n        let id = self.free_list.pop()?;\n        self.blocks[id].ref_count = 1;\n        self.blocks[id].dirty = true;\n        Some(id)\n    }\n    pub fn free_block(&mut self, id: usize) {\n        if id < self.blocks.len() && self.blocks[id].release() {\n            self.blocks[id].dirty = false;\n            self.free_list.push(id);\n        }\n    }\n    pub fn used_blocks(&self) -> usize {\n        self.max_blocks.saturating_sub(self.free_list.len())\n    }\n    pub fn free_blocks(&self) -> usize {\n        self.free_list.len()\n    }\n    pub fn utilization(&self) -> f64 {\n        if self.max_blocks == 0 { 0.0 } else { self.used_blocks() as f64 / self.max_blocks as f64 }\n    }\n}\n";
                    let hp = format_prompt(
                        &[ChatMessage { role: "user".into(), content: json!(format!("Output ONLY the exact `KvCachePool` struct and `impl KvCachePool` code verbatim (no markdown fences, no commentary):\n{edit_txt}")), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
                        false,
                    );
                    let hids: Vec<i32> = tokenizer.encode(hp, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
                    let mut hcur = vec![0i32; b];
                    for u in 0..b {
                        engine.reset_seq(u)?;
                        d.reset_seq(u);
                        let (t, _) = prefill_with_drafter(&mut engine, u, &hids, &[], None, Some(d), None)?;
                        hcur[u] = t;
                    }
                    let warm_d: Vec<Vec<i32>> = (0..b).map(|u| vec![hcur[u]; hk_max]).collect();
                    let _ = engine.verify_multi(&seq_ids, &warm_d)?;
                    for u in 0..b {
                        engine.reset_seq(u)?;
                        d.reset_seq(u);
                        let (t, _) = prefill_with_drafter(&mut engine, u, &hids, &[], None, Some(d), None)?;
                        hcur[u] = t;
                    }
                    let mut htot = 0usize;
                    let ht0 = Instant::now();
                    let mut hrounds = 0usize;
                    while htot < b * 240 && hrounds < 40 {
                        let mut cands: Vec<Vec<i32>> = Vec::with_capacity(b);
                        let mut min_l = hk_max;
                        let mut all_ng = true;
                        for u in 0..b {
                            if let Some(ng) = ngram_lookup_draft(&engine.seq_tokens[u], hcur[u], hk_max) {
                                min_l = min_l.min(ng.len());
                                cands.push(ng);
                            } else {
                                all_ng = false;
                                break;
                            }
                        }
                        let (drafts, rk) = if all_ng && ngram_bucket_len(min_l) >= 8 {
                            let bk = ngram_bucket_len(min_l);
                            for c in &mut cands { c.truncate(bk); }
                            (cands, bk)
                        } else {
                            (d.draft_multi(&seq_ids, &hcur, &engine, mk)?, mk)
                        };
                        let pos0: Vec<i32> = (0..b).map(|u| engine.seq_pos[u]).collect();
                        let preds = engine.verify_multi(&seq_ids, &drafts)?;
                        let mut emits = vec![0usize; b];
                        for u in 0..b {
                            let mut n_acc = 1usize;
                            for j in 0..(rk - 1) {
                                if preds[u][j] == drafts[u][j + 1] { n_acc += 1; } else { break; }
                            }
                            emits[u] = n_acc;
                            for j in 0..n_acc {
                                engine.seq_tokens[u].push(drafts[u][j]);
                            }
                            hcur[u] = preds[u][n_acc - 1];
                            htot += n_acc;
                        }
                        engine.rewind_verify_multi(&seq_ids, &emits, rk)?;
                        for u in 0..b {
                            d.inject_batch_seq_async(u, &engine, u * rk, pos0[u], emits[u])?;
                        }
                        d.sync();
                        hrounds += 1;
                    }
                    let hagg = htot as f64 / ht0.elapsed().as_secs_f64();
                    let hpu = hagg / b as f64;
                    println!("| Multi-User B={b} (Hybrid {spec_mode}+K={hk_max}) | {:.1} agg tok/s | {:.1} tok/s/user (ITL {:.1}ms) |", hagg, hpu, 1000.0 / hpu);
                }
            }
        }
    }

    for &b in &[8usize, 16usize, 32usize, 64usize] {
        if b <= max_seqs {
            let seq_ids: Vec<i32> = (0..b as i32).collect();
            let mut cur = vec![0i32; b];
            for u in 0..b {
                engine.reset_seq(u)?;
                cur[u] = engine.prefill_tokens_seq(u, &short_ids)?;
            }
            let _ = engine.step_multi(&seq_ids, &cur)?;
            let steps = 48usize;
            let t1 = Instant::now();
            for _ in 0..steps {
                cur = engine.step_multi(&seq_ids, &cur)?;
            }
            let agg = (b * steps) as f64 / t1.elapsed().as_secs_f64();
            let per_u = agg / b as f64;
            println!("| Multi-User B={b} (No-Spec) | {:.1} agg tok/s | {:.1} tok/s/user (ITL {:.1}ms) |", agg, per_u, 1000.0 / per_u);
        }
    }

    if let Some(ref mut d) = drafter {
        let k_len = d.spec_k();
        let targets: &[usize] = &[2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
        let mut long_targets: Vec<usize> = targets
            .iter()
            .copied()
            .filter(|&t| t <= opts.ctx)
            .map(|t| t.min(opts.ctx.saturating_sub(768)))
            .collect();
        if long_targets.len() >= 2 {
            println!();
            println!("Long-Context Scaling (prefill + hybrid {spec_mode}/ngram K=128 decode):");
            let block_txt = "pub struct KvBlock {\n    pub block_id: usize,\n    pub ref_count: u32,\n    pub tokens: [i32; 16],\n    pub dirty: bool,\n}\nimpl KvBlock {\n    pub fn new(block_id: usize) -> Self {\n        Self { block_id, ref_count: 1, tokens: [0; 16], dirty: false }\n    }\n    pub fn retain(&mut self) {\n        self.ref_count = self.ref_count.saturating_add(1);\n    }\n    pub fn release(&mut self) -> bool {\n        self.ref_count = self.ref_count.saturating_sub(1);\n        self.ref_count == 0\n    }\n}\n";
            let target_txt = "pub struct KvCachePool {\n    pub blocks: Vec<KvBlock>,\n    pub free_list: Vec<usize>,\n    pub block_size: usize,\n    pub max_blocks: usize,\n    pub hits: u64,\n    pub misses: u64,\n}\nimpl KvCachePool {\n    pub fn with_capacity(max_blocks: usize, block_size: usize) -> Self {\n        let mut blocks = Vec::with_capacity(max_blocks);\n        let mut free_list = Vec::with_capacity(max_blocks);\n        for i in 0..max_blocks {\n            blocks.push(KvBlock::new(i));\n            free_list.push(max_blocks - 1 - i);\n        }\n        Self { blocks, free_list, block_size, max_blocks, hits: 0, misses: 0 }\n    }\n    pub fn alloc_block(&mut self) -> Option<usize> {\n        let id = self.free_list.pop()?;\n        self.blocks[id].ref_count = 1;\n        self.blocks[id].dirty = true;\n        Some(id)\n    }\n    pub fn free_block(&mut self, id: usize) {\n        if id < self.blocks.len() && self.blocks[id].release() {\n            self.blocks[id].dirty = false;\n            self.free_list.push(id);\n        }\n    }\n    pub fn retain_block(&mut self, id: usize) -> bool {\n        if id < self.blocks.len() && self.blocks[id].ref_count > 0 {\n            self.blocks[id].retain();\n            self.hits += 1;\n            true\n        } else {\n            self.misses += 1;\n            false\n        }\n    }\n    pub fn write_token(&mut self, id: usize, slot: usize, tok: i32) -> bool {\n        if id < self.blocks.len() && slot < 16 {\n            self.blocks[id].tokens[slot] = tok;\n            self.blocks[id].dirty = true;\n            true\n        } else {\n            false\n        }\n    }\n    pub fn read_token(&self, id: usize, slot: usize) -> Option<i32> {\n        if id < self.blocks.len() && slot < 16 {\n            Some(self.blocks[id].tokens[slot])\n        } else {\n            None\n        }\n    }\n    pub fn used_blocks(&self) -> usize {\n        self.max_blocks.saturating_sub(self.free_list.len())\n    }\n    pub fn free_blocks(&self) -> usize {\n        self.free_list.len()\n    }\n    pub fn utilization(&self) -> f64 {\n        if self.max_blocks == 0 { 0.0 } else { self.used_blocks() as f64 / self.max_blocks as f64 }\n    }\n    pub fn hit_rate(&self) -> f64 {\n        let total = self.hits + self.misses;\n        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }\n    }\n    pub fn reset_all(&mut self) {\n        self.free_list.clear();\n        for i in 0..self.max_blocks {\n            self.blocks[i].ref_count = 0;\n            self.blocks[i].dirty = false;\n            self.free_list.push(self.max_blocks - 1 - i);\n        }\n        self.hits = 0;\n        self.misses = 0;\n    }\n}\n";
            let block_ids: Vec<i32> = tokenizer.encode(block_txt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let target_ids: Vec<i32> = tokenizer.encode(target_txt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let prefix_p = format_prompt(
                &[ChatMessage { role: "user".into(), content: json!("Here is a large Rust codebase file. At the end is `KvCachePool`. Output ONLY the exact `KvCachePool` struct and `impl KvCachePool` code verbatim from the file (no markdown fences, no commentary):\n"), tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }],
                false,
            );
            let mut prefix_ids: Vec<i32> = tokenizer.encode(prefix_p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let tail_ids = if prefix_ids.len() >= 2 { prefix_ids.split_off(prefix_ids.len() - 2) } else { Vec::new() };
            for tgt in long_targets.drain(..) {
                let mut ids = Vec::with_capacity(tgt + 64);
                ids.extend_from_slice(&prefix_ids);
                while ids.len() + block_ids.len() + target_ids.len() + tail_ids.len() <= tgt {
                    ids.extend_from_slice(&block_ids);
                }
                ids.extend_from_slice(&target_ids);
                ids.extend_from_slice(&tail_ids);
                engine.reset_seq(0)?;
                d.reset_seq(0);
                let tp = Instant::now();
                let (mut ct, _) = prefill_with_drafter(&mut engine, 0, &ids, &[], None, Some(d), None)?;
                let pp_s = tp.elapsed().as_secs_f64();
                let pp_tps = ids.len() as f64 / pp_s.max(1e-6);
                let mut gen = 0usize;
                let mut rounds = 0usize;
                let td = Instant::now();
                while gen < 512 {
                    let rem_kv = engine.max_ctx.saturating_sub(engine.pos as usize);
                    let draft = if let Some(ng) = ngram_lookup_draft(&engine.seq_tokens[0], ct, 128usize.min(rem_kv)) {
                        ng
                    } else {
                        d.draft(ct, &engine, k_len)?
                    };
                    let dk = draft.len();
                    let pos0 = engine.pos;
                    let preds = engine.verify_batch(&draft)?;
                    let mut n_acc = 1usize;
                    for j in 0..(dk - 1) {
                        if preds[j] == draft[j + 1] { n_acc += 1; } else { break; }
                    }
                    engine.rewind_verify_to(n_acc, dk)?;
                    engine.seq_pos[0] = engine.pos;
                    d.inject_batch_seq_async(0, &engine, 0, pos0, n_acc)?;
                    for j in 0..n_acc {
                        engine.seq_tokens[0].push(draft[j]);
                    }
                    ct = preds[n_acc - 1];
                    gen += n_acc;
                    rounds += 1;
                    if ct == 200006 || ct == 199999 { break; }
                }
                let dec_s = td.elapsed().as_secs_f64().max(1e-6);
                let dec_tps = gen as f64 / dec_s;
                println!(
                    "| Long-Ctx {:>6} tok | prefill {:>6.1} tok/s ({:.2}s) | decode {:>6.1} tok/s (emit={:.1}, {gen} tok) |",
                    ids.len(), pp_tps, pp_s, dec_tps, gen as f64 / rounds.max(1) as f64
                );
            }
        }
    }

    Ok(())
}
