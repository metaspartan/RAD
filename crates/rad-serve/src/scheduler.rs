use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use rad_core::{InklingEngine, VisionEncoder};
use tokio::sync::mpsc;
use crate::{grammar, metrics};
use crate::types::{ActiveSlot, AppState, Drafter, GenEvent, GenJob, PrefillSlot};

pub(crate) fn pick_best_slot(eng: &InklingEngine, free: &[bool], prompt: &[i32]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (s, &is_free) in free.iter().enumerate() {
        if !is_free {
            continue;
        }
        let cached = &eng.seq_tokens[s];
        let score = if !cached.is_empty() && prompt.len() > cached.len() && prompt.starts_with(cached) {
            cached.len()
        } else {
            0
        };
        if best.map(|(_, b)| score > b).unwrap_or(true) {
            best = Some((s, score));
        }
    }
    best.map(|(s, _)| s)
}

pub(crate) fn prefill_with_drafter(
    engine: &mut InklingEngine,
    seq: usize,
    prompt_ids: &[i32],
    images: &[String],
    image_tok_id: Option<i32>,
    mut drafter: Option<&mut Drafter>,
    mut vision: Option<&mut VisionEncoder>,
) -> anyhow::Result<(i32, usize)> {
    if !images.is_empty() {
        if let (Some(vis), Some(img_id)) = (vision.as_deref_mut(), image_tok_id) {
            engine.reset_seq(seq)?;
            if let Some(ref mut d) = drafter {
                d.reset_seq(seq);
            }
            let mut img_idx = 0usize;
            let mut seg_start = 0usize;
            let mut last_tok = 0i32;
            for idx in 0..prompt_ids.len() {
                if prompt_ids[idx] == img_id && img_idx < images.len() {
                    let seg = &prompt_ids[seg_start..idx];
                    if !seg.is_empty() {
                        let pos0 = engine.seq_pos[seq];
                        let _ = engine.prefill_tokens_seq(seq, seg)?;
                        if let Some(ref mut d) = drafter {
                            if seq < d.max_seqs() {
                                d.inject_batch_seq(seq, engine, 0, pos0, seg.len())?;
                            }
                        }
                    }
                    let embeds = vis.encode_data_url(&images[img_idx])?;
                    let n_vis = vis.out_tokens;
                    let pos0 = engine.seq_pos[seq];
                    last_tok = engine.prefill_embeds_seq(seq, &embeds, n_vis)?;
                    if let Some(ref mut d) = drafter {
                        if seq < d.max_seqs() {
                            d.inject_batch_seq(seq, engine, 0, pos0, n_vis)?;
                        }
                    }
                    img_idx += 1;
                    seg_start = idx + 1;
                }
            }
            if seg_start < prompt_ids.len() {
                let seg = &prompt_ids[seg_start..];
                let pos0 = engine.seq_pos[seq];
                last_tok = engine.prefill_tokens_seq(seq, seg)?;
                if let Some(ref mut d) = drafter {
                    if seq < d.max_seqs() {
                        d.inject_batch_seq(seq, engine, 0, pos0, seg.len())?;
                    }
                }
            }
            return Ok((last_tok, 0));
        }
    }
    match drafter {
        Some(Drafter::Dspark(d)) => engine.prefill_seq_dspark(seq, prompt_ids, Some(d)),
        Some(Drafter::DFlash2(d)) => {
            if seq < d.max_seqs {
                let cached = &engine.seq_tokens[seq];
                let mut common = 0usize;
                let lim = cached.len().min(prompt_ids.len().saturating_sub(1));
                while common < lim && cached[common] == prompt_ids[common] {
                    common += 1;
                }
                if common == 0 || common != cached.len() || d.seq_ctx_len[seq] as usize != common {
                    engine.reset_seq(seq)?;
                    d.reset_seq(seq);
                    common = 0;
                }
                let rem = &prompt_ids[common..];
                let total_len = prompt_ids.len();
                let inj_start = total_len.saturating_sub(d.swa_window);
                let mut last = 0i32;
                let mut i = 0usize;
                while i < rem.len() {
                    let chunk = &rem[i..(i + 1024).min(rem.len())];
                    let pos0 = engine.seq_pos[seq];
                    last = engine.prefill_tokens_seq(seq, chunk)?;
                    let chunk_start = common + i;
                    let chunk_end = chunk_start + chunk.len();
                    if chunk_end > inj_start {
                        let s0 = inj_start.saturating_sub(chunk_start);
                        let cnt = chunk.len() - s0;
                        let (fptr, mb) = engine.batch_dspark_feats_ptr();
                        d.inject_target_feats_batch_seq(seq, fptr, mb, s0, pos0 + s0 as i32, cnt)?;
                    } else {
                        d.advance_ctx_len_seq(seq, pos0 + chunk.len() as i32);
                    }
                    i += chunk.len();
                }
                Ok((last, common))
            } else {
                engine.prefill_seq_dspark(seq, prompt_ids, None)
            }
        }
        None => engine.prefill_seq_dspark(seq, prompt_ids, None),
    }
}

pub(crate) fn prefill_tail_bucket(n: usize) -> usize {
    if n <= 8 { 8 }
    else if n <= 12 { 12 }
    else if n <= 16 { 16 }
    else if n <= 24 { 24 }
    else if n <= 32 { 32 }
    else if n <= 48 { 48 }
    else if n <= 64 { 64 }
    else if n <= 96 { 96 }
    else { 128 }
}

pub(crate) fn ngram_bucket_len(avail: usize) -> usize {
    if avail >= 384 { 384 }
    else if avail >= 256 { 256 }
    else if avail >= 192 { 192 }
    else if avail >= 128 { 128 }
    else if avail >= 96 { 96 }
    else if avail >= 64 { 64 }
    else if avail >= 48 { 48 }
    else if avail >= 32 { 32 }
    else if avail >= 24 { 24 }
    else if avail >= 16 { 16 }
    else if avail >= 12 { 12 }
    else if avail >= 8 { 8 }
    else { 0 }
}

pub(crate) fn ngram_lookup_draft_ex(history: &[i32], anchor: i32, max_k: usize) -> Option<(Vec<i32>, usize)> {
    ngram_lookup_draft_cfg(history, anchor, max_k, false)
}

pub(crate) fn ngram_lookup_draft_cfg(history: &[i32], anchor: i32, max_k: usize, in_edit: bool) -> Option<(Vec<i32>, usize)> {
    let n = history.len();
    if n < 16 || max_k < 8 {
        return None;
    }
    let min_follow = 8usize;
    let c = history[n - 1];
    let b = history[n - 2];
    let mut best_pos: Option<usize> = None;
    let mut best_len: usize = 0;
    let mut best_avail: usize = 0;
    let mut i = n.saturating_sub(min_follow + 1);
    while i >= 2 {
        if history[i] == anchor && history[i - 1] == c && history[i - 2] == b {
            let mut mlen = 3usize;
            let max_m = 64usize.min(i).min(n);
            while mlen < max_m && history[i - mlen] == history[n - mlen] {
                mlen += 1;
            }
            let avail = (n - i).min(max_k);
            if mlen > best_len || (mlen == best_len && avail > best_avail) {
                best_len = mlen;
                best_pos = Some(i);
                best_avail = avail;
                if mlen >= 64 && avail >= max_k {
                    break;
                }
            }
        }
        i -= 1;
    }
    if let Some(pos) = best_pos {
        let cap_k = if in_edit {
            if best_len >= 5 {
                max_k
            } else {
                max_k.min(16)
            }
        } else if best_len >= 9 {
            max_k
        } else if best_len >= 7 {
            max_k.min(64)
        } else if best_len >= 5 {
            max_k.min(24)
        } else {
            max_k.min(8)
        };
        let take = ngram_bucket_len((n - pos).min(cap_k));
        if take >= min_follow {
            let mut out = Vec::with_capacity(take);
            out.push(anchor);
            out.extend_from_slice(&history[(pos + 1)..(pos + take)]);
            return Some((out, best_len));
        }
    }
    None
}

pub(crate) fn ngram_lookup_draft(history: &[i32], anchor: i32, max_k: usize) -> Option<Vec<i32>> {
    ngram_lookup_draft_ex(history, anchor, max_k).map(|(v, _)| v)
}

pub(crate) fn detect_degenerate_cycle(seq_tokens: &[i32], emitted: usize) -> bool {
    let gen_n = emitted.min(seq_tokens.len());
    rad_http::detect_degenerate_cycle(&seq_tokens[seq_tokens.len() - gen_n..])
}

pub(crate) fn sync_kv_metrics(engine: &InklingEngine, metrics: &metrics::Metrics) {
    let st = engine.paged_kv.stats();
    metrics.kv_blocks_total.store(st.total_blocks as u64, Ordering::Relaxed);
    metrics.kv_blocks_used.store(st.used_blocks as u64, Ordering::Relaxed);
    metrics.kv_blocks_shared.store(st.shared_blocks as u64, Ordering::Relaxed);
}

pub(crate) fn run_scheduler(
    mut engine: InklingEngine,
    mut drafter: Option<Drafter>,
    mut vision: Option<VisionEncoder>,
    image_tok_id: Option<i32>,
    vocab_grammar: Arc<grammar::VocabGrammarTable>,
    metrics: Arc<metrics::Metrics>,
    prefill_chunk_size: usize,
    mut rx: mpsc::UnboundedReceiver<GenJob>,
) {
    let max_seqs = engine.max_seqs;
    let mut free = vec![true; max_seqs];
    let mut active: Vec<ActiveSlot> = Vec::with_capacity(max_seqs);
    let mut prefilling: VecDeque<PrefillSlot> = VecDeque::new();
    let mut pending: VecDeque<GenJob> = VecDeque::new();
    let spec_k = drafter.as_ref().map(|d| d.spec_k()).unwrap_or(4);
    let mut catchup_streak = 0usize;
    sync_kv_metrics(&engine, &metrics);

    loop {
        if active.is_empty() && prefilling.is_empty() && pending.is_empty() {
            metrics.requests_active.store(0, Ordering::Relaxed);
            metrics.requests_queued.store(0, Ordering::Relaxed);
            metrics.requests_prefilling.store(0, Ordering::Relaxed);
            match rx.blocking_recv() {
                Some(job) => {
                    pending.push_back(job);
                    if max_seqs > 1 {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
                None => break,
            }
        }
        while let Ok(job) = rx.try_recv() {
            pending.push_back(job);
        }

        while !pending.is_empty() {
            let slot_opt = {
                let front = pending.front().unwrap();
                pick_best_slot(&engine, &free, &front.prompt_ids)
            };
            let Some(seq) = slot_opt else { break; };
            let job = pending.pop_front().unwrap();
            free[seq] = false;
            if !job.images.is_empty() && vision.is_some() && image_tok_id.is_some() {
                let tp = Instant::now();
                match prefill_with_drafter(&mut engine, seq, &job.prompt_ids, &job.images, image_tok_id, drafter.as_mut(), vision.as_mut()) {
                    Ok((mut first_tok, cached_n)) => {
                        let pp_s = tp.elapsed().as_secs_f64();
                        let pp_ms = pp_s * 1000.0;
                        metrics.prompt_tokens_total.fetch_add(job.prompt_ids.len() as u64, Ordering::Relaxed);
                        metrics.prefill_us_total.fetch_add((pp_s * 1e6) as u64, Ordering::Relaxed);
                        metrics.observe_ttft(pp_s);
                        let _ = engine.paged_kv.ensure_tokens(seq, engine.seq_pos[seq] as usize, &engine.seq_tokens[seq]);
                        sync_kv_metrics(&engine, &metrics);
                        let mut gram = if job.json_grammar { Some(grammar::JsonGrammarState::new_json()) } else { None };
                        let mut sampling = job.sampling;
                        if let Some(ref mut g) = gram {
                            if let Ok(logits) = engine.read_batch_logits(0) {
                                first_tok = vocab_grammar.select_best_token(&logits, g);
                            }
                        } else if !sampling.is_greedy() {
                            if let Ok(logits) = engine.read_batch_logits(0) {
                                first_tok = grammar::sample_from_logits(&logits, &engine.seq_tokens[seq], &mut sampling);
                            }
                        }
                        let _ = job.tx.send(GenEvent::Prefill { prompt_n: job.prompt_ids.len(), cached_n, pp_ms });
                        if job.max_tokens == 0 || first_tok == 200006 || first_tok == 199999 {
                            let _ = job.tx.send(GenEvent::Done { finish_reason: if job.max_tokens == 0 { "length" } else { "stop" }, pred_n: 0, dec_ms: 0.0 });
                            free[seq] = true;
                        } else {
                            let _ = job.tx.send(GenEvent::Token(first_tok as u32));
                            metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                            if job.max_tokens == 1 {
                                let _ = job.tx.send(GenEvent::Done { finish_reason: "length", pred_n: 1, dec_ms: 0.0 });
                                free[seq] = true;
                            } else {
                                let now = Instant::now();
                                active.push(ActiveSlot {
                                    seq, next_tok: first_tok, max_tokens: job.max_tokens, emitted: 1, last_emits: 0,
                                    enable_thinking: job.enable_thinking, grammar: gram, sampling,
                                    started_at: now, last_tok_at: now, tx: job.tx,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        let _ = job.tx.send(GenEvent::Error(format!("{e}")));
                        free[seq] = true;
                    }
                }
            } else {
                let mut common = 0usize;
                {
                    let cached = &engine.seq_tokens[seq];
                    let lim = cached.len().min(job.prompt_ids.len().saturating_sub(1));
                    while common < lim && cached[common] == job.prompt_ids[common] {
                        common += 1;
                    }
                }
                let drafter_matches = match drafter.as_ref() {
                    Some(d) if seq < d.max_seqs() => d.seq_ctx_len(seq) as usize == common,
                    _ => true,
                };
                if common == 0 || common != engine.seq_tokens[seq].len() || !drafter_matches {
                    let _ = engine.reset_seq(seq);
                    if let Some(ref mut d) = drafter {
                        if seq < d.max_seqs() {
                            d.reset_seq(seq);
                        }
                    }
                    common = 0;
                } else if common > 0 {
                    metrics.prefix_cache_hits_total.fetch_add(1, Ordering::Relaxed);
                    metrics.prefix_cache_tokens_saved_total.fetch_add(common as u64, Ordering::Relaxed);
                }
                prefilling.push_back(PrefillSlot {
                    seq,
                    job,
                    offset: common,
                    cached_n: common,
                    started_at: Instant::now(),
                });
            }
        }

        let mut pf_budget = if active.is_empty() { 16384 } else if prefilling.len() > active.len() { 8192 } else { prefill_chunk_size.max(64).min(1024) };
        let mut burst_ready: Vec<(PrefillSlot, i32, Option<grammar::JsonGrammarState>, grammar::SamplingParams)> = Vec::new();
        let mut pf_injected_any = false;
        while pf_budget > 0 {
            if prefilling.len() >= 2 && engine.multi_verify_cap >= 16 {
                let mut short_cnt = 0usize;
                let mut max_rem = 0usize;
                for pf in &prefilling {
                    let rem = pf.job.prompt_ids.len().saturating_sub(pf.offset);
                    if rem >= 1 && rem <= 128 {
                        let cand_max = max_rem.max(rem);
                        let cand_k = prefill_tail_bucket(cand_max);
                        if (short_cnt + 1) <= 16 && (short_cnt + 1) * cand_k <= engine.multi_verify_cap {
                            short_cnt += 1;
                            max_rem = cand_max;
                        }
                    }
                }
                if short_cnt >= 2 {
                    let k_pad = prefill_tail_bucket(max_rem);
                    let mut batch_pfs: Vec<PrefillSlot> = Vec::with_capacity(short_cnt);
                    let mut rest_pfs: VecDeque<PrefillSlot> = VecDeque::with_capacity(prefilling.len());
                    while let Some(pf) = prefilling.pop_front() {
                        let rem = pf.job.prompt_ids.len().saturating_sub(pf.offset);
                        if batch_pfs.len() < short_cnt && rem >= 1 && rem <= k_pad {
                            batch_pfs.push(pf);
                        } else {
                            rest_pfs.push_back(pf);
                        }
                    }
                    prefilling = rest_pfs;
                    let n_b = batch_pfs.len();
                    let seq_ids: Vec<i32> = batch_pfs.iter().map(|p| p.seq as i32).collect();
                    let pos_before: Vec<i32> = batch_pfs.iter().map(|p| engine.seq_pos[p.seq]).collect();
                    let mut actual_lens: Vec<usize> = Vec::with_capacity(n_b);
                    let mut padded_chunks: Vec<Vec<i32>> = Vec::with_capacity(n_b);
                    let mut tot_actual = 0usize;
                    for pf in &batch_pfs {
                        let slice = &pf.job.prompt_ids[pf.offset..];
                        actual_lens.push(slice.len());
                        tot_actual += slice.len();
                        let mut v = Vec::with_capacity(k_pad);
                        v.extend_from_slice(slice);
                        let pad = *slice.last().unwrap_or(&0);
                        v.resize(k_pad, pad);
                        padded_chunks.push(v);
                    }
                    pf_budget = pf_budget.saturating_sub(tot_actual);
                    match engine.verify_multi(&seq_ids, &padded_chunks) {
                        Ok(preds) => {
                            let any_samp = batch_pfs.iter().any(|p| !p.job.json_grammar && !p.job.sampling.is_greedy());
                            let topk_opt = if any_samp {
                                engine.read_batch_topk64(n_b * k_pad).ok()
                            } else {
                                None
                            };
                            let _ = engine.rewind_verify_multi(&seq_ids, &actual_lens, k_pad);
                            for (u, mut pf) in batch_pfs.into_iter().enumerate() {
                                let su = pf.seq;
                                let alen = actual_lens[u];
                                let end = pf.job.prompt_ids.len();
                                engine.seq_tokens[su].extend_from_slice(&pf.job.prompt_ids[pf.offset..end]);
                                if let Some(ref mut d) = drafter {
                                    if su < d.max_seqs() {
                                        match d {
                                            Drafter::DFlash2(df) => {
                                                let inj_start = end.saturating_sub(df.swa_window);
                                                if end > inj_start {
                                                    let s0 = inj_start.saturating_sub(pf.offset);
                                                    let cnt = alen - s0;
                                                    let (fptr, mb) = engine.batch_dspark_feats_ptr();
                                                    let _ = df.inject_target_feats_batch_seq_async(su, fptr, mb, u * k_pad + s0, pos_before[u] + s0 as i32, cnt);
                                                    pf_injected_any = true;
                                                } else {
                                                    df.advance_ctx_len_seq(su, pos_before[u] + alen as i32);
                                                }
                                            }
                                            Drafter::Dspark(_) => {
                                                let _ = d.inject_batch_seq(su, &engine, u * k_pad, pos_before[u], alen);
                                            }
                                        }
                                    }
                                }
                                let _ = engine.paged_kv.ensure_tokens(su, engine.seq_pos[su] as usize, &engine.seq_tokens[su]);
                                let mut last_tok = preds[u][alen - 1];
                                let mut gram = if pf.job.json_grammar { Some(grammar::JsonGrammarState::new_json()) } else { None };
                                let mut sampling = pf.job.sampling;
                                if let Some(ref mut g) = gram {
                                    if let Ok(logits) = engine.read_batch_logits(u * k_pad + alen - 1) {
                                        last_tok = vocab_grammar.select_best_token(&logits, g);
                                    }
                                } else if !sampling.is_greedy() {
                                    if let Some((ref tk_vals, ref tk_ids)) = topk_opt {
                                        let off = (u * k_pad + alen - 1) * 64;
                                        let seq_len = engine.seq_tokens[su].len();
                                        let recent = &engine.seq_tokens[su][seq_len.saturating_sub(128)..];
                                        last_tok = grammar::sample_from_topk(&tk_vals[off..off + 64], &tk_ids[off..off + 64], recent, &mut sampling);
                                    }
                                }
                                pf.offset = end;
                                burst_ready.push((pf, last_tok, gram, sampling));
                            }
                            sync_kv_metrics(&engine, &metrics);
                        }
                        Err(e) => {
                            let msg = format!("{e}");
                            for pf in batch_pfs {
                                let _ = pf.job.tx.send(GenEvent::Error(msg.clone()));
                                free[pf.seq] = true;
                            }
                        }
                    }
                    continue;
                }
            }
            let Some(mut pf) = prefilling.pop_front() else { break; };
            let end = (pf.offset + pf_budget.min(1024)).min(pf.job.prompt_ids.len());
            let chunk = &pf.job.prompt_ids[pf.offset..end];
            pf_budget = pf_budget.saturating_sub(chunk.len());
            let pos0 = engine.seq_pos[pf.seq];
            match engine.prefill_tokens_seq(pf.seq, chunk) {
                Ok(mut last_tok) => {
                    if let Some(ref mut d) = drafter {
                        if pf.seq < d.max_seqs() {
                            match d {
                                Drafter::DFlash2(df) => {
                                    let inj_start = pf.job.prompt_ids.len().saturating_sub(df.swa_window);
                                    if end > inj_start {
                                        let s0 = inj_start.saturating_sub(pf.offset);
                                        let cnt = chunk.len() - s0;
                                        let (fptr, mb) = engine.batch_dspark_feats_ptr();
                                        let _ = df.inject_target_feats_batch_seq_async(pf.seq, fptr, mb, s0, pos0 + s0 as i32, cnt);
                                        pf_injected_any = true;
                                    } else {
                                        df.advance_ctx_len_seq(pf.seq, pos0 + chunk.len() as i32);
                                    }
                                }
                                Drafter::Dspark(_) => {
                                    let _ = d.inject_batch_seq(pf.seq, &engine, 0, pos0, chunk.len());
                                }
                            }
                        }
                    }
                    let _ = engine.paged_kv.ensure_tokens(pf.seq, engine.seq_pos[pf.seq] as usize, &engine.seq_tokens[pf.seq]);
                    sync_kv_metrics(&engine, &metrics);
                    pf.offset = end;
                    if pf.offset < pf.job.prompt_ids.len() {
                        prefilling.push_back(pf);
                        if !active.is_empty() && prefilling.len() <= active.len() {
                            break;
                        }
                    } else {
                        let mut gram = if pf.job.json_grammar { Some(grammar::JsonGrammarState::new_json()) } else { None };
                        let mut sampling = pf.job.sampling;
                        if let Some(ref mut g) = gram {
                            if let Ok(logits) = engine.read_batch_logits(0) {
                                last_tok = vocab_grammar.select_best_token(&logits, g);
                            }
                        } else if !sampling.is_greedy() {
                            if let Ok((tk_vals, tk_ids)) = engine.read_batch_topk64(1) {
                                let seq_len = engine.seq_tokens[pf.seq].len();
                                let recent = &engine.seq_tokens[pf.seq][seq_len.saturating_sub(128)..];
                                last_tok = grammar::sample_from_topk(&tk_vals[..64], &tk_ids[..64], recent, &mut sampling);
                            }
                        }
                        burst_ready.push((pf, last_tok, gram, sampling));
                    }
                }
                Err(e) => {
                    let _ = pf.job.tx.send(GenEvent::Error(format!("{e}")));
                    free[pf.seq] = true;
                }
            }
        }
        if pf_injected_any {
            if let Some(ref mut d) = drafter {
                d.sync();
            }
        }
        for (pf, last_tok, gram, sampling) in burst_ready {
            let pp_s = pf.started_at.elapsed().as_secs_f64();
            let pp_ms = pp_s * 1000.0;
            metrics.prompt_tokens_total.fetch_add(pf.job.prompt_ids.len() as u64, Ordering::Relaxed);
            metrics.prefill_us_total.fetch_add((pp_s * 1e6) as u64, Ordering::Relaxed);
            metrics.observe_ttft(pp_s);
            let _ = pf.job.tx.send(GenEvent::Prefill {
                prompt_n: pf.job.prompt_ids.len(),
                cached_n: pf.cached_n,
                pp_ms,
            });
            if pf.job.max_tokens == 0 || last_tok == 200006 || last_tok == 199999 {
                let _ = pf.job.tx.send(GenEvent::Done {
                    finish_reason: if pf.job.max_tokens == 0 { "length" } else { "stop" },
                    pred_n: 0,
                    dec_ms: 0.0,
                });
                free[pf.seq] = true;
            } else {
                let _ = pf.job.tx.send(GenEvent::Token(last_tok as u32));
                metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                if pf.job.max_tokens == 1 {
                    let _ = pf.job.tx.send(GenEvent::Done { finish_reason: "length", pred_n: 1, dec_ms: 0.0 });
                    free[pf.seq] = true;
                } else {
                    let now = Instant::now();
                    active.push(ActiveSlot {
                        seq: pf.seq,
                        next_tok: last_tok,
                        max_tokens: pf.job.max_tokens,
                        emitted: 1,
                        last_emits: 0,
                        enable_thinking: pf.job.enable_thinking,
                        grammar: gram,
                        sampling,
                        started_at: now,
                        last_tok_at: now,
                        tx: pf.job.tx,
                    });
                }
            }
        }

        metrics.requests_active.store(active.len() as i64, Ordering::Relaxed);
        metrics.requests_queued.store(pending.len() as i64, Ordering::Relaxed);
        metrics.requests_prefilling.store(prefilling.len() as i64, Ordering::Relaxed);

        if active.is_empty() {
            continue;
        }

        if active.len() == 1 {
            if let Some(ref mut d) = drafter {
                let seq = active[0].seq;
                if seq < d.max_seqs() && d.seq_ctx_len(seq) == engine.seq_pos[seq] {
                    let mut slot = active.pop().unwrap();
                    let pos0 = engine.seq_pos[seq];
                    let t_step = Instant::now();
                    let needs_logits = slot.grammar.is_some() || !slot.sampling.is_greedy();
                    let res = (|| -> anyhow::Result<Vec<i32>> {
                        let rem_kv = engine.max_ctx.saturating_sub(pos0 as usize);
                        let in_edit = slot.last_emits >= 10;
                        let max_ng_k = if slot.last_emits >= 192 {
                            engine.multi_verify_cap.min(384).min(rem_kv)
                        } else if slot.last_emits >= 64 {
                            engine.multi_verify_cap.min(256).min(rem_kv)
                        } else {
                            128usize.min(rem_kv)
                        };
                        let draft = if let Some((ng, mlen)) = ngram_lookup_draft_cfg(&engine.seq_tokens[seq], slot.next_tok, max_ng_k, in_edit) {
                            if mlen >= 5 {
                                ng
                            } else if seq == 0 {
                                d.draft(slot.next_tok, &engine, spec_k)?
                            } else {
                                d.draft_seq(seq, slot.next_tok, &engine, spec_k)?
                            }
                        } else if seq == 0 {
                            d.draft(slot.next_tok, &engine, spec_k)?
                        } else {
                            d.draft_seq(seq, slot.next_tok, &engine, spec_k)?
                        };
                        let dk = draft.len();
                        let verified = engine.verify_batch_seq(seq, &draft)?;
                        let rem_emit = slot.max_tokens.saturating_sub(slot.emitted).max(1);
                        let acc_lim = dk.min(rem_emit);
                        let mut n_acc = 1usize;
                        let mut last_pred = verified[0];
                        if needs_logits {
                            if let Some(ref mut g) = slot.grammar {
                                for j in 0..acc_lim {
                                    let logits = engine.read_batch_logits(j)?;
                                    let pred = vocab_grammar.select_best_token(&logits, g);
                                    last_pred = pred;
                                    if pred == 200006 || pred == 199999 {
                                        break;
                                    }
                                    if j + 1 < acc_lim && pred == draft[j + 1] {
                                        n_acc += 1;
                                    } else {
                                        break;
                                    }
                                }
                            } else {
                                let (tk_vals, tk_ids) = engine.read_batch_topk64(dk)?;
                                let seq_len = engine.seq_tokens[seq].len();
                                let mut recent: Vec<i32> = engine.seq_tokens[seq][seq_len.saturating_sub(128)..].to_vec();
                                for j in 0..acc_lim {
                                    recent.push(draft[j]);
                                    let off = j * 64;
                                    let pred = grammar::sample_from_topk(
                                        &tk_vals[off..off + 64],
                                        &tk_ids[off..off + 64],
                                        &recent,
                                        &mut slot.sampling,
                                    );
                                    last_pred = pred;
                                    if pred == 200006 || pred == 199999 {
                                        break;
                                    }
                                    if j + 1 < acc_lim && pred == draft[j + 1] {
                                        n_acc += 1;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        } else {
                            for j in 0..(acc_lim - 1) {
                                if verified[j] == 200006 || verified[j] == 199999 {
                                    break;
                                }
                                if verified[j] == draft[j + 1] {
                                    n_acc += 1;
                                } else {
                                    break;
                                }
                            }
                            last_pred = verified[n_acc - 1];
                        }
                        engine.rewind_verify_seq_to(seq, n_acc, dk)?;
                        d.inject_batch_seq_async(seq, &engine, 0, pos0, n_acc)?;
                        for j in 0..n_acc {
                            engine.seq_tokens[seq].push(draft[j]);
                        }
                        let _ = engine.paged_kv.ensure_tokens(seq, engine.seq_pos[seq] as usize, &engine.seq_tokens[seq]);
                        let mut emitted_toks = Vec::with_capacity(n_acc);
                        for j in 1..n_acc {
                            emitted_toks.push(draft[j]);
                        }
                        emitted_toks.push(last_pred);
                        Ok(emitted_toks)
                    })();
                    let step_s = t_step.elapsed().as_secs_f64();
                    sync_kv_metrics(&engine, &metrics);
                    match res {
                        Ok(toks_out) => {
                            let n_out = toks_out.len();
                            metrics.spec_rounds_total.fetch_add(1, Ordering::Relaxed);
                            metrics.spec_accepted_tokens_total.fetch_add(n_out as u64, Ordering::Relaxed);
                            metrics.decode_us_total.fetch_add((step_s * 1e6) as u64, Ordering::Relaxed);
                            if n_out > 0 {
                                metrics.observe_itl(step_s / n_out as f64);
                            }
                            let mut finished = false;
                            let mut last_tok = slot.next_tok;
                            for nxt in toks_out {
                                let is_stop = nxt == 200006 || nxt == 199999;
                                if is_stop || slot.tx.is_closed() {
                                    let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                    let _ = slot.tx.send(GenEvent::Done {
                                        finish_reason: "stop",
                                        pred_n: slot.emitted,
                                        dec_ms,
                                    });
                                    free[slot.seq] = true;
                                    finished = true;
                                    break;
                                }
                                slot.emitted += 1;
                                metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                                let _ = slot.tx.send(GenEvent::Token(nxt as u32));
                                last_tok = nxt;
                                if slot.emitted >= slot.max_tokens {
                                    let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                    let _ = slot.tx.send(GenEvent::Done {
                                        finish_reason: "length",
                                        pred_n: slot.emitted,
                                        dec_ms,
                                    });
                                    free[slot.seq] = true;
                                    finished = true;
                                    break;
                                }
                            }
                            if !finished && detect_degenerate_cycle(&engine.seq_tokens[slot.seq], slot.emitted) {
                                let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                let _ = slot.tx.send(GenEvent::Done {
                                    finish_reason: "stop",
                                    pred_n: slot.emitted,
                                    dec_ms,
                                });
                                free[slot.seq] = true;
                                finished = true;
                            }
                            if !finished {
                                slot.next_tok = last_tok;
                                slot.last_emits = n_out;
                                slot.last_tok_at = Instant::now();
                                active.push(slot);
                            }
                        }
                        Err(e) => {
                            let _ = slot.tx.send(GenEvent::Error(format!("{e}")));
                            free[slot.seq] = true;
                        }
                    }
                    continue;
                }
            }
        }

        let any_grammar_only = active.iter().any(|a| a.grammar.is_some());
        if active.len() >= 1 && active.len() <= 16 && !any_grammar_only {
            let n_u = active.len();
            let max_user_k = (engine.multi_verify_cap / n_u).min(128);
            let mut ng_drafts: Option<Vec<Vec<i32>>> = None;
            if n_u >= 2 && max_user_k >= 8 {
                let mut cands: Vec<Option<(Vec<i32>, usize)>> = Vec::with_capacity(n_u);
                let mut hit_count = 0usize;
                let mut max_hit_len = 0usize;
                for a in &active {
                    let rem_kv = engine.max_ctx.saturating_sub(engine.seq_pos[a.seq] as usize);
                    let u_in_edit = a.last_emits >= 10;
                    if let Some((ng, mlen)) = ngram_lookup_draft_cfg(&engine.seq_tokens[a.seq], a.next_tok, max_user_k.min(rem_kv), u_in_edit) {
                        let req_mlen = if a.emitted >= 24 && a.last_emits > 0 && a.last_emits <= 16 { 9 } else { 5 };
                        if ng.len() >= 8 && mlen >= req_mlen {
                            hit_count += 1;
                            max_hit_len = max_hit_len.max(ng.len());
                            cands.push(Some((ng, mlen)));
                        } else if ng.len() >= 8 && a.last_emits == 0 {
                            hit_count += 1;
                            max_hit_len = max_hit_len.max(ng.len());
                            cands.push(Some((ng, mlen)));
                        } else {
                            cands.push(None);
                        }
                    } else {
                        cands.push(None);
                    }
                }
                if hit_count * 2 >= n_u {
                    let bk = ngram_bucket_len(max_hit_len);
                    if bk >= 8 {
                        let mut miss_idx: Vec<usize> = Vec::new();
                        let mut miss_seqs: Vec<i32> = Vec::new();
                        let mut miss_anchors: Vec<i32> = Vec::new();
                        let min_strong_ng = if bk >= 64 { 32 } else { 8 };
                        if let Some(ref d) = drafter {
                            for (idx, c) in cands.iter().enumerate() {
                                let su = active[idx].seq;
                                if !matches!(c, Some((ng, _)) if ng.len() >= min_strong_ng)
                                    && su < d.max_seqs()
                                    && d.seq_ctx_len(su) == engine.seq_pos[su]
                                {
                                    miss_idx.push(idx);
                                    miss_seqs.push(su as i32);
                                    miss_anchors.push(active[idx].next_tok);
                                }
                            }
                        }
                        if !miss_seqs.is_empty() && miss_seqs.len() <= 3 && bk >= 32 && catchup_streak < 12 && drafter.is_some() {
                            let cu_k = spec_k.min(max_user_k).max(4);
                            let any_cu_sampling = miss_idx.iter().any(|&mi| !active[mi].sampling.is_greedy());
                            let cu_pos_before: Vec<i32> = miss_seqs.iter().map(|&s| engine.seq_pos[s as usize]).collect();
                            let t_cu = Instant::now();
                            let cu_res = (|| -> anyhow::Result<Vec<Vec<i32>>> {
                                let d = drafter.as_mut().unwrap();
                                let drafts = if miss_seqs.len() == 1 {
                                    vec![d.draft_seq(miss_seqs[0] as usize, miss_anchors[0], &engine, cu_k)?]
                                } else {
                                    d.draft_multi(&miss_seqs, &miss_anchors, &engine, cu_k)?
                                };
                                let preds = if miss_seqs.len() == 1 {
                                    vec![engine.verify_batch_seq(miss_seqs[0] as usize, &drafts[0])?]
                                } else {
                                    engine.verify_multi(&miss_seqs, &drafts)?
                                };
                                let topk_opt = if any_cu_sampling {
                                    Some(engine.read_batch_topk64(miss_seqs.len() * cu_k)?)
                                } else {
                                    None
                                };
                                let mut emits = vec![0usize; miss_seqs.len()];
                                let mut outs = Vec::with_capacity(miss_seqs.len());
                                for (u, &mi) in miss_idx.iter().enumerate() {
                                    let su = miss_seqs[u] as usize;
                                    let rem_emit = active[mi].max_tokens.saturating_sub(active[mi].emitted).max(1);
                                    let acc_lim = cu_k.min(rem_emit);
                                    let mut n_acc = 1usize;
                                    let mut last_pred = preds[u][0];
                                    if let (Some((ref tk_vals, ref tk_ids)), true) = (&topk_opt, !active[mi].sampling.is_greedy()) {
                                        let seq_len = engine.seq_tokens[su].len();
                                        let mut recent: Vec<i32> = engine.seq_tokens[su][seq_len.saturating_sub(128)..].to_vec();
                                        for j in 0..acc_lim {
                                            recent.push(drafts[u][j]);
                                            let off = (u * cu_k + j) * 64;
                                            let pred = grammar::sample_from_topk(
                                                &tk_vals[off..off + 64],
                                                &tk_ids[off..off + 64],
                                                &recent,
                                                &mut active[mi].sampling,
                                            );
                                            last_pred = pred;
                                            if pred == 200006 || pred == 199999 {
                                                break;
                                            }
                                            if j + 1 < acc_lim && pred == drafts[u][j + 1] {
                                                n_acc += 1;
                                            } else {
                                                break;
                                            }
                                        }
                                    } else {
                                        for j in 0..(acc_lim - 1) {
                                            if preds[u][j] == 200006 || preds[u][j] == 199999 {
                                                break;
                                            }
                                            if preds[u][j] == drafts[u][j + 1] {
                                                n_acc += 1;
                                            } else {
                                                break;
                                            }
                                        }
                                        last_pred = preds[u][n_acc - 1];
                                    }
                                    emits[u] = n_acc;
                                    let mut tok_out = Vec::with_capacity(n_acc);
                                    for j in 1..n_acc {
                                        tok_out.push(drafts[u][j]);
                                    }
                                    tok_out.push(last_pred);
                                    outs.push(tok_out);
                                }
                                if miss_seqs.len() == 1 {
                                    engine.rewind_verify_seq_to(miss_seqs[0] as usize, emits[0], cu_k)?;
                                    d.inject_batch_seq_async(miss_seqs[0] as usize, &engine, 0, cu_pos_before[0], emits[0])?;
                                    for j in 0..emits[0] {
                                        engine.seq_tokens[miss_seqs[0] as usize].push(drafts[0][j]);
                                    }
                                    let _ = engine.paged_kv.ensure_tokens(miss_seqs[0] as usize, engine.seq_pos[miss_seqs[0] as usize] as usize, &engine.seq_tokens[miss_seqs[0] as usize]);
                                } else {
                                    engine.rewind_verify_multi(&miss_seqs, &emits, cu_k)?;
                                    for u in 0..miss_seqs.len() {
                                        let su = miss_seqs[u] as usize;
                                        d.inject_batch_seq_async(su, &engine, u * cu_k, cu_pos_before[u], emits[u])?;
                                        for j in 0..emits[u] {
                                            engine.seq_tokens[su].push(drafts[u][j]);
                                        }
                                        let _ = engine.paged_kv.ensure_tokens(su, engine.seq_pos[su] as usize, &engine.seq_tokens[su]);
                                    }
                                }
                                d.sync();
                                Ok(outs)
                            })();
                            let cu_s = t_cu.elapsed().as_secs_f64();
                            sync_kv_metrics(&engine, &metrics);
                            if let Ok(cu_outs) = cu_res {
                                let tot_out: usize = cu_outs.iter().map(|v| v.len()).sum();
                                metrics.spec_rounds_total.fetch_add(1, Ordering::Relaxed);
                                metrics.spec_accepted_tokens_total.fetch_add(tot_out as u64, Ordering::Relaxed);
                                metrics.decode_us_total.fetch_add((cu_s * 1e6) as u64, Ordering::Relaxed);
                                let mut finished_set = vec![false; active.len()];
                                for (u, &mi) in miss_idx.iter().enumerate() {
                                    let slot = &mut active[mi];
                                    let mut last_tok = slot.next_tok;
                                    for &nxt in &cu_outs[u] {
                                        let is_stop = nxt == 200006 || nxt == 199999;
                                        if is_stop || slot.tx.is_closed() {
                                            let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                            let _ = slot.tx.send(GenEvent::Done {
                                                finish_reason: "stop",
                                                pred_n: slot.emitted,
                                                dec_ms,
                                            });
                                            free[slot.seq] = true;
                                            finished_set[mi] = true;
                                            break;
                                        }
                                        slot.emitted += 1;
                                        metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                                        let _ = slot.tx.send(GenEvent::Token(nxt as u32));
                                        last_tok = nxt;
                                        if slot.emitted >= slot.max_tokens {
                                            let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                            let _ = slot.tx.send(GenEvent::Done {
                                                finish_reason: "length",
                                                pred_n: slot.emitted,
                                                dec_ms,
                                            });
                                            free[slot.seq] = true;
                                            finished_set[mi] = true;
                                            break;
                                        }
                                    }
                                    if !finished_set[mi] {
                                        slot.next_tok = last_tok;
                                        slot.last_emits = cu_outs[u].len();
                                    }
                                }
                                let mut keep = Vec::with_capacity(active.len());
                                for (idx, slot) in active.into_iter().enumerate() {
                                    if !finished_set[idx] {
                                        keep.push(slot);
                                    }
                                }
                                active = keep;
                                catchup_streak += 1;
                                continue;
                            }
                        }
                        catchup_streak = 0;
                        let df_k = spec_k.min(bk);
                        let mut miss_drafts: Vec<Vec<i32>> = if let Some(ref mut d) = drafter {
                            if miss_seqs.len() >= 2 {
                                d.draft_multi(&miss_seqs, &miss_anchors, &engine, df_k)
                                    .unwrap_or_else(|_| miss_anchors.iter().map(|&a| vec![a]).collect())
                            } else if miss_seqs.len() == 1 {
                                vec![d.draft_seq(miss_seqs[0] as usize, miss_anchors[0], &engine, df_k)
                                    .unwrap_or_else(|_| vec![miss_anchors[0]])]
                            } else {
                                Vec::new()
                            }
                        } else {
                            Vec::new()
                        };
                        let mut miss_cursor = 0usize;
                        let mut padded: Vec<Vec<i32>> = Vec::with_capacity(n_u);
                        for (idx, c) in cands.into_iter().enumerate() {
                            let mut v = match c {
                                Some((ng, _)) if ng.len() >= min_strong_ng => ng,
                                _ => {
                                    if miss_cursor < miss_idx.len() && miss_idx[miss_cursor] == idx {
                                        let dv = std::mem::take(&mut miss_drafts[miss_cursor]);
                                        miss_cursor += 1;
                                        dv
                                    } else {
                                        vec![active[idx].next_tok]
                                    }
                                }
                            };
                            if v.len() > bk {
                                v.truncate(bk);
                            } else if v.len() < bk {
                                let pad = *v.last().unwrap_or(&active[idx].next_tok);
                                v.resize(bk, pad);
                            }
                            padded.push(v);
                        }
                        ng_drafts = Some(padded);
                    }
                }
            }
            let use_drafter_multi = ng_drafts.is_none()
                && drafter.as_ref().map(|d| {
                    active.iter().all(|a| a.seq < d.max_seqs() && d.seq_ctx_len(a.seq) == engine.seq_pos[a.seq])
                }).unwrap_or(false);
            if ng_drafts.is_some() || use_drafter_multi {
                let multi_k = if let Some(ref ng) = ng_drafts {
                    ng[0].len()
                } else {
                    spec_k.min(max_user_k).max(4)
                };
                let any_sampling = active.iter().any(|a| !a.sampling.is_greedy());
                let seq_ids: Vec<i32> = active.iter().map(|a| a.seq as i32).collect();
                let anchors: Vec<i32> = active.iter().map(|a| a.next_tok).collect();
                let pos_before: Vec<i32> = active.iter().map(|a| engine.seq_pos[a.seq]).collect();
                let t_step = Instant::now();
                let res = (|| -> anyhow::Result<Vec<Vec<i32>>> {
                    let drafts = match ng_drafts {
                        Some(ng) => ng,
                        None => drafter.as_mut().unwrap().draft_multi(&seq_ids, &anchors, &engine, multi_k)?,
                    };
                    let preds = engine.verify_multi(&seq_ids, &drafts)?;
                    let topk_opt = if any_sampling {
                        Some(engine.read_batch_topk64(n_u * multi_k)?)
                    } else {
                        None
                    };
                    let mut emits = vec![0usize; active.len()];
                    let mut outs = Vec::with_capacity(active.len());
                    for u in 0..active.len() {
                        let su = seq_ids[u] as usize;
                        let rem_emit = active[u].max_tokens.saturating_sub(active[u].emitted).max(1);
                        let acc_lim = multi_k.min(rem_emit);
                        let mut n_acc = 1usize;
                        let mut last_pred = preds[u][0];
                        if let (Some((ref tk_vals, ref tk_ids)), true) = (&topk_opt, !active[u].sampling.is_greedy()) {
                            let seq_len = engine.seq_tokens[su].len();
                            let mut recent: Vec<i32> = engine.seq_tokens[su][seq_len.saturating_sub(128)..].to_vec();
                            for j in 0..acc_lim {
                                recent.push(drafts[u][j]);
                                let off = (u * multi_k + j) * 64;
                                let pred = grammar::sample_from_topk(
                                    &tk_vals[off..off + 64],
                                    &tk_ids[off..off + 64],
                                    &recent,
                                    &mut active[u].sampling,
                                );
                                last_pred = pred;
                                if pred == 200006 || pred == 199999 {
                                    break;
                                }
                                if j + 1 < acc_lim && pred == drafts[u][j + 1] {
                                    n_acc += 1;
                                } else {
                                    break;
                                }
                            }
                        } else {
                            for j in 0..(acc_lim - 1) {
                                if preds[u][j] == 200006 || preds[u][j] == 199999 {
                                    break;
                                }
                                if preds[u][j] == drafts[u][j + 1] {
                                    n_acc += 1;
                                } else {
                                    break;
                                }
                            }
                            last_pred = preds[u][n_acc - 1];
                        }
                        emits[u] = n_acc;
                        let mut tok_out = Vec::with_capacity(n_acc);
                        for j in 1..n_acc {
                            tok_out.push(drafts[u][j]);
                        }
                        tok_out.push(last_pred);
                        outs.push(tok_out);
                    }
                    engine.rewind_verify_multi(&seq_ids, &emits, multi_k)?;
                    let mut injected_any = false;
                    for u in 0..active.len() {
                        let su = seq_ids[u] as usize;
                        if let Some(ref mut d) = drafter {
                            if su < d.max_seqs() && d.seq_ctx_len(su) == pos_before[u] {
                                d.inject_batch_seq_async(su, &engine, u * multi_k, pos_before[u], emits[u])?;
                                injected_any = true;
                            }
                        }
                        for j in 0..emits[u] {
                            engine.seq_tokens[su].push(drafts[u][j]);
                        }
                        let _ = engine.paged_kv.ensure_tokens(su, engine.seq_pos[su] as usize, &engine.seq_tokens[su]);
                    }
                    if injected_any {
                        if let Some(ref mut d) = drafter {
                            d.sync();
                        }
                    }
                    Ok(outs)
                })();
                let step_s = t_step.elapsed().as_secs_f64();
                sync_kv_metrics(&engine, &metrics);
                match res {
                    Ok(all_outs) => {
                        let tot_out: usize = all_outs.iter().map(|v| v.len()).sum();
                        metrics.spec_rounds_total.fetch_add(1, Ordering::Relaxed);
                        metrics.spec_accepted_tokens_total.fetch_add(tot_out as u64, Ordering::Relaxed);
                        metrics.decode_us_total.fetch_add((step_s * 1e6) as u64, Ordering::Relaxed);
                        if tot_out > 0 {
                            metrics.observe_itl(step_s / (tot_out as f64 / active.len().max(1) as f64));
                        }
                        let mut keep = Vec::with_capacity(active.len());
                        for (mut slot, toks_out) in active.into_iter().zip(all_outs.into_iter()) {
                            let round_emits = toks_out.len();
                            let mut finished = false;
                            let mut last_tok = slot.next_tok;
                            for nxt in toks_out {
                                let is_stop = nxt == 200006 || nxt == 199999;
                                if is_stop || slot.tx.is_closed() {
                                    let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                    let _ = slot.tx.send(GenEvent::Done {
                                        finish_reason: "stop",
                                        pred_n: slot.emitted,
                                        dec_ms,
                                    });
                                    free[slot.seq] = true;
                                    finished = true;
                                    break;
                                }
                                slot.emitted += 1;
                                metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                                let _ = slot.tx.send(GenEvent::Token(nxt as u32));
                                last_tok = nxt;
                                if slot.emitted >= slot.max_tokens {
                                    let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                                    let _ = slot.tx.send(GenEvent::Done {
                                        finish_reason: "length",
                                        pred_n: slot.emitted,
                                        dec_ms,
                                    });
                                    free[slot.seq] = true;
                                    finished = true;
                                    break;
                                }
                            }
                            if !finished {
                                slot.next_tok = last_tok;
                                slot.last_emits = round_emits;
                                keep.push(slot);
                            }
                        }
                        active = keep;
                    }
                    Err(e) => {
                        let msg = format!("{e}");
                        for slot in active.drain(..) {
                            let _ = slot.tx.send(GenEvent::Error(msg.clone()));
                            free[slot.seq] = true;
                        }
                    }
                }
                continue;
            }
        }

        let seq_ids: Vec<i32> = active.iter().map(|a| a.seq as i32).collect();
        let toks: Vec<i32> = active.iter().map(|a| a.next_tok).collect();
        let any_step_sampling = active.iter().any(|a| a.grammar.is_none() && !a.sampling.is_greedy());
        let t_step = Instant::now();
        match engine.step_multi(&seq_ids, &toks) {
            Ok(nexts) => {
                let topk_opt = if any_step_sampling && active.len() > 1 {
                    engine.read_batch_topk64(active.len()).ok()
                } else {
                    None
                };
                let step_s = t_step.elapsed().as_secs_f64();
                metrics.decode_us_total.fetch_add((step_s * 1e6) as u64, Ordering::Relaxed);
                metrics.observe_itl(step_s);
                let mut keep = Vec::with_capacity(active.len());
                for (u, (mut slot, mut nxt)) in active.into_iter().zip(nexts.into_iter()).enumerate() {
                    if let Some(ref mut g) = slot.grammar {
                        if let Ok(logits) = engine.read_batch_logits(u) {
                            nxt = vocab_grammar.select_best_token(&logits, g);
                        }
                    } else if !slot.sampling.is_greedy() {
                        if let Some((ref tk_vals, ref tk_ids)) = topk_opt {
                            let off = u * 64;
                            let seq_len = engine.seq_tokens[slot.seq].len();
                            let recent = &engine.seq_tokens[slot.seq][seq_len.saturating_sub(128)..];
                            nxt = grammar::sample_from_topk(&tk_vals[off..off + 64], &tk_ids[off..off + 64], recent, &mut slot.sampling);
                        } else if let Ok(logits) = engine.read_batch_logits(u) {
                            nxt = grammar::sample_from_logits(&logits, &engine.seq_tokens[slot.seq], &mut slot.sampling);
                        }
                    }
                    let _ = engine.paged_kv.ensure_tokens(slot.seq, engine.seq_pos[slot.seq] as usize, &engine.seq_tokens[slot.seq]);
                    let is_stop = nxt == 200006 || nxt == 199999;
                    if is_stop || slot.tx.is_closed() {
                        let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                        let _ = slot.tx.send(GenEvent::Done {
                            finish_reason: "stop",
                            pred_n: slot.emitted,
                            dec_ms,
                        });
                        free[slot.seq] = true;
                    } else {
                        slot.emitted += 1;
                        metrics.generated_tokens_total.fetch_add(1, Ordering::Relaxed);
                        let _ = slot.tx.send(GenEvent::Token(nxt as u32));
                        if slot.emitted >= slot.max_tokens {
                            let dec_ms = slot.started_at.elapsed().as_secs_f64() * 1000.0;
                            let _ = slot.tx.send(GenEvent::Done {
                                finish_reason: "length",
                                pred_n: slot.emitted,
                                dec_ms,
                            });
                            free[slot.seq] = true;
                        } else {
                            slot.next_tok = nxt;
                            slot.last_tok_at = Instant::now();
                            keep.push(slot);
                        }
                    }
                }
                sync_kv_metrics(&engine, &metrics);
                active = keep;
            }
            Err(e) => {
                let msg = format!("{e}");
                for slot in active.drain(..) {
                    let _ = slot.tx.send(GenEvent::Error(msg.clone()));
                    free[slot.seq] = true;
                }
            }
        }
    }
}

pub(crate) async fn collect_job(
    st: &AppState,
    prompt_ids: Vec<i32>,
    images: Vec<String>,
    max_tokens: usize,
    #[allow(dead_code)]
    enable_thinking: bool,
    json_grammar: bool,
    sampling: grammar::SamplingParams,
) -> Result<(Vec<u32>, usize, usize, f64, f64, &'static str), String> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let in_flight = st
        .dispatch(GenJob { prompt_ids, images, max_tokens, enable_thinking, json_grammar, sampling, tx })
        .map_err(|_| "scheduler closed".to_string())?;
    let mut out_ids = Vec::with_capacity(max_tokens.min(4096));
    let mut prompt_n = 0usize;
    let mut cached_n = 0usize;
    let mut pp_ms = 0.0f64;
    let mut dec_ms = 0.0f64;
    let mut finish_reason = "stop";
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Prefill { prompt_n: pn, cached_n: cn, pp_ms: pm } => {
                prompt_n = pn;
                cached_n = cn;
                pp_ms = pm;
            }
            GenEvent::Token(t) => out_ids.push(t),
            GenEvent::Done { finish_reason: fr, dec_ms: dm, .. } => {
                finish_reason = fr;
                dec_ms = dm;
                break;
            }
            GenEvent::Error(e) => {
                in_flight.fetch_sub(1, Ordering::Relaxed);
                return Err(e);
            }
        }
    }
    in_flight.fetch_sub(1, Ordering::Relaxed);
    Ok((out_ids, prompt_n, cached_n, pp_ms, dec_ms, finish_reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_tail_and_ngram_buckets() {
        assert_eq!(prefill_tail_bucket(1), 8);
        assert_eq!(prefill_tail_bucket(9), 12);
        assert_eq!(prefill_tail_bucket(13), 16);
        assert_eq!(prefill_tail_bucket(17), 24);
        assert_eq!(prefill_tail_bucket(25), 32);
        assert_eq!(prefill_tail_bucket(64), 64);
        assert_eq!(prefill_tail_bucket(65), 96);
        assert_eq!(prefill_tail_bucket(97), 128);
        assert_eq!(ngram_bucket_len(10), 8);
        assert_eq!(ngram_bucket_len(64), 64);
        assert_eq!(ngram_bucket_len(200), 128);
    }

    #[test]
    fn ngram_lookup_draft_finds_longest_suffix() {
        let mut history = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        history.extend_from_slice(&[10, 20, 30, 40, 50]);
        let res = ngram_lookup_draft_ex(&history, 60, 16);
        assert!(res.is_some());
        let (draft, mlen) = res.unwrap();
        assert_eq!(mlen, 5);
        assert_eq!(&draft[..4], &[60, 70, 80, 90]);
    }

    #[test]
    fn degenerate_cycle_catches_whitespace_loop_without_false_positives() {
        let normal: Vec<i32> = (0..120).collect();
        assert!(!detect_degenerate_cycle(&normal, 120));
        let mut ws_loop = vec![10, 20, 30];
        for _ in 0..30 {
            ws_loop.extend_from_slice(&[220, 198, 220]);
        }
        assert!(detect_degenerate_cycle(&ws_loop, 90));
        assert!(!detect_degenerate_cycle(&ws_loop, 20));
    }
}
