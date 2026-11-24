use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

const TTFT_BUCKETS: [f64; 9] = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];
const ITL_BUCKETS: [f64; 7] = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25];

pub struct Metrics {
    pub requests_ok: AtomicU64,
    pub requests_err: AtomicU64,
    pub requests_active: AtomicI64,
    pub requests_queued: AtomicI64,
    pub requests_prefilling: AtomicI64,
    pub prompt_tokens_total: AtomicU64,
    pub generated_tokens_total: AtomicU64,
    pub prefill_us_total: AtomicU64,
    pub decode_us_total: AtomicU64,
    pub spec_rounds_total: AtomicU64,
    pub spec_accepted_tokens_total: AtomicU64,
    pub prefix_cache_hits_total: AtomicU64,
    pub prefix_cache_tokens_saved_total: AtomicU64,
    pub kv_blocks_total: AtomicU64,
    pub kv_blocks_used: AtomicU64,
    pub kv_blocks_shared: AtomicU64,
    ttft_buckets: [AtomicU64; 10],
    ttft_sum_us: AtomicU64,
    ttft_count: AtomicU64,
    itl_buckets: [AtomicU64; 8],
    itl_sum_us: AtomicU64,
    itl_count: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_ok: AtomicU64::new(0),
            requests_err: AtomicU64::new(0),
            requests_active: AtomicI64::new(0),
            requests_queued: AtomicI64::new(0),
            requests_prefilling: AtomicI64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            generated_tokens_total: AtomicU64::new(0),
            prefill_us_total: AtomicU64::new(0),
            decode_us_total: AtomicU64::new(0),
            spec_rounds_total: AtomicU64::new(0),
            spec_accepted_tokens_total: AtomicU64::new(0),
            prefix_cache_hits_total: AtomicU64::new(0),
            prefix_cache_tokens_saved_total: AtomicU64::new(0),
            kv_blocks_total: AtomicU64::new(0),
            kv_blocks_used: AtomicU64::new(0),
            kv_blocks_shared: AtomicU64::new(0),
            ttft_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            ttft_sum_us: AtomicU64::new(0),
            ttft_count: AtomicU64::new(0),
            itl_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            itl_sum_us: AtomicU64::new(0),
            itl_count: AtomicU64::new(0),
        }
    }

    pub fn observe_ttft(&self, secs: f64) {
        for (i, &le) in TTFT_BUCKETS.iter().enumerate() {
            if secs <= le {
                self.ttft_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.ttft_buckets[TTFT_BUCKETS.len()].fetch_add(1, Ordering::Relaxed);
        self.ttft_sum_us.fetch_add((secs * 1e6) as u64, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_itl(&self, secs: f64) {
        for (i, &le) in ITL_BUCKETS.iter().enumerate() {
            if secs <= le {
                self.itl_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.itl_buckets[ITL_BUCKETS.len()].fetch_add(1, Ordering::Relaxed);
        self.itl_sum_us.fetch_add((secs * 1e6) as u64, Ordering::Relaxed);
        self.itl_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn render(&self) -> String {
        let mut s = String::with_capacity(4096);
        let ok = self.requests_ok.load(Ordering::Relaxed);
        let err = self.requests_err.load(Ordering::Relaxed);
        let act = self.requests_active.load(Ordering::Relaxed);
        let q = self.requests_queued.load(Ordering::Relaxed);
        let pf = self.requests_prefilling.load(Ordering::Relaxed);
        let ptok = self.prompt_tokens_total.load(Ordering::Relaxed);
        let gtok = self.generated_tokens_total.load(Ordering::Relaxed);
        let p_sec = self.prefill_us_total.load(Ordering::Relaxed) as f64 / 1e6;
        let d_sec = self.decode_us_total.load(Ordering::Relaxed) as f64 / 1e6;
        let sr = self.spec_rounds_total.load(Ordering::Relaxed);
        let sa = self.spec_accepted_tokens_total.load(Ordering::Relaxed);
        let acc_rate = if sr > 0 { sa as f64 / sr as f64 } else { 0.0 };
        let pch = self.prefix_cache_hits_total.load(Ordering::Relaxed);
        let pct = self.prefix_cache_tokens_saved_total.load(Ordering::Relaxed);
        let kvt = self.kv_blocks_total.load(Ordering::Relaxed);
        let kvu = self.kv_blocks_used.load(Ordering::Relaxed);
        let kvs = self.kv_blocks_shared.load(Ordering::Relaxed);
        let kv_ratio = if kvt > 0 { kvu as f64 / kvt as f64 } else { 0.0 };

        s.push_str(&format!("rad_requests_total{{status=\"ok\"}} {ok}\n"));
        s.push_str(&format!("rad_requests_total{{status=\"error\"}} {err}\n"));
        s.push_str(&format!("rad_requests_active {act}\n"));
        s.push_str(&format!("rad_requests_queued {q}\n"));
        s.push_str(&format!("rad_requests_prefilling {pf}\n"));
        s.push_str(&format!("rad_prompt_tokens_total {ptok}\n"));
        s.push_str(&format!("rad_generated_tokens_total {gtok}\n"));
        s.push_str(&format!("rad_prefill_seconds_total {p_sec:.6}\n"));
        s.push_str(&format!("rad_decode_seconds_total {d_sec:.6}\n"));
        s.push_str(&format!("rad_spec_rounds_total {sr}\n"));
        s.push_str(&format!("rad_spec_accepted_tokens_total {sa}\n"));
        s.push_str(&format!("rad_spec_avg_emit {acc_rate:.4}\n"));
        s.push_str(&format!("rad_prefix_cache_hits_total {pch}\n"));
        s.push_str(&format!("rad_prefix_cache_tokens_saved_total {pct}\n"));
        s.push_str(&format!("rad_kv_cache_blocks_total {kvt}\n"));
        s.push_str(&format!("rad_kv_cache_blocks_used {kvu}\n"));
        s.push_str(&format!("rad_kv_cache_blocks_shared {kvs}\n"));
        s.push_str(&format!("rad_kv_cache_usage_ratio {kv_ratio:.4}\n"));

        for (i, &le) in TTFT_BUCKETS.iter().enumerate() {
            let c = self.ttft_buckets[i].load(Ordering::Relaxed);
            s.push_str(&format!("rad_ttft_seconds_bucket{{le=\"{le}\"}} {c}\n"));
        }
        let t_inf = self.ttft_buckets[TTFT_BUCKETS.len()].load(Ordering::Relaxed);
        let t_sum = self.ttft_sum_us.load(Ordering::Relaxed) as f64 / 1e6;
        let t_cnt = self.ttft_count.load(Ordering::Relaxed);
        s.push_str(&format!("rad_ttft_seconds_bucket{{le=\"+Inf\"}} {t_inf}\n"));
        s.push_str(&format!("rad_ttft_seconds_sum {t_sum:.6}\n"));
        s.push_str(&format!("rad_ttft_seconds_count {t_cnt}\n"));

        for (i, &le) in ITL_BUCKETS.iter().enumerate() {
            let c = self.itl_buckets[i].load(Ordering::Relaxed);
            s.push_str(&format!("rad_itl_seconds_bucket{{le=\"{le}\"}} {c}\n"));
        }
        let i_inf = self.itl_buckets[ITL_BUCKETS.len()].load(Ordering::Relaxed);
        let i_sum = self.itl_sum_us.load(Ordering::Relaxed) as f64 / 1e6;
        let i_cnt = self.itl_count.load(Ordering::Relaxed);
        s.push_str(&format!("rad_itl_seconds_bucket{{le=\"+Inf\"}} {i_inf}\n"));
        s.push_str(&format!("rad_itl_seconds_sum {i_sum:.6}\n"));
        s.push_str(&format!("rad_itl_seconds_count {i_cnt}\n"));
        s
    }
}
