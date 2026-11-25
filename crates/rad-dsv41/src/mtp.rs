pub fn ngram_draft(history: &[i32], anchor: i32, max_k: usize) -> Option<Vec<i32>> {
    let n = history.len();
    if n < 12 || max_k < 5 {
        return None;
    }
    let c = history[n - 1];
    let b = history[n - 2];
    let mut best_pos: Option<usize> = None;
    let mut best_len = 0usize;
    let mut i = n.saturating_sub(6);
    while i >= 2 {
        if history[i] == anchor && history[i - 1] == c && history[i - 2] == b {
            let mut mlen = 3usize;
            let lim = 32usize.min(i).min(n);
            while mlen < lim && history[i - mlen] == history[n - mlen] {
                mlen += 1;
            }
            if mlen > best_len {
                best_len = mlen;
                best_pos = Some(i);
                if mlen >= 16 {
                    break;
                }
            }
        }
        i -= 1;
    }
    let pos = best_pos?;
    let avail = (n - pos).min(max_k);
    if avail < 5 {
        return None;
    }
    let mut out = Vec::with_capacity(avail);
    out.push(anchor);
    out.extend_from_slice(&history[(pos + 1)..(pos + avail)]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_draft_matches_repeated_suffix() {
        let mut h: Vec<i32> = (1..20).collect();
        h.extend_from_slice(&[5, 6, 7, 8]);
        let d = ngram_draft(&h, 9, 8).expect("should match");
        assert_eq!(&d[..5], &[9, 10, 11, 12, 13]);
    }
}
