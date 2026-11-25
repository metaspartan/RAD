import argparse
import concurrent.futures
import json
import os
import time
import urllib.request


def make_headers(api_key):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def stream_sse(url, payload, headers, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    first = None
    out = []
    usage = {}
    saw_done = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                saw_done = True
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                txt = (choices[0].get("delta") or {}).get("content")
                if txt:
                    if first is None:
                        first = time.perf_counter()
                    out.append(txt)
    t1 = time.perf_counter()
    if first is None:
        first = t1
    return {
        "t0": t0,
        "first": first,
        "t1": t1,
        "text": "".join(out),
        "usage": usage,
        "saw_done": saw_done,
    }


def load_context_block(repo_root, target_chars):
    files = [
        "crates/rad-core/src/model/mod.rs",
        "crates/rad-serve/src/scheduler.rs",
        "kernels/rocm/rad_q8kv_attn.hip",
    ]
    buf = []
    for rf in files:
        p = os.path.join(repo_root, rf)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                buf.append(f"=== {rf} ===\n" + f.read() + "\n")
    s = "\n".join(buf)
    while len(s) < target_chars + 4096:
        s = s + "\n" + s
    return s[:target_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--repo-root", default=os.path.expanduser("~/RAD"))
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--ctx-chars", type=int, default=42000)
    args = ap.parse_args()

    headers = make_headers(args.api_key)
    chat_url = f"{args.url}/v1/chat/completions"
    base_ctx = load_context_block(args.repo_root, args.ctx_chars)

    def run_turn1(u):
        msgs = [
            {"role": "system", "content": f"Session {u}: concise Rust/Python systems assistant."},
            {
                "role": "user",
                "content": (
                    f"Codebase context for user {u}:\n{base_ctx}\n\n"
                    f"Task: Write a standalone 120-line Python ring-buffer metrics collector for worker {u} "
                    "with p50/p95/p99 latency tracking and JSON export."
                ),
            },
        ]
        r = stream_sse(
            chat_url,
            {
                "model": "inkling-2-mini",
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers,
        )
        assert r["saw_done"], f"T1 U{u} missing [DONE]"
        return u, msgs, r

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as ex:
        t1_res = list(ex.map(run_turn1, range(args.users)))

    tot_ct1 = sum(r["usage"]["completion_tokens"] for _, _, r in t1_res)
    avg_pt1 = sum(r["usage"]["prompt_tokens"] for _, _, r in t1_res) / args.users
    dec_span1 = max(r["t1"] for _, _, r in t1_res) - min(r["first"] for _, _, r in t1_res)
    agg_dec1 = tot_ct1 / max(1e-6, dec_span1)
    print(
        f"[T1 B={args.users} @ {avg_pt1:,.0f} tok pure DFlash2 K=8] "
        f"ctoks={tot_ct1} span={dec_span1:.2f}s agg={agg_dec1:.1f} tok/s ({agg_dec1/args.users:.1f} tok/s/user)"
    )

    def run_turn2(item):
        u, msgs, r1 = item
        msgs2 = list(msgs) + [
            {"role": "assistant", "content": r1["text"]},
            {
                "role": "user",
                "content": (
                    "Now repeat your complete Python ring-buffer collector verbatim, adding a `reset()` method "
                    "that clears samples and resets counters to zero."
                ),
            },
        ]
        r2 = stream_sse(
            chat_url,
            {
                "model": "inkling-2-mini",
                "messages": msgs2,
                "temperature": 0.0,
                "max_tokens": 384,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers,
        )
        assert r2["saw_done"], f"T2 U{u} missing [DONE]"
        return u, r2

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as ex:
        t2_res = list(ex.map(run_turn2, t1_res))

    tot_ct2 = sum(r["usage"]["completion_tokens"] for _, r in t2_res)
    caches = [
        100.0 * (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) / max(1, r["usage"].get("prompt_tokens", 1))
        for _, r in t2_res
    ]
    ttfts2 = [(r["first"] - r["t0"]) * 1000.0 for _, r in t2_res]
    dec_span2 = max(r["t1"] for _, r in t2_res) - min(r["first"] for _, r in t2_res)
    agg_dec2 = tot_ct2 / max(1e-6, dec_span2)
    print(
        f"[T2 B={args.users} @ {avg_pt1:,.0f} tok cached Hybrid] "
        f"cache_avg={sum(caches)/len(caches):.1f}% ttft_avg={sum(ttfts2)/len(ttfts2):.1f}ms "
        f"ctoks={tot_ct2} span={dec_span2:.2f}s agg={agg_dec2:.1f} tok/s ({agg_dec2/args.users:.1f} tok/s/user)"
    )
    assert min(caches) >= 95.0, f"T2 multi-user 16K prefix cache too low: min={min(caches):.1f}%"
    print("MULTI-USER LONG-CONTEXT E2E TEST PASSED!")


if __name__ == "__main__":
    main()
