import argparse
import concurrent.futures
import json
import socket
import sys
import time
import urllib.request


def make_headers(api_key):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def post_json(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_sse(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    t0 = time.perf_counter()
    first = None
    content = []
    tcs = {}
    usage = {}
    saw_done = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            p = line[6:]
            if p == "[DONE]":
                saw_done = True
                break
            obj = json.loads(p)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            c = delta.get("content") or ""
            if c:
                if first is None:
                    first = time.perf_counter()
                content.append(c)
            for tc in delta.get("tool_calls") or []:
                if first is None:
                    first = time.perf_counter()
                idx = tc.get("index", 0)
                cur = tcs.setdefault(idx, {"id": tc.get("id", f"call_{idx}"), "type": "function", "function": {"name": "", "arguments": ""}})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    cur["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    cur["function"]["arguments"] += fn["arguments"]
    t1 = time.perf_counter()
    return {
        "t0": t0,
        "first": first or t0,
        "t1": t1,
        "content": "".join(content),
        "tool_calls": [tcs[k] for k in sorted(tcs)],
        "usage": usage,
        "saw_done": saw_done,
    }


def test_tool_call_reformatted_prefix_cache(base_url, headers):
    url = f"{base_url}/v1/chat/completions"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "exec",
                "description": "Run shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    big_file = "\n".join([f"def bucket_{i}(x: float) -> float:\n    return x * {i}.0 + 1.0\n" for i in range(1, 80)])
    msgs = [
        {"role": "system", "content": "You are an autonomous coding agent. Use tools cleanly."},
        {"role": "user", "content": f"Here is `metrics.py`:\n```python\n{big_file}\n```\nFirst, run `python3 --version` using the `exec` tool."},
    ]
    d1 = post_json(url, {"model": "inkling-2-mini", "messages": msgs, "tools": tools, "temperature": 0.0, "max_tokens": 128}, headers)
    m1 = d1["choices"][0]["message"]
    tcs = m1.get("tool_calls") or []
    assert len(tcs) >= 1, f"expected tool call on T1, got {m1}"
    tc = tcs[0]
    parsed = json.loads(tc["function"]["arguments"])
    tc["function"]["arguments"] = json.dumps(parsed, indent=2, sort_keys=True)
    msgs2 = msgs + [
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": "Python 3.12.3"},
        {"role": "user", "content": "Reply with just the Python version string."},
    ]
    t0 = time.perf_counter()
    d2 = post_json(url, {"model": "inkling-2-mini", "messages": msgs2, "tools": tools, "temperature": 0.0, "max_tokens": 64}, headers)
    ms2 = (time.perf_counter() - t0) * 1000.0
    u2 = d2["usage"]
    c2 = (u2.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    pct2 = 100.0 * c2 / max(1, u2["prompt_tokens"])
    assert pct2 >= 95.0, f"T2 prefix cache too low: {pct2:.1f}% ({c2}/{u2['prompt_tokens']})"
    r2 = d2["choices"][0]["message"].get("content") or ""
    assert "3.12.3" in r2, f"unexpected T2 reply: {r2!r}"
    msgs3 = msgs2 + [
        {"role": "assistant", "content": r2},
        {"role": "user", "content": "What major version is that?"},
    ]
    t0 = time.perf_counter()
    d3 = post_json(url, {"model": "inkling-2-mini", "messages": msgs3, "tools": tools, "temperature": 0.0, "max_tokens": 32}, headers)
    ms3 = (time.perf_counter() - t0) * 1000.0
    u3 = d3["usage"]
    c3 = (u3.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    pct3 = 100.0 * c3 / max(1, u3["prompt_tokens"])
    assert pct3 >= 95.0, f"T3 prefix cache too low: {pct3:.1f}% ({c3}/{u3['prompt_tokens']})"
    print(f"[PASS] tool_call_reformatted_prefix_cache: T2={pct2:.1f}% ({ms2:.1f}ms) T3={pct3:.1f}% ({ms3:.1f}ms)")


def test_json_grammar(base_url, headers):
    url = f"{base_url}/v1/chat/completions"
    d = post_json(
        url,
        {
            "model": "inkling-2-mini",
            "messages": [{"role": "user", "content": "Return a JSON object with keys language, fast (bool), and score (int)."}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 96,
        },
        headers,
    )
    txt = d["choices"][0]["message"].get("content") or ""
    obj = json.loads(txt)
    assert isinstance(obj, dict), f"expected JSON object, got {type(obj)}"
    print(f"[PASS] json_grammar: {obj}")


def test_midstream_disconnect_resilience(base_url, headers):
    host_port = base_url.replace("http://", "").replace("https://", "").split("/")[0]
    host, port = host_port.split(":")
    body = json.dumps({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "Write a 500-word essay on GPUs."}],
        "max_tokens": 512,
        "stream": True,
    }).encode()
    auth = f"Authorization: {headers['Authorization']}\r\n" if "Authorization" in headers else ""
    req = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"{auth}"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body
    s = socket.create_connection((host, int(port)), timeout=15)
    s.sendall(req)
    _ = s.recv(512)
    s.close()
    time.sleep(0.25)
    d = post_json(
        f"{base_url}/v1/chat/completions",
        {"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 16, "temperature": 0.0},
        headers,
    )
    assert d["choices"][0]["message"].get("content"), "server failed to recover after disconnect"
    print("[PASS] midstream_disconnect_resilience")


def test_multiuser_sse_hybrid(base_url, headers, n_users=8):
    url = f"{base_url}/v1/chat/completions"
    code = "\n".join(
        ["from dataclasses import dataclass", "from typing import Dict, List, Optional", ""]
        + [
            f'@dataclass\nclass WorkerPool{i}:\n    name: str = "pool_{i}"\n    max_concurrency: int = {i * 4}\n    timeout_ms: int = {100 + i * 25}\n    retries: int = 3\n\n    def budget(self) -> int:\n        return self.max_concurrency * self.timeout_ms\n'
            for i in range(1, 21)
        ]
    )

    def run_u(uid):
        prompt = f'Here is `service_{uid}.py`:\n```python\n{code}\n```\nChange `WorkerPool1.name` default to `"pool_primary_{uid}"` and output the COMPLETE updated `service_{uid}.py` verbatim inside a single ```python code block with no commentary.'
        r = stream_sse(
            url,
            {
                "model": "inkling-2-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1200,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers,
        )
        assert r["saw_done"], f"U{uid} missing [DONE]"
        assert f"pool_primary_{uid}" in r["content"], f"U{uid} missing edited token"
        ct = r["usage"].get("completion_tokens", 0)
        assert ct >= 900, f"U{uid} too few completion tokens: {ct}"
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_users) as ex:
        res = list(ex.map(run_u, range(1, n_users + 1)))
    tot_ct = sum(r["usage"]["completion_tokens"] for r in res)
    first_min = min(r["first"] for r in res)
    first_max = max(r["first"] for r in res)
    end_max = max(r["t1"] for r in res)
    full_agg = tot_ct / max(1e-6, end_max - first_min)
    ov_agg = tot_ct / max(1e-6, end_max - first_max)
    per_u = [r["usage"]["completion_tokens"] / max(1e-6, r["t1"] - r["first"]) for r in res]
    print(f"[PASS] multiuser_sse_hybrid (B={n_users}, temp=0.2): full={full_agg:.1f} tok/s overlap={ov_agg:.1f} tok/s per_user_avg={sum(per_u)/len(per_u):.1f} min={min(per_u):.1f} max={max(per_u):.1f}")


def test_multiuser_sse_pure_gen(base_url, headers, n_users=8):
    url = f"{base_url}/v1/chat/completions"
    topics = [
        "quicksort in Rust",
        "B-tree insertion in C++",
        "lock-free ring buffer in Rust",
        "HTTP/1.1 parser in Python",
        "Raft leader election overview",
        "SIMD dot product in C",
        "radix sort in Go",
        "bloom filter in TypeScript",
    ]

    def run_u(idx):
        r = stream_sse(
            url,
            {
                "model": "inkling-2-mini",
                "messages": [{"role": "user", "content": f"Write a concise, clean implementation of {topics[idx % len(topics)]}."}],
                "temperature": 0.2,
                "max_tokens": 256,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers,
        )
        assert r["saw_done"], f"pure_gen U{idx} missing [DONE]"
        assert r["usage"].get("completion_tokens", 0) >= 32
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_users) as ex:
        res = list(ex.map(run_u, range(n_users)))
    tot_ct = sum(r["usage"]["completion_tokens"] for r in res)
    first_min = min(r["first"] for r in res)
    end_max = max(r["t1"] for r in res)
    agg = tot_ct / max(1e-6, end_max - first_min)
    print(f"[PASS] multiuser_sse_pure_gen (B={n_users}, temp=0.2): ctoks={tot_ct} agg={agg:.1f} tok/s ({agg/n_users:.1f} tok/s/user)")


def test_metrics_endpoint(base_url, headers):
    req = urllib.request.Request(f"{base_url}/metrics", headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        txt = resp.read().decode("utf-8")
    for key in [
        "rad_prefix_cache_hits_total",
        "rad_prefix_cache_tokens_saved_total",
        "rad_prompt_tokens_total",
        "rad_spec_avg_emit",
    ]:
        assert key in txt, f"missing metric {key}"
    print("[PASS] metrics_endpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:6733")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--users", type=int, default=8)
    args = ap.parse_args()
    headers = make_headers(args.api_key)
    test_tool_call_reformatted_prefix_cache(args.url, headers)
    test_json_grammar(args.url, headers)
    test_midstream_disconnect_resilience(args.url, headers)
    test_multiuser_sse_pure_gen(args.url, headers, args.users)
    test_multiuser_sse_hybrid(args.url, headers, args.users)
    test_metrics_endpoint(args.url, headers)
    print("ALL REAL-WORLD E2E TESTS PASSED!")


if __name__ == "__main__":
    main()
