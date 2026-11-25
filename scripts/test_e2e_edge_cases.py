import argparse
import json
import urllib.error
import urllib.request


def make_headers(api_key, use_x_api_key=False):
    h = {"Content-Type": "application/json"}
    if api_key:
        if use_x_api_key:
            h["x-api-key"] = api_key
        else:
            h["Authorization"] = f"Bearer {api_key}"
    return h


def post_json(url, payload, headers, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_auth_enforcement(base_url, api_key):
    if not api_key:
        print("[SKIP] auth_enforcement (server has no api key)")
        return
    for path, body in [
        ("/v1/chat/completions", {"model": "inkling-2-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
        ("/v1/messages", {"model": "inkling-2-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
        ("/v1/responses", {"model": "inkling-2-mini", "input": "hi", "max_output_tokens": 4}),
        ("/tokenize", {"content": "hello"}),
    ]:
        bad_headers = {"Content-Type": "application/json", "Authorization": "Bearer wrong-key-xyz"}
        req = urllib.request.Request(f"{base_url}{path}", data=json.dumps(body).encode("utf-8"), headers=bad_headers, method="POST")
        got_401 = False
        try:
            urllib.request.urlopen(req, timeout=15)
        except urllib.error.HTTPError as e:
            got_401 = e.code == 401
        assert got_401, f"expected 401 on {path} with bad key"
    st, d = post_json(
        f"{base_url}/tokenize",
        {"content": "auth via x-api-key"},
        make_headers(api_key, use_x_api_key=True),
    )
    assert st == 200 and d.get("count", 0) >= 3
    print("[PASS] auth_enforcement (401 on bad key + x-api-key accepted)")


def test_ctx_overflow_graceful_400(base_url, headers):
    req = urllib.request.Request(f"{base_url}/health", headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        h = json.loads(resp.read().decode("utf-8"))
    max_ctx = h["max_ctx"]
    huge = "word " * (max_ctx + 512)
    req2 = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps({"model": "inkling-2-mini", "messages": [{"role": "user", "content": huge}], "max_tokens": 8}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    got_400 = False
    try:
        urllib.request.urlopen(req2, timeout=30)
    except urllib.error.HTTPError as e:
        got_400 = e.code == 400
    assert got_400, "expected 400 on prompt exceeding max_ctx"
    st, d = post_json(
        f"{base_url}/v1/chat/completions",
        {"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 8, "temperature": 0.0},
        headers,
    )
    assert st == 200 and d["choices"][0]["message"]["content"].strip()
    print(f"[PASS] ctx_overflow_graceful_400 (max_ctx={max_ctx} rejected cleanly + recovery ok)")


def test_max_tokens_edge_cases(base_url, headers):
    st1, d1 = post_json(
        f"{base_url}/v1/chat/completions",
        {"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Write a long poem."}], "max_tokens": 1, "temperature": 0.0},
        headers,
    )
    assert st1 == 200 and d1["usage"]["completion_tokens"] == 1
    assert d1["choices"][0]["finish_reason"] in ("length", "stop")
    st0, d0 = post_json(
        f"{base_url}/v1/chat/completions",
        {"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 0, "temperature": 0.0},
        headers,
    )
    assert st0 == 200 and d0["usage"]["completion_tokens"] == 0
    print("[PASS] max_tokens_edge_cases (max_tokens=1 and max_tokens=0)")


def test_stop_sequences(base_url, headers):
    st, d = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "inkling-2-mini",
            "messages": [{"role": "user", "content": "Count from 1 to 10 separated by commas: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
            "stop": ["5", "6"],
            "max_tokens": 64,
            "temperature": 0.0,
        },
        headers,
    )
    assert st == 200
    txt = d["choices"][0]["message"]["content"]
    assert "5" not in txt and "6" not in txt, f"stop sequence leaked into output: {txt!r}"
    print(f"[PASS] stop_sequences: stopped cleanly at {txt!r}")


def test_unicode_emoji_cjk_sse(base_url, headers):
    payload = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "Reply with this exact line: Hello 世界 🚀 café"}],
        "max_tokens": 48,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    parts = []
    saw_done = False
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line.startswith("data:"):
                body = line[5:].strip()
                if body == "[DONE]":
                    saw_done = True
                    break
                obj = json.loads(body)
                c = obj.get("choices") or []
                if c:
                    t = (c[0].get("delta") or {}).get("content")
                    if t:
                        parts.append(t)
    full = "".join(parts)
    assert saw_done and "\ufffd" not in full and "世界" in full, f"unicode SSE issue: {full!r}"
    print(f"[PASS] unicode_emoji_cjk_sse: {full.strip()!r}")


def test_anthropic_multiturn_tools_prefix_cache(base_url, headers):
    sys_prompt = "You are an autonomous agent with tools. Always invoke tools directly instead of describing what you will do. " + ("Prefer exact tool calls. " * 140)
    tools = [
        {
            "name": "get_metric",
            "description": "Fetch a named system metric.",
            "input_schema": {
                "type": "object",
                "properties": {"metric": {"type": "string"}},
                "required": ["metric"],
            },
        }
    ]
    st1, d1 = post_json(
        f"{base_url}/v1/messages",
        {
            "model": "inkling-2-mini",
            "max_tokens": 128,
            "system": sys_prompt,
            "tools": tools,
            "messages": [{"role": "user", "content": "What is the current gpu_temp? Use get_metric."}],
            "temperature": 0.0,
        },
        headers,
    )
    assert st1 == 200
    tuses = [b for b in d1.get("content", []) if b.get("type") == "tool_use"]
    assert tuses and tuses[0]["name"] == "get_metric", f"expected tool_use: {d1}"
    tid = tuses[0]["id"]

    st2, d2 = post_json(
        f"{base_url}/v1/messages",
        {
            "model": "inkling-2-mini",
            "max_tokens": 64,
            "system": sys_prompt,
            "tools": tools,
            "messages": [
                {"role": "user", "content": "What is the current gpu_temp? Use get_metric."},
                {"role": "assistant", "content": d1["content"]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": "{\"gpu_temp\": 62}"}]},
            ],
            "temperature": 0.0,
        },
        headers,
    )
    assert st2 == 200
    u2 = d2["usage"]
    cached = u2.get("cache_read_input_tokens", 0)
    total_in = cached + u2.get("input_tokens", 0)
    pct = 100.0 * cached / max(1, total_in)
    assert pct >= 90.0, f"Anthropic T2 prefix cache too low: {pct:.1f}% ({cached}/{total_in})"
    txt2 = "".join(b.get("text", "") for b in d2["content"] if b.get("type") == "text")
    assert "62" in txt2, f"Anthropic T2 missed tool_result value: {txt2!r}"
    print(f"[PASS] anthropic_multiturn_tools_prefix_cache: cache={pct:.1f}% ({cached}/{total_in}) reply={txt2.strip()!r}")


def test_batch_completions_endpoint(base_url, headers):
    st, d = post_json(
        f"{base_url}/v1/completions",
        {"model": "inkling-2-mini", "prompt": ["2 + 2 =", "3 + 3 ="], "max_tokens": 12, "temperature": 0.0},
        headers,
    )
    assert st == 200 and len(d.get("choices", [])) == 2
    print("[PASS] batch_completions_endpoint (2 prompts batched)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()
    headers = make_headers(args.api_key)
    test_auth_enforcement(args.url, args.api_key)
    test_ctx_overflow_graceful_400(args.url, headers)
    test_max_tokens_edge_cases(args.url, headers)
    test_stop_sequences(args.url, headers)
    test_unicode_emoji_cjk_sse(args.url, headers)
    test_anthropic_multiturn_tools_prefix_cache(args.url, headers)
    test_batch_completions_endpoint(args.url, headers)
    print("ALL 7 EDGE-CASE E2E TESTS PASSED!")


if __name__ == "__main__":
    main()
