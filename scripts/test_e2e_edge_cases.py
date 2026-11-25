import argparse
import json
import time
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


def get_json(url, headers):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


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


def test_sse_client_disconnect_cancellation(base_url, headers):
    body = json.dumps({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "Write a 1500 word detailed essay about GPU architecture."}],
        "max_tokens": 1500,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    chunks = 0
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("data: "):
            chunks += 1
            if chunks >= 5:
                break
    t_close = time.perf_counter()
    resp.close()
    freed_ms = None
    for _ in range(40):
        time.sleep(0.05)
        st, slots = get_json(f"{base_url}/slots", headers)
        assert st == 200
        active = int(slots.get("active", 0)) + int(slots.get("prefilling", 0)) + int(slots.get("queued", 0))
        if active == 0:
            freed_ms = (time.perf_counter() - t_close) * 1000.0
            break
    assert freed_ms is not None and freed_ms < 2000.0, f"slot not freed promptly after client disconnect: {freed_ms}"
    print(f"[PASS] sse_client_disconnect_cancellation: slot freed in {freed_ms:.0f} ms after {chunks} chunks")


def test_streaming_stop_sequences(base_url, headers):
    body = json.dumps({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "Count from 1 to 10 separated by commas: 1, 2, 3, ..."}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stop": ["5"],
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body, headers=headers, method="POST")
    text = []
    finish_reason = None
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            ch = ev.get("choices", [])
            if ch:
                c = ch[0].get("delta", {}).get("content")
                if c:
                    text.append(c)
                if ch[0].get("finish_reason"):
                    finish_reason = ch[0]["finish_reason"]
    oai_txt = "".join(text)
    assert "5" not in oai_txt and finish_reason == "stop", f"OAI SSE stop failed: fr={finish_reason} txt={oai_txt!r}"

    anth_body = json.dumps({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "Count from 1 to 10 separated by commas: 1, 2, 3, ..."}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stop_sequences": ["5"],
        "stream": True,
    }).encode("utf-8")
    req2 = urllib.request.Request(f"{base_url}/v1/messages", data=anth_body, headers=headers, method="POST")
    anth_txt = []
    stop_reason = None
    with urllib.request.urlopen(req2, timeout=30) as resp2:
        for raw in resp2:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                anth_txt.append(ev["delta"]["text"])
            elif ev.get("type") == "message_delta":
                stop_reason = ev.get("delta", {}).get("stop_reason")
    atxt = "".join(anth_txt)
    assert "5" not in atxt and stop_reason == "stop_sequence", f"Anthropic SSE stop failed: sr={stop_reason} txt={atxt!r}"
    print(f"[PASS] streaming_stop_sequences: oai={oai_txt.strip()!r} anth={atxt.strip()!r}")


def test_sampling_top_k_min_p_seed(base_url, headers):
    p_det = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "What is 17 * 19? Reply with just the number."}],
        "max_tokens": 16,
        "temperature": 0.7,
        "top_k": 1,
        "seed": 4242,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    st1, d1 = post_json(f"{base_url}/v1/chat/completions", p_det, headers)
    st2, d2 = post_json(f"{base_url}/v1/chat/completions", p_det, headers)
    assert st1 == 200 and st2 == 200
    c1 = d1["choices"][0]["message"]["content"]
    c2 = d2["choices"][0]["message"]["content"]
    assert len(c1.strip()) > 0 and c1 == c2, f"top_k=1 mismatch: {c1!r} vs {c2!r}"

    p_smp = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": "Give one short creative sentence about oceans."}],
        "max_tokens": 24,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "seed": 4242,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    st3, d3 = post_json(f"{base_url}/v1/chat/completions", p_smp, headers)
    assert st3 == 200
    c3 = d3["choices"][0]["message"]["content"]
    assert len(c3.strip()) > 0, f"top_k/min_p empty: {d3}"
    print(f"[PASS] sampling_top_k_min_p_seed: top_k1={c1.strip()!r} sampled={c3.strip()!r}")


def test_mixed_concurrent_prefill_decode_ttft(base_url, headers):
    import threading
    long_done = {}
    def run_long():
        st, d = post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": "inkling-2-mini",
                "messages": [{"role": "user", "content": "Write a detailed Python class implementing a red-black tree with insert and search."}],
                "max_tokens": 256,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            headers,
        )
        long_done["status"] = st
    th = threading.Thread(target=run_long)
    th.start()
    time.sleep(0.15)
    t0 = time.perf_counter()
    st_s, d_s = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "inkling-2-mini",
            "messages": [{"role": "user", "content": "What is 7 * 8? Reply with just the number."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers,
    )
    short_ms = (time.perf_counter() - t0) * 1000.0
    th.join(timeout=60)
    assert st_s == 200 and long_done.get("status") == 200
    reply = d_s["choices"][0]["message"]["content"]
    assert "56" in reply, f"concurrent short reply wrong: {reply!r}"
    print(f"[PASS] mixed_concurrent_prefill_decode_ttft: short_e2e={short_ms:.1f} ms reply={reply.strip()!r}")


def test_tool_loop_breaker_and_post_loop_prose(base_url, headers):
    cmd = "python -c \"print('ok')\""
    tools = [{
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
    }]
    msgs = [{"role": "user", "content": "Write Chapter 2 of the dragon fantasy story."}]
    for i in range(3):
        msgs.append({
            "role": "assistant",
            "content": "Let me test python first.",
            "tool_calls": [{
                "id": f"call_loop_{i}",
                "type": "function",
                "function": {"name": "exec", "arguments": json.dumps({"command": cmd})},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"call_loop_{i}", "name": "exec", "content": "ok"})

    st1, d1 = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "inkling-2-mini",
            "messages": msgs,
            "tools": tools,
            "max_tokens": 128,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers,
    )
    assert st1 == 200, f"loop-break request failed: {st1} {d1}"
    m1 = d1["choices"][0]["message"]
    tcs1 = m1.get("tool_calls") or []
    dup = any(
        tc["function"]["name"] == "exec" and json.loads(tc["function"]["arguments"]).get("command") == cmd
        for tc in tcs1
    )
    assert not dup, f"duplicate tool call was not blocked: {tcs1}"

    msgs_post = list(msgs)
    msgs_post.append({
        "role": "assistant",
        "content": "I will deliver Chapter 2 directly in plain text below.",
    })
    msgs_post.append({
        "role": "user",
        "content": "Great, please write the opening two paragraphs of Chapter 2 now.",
    })
    st2, d2 = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "inkling-2-mini",
            "messages": msgs_post,
            "tools": tools,
            "max_tokens": 192,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers,
    )
    assert st2 == 200, f"post-loop prose failed: {st2} {d2}"
    c2 = (d2["choices"][0]["message"].get("content") or "").strip()
    assert len(c2) >= 60, f"post-loop prose too short or degenerate: {c2!r}"
    assert "\r\n\n  \r\n\n  \r\n\n" not in c2, f"degenerate whitespace loop detected: {c2!r}"
    print(f"[PASS] tool_loop_breaker_and_post_loop_prose: dup_blocked=True prose_len={len(c2)}")

    cmd_a = "python -c \"print('ok')\""
    cmd_b = "python -c \"import pathlib; print('check')\""
    msgs_alt = [{"role": "user", "content": "Write the story file now."}]
    for i, c in enumerate([cmd_a, cmd_b, cmd_a, cmd_b]):
        msgs_alt.append({
            "role": "assistant",
            "content": "Checking environment.",
            "tool_calls": [{
                "id": f"call_alt_{i}",
                "type": "function",
                "function": {"name": "exec", "arguments": json.dumps({"command": c})},
            }],
        })
        msgs_alt.append({"role": "tool", "tool_call_id": f"call_alt_{i}", "name": "exec", "content": "ok"})
    st3, d3 = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "inkling-2-mini",
            "messages": msgs_alt,
            "tools": tools,
            "max_tokens": 128,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers,
    )
    assert st3 == 200
    tcs3 = d3["choices"][0]["message"].get("tool_calls") or []
    alt_dup = any(
        tc["function"]["name"] == "exec" and json.loads(tc["function"]["arguments"]).get("command") in (cmd_a, cmd_b)
        for tc in tcs3
    )
    assert not alt_dup, f"alternating loop A/B was not blocked: {tcs3}"

    st4, d4 = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "inkling-2-mini",
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "tools": tools,
            "tool_choice": "none",
            "max_tokens": 64,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers,
    )
    assert st4 == 200
    assert not (d4["choices"][0]["message"].get("tool_calls") or []), f"tool_choice=none emitted tool call: {d4}"

    st5, d5 = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "inkling-2-mini",
            "messages": [{"role": "user", "content": "Print hello using python."}],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "exec"}},
            "max_tokens": 128,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers,
    )
    assert st5 == 200
    m5 = d5["choices"][0]["message"]
    tcs5 = m5.get("tool_calls") or []
    assert len(tcs5) >= 1 and tcs5[0]["function"]["name"] == "exec", f"forced tool_choice failed: {m5}"
    assert "[3" not in (m5.get("content") or ""), f"forced tool_choice emitted garbage: {m5}"
    print("[PASS] alternating_tool_loop_and_tool_choice_none_and_forced")


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
    test_sse_client_disconnect_cancellation(args.url, headers)
    test_streaming_stop_sequences(args.url, headers)
    test_sampling_top_k_min_p_seed(args.url, headers)
    test_mixed_concurrent_prefill_decode_ttft(args.url, headers)
    test_tool_loop_breaker_and_post_loop_prose(args.url, headers)
    print("ALL 12 EDGE-CASE E2E TESTS PASSED!")


if __name__ == "__main__":
    main()
