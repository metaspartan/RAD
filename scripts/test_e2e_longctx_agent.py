import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a relative file path inside the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file inside the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Run a shell command inside the project workspace and return stdout, stderr, and exit_code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]


def make_headers(api_key):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def stream_chat(url, payload, headers, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    first = None
    content_parts = []
    tool_calls = {}
    usage = {}
    finish_reason = None
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
            if not choices:
                continue
            c0 = choices[0]
            if c0.get("finish_reason"):
                finish_reason = c0["finish_reason"]
            delta = c0.get("delta") or {}
            txt = delta.get("content")
            if txt:
                if first is None:
                    first = time.perf_counter()
                content_parts.append(txt)
            for tc in delta.get("tool_calls") or []:
                if first is None:
                    first = time.perf_counter()
                idx = tc.get("index", 0)
                entry = tool_calls.setdefault(
                    idx,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    entry["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    entry["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn["arguments"]
    t1 = time.perf_counter()
    if first is None:
        first = t1
    tc_list = [tool_calls[k] for k in sorted(tool_calls.keys())]
    return {
        "t0": t0,
        "first": first,
        "t1": t1,
        "content": "".join(content_parts),
        "tool_calls": tc_list,
        "usage": usage,
        "finish_reason": finish_reason,
        "saw_done": saw_done,
    }


def load_corpus_chunks(repo_root):
    rel_files = [
        "crates/rad-core/src/model.rs",
        "kernels/rad_wmma.hip",
        "crates/rad-serve/src/main.rs",
        "kernels/rad_batch.hip",
        "kernels/rad_q8kv.hip",
        "kernels/rad_sudot4.hip",
        "crates/rad-core/src/dflash2.rs",
    ]
    pieces = []
    for rf in rel_files:
        p = os.path.join(repo_root, rf)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                pieces.append(f"=== FILE: {rf} ===\n" + f.read() + "\n")
    full = "\n".join(pieces)
    if len(full) < 1_200_000:
        reps = (1_200_000 // max(1, len(full))) + 2
        full = (full + "\n") * reps
    return full


def exec_tool(proj_dir, tc):
    name = tc["function"]["name"]
    args = json.loads(tc["function"]["arguments"])
    if name == "write_file":
        rel = args["path"].lstrip("/\\")
        dst = os.path.join(proj_dir, rel)
        os.makedirs(os.path.dirname(dst) or proj_dir, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return json.dumps({"ok": True, "path": rel, "bytes": len(args["content"])})
    if name == "read_file":
        rel = args["path"].lstrip("/\\")
        src = os.path.join(proj_dir, rel)
        with open(src, "r", encoding="utf-8") as f:
            txt = f.read()
        return json.dumps({"ok": True, "path": rel, "content": txt})
    if name == "run_cmd":
        p = subprocess.run(
            args["command"],
            shell=True,
            cwd=proj_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.dumps(
            {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        )
    return json.dumps({"ok": False, "error": f"unknown tool {name}"})


def append_assistant_and_tools(messages, proj_dir, res):
    msg = {"role": "assistant"}
    if res["content"]:
        msg["content"] = res["content"]
    else:
        msg["content"] = None
    if res["tool_calls"]:
        msg["tool_calls"] = res["tool_calls"]
    messages.append(msg)
    tool_results = []
    for tc in res["tool_calls"]:
        out = exec_tool(proj_dir, tc)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": out,
            }
        )
        tool_results.append((tc["function"]["name"], out))
    return tool_results


def report_turn(label, res):
    u = res["usage"]
    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", 0)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    uncached = max(0, pt - cached)
    pct = (100.0 * cached / pt) if pt > 0 else 0.0
    ttft_s = max(1e-6, res["first"] - res["t0"])
    dec_s = max(1e-6, res["t1"] - res["first"])
    pp_tps = uncached / ttft_s if uncached > 0 else 0.0
    dec_tps = ct / dec_s if ct > 0 else 0.0
    tcs = [tc["function"]["name"] for tc in res["tool_calls"]]
    print(
        f"[{label}] prompt={pt:,} cached={cached:,} ({pct:.1f}%) "
        f"new={uncached:,} ttft={ttft_s*1000:.1f}ms pp={pp_tps:.1f} tok/s "
        f"out={ct} dec={dec_tps:.1f} tok/s tools={tcs}"
    )
    return {
        "label": label,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "cache_pct": pct,
        "ttft_ms": ttft_s * 1000.0,
        "prefill_tps": pp_tps,
        "completion_tokens": ct,
        "decode_tps": dec_tps,
        "tools": tcs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--repo-root", default=os.path.expanduser("~/RAD"))
    ap.add_argument("--proj-dir", default="/tmp/rad_longctx_proj")
    ap.add_argument("--target-ctx", type=int, default=240000)
    args = ap.parse_args()

    if os.path.exists(args.proj_dir):
        shutil.rmtree(args.proj_dir)
    os.makedirs(args.proj_dir, exist_ok=True)

    corpus = load_corpus_chunks(args.repo_root)
    cursor = 0

    def take_chars(n_chars):
        nonlocal cursor
        s = corpus[cursor : cursor + n_chars]
        cursor += n_chars
        return s

    headers = make_headers(args.api_key)
    chat_url = f"{args.url}/v1/chat/completions"
    needle = "RAD_SENTINEL_ALPHA_94721"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous Python software engineer. Always use the provided tools "
                "(write_file, read_file, run_cmd) to build and verify code. Keep code clean and concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Project spec ID: {needle}.\n"
                "Step 1: Call write_file to create `store.py` implementing a class `KVStore` with "
                "methods `put(key: str, value: int)`, `get(key: str, default=None)`, `delete(key: str) -> bool`, "
                "and `keys_with_prefix(prefix: str) -> list[str]` (sorted). Use only Python stdlib."
            ),
        },
    ]

    stats = []

    def call_turn(label, max_tokens=1024):
        res = stream_chat(
            chat_url,
            {
                "model": "inkling-2-mini",
                "messages": messages,
                "tools": TOOLS,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers,
        )
        assert res["saw_done"], f"{label}: missing [DONE]"
        st = report_turn(label, res)
        stats.append(st)
        return res, st

    r1, s1 = call_turn("T1_cold_write_store")
    assert any(tc["function"]["name"] == "write_file" for tc in r1["tool_calls"]), f"T1 expected write_file: {r1}"
    append_assistant_and_tools(messages, args.proj_dir, r1)
    assert os.path.exists(os.path.join(args.proj_dir, "store.py"))

    messages.append(
        {
            "role": "user",
            "content": (
                "Reference codebase snapshot A (~30K tokens):\n"
                + take_chars(105_000)
                + "\n\nStep 2: Call write_file to create `analytics.py` that imports `KVStore` from `store` "
                "and implements `sum_prefix(store: KVStore, prefix: str) -> int` and "
                "`top_k(store: KVStore, k: int) -> list[tuple[str, int]]` sorted by (-value, key)."
            ),
        }
    )
    r2, s2 = call_turn("T2_32K_write_analytics")
    assert any(tc["function"]["name"] == "write_file" for tc in r2["tool_calls"]), f"T2 expected write_file: {r2}"
    append_assistant_and_tools(messages, args.proj_dir, r2)
    assert os.path.exists(os.path.join(args.proj_dir, "analytics.py"))

    messages.append(
        {
            "role": "user",
            "content": (
                "Reference codebase snapshot B (~32K tokens):\n"
                + take_chars(115_000)
                + "\n\nStep 3: Call write_file to create `test_proj.py` using `unittest` with at least 4 tests "
                "covering `KVStore` (put/get/delete/keys_with_prefix) and `analytics` (sum_prefix/top_k)."
            ),
        }
    )
    r3, s3 = call_turn("T3_64K_write_tests")
    assert any(tc["function"]["name"] == "write_file" for tc in r3["tool_calls"]), f"T3 expected write_file: {r3}"
    append_assistant_and_tools(messages, args.proj_dir, r3)
    assert os.path.exists(os.path.join(args.proj_dir, "test_proj.py"))

    messages.append(
        {
            "role": "user",
            "content": (
                "Reference codebase snapshot C (~64K tokens):\n"
                + take_chars(230_000)
                + "\n\nStep 4: Call run_cmd with `python3 -m unittest -v test_proj.py` to verify all tests pass."
            ),
        }
    )
    r4, s4 = call_turn("T4_128K_run_tests")
    assert any(tc["function"]["name"] == "run_cmd" for tc in r4["tool_calls"]), f"T4 expected run_cmd: {r4}"
    tr4 = append_assistant_and_tools(messages, args.proj_dir, r4)
    out4 = json.loads(tr4[0][1])
    assert out4["exit_code"] == 0, f"T4 unit tests failed: {out4}"

    cur_toks = s4["prompt_tokens"] + s4["completion_tokens"]
    t4_new = max(1, s4["prompt_tokens"] - s4["cached_tokens"])
    chars_per_tok = 230_000.0 / t4_new
    need_toks = max(10_000, args.target_ctx - cur_toks)
    chars_for_256k = int(need_toks * chars_per_tok * 0.96)
    messages.append(
        {
            "role": "user",
            "content": (
                "Reference codebase snapshot D (pushing context to ~240K-250K tokens):\n"
                + take_chars(chars_for_256k)
                + "\n\nStep 5: Call write_file to add `cli.py` that imports `KVStore` and `sum_prefix`, "
                "populates keys `a:1=10`, `a:2=25`, `b:1=7`, and prints `SUM_A=<sum>` and `SPEC=<project spec ID from Turn 1>`."
            ),
        }
    )
    r5, s5 = call_turn("T5_256K_write_cli")
    assert any(tc["function"]["name"] == "write_file" for tc in r5["tool_calls"]), f"T5 expected write_file: {r5}"
    append_assistant_and_tools(messages, args.proj_dir, r5)
    assert os.path.exists(os.path.join(args.proj_dir, "cli.py"))

    messages.append(
        {
            "role": "user",
            "content": "Step 6: Call run_cmd with `python3 cli.py` to verify the CLI output.",
        }
    )
    r6, s6 = call_turn("T6_256K_cached_run_cli", max_tokens=256)
    assert s6["cache_pct"] >= 98.0, f"T6 256K prefix cache hit too low: {s6['cache_pct']:.1f}%"
    assert any(tc["function"]["name"] == "run_cmd" for tc in r6["tool_calls"]), f"T6 expected run_cmd: {r6}"
    tr6 = append_assistant_and_tools(messages, args.proj_dir, r6)
    out6 = json.loads(tr6[0][1])
    assert out6["exit_code"] == 0, f"T6 cli.py failed: {out6}"
    assert "SUM_A=35" in out6["stdout"], f"T6 wrong SUM_A: {out6['stdout']}"
    assert needle in out6["stdout"], f"T6 needle recall failed at ~250K ctx: {out6['stdout']}"

    messages.append(
        {
            "role": "user",
            "content": "Step 7: Confirm in one short sentence that all tests and cli.py passed, quoting the exact Project spec ID.",
        }
    )
    r7, s7 = call_turn("T7_256K_cached_summary", max_tokens=128)
    assert s7["cache_pct"] >= 99.0, f"T7 256K prefix cache hit too low: {s7['cache_pct']:.1f}%"
    assert needle in r7["content"], f"T7 summary missing needle: {r7['content']}"

    print("\n=== REAL-WORLD 256K AGENTIC PROJECT E2E SUMMARY ===")
    for st in stats:
        print(
            f"  {st['label']:<24} ctx={st['prompt_tokens']:>7,} cache={st['cache_pct']:5.1f}% "
            f"ttft={st['ttft_ms']:8.1f}ms pp={st['prefill_tps']:6.1f} tok/s dec={st['decode_tps']:6.1f} tok/s"
        )
    print(f"CLI STDOUT: {out6['stdout'].strip()}")
    print(f"FINAL SUMMARY: {r7['content'].strip()}")
    print("ALL 7 LONG-CONTEXT AGENTIC PROJECT TURNS PASSED (UP TO ~250K CTX)!")


if __name__ == "__main__":
    main()
