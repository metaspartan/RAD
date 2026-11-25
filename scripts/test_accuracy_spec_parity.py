import argparse
import difflib
import json
import subprocess
import time
import urllib.request


SAMPLE_FILE = """from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Task:
    id: int
    title: str
    done: bool = False

class TaskBoard:
    def __init__(self) -> None:
        self.tasks: Dict[int, Task] = {}
        self.next_id: int = 1

    def add(self, title: str) -> Task:
        t = Task(id=self.next_id, title=title)
        self.tasks[t.id] = t
        self.next_id += 1
        return t

    def complete(self, task_id: int) -> bool:
        if task_id not in self.tasks:
            return False
        self.tasks[task_id].done = True
        return True

    def pending(self) -> List[Task]:
        return [t for t in self.tasks.values() if not t.done]
"""

PROMPTS = [
    (
        "math_logic",
        "Compute (37 * 24) - (19 * 13) + 145 step by step and end with `ANSWER=<integer>`.",
        256,
        lambda txt: "ANSWER=786" in txt.replace(" ", ""),
    ),
    (
        "code_algo",
        "Write a concise Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
        "inside a single ```python code block with no extra commentary.",
        256,
        lambda txt: "def merge_intervals" in txt and "sort" in txt,
    ),
    (
        "hybrid_file_edit",
        f"Here is `board.py`:\n```python\n{SAMPLE_FILE}```\n\n"
        "Add a method `remove(self, task_id: int) -> bool` to `TaskBoard` that deletes and returns True if present else False. "
        "Output the complete updated `board.py` inside a single ```python code block.",
        512,
        lambda txt: "def remove" in txt and "def pending" in txt and "def complete" in txt,
    ),
]


def post_chat(url, api_key, prompt, max_tokens):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    txt = d["choices"][0]["message"]["content"]
    ct = d["usage"]["completion_tokens"]
    tps = d.get("timings", {}).get("predicted_per_second", ct / max(1e-6, dt))
    return txt, ct, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", default="")
    args = ap.parse_args()

    results = {}
    for name, prompt, max_toks, check_fn in PROMPTS:
        txt, ct, tps = post_chat(args.url, args.api_key, prompt, max_toks)
        ok = check_fn(txt)
        results[name] = {"text": txt, "tokens": ct, "tps": tps, "ok": ok}
        print(f"[{name}] ok={ok} tokens={ct} speed={tps:.1f} tok/s")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f)
        assert all(v["ok"] for v in results.values()), f"functional check failed: {results}"
        print(f"Saved baseline to {args.save}")
        return

    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            ref = json.load(f)
        all_ok = True
        for name, _, _, _ in PROMPTS:
            a = ref[name]
            b = results[name]
            exact = a["text"] == b["text"]
            ratio = difflib.SequenceMatcher(None, a["text"], b["text"]).ratio()
            speedup = a["tps"] / max(1e-6, b["tps"])
            print(
                f"[PARITY {name}] exact={exact} similarity={ratio*100:.2f}% "
                f"ok(spec={a['ok']}, nospec={b['ok']}) "
                f"spec={a['tps']:.1f} tok/s vs nospec={b['tps']:.1f} tok/s [{speedup:.2f}x]"
            )
            if not a["ok"] or not b["ok"] or ratio < 0.95:
                all_ok = False
        assert all_ok, "Spec vs No-Spec accuracy parity failed!"
        print("ALL DFLASH 2 HYBRID VS NO-SPEC ACCURACY PARITY CHECKS PASSED!")


if __name__ == "__main__":
    main()
