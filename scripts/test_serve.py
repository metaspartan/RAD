import base64
import hashlib
import json
import os
import socket
import struct
import time
import urllib.error
import urllib.request

BASE = os.environ.get("RAD_URL", "http://127.0.0.1:8088")
KEY = os.environ.get("RAD_API_KEY", "")
HDR = {"Content-Type": "application/json"}
if KEY:
    HDR["Authorization"] = f"Bearer {KEY}"

def req(method, path, body=None, auth=True):
    data = json.dumps(body).encode() if body is not None else None
    headers = dict(HDR) if auth else {"Content-Type": "application/json"}
    r = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def ws_test():
    sock = socket.create_connection(("127.0.0.1", 8088), timeout=60)
    ws_key = base64.b64encode(os.urandom(16)).decode()
    auth_hdr = f"Authorization: Bearer {KEY}\r\n" if KEY else ""
    hs = (
        f"GET /v1/responses HTTP/1.1\r\n"
        f"Host: 127.0.0.1:8088\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"{auth_hdr}"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(hs.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(4096)
    status_line = resp.split(b"\r\n", 1)[0].decode()
    assert "101" in status_line, f"WS handshake failed: {status_line}"

    def send_text(obj):
        payload = json.dumps(obj).encode()
        mask = os.urandom(4)
        hdr = bytearray([0x81])
        n = len(payload)
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr.extend(struct.pack("!H", n))
        else:
            hdr.append(0x80 | 127)
            hdr.extend(struct.pack("!Q", n))
        hdr.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        sock.sendall(bytes(hdr) + masked)

    buf = bytearray(resp.split(b"\r\n\r\n", 1)[1])
    def recv_exact(k):
        while len(buf) < k:
            chunk = sock.recv(4096)
            if not chunk:
                raise EOFError()
            buf.extend(chunk)
        out = bytes(buf[:k])
        del buf[:k]
        return out

    def recv_frame():
        b0, b1 = recv_exact(2)
        ln = b1 & 0x7F
        if ln == 126:
            ln = struct.unpack("!H", recv_exact(2))[0]
        elif ln == 127:
            ln = struct.unpack("!Q", recv_exact(8))[0]
        data = recv_exact(ln)
        return b0 & 0x0F, data.decode("utf-8", errors="replace")

    t0 = time.perf_counter()
    send_text({
        "type": "response.create",
        "response": {
            "model": "inkling-2-mini",
            "input": "Write a 3-line Python function to compute factorial(n).",
            "max_output_tokens": 96,
        },
    })
    ttft_ms = None
    deltas = []
    completed = None
    while True:
        op, txt = recv_frame()
        if op == 8:
            break
        ev = json.loads(txt)
        if ev.get("type") == "response.output_text.delta":
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            deltas.append(ev.get("delta", ""))
        elif ev.get("type") == "response.completed":
            completed = ev
            break
    sock.close()
    print(f"ws_responses: status=101 ttft={ttft_ms:.1f}ms deltas={len(deltas)} timings={completed['response']['timings']}")
    print(f"  ws_text: {completed['response']['output_text'][:120]!r}")

code, body = req("GET", "/health", auth=False)
print("health:", code, body)
code_unauth, _ = req("GET", "/v1/models", auth=False)
code_auth, body_models = req("GET", "/v1/models", auth=True)
print("auth_check:", code_unauth, code_auth, len(json.loads(body_models)["data"]))

c, b = req("POST", "/v1/chat/completions", {
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
    "max_tokens": 32,
})
j = json.loads(b)
print("chat_math:", c, repr(j["choices"][0]["message"]["content"]), j["timings"])

c, b = req("POST", "/v1/chat/completions", {
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write a complete Python LRU cache class using OrderedDict with get and put methods."}],
    "max_tokens": 128,
})
j = json.loads(b)
print("chat_code_spec:", c, j["timings"])
print("  code_preview:", repr(j["choices"][0]["message"]["content"][:120]))

sys_long = "You are a strict distributed systems reviewer. " + ("Rule: maintain linearizability and p99 < 10ms. " * 60)
c1, b1 = req("POST", "/v1/chat/completions", {
    "model": "inkling-2-mini",
    "messages": [{"role": "system", "content": sys_long}, {"role": "user", "content": "Say OK."}],
    "max_tokens": 16,
})
j1 = json.loads(b1)
c2, b2 = req("POST", "/v1/chat/completions", {
    "model": "inkling-2-mini",
    "messages": [{"role": "system", "content": sys_long}, {"role": "user", "content": "Say OK. Now list 2 risks."}],
    "max_tokens": 32,
})
j2 = json.loads(b2)
print("prefix_cache_1:", j1["timings"])
print("prefix_cache_2:", j2["timings"])

prompts8 = [
    "<|begin_of_text|><|message_user|><|content_text|>What is 12 + 34?<|end_message|><|message_model|><|content_thinking|><|content_text|>",
    "<|begin_of_text|><|message_user|><|content_text|>Name 3 planets.<|end_message|><|message_model|><|content_thinking|><|content_text|>",
    "<|begin_of_text|><|message_user|><|content_text|>Capital of France?<|end_message|><|message_model|><|content_thinking|><|content_text|>",
    "<|begin_of_text|><|message_user|><|content_text|>Hello world in Rust.<|end_message|><|message_model|><|content_thinking|><|content_text|>",
    "<|begin_of_text|><|message_user|><|content_text|>Explain TCP briefly.<|end_message|><|message_model|><|content_thinking|><|content_text|>",
    "<|begin_of_text|><|message_user|><|content_text|>What is 9 * 9?<|end_message|><|message_model|><|content_thinking|><|content_text|>",
    "<|begin_of_text|><|message_user|><|content_text|>Name 3 languages.<|end_message|><|message_model|><|content_thinking|><|content_text|>",
    "<|begin_of_text|><|message_user|><|content_text|>What color is the sky?<|end_message|><|message_model|><|content_thinking|><|content_text|>",
]
c, b = req("POST", "/v1/completions", {
    "model": "inkling-2-mini",
    "prompt": prompts8,
    "max_tokens": 48,
})
j = json.loads(b)
print("batch_completions_8:", c, j["timings"])
for ch in j["choices"][:3]:
    print(f"  batch[{ch['index']}]:", repr(ch["text"][:80]))

c, b = req("POST", "/v1/responses", {
    "model": "inkling-2-mini",
    "input": "Explain what a hash table is in one sentence.",
    "max_output_tokens": 64,
})
j = json.loads(b)
print("http_responses:", c, repr(j["output_text"][:100]), j["timings"])

ws_test()
print("ALL_TESTS_PASSED")
