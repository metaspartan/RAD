# RAD — Inference Engine

**RAD** is a high-performance, from-scratch LLM inference engine written in **Rust** and **ROCm/HIP**, engineered for low-latency single-user generation and high-throughput multi-user serving.

The current focus of RAD is **Inkling-2-Mini** (26B MoE, 48 looped visits, hybrid SWA + full attention, dynamic shortconv, rank-64 loop controllers, and DSpark speculative decoding) on **AMD Radeon AI PRO R9700 (RDNA4 `gfx1201`)**, with a modular crate architecture built to expand to additional hardware targets and model families.

---

## Highlights

- **RDNA4 (`gfx1201`) Wave32 WMMA Matrix Core Suite**
  - Custom `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` kernels for `Q4_0_SOA` × `Q8_1` GEMM (`qkvrg`, `wo`, `lm_head`, grouped MoE gate/up, and zero-atomic grouped MoE down).
  - Custom `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` **FlashAttention-2** (64 query tokens sharing an 8 KB LDS KV tile), WMMA FP16 loop controllers, and WMMA FP16 MoE router.
- **On-GPU `Q4_0_SOA` Weight Repacking**
  - Converts `Q8_0` and `NVFP4` tensors directly on GPU at load time (~10 ms, **~1.2 s** total cold load) into 16-byte-aligned structure-of-arrays blocks for coalesced `uint4` loads and `sudot4` / WMMA execution.
- **`hipGraph`-Captured Decode & DSpark Speculative Decoding**
  - Full 48-visit decode step and multi-step `K`-token DSpark draft loop captured into `hipGraph`s (zero mid-loop host syncs), paired with exact O(1) shortconv & ring-buffer KV rewind.
- **vLLM-Style Continuous Batching & Prefix Caching**
  - Dedicated GPU scheduler thread with prefix-cache affinity, automatic DSpark speculative decoding at `B=1`, and continuous batched decode (`step_multi`) up to `B=32+` concurrent sequences.
- **Full OpenAI-Compatible Serving API (`rad-serve`)**
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/chat/completions` (JSON + token-by-token SSE + `timings` + `cached_tokens`)
  - `POST /v1/completions` (single prompt and batch array)
  - `POST /v1/responses` (HTTP + SSE)
  - `GET /v1/responses` (**WebSocket** low-latency streaming)

---

## Benchmarks (AMD Radeon AI PRO R9700, `gfx1201`)

Measured end-to-end on **Inkling-2-Mini** (`Inkling-2-Mini-UD-Q4_K_XL-v2.gguf` + `dspark-Inkling-2-Mini-Q8_0.gguf`, coherence-verified):

| Workload | Throughput / Latency | Notes |
| --- | --- | --- |
| **Cold Load (`max_seqs=32`)** | **1.16 s** | On-GPU `Q4_0_SOA` conversion + `hipGraph` capture (`51.9 ms`) |
| **Long Prefill (`1,639 tok`)** | **1,643.9 tok/s** | WMMA INT8 + WMMA FlashAttention-2 (`MAX_BATCH=1024`) |
| **Prefix-Cached TTFT (`1,645 tok`, `1,639` cached)** | **32.9 ms** | Prefix-cache affinity hit |
| **Single-User Decode (`B=1`, no-spec)** | **93.2 tok/s** | `hipGraph`-captured 48-visit loop |
| **Single-User Decode (DSpark `K=6`, code)** | **138.2 tok/s** | `hipGraph` draft + verify (`emit=4.41 tok/round`) |
| **Multi-User Decode (`B=8`)** | **229.5 agg tok/s** | `28.7 tok/s/user` |
| **Multi-User Decode (`B=16`)** | **366.1 agg tok/s** | `22.9 tok/s/user` |
| **Multi-User Decode (`B=32`)** | **622.2 agg tok/s** | `19.4 tok/s/user` |

---

## Repository Structure

```text
RAD/
├── Cargo.toml                 Workspace manifest
├── kernels/
│   ├── build.sh               Builds librad_ops.so via hipcc
│   ├── rad_ops.hip            Core decode, RMSNorm, shortconv, attention, loop-ctrl kernels
│   ├── rad_batch.hip          Batched prefill & multi-user kernels
│   ├── rad_dspark.hip         DSpark speculative drafter kernels
│   ├── rad_sudot4.hip         Q4_0_SOA / Q8_1 / NVFP4 sudot4 & fused B=1 kernels
│   └── rad_wmma.hip           RDNA4 Wave32 WMMA INT8 GEMM, MoE, & FP16 FlashAttention-2
├── crates/
│   ├── rad-gguf/              Zero-copy mmap GGUF reader
│   ├── rad-hip/               Minimal ROCm/HIP FFI, RAII DevBuf, and CapturedGraph
│   ├── rad-core/              Model engine, DSpark drafter, and benchmark examples
│   └── rad-serve/             Axum HTTP/SSE/WebSocket server with continuous batching
└── scripts/
    └── test_serve.py          End-to-end HTTP, SSE, batch, and WebSocket test suite
```

---

## Quickstart

### Prerequisites

- Linux with **ROCm 6.x / 7.x** (`hipcc`, `libamdhip64`, `libhipblas`, `librocblas`)
- **Rust** toolchain (`cargo`, edition 2021)
- GGUF model weights + `tokenizer.json` (and optional DSpark draft GGUF)

### 1. Build HIP Kernels

```bash
RAD_ARCH=gfx1201 bash kernels/build.sh
```

### 2. Build RAD

`crates/rad-core/build.rs` automatically links `kernels/librad_ops.so` with `-Wl,-rpath`, so no manual `RUSTFLAGS` or `LD_LIBRARY_PATH` is required:

```bash
cargo build --release
```

### 3. Run the Server

```bash
RAD_MODEL=/path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf \
RAD_TOKENIZER=/path/to/tokenizer.json \
RAD_DSPARK=/path/to/dspark-Inkling-2-Mini-Q8_0.gguf \
RAD_PORT=8088 \
./target/release/rad-serve
```

### 4. Run Benchmarks & API Tests

```bash
RAD_MODEL=/path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf \
RAD_TOKENIZER=/path/to/tokenizer.json \
cargo run --release --example bench_all

RAD_MODEL=/path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf \
RAD_DSPARK=/path/to/dspark-Inkling-2-Mini-Q8_0.gguf \
RAD_TOKENIZER=/path/to/tokenizer.json \
cargo run --release --example bench_spec

python3 scripts/test_serve.py
```

---

## Configuration

| Environment Variable | Default | Description |
| --- | --- | --- |
| `RAD_MODEL` | `./models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf` | Target model GGUF path |
| `RAD_TOKENIZER` | `./models/tokenizer.json` | HuggingFace `tokenizer.json` path |
| `RAD_DSPARK` | `./models/dspark-Inkling-2-Mini-Q8_0.gguf` | Optional DSpark drafter GGUF path |
| `RAD_SPEC` | `1` | Enable (`1`) or disable (`0`) DSpark speculative decoding |
| `RAD_CTX` | `8192` | Per-sequence KV context length |
| `RAD_MAX_SEQS` | `16` | Maximum concurrent sequences in KV pool |
| `RAD_PORT` | `8088` | HTTP/WebSocket bind port |
| `RAD_API_KEY` | *(empty)* | Optional Bearer token (if empty, auth is disabled) |
| `RAD_ARCH` | `gfx1201` | Target GPU architecture for `kernels/build.sh` |

---

## Roadmap

- **Hardware backends:** CDNA (`gfx942` / MI300X), RDNA3 (`gfx1100`), NVIDIA CUDA, and Apple Metal backends behind a unified device trait.
- **Model families:** Standard GQA/MLA transformers, Qwen3-MoE, Llama, DeepSeek, and hybrid SSM/attention architectures.
- **Serving features:** Paged KV cache, chunked prefill, tool-calling grammar constraints, and Prometheus metrics.

---

## License

MIT — see [LICENSE](LICENSE).
