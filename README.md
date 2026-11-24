# RAD — Inference Engine

**RAD** is a high-performance, from-scratch multimodal LLM inference engine written in **Rust** and **ROCm/HIP**, engineered for ultra-fast single-user speculative generation and high-throughput multi-user serving.

The current focus of RAD is **Inkling-2-Mini** (26B MoE, 48 looped visits, hybrid SWA + full attention, dynamic shortconv, rank-64 loop controllers, native **DFlash 2** & **DSpark** speculative decoding, and native **GPU Vision** via `mmproj-F16.gguf`) on **AMD Radeon AI PRO R9700 (RDNA4 `gfx1201`)**, with a modular crate architecture built to expand to additional hardware targets and model families.

---

## Highlights

- **Native `incoai/Inkling-2-Mini-DFlash2` Speculative Decoding (`285.4 tok/s` Single-User, `3,354.9 tok/s` `pp2048` Prefill)**
  - Zero-copy `safetensors` loader + 5-layer parallel block-diffusion drafter with `Q8_0` conv/selector projections, precomputed mask embeddings, 1 parallel `lm_head` pass, GPU top-16 candidate path selector, token-tiled 8-split Split-KV attention (`draft=2.24 ms` short ctx, `2.74 ms` at `2K ctx`), zero-spill compile-time `<NT>` `sudot4` kernels, and batched WMMA feature-injection (`verify=22.19 ms`, `inject=0.16 ms`, `avg_emit=7.13` on code; **`186.2 tok/s`** on math; **`129.5 tok/s`** at `2K ctx` with `verify=22.93 ms`).
  - Full `hipGraph`-captured draft & verify loops for both single-user (`B=1` `tg128 = 102.5 tok/s`, `tg128 @ d=2048 = 94.1 tok/s` via GQA 3:1 Split-KV FlashDecoding + fused single-pass `float4` state mixing & loop controllers) and multi-user (`B=4` at **`458.5 agg tok/s` / `114.6 tok/s/user`**, `B=8` up to **`638.1 agg tok/s`**), plus fallback support for **DSpark** (`dspark-Inkling-2-Mini-Q8_0.gguf`).
- **Native GPU Vision (`mmproj-F16.gguf`, Enabled by Default)**
  - Full 23-layer HIP ViT encoder (`640×640`, `20×20` patches, LayerScale `ls1`/`ls2`, bidirectional self-attention, `2×2` spatial merge, `inkling2v` MLP projector) outputting 256 multimodal embeddings directly into `InklingEngine` and `DFlash2Drafter`.
  - Works out of the box via OpenAI `image_url` (data URLs / local files) in `/v1/chat/completions` and `rad run --image <PATH>`.
- **RDNA4 (`gfx1201`) 8-Warp Coalesced LDS-Tiled Wave32 WMMA Matrix Core Suite (ROCm 7.x)**
  - Custom `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` 8-warp `128×64` & `64×64` `BK=4` coalesced LDS-tiled kernels for `Q4_0_SOA` × `Q8_1` GEMM (`qkvrg`, `wo`, `lm_head`, grouped MoE gate/up, and zero-atomic grouped MoE down with 6-way hot-expert token splitting), compiled with `-ffast-math -munsafe-fp-atomics`.
  - Custom `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` **FlashAttention-2** (bank-conflict-free LDS KV tile, register-preloaded `float4` relative-position projections, plus 4-warp Split-KV WMMA FlashAttention for `B=4..16`), fused multi-query `float4` state mixing (`mix_batch_f4_k`), register-cached `float4` RMSNorm, WMMA `sink_dyn_bf16`, single-pass `float4` & LDS WMMA loop controllers, and 64-CU `float4` router GEMM/GEMV.
- **On-GPU `Q4_0_SOA` Weight Repacking & `hipGraph` Execution**
  - Converts `Q8_0`, `NVFP4`, and `BF16` tensors directly on GPU at load time (**~1.16 s** cold load) into 16-byte-aligned structure-of-arrays blocks for coalesced `uint4` loads and `sudot4` / WMMA execution.
- **vLLM-Style Continuous Batching, Prefix Caching & Full OpenAI API (`rad`)**
  - Dedicated GPU scheduler thread with prefix-cache affinity and continuous batched decode (`step_multi` with GQA 3:1 attention) scaling up to `B=128` concurrent sequences (**`1,821.5 agg tok/s`**).
  - `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (text + vision, JSON + SSE), `POST /v1/completions` (single + batch), `POST /v1/responses` (HTTP + SSE), `GET /v1/responses` (WebSocket).

---

## Benchmarks (AMD Radeon AI PRO R9700, `gfx1201`, ROCm 7.14)

### Single-User Decode & Prefill (RAD vs. Stock `llama.cpp` & RTX PRO 6000 Blackwell)

![RAD vs Stock llama.cpp and RTX PRO 6000 Blackwell](assets/rad_benchmark.svg)

Measured end-to-end on the **same AMD Radeon AI PRO R9700 (32 GB, ~\$1,250)** (`Inkling-2-Mini-UD-Q4_K_XL` + `incoai/Inkling-2-Mini-DFlash2` / `dspark-Inkling-2-Mini-Q8_0.gguf`, `llama-bench` / vLLM-parity 3-run `mean ± stddev` via `rad bench`) — achieving **4.05× faster decode** and **4.43× faster `pp2048` prefill** than stock `llama.cpp`, and outperforming Unsloth's claimed **125 tok/s** on an **NVIDIA RTX PRO 6000 Blackwell (96 GB, ~\$8,500)** by **+128.3% (2.28×)**:

| Engine / Hardware | Decode (`tok/s`) | `pp2048` Prefill (`tok/s`) | Notes |
| --- | --- | --- | --- |
| **Stock `llama.cpp` (R9700 + DSpark)** | `70.5 tok/s` | `765.5 tok/s` | `-fa on -ctk q8_0 -ctv q8_0 --spec-type draft-dspark` |
| **RAD (R9700, No-Spec `hipGraph`)** | **`102.5 ± 0.6 tok/s`** (`1.45×`) | **`3,394.1 ± 1.1 tok/s`** (**`4.43×`**) | `tg128 @ d=2048 = 94.1 tok/s`; `pp512 = 2,934.2 ± 2.1 tok/s` |
| **Unsloth RTX PRO 6000 Blackwell** | `125.0 tok/s` *(claimed)* | — | 96 GB workstation GPU (~\$8,500) |
| **RAD (R9700 + DSpark `K=6`)** | **`148.2 tok/s`** (`2.10×`) | **`3,394.1 tok/s`** (**`4.43×`**) | `verify=23.7 ms`, `avg_emit=4.27` |
| **RAD (R9700 + DFlash 2 `K=8`)** | **`285.4 tok/s`** (**`4.05×`**) | **`3,354.9 ± 1.0 tok/s`** (**`4.38×`**) | **2.28× vs. RTX PRO 6000 Blackwell**, `draft=2.29 ms`, `verify=22.35 ms`, `2K ctx = 129.5 tok/s` |

### Multi-User Continuous Batching & Batched Speculative Decoding

![RAD Multi-User Scaling on Single R9700](assets/rad_multiuser.svg)

| Workload | Throughput / Latency | Notes |
| --- | --- | --- |
| **Cold Load (`max_seqs=16`, DFlash 2)** | **1.82 s** | On-GPU `Q4_0_SOA` + `Q8_0` conversion & `hipGraph` capture |
| **Short Prompt TTFT (`15 tok`)** | **34.6 ± 0.1 ms** | 3-run mean; 8-warp LDS WMMA + fused `float4` state mixing |
| **Prefix-Cached TTFT (`1,645 tok`, `1,639` cached)** | **32.5 ms** | Prefix-cache affinity hit |
| **Multi-User Decode (`B=4`, Batched DFlash 2 `K=4`)** | **458.5 agg tok/s** | **`114.6 tok/s/user`** (`ITL 8.7 ms`, zero-spill MoE + LDS WMMA) |
| **Multi-User Decode (`B=8`, no-spec)** | **515.7 agg tok/s** | `64.5 tok/s/user` (`ITL 15.5 ms`, `hipGraph` + fused `float4` mix) |
| **Multi-User Decode (`B=8`, Batched DSpark `K=4`)** | **458.6 agg tok/s** | `57.3 tok/s/user` |
| **Multi-User Decode (`B=8`, Batched DFlash 2 `K=4`)** | **638.1 agg tok/s** | **`79.8 tok/s/user`** (8 concurrent users each faster than single-user `llama.cpp`) |
| **Multi-User Decode (`B=16`, no-spec)** | **679.9 agg tok/s** | `42.5 tok/s/user` (`ITL 23.5 ms`) |
| **Multi-User Decode (`B=32`, no-spec)** | **848.3 agg tok/s** | `26.5 tok/s/user` (GQA 3:1 multi-user attention + `float4` RMSNorm) |
| **Multi-User Decode (`B=64`, no-spec)** | **1,162.2 agg tok/s** | `18.2 tok/s/user` (8-warp `128×64` LDS WMMA MoE) |
| **Multi-User Decode (`B=128`, no-spec)** | **1,821.5 agg tok/s** | **`14.2 tok/s/user`** (8-warp LDS WMMA + `float4` MoE slot reduction) |

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
│   ├── rad_dflash2.hip        DFlash 2 grouped dynamic conv, top-16 & path selector kernels
│   ├── rad_vision.hip         ViT FP16 GEMM, LayerNorm, GELU, LayerScale, bidirectional attn & 2x2 merge
│   ├── rad_sudot4.hip         Q4_0_SOA / Q8_1 / NVFP4 sudot4 & fused B=1 kernels
│   └── rad_wmma.hip           RDNA4 Wave32 WMMA INT8 GEMM, MoE, & FP16 FlashAttention-2
├── crates/
│   ├── rad-gguf/              Zero-copy mmap GGUF reader
│   ├── rad-hip/               Minimal ROCm/HIP FFI, RAII DevBuf, and CapturedGraph
│   ├── rad-core/              Model engine, DFlash 2 & DSpark drafters, VisionEncoder, safetensors reader
│   └── rad-serve/             `rad` CLI + Axum HTTP/SSE/WebSocket server with continuous batching
├── assets/
│   ├── rad_benchmark.svg      Single-user decode & prefill comparison chart
│   └── rad_multiuser.svg      Multi-user throughput scaling chart
└── scripts/
    └── test_serve.py          End-to-end HTTP, SSE, batch, and WebSocket test suite
```

---

## Quickstart

### Prerequisites

- Linux with **ROCm 6.x / 7.x** (`hipcc`, `libamdhip64`, `libhipblas`, `librocblas`)
- **Rust** toolchain (`cargo`, edition 2021)
- GGUF model weights + `tokenizer.json` (plus optional `incoai/Inkling-2-Mini-DFlash2`, `dspark-Inkling-2-Mini-Q8_0.gguf`, and `mmproj-F16.gguf`)

### 1. Build HIP Kernels

```bash
RAD_ARCH=gfx1201 bash kernels/build.sh
```

### 2. Build RAD

`crates/rad-core/build.rs` and `crates/rad-serve/build.rs` automatically link `kernels/librad_ops.so` with `-Wl,-rpath`, so no manual `RUSTFLAGS` or `LD_LIBRARY_PATH` is required:

```bash
cargo build --release
```

### 3. Run the CLI (`rad`)

`rad` automatically detects sibling `tokenizer.json`, **DFlash 2** (`../inkling2-dflash2/model.safetensors`, preferred over DSpark), **DSpark** (`dspark-Inkling-2-Mini-Q8_0.gguf`), and **Vision** (`mmproj-F16.gguf`, enabled by default):

```bash
./target/release/rad serve -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --port 8088

./target/release/rad run "Show me a Python quicksort function." -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf

./target/release/rad run "Describe this image." --image /path/to/photo.png -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf

./target/release/rad info -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf
```

### 4. Run Built-In Benchmarks (`rad bench`)

Includes Cold Load, TTFT, No-Spec & DFlash 2 Long Prefill, Single-User Code, Large-Context Code (2K ctx), Math, Prose, and Multi-User `B=4/8/16`:

```bash
./target/release/rad bench -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --ctx 4096 --max-seqs 16

python3 scripts/test_serve.py
```

### 5. Install & Deploy (Docker, `.deb`, Homebrew, Flatpak)

- **Docker with AMD ROCm GPU Passthrough:**
  ```bash
  RAD_MODELS_DIR=/path/to/models docker compose up -d
  ```
  Or directly:
  ```bash
  docker run --rm -p 8088:8088 \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined --ipc=host \
    -v /path/to/models:/models:ro \
    ghcr.io/metaspartan/rad:latest serve -m /models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --port 8088
  ```
- **Debian / Ubuntu (`.deb`):**
  Download `rad_<version>_amd64.deb` from GitHub Releases (built automatically by `.github/workflows/release.yml` on `v*` tags) or build locally:
  ```bash
  bash packaging/deb/build_deb.sh 0.1.0
  sudo dpkg -i dist/rad_0.1.0_amd64.deb
  ```
- **Homebrew (Linuxbrew):**
  ```bash
  brew install --build-from-source packaging/homebrew/rad.rb
  ```
- **Flatpak:**
  ```bash
  flatpak-builder --user --install --force-clean build-dir packaging/flatpak/io.github.metaspartan.RAD.yml
  ```

---

## Configuration

| Environment Variable | Default | Description |
| --- | --- | --- |
| `RAD_MODEL` | `./models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf` | Target model GGUF path |
| `RAD_TOKENIZER` | *(auto-detected)* | HuggingFace `tokenizer.json` path |
| `RAD_DFLASH2` | *(auto-detected)* | Optional DFlash 2 safetensors dir/file (preferred over DSpark) |
| `RAD_DSPARK` | *(auto-detected)* | Optional DSpark drafter GGUF path |
| `RAD_MMPROJ` | *(auto-detected)* | Optional vision `mmproj-F16.gguf` path |
| `RAD_SPEC` | `1` | Enable (`1`) or disable (`0`) speculative decoding |
| `RAD_VISION` | `1` | Enable (`1`) or disable (`0`) GPU vision encoder |
| `RAD_CTX` | `8192` | Per-sequence KV context length |
| `RAD_MAX_SEQS` | `32` | Maximum concurrent sequences in KV pool |
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

MIT — Copyright (c) 2026 Carsen Klock. See [LICENSE](LICENSE).
