# RAD — Inference Engine

**RAD** is a high-performance, from-scratch multimodal LLM inference engine written in **Rust** and **ROCm/HIP**, engineered for ultra-fast single-user speculative generation, 256K long-context inference, and high-throughput multi-user serving on **AMD Radeon AI PRO R9700 (RDNA4 `gfx1201`)**.

RAD natively runs **Inkling-2-Mini** (26B MoE, 48 looped visits, hybrid SWA + full attention, dynamic shortconv, rank-64 loop controllers, native **DFlash 2** & **DSpark** speculative decoding, and native **GPU Vision** via `mmproj-F16.gguf`) with zero external inference runtime dependencies.

---

## Highlights

- **256K Long-Context Speculative Frontier (`546.7 tok/s @ 32K`, `521.4 tok/s @ 64K`, `284.9 tok/s @ 128K`, `245.1 tok/s @ 256K` on Single 32 GB R9700)**
  - **INT8 Full-Attention KV Cache (`kernels/rad_q8kv.hip`):** Per-head-token `int8_t` + FP16 scale KV cache for all 12 full-attention layers across `B=1`, `verify_b8..48`, prefill, and multi-user — fitting full **256K context (`262,144` tokens)** on a single 32 GB R9700 with **1.79 s** cold load.
  - **Q48-Shared Split-64 WMMA FlashAttention (`attn_b48_gqa3_flash_split_q8kv_k`):** Processes all `B <= 48` speculative draft tokens across 3 Q-tiles × 3 GQA heads in a single `uint4` 128-bit coalesced KV-streaming pass (`3×` less KV HBM bandwidth at 32K–256K) with pre-captured `K=48` HIP graphs.
  - **64-Q-Token GQA-3:1 Shared-KV Prefill FlashAttention (`attn_batch_gqa3_flash_q8kv_k`):** Shares INT8 K/V loads across 64 Q tokens × 3 GQA heads (`384` threads, 10.5 KB LDS) with `tile_all_far` relative-bias skipping — driving **`3,448.7 tok/s`** `pp2048` prefill (`4.51×` stock `llama.cpp`), **`2,984.9 tok/s`** at 32K, **`2,707.1 tok/s`** at 64K, **`2,284.7 tok/s`** at 128K, and **`1,744.4 tok/s`** at 256K.
  - **Native `incoai/Inkling-2-Mini-DFlash2` Speculative Decoding (`287.2 tok/s` Pure-Gen Code, `249.0 tok/s` Math, `236.4 tok/s` Pure-Gen 2K, `105.0 tok/s` Prose):** Zero-copy `safetensors` loader + 5-layer parallel block-diffusion drafter with WMMA GQA 4:1 Split-16 FlashAttention, fused `dflash2_rmsnorm_q8` / `dflash2_conv_q8` / `dflash2_proj_conv_add_q4` / `dflash2_gu_swiglu_q8`, 16-chunk parallel top-16 candidate selector (`draft=1.68–1.83 ms`), hoisted MoE nibble unpacks, coalesced `float4`/`uint32_t` attention, and async SWA-windowed feature injection (`inject=0.01–0.02 ms`).
  - **Native Agentic Tool Calling & Multi-Turn Prefix Caching:** Full Inkling-2-Mini native `tool_declare` / `<|content_invoke_tool_json|>` (`{"name":...,"args":...}`) / `<|message_tool|>` prompt formatting, real-time token-ID `InklingStreamParser`, balanced JSON tool-arg extraction, automatic repeated-tool-loop breaker with nucleus/repetition-penalty sampling, and multi-turn KV prefix caching with `cached_tokens` reported in both JSON and SSE `stream_options.include_usage`.
- **Native GPU Vision (`mmproj-F16.gguf`, Enabled by Default)**
  - Full 23-layer HIP ViT encoder (`640×640`, `20×20` patches, LayerScale `ls1`/`ls2`, bidirectional self-attention, `2×2` spatial merge, `inkling2v` MLP projector) outputting 256 multimodal embeddings directly into `InklingEngine` and `DFlash2Drafter`.
  - Works out of the box via OpenAI `image_url` (data URLs / local files) in `/v1/chat/completions` and `rad run --image <PATH>`.
- **RDNA4 (`gfx1201`) 8-Warp Coalesced LDS-Tiled Wave32 WMMA Matrix Core Suite (ROCm 7.x)**
  - Custom `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` 8-warp `128×64` & `64×64` `BK=4` coalesced LDS-tiled kernels for `Q4_0_SOA` × `Q8_1` GEMM (`qkvrg`, `wo`, `lm_head`, fused MoE gate/up-to-`Q8_1`, and zero-atomic grouped MoE down with 6-way hot-expert token splitting) and zero-LDS warp-shuffle `Q8_1` quantization.
  - Custom `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` **FlashAttention-2** (reverse causal tile scheduling, bank-conflict-free LDS KV tiles, register-preloaded `float4` relative-position projections, and Split-KV WMMA FlashAttention up to 64 splits).
- **Multi-GPU R9700 Data-Parallel Serving (`--gpus 0,1` / `--gpus auto`) & HIP P2P Collective Layer**
  - Spawns one dedicated GPU scheduler + DFlash 2 + Vision + Paged KV worker per GPU with **prefix-aware least-loaded routing** (matching prompt prefixes stick to the same GPU for KV cache hits while balancing concurrent requests across all GPUs) — zero PCIe all-reduce overhead and linear multi-user throughput & KV capacity scaling.
  - Automatic HIP peer-access detection (`hipDeviceCanAccessPeer` / `hipDeviceEnablePeerAccess`), direct `hipMemcpyPeerAsync` + on-GPU `ink_vec_add` AllReduce/Broadcast with pinned-host fallback (`crates/rad-core/src/comm.rs`), tensor-parallel sharding plan (`crates/rad-core/src/tp.rs`), and GPU/P2P topology inspection via `rad info` and `GET /health`.
- **Production OpenAI-Compatible Serving (`rad serve`)**
  - Continuous batching up to `B=128` (**`1,856.0 agg tok/s`**), **Paged KV block manager** (`--kv-block-size 16`, copy-on-write prefix sharing & refcounting), **Chunked Prefill** (`--prefill-chunk-size 512`), **JSON & Tool-Calling Grammar Constraints**, **`stop` sequences**, **`seed` / `presence_penalty` / `frequency_penalty`**, **`400 context_length_exceeded`** overflow guard, **Prometheus `GET /metrics`**, optional **multi-key API auth** (`Authorization: Bearer` & `X-API-Key`, `--metrics-auth`), CORS + `x-request-id` middleware, and non-conflicting default port **`6733`**.
  - Endpoints: `GET /health`, `GET /metrics`, `GET /v1/models`, `GET /v1/models/:id`, `POST /v1/tokenize`, `POST /v1/detokenize`, `POST /v1/chat/completions` (JSON + SSE + `stream_options.include_usage`), `POST /v1/completions` (single + batch), `POST /v1/responses` (HTTP + SSE), `GET /v1/responses` (WebSocket).

---

## Benchmarks (AMD Radeon AI PRO R9700, `gfx1201`, ROCm 7.14)

### Single-User Decode & Prefill (RAD vs. Stock `llama.cpp` & RTX PRO 6000 Blackwell)

![RAD vs Stock llama.cpp and RTX PRO 6000 Blackwell](assets/rad_benchmark.svg)

Measured end-to-end on the **same AMD Radeon AI PRO R9700 (32 GB, ~\$1,250)**, **same GGUF (`Inkling-2-Mini-UD-Q4_K_XL-v2.gguf`)**, and **same `B=1` / `pp2048` / `tg128`** (`rad bench` 3-run `mean ± stddev`, on-GPU `Q4_0_SOA` weight repacking, **INT8 full-attention KV cache + FP16 SWA KV**) — achieving **4.07×–7.75× faster decode** and **4.50× faster `pp2048` prefill** than stock `llama.cpp`, and outperforming Unsloth's claimed **125 tok/s** on an **NVIDIA RTX PRO 6000 Blackwell (96 GB, ~\$8,500)** by **2.30×–4.37×**:

| Engine / Hardware | Decode (`tok/s`) | `pp2048` Prefill (`tok/s`) | Notes |
| --- | --- | --- | --- |
| **Stock `llama.cpp` (R9700 + DSpark)** | `70.5 tok/s` | `765.5 tok/s` | `-fa on -ctk q8_0 -ctv q8_0 --spec-type draft-dspark` |
| **RAD (R9700, No-Spec `hipGraph`)** | **`108.7 ± 0.7 tok/s`** (`1.54×`) | **`3,448.7 ± 3.3 tok/s`** (**`4.51×`**) | `tg128 @ d=2048 = 99.4 tok/s`; `pp512 = 2,928.3 ± 4.2 tok/s` |
| **Unsloth RTX PRO 6000 Blackwell** | `125.0 tok/s` *(claimed)* | — | 96 GB workstation GPU (~\$8,500) |
| **RAD (R9700 + DSpark `K=6`)** | **`148.2 tok/s`** (`2.10×`) | **`3,448.7 tok/s`** (**`4.51×`**) | `verify=23.7 ms`, `avg_emit=4.27` |
| **RAD (R9700 + DFlash 2 `K=8`, Pure Gen)** | **`287.2 tok/s`** (**`4.07×`**) | **`3,406.4 ± 1.6 tok/s`** (**`4.45×`**) | **2.30× vs. RTX PRO 6000 Blackwell**, `Math=249.0 tok/s`, `2K Pure Gen=236.4 tok/s`, `Prose=105.0 tok/s` |
| **RAD (Hybrid DFlash 2 + `K=48`, 32K–64K)** | **`546.7 tok/s`** (**`7.75×`**) | **`2,984.9 tok/s`** (**`3.90×`**) | **4.37× vs. RTX PRO 6000 Blackwell**; **`521.4 tok/s`** at 64K, **`245.1 tok/s`** at 256K |

### Long-Context Decode & Prefill (`2K → 256K`, Single 32 GB R9700)

![RAD Long-Context 2K to 256K Scaling on Single R9700](assets/rad_longctx.svg)

Measured via `rad bench --ctx 262144 --max-seqs 1` on a single AMD Radeon AI PRO R9700 (32 GB) with INT8 full-attention KV cache, Q48-Shared Split-64 WMMA FlashAttention, GQA-3:1 Shared-KV Prefill FlashAttention, and hybrid DFlash 2 + `K=48` prompt-lookup speculation (`1.79 s` cold load at 256K):

| Context Depth | Prompt Tokens | Prefill (`tok/s`) | Prefill Time | Decode (`tok/s`) | Avg Emit / Round |
| --- | --- | --- | --- | --- | --- |
| **2K** | 1,977 | **3,288.5 tok/s** | 0.60 s | **355.3 tok/s** | 21.7 |
| **4K** | 4,067 | **3,278.7 tok/s** | 1.24 s | **339.4 tok/s** | 21.7 |
| **8K** | 8,057 | **3,230.1 tok/s** | 2.49 s | **362.3 tok/s** | 21.7 |
| **16K** | 16,227 | **3,144.9 tok/s** | 5.16 s | **356.8 tok/s** | 21.7 |
| **32K** | 32,757 | **2,984.9 tok/s** | 10.97 s | **546.7 tok/s** | 35.0 |
| **64K** | 65,437 | **2,707.1 tok/s** | 24.17 s | **521.4 tok/s** | 35.0 |
| **128K** | 130,987 | **2,284.7 tok/s** | 57.33 s | **284.9 tok/s** | 21.7 |
| **256K** | 261,707 | **1,744.4 tok/s** | 150.03 s | **245.1 tok/s** | 21.7 |

### Multi-User Continuous Batching & Batched Speculative Decoding

![RAD Multi-User Scaling on Single R9700](assets/rad_multiuser.svg)

| Workload | Throughput / Latency | Notes |
| --- | --- | --- |
| **Cold Load (`ctx=262144`, DFlash 2)** | **1.79 s** | On-GPU `Q4_0_SOA` + `Q8_0` conversion & `hipGraph` capture |
| **Short Prompt TTFT (`15 tok`)** | **34.8 ± 0.1 ms** | 3-run mean; 8-warp LDS WMMA + fused `float4` state mixing |
| **Prefix-Cached TTFT (`1,645 tok`, `1,639` cached)** | **21.2 ms** | Prefix-cache affinity hit + GQA 3:1 Split-KV FlashAttention |
| **Multi-User Decode (`B=4`, Batched DFlash 2 `K=4`)** | **485.0 agg tok/s** | **`121.2 tok/s/user`** (`ITL 8.2 ms`, zero-spill MoE + LDS WMMA) |
| **Multi-User Decode (`B=8`, no-spec)** | **559.2 agg tok/s** | **`69.9 tok/s/user`** (`ITL 14.3 ms`, `hipGraph` + fused `router+groups+Q8_1`) |
| **Multi-User Decode (`B=8`, Batched DSpark `K=4`)** | **458.6 agg tok/s** | `57.3 tok/s/user` |
| **Multi-User Decode (`B=8`, Batched DFlash 2 `K=4`)** | **676.2 agg tok/s** | **`84.5 tok/s/user`** (`ITL 11.8 ms`; 8 concurrent users each faster than single-user `llama.cpp`) |
| **Multi-User Decode (`B=16`, no-spec)** | **688.3 agg tok/s** | **`43.0 tok/s/user`** (`ITL 23.2 ms`) |
| **Multi-User Decode (`B=32`, no-spec)** | **867.2 agg tok/s** | `27.1 tok/s/user` |
| **Multi-User Decode (`B=64`, no-spec)** | **1,162.2 agg tok/s** | `18.2 tok/s/user` (8-warp `128×64` LDS WMMA MoE) |
| **Multi-User Decode (`B=128`, no-spec)** | **1,856.0 agg tok/s** | **`14.5 tok/s/user`** (fused WMMA MoE-to-`Q8_1` + `float4` slot reduction) |

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
│   ├── rad_wmma.hip           RDNA4 Wave32 WMMA INT8 GEMM, MoE, & FP16 FlashAttention-2
│   └── rad_q8kv.hip           INT8 KV cache, GQA-3:1 prefill & Q48-shared Split-64 WMMA FlashAttention
├── crates/
│   ├── rad-gguf/              Zero-copy mmap GGUF reader
│   ├── rad-hip/               Minimal ROCm/HIP FFI, RAII DevBuf, P2P & CapturedGraph
│   ├── rad-core/              InklingEngine, DFlash 2 & DSpark, Paged KV, MultiGpuComm, TpShardPlan, VisionEncoder
│   └── rad-serve/             `rad` CLI + Axum HTTP/SSE/WebSocket server, grammar, sampling, metrics
├── assets/
│   ├── rad_benchmark.svg      Single-user decode & prefill comparison chart
│   ├── rad_longctx.svg        Long-context 2K–256K decode & prefill scaling chart
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
./target/release/rad serve -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --port 6733

./target/release/rad serve -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --gpus 0,1 --port 6733

./target/release/rad serve -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --ctx 131072 --max-seqs 2 --port 6733

./target/release/rad run "Show me a Python quicksort function." -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf

./target/release/rad run "Describe this image." --image /path/to/photo.png -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf

./target/release/rad info -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf
```

### 4. Run Built-In Benchmarks (`rad bench`)

Includes Cold Load, TTFT, `pp512`/`pp2048` Prefill, Single-User Code/Math/Prose, Multi-User `B=4/8/16`, and **2K–256K Long-Context Scaling**:

```bash
./target/release/rad bench -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --ctx 4096 --max-seqs 16

./target/release/rad bench -m /path/to/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --ctx 262144 --max-seqs 1

python3 scripts/test_serve.py
```

### 5. Install & Deploy (Docker, `.deb`, Homebrew, Flatpak)

- **Docker with AMD ROCm GPU Passthrough:**
  ```bash
  RAD_MODELS_DIR=/path/to/models docker compose up -d
  ```
  Or directly:
  ```bash
  docker run --rm -p 6733:6733 \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined --ipc=host \
    -v /path/to/models:/models:ro \
    ghcr.io/metaspartan/rad:latest serve -m /models/Inkling-2-Mini-UD-Q4_K_XL-v2.gguf --port 6733
  ```
- **Debian / Ubuntu (`.deb`):**
  Download `rad_<version>_amd64.deb` from GitHub Releases or build locally:
  ```bash
  bash packaging/deb/build_deb.sh 0.6.0
  sudo dpkg -i dist/rad_0.6.0_amd64.deb
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
| `RAD_CTX` | `8192` | Per-sequence KV context length (tested up to `262144`) |
| `RAD_MAX_SEQS` | `32` | Maximum concurrent sequences in KV pool |
| `RAD_PORT` | `6733` | HTTP/WebSocket bind port |
| `RAD_GPUS` | `0` | Comma-separated HIP GPU IDs (`0,1`) or `auto` for multi-GPU data-parallel serving |
| `RAD_API_KEY` | *(empty)* | Optional comma-separated Bearer / `X-API-Key` tokens (disabled if empty) |
| `RAD_ARCH` | `gfx1201` | Target GPU architecture for `kernels/build.sh` |

---

## License

MIT — Copyright (c) 2026 Carsen Klock. See [LICENSE](LICENSE).
