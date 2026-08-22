<div align="center">

# Celiums BitNet Runtime

### Ternary models on CPU. Packed on disk. RAM as a serving lever. Exact math.

[![License](https://img.shields.io/badge/Celiums%20code-Apache--2.0-blue.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-Microsoft%20BitNet-5C2D91.svg)](UPSTREAM.md)
[![Version](https://img.shields.io/badge/version-0.3.0-111827.svg)](CHANGES.md)
[![I2_S](https://img.shields.io/badge/I2__S-contract%20validated-0F766E.svg)](docs/NUMERICAL_CONTRACT.md)
[![Q1](https://img.shields.io/badge/Bonsai%20Q1-CPU%20text-111827.svg)](docs/VENDORED_CPU_HOT_PATH.md)

</div>

Celiums BitNet is a **CPU inference runtime for ternary networks**, not a
wrapper around `llama-cli`. The public product is `celiums-bitnet`: one-shot
generation, a native HTTP server, an experimental C ABI, and JSONL
prefill/decode benches. The engine is vendored llama.cpp/ggml at
`3rdparty/llama.cpp`. Applications talk to Celiums, not to GGML.

Two families are in the product surface:

| Family | Model | On disk | Why it exists |
| --- | --- | --- | --- |
| `bitnet` (default) | [BitNet b1.58 2B](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | I2_S, ~1.1 GiB | Certified ternary contract, 2-bit packed codes |
| `bonsai` | [Bonsai 27B Q1_0](https://huggingface.co/prism-ml/Bonsai-27B-gguf) | Q1_0, ~3.53 GiB | 27B ternary CPU target; 1-bit signs, Gated DeltaNet + attention |

A Llama/Qwen `Q4_K_M` GGUF is not a Celiums model. The engine still contains
those kernels; the loader refuses them. Prism `Q2_0` is unsupported (its
type IDs collide with Celiums TL2/I2_S).

> [!IMPORTANT]
> Maintained by Celiums Solutions LLC. Not affiliated with, sponsored by,
> endorsed by, or supported by Microsoft. Upstream BitNet, llama.cpp, ggml,
> and model artifacts keep their own copyright and licenses.

## Why this runtime exists

Ternary weights exist to cut **memory bandwidth**, the thing decode actually
pays. A 27B model at ~1.13 bits/parameter is 3.53 GiB on disk. Streaming that
once per generated token is the decode roofline. Prefill is different: the
same weight image is reread once per activation tile. A 4-row tile on a
128-token prompt rereads Q1 **32 times**.

The 0.3.0 product already had a strict I2_S contract and a C API. What was
missing for serving was the right to **spend RAM on purpose**, under a cap so
the host stays usable, and to stop rereading Q1 panels on prefill.

That is this work:

1. **Packed GGUF stays the durable store.** We do not clone the file in RAM
   and we do not treat NVMe as the inference path.
2. **RAM is a compute working set.** On ARM with i8mm, Q1 expands to int8
   ±1 so existing `usdot` kernels run. On x86, Q1 stays bit-packed 4×8 and
   uses `vpdpbusd` (2P−S). Expanding x86 Q1 to bytes would blow DRAM on
   decode; Graviton already showed that at 96 threads.
3. **Every extra byte is capped.** `--ram-budget-bytes` is fail-closed:
   `CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED` at model load or session
   create. Production default in the metal receipts is 64 GiB on 755 GiB
   hosts.
4. **Prefill reuses weights.** Q1 GEMM applies one 4×8 panel to **eight**
   activation rows (FBGEMM-style blocking), then four. Decode is still
   GEMV: one token, one pass over the image.

LUT-GEMM and T-MAC stay out. They are approximate. I2_S and Q1 keep exact
integer accumulation: `y = (D − S) ρ` and `dot = d_w d_x (2P − S)`.

## How inference actually runs

```text
GGUF on disk (I2_S or Q1_0, mmap)
        |
        |  optional compute layout, charged against ram_budget
        v
in-RAM image
  ARM i8mm : Q1 -> q8_0 4x8  (~8x bytes, ~26 GiB for Bonsai)
  x86 VNNI : Q1 4x8 bit-packed panels (~3.4 GiB)
  I2_S     : tinyBLAS VNNI, no Q1 expand
        |
        |  MUL_MAT
        v
prefill  n>3  -> GEMM, 8-row then 4-row tiles   (prompt)
decode   n=1  -> GEMV, stream weights once      (next token)
        |
        v
exact epilogue   I2_S: (D-S)*rho    Q1: 2P-S
```

The public call is still `celiums_bitnet_session_prefill` /
`celiums_bitnet_session_decode`. Decode is prefill of length 1; the kernel
split is inside vendored ggml.

Exact math lives in `celiums-exact`
(`3rdparty/llama.cpp/ggml/src/ggml-cpu/celiums-exact.{c,h}`). Kernels and
oracles call it so tests do not reimplement the unit under test.

| Knob | Default | What it buys |
| --- | --- | --- |
| `--model-family bitnet\|bonsai` | `bitnet` | Architecture + file-type gate |
| `--compute-layout 0\|1` | `1` | ISA working set (expand or 4×8 panels) |
| `--ram-budget-bytes N` | auto: ~half of host RAM, always headroom | Fail-closed cap |
| `--n-seq N` | `1` | Concurrent decode states sharing one weight image |
| `--threads` / `--threads-batch` | | Decode vs prefill parallelism |

Zero budget means auto: half of host RAM, never more than 90%, always
leaving at least 4 GiB or 10% free.

## What the changes add (measured)

All figures are `celiums-bitnet bench`, Bonsai 27B Q1_0 SHA
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`,
pp128 / tg128, cap 64 GiB. Metals were destroyed after the receipts.

### Graviton 4 — `r8g.metal-24xl` (96× Neoverse-V2)

Packed mmap was compute-bound. Expanding Q1 to int8 spends ~29 GiB RSS and
feeds i8mm.

| Threads | Packed mmap pp / tg | Expand-to-int8 pp / tg |
| ---: | ---: | ---: |
| 1 | 0.528 / 0.500 | **2.94 / 1.13** |
| 96 | — / 18.16 | **147.0 / 12.11** |

Prefill at t=1 is **5.6×**. Decode at t=1 is **2.3×**. At 96 threads decode
**drops**: the int8 image saturates DRAM. Use the expand for prefill and
moderate concurrency, not as “fill the machine.”

### Xeon 8559C — `i7i.metal-24xl` (48c SMT-2, AVX-512 VNNI)

x86 does **not** expand Q1. The lever is VNNI on bit-packed 4×8, then
8-row prefill tiles.

| Threads | 4-row GEMM pp / tg | 8-row GEMM pp / tg |
| ---: | ---: | ---: |
| 1 | 2.389 / 1.234 | **2.593 / 1.227** |
| 48 | 75.41 / 18.99 | **80.40 / 18.69** |

Prefill **+8.6%** at t=1, **+6.6%** at t=48. Decode is unchanged: GEMV
still streams ~3.5 GiB/token (t=1 cache miss was ~65%). RSS stays ~7.05 GiB.

A 64-byte budget fails closed (exit 1, ~9 MiB RSS, no model alloc) on both
ISAs.

Same i7i, certified BitNet 2B I2_S: t=1 **77.82 / 9.58**, t=48 **1304 / 153**
tok/s. Different model, different roofline; listed so 27B Q1 is not confused
with 2B I2_S.

Raw JSON: [`docs/benchmark-results-2026-08-22-r8g-bonsai-ram-lever-rerun.json`](docs/benchmark-results-2026-08-22-r8g-bonsai-ram-lever-rerun.json),
[`docs/benchmark-results-2026-08-22-i7i-ram-lever-rerun.json`](docs/benchmark-results-2026-08-22-i7i-ram-lever-rerun.json),
[`docs/benchmark-results-2026-08-22-i7i-q1-8row.json`](docs/benchmark-results-2026-08-22-i7i-q1-8row.json).
Hot-path review: [`docs/VENDORED_CPU_HOT_PATH.md`](docs/VENDORED_CPU_HOT_PATH.md).

## 0.3.0 Support Matrix

| Area | Supported product surface |
| --- | --- |
| Certified model | [`microsoft/BitNet-b1.58-2B-4T`](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) I2_S |
| CPU text family | PrismML Bonsai 27B `Q1_0` via `--model-family bonsai` |
| Runtime | CPU only; one-shot generation, tokenize, prefill, decode, logits, sampling, callbacks, stops, cancel |
| CPU profiles | `native`, portable x86-64 `avx2`, scalar reference |
| Public interfaces | `celiums-bitnet` CLI, experimental C ABI v1, native HTTP API |
| Serving knobs | RAM budget, compute layout, `n_seq` |
| Optional local AI | Hyphae RAG / memory / receipts / MCP |
| Release platform | Linux x86-64 shared-library archives |

Product CMake disables GPU and generic accelerator backends. ARM64 is
measured on Graviton 4 for Bonsai Q1; it is not the 0.3.0 I2_S
certification target.

## Quick Start

### Tested environment

- official release archives: Linux x86-64, Ubuntu 24.04 / glibc 2.39;
- source builds: Linux x86-64 continuously tested; ARM64 measured for Bonsai Q1;
- Python 3.10–3.12 for conversion;
- CMake 3.22+, GCC or Clang with C++17, Git;
- Rust 1.97.1 only for the optional gateway.

### Install a release archive

After `v0.3.0` is published, download an archive and its matching manifest from
[GitHub Releases](https://github.com/celiumsai/celiums-bitnet/releases). Choose:

- `native` only for the same CPU class as the release builder;
- `avx2` for x86-64 CPUs with SSE4.2, AVX, AVX2, FMA, and F16C;
- `scalar` as the compatibility/reference fallback.

```bash
PROFILE=avx2
ARCHIVE="celiums-bitnet-runtime-0.3.0-linux-x86_64-${PROFILE}.tar.gz"
MANIFEST="${ARCHIVE}.json"

expected=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"][0]["sha256"])' "$MANIFEST")
printf '%s  %s\n' "$expected" "$ARCHIVE" | sha256sum -c -

mkdir celiums-bitnet-0.3.0
tar -xzf "$ARCHIVE" -C celiums-bitnet-0.3.0
celiums-bitnet-0.3.0/usr/bin/celiums-bitnet version
```

The archive is a `/usr`-shaped relocatable payload.

### Build from source

```bash
git clone https://github.com/celiumsai/celiums-bitnet.git
cd celiums-bitnet

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python setup_env.py --build-only --build-server
```

The engine is vendored at `3rdparty/llama.cpp`. A plain clone is complete.

### Certified BitNet 2B

Hugging Face revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`:

```bash
python setup_env.py \
  --hf-repo microsoft/BitNet-b1.58-2B-4T \
  --revision 04c3b9ad9361b824064a1f25ea60a8be9599b127 \
  --quant-type i2_s

MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
printf '%s  %s\n' \
  e23b16fa81b890e8b65e676262b645e8ffa5ae1f6df89dadaf793246826bbd90 \
  "$MODEL" | sha256sum -c -
```

### Bonsai 27B Q1_0

Use a pre-quantized `Bonsai-27B-Q1_0.gguf` (SHA-256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`).
Celiums does not convert Bonsai.

```bash
BONSAI=models/Bonsai-27B-Q1_0.gguf
build/bin/celiums-bitnet validate --model "$BONSAI" --model-family bonsai
build/bin/celiums-bitnet run \
  --model "$BONSAI" --model-family bonsai \
  --prompt "Explain why ternary weights reduce memory bandwidth." \
  --threads 8 --threads-batch 48 --n-predict 128 \
  --ram-budget-bytes 68719476736
```

### Validate, run, and serve (BitNet)

```bash
build/bin/celiums-bitnet version
build/bin/celiums-bitnet validate --model "$MODEL"
build/bin/celiums-bitnet run \
  --model "$MODEL" \
  --prompt "Explain why ternary weights reduce memory bandwidth." \
  --threads 6 --threads-batch 14 --n-predict 128
```

`run` is one-shot generation, not a chat loop.

```bash
CELIUMS_BITNET_API_KEY='replace-me' \
build/bin/celiums-bitnet serve \
  --model "$MODEL" --host 127.0.0.1 --port 8080
```

Non-loopback binding is refused unless an API key is configured or the
explicit unsafe override is supplied.

```bash
build/bin/celiums-bitnet bench --model "$MODEL" -p 128 -n 128 -t 8 -r 3
```

## Core Capabilities

### Strict I2_S CPU path

- canonical 128-coefficient ternary packing;
- tensor-wide F32 weight scale;
- per-row I8_S activation quantization and activation-code sum;
- unsigned-code dot product and offset recovery `y = (D − S) ρ`;
- scalar, AVX2-without-VNNI, and native/VNNI paths;
- GEMV and GEMM, multiple threads, zero rows, multi-plane activations;
- squared-ReLU BitNet FFN graph;
- structural conversion checks and `celiums-bitnet validate`.

### Bonsai Q1_0 CPU path

- 1-bit signs, FP16 block scale, `dot = d_w d_x (2P − S)`;
- 4×8 repack; ARM expand-to-int8 or x86 VNNI bit-packed panels;
- 8-row prefill GEMM weight reuse;
- Qwen35 Gated DeltaNet + full attention (inherited graph, F32 state);
- packed vs layout vs generic oracle (`tests/test-q1-repack-oracle.cpp`).

### Runtime and serving

- opaque Runtime, Model, Session, and Request C handles;
- fail-closed RAM budget and optional compute layout;
- tokenization / detokenization, prefill, decode, logits, sampling;
- greedy or temperature/top-k/top-p;
- stop sequences, cooperative cancellation, streaming callbacks;
- JSONL prefill/decode benchmark;
- OpenAI-shaped HTTP subset, SSE, bearer / `X-Api-Key`, `/metrics`.

### Optional Hyphae gateway

BM25, exact cosine, HNSW, hybrid retrieval, memory, receipts, proofs,
registry, semantic cache, MCP. The gateway is process-separated: it does
not link Hyphae into ggml.

## Supported Models

### BitNet b1.58 2B (certified)

| Property | Value |
| --- | --- |
| Repository | `microsoft/BitNet-b1.58-2B-4T` |
| Architecture | `bitnet-b1.58` |
| File type | `MOSTLY_I2_S` (41) |
| Parameters | ~2.41B |
| Validated SHA-256 | `e23b16fa81b890e8b65e676262b645e8ffa5ae1f6df89dadaf793246826bbd90` |

### Bonsai 27B Q1_0 (CPU text family)

| Property | Value |
| --- | --- |
| File | `Bonsai-27B-Q1_0.gguf` |
| Architecture | `qwen35` |
| File type | `MOSTLY_Q1_0` (40) |
| Parameters | ~26.9B |
| SHA-256 | `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0` |

Model files are not in this repository. Licenses stay with their publishers.
`celiums_bitnet_model_load()` checks architecture and file type, not the
Hugging Face identity. Pin the revision and digest in production.

`celiums_bitnet_model_load()` is BitNet-only. Bonsai requires
`celiums_bitnet_model_load_family(..., CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0)`
or `--model-family bonsai`.

## Mathematical Contract Summary

Let a logical ternary weight matrix be

```text
W in {-1, 0, +1}^{M x K},  K > 0,  K mod 128 = 0.
```

Only `K` must be divisible by 128.

### 1. Ternary codes and packing (I2_S)

```text
u[r,i] = W[r,i] + 1     # -1 -> 0,  0 -> 1,  +1 -> 2
```

Each 128-coefficient block occupies 32 packed bytes. A 32-byte trailer holds
the tensor-wide scale `alpha_W`.

### 2. Activation quantization

```text
beta_t = 127 / amax_t     (0 if the row is zero)
q_t[i] = clamp(round_half_away(beta_t * x_t[i]), -128, 127)
S_t    = sum_i q_t[i]
```

### 3. Integer dot and recovery

```text
D[r,t] = sum_i u[r,i] * q_t[i]
T[r,t] = D[r,t] - S_t = sum_i W[r,i] * q_t[i]
y_hat  = (alpha_W / beta_t) * T     (0 if beta_t = 0)
```

Q1_0 is the 1-bit sibling: stored bit `b`, `P = sum b q`, `S = sum q`,
`dot = d_w d_x (2P − S)`. Same VNNI instruction, different packing.

Full layout, bounds, and evidence:
[`docs/NUMERICAL_CONTRACT.md`](docs/NUMERICAL_CONTRACT.md).

## Public Interfaces

### CLI

| Command | Purpose |
| --- | --- |
| `celiums-bitnet run` | One-shot prompt generation |
| `celiums-bitnet serve` | Native HTTP completion server |
| `celiums-bitnet bench` | Prefill/decode throughput in JSONL |
| `celiums-bitnet validate` | Checked model load |
| `celiums-bitnet version` | Product, engine, tree, profile, strict |

`run`, `serve`, `bench`, and `validate` accept `--model-family bitnet|bonsai`,
`--ram-budget-bytes`, `--compute-layout`, and `--n-seq`.

### Experimental C ABI v1

Header: [`include/celiums/bitnet_runtime.h`](include/celiums/bitnet_runtime.h).

Runtime / Model / Session / Request handles; provenance; family-gated load;
tokenize; prefill/decode/logits/sample; RAM budget; compute layout; `n_seq`.
The ABI is 0.x. A Session serializes mutable context operations.

### Native HTTP API

| Method | Endpoint |
| --- | --- |
| `GET` | `/health`, `/v1/health` |
| `GET` | `/v1/models` |
| `POST` | `/v1/completions`, `/v1/chat/completions` |
| `GET` | `/metrics` |

OpenAI-shaped **subset**. No continuous batching, tools, logprobs, or
complete usage counts.

## Optional Local AI Gateway

```bash
cmake -S . -B build-gateway \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCELIUMS_BITNET_BUILD_GATEWAY=ON
cmake --build build-gateway --parallel
```

```text
client -> celiums-runtime-gateway
            | UDS                    | HTTP loopback
            v                        v
       Hyphae sidecar          celiums-bitnet serve
```

See [`docs/LOCAL_AI_GATEWAY.md`](docs/LOCAL_AI_GATEWAY.md) and
[`docs/GATEWAY_THREAT_MODEL.md`](docs/GATEWAY_THREAT_MODEL.md).

## CPU Profiles and Hardware

| Profile | ISA contract | Use |
| --- | --- | --- |
| `native` | `-march=native` | Same CPU class as the builder |
| `avx2` | SSE4.2, AVX, AVX2, FMA, F16C; no AVX-VNNI/AVX-512/AMX | Portable x86-64 |
| `scalar` | SIMD off in the product profile | Reference / fallback |

Product builds disable BLAS, CUDA, HIP, Metal, Vulkan, OpenCL, SYCL, CANN,
RPC, and OpenVINO. TL1/TL2 need `CELIUMS_BITNET_EXPERIMENTAL=ON`. Non-strict
mode is not implemented.

## Validation

CI covers native / AVX2-without-VNNI / scalar, C ABI, I2_S GEMV/GEMM, I8_S
quantization, ASan/UBSan, Python wrappers, and gateway tests. Q1 packed vs
layout vs generic is `test-q1-repack-oracle`. I2_S mul_mat is
`test-i2s-mul-mat`.

The model-backed release gate still requires the pinned I2_S GGUF and
full-vocabulary logits identity against the exactness oracle.
See [`docs/EXACTNESS_ORACLE.md`](docs/EXACTNESS_ORACLE.md).

```bash
CELIUMS_BITNET_TEST_MODEL=/absolute/path/ggml-model-i2_s.gguf \
CELIUMS_BITNET_TEST_ORACLE=/absolute/path/runtime-reference.gguf \
CELIUMS_BITNET_TEST_ORACLE_SIDECAR=/absolute/path/runtime-reference.gguf.json \
JOBS=2 scripts/validate-release.sh
```

## Release Packages

```bash
JOBS=8 ./scripts/package-runtime.sh native
JOBS=8 ./scripts/package-runtime.sh avx2
JOBS=8 ./scripts/package-runtime.sh scalar
```

Packaging refuses a dirty tree unless `ALLOW_DIRTY=1`.

## Explicit Limits

Celiums BitNet does not claim:

- arbitrary llama.cpp models or quantizations;
- source-checkpoint numerical certification;
- universal bitwise identity across ISAs;
- GPU inference in the product build;
- that expanding Q1 to int8 always helps decode (it can hurt at high thread
  counts when the int8 image saturates DRAM);
- that 8-row GEMM moves decode (it does not; decode is GEMV);
- continuous batching, a complete OpenAI API, embeddings, or TLS.

Packed I2_S code `3` is invalid. The converter never emits it; ordinary load
does not scan every packed field. Verify hashes and run `validate`.

## Documentation

| Document | Purpose |
| --- | --- |
| [`docs/RUNTIME_PRODUCT.md`](docs/RUNTIME_PRODUCT.md) | Runtime boundary, ABI, RAM cap |
| [`docs/VENDORED_CPU_HOT_PATH.md`](docs/VENDORED_CPU_HOT_PATH.md) | CPU hot-path review and 8-row GEMM |
| [`docs/NUMERICAL_CONTRACT.md`](docs/NUMERICAL_CONTRACT.md) | I2_S / I8_S equations |
| [`docs/BONSAI_Q1_OPTIMIZATION_DEEP_DIVE.md`](docs/BONSAI_Q1_OPTIMIZATION_DEEP_DIVE.md) | Q1 kernel analysis |
| [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) | Methodology |
| [`include/celiums/bitnet_runtime.h`](include/celiums/bitnet_runtime.h) | C ABI v1 |
| [`CHANGES.md`](CHANGES.md) | Release history |

## Ownership, License, and Attribution

Copyright 2026 Celiums Solutions LLC.

Celiums-authored product code, documentation, tests, and tooling outside the
vendored runtime are [Apache-2.0](LICENSE). Microsoft BitNet portions retain
[MIT](LICENSE-MIT). `3rdparty/llama.cpp` and Celiums modifications inside
that tree are MIT, as stated in [NOTICE](NOTICE).

This repository does not transfer ownership of Microsoft, ggml, llama.cpp,
model, Hyphae, or third-party code to Celiums Solutions LLC.

## Maintainer

- Mario Gutierrez, Celiums Solutions LLC

Security: [SECURITY.md](SECURITY.md). Bugs: open a GitHub issue with product
version, model hash, CPU profile, command, and logs. Contributions must keep
the numerical contract, provenance pins, upstream licenses, and the
[Code of Conduct](CODE_OF_CONDUCT.md).
