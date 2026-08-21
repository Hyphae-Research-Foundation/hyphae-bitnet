<div align="center">

# Celiums BitNet Runtime

### CPU inference for ternary BitNet b1.58 with an explicit numerical contract

[![License](https://img.shields.io/badge/Celiums%20code-Apache--2.0-blue.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-Microsoft%20BitNet-5C2D91.svg)](UPSTREAM.md)
[![Version](https://img.shields.io/badge/version-0.3.0-111827.svg)](CHANGES.md)
[![Status](https://img.shields.io/badge/I2__S-contract%20validated-0F766E.svg)](docs/NUMERICAL_CONTRACT.md)

</div>

Celiums BitNet is an independent engineering fork of
[Microsoft BitNet](https://github.com/microsoft/BitNet). Version 0.3.0 provides
a CPU-only runtime for the BitNet b1.58 I2_S path, a public experimental C ABI,
a native HTTP server, reproducible release tooling, and an optional local AI
gateway backed by [Hyphae](https://github.com/celiumsai/hyphae).

The project is correctness-first: packed weights, activation quantization,
integer accumulation, scale recovery, model metadata, and release provenance
are treated as versioned contracts rather than undocumented implementation
details.

The 0.3.0 release candidate is prepared in source and CI. Until the `v0.3.0`
tag and GitHub assets are published, use the source-build workflow below; the
release-installation section describes the final asset contract.

> [!IMPORTANT]
> Celiums BitNet is maintained by Celiums Solutions LLC and is not affiliated
> with, sponsored by, endorsed by, or supported by Microsoft. Microsoft and the
> original contributors retain copyright in upstream code and model artifacts.

## 0.3.0 Support Matrix

| Area | Supported product surface |
| --- | --- |
| Model | [`microsoft/BitNet-b1.58-2B-4T`](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) release fixture |
| Architecture | Decoder-only causal `bitnet-b1.58`, RMSNorm, RoPE, grouped-query attention, squared-ReLU gated FFN |
| Weight format | GGUF v3, `MOSTLY_I2_S` file type 41, custom GGML I2_S type 36 |
| Runtime | CPU only; one-shot generation, tokenization, prefill, decode, logits, sampling, callbacks, stops, cancellation |
| CPU profiles | `native`, portable x86-64 `avx2`, and scalar reference/fallback |
| Public interfaces | `celiums-bitnet` CLI, experimental C ABI v1, native HTTP API |
| Optional local AI | Hyphae-backed lexical/vector/hybrid RAG, memory, receipts, proofs, registry, semantic cache, MCP |
| Release platform | Linux x86-64 shared-library archives |

The engine contains inherited support for additional model families and
backends, but those entries are not part of the Celiums 0.3.0 compatibility
claim. Product CMake explicitly disables GPU and generic accelerator backends.

## Quick Start

### Tested environment

- official release archives: Linux x86-64, Ubuntu 24.04 / glibc 2.39 baseline;
- source builds: Linux x86-64 is continuously tested; ARM64 is best-effort and
  not strict-certified;
- Python 3.10 through 3.12 for the documented conversion environment;
- CMake 3.22 or newer;
- GCC or Clang with C++17 support;
- Git;
- Rust 1.97.1 only when building the optional gateway.

### Install a release archive

After `v0.3.0` is published, download an archive and its matching manifest from
[GitHub Releases](https://github.com/celiumsai/celiums-bitnet/releases). Choose:

- `native` only for the same CPU class as the release builder;
- `avx2` for x86-64 CPUs with SSE4.2, AVX, AVX2, FMA, and F16C;
- `scalar` as the compatibility/reference fallback.

Verify and extract:

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

The archive is a `/usr`-shaped relocatable payload. Install it under `/` with
your package/image tooling, or invoke it from the extracted tree with its
relative library layout intact.

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

The engine is vendored at `3rdparty/llama.cpp`; a plain clone is complete and
does not fetch a second engine repository. Even build-only setup imports the
conversion validator, so install the Python requirements first.

### Download and convert the certified source revision

The release fixture uses Hugging Face source revision
`04c3b9ad9361b824064a1f25ea60a8be9599b127`:

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

This invocation configures/builds the runtime and converts the model. If you
already built, reuse the same build directory; CMake and Cargo perform
incremental work.

### Validate, run, and serve

```bash
build/bin/celiums-bitnet version
build/bin/celiums-bitnet validate --model "$MODEL"
build/bin/celiums-bitnet run \
  --model "$MODEL" \
  --prompt "Explain why ternary weights reduce memory bandwidth." \
  --threads 6 \
  --threads-batch 14 \
  --n-predict 128
```

`run` is one-shot generation, not a terminal conversation loop.

```bash
CELIUMS_BITNET_API_KEY='replace-me' \
build/bin/celiums-bitnet serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8080
```

Non-loopback binding is refused unless an API key is configured or the explicit
unsafe override is supplied. Authentication applies to health, models,
completions, chat completions, and metrics.

## Core Capabilities

### Strict I2_S CPU path

- canonical 128-coefficient ternary packing;
- tensor-wide F32 weight dequantization scale;
- per-row I8_S activation quantization and activation-code sum;
- corrected unsigned-code dot product and offset recovery;
- scalar, AVX2-without-VNNI, and native/VNNI execution paths;
- GEMV and GEMM, multiple threads, zero rows, and multi-plane activations;
- squared-ReLU BitNet FFN graph;
- structural conversion checks for architecture, activation, projection
  presence, shape, byte count, scale, and GGUF metadata;
- checked model load through `celiums-bitnet validate`.

### Runtime and serving

- opaque Runtime, Model, Session, and Request C handles;
- model metadata and chat-template application;
- tokenization and detokenization;
- prefill, single-token decode, copied logits, and sampling;
- greedy or temperature/top-k/top-p generation;
- repeatable stop sequences and cooperative cancellation;
- synchronous token streaming callbacks;
- one-shot CLI generation and JSONL prefill/decode benchmarking;
- an OpenAI-shaped HTTP subset for text and chat completions;
- SSE streaming with `data: [DONE]` and disconnect cancellation;
- bearer or `X-Api-Key` authentication on every server endpoint;
- Prometheus counters and gauges at `/metrics`.

### Optional Hyphae gateway

- BM25 lexical retrieval;
- exact cosine-vector retrieval;
- HNSW ANN retrieval with exact reranking;
- lexical + vector hybrid retrieval;
- scoped persistent documents and agent memory;
- idempotent request state and generation receipts;
- opt-in semantic response cache;
- proof/witness generation and offline semantic re-execution;
- artifact, dataset, lineage, run, evaluation, and contamination records;
- hard-negative mining manifests;
- separate MCP stdio adapter for retrieval, RAG, memory, and verification.

The gateway is model-independent at the orchestration boundary: another
generator can use the same Hyphae knowledge plane if it exposes the expected
loopback completion API. Vector modes require a separately configured
OpenAI-compatible embedding service; Celiums BitNet 0.3.0 does not provide a
built-in embedding endpoint.

## Supported Model

The certified release fixture is:

| Property | Value |
| --- | --- |
| Repository | `microsoft/BitNet-b1.58-2B-4T` |
| Source class | `BitnetForCausalLM` / `BitNetForCausalLM` |
| Required source activation | `relu2` |
| Converted architecture | `bitnet-b1.58` |
| File type | `MOSTLY_I2_S` (41) |
| Training context metadata | 4096 tokens |
| Layers | 30 |
| Parameters reported by runtime | approximately 2.41B |
| Validated GGUF SHA-256 | `e23b16fa81b890e8b65e676262b645e8ffa5ae1f6df89dadaf793246826bbd90` |
| Tensor inventory | 210 I2_S, 121 F32, 1 F16 |

Model files are not distributed in this repository or in runtime release
archives. Their licenses and terms remain those of their publishers.

`celiums_bitnet_model_load()` enforces architecture and file-type metadata, not
the model repository identity or known SHA-256. Production deployments should
pin the model revision and verify the expected digest before loading it.

## Mathematical Contract Summary

Let a logical ternary weight matrix be

```text
W in {-1, 0, +1}^{M x K},  K > 0,  K mod 128 = 0.
```

Only the contraction dimension `K` must be divisible by 128; `M` and the number
of activation rows do not have that restriction.

### 1. Ternary codes and packing

The persisted unsigned code is

```text
u[r,i] = W[r,i] + 1

W = -1  ->  u = 0
W =  0  ->  u = 1
W = +1  ->  u = 2
```

For each logical block of 128 coefficients, packed byte `g` stores positions
`g`, `g+32`, `g+64`, and `g+96` in bit fields `7:6`, `5:4`, `3:2`, and `1:0`:

```text
P[g] = (u[g] << 6) | (u[g+32] << 4) |
       (u[g+64] << 2) | u[g+96],       0 <= g < 32.
```

Each 128-coefficient block occupies 32 packed bytes. The tensor adds a 32-byte
trailer whose first F32 is the tensor-wide weight scale `alpha_W`; remaining
trailer bytes are reserved and ignored by the runtime.

### 2. Activation quantization

For each finite activation row `x_t` in the validated numerical domain:

```text
amax_t = max_i abs(x_t[i])

beta_t = 127 / amax_t,  if amax_t > 0
         0,             if amax_t = 0

q_t[i] = clamp(round_half_away(beta_t * x_t[i]), -128, 127)
S_t    = sum_i q_t[i].
```

`beta_t` is the activation quantization multiplier. A zero row produces
`q_t = 0`, `S_t = 0`, and exact zero matrix output.

### 3. Integer dot product and recovery

The kernel computes unsigned-code products:

```text
D[r,t] = sum_i u[r,i] * q_t[i]
T[r,t] = D[r,t] - S_t
       = sum_i W[r,i] * q_t[i].
```

The active CPU path recovers the approximation with the piecewise post-scale

```text
rho_t = alpha_W / beta_t,  if beta_t > 0
        0,                 if beta_t = 0

y_hat[r,t] = rho_t * T[r,t].
```

This avoids division by zero for all-zero activation rows. The complete contract
documents byte layout, constructor behavior, accumulator bounds, invalid code
3, graph activation, validation gaps, and exactness evidence in
[`docs/NUMERICAL_CONTRACT.md`](docs/NUMERICAL_CONTRACT.md).

### What “contract validated” means

- AVX2 activation quantization matches the scalar `roundf` definition for the
  enumerated unit fixtures in multiplier, I8 bytes, and activation sum.
- Scalar, AVX2-without-VNNI, and native/VNNI kernels implement the same
  unsigned-code correction for the supported model dimensions.
- GEMV/GEMM fixtures pass across 1, 2, and 4 threads, zero rows, and multi-plane
  input.
- On one fixed model, prompt, and runtime configuration, native AVX-VNNI and
  AVX2-without-VNNI produced bitwise-identical full-vocabulary logits.
- Scalar whole-model logits were not bitwise identical; the first observed
  divergence was in RoPE, outside the I2_S matrix kernel.

This evidence does **not** establish universal cross-ISA bitwise identity or
equivalence to the original source-checkpoint implementation. Source-checkpoint
logits, perplexity, or KL certification remains future work.

## Public Interfaces

### CLI

| Command | Purpose |
| --- | --- |
| `celiums-bitnet run` | One-shot prompt generation with streamed stdout |
| `celiums-bitnet serve` | Native HTTP completion server |
| `celiums-bitnet bench` | Prefill/decode throughput in JSONL |
| `celiums-bitnet validate` | Checked model load and metadata report |
| `celiums-bitnet version` | Product, engine, tree, profile, and strict-state identity |

### Experimental C ABI v1

The public header is [`include/celiums/bitnet_runtime.h`](include/celiums/bitnet_runtime.h).
It provides:

- runtime/model/session/request lifetime management;
- product and vendored-engine provenance;
- model load, validation, metadata, descriptions, and chat templates;
- token-to-piece, tokenization, and detokenization;
- session reset, position/context/vocabulary inspection;
- prefill, decode, logits copy, and sampling;
- synchronous generation, token callbacks, stops, and cancellation.

The ABI is experimental while asynchronous ownership, multi-sequence scheduling,
and broader compatibility mature. A Session serializes mutable context
operations. Request destruction must occur after its synchronous generation
call returns.

### Native HTTP API

| Method | Endpoint | Capability |
| --- | --- | --- |
| `GET` | `/health`, `/v1/health` | Health status |
| `GET` | `/v1/models` | Static Celiums model record |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/chat/completions` | Chat-template completion |
| `GET` | `/metrics` | Prometheus metrics |

The server implements an OpenAI-shaped **subset**, not the complete OpenAI API.
Accepted generation fields are `prompt` or `messages`, `max_tokens`,
`temperature`, `top_k`, `top_p`, `seed`, and `stream`. It does not currently
implement continuous batching, tools/function calling, logprobs, penalties,
multiple completions, structured output, or complete usage counts.

Exported metrics:

```text
celiums_bitnet_http_requests_total
celiums_bitnet_completions_total
celiums_bitnet_failures_total
celiums_bitnet_cancelled_total
celiums_bitnet_generated_tokens_total
celiums_bitnet_active_requests
```

## Optional Local AI Gateway

Build the gateway with:

```bash
cmake -S . -B build-gateway \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCELIUMS_BITNET_BUILD_GATEWAY=ON
cmake --build build-gateway --parallel
```

The architecture remains process-separated:

```text
client
  |
  v
celiums-runtime-gateway
  | authenticated UDS                 | HTTP loopback
  v                                   v
celiums-hyphae-sidecar          celiums-bitnet serve
  |                                   |
  v                                   v
Hyphae Native directory         read-only GGUF model
```

Installed gateway binaries:

- `celiums-runtime-gateway`
- `celiums-hyphae-sidecar`
- `celiums-runtime-mcp`

Retrieval modes are `lexical`, `exact`, `ann`, `hybrid_exact`, and
`hybrid_ann`. RAG is opt-in on the completion routes. Hyphae is pinned to
release 1.2.2 at commit
`0471ae25b263fd506da1578068ec57429a6783de`.

See [`docs/LOCAL_AI_GATEWAY.md`](docs/LOCAL_AI_GATEWAY.md) for deployment and
API examples, and [`docs/GATEWAY_THREAT_MODEL.md`](docs/GATEWAY_THREAT_MODEL.md)
for trust boundaries and non-claims.

## CPU Profiles and Hardware

`CELIUMS_BITNET_CPU_PROFILE` is selected at build time:

| Profile | ISA contract | Intended use |
| --- | --- | --- |
| `native` | `-march=native`; inherits the release builder CPU features | Maximum throughput on the same CPU class; not portable |
| `avx2` | x86-64 SSE4.2, AVX, AVX2, FMA, F16C; no AVX-VNNI/AVX-512/AMX | Portable package for compatible x86-64 CPUs |
| `scalar` | AVX/SSE4.2/FMA/F16C disabled in the product profile | Correctness fallback and reference testing |

Certified release evidence is Linux x86-64 CPU-focused. ARM64 NEON/DOTPROD is
present in inherited code but remains outside the strict 0.3.0 certification.
See [`docs/SUPPORTED_HARDWARE.md`](docs/SUPPORTED_HARDWARE.md).

Product builds forcibly disable BLAS, CUDA, HIP/ROCm, Metal, Vulkan, OpenCL,
SYCL, CANN, RPC, and OpenVINO. TL1/TL2 require both their format option and
`CELIUMS_BITNET_EXPERIMENTAL=ON`. Non-strict mode is not implemented.

## Performance Evidence

A controlled historical baseline on a DigitalOcean `c-60-intel` instance with
the certified 2B I2_S model is recorded in the benchmark documents. It includes
one- and multi-thread prompt-processing/decode results and the first strict
GEMV8 optimization. These are development measurements, not 0.3.0 universal
speedup claims. Hardware, compiler, profile, thermal state, memory
configuration, prompt shape, and model hash affect results. See
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md),
[`docs/benchmark-results-2026-08-19.json`](docs/benchmark-results-2026-08-19.json),
and the [performance audit](docs/performance-audit-2026-08-18.md).

The machine-readable JSON records the exact-commit baseline/GEMV8 run. The
separate higher-throughput development-validation table in `BENCHMARKING.md`
belongs to another archived run and is identified by its archive hash there.

## Validation and Evidence

The regular CI matrix covers:

- native, AVX2-without-VNNI, and scalar builds;
- C ABI and installed CMake SDK tests;
- I8_S quantization fixtures;
- I2_S GEMV/GEMM fixtures;
- zero and multi-plane matrix cases;
- ASan and UBSan;
- Python conversion, logits-comparison, and wrapper tests;
- Rust format, check, Clippy, gateway unit/E2E tests, packaging, and licenses.

The model-backed release gate additionally runs:

- strict checked model load;
- one-shot generation;
- native benchmark smoke tests;
- authenticated HTTP and SSE tests;
- full-vocabulary runtime logits comparison against the fixed converted-model
  oracle, requiring bitwise identity.

The oracle is a regression authority for the pinned converted GGUF and runtime
path. It does not replace source-checkpoint logits, perplexity, or KL validation.
See [`docs/EXACTNESS_ORACLE.md`](docs/EXACTNESS_ORACLE.md).

Run the complete local gate on an Ubuntu 24.04 / glibc 2.39 builder with Rust
1.97.1 and `cargo-about` 0.9.2. The model, oracle, and oracle sidecar are external
fixtures:

```bash
CELIUMS_BITNET_TEST_MODEL=/absolute/path/ggml-model-i2_s.gguf \
CELIUMS_BITNET_TEST_ORACLE=/absolute/path/runtime-reference.gguf \
CELIUMS_BITNET_TEST_ORACLE_SIDECAR=/absolute/path/runtime-reference.gguf.json \
JOBS=2 scripts/validate-release.sh
```

The reference oracle is generated and governed by the workflow documented in
[`docs/EXACTNESS_ORACLE.md`](docs/EXACTNESS_ORACLE.md); it is intentionally not
stored in Git.

## Release Packages

Release packaging is supported on Linux x86-64:

```bash
JOBS=8 ./scripts/package-runtime.sh native
JOBS=8 ./scripts/package-runtime.sh avx2
JOBS=8 ./scripts/package-runtime.sh scalar
```

Each profile produces:

```text
celiums-bitnet-runtime-0.3.0-linux-x86_64-<profile>.tar.gz
celiums-bitnet-runtime-0.3.0-linux-x86_64-<profile>.tar.gz.json
```

The archive contains the unified runtime, optional gateway binaries, public C
header/CMake package, private shared libraries, `Cargo.lock`, and complete
license notices. The matching manifest records product and engine commits,
engine tree, CPU profile, C/C++ compiler identities, Rust toolchain, Hyphae commit,
Cargo lock digest, platform, and archive SHA-256.

`native` archives are valid only for the release builder CPU class. `avx2` and
`scalar` close the CPU ISA profile, but binary portability also depends on the
Linux userspace ABI of the release builder. The official 0.3.0 assets are built on
Ubuntu 24.04. The CPU profile does not by itself guarantee compatibility with
older Linux userspaces; inspect the manifest and ELF symbol requirements before
deploying outside that baseline.

Packaging refuses a dirty source tree unless `ALLOW_DIRTY=1` is explicitly
used for non-release smoke testing.

## Explicit Limits and Non-Claims

Celiums BitNet 0.3.0 does not claim:

- support for arbitrary llama.cpp models or quantization formats;
- source-checkpoint numerical certification;
- universal full-model bitwise identity across CPU ISAs;
- GPU or accelerator inference in the supported product build;
- runtime CPU-profile switching or public affinity-mask controls;
- continuous/dynamic batching or asynchronous C request execution;
- a complete OpenAI API implementation;
- built-in embedding inference;
- TLS, rate limiting, or distributed serving;
- that a valid retrieval proof makes generated prose true;
- complete nearest-neighbor results from ANN;
- clustering, replication, or object storage through Hyphae.

The packed code `3` is reserved and invalid under the I2_S contract. The
trusted converter never emits it, but the current runtime loader does not scan
every packed field during ordinary model load. Use trusted converter output,
verify model hashes, and run `celiums-bitnet validate` before deployment.

## Documentation

| Document | Purpose |
| --- | --- |
| [`docs/RUNTIME_PRODUCT.md`](docs/RUNTIME_PRODUCT.md) | Runtime boundary, ABI, installation, CI, packaging |
| [`include/celiums/bitnet_runtime.h`](include/celiums/bitnet_runtime.h) | Experimental public C ABI v1 |
| [`docs/NUMERICAL_CONTRACT.md`](docs/NUMERICAL_CONTRACT.md) | I2_S/I8_S format and equations |
| [`docs/SUPPORTED_HARDWARE.md`](docs/SUPPORTED_HARDWARE.md) | Certified and experimental hardware |
| [`docs/LOCAL_AI_GATEWAY.md`](docs/LOCAL_AI_GATEWAY.md) | Gateway deployment and APIs |
| [`docs/GATEWAY_THREAT_MODEL.md`](docs/GATEWAY_THREAT_MODEL.md) | Gateway security boundaries |
| [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) | Benchmark methodology |
| [`docs/EXACTNESS_ORACLE.md`](docs/EXACTNESS_ORACLE.md) | Raw-logits regression workflow |
| [`docs/EXPERIMENTAL_ROADMAP.md`](docs/EXPERIMENTAL_ROADMAP.md) | Non-certified optimization roadmap |
| [`CHANGES.md`](CHANGES.md) | Release history |

The vendored `3rdparty/llama.cpp` tree retains upstream documentation that may
describe binaries and backends disabled by the Celiums product build. The root
README and product-owned documents define the supported surface.

## Ownership, License, and Attribution

Copyright 2026 Celiums Solutions LLC.

Celiums-authored product code, documentation, tests, build integration, and
tooling outside the vendored runtime are licensed under the
[Apache License 2.0](LICENSE). Original Microsoft BitNet portions retain their
[MIT License](LICENSE-MIT). The `3rdparty/llama.cpp` runtime and Celiums
modifications made inside that tree are distributed under its MIT terms, as
stated in [NOTICE](NOTICE). Hyphae crates and bundled dependencies retain their
own licenses and notices.

This repository does not transfer ownership of Microsoft, ggml, llama.cpp,
model, Hyphae, or third-party code to Celiums Solutions LLC. See
[NOTICE](NOTICE), [UPSTREAM.md](UPSTREAM.md), and the installed release notices
for exact provenance.

## Maintainer

- Mario Gutierrez, Celiums Solutions LLC

Security reports should follow [SECURITY.md](SECURITY.md).

For bugs and usage questions, open a GitHub issue with the product version,
model hash, CPU profile, reproduction command, and relevant logs. Contributions
must preserve the numerical contract, provenance files, upstream licenses, and
the project [Code of Conduct](CODE_OF_CONDUCT.md).
