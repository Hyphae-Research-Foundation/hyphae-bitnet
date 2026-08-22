# Celiums BitNet Runtime

Celiums BitNet Runtime is the public product layer over the vendored
BitNet engine at `3rdparty/llama.cpp`. Applications should use the Celiums
command and API instead of depending directly on llama.cpp or GGML interfaces.

## Commands

```bash
celiums-bitnet run --model model.gguf --prompt "Hello"
celiums-bitnet serve --model model.gguf --host 127.0.0.1 --port 8080
celiums-bitnet bench --model model.gguf -p 128 -n 128
celiums-bitnet validate --model model.gguf
celiums-bitnet version
```

Commands default to the certified `bitnet` model family. A pre-quantized
PrismML Bonsai 27B `Q1_0` model can be selected explicitly with
`--model-family bonsai`. Bonsai `Q2_0` is not supported.

`run`, `bench`, and `serve` use the Celiums Runtime C API directly. The native
server exposes `/health`, `/v1/health`, `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`. The `llama-*` binaries remain optional compatibility
tools and `llama-bench` remains the performance oracle during the transition.
The native server supports an OpenAI-shaped SSE subset for `stream=true`, applies the
model chat template, exports Prometheus metrics at `/metrics`, and cancels
generation when a streaming client disconnects. Continuous batching remains
explicit follow-up work rather than being silently emulated.

`celiums-runtime-gateway` is an optional separate Rust process. It composes the
loopback native server with authenticated Hyphae UDS for RAG, memory, receipts,
registry/evidence, hybrid retrieval, and proofs. It does not expand the runtime
C ABI and does not link Hyphae into llama.cpp or GGML. See
[LOCAL_AI_GATEWAY.md](LOCAL_AI_GATEWAY.md).

`serve` refuses non-loopback hosts unless an API key is configured or the
operator explicitly passes `--allow-unauthenticated-remote`.
Set `CELIUMS_BITNET_BUILD_SERVER=OFF` to omit the public `serve` subcommand.
`CELIUMS_BITNET_API_KEY` is the canonical environment variable. `LLAMA_API_KEY`
is accepted for migration and, unlike the unsafe 0.2.0 precheck, enforces the
key on every request. The policy is enforced inside
the native server, including when `celiums-runtime-server` is run directly.

## C API

The experimental v1 C API is declared in
`include/celiums/bitnet_runtime.h`. It provides opaque Runtime and Model
handles, version queries, model loading, and model metadata without exposing
`llama_model`, `llama_context`, `ggml_tensor`, or C++ standard-library types.

The API now provides opaque Runtime, Model, Session, and Request handles;
tokenization and detokenization; prefill and single-token decode; copied logits;
sampling; synchronous generation; streaming callbacks; stop sequences; and
cooperative cancellation. A Session serializes its mutable context operations.
Models and runtimes remain alive while dependent handles exist. The API remains
0.x while multi-sequence scheduling and asynchronous request ownership mature.
`celiums_bitnet_request_cancel()` is thread-safe; Request destruction must occur
after its synchronous generation call has returned.

`celiums_bitnet_model_validate_strict()` performs a checked model load and
accepts only `bitnet-b1.58` metadata marked as `MOSTLY_I2_S` (file type 41).
The stronger Python conversion validator additionally checks required
projections, rank, dimensions, byte counts, and scales. Neither path currently
scans every packed field for reserved code 3.

The additive `celiums_bitnet_model_load_family()` and
`celiums_bitnet_model_validate_family()` APIs expose the inherited CPU text
path for `qwen35` models marked as `MOSTLY_Q1_0` (file type 40). Callers must
select `CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0` explicitly. This does
not extend the strict I2_S numerical contract to Bonsai and does not add model
conversion, vision, GPU offload, speculative decoding, or Prism `Q2_0`.
Native x86 builds keep Q1_0 as bit-packed 4×8 VNNI panels and reuse each
panel across eight activation rows in prefill GEMM. AVX2 and scalar builds
keep the ordinary Q1_0 path. ARM with i8mm expands Q1_0 to q8_0 ±1 so the
existing i8mm kernels run.

A runtime RAM budget (`celiums_bitnet_runtime_options.ram_budget_bytes`,
`--ram-budget-bytes`) bounds both the in-RAM compute image and extra decode
slots (`n_seq`). Zero selects an automatic cap: half of host RAM, never more
than 90%, always leaving at least 4 GiB or 10% free so the host stays usable
for serving. Model load and session create fail closed with
`CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED` instead of allocating past the
cap. `--compute-layout 1` (the default) materializes the ISA compute image:
ARM with i8mm expands Q1_0 1-bit weights to q8_0 4×8 ±1; x86 keeps bit-packed
Q1 4×8 VNNI panels. `run`, `bench`, and `serve` all honor these options.

## Build Profiles

`CELIUMS_BITNET_CPU_PROFILE` selects one of:

- `native`: optimize for the build host.
- `avx2`: portable x86-64 SSE4.2/AVX/AVX2/FMA/F16C without AVX-VNNI or AVX-512.
- `scalar`: scalar reference path without AVX.

The `native` release archive is host-class-specific because it uses
`-march=native`. Use `avx2` or `scalar` for CPU-ISA portability. Official Linux
archives are built on Ubuntu 24.04; profile names do not promise compatibility
with older glibc or libstdc++ userspaces.

Strict I2_S is the supported product conversion. Pre-quantized Bonsai Q1_0 is
an explicit inherited runtime compatibility profile. TL1/TL2, quantized
embeddings, GPU, Swift, and Android remain experimental or inherited and are
outside the release gate. Strict mode is currently mandatory. CMake rejects
`CELIUMS_BITNET_STRICT=OFF`, because a non-strict product profile has not been
implemented. TL1 and TL2 require `CELIUMS_BITNET_EXPERIMENTAL=ON`.

## Installation

The default installation contains:

- `bin/celiums-bitnet`.
- `bin/celiums-runtime-gateway`, `bin/celiums-hyphae-sidecar`, and
  `bin/celiums-runtime-mcp` in release packages or gateway-enabled builds.
- `include/celiums/bitnet_runtime.h`.
- `lib*/celiums-bitnet-runtime/` with the public library and private engine.
- Apache-2.0, MIT, BSD-3-Clause, and attribution notices.

Set `CELIUMS_BITNET_INSTALL_COMPAT=ON` only when the inherited llama.cpp tools
and development packages are required.
The public SDK supports shared libraries. Static builds are used for internal
sanitizer validation and intentionally do not install the SDK.

## CI and packages

`.github/workflows/runtime-ci.yml` validates native, AVX2, scalar, and
ASan/UBSan builds, the installed CMake consumer, server security policy, and a
package smoke test. Model-backed generation, HTTP, benchmark, and exactness
checks are a separate release gate. A manual GitHub dispatch runs that gate only
when the self-hosted runner and model/oracle variables are configured; local
release publication must run `scripts/validate-release.sh` and retain its
result. `scripts/package-runtime.sh` creates a Linux
x86_64 profile-specific
archive and `utils/release_manifest.py` records product/engine commits, C and
C++ compiler identities, platform, profile, model hash when supplied, and
artifact hashes.
Gateway-enabled manifests also record the Rust compiler. Hyphae is pinned by
exact Git commit in the gateway lockfile.

The engine is vendored in-tree at `3rdparty/llama.cpp` (single-repository
layout). `3rdparty/llama.cpp/ENGINE_COMMIT` records the historical engine commit
and `cmake/ENGINE_TREE` records the exact vendored Git tree. The archived engine
repository is provenance only and is not accessed by clone or build:
`https://github.com/celiumsai/celiums-bitnet-llama`:

```text
Celiums BitNet        product, API, packaging, CI, integration tests,
                      and the vendored engine, graph, GGML, and I2_S kernels
```
