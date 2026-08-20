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

`run`, `bench`, and `serve` use the Celiums Runtime C API directly. The native
server exposes `/health`, `/v1/health`, `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`. The `llama-*` binaries remain optional compatibility
tools and `llama-bench` remains the performance oracle during the transition.
The native server supports OpenAI-compatible SSE for `stream=true`, applies the
model chat template, exports Prometheus metrics at `/metrics`, and cancels
generation when a streaming client disconnects. Continuous batching remains
explicit follow-up work rather than being silently emulated.

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

`celiums_bitnet_model_validate_strict()` performs a full tensor-checking load
and accepts only `bitnet-b1.58` models marked as `MOSTLY_I2_S` (file type 41).

## Build Profiles

`CELIUMS_BITNET_CPU_PROFILE` selects one of:

- `native`: optimize for the build host.
- `avx2`: portable x86 AVX2 without AVX-VNNI or AVX-512.
- `scalar`: scalar reference path without AVX.

The `native` release archive is host-class-specific because it uses
`-march=native`. Use `avx2` or `scalar` for portable distribution.

Strict I2_S is the supported product conversion. TL1/TL2, quantized embeddings,
GPU, Swift, and Android remain experimental or inherited and are outside the
release gate. Strict mode is currently mandatory. CMake rejects
`CELIUMS_BITNET_STRICT=OFF`, because a non-strict product profile has not been
implemented. TL1 and TL2 require `CELIUMS_BITNET_EXPERIMENTAL=ON`.

## Installation

The default installation contains:

- `bin/celiums-bitnet`.
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
checks are a separate release gate. `scripts/package-runtime.sh` creates a Linux
x86_64 profile-specific
archive and `utils/release_manifest.py` records product/engine commits,
compiler, platform, profile, model hash when supplied, and artifact hashes.

The engine is vendored in-tree at `3rdparty/llama.cpp` (single-repository
layout). `3rdparty/llama.cpp/ENGINE_COMMIT` records the historical engine commit
and `cmake/ENGINE_TREE` records the exact vendored Git tree. The archived engine
repository is provenance only and is not accessed by clone or build:
`https://github.com/celiumsai/celiums-bitnet-llama`:

```text
Celiums BitNet        product, API, packaging, CI, integration tests,
                      and the vendored engine, graph, GGML, and I2_S kernels
```
