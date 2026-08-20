# Celiums BitNet Runtime

Celiums BitNet Runtime is the public product layer over the pinned
`celiums-bitnet-llama` engine. Applications should use the Celiums command and
API instead of depending directly on llama.cpp or GGML interfaces.

## Commands

```bash
celiums-bitnet run --model model.gguf --prompt "Hello"
celiums-bitnet serve --model model.gguf --host 127.0.0.1 --port 8080
celiums-bitnet bench --model model.gguf -p 128 -n 128
celiums-bitnet validate --model model.gguf
celiums-bitnet version
```

`run`, `serve`, and `bench` currently delegate to the pinned engine's mature
implementations. Their Celiums entry points are the supported product surface;
the `llama-*` binaries are optional compatibility tools.

`serve` refuses non-loopback hosts unless an API key is configured or the
operator explicitly passes `--allow-unauthenticated-remote`.
Set `CELIUMS_BITNET_BUILD_SERVER=OFF` to omit the public `serve` subcommand.
The inherited CLI still compiles its private server-context dependency during
this transition.

## C API

The experimental v1 C API is declared in
`include/celiums/bitnet_runtime.h`. It provides opaque Runtime and Model
handles, version queries, model loading, and model metadata without exposing
`llama_model`, `llama_context`, `ggml_tensor`, or C++ standard-library types.

Generation and Session handles will be added incrementally after their
ownership, cancellation, and concurrency contracts are fixed. The API remains
0.x until that lifecycle is complete.

`celiums_bitnet_model_validate_strict()` performs a full tensor-checking load
and accepts only `bitnet-b1.58` models marked as `MOSTLY_I2_S` (file type 41).

## Build Profiles

`CELIUMS_BITNET_CPU_PROFILE` selects one of:

- `native`: optimize for the build host.
- `avx2`: portable x86 AVX2 without AVX-VNNI or AVX-512.
- `scalar`: scalar reference path without AVX.

Strict mode is currently mandatory. CMake rejects
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

The two-repository boundary remains:

```text
celiums-bitnet        public product, API, packaging, CI, and integration tests
celiums-bitnet-llama  pinned internal engine, graph, GGML, and I2_S kernels
```
