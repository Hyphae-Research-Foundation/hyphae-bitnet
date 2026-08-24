# Changelog

## 0.3.2 - 2026-08-24

- Restore the v0.3.0 LP64 layouts and by-value return ABI for the original
  Runtime, Model, and Session option structs. Add explicit `_ex` option structs,
  caller-buffer initializers, consuming APIs, and RAM estimators for the 0.3.1
  RAM budget, compute-layout, `n_seq`, and model-family behavior.
- Compile and run a frozen-header v0.3.0 client against the current shared
  library in CTest and installed-artifact validation. Binaries built against the
  broken v0.3.1 expanded option-return layouts must rebuild. No v0.3.1 binary
  release was published; the `.so.0` identity continues the released v0.3.0 ABI.
- Correct the ARM packed compute-layout RAM estimate from 8x to 1x packed model
  bytes and hide non-API C++ symbols by default.
- Add an experimental Neoverse V2 SVE2 VL=128 packed Q1 GEMV path with
  eight-output activation reuse and static contiguous decode scheduling. NEON
  remains the default after whole-model A/B measurement; `GGML_Q1_SVE2=1`
  enables the experimental path.
- Reuse an exact Q8 activation packing between consecutive eligible Bonsai Q1
  `CPU_REPACK` matrix multiplications with the same immutable F32 projection
  source and packing layout. Non-eligible graph nodes may occur between the
  projections; another eligible key replaces the one-entry cache. Workspace is
  sized for the largest eligible candidate. The cache is
  execution-owned, reset for every graph compute, and can be disabled with
  `GGML_Q1_ACT_CACHE=0` for same-binary comparisons.

## 0.3.1 - 2026-08-22

- Treat host RAM as a serving lever under an explicit cap. Model load and
  session create fail closed with `CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED`
  instead of allocating past `--ram-budget-bytes`. The runtime keeps an
  aggregate `reserved_bytes` ledger.
- Materialize an ISA compute layout for Bonsai Q1_0: ARM i8mm expands 1-bit
  weights to q8_0 ±1; x86 keeps bit-packed 4×8 VNNI panels. Packed GGUF stays
  the durable store.
- Reuse each Q1 4×8 weight panel across eight activation rows in prefill
  GEMM (AVX-512 VNNI and the generic kernel). Decode remains GEMV.
- Add in-tree `celiums-exact` for 2P−S, Q1 4×8 pack, the tile planner, and
  I2_S `(D−S)ρ` recovery. Kernels and oracles call the same math.
- Reject `n_seq > 1` at session create and CLI/server/bench parse. Decode
  always uses `seq_id 0`; estimates may still use `n_seq` for planning.
- Close HTTP audit holes: stream `max_tokens` captured by value, no `setenv`
  of the API key, 4 MiB body cap, `max_tokens` ≤ 4096, ≤ 8 concurrent
  generations.
- Expose `--model-family bonsai`, `--compute-layout`, `--n-seq`, and
  `--ram-budget-bytes` on `run`, `bench`, and `serve`.
- Record AWS metal receipts for Bonsai Q1_0 on r8g.metal-24xl (Graviton 4)
  and i7i.metal-24xl (Xeon 8559C). Q1 oracle passed on Graviton 4
  (`c8g.2xlarge`, Neoverse-V2, i8mm).

## 0.3.0 - 2026-08-20

- Added the optional Celiums Runtime Gateway with authenticated Hyphae UDS,
  OpenAI-compatible BitNet proxying, bounded local RAG, memory, BM25, exact/ANN
  vectors, hybrid retrieval, streaming, idempotent request state, and semantic
  cache.
- Added proof-bearing retrieval, durable generation receipts, offline semantic
  proof verification, and model/retrieval/generation provenance.
- Added dataset, artifact, lineage, run, evaluation, contamination, and
  hard-negative registry surfaces for CTT-1 development.
- Added a separate MCP stdio adapter, least-privilege Hyphae bootstrap, threat
  model, CMake opt-in, release packaging, Rust CI, and native end-to-end tests.
- Reorganized the release documentation around the 0.3.0 support matrix,
  formalized the I2_S/I8_S equations and evidence scope, and documented Ubuntu
  24.04 as the official Linux package build environment.

## 0.2.1 - 2026-08-20

- Completed the single-repository migration with an exact vendored engine tree
  hash and consistent product, engine, and exactness provenance.
- Centralized remote-bind and API-key enforcement in the native server, with
  `CELIUMS_BITNET_API_KEY` as the canonical variable. `LLAMA_API_KEY` is accepted
  to close the unsafe 0.2.0 precheck, and now enforces authentication.
- Reduced default builds to the Celiums runtime surface and disabled inherited
  server, application, and Web UI downloads unless compatibility is requested.
- Made server-disabled builds omit the public command and server targets.
- Exported a relocatable shared CMake SDK and made static builds internal-only.
- Closed the AVX2 ISA contract, strengthened CI and product tests, and limited
  release packages to validated Linux x86_64 profiles.
- Corrected an x86 quantization read that UBSan identified as unaligned.
- Restricted supported setup and wrapper options to the strict I2_S product path.

## 0.2.0 - 2026-08-19

- Added the Celiums BitNet Runtime product layer: `celiums-bitnet`
  with `run`, `serve`, `bench`, `validate`, and `version` commands
  built on the experimental v1 C API.
- Added the native C API surface: Runtime, Model, Session, and
  Request handles; tokenization and detokenization; prefill and
  single-token decode; copied logits; sampling; synchronous
  generation; streaming callbacks; stop sequences; chat template
  application; and cooperative, thread-safe cancellation.
- Added the native HTTP server with OpenAI-compatible
  `/v1/completions` and `/v1/chat/completions`, SSE streaming for
  `stream=true` with the `data: [DONE]` terminator, model chat
  template rendering through the engine Jinja path, Prometheus
  metrics at `/metrics`, and generation cancellation when a
  streaming client disconnects.
- Added loopback-only binding by default; remote hosts require an
  API key or an explicit unauthenticated-remote opt-in.
- Added `CELIUMS_BITNET_BUILD_SERVER` to omit the server and made
  strict mode the only accepted build policy.
- Added native, AVX2 (no AVX-VNNI), scalar, and ASan/UBSan CI
  validation for the runtime tests.
- Added the exactness oracle: deterministic full-vocabulary logits
  captures with prompt positions and named-tensor probes, recording
  model, binary, prompt, and build metadata with SHA-256 hashes.
  Native AVX-VNNI and AVX2-without-VNNI captures were bitwise
  identical on the fixed oracle prompt.
- Added the runtime logits comparator, which verifies the runtime
  C API against a final-position logits capture bitwise.
- Added `scripts/package-runtime.sh` and `utils/release_manifest.py`
  for profile-specific release archives with product/engine commit,
  compiler, and artifact-hash manifest entries.
- Kept the `llama-*` compatibility tools available behind
  `CELIUMS_BITNET_INSTALL_COMPAT=ON`.
- Consolidated the pinned engine into the product repository
  (`3rdparty/llama.cpp`), recording the engine snapshot commit in
  `3rdparty/llama.cpp/ENGINE_COMMIT`. The archived engine repository is
  `https://github.com/celiumsai/celiums-bitnet-llama`.

## 0.1.0 - 2026-08-18

- Established the independent Celiums BitNet fork.
- Added explicit upstream attribution and a Celiums security policy.
- Added a strict numerical contract for I2_S and I8_S inference.
- Added exact AVX2 activation quantization and edge-case tests.
- Corrected the BitNet b1.58 FFN activation from SiLU to squared ReLU.
- Corrected I2_S packing, scalar dot-product semantics, and non-VNNI overflow.
- Added zero-activation handling, post-scale precomputation, and contiguous
  activation-workspace scheduling.
- Added a phase-aware hybrid Intel policy that uses performance cores for
  single-token work and all available cores for batch work.
- Disabled invalid generic BLAS dequantization for I2_S/TL formats.
- Repaired scalar builds, CMake installation, setup, inference wrappers, and
  end-to-end benchmark batching.
- Added controlled CPU benchmark and implementation-review artifacts.
- Added structural GGUF validation for architecture, file type, required block
  projections, positive dimensions, packed byte counts, and finite scales.
- Added separate attached decode and batch threadpools so hybrid affinity is
  effective in both `llama-cli` and `llama-server`.
- Validated native, scalar, AVX2-without-VNNI, and ASan/UBSan builds.
- Licensed Celiums contributions under Apache-2.0 while preserving upstream
  MIT licenses and attribution.
