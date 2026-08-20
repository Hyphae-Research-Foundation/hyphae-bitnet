# Changelog

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
