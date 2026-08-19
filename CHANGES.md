# Changelog

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
