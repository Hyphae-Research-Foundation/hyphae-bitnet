# Supported Hardware

## Strict CPU

- x86-64 AVX2
- x86-64 AVX-VNNI
- Scalar reference and fallback builds

ARM NEON/DOTPROD remains under validation. Celiums standardizes the persisted
I2_S layout on blocks of 128 weights regardless of runtime ISA.

Validated release builds must include both an AVX2/VNNI build and a build with
AVX disabled. The scalar build is a correctness fallback, not a performance
target.

## Experimental

- AVX-512 VNNI/VBMI kernels
- AMX prefill kernels
- TL1/TL2 lookup-table kernels
- CUDA, ROCm, and Metal backends

Experimental backends are not part of the strict compatibility guarantee until
they pass the same numerical and model-level test matrix.
