# Supported Hardware and Build Profiles

## Certified Linux x86-64 CPU Profiles

- `native`: release-builder CPU features, including AVX-VNNI when available;
- `avx2`: SSE4.2, AVX, AVX2, FMA, and F16C, with AVX-VNNI/AVX-512/AMX disabled;
- `scalar`: AVX/SSE4.2/FMA/F16C disabled as a correctness fallback.

ARM NEON/DOTPROD remains under validation. Celiums standardizes the persisted
I2_S layout on blocks of 128 weights regardless of runtime ISA.

Validated releases include native, portable AVX2-without-VNNI, and scalar
archives. The native archive is host-class-specific because it uses
`-march=native`. The scalar build is a correctness fallback, not a performance
target.

Official Linux archives are built on Ubuntu 24.04. CPU portability and Linux
userspace ABI portability are separate contracts; profile names do not promise
compatibility with older glibc or libstdc++ versions.

## Experimental

- AVX-512 VNNI/VBMI kernels
- AMX prefill kernels
- TL1/TL2 lookup-table kernels
- CUDA, ROCm, and Metal backends

Experimental backends are not part of the strict compatibility guarantee until
they pass the same numerical and model-level test matrix.
