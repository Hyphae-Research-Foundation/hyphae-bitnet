# Experimental Roadmap

The strict CPU path is the release baseline. The following work remains behind
`CELIUMS_BITNET_EXPERIMENTAL` until it passes the strict numerical tests and a
controlled benchmark matrix.

## Kernel Tiling

- AVX2 and AVX-VNNI: compare 4x1, 2x2, 4x2, and 4x4 tiles.
- AVX-512 VNNI/VBMI: add real ZMM kernels and test 8x1 decode tiles.
- Reject a tile if register spills erase its arithmetic advantage.

## Shared Activation Quantization

- Quantize once for Q/K/V projections.
- Quantize once for FFN gate/up projections.
- Preserve independent weight scales and exact I8_S bytes.
- Measure the reduced barriers and workspace lifetime.

This requires an explicit reusable quantized-activation graph value or a
multi-matrix CPU operator. It is not enabled in 0.1 development builds.

## Exact FFN Channel Compaction

Microsoft's 2B checkpoint contains gate/up channels whose complete rows are
zero. A future experimental loader can compact those rows and the matching down
columns while retaining the original logical RMSNorm denominator.

No approximate pruning is allowed in strict mode.

## Compact Ternary Formats

- Three trits in five bits (about 1.667 bpw).
- Five trits per byte (1.6 bpw).
- Exact TQ1-style encoding with the existing global I2_S scale.

Any format must include conversion time, mmap behavior, RSS, workspace, prefill,
decode, and model-quality results.

## AMX

AMX is a prefill-only candidate. Persisted weights remain at 2 bpw; panels may
be expanded temporarily to U8 before `TDPBUSD`. An 8-bit persistent copy is not
acceptable for the strict memory objective.

## GPU

The inherited CUDA code remains experimental and shape-specific. Planned work:

1. Generic I2_S GEMV shapes and explicit errors.
2. Fat binaries selected for the target GPU architecture.
3. I32 reference tests and scale validation.
4. Batched/prefill kernels.
5. GGML integration.
6. Equivalent ROCm and Metal paths where hardware is available.

GPU validation uses one accelerator at a time. NVIDIA work targets a B300 x1
when capacity is available; AMD work uses an available x1 accelerator.
