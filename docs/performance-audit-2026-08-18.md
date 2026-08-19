# BitNet CPU performance audit

Date: 2026-08-18

This audit began from Microsoft BitNet and now documents the baseline used by
the independent Celiums BitNet fork. Historical findings describe the state at
the cited commits; several P0 defects have since been corrected in the Celiums
worktree.

## Scope

- BitNet commit: `0b341e582afbf9e1011f24744b554c96a3477eb5`
- llama.cpp submodule: `390c307752ab78fd8189f359d6954c9ba1be74af`
- Host: Intel Core Ultra 9 285H, 6 P-cores, 8 E-cores, 2 LP E-cores
- ISA available: AVX2 and AVX-VNNI; no AVX-512
- Compiler: GCC 16.1.1 with `-march=native`
- Model: `microsoft/bitnet-b1.58-2B-4T-gguf`, I2_S, 2.4B parameters

## Critical path

For I2_S weights and F32 activations, each matrix multiplication performs:

1. Per-token activation quantization from F32 to I8_S.
2. A thread barrier.
3. I2_S x I8_S GEMV for decode or GEMM for prefill.
4. Offset correction and scaling.

The packed weight code stores `u = w + 1` for `w` in `{-1, 0, 1}`. The integer
kernel computes `D = sum(u_i * q_i)`, so the final result is:

```text
y = (D - sum(q_i)) * weight_scale / activation_scale
```

On this host, the productive I2_S kernel uses AVX-VNNI `vpdpbusd`. The default
llamafile path uses register tiles up to 4x4, which reuses unpacked weights for
prefill. Decode uses a 4x1 tile and cannot amortize unpacking across tokens.

## Implemented experiment

`quantize_row_i8_s()` made two scalar passes and called `roundf()` once per
activation. The experiment adds an AVX2 implementation that preserves the
existing round-half-away-from-zero behavior, packs 32 signed bytes per
iteration, and accumulates activation sums in int32 vectors. Scalar tails and
non-AVX2 platforms keep the original path.

New tests compare scale, every quantized byte, and the activation sum against
the scalar definition for zero, half-integer, trigonometric, vector, and tail
sizes.

## Results

Microbenchmarks were pinned to P-core 0. Times are medians after warmup.

| Inner dimension | Activation rows | Scalar | AVX2 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2,560 | 1 | 7.8 us | 1.2 us | 6.52x |
| 6,912 | 1 | 20.2 us | 2.1 us | 9.50x |
| 13,824 | 1 | 39.4 us | 3.6 us | 11.09x |
| 2,560 | 128 | 941.0 us | 126.8 us | 7.42x |
| 6,912 | 128 | 2,549.7 us | 242.5 us | 10.52x |
| 13,824 | 128 | 5,218.8 us | 443.6 us | 11.76x |

The 1M-element P-core test improved from 2,933.1 us to 249.8 us, or 11.74x.
On LP E-core 15 it improved from 7,747.6 us to 1,056.3 us, or 7.33x. These
mobile-host figures have been superseded by the controlled C-60 experiment
below for end-to-end conclusions.

## Controlled DigitalOcean C-60 experiment

The experiment was repeated on a temporary DigitalOcean `c-60-intel` Droplet:

- Region: NYC3
- 60 dedicated vCPUs, one thread per core
- Intel Xeon Platinum 8358 at 2.60 GHz
- 120 GiB RAM
- One NUMA node
- AVX2, AVX-512, AVX-512 VNNI, and AVX-512 VBMI exposed by KVM
- Ubuntu 24.04, kernel 6.8.0-124
- GCC 13.3.0 with `-march=native`

Both variants used identical commits, compiler options, model bytes, strict
CPU affinity over cores 0-59, poll 100, batch/ubatch 128, five samples per
invocation, and five alternated invocations per thread count. The model SHA256
was `e23b16fa81b890e8b65e676262b645e8ffa5ae1f6df89dadaf793246826bbd90`
after correcting `general.file_type` from legacy value 40 to `MOSTLY_I2_S` (41).

### Activation quantization microbenchmark

Times are medians of 200 samples pinned to core 0.

| Inner dimension | Rows | Baseline | SIMD | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2,560 | 1 | 24.67 us | 1.80 us | 13.73x |
| 6,912 | 1 | 64.86 us | 2.85 us | 22.74x |
| 13,824 | 1 | 128.91 us | 4.55 us | 28.30x |
| 2,560 | 128 | 3,185.10 us | 212.56 us | 14.99x |
| 6,912 | 128 | 8,387.56 us | 344.44 us | 24.35x |
| 13,824 | 128 | 16,680.15 us | 608.63 us | 27.41x |
| 1,048,576 | 1 | 9,866.53 us | 320.72 us | 30.76x |

### End-to-end results

The table reports the median of all 25 `samples_ts` values for each cell.
Intervals are 95% bootstrap confidence intervals for the speedup ratio.

| Threads | Phase | Baseline tok/s | SIMD tok/s | Speedup | 95% CI |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 1 | pp128 | 41.09 | 57.33 | 1.395x | [1.372, 1.412] |
| 1 | tg128 | 8.60 | 8.85 | 1.029x | [1.014, 1.047] |
| 8 | pp128 | 302.15 | 415.35 | 1.375x | [1.346, 1.394] |
| 8 | tg128 | 32.75 | 44.10 | 1.347x | [1.342, 1.350] |
| 30 | pp128 | 721.54 | 1,010.90 | 1.401x | [1.262, 1.472] |
| 30 | tg128 | 51.43 | 76.26 | 1.483x | [1.453, 1.500] |
| 60 | pp128 | 836.00 | 1,029.90 | 1.232x | [1.223, 1.240] |
| 60 | tg128 | 46.61 | 78.73 | 1.689x | [1.685, 1.693] |

Scaling peaked at 60 threads for prefill. Baseline decode peaked around 30
threads and regressed at 60, while the SIMD version continued improving through
60 threads. This confirms that scalar activation quantization was both a direct
cost and a thread-scaling bottleneck.

A separate `perf stat` pp128 run at 30 threads showed that SIMD retired about
103.9B instructions versus 118.0B for baseline, a reduction of 11.9%. The
whole-run IPC changed from 1.65 to 1.39, so further work should reduce cache
misses and synchronization rather than treating instruction count alone as the
objective.

The raw JSONL samples, microbenchmarks, system manifest, assembly, test output,
execution order, and perf counters are archived in
`docs/bitnet-c60-results-2026-08-18.tar.gz` with SHA256
`5d1b728230b035c0e160db884db3154c3a15f671e18f639ed58723faed9a44bf`.

## Follow-up implementation review

A second temporary C-60 was used to review the SIMD implementation rather than
only benchmark it. The review found a subtle weakness in the original
`v + copysign(0.5, v)` shortcut: the F32 addition can round a non-tie input to
an exact tie before integer conversion. The implementation now computes
`roundf()` explicitly as truncation plus a signed unit adjustment when the
absolute fractional part is at least 0.5.

The reviewed implementation was checked against scalar `roundf()` with exact
scale, byte, sum, and guard-byte comparisons for:

- 31/32/33, 63/64/65, 127/128/129, 2,560, 6,912, and 13,824 elements.
- Zero rows and deterministic random inputs.
- Values immediately below, exactly on, and immediately above positive and
  negative half ties.
- GCC 13 and Clang 18 native builds.
- AVX2-only and scalar standalone builds.
- Native and AVX2-only full CMake tests.

The full scalar CMake build exposed a pre-existing unrelated build defect:
`gemm-config.h` leaves `ROW_BLOCK_SIZE` and `COL_BLOCK_SIZE` undefined when all
x86 SIMD macros are disabled. The standalone scalar test still validates the
unchanged fallback directly.

The corrected implementation retained the end-to-end improvement on the
second C-60:

| Threads | Phase | Baseline tok/s | Reviewed SIMD tok/s | Speedup |
| ---: | :--- | ---: | ---: | ---: |
| 1 | pp128 | 42.16 | 57.81 | 1.371x [1.366, 1.375] |
| 1 | tg128 | 6.95 | 7.40 | 1.065x [1.040, 1.066] |
| 8 | pp128 | 307.48 | 421.04 | 1.369x [1.358, 1.385] |
| 8 | tg128 | 32.57 | 43.30 | 1.329x [1.326, 1.334] |
| 30 | pp128 | 791.62 | 1,091.67 | 1.379x [1.323, 1.419] |
| 30 | tg128 | 53.56 | 80.79 | 1.508x [1.505, 1.511] |
| 60 | pp128 | 841.87 | 1,033.46 | 1.228x [1.223, 1.234] |
| 60 | tg128 | 49.18 | 84.45 | 1.717x [1.713, 1.722] |

The second review archive, including the exact patch, standalone harness,
proof notes, native/AVX2 assembly, test outputs, microbenchmarks, and raw JSONL,
is `docs/bitnet-c60-review-2026-08-18.tar.gz` with SHA256
`78b3668482a8229bbf4a1988b2b93087ba4674e6f620d8c534b259d242e3ff65`.

## Correctness and infrastructure findings

These findings should precede more aggressive kernel work:

1. The BitNet graph currently selects SiLU for the FFN, while the official
   model requires ReLU squared. GitHub issue #588 and PR #604 track the measured
   perplexity regression and fix.
2. I2_S, TL1, and TL2 are omitted from the standard backend operation tests.
   The existing quantization test prints their names but did not execute I2_S
   round-trip, dot-product, GEMV, or GEMM validation.
3. `quantize_i2_s()` packs four consecutive weights per byte, but the kernels
   and Python converter expect positions `[i, i+32, i+64, i+96]` in each byte.
4. TL2 batch output uses no batch offset, so later batches overwrite earlier
   results. PR #521 tracks this issue.
5. The setup script offers TL1/TL2 but compiles both specialized paths OFF.
6. The setup script calls `llama-quantize ... I2_S`, although the included
   quantizer does not register I2_S as a CLI option.
7. `e2e_benchmark.py` forces batch size 1 and exits with status 1 after a
   successful benchmark. It therefore neither exercises prompt GEMM correctly
   nor behaves as a reliable automation target.
8. The documented tuning macros do not control the default x86 llamafile
   I2_S path. Tuning the header can therefore optimize a non-productive path.

## Prioritized roadmap

### P0: establish correctness

- Resolve the ReLU-squared graph regression before claiming lossless inference.
- Add model-free I2_S tests for packing, dequantization, dot products, GEMV,
  GEMM, tails, zero activations, saturation, and multiple threads.
- Consolidate the existing I2_S BLAS, TL2 batch, workspace, and server-race
  fixes rather than creating duplicate patches.
- Make unsupported TL shapes and batch sizes fail explicitly.

### P1: optimize the active x86 path

- Keep activation quantization vectorized and measure it inside full MUL_MAT.
- Benchmark separate 4x1 decode and 4x4 prefill tiles by model shape.
- Reduce decode unpack cost with a tested repacked layout or an ELUT/LUT design;
  include the memory and load-time cost in the comparison.
- Add runtime dispatch for scalar, AVX2, and AVX-VNNI instead of relying only on
  a native build. Evaluate AVX-512/VBMI only on matching hardware.

### P1: schedule heterogeneous CPUs correctly

- Detect P, E, and LP E cores and benchmark separate policies for prefill and
  decode.
- Compare P-only affinity against weighted static partitioning and work
  stealing. A slow core must not hold every matrix-operation barrier.
- Persist tuning by CPU signature, model shape, phase, thread count, and cache
  topology instead of recompiling one global header.

### P2: extend the mathematical design

- Evaluate BitNet a4.8-style activation quantization and sparse outlier paths;
  this requires compatible trained checkpoints and cannot be an inference-only
  substitution.
- Explore ternary signed-position algebra: split nonzero weights into positive
  and negative index streams and compare sparse gather/add-subtract kernels
  against packed VNNI as zero density changes.
- Evaluate balanced-ternary ELUT groups, but include LUT construction,
  register pressure, cache footprint, and batch reuse in the objective.
- Treat the 3-bit KV cache and conditional parameter activation from a4.8 as
  architecture-level work, not a drop-in kernel patch.

## Benchmark discipline

Every optimization should report:

- Exact commits, compiler, ISA, CPU topology, affinity, power mode, and model.
- Separate prefill and decode results over realistic shapes and token counts.
- Median, dispersion, samples, and thermal warmup.
- Kernel-only and end-to-end measurements.
- Exact byte-level output tests, logits/KL or perplexity, and failure thresholds.
- Packing/load time and extra memory when changing weight layout.

The immediate opportunity is not another isolated microkernel. It is a
correctness-tested, phase-aware path that combines vectorized activation
quantization, VNNI kernels, and hybrid-core scheduling.
