# Graviton4 Bonsai Q1 Packed Repack

The current ARM Q1 path keeps weights bit-packed and decodes signs inside
NEON DOTPROD/I8MM kernels. This replaces the earlier 7.56x Q1-to-Q8 compute
image while preserving the exact `+1/-1` interpretation.

The measurements below used an AWS `r8g.metal-24xl`, 96 physical Neoverse V2
cores, Bonsai 27B Q1_0 SHA256
`17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`,
128 prompt tokens, 128 generated tokens, and one timed repetition in the thread
sweep.

| Threads | mmap pp128 | packed pp128 | Speedup | mmap tg128 | packed tg128 | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.52 | 2.73 | 5.22x | 0.50 | 0.99 | 1.97x |
| 72 | 29.50 | 104.80 | 3.55x | 17.30 | 22.42 | 1.30x |
| 96 | 37.60 | 122.95 | 3.27x | 18.42 | 21.91 | 1.19x |

The full packed thread sweep peaked at 96 threads for prefill and 72 threads
for decode:

| Threads | pp128 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 1 | 2.73 | 0.99 |
| 8 | 20.92 | 6.83 |
| 24 | 53.30 | 14.58 |
| 48 | 84.64 | 20.01 |
| 72 | 104.80 | 22.42 |
| 96 | 122.95 | 21.91 |

Peak RSS was 7.04 GiB for packed repack and 3.83 GiB for mmap without the
compute layout. The previous int8 expansion consumed about 29 GiB and reached
147.0 pp128 but only 12.11 tg128 at 96 threads. Packed repack therefore gives
up about 16% of that expansion's peak prefill while using about one quarter of
its process memory, and improves decode by about 81%.

## PMU and runtime profile

The 96-thread decode PMU runs show that packed repack reduces retired
instructions by 18.6% and cycles by 15.0% relative to mmap. It also shifts the
bottleneck toward memory: backend-stalled cycles rise from 28.8% to 37.4%, and
LLC read misses rise from 4.31B to 4.49B in the profiled runs. A frame-pointer
`perf record` attributes 35.35% of sampled cycles directly to
`ggml_gemv_q1_0_4x8_q8_0`; OpenMP/libgomp wait and scheduling sites account
for most of the remaining large samples.

The fused Gated DeltaNet operator is material but not dominant:

| Phase | Threads | End-to-end time | GDN time | Share |
| --- | ---: | ---: | ---: | ---: |
| pp128 | 1 | 46.84 s | 2.41 s | 5.14% |
| pp128 | 96 | 1.04 s | 0.063 s | 6.04% |
| tg64 | 1 | 64.71 s | 1.72 s | 2.65% |
| tg64 | 96 | 2.99 s | 0.097 s | 3.23% |

The packed Q1 decode path includes an experimental SVE2 GEMV kernel for VL=128
targets. It pairs two adjacent four-row panels so one Q8 activation load feeds
eight outputs. Graviton4 whole-model A/B measurements found NEON DOTPROD faster,
so NEON remains the default and `GGML_Q1_SVE2=1` opts into SVE2. Single-row Q1
decode also uses a static,
contiguous panel split after activation quantization instead of work stealing.
Consecutive eligible Q1 `MUL_MAT` projections now also share the packed Q8
activation when their F32 source tensor and packing layout match exactly;
unrelated graph nodes may occur between them. Eligibility is limited to F32
projection sources that remain immutable during the eligible sequence. A
different eligible source or layout replaces the one-entry cache, and a node
whose output overlaps the source storage invalidates it conservatively. The
execution-owned allocation is sized for the largest eligible candidate so
one-entry replacement remains available even when the planner cannot predict a
future reuse pair. Overlap invalidation is performed by thread zero after
computation and published by the existing end-of-node barrier. This
targets the Q/K/V, recurrent QKV/Z/alpha/beta, and FFN up/gate groups without
retaining data across graph executions or scheduler splits.
`GGML_Q1_ACT_CACHE=0` disables this reuse; `GGML_Q1_ACT_CACHE_DEBUG=1` prints
per-execution hit/miss counts.

The 2026-08-24 A/B run confirmed exact SVE2 and NEON oracle results. At 72
threads, SVE2 reached 20.50 tg128 versus 21.99 for NEON with the activation
cache enabled; at 96 threads it reached 20.63 versus 21.51. Q8 activation reuse
was neutral at full-model scale, with changes below normal single-run noise.
Raw summary: `benchmark-results-2026-08-24-r8g-sve2-cache.json`.

The raw archive SHA256 is
`4a7dfe9b82463862f18f424667cbc47aeff3129bfaa7df5bc079a50289dd9116`.

## Provenance note

The packed thread sweep and packed PMU runs were captured before adding a
temporary Gated DeltaNet timer. The GDN attribution runs and mmap PMU run used
that timer; it added one end-of-operator barrier and was removed after the
archive was captured. The mmap throughput agrees with the independent prior
baseline, but the full 27B sweep was not repeated after removing the timer.

The remote incremental build also retained the earlier configured engine-tree
string `63e2234c0268e540b8f63a8fe8ca9c4bd44d01cb` in its JSON output. The final
source tree, rebuilt and tested locally after removing the timer, is
`f9b4b08abf818ca0ef5ad6d4f069f51d515bc747`.
