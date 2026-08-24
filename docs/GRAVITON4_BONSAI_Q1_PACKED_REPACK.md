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

The next optimization target is the packed Q1 GEMV loop and thread scheduling,
not Gated DeltaNet. An SVE2 packed kernel is still unimplemented; SVE PMU
events observed in the whole model come from other ggml kernels.

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
