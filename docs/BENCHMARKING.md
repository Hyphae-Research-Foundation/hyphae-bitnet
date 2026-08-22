# Benchmarking

Report prefill and decode separately. Every result must include:

- Celiums BitNet product commit, engine commit, and vendored engine tree hash.
- Model filename and SHA256.
- CPU/GPU model and exposed ISA.
- Compiler and flags.
- Thread count, affinity, NUMA policy, batch, and ubatch.
- Warmup, individual samples, median, dispersion, and P95.
- Kernel-only and end-to-end measurements when relevant.
- Correctness, logits, KL, or perplexity checks appropriate to the change.

Use `utils/e2e_benchmark.py` for the standard CPU matrix. Raw result archives
are release artifacts; Git should retain scripts, summaries, and hashes.

## I2_S Phase Profile

Set `GGML_I2S_PROFILE=1` to print one aggregate line at process exit. The
profiler is disabled by default. It reports activation quantization, barrier
waits, tinyBLAS operations, fallback operations, and separately timed fallback
post-processing. All `*_thread_us` fields sum elapsed time across participating
threads, so they are not wall-clock phase durations. Fused tinyBLAS output
scaling and stores are reported separately in `postprocess_thread_us`.
Each `ggml_i2s_profile_shape` JSON record adds the phase, projection, and
`m`/`n`/`k` shape. Projections are grouped as Q, K, V, attention output, gate,
up, down, or other; `n=1` is decode and larger `n` is prefill.

```bash
GGML_I2S_PROFILE=1 build/bin/llama-bench \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p 128 -n 128 -t 1
```

Profiling adds timers and atomic counters to the measured path. Use it to
locate costs, not as the source of publishable throughput numbers.

## Celiums 0.1 Development Validation

The corrected strict path was validated on a DigitalOcean `c-60-intel` with
60 dedicated Xeon Platinum 8358 vCPUs, strict affinity, batch/ubatch 128, and
five samples per workload:

| Threads | pp128 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 1 | 66.61 | 8.20 |
| 8 | 461.51 | 52.61 |
| 30 | 924.87 | 110.08 |
| 60 | 1,200.83 | 103.84 |

Decode peaks before all 60 cores, reinforcing the need for phase-specific
thread counts. The raw validation archive has SHA256
`2d42d7e8868aa80f666d81ca5c0fce1ab55603ce530b26726a6b9b7e8b4958e2`.

## Phase 0 Current-Source Rerun

A clean DigitalOcean `c-60-intel` rerun on an Intel Xeon Gold 6548N produced:

| Threads | pp128 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 30 | 1,086.44 | 127.51 |
| 60 | 1,232.25 | 133.57 |

The source snapshot was transferred without Git metadata, so llama-bench
reported build commit `unknown`; the archived source/model hashes and logits
capture identify this run. Future remote runs should use a Git clone or inject
the exact build commit into CMake metadata.

## Exact-Commit C-60 Baseline and GEMV8

The Phase 0 source was reconstructed from Git bundles on a clean
DigitalOcean `c-60-intel`, preserving historical product commit `df0165b` and
the then-separate engine commit `d608d85b7`. The host exposed 60 dedicated Intel Xeon Platinum 8358
cores, AVX-512 VNNI, 120 GiB RAM, and one NUMA node. GCC 13.3.0 built Release
targets with `-march=native`. Every cell used strict affinity, batch/ubatch
128, and five samples.

| Threads | Baseline pp128 | GEMV8 pp128 | Baseline tg128 | GEMV8 tg128 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 57.91 | 57.79 | 8.07 | 8.70 |
| 30 | 741.76 | 780.31 | 64.90 | 68.70 |
| 60 | 836.12 | 865.66 | 56.43 | 56.86 |

The first strict optimization adds an 8-row tinyBLAS tile only for decode
(`n=1`). It preserves the same integer accumulation order and leaves prefill
unchanged. Decode improved 8.34% at one thread and 3.46% at 30 threads. At 60
threads decode remains slower than 30 threads. A clean rebuild of the final
commits measured decode improvements of 7.82%, 5.86%, and 0.75% at 1, 30, and
60 threads. pp128 movement is benchmark variance because GEMV8 is not selected
there. The pre/post remote oracle tensor values are bitwise identical; their
container hashes differ because build metadata records different commits.

The checked-in machine-readable summary is
`docs/benchmark-results-2026-08-19.json`. The temporary benchmark Droplet was
destroyed after artifacts were downloaded.

### Profile attribution

The one-thread C-60 profile identifies gate, up, and down as the dominant
tinyBLAS shapes. For tg128 their accumulated tinyBLAS times were approximately
2.43 s, 2.29 s, and 2.01 s, respectively; Q was 0.86 s, attention output
0.88 s, and K/V about 0.21-0.23 s each. For pp128, gate/up/down likewise
dominated at roughly 0.96 s, 1.00 s, and 0.89 s. This supports targeting
Gate/Up and QKV while keeping each strict kernel change independently
verifiable.

## PrismML Bonsai 27B Cross-Runtime Run

A CPU-only run on a DigitalOcean `c-60-intel` used the PrismML llama.cpp
release `prism-b9599-9ca265a` at commit `9ca265a57`, GCC 13.3.0 with
`GGML_NATIVE=ON`, strict CPU placement, batch/ubatch 128, and five samples
per workload. The host exposed 60 dedicated Intel Xeon Gold 6548N cores,
AVX-512 VNNI, 120 GiB RAM, and one NUMA node.

| Format | Threads | pp128 tok/s | tg128 tok/s |
| --- | ---: | ---: | ---: |
| `Q2_0` ternary | 1 | 2.40 | 1.04 |
| `Q2_0` ternary | 8 | 18.34 | 6.67 |
| `Q2_0` ternary | 30 | 48.33 | 13.88 |
| `Q2_0` ternary | 60 | 46.66 | 14.08 |
| `Q1_0` binary | 1 | 2.37 | 1.42 |
| `Q1_0` binary | 8 | 18.37 | 8.78 |
| `Q1_0` binary | 30 | 46.37 | 16.92 |
| `Q1_0` binary | 60 | 46.15 | 16.42 |

`Q2_0` peaked at 30 threads for prefill and 60 for decode. `Q1_0` peaked at 30
threads for both phases. `Q1_0`'s smaller 3.80 GB file improved decode by 20%
at each format's best thread count, but did not improve prefill. Measured peak
RSS was 13.11 GiB for `Q2_0` and 7.03 GiB for `Q1_0`.

Celiums BitNet 0.3.0 could not load these models because that release exposed
only `bitnet-b1.58` `I2_S`. The engine already contained inherited `qwen35`
and `Q1_0` support; a later explicit Bonsai compatibility profile enables that
binary format without broadening the strict I2_S default. Prism `Q2_0` remains
unsupported. The 0.3.0 strict-load rejection was recorded. As a
same-host product control, Celiums 0.3.0 with its 2.41B I2_S model reached
1,311.88 pp128 tok/s and 127.03 tg128 tok/s at 60 threads. This is not a
same-model or architecture-normalized comparison.

The machine-readable summary is
`docs/benchmark-results-2026-08-21-bonsai-27b.json`. The raw archive SHA256
is `779f8bd141449b7ce36780efacdb7b0c03d532d2dbaee4be8bead19b7e266934`.
The temporary benchmark Droplet was destroyed after artifacts were downloaded.

### Same-model Q1_0 comparison

After exposing the engine's existing `qwen35` and `Q1_0` path through the
Celiums family API, both runtimes were measured on the exact same
`Bonsai-27B-Q1_0.gguf` file on a second `c-60-intel`. This host exposed 60
dedicated Intel Xeon Platinum 8358 cores and one NUMA node.

| Runtime | Threads | pp128 tok/s | tg128 tok/s |
| --- | ---: | ---: | ---: |
| Celiums | 1 | 0.89 | 0.75 |
| Celiums | 8 | 7.02 | 4.81 |
| Celiums | 30 | 20.76 | 8.14 |
| Celiums | 60 | 20.71 | 8.61 |
| Prism | 1 | 1.53 | 1.04 |
| Prism | 8 | 11.99 | 5.82 |
| Prism | 30 | 35.23 | 10.80 |
| Prism | 60 | 35.45 | 10.14 |

The result is negative: enabling the same model does not accelerate it.
Prism's peak throughput is 1.71x higher for prefill and 1.26x higher for
decode. The current Celiums Q1_0 path therefore proves compatibility, not a
performance advantage. Celiums used 3.86 GiB peak RSS versus Prism's 7.03 GiB,
but the processes use different benchmark/session setup and that memory result
is not yet a runtime efficiency claim.

The same-model raw archive SHA256 is
`1fa7a5dd4bab6589760aea29c228863e3fae0db756b09e47fb350bf7851d70c0`.

A follow-up direct `llama-bench` run removed the Celiums product benchmark
wrapper from both sides. It confirmed the same result: Celiums peaked at 20.72
pp128 and 8.56 tg128 tok/s, while Prism peaked at 35.42 pp128 and 10.59 tg128
tok/s. Prism was 1.71x faster for prefill and 1.24x faster for decode. The
direct-engine archive SHA256 is
`9146d9f411fc4804393e875b9ebbf98546b6e749a65da7523cc8a26ec081c75f`.

### Q1_0 AVX-512 VNNI repack

A Q1_0-only port of Prism's four-row AVX-512 VNNI repack closed the direct
engine gap. On a `c-60-intel` with an Intel Xeon Platinum 8358, the optimized
Celiums engine reached:

| Threads | pp128 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 1 | 1.54 | 1.05 |
| 8 | 12.12 | 5.89 |
| 30 | 36.24 | 10.99 |
| 60 | 35.93 | 10.20 |

Against the previous direct Celiums run, peak prefill improved 74.9% and peak
decode improved 28.4%. Against Prism on the same CPU class and model, the
optimized Celiums peak was 2.3% faster for prefill and 3.8% faster for decode.
The repacked model used 7.03 GiB peak RSS, up from the previous Celiums process
measurement of 3.86 GiB and effectively equal to Prism.

The raw archive SHA256 is
`c00d339ebf385f6f9f44bdfcfc67ac672c8cbb1e803c017ec3c743bf43a6a5d2`.
