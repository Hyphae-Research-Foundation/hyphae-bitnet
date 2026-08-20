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
