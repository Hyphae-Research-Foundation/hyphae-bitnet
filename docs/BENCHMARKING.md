# Benchmarking

Report prefill and decode separately. Every result must include:

- Celiums BitNet and submodule commits.
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
