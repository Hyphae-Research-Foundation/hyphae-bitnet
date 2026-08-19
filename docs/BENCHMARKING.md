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
