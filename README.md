<div align="center">

# Celiums BitNet Runtime

### Correctness-first 1.58-bit inference, hardened and tuned for production CPUs

[![License](https://img.shields.io/badge/Celiums%20code-Apache--2.0-blue.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-Microsoft%20BitNet-5C2D91.svg)](UPSTREAM.md)
[![Version](https://img.shields.io/badge/version-0.1.0-111827.svg)](CHANGES.md)
[![Status](https://img.shields.io/badge/strict%20I2__S-validated-0F766E.svg)](docs/NUMERICAL_CONTRACT.md)

</div>

Celiums BitNet is an independent engineering fork of
[Microsoft BitNet](https://github.com/microsoft/BitNet). It repairs numerical
and model-graph defects in the inherited I2_S path, adds exact SIMD kernels,
hardens GGUF conversion, introduces phase-aware CPU scheduling, and wraps the
result in reproducible tests and benchmark tooling.

The goal is simple: make ternary inference fast **without silently changing the
model computation**.

> [!IMPORTANT]
> Celiums BitNet is maintained by Celiums Solutions LLC and is not affiliated
> with, sponsored by, endorsed by, or supported by Microsoft. Microsoft and the
> original contributors retain copyright in upstream code and model artifacts.

## Why This Fork Exists

The inherited BitNet implementation could compile and generate text while still
violating the intended numerical contract. The problems were not cosmetic:

- the BitNet b1.58 graph used SiLU instead of squared ReLU;
- I2_S packing and scalar dot-product semantics were inconsistent;
- the AVX2 path without VNNI could overflow 16-bit intermediate sums;
- generic BLAS could interpret custom packed tensors through an invalid path;
- zero activations and multi-plane matrix inputs were not handled safely;
- model conversion could emit incorrect metadata, scales, or tensor layouts;
- CPU affinity options changed thread counts without attaching the requested
  threadpools to `llama-cli` and `llama-server`.

Celiums BitNet treats these as correctness bugs first and performance problems
second. Optimizations enter the strict path only after they match the scalar
reference and survive architecture-specific tests.

## What Celiums Changed

| Area | Celiums work |
| --- | --- |
| Model graph | Corrected BitNet b1.58 FFN activation from SiLU to ReLU squared. |
| I2_S format | Standardized 128-value blocks, 2-bit packing, scale placement, byte counts, and shape validation. |
| Integer math | Defined the unsigned packed-weight dot-product contract and corrected scalar recovery. |
| SIMD | Added exact AVX2 I8_S activation quantization and corrected non-VNNI overflow before I32 widening. |
| Matrix kernels | Hardened GEMV/GEMM, scalar fallback, multi-thread execution, zero activations, and multi-plane inputs. |
| Runtime | Precomputed post-scales, distributed activation quantization by contiguous row ranges, and removed invalid generic BLAS dispatch. |
| CPU scheduling | Added `--celiums-hybrid-auto`: P-cores for single-token work and the full allowed CPU set for batch work. |
| Threadpools | Attached separate decode and batch pools to common llama contexts so affinity is effective in CLI and server execution. |
| Conversion | Unified I2_S packing, fixed offline scale direction, fixed GGUF file type, added quantization metadata, and rejected malformed tensors. |
| Python tooling | Rebuilt setup, inference, server, and benchmark wrappers with explicit failures and real batch parameters. |
| Validation | Added scalar, AVX2, AVX2-without-VNNI, sanitizer, matrix, conversion, wrapper, and hybrid-policy tests. |
| Governance | Added explicit provenance, security policy, numerical contract, hardware matrix, benchmark methodology, and experimental roadmap. |

Detailed implementation notes are in [CHANGES.md](CHANGES.md) and
[the 2026-08-18 performance audit](docs/performance-audit-2026-08-18.md).

## Strict Numerical Contract

The strict I2_S path represents ternary weights as unsigned 2-bit codes:

```text
ternary -1 -> code 0
ternary  0 -> code 1
ternary +1 -> code 2
```

For quantized activation `q`, activation sum `sum`, integer dot product `D`,
weight scale `weight_scale`, and activation scale `scale`, output recovery is:

```text
y[j] = (D[j] - sum) * weight_scale / scale
```

Strict kernels must preserve packed bytes, activation sums, integer products,
scale behavior, and output recovery relative to the scalar reference. Current
I2_S matrix dimensions must be positive and divisible by 128.

See [docs/NUMERICAL_CONTRACT.md](docs/NUMERICAL_CONTRACT.md) for the complete
contract.

## Execution Policy

Celiums development is organized around three policy levels:

- **Strict:** exact compatibility with the documented numerical contract.
- **Fast:** optimizations that still pass strict equivalence tests.
- **Experimental:** compact formats, AMX, GPU, and other work that has not yet
  completed strict model-level certification.

`CELIUMS_BITNET_STRICT=ON` is the default CMake configuration.
`CELIUMS_BITNET_EXPERIMENTAL=OFF` keeps experimental kernels out of normal
builds.

## Supported Model

The strict end-to-end setup currently accepts one model family:

| Model | Architecture | Format | Strict status |
| --- | --- | --- | --- |
| [microsoft/BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | `bitnet-b1.58` | I2_S | CPU kernel and structural conversion validation complete |

Other inherited model entries are intentionally not advertised as strict. Some
use a different activation graph, have unsupported dimensions, require a
separate embedding workflow, or still need model-level logits/perplexity
certification.

Model files are not distributed by this repository. The locally validated
GGUF after correcting `general.file_type` to `MOSTLY_I2_S` (41) had:

```text
SHA256 e23b16fa81b890e8b65e676262b645e8ffa5ae1f6df89dadaf793246826bbd90
210 I2_S tensors, 121 F32 tensors, 1 F16 tensor
```

Model licenses and terms remain those of their publishers.

## Performance Snapshot

Controlled CPU measurements on a DigitalOcean `c-60-intel` instance with the
Microsoft 2B I2_S model:

| Threads | Prompt processing, pp128 | Token generation, tg128 |
| ---: | ---: | ---: |
| 1 | 66.605 tok/s | 8.197 tok/s |
| 8 | 461.511 tok/s | 52.608 tok/s |
| 30 | 924.867 tok/s | 110.076 tok/s |
| 60 | 1200.831 tok/s | 103.840 tok/s |

These are controlled throughput results, not universal speedup claims. Decode
peaked before the full logical CPU count, which motivated separate decode and
batch scheduling. See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for the test
rules and [docs/performance-audit-2026-08-18.md](docs/performance-audit-2026-08-18.md)
for the recorded environment.

## Quick Start

### Requirements

- Python 3.10 or newer
- CMake 3.22 or newer
- A C/C++ compiler with C++17 support
- Git with submodule support
- Python packages from `requirements.txt` for model conversion

### Clone

```bash
git clone --recursive https://github.com/celiumsai/celiums-bitnet.git
cd celiums-bitnet
```

### Build Only

```bash
python setup_env.py --build-only --build-server
```

The default build includes the `celiums-bitnet` product command, inherited
engine tools used during the compatibility transition, the Celiums I2_S tests,
and `celiums-logits-capture` for exactness baselines. A default installation
exposes only the Celiums command and C header; set
`CELIUMS_BITNET_INSTALL_COMPAT=ON` to install inherited development tools.

The supported product entry point is `build/bin/celiums-bitnet`:

```bash
build/bin/celiums-bitnet version
build/bin/celiums-bitnet validate --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
build/bin/celiums-bitnet run --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --prompt "Hello"
```

See [docs/RUNTIME_PRODUCT.md](docs/RUNTIME_PRODUCT.md) for the product boundary,
experimental C API, installation layout, and compatibility policy.

The runtime C API now includes opaque Session and Request handles,
tokenization, prefill/decode, copied logits, generation, streaming callbacks,
stop sequences, and cooperative cancellation. `run`, `bench`, and `serve` use
that API directly.

`serve` exposes the initial OpenAI-compatible `/v1/completions` and
`/v1/chat/completions` endpoints. HTTP streaming and continuous batching are
not yet part of the native server; use compatibility tooling only when those
features are required during the transition.

### Download, Build, and Convert

```bash
python setup_env.py \
  --hf-repo microsoft/BitNet-b1.58-2B-4T \
  --quant-type i2_s
```

The setup script fails closed when the source architecture, activation,
generated metadata, required projections, dimensions, byte counts, or scales do
not match the strict contract.

### Run Inference

```bash
python run_inference.py \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "Explain why ternary weights reduce memory bandwidth." \
  --threads 6 \
  --threads-batch 14 \
  --n-predict 128
```

### Hybrid Intel Scheduling

```bash
python run_inference.py \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "Write a concise technical summary." \
  --hybrid-auto
```

`--hybrid-auto` discovers the effective process/cgroup CPU set, uses allowed
performance cores for single-token work, uses the full allowed set for batch
work, and attaches separate threadpools with phase-specific affinity. On
non-hybrid or P-core-restricted systems it safely falls back to the allowed CPU
set.

### Run the Server

```bash
python run_inference_server.py \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --hybrid-auto
```

The wrapper keeps continuous batching enabled. Binding to a non-loopback host
prints a warning because authentication is not configured by the wrapper.

### Benchmark

```bash
python utils/e2e_benchmark.py \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --n-prompt 128 \
  --n-token 128 \
  --batch 128 \
  --ubatch 64 \
  --threads 8 \
  --repetitions 5 \
  --output jsonl
```

## Validation

### Python and conversion tests

```bash
python -m unittest discover -s tests -v
```

### Native targeted tests

```bash
cmake --build build --parallel --target \
  test-quantize-fns test-i2s-mul-mat test-celiums-hybrid

ctest --test-dir build \
  -R 'test-(quantize-fns|i2s-mul-mat|celiums-hybrid)$' \
  --output-on-failure
```

The validation matrix also covers:

- scalar CPU builds without AVX;
- AVX2 builds with VNNI disabled;
- ASan and UBSan;
- GEMV and GEMM with 1, 2, and 4 threads;
- zero activation rows;
- multi-plane inputs;
- malformed GGUF dimensions, byte counts, scales, and file types;
- restricted CPU affinity and hybrid fallback behavior.

Raw full-vocabulary regression baselines are documented in
[docs/EXACTNESS_ORACLE.md](docs/EXACTNESS_ORACLE.md).
The fixed Phase 0 result manifest is
[`docs/exactness-oracle-results-2026-08-19.json`](docs/exactness-oracle-results-2026-08-19.json).
The exact-commit C-60 baseline and first strict tinyBLAS optimization are in
[`docs/benchmark-results-2026-08-19.json`](docs/benchmark-results-2026-08-19.json).

## Repository Map

```text
3rdparty/llama.cpp/                  pinned Celiums llama.cpp dependency
docs/NUMERICAL_CONTRACT.md          strict I2_S/I8_S semantics
docs/BENCHMARKING.md                reproducible benchmark rules
docs/SUPPORTED_HARDWARE.md          validated hardware/backend matrix
docs/EXPERIMENTAL_ROADMAP.md        non-strict optimization roadmap
docs/EXACTNESS_ORACLE.md            deterministic raw-logits regression workflow
docs/performance-audit-2026-08-18.md controlled implementation audit
setup_env.py                        strict build/download/conversion entry point
run_inference.py                    CLI inference wrapper
run_inference_server.py             server wrapper
utils/i2s_format.py                 canonical Python I2_S packer and validator
tests/                              Python conversion and wrapper tests
```

## Current Boundaries

- Strict certification is currently CPU-focused and model-specific.
- TL1, TL2, embedding conversion, compact formats, AMX, and GPU work remain
  experimental unless explicitly listed in the hardware matrix.
- Full source-checkpoint logits/perplexity comparison remains a release gate
  for broader model compatibility claims.
- Clone reproducibility depends on the pinned Celiums llama.cpp submodule
  commit recorded by this repository.

## Roadmap

Near-term work includes:

1. Model-level logits and perplexity certification.
2. AVX2, VNNI, and AVX-512 tuning without changing strict results.
3. Reuse of activation quantization across QKV and gate/up projections.
4. Exact compaction of dead FFN channels.
5. Experimental compact formats, AMX kernels, and GPU backends.

See [docs/EXPERIMENTAL_ROADMAP.md](docs/EXPERIMENTAL_ROADMAP.md).

## Ownership, License, and Attribution

Copyright 2026 Celiums Solutions LLC.

Celiums-authored code, modifications, documentation, tests, and tooling are
licensed under the [Apache License 2.0](LICENSE). The original Microsoft BitNet
portions remain available under their retained [MIT License](LICENSE-MIT).
The pinned llama.cpp dependency and bundled third-party components retain their
own licenses and notices.

This repository does not transfer ownership of Microsoft, ggml, llama.cpp,
model, or third-party code to Celiums Solutions LLC. Celiums Solutions LLC owns
its original contributions and modifications only. See [NOTICE](NOTICE) and
[UPSTREAM.md](UPSTREAM.md) for exact provenance.

## Maintainer

- Mario Gutierrez, Celiums Solutions LLC

Security reports should follow [SECURITY.md](SECURITY.md).
