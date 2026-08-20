# Exactness Oracle

Phase 0 captures deterministic full-vocabulary logits before changing runtime
dispatch, scheduling, graph fusion, or I2_S kernels. A capture stores token IDs,
selected prompt positions, raw F32 logits, and build/runtime metadata in GGUF.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON
cmake --build build --parallel --target celiums-logits-capture
```

## Capture

By default the tool captures logits after the last tokenized prompt position.
Repeat `--position` to capture additional zero-based positions. Negative
positions count from the end.

```bash
python utils/logits_capture.py \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "The strict I2_S oracle" \
  --position 0 \
  --position -1 \
  --threads 1 \
  --output captures/reference.gguf
```

To localize a numerical divergence, capture exact named graph tensors. Append
`@OP` where a graph name is reused:

```bash
python utils/logits_capture.py \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "The strict I2_S oracle" \
  --tensor attn_norm-0 \
  --tensor Qcur-0@ROPE \
  --tensor Kcur-0@ROPE \
  --tensor Vcur-0@RESHAPE \
  --tensor attn_out-0 \
  --tensor l_out-0 \
  --threads 1 \
  --output captures/probes.gguf
```

Probe captures add scheduler observation boundaries and are diagnostic runs.
Always compare their final logits against a callback-free capture before
treating them as non-perturbing observations.

```bash
python utils/compare_logits.py \
  captures/reference.gguf \
  captures/probes.gguf \
  --ignore-probes \
  --require-bitwise
```

The wrapper also writes `captures/reference.gguf.json` with SHA-256 hashes for
the model, executable, prompt, and capture artifact, plus the root and submodule
commits and whether the source tree was dirty. Commit that metadata with
release evidence; large logits captures can remain external artifacts.

## Compare

```bash
python utils/compare_logits.py \
  captures/reference.gguf \
  captures/candidate.gguf \
  --require-bitwise \
  --require-probes-bitwise \
  --output captures/comparison.json
```

The comparator verifies token IDs, capture positions, tensor shapes, and finite
values. It reports differing F32 bit patterns, max and mean absolute error, MSE,
NMSE, and top-1 agreement.

## Runtime Validation

```bash
python utils/logits_capture.py \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "Celiums BitNet exactness oracle." \
  --position -1 \
  --threads 1 \
  --build-dir build-runtime \
  --output captures/runtime-reference.gguf

python utils/compare_runtime_logits.py \
  --library build-runtime/src/libceliums-bitnet-runtime.so \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --reference captures/runtime-reference.gguf \
  --require-bitwise
```

The runtime prefill materializes logits for a single output row (the final
prompt token), so the runtime reference must capture only the final prompt
position. Capturing extra positions changes the number of requested outputs,
which perturbs flash-attention scheduling and produces ULP-level differences
even against the same build. With a final-position-only reference the runtime
matches the engine bitwise.

## Promotion Policy

Transparent strict optimizations must produce zero differing F32 bits against
the same build policy, prompt tokens, positions, context, and thread settings.
Cross-ISA comparisons are recorded separately until each ISA is explicitly
included in the strict equivalence claim.

The first Phase 0 comparison found native AVX-VNNI and AVX2-without-VNNI
bitwise identical on the fixed oracle prompt. A forced scalar build kept the
same top-1 token but differed in F32 logits (`NMSE=3.31e-3`), so scalar versus
SIMD is currently a numerical-diagnostics comparison rather than a bitwise
strict claim.

Layer-0 probes localized the first scalar/SIMD difference to rotary embedding.
`attn_norm-0` and the Q/K/V projection outputs are bitwise identical. Q and K
first differ after `ROPE`, with maximum absolute error `4.77e-7`; the I2_S QKV
matrix products are therefore not the source of this cross-ISA difference.
The native probe run's final logits were bitwise identical to a callback-free
capture.

The fixed Phase 0 evidence, including capture hashes and corrected
floating-point difference counts, is stored in
`docs/exactness-oracle-results-2026-08-19.json`.

The exactness oracle complements, but does not replace, source-checkpoint
perplexity and KL validation.
