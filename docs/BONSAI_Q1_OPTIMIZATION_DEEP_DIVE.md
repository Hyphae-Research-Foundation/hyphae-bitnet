# Bonsai Q1 CPU Optimization Deep Dive

## Scope

This analysis targets the exact Bonsai 27B `Q1_0` GGUF on CPU without
changing its stored weights or Q8 activation values. The current reference
machine is a DigitalOcean `c-60-intel` exposing AVX-512 VNNI.

## Q1_0 Mathematics

A block stores 128 sign bits and one FP16 weight scale. Let `b_i` be a stored
bit, `s_i = 2*b_i - 1`, `d_w` the block scale, `q_i` a Q8 integer, and `d_x`
the Q8 scale. The block dot product is:

```text
dot = d_w * d_x * sum_i ((2*b_i - 1) * q_i)
```

For each group of 32 activations:

```text
P = sum_i b_i*q_i
S = sum_i q_i
sum_i ((2*b_i - 1)*q_i) = 2*P - S
```

`P` and `S` map directly to unsigned-by-signed `VPDPBUSD`. This is a better
fit for W1A8 than XNOR-popcount, which would require decomposing each Q8 value
into seven or eight bit planes plus a dynamic transpose.

The current four-row kernel preserves eight FP32 accumulation stripes per
output. Integer partials are exact; low-bit floating-point identity depends on
preserving scale multiplication and the final reduction tree.

## Model Structure

Bonsai uses 64 Qwen35 blocks:

```text
16 x [Gated DeltaNet, Gated DeltaNet, Gated DeltaNet, full attention]
```

The model has 497 Q1 matrix multiplications:

| Projection | Shape | Count |
| --- | ---: | ---: |
| FFN up | 5120 x 17408 | 64 |
| FFN gate | 5120 x 17408 | 64 |
| FFN down | 17408 x 5120 | 64 |
| Recurrent QKV | 5120 x 10240 | 48 |
| Recurrent Z | 5120 x 6144 | 48 |
| Recurrent output | 6144 x 5120 | 48 |
| Recurrent alpha/beta | 5120 x 48 | 96 |
| Full Q+gate/K/V/output | mixed | 64 |
| LM head | 5120 x 248320 | 1 |

The FFN accounts for about 67% of Q1 decode MACs. Gated DeltaNet contains a
144 MiB aggregate F32 recurrent state across its 48 layers.

## Roofline

The model contains 26,895,998,464 parameters in 3,792,459,776 resident model
bytes, or approximately 1.128 bits per parameter. At the measured 10.99 decode
tokens/s, model-byte throughput is only 41.7 GB/s. The useful dense rate is
approximately 0.59 TOPS.

For pp128 at 36.24 tokens/s, the current four-token GEMM tile implies about
34.4 GB/s of physical weight traffic and 1.95 useful TOPS. Prefill is therefore
not saturating DRAM; it is primarily limited by matrix-kernel instruction mix,
small tiles, repeated weight traversals, and synchronization.

The VM reports 60 cores, but Xeon Platinum 8358 has 32 physical cores and 64
threads. The 30-thread performance peak strongly suggests that the effective
allocation is about 30 physical cores plus SMT siblings hidden by KVM.

## Measured Profile

Hardware-counter and sampled profiles on the optimized four-row build found:

| Metric | Decode | pp128 |
| --- | ---: | ---: |
| Effective clock | 3.06 GHz | 2.47 GHz |
| Instructions/cycle | 0.90 | 2.77 |
| Cache misses/reference | 46.2% | 8.9% |
| Q1 GEMV/GEMM sampled time | 39.4% | 72.5% |
| OpenMP/runtime sampled time | about 43% | about 17% |
| Gated DeltaNet sampled time | 1.3% | 4.5% |

Decode is a mixed memory, thread-runtime, and Q1 GEMV problem. Prefill is
dominated by the Q1 GEMM kernel and is the clearest target for a larger tile.
The profile archive SHA256 is
`56b0213614bb0671bc8d96d528d1e57efe093acf2916af66479d905b774eb2d2`.

## Literature

The most applicable sources are:

1. **Fast and Lossless BitNet b1.58 Inference on CPUs**, Wang et al., 2024,
   [arXiv:2410.16144](https://arxiv.org/abs/2410.16144). Lossless I2_S and LUT
   CPU kernels establish the value of format-specific packing.
2. **bitnet.cpp**, Wang et al., 2025,
   [arXiv:2502.11880](https://arxiv.org/abs/2502.11880). Provides exact and
   approximate LUT formulations for low-bit CPU inference.
3. **T-MAC**, Wei et al., 2024/2025,
   [arXiv:2407.00088](https://arxiv.org/abs/2407.00088). Uses activation-side
   lookup tables and mirror consolidation; most relevant on AVX2/ARM targets.
4. **LUT-GEMM**, Park et al., 2022/2024,
   [arXiv:2206.09557](https://arxiv.org/abs/2206.09557). General binary-code
   lookup formulation.
5. **XNOR-Net**, Rastegari et al., 2016,
   [arXiv:1603.05279](https://arxiv.org/abs/1603.05279). Canonical W1A1
   popcount formulation, but not the right primitive for dynamic W1A8.
6. **FBGEMM**, Khudia et al., 2021,
   [arXiv:2101.05615](https://arxiv.org/abs/2101.05615). Packing, generated
   microkernels, cache blocking, and fused epilogues.
7. **Gated Delta Networks**, Yang et al., 2024,
   [arXiv:2412.06464](https://arxiv.org/abs/2412.06464). Defines the recurrent
   state update used by 48 layers.
8. **Parallelizing Linear Transformers with the Delta Rule**, Yang et al.,
   2024, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484). Motivates a
   separate chunkwise prefill algorithm.
9. **Speculative Decoding**, Leviathan et al., 2022/2023,
   [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). Exact target
   distribution with multi-token verification.

BiLLM, BitDistiller, OneBit, FBI-LLM, SmoothQuant, AWQ, BitNet a4.8,
Q-Sparse, and TEAL are relevant only to a separate model-changing or
approximate quality program.

## Optimization Program

### Exact, High Priority

1. **Wider Q1 output tiles.** Move from four to eight or sixteen output rows.
   This reuses each activation block, scale, and correction sum across more
   weights. All major Bonsai projection dimensions are divisible by 16.
2. **Prefill cache blocking and loop interchange.** The current four-row tile
   traverses the model once per four prompt rows. Reusing one weight panel
   across 16-32 activation rows can substantially increase arithmetic
   intensity.
3. **Share Q8 activation packing.** QKV/Z/alpha/beta, Q/K/V, and FFN gate/up
   consume identical F32 inputs but currently quantize them independently.
   Approximately 240 of 497 activation packing operations are redundant.
4. **Thread-runtime reduction.** Small projections and elementwise nodes pay
   global OpenMP/barrier costs. Use phase- and shape-specific thread caps and
   one persistent pinned pool.
5. **Gated DeltaNet row-hot update.** Fuse decay, `S^T k`, update, and `S^T q`
   per state-row block, cutting state traversals and exposing more than 48
   tasks to the scheduler.
6. **Fuse RMSNorm into shared Q8 packing.** Avoid materializing a full F32 norm
   output followed by repeated absmax scans.
7. **Fuse SSM convolution and SiLU.** Avoid the 10240 x token F32 temporary in
   recurrent layers.

### Exact or Mathematically Equivalent, Medium Priority

1. Use direct mask-ready bit layouts and `0/2` codes to reduce expansion work.
2. Fuse projection epilogues, residual adds, and gating while carefully
   controlling FP association.
3. Stream LM-head top-k/argmax when API semantics do not require all logits.
4. Add exact speculative verification after multi-token CPU prefill improves.
5. Prototype exact 4-bit-index, INT16 LUT kernels for AVX2 and ARM.

### Rejected as First Steps

- Dynamic Q8 bit-plane popcount: too much transpose and seven/eight passes.
- Persistent byte-expanded weights: destroys the 1-bit memory advantage.
- AMX on Xeon 8358: the processor does not expose AMX.
- Approximate activation/KV/state quantization before the exact path is fully
  optimized.

## Expected Headroom

There is no evidence of a physical wall at 36 pp128 or 11 decode tok/s.
However, compute capability alone is not sufficient: decode must stream a
3.79 GB model and maintain recurrent state. Realistic exact targets are:

- Prefill: 1.5-2.5x from wider tiles and weight-panel reuse.
- Decode: 1.1-1.4x from wider tiles, lower synchronization, and GDN fusion.
- Additional exact speculation: potentially 1.1-1.5x if verification accepts
  multiple tokens efficiently.

These gains overlap and must be measured independently rather than multiplied.

## First Follow-up Experiment

Interchanging the Q1 GEMM loops from activation-tile-first to weight-tile-first
was tested on another `c-60-intel`. It did not improve pp128: 36.11 tok/s versus
36.24 tok/s in the previous run, within variance. Decode, which does not use
that loop, measured 11.49 tok/s on the new host. The experiment was not retained;
it confirms that meaningful prefill gains require a larger activation tile or
explicit panel blocking rather than a simple loop swap.
