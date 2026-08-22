# Vendored llama.cpp CPU hot-path review

Scope: CPU text inference for BitNet I2_S and Bonsai Q1_0 inside
`3rdparty/llama.cpp`. CUDA, Vulkan, SYCL, Metal, Android, and conversion
scripts are out of depth.

Papers used: Wang et al. BitNet CPU (arXiv:2410.16144) and bitnet.cpp
(arXiv:2502.11880) for exact 2P−S / I2_S correction; Park et al. LUT-GEMM
(arXiv:2206.09557) and Wei et al. T-MAC (arXiv:2407.00088) as *approximate*
LUT methods we do **not** ship; Khudia et al. FBGEMM (arXiv:2101.05615) for
packing and cache blocking; Yang et al. Gated DeltaNet (arXiv:2412.06464).

The exact math and tile sizes live in `celiums-exact`
(`ggml/src/ggml-cpu/celiums-exact.{c,h}`). Kernels and oracles call it.

## 1. MUL_MAT dispatch

**Files / functions**

- `ggml-cpu.c` `ggml_compute_forward` → `ggml_cpu_extra_compute_forward`
  (`traits.cpp`) then `ggml_compute_forward_mul_mat`.
- Extra buffers registered in `ggml-cpu.cpp`
  (`ggml_backend_cpu_repack_buffer_type`). I2_S extra buffer exists but is
  commented as slower than tinyBLAS VNNI.
- Repack traits: `repack.cpp` `tensor_traits::forward_mul_mat` /
  `forward_mul_mat_one_chunk` (`ggml_gemm_q1_0_4x8_q8_0` when `nrows > 3`).

**Bottleneck**

The default path is “add a kernel, keep 4-row activation tiles.” Prefill of
128 tokens therefore rereads the Q1 weight image 32 times. OpenMP/barrier
sits between quantize and matmul (`ggml_barrier` in
`ggml_compute_forward_mul_mat`).

**Opportunity**

FBGEMM-style panel reuse: more activation rows per weight panel. Shipped as
`celiums_exact_q1_gemm_act_tile_rows` = 8, used by
`ggml_gemm_q1_0_4x8_q8_0` (x86 VNNI). LUT-GEMM / T-MAC are rejected for the
strict I2_S/Q1 contract (`docs/NUMERICAL_CONTRACT.md`).

## 2. Q1_0 4×8 VNNI / repack

**Files / functions**

- Select: `ggml_repack_get_optimal_repack_type` (`repack.cpp`) →
  `tensor_traits<block_q1_0, 8, 4, GGML_TYPE_Q8_0>`.
- Pack: `repack_q1_0_to_q1_0_4_bl` → `celiums_exact_q1_make_4x8`.
  ARM i8mm instead expands to q8_0 4×8 (`q1_repack_expand_to_q8`).
- Kernels: `arch/x86/repack.cpp` `ggml_gemv_q1_0_4x8_q8_0` (decode) and
  `ggml_gemm_q1_0_4x8_q8_0` (prefill, 8-row then 4-row). Generic reference:
  `ggml_gemm_q1_0_4x8_q8_0_generic` (same 8-row weight reuse via
  `celiums_exact_q1_gemm_act_tile_rows`; used on hosts without AVX-512 VNNI).
- Math: `celiums_exact_q1_corr_int` = `2P − S` (Wang / bitnet.cpp).
- Direct kernel calls require `ggml_cpu_init()` so the x86 `GGML_CPU_FP16_TO_FP32`
  LUT (`ggml_table_f32_f16`) is filled; otherwise scales are 0.

**Bottleneck**

GEMV decode streams ~3.5 GiB/token (i7i t=1 cache miss ~65%). Prefill was
limited by 4-row tiles, not by `vpdpbusd` itself. i7i: VNNI vs generic
~1.88× pp / ~1.22× tg; t=1 2.39/1.24 tok/s.

**Opportunity**

Shipped: 8 activation rows share one weight panel (halves Q1 rereads on
pp128) in both AVX-512 VNNI and the generic GEMM. Next: 16-row tiles (all
Bonsai projections divisible by 16). Do not persist byte-expanded Q1 on x86
for decode (ARM t=96 tg dropped when the int8 image saturated DRAM).

## 3. I2_S

**Files / functions**

- Contract: `docs/NUMERICAL_CONTRACT.md` — `u = W+1`, `y = (D − S) ρ`.
- Fast path: `llamafile/sgemm.cpp` `llamafile_sgemm_i2s` (tinyBLAS VNNI).
- Fallback: `ggml-cpu.c` `ggml_gemm_i2_i8_s` / `ggml_gemv_i2_i8_s`
  (`ggml-cpu-i2s.c`) then per-column recover.
- Recover now: `celiums_exact_i2s_recover` in both tinyBLAS epilogue and
  the vec_dot / gemm fallback.

**Bottleneck**

Quantize-then-barrier before tinyBLAS (`GGML_I2S_PROFILE_BARRIER`). Decode
profiles on c-60 showed ~43% OpenMP/runtime vs ~39% Q1/I2_S kernel.
Activation packing is not shared across projections that consume the same
F32 row (deep dive: ~240/497 redundant packs on Bonsai analog).

**Opportunity**

Keep the exact `(D − S) ρ` epilogue (shipped through `celiums-exact`).
Shared Q8/I8 packing across fused projections is the next exact win
(FBGEMM packing, not T-MAC LUTs).

## 4. Threadpool / chunking

**Files / functions**

- `ggml_compute_forward_mul_mat` (`ggml-cpu.c`): per-thread quantize of
  src1, `ggml_barrier`, then `current_chunk` work steal.
- Repack `forward_mul_mat`: quantize in planes of 4 rows
  (`ggml_quantize_mat_t<8, Q8_0>` → `ggml_quantize_mat_q8_0_4x8`), then
  chunks aligned to `NB_COLS` (4). `nth_scaled = nth * 4`.
- NUMA: `disable_chunking` when `ggml_is_numa()`.

**Bottleneck**

Small projections (GDN alpha/beta 5120×48) still pay a global barrier.
Chunk alignment to 4 columns plus `nth * 4` over-partitions at 48–96
threads.

**Opportunity**

Phase-specific thread caps (deep dive item 4). Tile planner already
returns 8 for `nr ≥ 8` without changing chunk alignment of weights.

## 5. Qwen35 Gated DeltaNet

**Files / functions**

- Graph: `src/models/delta-net-base.cpp` `ggml_gated_delta_net`.
- CPU: `ops.cpp` `ggml_compute_forward_gated_delta_net` /
  `_one_chunk` / `_f32`. Parallelism is over heads × sequences (`ir` =
  head + seq), not over the `S_v` state row.
- State: F32 `[S_v, S_v, H, n_seqs]` (~144 MiB aggregate on Bonsai).

**Bottleneck**

Per-token copy of the full head state into `state_work`, then decay,
`S^T k`, rank-1 update, `S^T q` as separate passes (Yang et al. 2024).
Sampled GDN time was small vs Q1 GEMM on prefill (~4.5%) but the state
is walked every decode step.

**Opportunity**

Row-hot fused update (deep dive item 5) cutting state traversals. Not
shipped this change; Q1 GEMM reuse is the larger measured prefill lever.

## Shipped change (this goal)

Not a new ISA wrapper. `ggml_gemm_q1_0_4x8_q8_0` (AVX-512 VNNI) and
`ggml_gemm_q1_0_4x8_q8_0_generic` reuse each Q1 4×8 weight panel across
8 activation rows when `celiums_exact_q1_gemm_act_tile_rows(nr) >= 8`.
Packing uses `celiums_exact_q1_make_4x8`. Exactness:
`tests/test-q1-repack-oracle.cpp` (packed `celiums_exact_q1_dot` vs layout
vs generic, `nr=4` and `nr=8`; calls `ggml_cpu_init()` for the x86 fp16
LUT) and `3rdparty/llama.cpp/tests/test-i2s-mul-mat.cpp`
(`celiums_exact_i2s_recover`).
