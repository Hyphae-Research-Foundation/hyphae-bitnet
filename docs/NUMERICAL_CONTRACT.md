# Celiums BitNet I2_S / I8_S Numerical Contract

This document defines the strict CPU matrix contract implemented and tested by
Celiums BitNet 0.3.0. It describes the persisted I2_S weights, temporary I8_S
activations, integer correction, output recovery, accepted shape, and evidence
scope.

It does not claim numerical equivalence to the original source-checkpoint
implementation. The release evidence compares CPU paths using the same
converted GGUF model; source-checkpoint logits, perplexity, and KL validation
remain separate work.

## Symbols

| Symbol | Meaning |
| --- | --- |
| `M` | Number of output rows in the weight matrix |
| `K` | Contraction dimension / logical I2_S row width |
| `N` | Number of activation rows after flattening leading planes |
| `W[r,i]` | Logical ternary weight |
| `u[r,i]` | Unsigned persisted code, `W[r,i] + 1` |
| `alpha_W` | Tensor-wide weight dequantization scale |
| `x_t[i]` | F32 activation value in flattened row `t` |
| `beta_t` | Per-row activation quantization multiplier |
| `q_t[i]` | Signed I8 activation code |
| `S_t` | Sum of activation codes in row `t` |
| `D[r,t]` | Unsigned-code integer dot product |
| `T[r,t]` | Corrected ternary integer dot product |
| `rho_t` | Per-row output post-scale |

## 1. Logical Weight Domain

The logical matrix is rank 2:

```text
W in {-1, 0, +1}^{M x K}
M > 0
K > 0
K mod 128 = 0
```

Only `K` must be divisible by 128. `M` and `N` may be any positive dimensions
admitted by the surrounding GGML graph and memory limits.

The persisted unsigned code is

```text
u[r,i] = W[r,i] + 1

W = -1  ->  u = 0
W =  0  ->  u = 1
W = +1  ->  u = 2
```

Code `3` is reserved and invalid under this contract. The Celiums converters do
not emit it. The current ordinary runtime loader does not scan every packed
field, so strict model provenance also requires trusted converter output or an
independently verified model digest.

## 2. I2_S Packing

For logical row `r`, 128-coefficient block `b`, and packed-byte position
`g in [0,31]`, define

```text
P[r,b,g] = (u[r,128*b + g]    << 6) |
           (u[r,128*b + g+32] << 4) |
           (u[r,128*b + g+64] << 2) |
            u[r,128*b + g+96].
```

Equivalently:

| Packed bits | Logical position inside the 128-value block |
| --- | ---: |
| `7:6` | `g` |
| `5:4` | `g + 32` |
| `3:2` | `g + 64` |
| `1:0` | `g + 96` |

Each block occupies 32 bytes. Rows and blocks are stored consecutively.

For `M*K` logical weights, the I2_S tensor byte count is

```text
packed_payload_bytes = M*K/4
tensor_bytes         = M*K/4 + 32.
```

The first four bytes of the final 32-byte trailer contain one native-endian F32
weight dequantization scale `alpha_W` in the current host-produced GGUF
artifact. The remaining 28 bytes are reserved and ignored by the current CPU
kernel; existing converted artifacts may contain nonzero data there, and those
bytes are not part of the mathematical recovery formula.

## 3. Weight Scale Construction

The supported production conversion path writes one tensor-wide positive F32
`alpha_W`. Converter behavior depends on the source representation:

- the default Python floating-point path uses
  `alpha_W = max(mean(abs(W_source)), 1e-5)` before ternary classification;
- already-quantized or override-scale paths preserve their explicitly selected
  logical ternary values and scale;
- the C reference quantizer uses its documented first-nonzero rule and an
  all-zero fallback.

These constructors are not claimed to be interchangeable for arbitrary F32
tensors. Release certification concerns the converted model artifact and the
runtime interpretation of its persisted codes and scale.

## 4. I8_S Activation Quantization

I8_S is a temporary workspace representation. It is not a persisted GGUF
blocked tensor format.

For each flattened finite F32 activation row `x_t` in the validated domain:

```text
amax_t = max_i abs(x_t[i])

beta_t = 127 / amax_t,  if amax_t > 0
         0,             if amax_t = 0
```

The signed activation codes and correction sum are

```text
q_t[i] = clamp(roundf(beta_t * x_t[i]), -128, 127)
S_t    = sum_i q_t[i].
```

`roundf` means round to nearest with halfway cases away from zero. `beta_t` is
the activation quantization multiplier; `1/beta_t` is the dequantization step
for nonzero rows.

If `amax_t = 0`, then

```text
beta_t = 0
q_t[i] = 0 for every i
S_t    = 0.
```

Non-finite activations and finite values for which `127/amax_t` is not finite
are outside the validated numerical domain. The current low-level quantizer
does not provide a recoverable rejection status for those values; callers must
not supply them under the strict contract.

## 5. Unsigned Integer Dot Product

For each output row `r` and activation row `t`, the kernel computes

```text
D[r,t] = sum_i u[r,i] * q_t[i].
```

Because `u = W + 1`, subtracting the activation-code sum recovers the signed
ternary integer product:

```text
T[r,t] = D[r,t] - S_t
       = sum_i (u[r,i] - 1) * q_t[i]
       = sum_i W[r,i] * q_t[i].
```

The AVX2 path without VNNI widens bounded partial sums to I32 before long
accumulation. Strict kernels must agree on the activation codes, `S_t`, and the
corrected integer product for the validated dimensions.

For valid codes and I8 values, conservative bounds are

```text
abs(D[r,t]) <= 256*K
abs(S_t)    <= 128*K.
```

The certified model uses `K` values 2560 and 6912, safely inside signed-I32
accumulation for these bounds. The contract does not claim unbounded positive
multiples of 128 are safe.

## 6. Output Recovery

The active CPU path defines the post-scale piecewise:

```text
rho_t = alpha_W / beta_t,  if beta_t > 0
        0,                 if beta_t = 0.
```

The recovered approximation is

```text
y_hat[r,t] = rho_t * T[r,t]
           = rho_t * (D[r,t] - S_t).
```

The zero-row branch therefore produces exact zero without dividing by zero.

The BitNet graph may support additional projection-level scale tensors through
the inherited engine. The certified 2B release fixture has no separate `.scale`
tensors, so the projection-level multiplier is 1 for this contract.

## 7. BitNet b1.58 FFN Graph

The supported graph uses a parallel gated feed-forward block. If `z` is the
normalized residual input, `G` the gate projection, and `U` the up projection,
the nonlinear product is

```text
FFN_gate(z) = relu(G*z)^2
FFN_mix(z)  = FFN_gate(z) elementwise-multiplied-by (U*z).
```

This is squared ReLU, `relu(a)^2`, not `relu(a^2)` and not SiLU. The mixed
activation is sub-normalized and passed through the FFN down projection before
the residual addition.

## 8. Shape and Layout Preconditions

The strict converted I2_S weight tensor requires:

- rank exactly 2;
- positive `M` and `K`;
- `K mod 128 = 0`;
- exact byte count `M*K/4 + 32`;
- finite, positive trailer scale;
- canonical Celiums bit ordering;
- only logical codes 0, 1, and 2.

The Python conversion validator checks rank, dimensions, byte count, projection
presence, metadata, and scale. The trusted packer defines bit ordering and emits
only valid codes. The generic runtime checked-load path does not independently
prove every semantic precondition: it does not scan all packed codes or the
reserved trailer bytes.

## 9. Evidence and Exactness Scope

The repository tests establish the following bounded claims:

1. AVX2 activation quantization matches the scalar `roundf` formula for the
   enumerated patterns and lengths through 4096 in multiplier, I8 bytes, and
   activation sum.
2. The direct unsigned dot-product fixture exercises the packed-code correction.
3. I2_S GEMV/GEMM fixtures agree within the declared tolerance for selected
   model-relevant shapes, 1/2/4 threads, zero rows, and multi-plane input.
4. Native, AVX2-without-VNNI, and scalar profiles execute the same contract
   tests in CI.
5. For one fixed model, prompt, and runtime configuration, native AVX-VNNI and
   AVX2-without-VNNI produced bitwise-identical full-vocabulary logits.
6. Scalar whole-model logits were not bitwise identical; the first observed
   divergence was in RoPE, while selected layer-0 Q/K/V projections were
   bitwise equal.

Accordingly, “contract validated” means the documented I2_S/I8_S computation is
the optimization boundary for supported dimensions and fixtures. It does not
mean:

- every CPU ISA produces universally bitwise-identical whole-model output;
- every prompt, shape, or inherited model has been exhaustively tested;
- the converted GGUF has been certified against source-checkpoint logits,
  perplexity, or KL divergence;
- malformed externally authored packed tensors are always rejected by ordinary
  model loading.

See [`EXACTNESS_ORACLE.md`](EXACTNESS_ORACLE.md) and
[`exactness-oracle-results-2026-08-19.json`](exactness-oracle-results-2026-08-19.json)
for the fixed model-level regression evidence.
