# Strict Numerical Contract

Celiums BitNet strict mode preserves the I2_S and I8_S computation used by the
Microsoft-published BitNet b1.58 checkpoints.

## I2_S Weights

Weights are ternary:

```text
w[i] in {-1, 0, 1}
u[i] = w[i] + 1 in {0, 1, 2}
```

Each block of 128 weights occupies 32 bytes. Byte `gp` contains positions
`gp`, `gp + 32`, `gp + 64`, and `gp + 96` in bits `7:6`, `5:4`, `3:2`, and
`1:0` respectively. The tensor uses one F32 weight scale stored after all
packed bytes in a 32-byte trailer.

## I8_S Activations

For each finite F32 activation row:

```text
amax  = max(abs(x[i]))
scale = amax > 0 ? 127 / amax : 0
q[i]  = clamp(roundf(x[i] * scale), -128, 127)
sum   = sum(q[i])
```

`roundf` means round to nearest with half-way cases away from zero. A row whose
maximum is zero produces zero bytes, zero sum, and an exact zero matrix-product
output.

## Matrix Product

The integer kernel computes unsigned-weight products:

```text
D[j] = sum(u[j, i] * q[i])
```

The ternary output is recovered as:

```text
y[j] = (D[j] - sum) * weight_scale / scale
```

Integer accumulation must not overflow before widening to I32. Strict kernels
must preserve the same quantized bytes, activation sums, and integer dot
products as the scalar reference.

## Unsupported Inputs

I2_S dimensions must currently be multiples of 128. Non-finite activations and
non-standard packed layouts are outside the strict contract and must be
rejected rather than interpreted heuristically.
