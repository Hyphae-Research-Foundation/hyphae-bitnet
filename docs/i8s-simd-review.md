# `quantize_row_i8_s` SIMD review

## Contract

For a finite F32 row `x[0:n]`, the function computes:

```text
amax  = max(abs(x[i]))
scale = amax > 0 ? 127 / amax : 0
q[i]  = clamp(roundf(x[i] * scale), -128, 127)
sum   = sum(q[i])
```

The I2_S matrix product stores ternary weights as unsigned codes
`u = w + 1`. It therefore computes `sum(u[i] * q[i])` and needs `sum(q[i])`
to recover `sum(w[i] * q[i])`.

## Maximum Reduction

`andnot(-0.0f, value)` clears only the IEEE-754 sign bit, so it produces the
same absolute value as `fabsf()` for finite values. Eight independent lanes
accumulate maxima across the vector loop. The 256-bit value is then reduced to
four lanes, two lanes, and one lane. Elements after the last complete group of
eight use the unchanged scalar loop.

The reduction is equivalent to the scalar loop for finite activations because
`max` is associative and no rounding operation is involved. NaN behavior is
not part of this proof; model activations are expected to be finite and the
existing scalar function does not define a useful NaN contract either.

## Round Half Away From Zero

`roundf()` does not use the current floating-point rounding mode. For finite
values in the quantizer range, it truncates toward zero and increments the
magnitude when the absolute fractional part is at least 0.5:

```text
t = trunc(v)
f = abs(v - float(t))
r = t + (f >= 0.5 ? copysign(1, v) : 0)
```

The SIMD helper implements this definition explicitly with `cvttps_epi32`,
conversion back to F32, subtraction, absolute value, comparison, and a signed
unit adjustment. This avoids the earlier `v +/- 0.5` shortcut, which can change
a non-tie value into an exact tie because the addition itself rounds in F32.

The sign is read from the original F32 bits rather than the truncated integer.
This is required for inputs in `(-1, -0.5]`, where truncation produces zero but
the rounding direction must still be `-1`.

## Packing And Lane Permutation

Four vectors contain 32 signed int32 values:

```text
i0 = q[ 0: 8], i1 = q[ 8:16]
i2 = q[16:24], i3 = q[24:32]
```

`packs_epi32(i0, i1)` and `packs_epi32(i2, i3)` saturate to int16. The values
are already in `[-127, 127]`, so saturation does not alter them. AVX2 packing
operates independently in each 128-bit lane, producing this logical order:

```text
[0..3, 8..11, 4..7, 12..15]
[16..19, 24..27, 20..23, 28..31]
```

The subsequent `packs_epi16` keeps the same lane-local interleaving. Treating
the result as eight 32-bit groups, its order is `[0,4,1,5,2,6,3,7]` relative
to the desired contiguous order. `permutevar8x32` with that same self-inverse
permutation restores bytes `q[0]` through `q[31]` before the unaligned store.

Guard bytes before and after the output prove that the vector store and scalar
tail remain within `y[0:n]`.

## Sum Reduction And Overflow Bound

The four int32 vectors are added lane-wise into eight int32 accumulators. Every
input is at most 127 in magnitude. An int32 lane receives at most `n / 8`
values, so it cannot overflow for the BitNet dimensions used here. Even a row
of 13,824 values has a per-lane bound of `1,728 * 127 = 219,456` and a total
bound of `1,755,648`.

The horizontal reduction first adds the low and high 128-bit halves, then the
two 64-bit groups, then adjacent 32-bit lanes. It therefore returns the exact
sum of all eight accumulators. The scalar tail adds any remaining elements.

## Validation Matrix

- Exact comparison of scale, every byte, and sum against scalar `roundf()`.
- Sizes around every boundary: 31/32/33, 63/64/65, 127/128/129, and 4096.
- Zero rows, alternating half-integers, trigonometric inputs, values immediately
  below/on/above positive and negative half ties, and deterministic random data.
- Guard bytes around output.
- Native AVX-512/VNNI build, AVX2-only build, and scalar standalone build.
- GCC 13 and Clang 18.
- End-to-end pp128/tg128 benchmark with strict affinity.

## Maintenance

Celiums maintains this implementation in its pinned llama.cpp dependency. The
exact dependency commit is part of every Celiums BitNet release and must pass
the same strict numerical and benchmark matrix before it is updated.
