#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exact Q1 2P-S / I2_S correction and pack helpers.
 * Kernels and oracles call this module so tests do not reimplement the math.
 *
 * Q1:  dot = d_w * d_x * (2P - S)   (Wang et al. bitnet.cpp / BitNet CPU)
 * I2_S: y = (D - S) * rho           (Celiums NUMERICAL_CONTRACT)
 * Prefill tile: 8 activation rows share one 4x8 weight panel (FBGEMM packing).
 */

enum {
    CELIUMS_EXACT_Q1_BLOCK = 128,
    CELIUMS_EXACT_Q1_ACT_TILE_PREF = 8,
    CELIUMS_EXACT_Q1_WEIGHT_COLS = 4,
};

int celiums_exact_q1_gemm_act_tile_rows(int nr);

void celiums_exact_q1_make_4x8(
    const void * row0,
    const void * row1,
    const void * row2,
    const void * row3,
    void * dst);

void celiums_exact_q1_pack_4x8(const void * packed_4rows, int n_blocks, void * dst);

int32_t celiums_exact_q1_corr_int(int32_t packed_dot, int32_t act_sum);

float celiums_exact_q1_dot(const void * q1_row, const void * q8_row, int n);

float celiums_exact_i2s_recover(float unsigned_dot, int32_t act_sum, float post_scale);

#ifdef __cplusplus
}
#endif
