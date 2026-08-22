#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "celiums-exact.h"

#include "ggml.h"

#include <string.h>

int celiums_exact_q1_gemm_act_tile_rows(int nr) {
    if (nr >= CELIUMS_EXACT_Q1_ACT_TILE_PREF) {
        return CELIUMS_EXACT_Q1_ACT_TILE_PREF;
    }
    if (nr >= CELIUMS_EXACT_Q1_WEIGHT_COLS) {
        return CELIUMS_EXACT_Q1_WEIGHT_COLS;
    }
    return 1;
}

int32_t celiums_exact_q1_corr_int(int32_t packed_dot, int32_t act_sum) {
    return packed_dot + packed_dot - act_sum;
}

float celiums_exact_i2s_recover(float unsigned_dot, int32_t act_sum, float post_scale) {
    return (unsigned_dot - (float) act_sum) * post_scale;
}

void celiums_exact_q1_make_4x8(
        const void * row0,
        const void * row1,
        const void * row2,
        const void * row3,
        void * dst) {
    const block_q1_0 * in[4] = {
        (const block_q1_0 *) row0,
        (const block_q1_0 *) row1,
        (const block_q1_0 *) row2,
        (const block_q1_0 *) row3,
    };
    ggml_half * d = (ggml_half *) dst;
    uint8_t * qs = (uint8_t *) (d + 4);
    for (int i = 0; i < 4; ++i) {
        d[i] = in[i]->d;
    }
    for (int byte_idx = 0; byte_idx < QK1_0 / 8; ++byte_idx) {
        qs[byte_idx * 4 + 0] = in[0]->qs[byte_idx];
        qs[byte_idx * 4 + 1] = in[1]->qs[byte_idx];
        qs[byte_idx * 4 + 2] = in[2]->qs[byte_idx];
        qs[byte_idx * 4 + 3] = in[3]->qs[byte_idx];
    }
}

void celiums_exact_q1_pack_4x8(const void * packed_4rows, int n_blocks, void * dst) {
    const block_q1_0 * packed = (const block_q1_0 *) packed_4rows;
    uint8_t * out = (uint8_t *) dst;
    const size_t block_bytes = 4 * sizeof(ggml_half) + (size_t) QK1_0 / 2;
    for (int l = 0; l < n_blocks; ++l) {
        celiums_exact_q1_make_4x8(
            &packed[0 * n_blocks + l],
            &packed[1 * n_blocks + l],
            &packed[2 * n_blocks + l],
            &packed[3 * n_blocks + l],
            out + (size_t) l * block_bytes);
    }
}

float celiums_exact_q1_dot(const void * q1_row, const void * q8_row, int n) {
    const int nb = n / QK1_0;
    const block_q1_0 * x = (const block_q1_0 *) q1_row;
    const block_q8_0 * y = (const block_q8_0 *) q8_row;
    float sum = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = ggml_fp16_to_fp32(x[ib].d);
        const block_q8_0 * y_ptr = y + ib * (QK1_0 / QK8_0);
        for (int k = 0; k < QK1_0 / QK8_0; ++k) {
            const float d1 = ggml_fp16_to_fp32(y_ptr[k].d);
            int32_t packed_dot = 0;
            int32_t act_sum = 0;
            const uint8_t * bits = x[ib].qs + k * 4;
            for (int b = 0; b < 4; ++b) {
                const uint8_t mask = bits[b];
                for (int p = 0; p < 8; ++p) {
                    const int8_t q = y_ptr[k].qs[b * 8 + p];
                    act_sum += q;
                    if (mask & (1u << p)) {
                        packed_dot += q;
                    }
                }
            }
            sum += d0 * d1 * (float) celiums_exact_q1_corr_int(packed_dot, act_sum);
        }
    }
    return sum;
}
