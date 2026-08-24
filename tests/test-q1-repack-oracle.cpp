/* Packed Q1_0 vs in-RAM compute layout. No 27B fixture. */
#define GGML_COMMON_IMPL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"
#include "ggml-cpu/quants.h"
#include "celiums-exact.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static void interleave_q1_4x8(const block_q1_0 * packed, int nb, block_q1_0x4 * interleaved) {
    celiums_exact_q1_pack_4x8(packed, nb, interleaved);
}

static void interleave_q1_4x4(const block_q1_0 * packed, int nb, block_q1_0x4 * interleaved) {
    for (int l = 0; l < nb; ++l) {
        for (int row = 0; row < 4; ++row) {
            interleaved[l].d[row] = packed[row * nb + l].d;
        }
        for (int k = 0; k < QK1_0 / QK8_0; ++k) {
            for (int tile = 0; tile < QK8_0 / 4; ++tile) {
                uint8_t packed_lo = 0;
                uint8_t packed_hi = 0;
                const int weight_base = k * QK8_0 + tile * 4;
                for (int pos = 0; pos < 4; ++pos) {
                    const int weight_idx = weight_base + pos;
                    const int byte_idx = weight_idx / 8;
                    const int bit_idx = weight_idx % 8;
                    packed_lo |= ((packed[0 * nb + l].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_lo |= ((packed[1 * nb + l].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                    packed_hi |= ((packed[2 * nb + l].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_hi |= ((packed[3 * nb + l].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                    }
                interleaved[l].qs[k * 16 + 2 * tile + 0] = packed_lo;
                interleaved[l].qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
    }
}

static void interleave_q8(const block_q8_0 rows[4], int blocklen, block_q8_0x4 * out) {
    for (int i = 0; i < 4; ++i) {
        out->d[i] = rows[i].d;
    }
    const int end = QK8_0 * 4 / blocklen;
    for (int i = 0; i < end; ++i) {
        const int src_id = i % 4;
        const int src_offset = (i / 4) * blocklen;
        memcpy(out->qs + i * blocklen, rows[src_id].qs + src_offset, blocklen);
    }
}

static int check3(const char * what, int idx, float packed_v, float layout_v, float generic_v) {
    const float tol = 1e-3f + 1e-4f * std::fabs(packed_v);
    if (!std::isfinite(layout_v) || !std::isfinite(generic_v) ||
            std::fabs(layout_v - packed_v) > tol ||
            std::fabs(generic_v - packed_v) > tol) {
        fprintf(stderr, "q1 oracle %s idx %d packed=%.6f layout=%.6f generic=%.6f\n",
                what, idx, packed_v, layout_v, generic_v);
        return 1;
    }
    return 0;
}

static int test_gemv(int n) {
    const int nb = n / QK1_0;
    const int nc = 4;
    std::vector<block_q1_0> packed((size_t) nc * (size_t) nb);
    std::vector<block_q8_0> act((size_t) (n / QK8_0));
    for (int row = 0; row < nc; ++row) {
        for (int l = 0; l < nb; ++l) {
            block_q1_0 & blk = packed[row * nb + l];
            /* IEEE fp16 0.5, 0.5625, ... so VNNI F16C and ggml_fp16_to_fp32 agree. */
            const uint16_t d0_bits[4] = { 0x3800, 0x3880, 0x3900, 0x3980 };
            blk.d = d0_bits[row % 4];
            for (int b = 0; b < QK1_0 / 8; ++b) {
                blk.qs[b] = (uint8_t) (0xA5 ^ (row * 17 + l * 13 + b));
            }
        }
    }
    for (int i = 0; i < n / QK8_0; ++i) {
        const uint16_t d1_bits[8] = { 0x3C00, 0x3C00, 0x3A00, 0x3B00, 0x3C00, 0x3D00, 0x3A80, 0x3B80 };
        act[i].d = d1_bits[i % 8];
        for (int q = 0; q < QK8_0; ++q) {
            act[i].qs[q] = (int8_t) ((q + i * 3) % 17 - 8);
        }
    }

    float packed_out[4] = { 0, 0, 0, 0 };
    float mag = 0.0f;
    for (int row = 0; row < nc; ++row) {
        packed_out[row] = celiums_exact_q1_dot(
            &packed[(size_t) row * (size_t) nb], act.data(), n);
        mag += std::fabs(packed_out[row]);
    }
    if (mag < 1e-6f) {
        fprintf(stderr, "q1 oracle gemv packed reference is degenerate d0=%g lib=%g %g %g %g\n",
                (double) ggml_fp16_to_fp32(packed[0].d),
                (double) packed_out[0], (double) packed_out[1],
                (double) packed_out[2], (double) packed_out[3]);
        return 1;
    }

    std::vector<block_q1_0x4> interleaved((size_t) nb);
    float layout_out[4] = { 0, 0, 0, 0 };
    float generic_out[4] = { 0, 0, 0, 0 };
    interleave_q1_4x8(packed.data(), nb, interleaved.data());
    ggml_gemv_q1_0_4x8_q8_0(n, layout_out, 0, interleaved.data(), act.data(), 1, nc);
    ggml_gemv_q1_0_4x8_q8_0_generic(n, generic_out, 0, interleaved.data(), act.data(), 1, nc);
    for (int row = 0; row < nc; ++row) {
        if (check3("gemv-4x8", row, packed_out[row], layout_out[row], generic_out[row])) {
            return 1;
        }
    }

    interleave_q1_4x4(packed.data(), nb, interleaved.data());
    memset(layout_out, 0, sizeof(layout_out));
    ggml_gemv_q1_0_4x4_q8_0(n, layout_out, 0, interleaved.data(), act.data(), 1, nc);
    for (int row = 0; row < nc; ++row) {
        if (check3("gemv-4x4", row, packed_out[row], layout_out[row], layout_out[row])) {
            return 1;
        }
    }
    return 0;
}

static int test_gemm(int n, int nr) {
    const int nb = n / QK1_0;
    const int nq8 = n / QK8_0;
    const int nc = 4;
    if (nr % 4 != 0 || nr <= 0) {
        return 1;
    }
    std::vector<block_q1_0> packed((size_t) nc * (size_t) nb);
    std::vector<block_q8_0> act((size_t) nr * (size_t) nq8);
    for (int row = 0; row < nc; ++row) {
        for (int l = 0; l < nb; ++l) {
            block_q1_0 & blk = packed[row * nb + l];
            const uint16_t d0_bits[4] = { 0x3666, 0x3800, 0x3880, 0x3900 };
            blk.d = d0_bits[row % 4];
            for (int b = 0; b < QK1_0 / 8; ++b) {
                blk.qs[b] = (uint8_t) (0x3C ^ (row * 29 + l * 11 + b));
            }
        }
    }
    for (int m = 0; m < nr; ++m) {
        for (int i = 0; i < nq8; ++i) {
            block_q8_0 & blk = act[(size_t) m * (size_t) nq8 + (size_t) i];
            const uint16_t d1_bits[8] = { 0x3B80, 0x3C00, 0x3A00, 0x3C80, 0x3B00, 0x3A80, 0x3D00, 0x3980 };
            blk.d = d1_bits[(m + i) % 8];
            for (int q = 0; q < QK8_0; ++q) {
                blk.qs[q] = (int8_t) ((q + m * 5 + i * 2) % 19 - 9);
            }
        }
    }

    std::vector<block_q8_0x4> act4x8((size_t) (nr / 4) * (size_t) nq8);
    std::vector<block_q8_0x4> act4x4((size_t) (nr / 4) * (size_t) nq8);
    for (int group = 0; group < nr / 4; ++group) {
        for (int i = 0; i < nq8; ++i) {
            block_q8_0 rows[4];
            for (int m = 0; m < 4; ++m) {
                rows[m] = act[(size_t) (group * 4 + m) * (size_t) nq8 + (size_t) i];
            }
            interleave_q8(rows, 8, &act4x8[(size_t) group * (size_t) nq8 + (size_t) i]);
            interleave_q8(rows, 4, &act4x4[(size_t) group * (size_t) nq8 + (size_t) i]);
        }
    }

    std::vector<float> packed_out((size_t) nr * (size_t) nc);
    for (int m = 0; m < nr; ++m) {
        for (int j = 0; j < nc; ++j) {
            packed_out[(size_t) m * (size_t) nc + (size_t) j] = celiums_exact_q1_dot(
                &packed[(size_t) j * (size_t) nb],
                &act[(size_t) m * (size_t) nq8], n);
        }
    }

    std::vector<float> layout_out((size_t) nr * (size_t) nc, 0.0f);
    char what[32];
    std::snprintf(what, sizeof(what), "gemm-nr%d", nr);
    std::vector<block_q1_0x4> interleaved((size_t) nb);
    interleave_q1_4x8(packed.data(), nb, interleaved.data());
    ggml_gemm_q1_0_4x8_q8_0(n, layout_out.data(), nc, interleaved.data(), act4x8.data(), nr, nc);
    std::vector<float> generic_out((size_t) nr * (size_t) nc, 0.0f);
    ggml_gemm_q1_0_4x8_q8_0_generic(n, generic_out.data(), nc, interleaved.data(), act4x8.data(), nr, nc);
    for (int idx = 0; idx < nr * nc; ++idx) {
        if (check3("gemm-4x8", idx, packed_out[(size_t) idx], layout_out[(size_t) idx], generic_out[(size_t) idx])) {
            return 1;
        }
    }

    interleave_q1_4x4(packed.data(), nb, interleaved.data());
    std::fill(layout_out.begin(), layout_out.end(), 0.0f);
    ggml_gemm_q1_0_4x4_q8_0(n, layout_out.data(), nc, interleaved.data(), act4x4.data(), nr, nc);
    for (int idx = 0; idx < nr * nc; ++idx) {
        if (check3(what, idx, packed_out[(size_t) idx], layout_out[(size_t) idx], layout_out[(size_t) idx])) {
            return 1;
        }
    }
    return 0;
}

int main(void) {
    /* Q1 kernels use GGML_CPU_FP16_TO_FP32 (x86 LUT). Fill it before any gemv/gemm. */
    ggml_cpu_init();
    if (celiums_exact_q1_gemm_act_tile_rows(128) != CELIUMS_EXACT_Q1_ACT_TILE_PREF) {
        fprintf(stderr, "tile planner expected %d for nr=128 got %d\n",
                CELIUMS_EXACT_Q1_ACT_TILE_PREF, celiums_exact_q1_gemm_act_tile_rows(128));
        return 1;
    }
    if (celiums_exact_q1_gemm_act_tile_rows(4) != 4) {
        fprintf(stderr, "tile planner expected 4 for nr=4\n");
        return 1;
    }
    printf("celiums_exact_q1_gemm_act_tile_rows(128)=%d shipped 8-row GEMM weight reuse\n",
           celiums_exact_q1_gemm_act_tile_rows(128));
    printf("celiums_exact_q1_corr_int / pack_4x8 / q1_dot / i2s_recover linked\n");
    if (test_gemv(QK1_0) != 0) return 1;
    if (test_gemv(2 * QK1_0) != 0) return 1;
    if (test_gemm(QK1_0, 4) != 0) return 1;
    if (test_gemm(2 * QK1_0, 4) != 0) return 1;
    if (test_gemm(QK1_0, 8) != 0) return 1;
    if (test_gemm(2 * QK1_0, 8) != 0) return 1;
    printf("q1 oracle packed/exact-lib vs layout vs generic: ok\n");
    return 0;
}
