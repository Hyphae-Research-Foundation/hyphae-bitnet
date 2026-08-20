// BitNet I2_S GEMV/GEMM declarations
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_i2s_profile_phase {
    GGML_I2S_PROFILE_OP,
    GGML_I2S_PROFILE_QUANTIZE,
    GGML_I2S_PROFILE_BARRIER,
    GGML_I2S_PROFILE_TINYBLAS,
    GGML_I2S_PROFILE_FALLBACK,
    GGML_I2S_PROFILE_POSTPROCESS,
    GGML_I2S_PROFILE_PHASE_COUNT,
};

bool ggml_i2s_profile_enabled(void);
void ggml_i2s_profile_record(
    enum ggml_i2s_profile_phase phase,
    const char * tensor_name,
    int64_t m,
    int64_t n,
    int64_t k,
    uint64_t count,
    uint64_t time_us);

void ggml_gemv_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);

// Interleaved (8-row) variants — operate on repacked weight data
void ggml_gemv_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc);

#ifdef __cplusplus
}
#endif
