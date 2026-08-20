// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_quantize_row_i8_s(bool verbose) {
    int num_failed = 0;
    uint32_t rng = 0x5eed1234u;

    for (int pattern = 0; pattern < 5; ++pattern) {
        for (int n : {1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 4096}) {
            std::vector<float> input(n);
            std::vector<int8_t> output(n + 64, 42);

            for (int i = 0; i < n; ++i) {
                input[i] = pattern == 0 ? 0.0f :
                    pattern == 1 ? (i % 2 == 0 ? 1.0f : -1.0f) * (0.5f + i % 127) :
                    pattern == 2 ? 0.1f + 2.0f*cosf(i + 0.25f) :
                    pattern == 3 ?
                        (i % 6 == 0 ? nextafterf( 0.5f, 0.0f) :
                         i % 6 == 1 ? 0.5f :
                         i % 6 == 2 ? nextafterf( 0.5f, 1.0f) :
                         i % 6 == 3 ? nextafterf(-0.5f, 0.0f) :
                         i % 6 == 4 ? -0.5f : nextafterf(-0.5f, -1.0f)) :
                        (rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5,
                         ((int32_t)(rng & 0x00ffffffu) - 0x007fffff) / 8192.0f);
            }
            if (pattern == 1 || pattern == 3) {
                input[n - 1] = 127.0f;
            }

            float scale = 0.0f;
            int32_t sum = 0;
            quantize_row_i8_s(input.data(), output.data() + 32, n, &scale, &sum);

            float amax = 0.0f;
            for (float value : input) {
                amax = fmaxf(amax, fabsf(value));
            }
            const float scale_ref = amax > 0.0f ? 127.0f / amax : 0.0f;
            int32_t sum_ref = 0;
            bool failed = scale != scale_ref;
            for (int i = 0; i < n; ++i) {
                int value = (int)roundf(input[i] * scale_ref);
                value = value > 127 ? 127 : value;
                value = value < -128 ? -128 : value;
                failed = failed || output[i + 32] != value;
                sum_ref += value;
            }
            failed = failed || sum != sum_ref;
            for (int i = 0; i < 32; ++i) {
                failed = failed || output[i] != 42 || output[n + 32 + i] != 42;
            }
            num_failed += failed;
            if (failed || verbose) {
                printf(" i8_s quantize pattern=%d n=%4d:     %s (scale=%f sum=%d)\n",
                       pattern, n, RESULT_STR[failed], scale, sum);
            }
        }
    }

    return num_failed;
}

static uint8_t i2_s_code(float value, float scale) {
    if (fabsf(value) < 1e-6f) {
        return 1;
    }
    return value * scale > 0.0f ? 2 : 0;
}

static int test_quantize_i2_s(bool verbose) {
    int num_failed = 0;

    for (int nrows : {1, 2}) {
        const int n_per_row = 256;
        const int64_t n = nrows * n_per_row;
        std::vector<float> input(n);
        std::vector<uint8_t> output(n / 4 + 96, 0xA5);

        for (int64_t i = 0; i < n; ++i) {
            input[i] = i % 3 == 0 ? -2.0f : i % 3 == 1 ? 0.0f : 2.0f;
        }

        const size_t written = quantize_i2_s(input.data(), output.data() + 32, nrows, n_per_row, nullptr);
        const size_t packed_bytes = n / 4;
        const float scale = *(const float *)(output.data() + 32 + packed_bytes);
        bool failed = written != packed_bytes + 32 || scale != 2.0f ||
                      !ggml_validate_row_data(GGML_TYPE_I2_S, output.data() + 32, written);

        for (int row = 0; row < nrows; ++row) {
            for (int block = 0; block < n_per_row / 128; ++block) {
                for (int gp = 0; gp < 32; ++gp) {
                    uint8_t expected = 0;
                    for (int group = 0; group < 4; ++group) {
                        const int64_t index = row * n_per_row + block * 128 + group * 32 + gp;
                        expected |= i2_s_code(input[index], scale) << (6 - 2 * group);
                    }
                    const size_t byte_index = 32 + row * n_per_row / 4 + block * 32 + gp;
                    failed = failed || output[byte_index] != expected;
                }
            }
        }

        std::vector<float> dequantized(n_per_row);
        dequantize_row_i2_s(output.data() + 32, dequantized.data(), n_per_row, scale);
        for (int i = 0; i < n_per_row; ++i) {
            const float expected = input[i] < 0 ? -scale : input[i] > 0 ? scale : 0.0f;
            failed = failed || dequantized[i] != expected;
        }

        for (int i = 0; i < 32; ++i) {
            failed = failed || output[i] != 0xA5;
            failed = failed || output[32 + packed_bytes + 32 + i] != 0xA5;
        }

        num_failed += failed;
        if (failed || verbose) {
            printf(" i2_s packing rows=%d:               %s (scale=%f bytes=%zu)\n",
                   nrows, RESULT_STR[failed], scale, written);
        }
    }

    {
        const int n = 128;
        std::vector<float> input(n, 0.0f);
        std::vector<uint8_t> output(n / 4 + 32);
        const size_t written = quantize_i2_s(input.data(), output.data(), 1, n, nullptr);
        const float scale = *(const float *)(output.data() + n / 4);
        const bool failed = scale != 1e-5f || !ggml_validate_row_data(GGML_TYPE_I2_S, output.data(), written);
        num_failed += failed;
    }

    return num_failed;
}

static int test_vec_dot_i2_s(bool verbose) {
    int num_failed = 0;

    for (int n : {128, 256, 2560, 4096, 6912, 13824}) {
        std::vector<uint8_t> weights(n / 4, 0xAA); // four unsigned +1 codes (u=2)
        std::vector<int8_t> activations(n, 127);
        float result = 0.0f;
        ggml_vec_dot_i2_i8_s(n, &result, 0, weights.data(), n, activations.data(), 0, 1);

        const int32_t expected_unsigned = 2 * 127 * n;
        const int32_t activation_sum = 127 * n;
        const bool failed = result != expected_unsigned || result - activation_sum != activation_sum;
        num_failed += failed;
        if (failed || verbose) {
            printf(" i2_s vec_dot n=%5d:                %s (got=%.0f expected=%d)\n",
                   n, RESULT_STR[failed], result, expected_unsigned);
        }
    }

    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_quantize_row_i8_s(verbose);
    num_failed += test_quantize_i2_s(verbose);
    num_failed += test_vec_dot_i2_s(verbose);
    num_failed += test_vec_dot_q(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
