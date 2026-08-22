#include "ggml.h"
#include "ggml-cpu.h"
#include "celiums-exact.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>


static uint8_t code_for_weight(int value) {
    return (uint8_t)(value + 1);
}

static void pack_weights(const std::vector<int> & weights, int64_t m, int64_t k, std::vector<uint8_t> & packed) {
    packed.assign(m * k / 4 + 32, 0);
    for (int64_t row = 0; row < m; row++) {
        for (int64_t block = 0; block < k / 128; block++) {
            for (int gp = 0; gp < 32; gp++) {
                uint8_t value = 0;
                for (int group = 0; group < 4; group++) {
                    const int64_t index = row*k + block*128 + group*32 + gp;
                    value |= code_for_weight(weights[index]) << (6 - 2*group);
                }
                packed[row*k/4 + block*32 + gp] = value;
            }
        }
    }
    *(float *)(packed.data() + m*k/4) = 1.0f;
}

static int run_case(int64_t m, int64_t n, int64_t k, int64_t planes, int threads, bool zero_activations) {
    const size_t ctx_size = 64*1024*1024 + m*k/4 + n*k*planes*sizeof(float) + m*n*planes*sizeof(float);
    ggml_context * ctx = ggml_init({ ctx_size, nullptr, false });

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, k, m);
    ggml_tensor * activations = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n, planes);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, activations);

    std::vector<int> logical_weights(m*k);
    for (int64_t i = 0; i < m*k; i++) {
        logical_weights[i] = (int)(i % 3) - 1;
    }
    std::vector<uint8_t> packed;
    pack_weights(logical_weights, m, k, packed);
    memcpy(weights->data, packed.data(), packed.size());

    float * input = (float *)activations->data;
    for (int64_t i = 0; i < n*k*planes; i++) {
        input[i] = zero_activations ? 0.0f : 0.1f + 2.0f*std::cos((float)i + 0.25f);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, threads);

    bool failed = status != GGML_STATUS_SUCCESS;
    const float * got = (const float *)output->data;
    for (int64_t token = 0; token < n*planes; token++) {
        float amax = 0.0f;
        for (int64_t col = 0; col < k; col++) {
            amax = std::max(amax, std::fabs(input[token*k + col]));
        }
        const float scale = amax > 0.0f ? 127.0f/amax : 0.0f;
        std::vector<int8_t> quantized(k);
        for (int64_t col = 0; col < k; col++) {
            quantized[col] = (int8_t)std::round(input[token*k + col]*scale);
        }
        for (int64_t row = 0; row < m; row++) {
            int32_t dot = 0;
            for (int64_t col = 0; col < k; col++) {
                dot += logical_weights[row*k + col]*quantized[col];
            }
            const float expected = scale > 0.0f ? dot/scale : 0.0f;
            const float actual = got[token*m + row];
            failed = failed || !std::isfinite(actual) || std::fabs(actual - expected) > 1e-3f;
            if (scale > 0.0f) {
                int32_t unsigned_dot = 0;
                int32_t act_sum = 0;
                for (int64_t col = 0; col < k; col++) {
                    unsigned_dot += (logical_weights[row*k + col] + 1) * quantized[col];
                    act_sum += quantized[col];
                }
                const float recovered = celiums_exact_i2s_recover(
                    (float) unsigned_dot, act_sum, 1.0f / scale);
                failed = failed || std::fabs(recovered - expected) > 1e-3f;
            }
        }
    }

    printf("i2_s mul_mat m=%lld n=%lld k=%lld planes=%lld threads=%d zero=%d: %s\n",
           (long long)m, (long long)n, (long long)k, (long long)planes,
           threads, zero_activations, failed ? "FAILED" : "ok");
    ggml_free(ctx);
    return failed;
}

int main() {
    printf("celiums_exact_i2s_recover linked; I2_S mul_mat uses exact (D-S)*rho\n");
    int failures = 0;
    for (int threads : {1, 2, 4}) {
        failures += run_case(5, 1, 128, 1, threads, false);
        failures += run_case(8, 1, 256, 1, threads, false);
        failures += run_case(13, 1, 256, 1, threads, false);
        failures += run_case(5, 5, 256, 1, threads, false);
        failures += run_case(5, 3, 128, 1, threads, true);
        failures += run_case(5, 3, 128, 2, threads, false);
    }
    return failures != 0;
}
