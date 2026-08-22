#include "celiums/bitnet_runtime.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    celiums_bitnet_runtime_options options = celiums_bitnet_runtime_default_options();
    celiums_bitnet_runtime * runtime = NULL;

    if (options.struct_size != sizeof(options) ||
            options.api_version != CELIUMS_BITNET_API_VERSION ||
            strcmp(celiums_bitnet_version(), CELIUMS_BITNET_TEST_VERSION) != 0 ||
            !(strlen(celiums_bitnet_product_commit()) == 9 ||
              strcmp(celiums_bitnet_product_commit(), "unknown") == 0) ||
            strcmp(celiums_bitnet_model_family_string(
                CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S), "bitnet-b1.58-i2_s") != 0 ||
            strcmp(celiums_bitnet_model_family_string(
                CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0), "bonsai-qwen35-q1_0") != 0 ||
            strcmp(celiums_bitnet_model_family_string(
                CELIUMS_BITNET_MODEL_FAMILY_UNKNOWN), "unknown") != 0 ||
            celiums_bitnet_runtime_create(&options, &runtime) != CELIUMS_BITNET_STATUS_OK ||
            runtime == NULL) {
        fprintf(stderr, "Celiums Runtime C API smoke test failed\n");
        return 1;
    }

    celiums_bitnet_model * model = (celiums_bitnet_model *) runtime;
    if (celiums_bitnet_model_load_family(
            runtime, "unused.gguf", (celiums_bitnet_model_family) 99, NULL, &model) !=
            CELIUMS_BITNET_STATUS_INVALID_ARGUMENT || model != NULL ||
            celiums_bitnet_model_get_family(NULL) != CELIUMS_BITNET_MODEL_FAMILY_UNKNOWN) {
        fprintf(stderr, "Celiums Runtime model family API smoke test failed\n");
        return 1;
    }

    if (strcmp(celiums_bitnet_status_string(CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED),
               "ram budget exceeded") != 0) {
        fprintf(stderr, "Celiums Runtime RAM budget status string failed\n");
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }

    celiums_bitnet_session_options session_options = celiums_bitnet_session_default_options();
    session_options.n_seq = 8;
    session_options.context_size = 8192;
    session_options.use_compute_layout = true;
    session_options.ram_budget_bytes = 1024;
    const uint64_t needed = celiums_bitnet_estimate_session_ram_bytes(&session_options, 8ull * 1024ull * 1024ull);
    session_options.use_compute_layout = false;
    const uint64_t packed_only = celiums_bitnet_estimate_session_ram_bytes(
        &session_options, 8ull * 1024ull * 1024ull);
    if (needed <= session_options.ram_budget_bytes ||
            needed <= packed_only ||
            celiums_bitnet_default_ram_budget_bytes() == 0 ||
            celiums_bitnet_host_ram_bytes() < celiums_bitnet_default_ram_budget_bytes()) {
        fprintf(stderr, "Celiums Runtime RAM budget estimate failed needed=%llu budget=%llu host=%llu\n",
                (unsigned long long) needed,
                (unsigned long long) session_options.ram_budget_bytes,
                (unsigned long long) celiums_bitnet_host_ram_bytes());
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }

    celiums_bitnet_runtime_destroy(runtime);
    return 0;
}
