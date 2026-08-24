#include "celiums/bitnet_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    celiums_bitnet_runtime_options options = celiums_bitnet_runtime_default_options();
    celiums_bitnet_runtime * runtime = NULL;

    if (sizeof(celiums_bitnet_runtime_options) != 16 ||
            sizeof(celiums_bitnet_model_options) != 16 ||
            sizeof(celiums_bitnet_session_options) != 32 ||
            offsetof(celiums_bitnet_model_options, check_tensors) != 14 ||
            offsetof(celiums_bitnet_session_options, threads_batch) != 28 ||
            options.struct_size != sizeof(options) ||
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

    celiums_bitnet_model_options legacy_model = celiums_bitnet_model_default_options();
    celiums_bitnet_model * missing_model = NULL;
    ((unsigned char *) &legacy_model)[15] = 0xa5;
    const celiums_bitnet_status padded_a = celiums_bitnet_model_load(
        runtime, "padding-canary-does-not-exist.gguf", &legacy_model, &missing_model);
    ((unsigned char *) &legacy_model)[15] = 0x5a;
    const celiums_bitnet_status padded_b = celiums_bitnet_model_load(
        runtime, "padding-canary-does-not-exist.gguf", &legacy_model, &missing_model);
    if (padded_a != CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED || padded_b != padded_a ||
            missing_model != NULL) {
        fprintf(stderr, "Celiums Runtime legacy model padding test failed\n");
        celiums_bitnet_runtime_destroy(runtime);
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

    celiums_bitnet_session_options legacy_session = celiums_bitnet_session_default_options();
    legacy_session.context_size = 8192;
    const uint64_t legacy_zero = celiums_bitnet_estimate_session_ram_bytes(&legacy_session, 0);
    const uint64_t legacy_packed = celiums_bitnet_estimate_session_ram_bytes(
        &legacy_session, 8ull * 1024ull * 1024ull);

    struct {
        celiums_bitnet_session_options_ex options;
        unsigned char future[16];
    } session_buffer;
    memset(&session_buffer, 0xa5, sizeof(session_buffer));
    celiums_bitnet_runtime_options_ex runtime_ex;
    celiums_bitnet_model_options_ex model_ex;
    if (celiums_bitnet_runtime_options_ex_init(&runtime_ex, sizeof(runtime_ex)) !=
                CELIUMS_BITNET_STATUS_OK ||
            celiums_bitnet_model_options_ex_init(&model_ex, sizeof(model_ex)) !=
                CELIUMS_BITNET_STATUS_OK ||
            celiums_bitnet_session_options_ex_init(
                &session_buffer.options, sizeof(session_buffer)) != CELIUMS_BITNET_STATUS_OK ||
            celiums_bitnet_session_options_ex_init(
                &session_buffer.options, sizeof(session_buffer.options) - 1) !=
                CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL ||
            celiums_bitnet_runtime_options_ex_init(NULL, sizeof(runtime_ex)) !=
                CELIUMS_BITNET_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "Celiums Runtime extended option initialization failed\n");
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }
    if (runtime_ex.struct_size != sizeof(runtime_ex) ||
            model_ex.struct_size != sizeof(model_ex) || !model_ex.use_mmap ||
            !model_ex.use_compute_layout ||
            session_buffer.options.struct_size != sizeof(session_buffer) ||
            session_buffer.options.n_seq != 1 || !session_buffer.options.use_compute_layout) {
        fprintf(stderr, "Celiums Runtime extended option defaults failed\n");
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }
    for (size_t index = 0; index < sizeof(session_buffer.future); ++index) {
        if (session_buffer.future[index] != 0) {
            fprintf(stderr, "Celiums Runtime extended option future bytes were not cleared\n");
            celiums_bitnet_runtime_destroy(runtime);
            return 1;
        }
    }
    session_buffer.options.n_seq = 8;
    session_buffer.options.context_size = 8192;
    session_buffer.options.use_compute_layout = true;
    session_buffer.options.ram_budget_bytes = 1024;
    const uint64_t needed = celiums_bitnet_estimate_session_ram_bytes_ex(
        &session_buffer.options, 8ull * 1024ull * 1024ull);
    session_buffer.options.use_compute_layout = false;
    const uint64_t packed_only = celiums_bitnet_estimate_session_ram_bytes_ex(
        &session_buffer.options, 8ull * 1024ull * 1024ull);
    if (needed <= session_buffer.options.ram_budget_bytes ||
            needed <= packed_only ||
            legacy_zero != legacy_packed ||
            celiums_bitnet_default_ram_budget_bytes() == 0 ||
            celiums_bitnet_host_ram_bytes() < celiums_bitnet_default_ram_budget_bytes()) {
        fprintf(stderr, "Celiums Runtime RAM budget estimate failed needed=%llu budget=%llu host=%llu\n",
                (unsigned long long) needed,
                (unsigned long long) session_buffer.options.ram_budget_bytes,
                (unsigned long long) celiums_bitnet_host_ram_bytes());
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }

    celiums_bitnet_runtime * runtime_extended = NULL;
    runtime_ex.ram_budget_bytes = 1;
    if (celiums_bitnet_runtime_create_ex(&runtime_ex, &runtime_extended) !=
            CELIUMS_BITNET_STATUS_OK || runtime_extended == NULL) {
        fprintf(stderr, "Celiums Runtime extended create failed\n");
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }
    const char * padding_path = "celiums-runtime-padding-canary.gguf";
    FILE * padding_file = fopen(padding_path, "wb");
    if (!padding_file) {
        fprintf(stderr, "Celiums Runtime padding canary creation failed\n");
        celiums_bitnet_runtime_destroy(runtime_extended);
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }
    if (fputc(0, padding_file) == EOF || fclose(padding_file) != 0) {
        fprintf(stderr, "Celiums Runtime padding canary creation failed\n");
        remove(padding_path);
        celiums_bitnet_runtime_destroy(runtime_extended);
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }
    legacy_model = celiums_bitnet_model_default_options();
    ((unsigned char *) &legacy_model)[15] = 1;
    const celiums_bitnet_status ignored_one = celiums_bitnet_model_load(
        runtime_extended, padding_path, &legacy_model, &missing_model);
    ((unsigned char *) &legacy_model)[15] = 0;
    const celiums_bitnet_status ignored_zero = celiums_bitnet_model_load(
        runtime_extended, padding_path, &legacy_model, &missing_model);
    remove(padding_path);
    if (ignored_zero != CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED ||
            ignored_one != ignored_zero || missing_model != NULL) {
        fprintf(stderr, "Celiums Runtime legacy model byte 15 was observed\n");
        celiums_bitnet_runtime_destroy(runtime_extended);
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }
    celiums_bitnet_runtime_destroy(runtime_extended);

    celiums_bitnet_runtime_destroy(runtime);
    return 0;
}
