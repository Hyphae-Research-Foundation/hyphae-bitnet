#define _POSIX_C_SOURCE 200809L

#include "celiums/bitnet_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

_Static_assert(sizeof(void *) == 8, "v0.3.0 ABI smoke test requires LP64");
_Static_assert(sizeof(size_t) == 8, "v0.3.0 ABI smoke test requires LP64");
_Static_assert(sizeof(celiums_bitnet_runtime_options) == 16, "runtime options ABI");
_Static_assert(sizeof(celiums_bitnet_model_options) == 16, "model options ABI");
_Static_assert(sizeof(celiums_bitnet_model_info) == 48, "model info ABI");
_Static_assert(sizeof(celiums_bitnet_session_options) == 32, "session options ABI");
_Static_assert(sizeof(celiums_bitnet_generation_options) == 48, "generation options ABI");
_Static_assert(sizeof(celiums_bitnet_generation_result) == 24, "generation result ABI");
_Static_assert(sizeof(celiums_bitnet_chat_message) == 16, "chat message ABI");
_Static_assert(offsetof(celiums_bitnet_runtime_options, api_version) == 8, "runtime api_version ABI");
_Static_assert(offsetof(celiums_bitnet_model_options, use_mmap) == 12, "model use_mmap ABI");
_Static_assert(offsetof(celiums_bitnet_model_options, check_tensors) == 14, "model check_tensors ABI");
_Static_assert(offsetof(celiums_bitnet_session_options, context_size) == 12, "session context ABI");
_Static_assert(offsetof(celiums_bitnet_session_options, threads_batch) == 28, "session threads ABI");
_Static_assert(CELIUMS_BITNET_STATUS_OK == 0, "status ABI");
_Static_assert(CELIUMS_BITNET_STATUS_CALLBACK_ABORTED == 10, "status ABI");

int main(int argc, char ** argv) {
    struct {
        uint64_t before;
        celiums_bitnet_runtime_options options;
        uint64_t after;
    } runtime_return = { UINT64_C(0x0123456789abcdef), { 0 }, UINT64_C(0xfedcba9876543210) };
    struct {
        uint64_t before;
        celiums_bitnet_model_options options;
        uint64_t after;
    } model_return = { UINT64_C(0x1122334455667788), { 0 }, UINT64_C(0x8877665544332211) };
    struct {
        uint64_t before;
        celiums_bitnet_session_options options;
        uint64_t after;
    } session_return = { UINT64_C(0x1020304050607080), { 0 }, UINT64_C(0x8070605040302010) };

    runtime_return.options = celiums_bitnet_runtime_default_options();
    model_return.options = celiums_bitnet_model_default_options();
    session_return.options = celiums_bitnet_session_default_options();
    if (runtime_return.before != UINT64_C(0x0123456789abcdef) ||
            runtime_return.after != UINT64_C(0xfedcba9876543210) ||
            model_return.before != UINT64_C(0x1122334455667788) ||
            model_return.after != UINT64_C(0x8877665544332211) ||
            session_return.before != UINT64_C(0x1020304050607080) ||
            session_return.after != UINT64_C(0x8070605040302010) ||
            runtime_return.options.struct_size != 16 ||
            runtime_return.options.api_version != CELIUMS_BITNET_API_VERSION ||
            model_return.options.struct_size != 16 || !model_return.options.use_mmap ||
            model_return.options.use_mlock || model_return.options.check_tensors ||
            session_return.options.struct_size != 32 || session_return.options.context_size != 2048 ||
            session_return.options.batch_size != 512 || session_return.options.ubatch_size != 512 ||
            session_return.options.threads != 2 || session_return.options.threads_batch != 2) {
        fputs("v0.3.0 by-value option ABI mismatch\n", stderr);
        return 1;
    }

    celiums_bitnet_runtime * runtime = NULL;
    if (celiums_bitnet_runtime_create(&runtime_return.options, &runtime) != CELIUMS_BITNET_STATUS_OK ||
            runtime == NULL) {
        fputs("v0.3.0 runtime_create failed\n", stderr);
        return 1;
    }

    char padding_path[] = "/tmp/hyphae-bitnet-v0.3.0-padding-XXXXXX";
    const int padding_fd = mkstemp(padding_path);
    FILE * padding_file = padding_fd >= 0 ? fdopen(padding_fd, "wb") : NULL;
    if (!padding_file) {
        fputs("v0.3.0 padding canary creation failed\n", stderr);
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }
    for (int index = 0; index < 1024; ++index) {
        if (fputc(0, padding_file) == EOF) {
            fclose(padding_file);
            remove(padding_path);
            celiums_bitnet_runtime_destroy(runtime);
            return 1;
        }
    }
    if (fclose(padding_file) != 0) {
        remove(padding_path);
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }

    celiums_bitnet_model * model = NULL;
    unsigned char * model_bytes = (unsigned char *) &model_return.options;
    model_bytes[15] = 1;
    const celiums_bitnet_status first = celiums_bitnet_model_load(
        runtime, padding_path, &model_return.options, &model);
    model_bytes[15] = 0;
    const celiums_bitnet_status second = celiums_bitnet_model_load(
        runtime, padding_path, &model_return.options, &model);
    remove(padding_path);
    if (first != CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED || second != first || model != NULL) {
        fputs("v0.3.0 model option padding changed behavior\n", stderr);
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }

    if (argc == 2) {
        celiums_bitnet_session * session = NULL;
        celiums_bitnet_model_options model_options = celiums_bitnet_model_default_options();
        celiums_bitnet_session_options session_options = celiums_bitnet_session_default_options();
        session_options.context_size = 128;
        session_options.batch_size = 64;
        session_options.ubatch_size = 64;
        session_options.threads = 1;
        session_options.threads_batch = 1;
        if (celiums_bitnet_model_load(runtime, argv[1], &model_options, &model) != CELIUMS_BITNET_STATUS_OK ||
                model == NULL ||
                celiums_bitnet_session_create(model, &session_options, &session) != CELIUMS_BITNET_STATUS_OK ||
                session == NULL) {
            fputs("v0.3.0 model/session creation failed\n", stderr);
            celiums_bitnet_model_destroy(model);
            celiums_bitnet_runtime_destroy(runtime);
            return 1;
        }

        size_t token_count = 0;
        if (celiums_bitnet_tokenize(model, "Hello", true, false, NULL, &token_count) !=
                    CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL || token_count == 0) {
            fputs("v0.3.0 tokenize sizing failed\n", stderr);
            celiums_bitnet_session_destroy(session);
            celiums_bitnet_model_destroy(model);
            celiums_bitnet_runtime_destroy(runtime);
            return 1;
        }
        celiums_bitnet_token tokens[32];
        if (token_count > sizeof(tokens) / sizeof(tokens[0]) ||
                celiums_bitnet_tokenize(model, "Hello", true, false, tokens, &token_count) !=
                    CELIUMS_BITNET_STATUS_OK ||
                celiums_bitnet_session_prefill(session, tokens, token_count, true) !=
                    CELIUMS_BITNET_STATUS_OK) {
            fputs("v0.3.0 prefill failed\n", stderr);
            celiums_bitnet_session_destroy(session);
            celiums_bitnet_model_destroy(model);
            celiums_bitnet_runtime_destroy(runtime);
            return 1;
        }
        size_t logits_count = 0;
        if (celiums_bitnet_session_copy_logits(session, NULL, &logits_count) !=
                    CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL || logits_count == 0) {
            fputs("v0.3.0 logits sizing failed\n", stderr);
            celiums_bitnet_session_destroy(session);
            celiums_bitnet_model_destroy(model);
            celiums_bitnet_runtime_destroy(runtime);
            return 1;
        }
        celiums_bitnet_session_destroy(session);
        celiums_bitnet_model_destroy(model);
    } else if (argc != 1) {
        fputs("usage: old-client [model.gguf]\n", stderr);
        celiums_bitnet_runtime_destroy(runtime);
        return 2;
    }

    celiums_bitnet_runtime_destroy(runtime);
    return 0;
}
