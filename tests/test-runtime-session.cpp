#include "celiums/bitnet_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool collect(celiums_bitnet_token, const char * piece, size_t size, void * user_data) {
    static_cast<std::vector<char> *>(user_data)->insert(
        static_cast<std::vector<char> *>(user_data)->end(), piece, piece + size);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "model path is required\n");
        return 2;
    }

    celiums_bitnet_runtime * runtime = nullptr;
    celiums_bitnet_model * model = nullptr;
    celiums_bitnet_session * session = nullptr;
    celiums_bitnet_request * request = nullptr;
    auto runtime_options = celiums_bitnet_runtime_default_options();
    auto model_options = celiums_bitnet_model_default_options();
    auto session_options = celiums_bitnet_session_default_options();
    session_options.context_size = 128;
    session_options.batch_size = 64;
    session_options.ubatch_size = 64;
    session_options.threads = 1;
    session_options.threads_batch = 1;

    auto status = celiums_bitnet_runtime_create(&runtime_options, &runtime);
    if (status == CELIUMS_BITNET_STATUS_OK) status = celiums_bitnet_model_load(runtime, argv[1], &model_options, &model);
    if (status == CELIUMS_BITNET_STATUS_OK) status = celiums_bitnet_session_create(model, &session_options, &session);
    if (status != CELIUMS_BITNET_STATUS_OK) return 1;

    size_t token_count = 0;
    status = celiums_bitnet_tokenize(model, "Hello", true, false, nullptr, &token_count);
    if (status != CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL || token_count == 0) return 1;
    std::vector<celiums_bitnet_token> tokens(token_count);
    status = celiums_bitnet_tokenize(model, "Hello", true, false, tokens.data(), &token_count);
    if (status != CELIUMS_BITNET_STATUS_OK) return 1;

    size_t text_size = 0;
    status = celiums_bitnet_detokenize(model, tokens.data(), token_count, true, false, nullptr, &text_size);
    std::vector<char> text(text_size);
    if (status != CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL ||
            celiums_bitnet_detokenize(model, tokens.data(), token_count, true, false, text.data(), &text_size) != CELIUMS_BITNET_STATUS_OK ||
            std::strstr(text.data(), "Hello") == nullptr) return 1;

    status = celiums_bitnet_session_prefill(session, tokens.data(), token_count, true);
    size_t logits_count = 0;
    if (status != CELIUMS_BITNET_STATUS_OK ||
            celiums_bitnet_session_copy_logits(session, nullptr, &logits_count) != CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL ||
            logits_count != 128256) return 1;
    std::vector<float> logits(logits_count);
    if (celiums_bitnet_session_copy_logits(session, logits.data(), &logits_count) != CELIUMS_BITNET_STATUS_OK) return 1;
    for (float value : logits) if (!std::isfinite(value)) return 1;

    const celiums_bitnet_chat_message messages[] = {{"user", "Hello"}};
    size_t chat_size = 0;
    status = celiums_bitnet_model_apply_chat_template(model, messages, 1, true, nullptr, &chat_size);
    if (status != CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL || chat_size == 0) return 1;
    std::vector<char> chat(chat_size);
    status = celiums_bitnet_model_apply_chat_template(model, messages, 1, true, chat.data(), &chat_size);
    if (status != CELIUMS_BITNET_STATUS_OK || std::strstr(chat.data(), "Hello") == nullptr) return 1;

    if (celiums_bitnet_request_create(session, &request) != CELIUMS_BITNET_STATUS_OK) return 1;
    auto generation = celiums_bitnet_generation_default_options();
    generation.max_tokens = 4;
    generation.temperature = 0.0f;
    celiums_bitnet_generation_result result = {};
    result.struct_size = sizeof(result);
    result.api_version = CELIUMS_BITNET_API_VERSION;
    std::vector<char> output;
    status = celiums_bitnet_generate(request, "Hello", &generation, collect, &output, &result);
    const std::string generated(output.begin(), output.end());
    if (status != CELIUMS_BITNET_STATUS_OK || generated != ", I am a" || result.generated_tokens != 4) return 1;

    celiums_bitnet_request_cancel(request);
    if (!celiums_bitnet_request_is_cancelled(request)) return 1;
    celiums_bitnet_request_destroy(request);
    celiums_bitnet_session_destroy(session);
    celiums_bitnet_model_destroy(model);
    celiums_bitnet_runtime_destroy(runtime);
    return 0;
}
