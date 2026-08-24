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
    if (argc != 3) {
        fprintf(stderr, "model path and family are required\n");
        return 2;
    }
    const bool bonsai = std::strcmp(argv[2], "bonsai") == 0;
    const celiums_bitnet_model_family family = bonsai
        ? CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0
        : CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S;

    celiums_bitnet_runtime * runtime = nullptr;
    celiums_bitnet_model * model = nullptr;
    celiums_bitnet_session * session = nullptr;
    celiums_bitnet_request * request = nullptr;
    celiums_bitnet_runtime_options_ex runtime_options;
    celiums_bitnet_model_options_ex model_options;
    celiums_bitnet_session_options_ex session_options;
    if (celiums_bitnet_runtime_options_ex_init(&runtime_options, sizeof(runtime_options)) !=
                CELIUMS_BITNET_STATUS_OK ||
            celiums_bitnet_model_options_ex_init(&model_options, sizeof(model_options)) !=
                CELIUMS_BITNET_STATUS_OK ||
            celiums_bitnet_session_options_ex_init(&session_options, sizeof(session_options)) !=
                CELIUMS_BITNET_STATUS_OK) {
        return 1;
    }
    session_options.context_size = 128;
    session_options.batch_size = 64;
    session_options.ubatch_size = 64;
    session_options.threads = 1;
    session_options.threads_batch = 1;

    {
        auto tight_runtime_options = runtime_options;
        tight_runtime_options.ram_budget_bytes = 64;
        celiums_bitnet_runtime * tight_runtime = nullptr;
        celiums_bitnet_model * denied_model = nullptr;
        auto tight_model_options = model_options;
        tight_model_options.use_compute_layout = true;
        if (celiums_bitnet_runtime_create_ex(&tight_runtime_options, &tight_runtime) !=
                CELIUMS_BITNET_STATUS_OK ||
                celiums_bitnet_model_load_family_ex(
                    tight_runtime, argv[1], family, &tight_model_options, &denied_model) !=
                CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED ||
                denied_model != nullptr) {
            fprintf(stderr, "RAM budget fail-closed at model load failed\n");
            return 1;
        }
        celiums_bitnet_runtime_destroy(tight_runtime);
    }

    auto status = celiums_bitnet_runtime_create_ex(&runtime_options, &runtime);
    if (status == CELIUMS_BITNET_STATUS_OK) {
        status = celiums_bitnet_model_load_family_ex(runtime, argv[1], family, &model_options, &model);
    }
    if (status == CELIUMS_BITNET_STATUS_OK) {
        status = celiums_bitnet_session_create_ex(model, &session_options, &session);
    }
    if (status != CELIUMS_BITNET_STATUS_OK || celiums_bitnet_model_get_family(model) != family) return 1;

    {
        celiums_bitnet_session_options legacy = celiums_bitnet_session_default_options();
        legacy.context_size = 128;
        legacy.batch_size = 64;
        legacy.ubatch_size = 64;
        legacy.threads = 1;
        legacy.threads_batch = 1;
        celiums_bitnet_session * legacy_session = nullptr;
        if (celiums_bitnet_session_create(model, &legacy, &legacy_session) !=
                CELIUMS_BITNET_STATUS_OK || legacy_session == nullptr) {
            fprintf(stderr, "legacy session create failed\n");
            return 1;
        }
        celiums_bitnet_session_destroy(legacy_session);
    }

    {
        auto multi = session_options;
        multi.n_seq = 2;
        celiums_bitnet_session * multi_denied = nullptr;
        if (celiums_bitnet_session_create_ex(model, &multi, &multi_denied) !=
                CELIUMS_BITNET_STATUS_INVALID_ARGUMENT || multi_denied != nullptr) {
            fprintf(stderr, "n_seq>1 must fail closed\n");
            return 1;
        }

        auto tight = session_options;
        tight.ram_budget_bytes = 64;
        tight.n_seq = 1;
        tight.context_size = 2048;
        tight.use_compute_layout = true;
        celiums_bitnet_session * denied = nullptr;
        if (celiums_bitnet_session_create_ex(model, &tight, &denied) !=
                CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED || denied != nullptr) {
            fprintf(stderr, "RAM budget fail-closed test failed\n");
            return 1;
        }
    }

    {
        auto kv = session_options;
        kv.context_size = 256;
        kv.n_seq = 1;
        kv.use_compute_layout = false;
        kv.ram_budget_bytes = 8ull * 1024ull * 1024ull;
        const uint64_t needed = celiums_bitnet_estimate_session_ram_bytes_for_model_ex(model, &kv);
        const uint64_t naive = (uint64_t) kv.context_size * 16384ull + 65536ull;
        if (needed <= kv.ram_budget_bytes || needed <= naive) {
            fprintf(stderr, "session KV estimate undercounts needed=%llu budget=%llu naive=%llu\n",
                    (unsigned long long) needed,
                    (unsigned long long) kv.ram_budget_bytes,
                    (unsigned long long) naive);
            return 1;
        }
        celiums_bitnet_session * denied = nullptr;
        if (celiums_bitnet_session_create_ex(model, &kv, &denied) !=
                CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED || denied != nullptr) {
            fprintf(stderr, "RAM budget fail-closed on real KV failed needed=%llu\n",
                    (unsigned long long) needed);
            return 1;
        }
    }

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
            logits_count != (size_t) celiums_bitnet_session_vocab_size(session)) return 1;
    std::vector<float> logits(logits_count);
    if (celiums_bitnet_session_copy_logits(session, logits.data(), &logits_count) != CELIUMS_BITNET_STATUS_OK) return 1;
    for (float value : logits) if (!std::isfinite(value)) return 1;

    celiums_bitnet_token sampled = 0;
    auto greedy = celiums_bitnet_generation_default_options();
    greedy.temperature = 0.0f;
    if (celiums_bitnet_session_sample(session, &greedy, &sampled) != CELIUMS_BITNET_STATUS_OK ||
            celiums_bitnet_session_decode(session, sampled, true) != CELIUMS_BITNET_STATUS_OK) return 1;
    celiums_bitnet_session_reset(session);
    size_t reset_logits = 0;
    if (celiums_bitnet_session_position(session) != 0 ||
            celiums_bitnet_session_copy_logits(session, nullptr, &reset_logits) !=
                CELIUMS_BITNET_STATUS_INVALID_ARGUMENT ||
            celiums_bitnet_session_prefill(session, tokens.data(), token_count, true) != CELIUMS_BITNET_STATUS_OK) return 1;
    std::vector<float> replay_logits(logits_count);
    if (celiums_bitnet_session_copy_logits(session, replay_logits.data(), &logits_count) != CELIUMS_BITNET_STATUS_OK) return 1;
    if (std::memcmp(logits.data(), replay_logits.data(), logits_count * sizeof(float)) != 0) return 1;

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
    if (status != CELIUMS_BITNET_STATUS_OK || generated.empty() ||
            (!bonsai && generated != ", I am a") ||
            result.generated_tokens <= 0 || result.generated_tokens > generation.max_tokens) return 1;

    celiums_bitnet_request_cancel(request);
    if (!celiums_bitnet_request_is_cancelled(request)) return 1;
    celiums_bitnet_request_destroy(request);
    celiums_bitnet_session_destroy(session);
    celiums_bitnet_model_destroy(model);
    celiums_bitnet_runtime_destroy(runtime);
    return 0;
}
