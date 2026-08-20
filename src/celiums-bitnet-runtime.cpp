#include "celiums/bitnet_runtime.h"

#include "llama.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

struct celiums_bitnet_runtime {
    std::atomic<uint32_t> references { 1 };
    std::atomic<bool> active { true };
};

struct celiums_bitnet_model {
    celiums_bitnet_runtime * runtime;
    llama_model * handle;
    std::atomic<uint32_t> references { 1 };
};

struct celiums_bitnet_session {
    celiums_bitnet_model * model;
    llama_context * context;
    std::mutex mutex;
    std::atomic<uint32_t> references { 1 };
    std::atomic<bool> active { true };
    llama_sampler * sampler = nullptr;
    float sampler_temperature = -1.0f;
    int32_t sampler_top_k = 0;
    float sampler_top_p = 0.0f;
    uint32_t sampler_seed = 0;
    int32_t position = 0;
    int32_t last_logits_index = -1;
};

struct celiums_bitnet_request {
    celiums_bitnet_session * session;
    std::atomic<bool> cancelled { false };
    std::atomic<bool> running { false };
};

namespace {

std::mutex backend_mutex;
uint32_t backend_users = 0;

bool valid_header(size_t struct_size, uint32_t api_version, size_t expected_size) {
    return struct_size >= expected_size && api_version == CELIUMS_BITNET_API_VERSION;
}

void release_runtime(celiums_bitnet_runtime * runtime) {
    if (runtime->references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(backend_mutex);
        if (backend_users > 0 && --backend_users == 0) {
            llama_backend_free();
        }
    }
    delete runtime;
}

void release_model(celiums_bitnet_model * model) {
    if (model->references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    llama_model_free(model->handle);
    release_runtime(model->runtime);
    delete model;
}

void release_session(celiums_bitnet_session * session) {
    if (session->references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    llama_sampler_free(session->sampler);
    llama_free(session->context);
    release_model(session->model);
    delete session;
}

celiums_bitnet_status decode_tokens(
        celiums_bitnet_session * session,
        const celiums_bitnet_token * tokens,
        size_t token_count,
        bool output_logits) {
    if (!session || !tokens || token_count == 0 || token_count > (size_t) std::numeric_limits<int32_t>::max()) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    if ((size_t) session->position + token_count > llama_n_ctx(session->context)) {
        return CELIUMS_BITNET_STATUS_CONTEXT_FULL;
    }
    const uint32_t batch_capacity = llama_n_batch(session->context);
    size_t offset = 0;
    while (offset < token_count) {
        const int32_t count = (int32_t) std::min<size_t>(batch_capacity, token_count - offset);
        llama_batch batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int32_t index = 0; index < count; ++index) {
            batch.token[index] = tokens[offset + index];
            batch.pos[index] = session->position + index;
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = output_logits && offset + index + 1 == token_count;
        }
        const int32_t decode_status = llama_decode(session->context, batch);
        llama_batch_free(batch);
        if (decode_status == 2) {
            return CELIUMS_BITNET_STATUS_CANCELLED;
        }
        if (decode_status != 0) {
            return CELIUMS_BITNET_STATUS_DECODE_FAILED;
        }
        session->position += count;
        offset += count;
    }
    session->last_logits_index = output_logits ? -1 : -2;
    return CELIUMS_BITNET_STATUS_OK;
}

std::vector<llama_token> tokenize_text(
        const celiums_bitnet_model * model,
        const char * text,
        bool add_special,
        bool parse_special,
        celiums_bitnet_status & status) {
    status = CELIUMS_BITNET_STATUS_OK;
    if (!model || !text) {
        status = CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
        return {};
    }
    const llama_vocab * vocab = llama_model_get_vocab(model->handle);
    const size_t text_size = std::strlen(text);
    if (text_size > (size_t) std::numeric_limits<int32_t>::max()) {
        status = CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
        return {};
    }
    int32_t count = llama_tokenize(vocab, text, (int32_t) text_size, nullptr, 0, add_special, parse_special);
    if (count == INT32_MIN) {
        status = CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
        return {};
    }
    count = count < 0 ? -count : count;
    std::vector<llama_token> tokens((size_t) count);
    if (count > 0) {
        const int32_t written = llama_tokenize(
            vocab, text, (int32_t) text_size, tokens.data(), count, add_special, parse_special);
        if (written < 0) {
            status = CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
            return {};
        }
        tokens.resize((size_t) written);
    }
    return tokens;
}

std::string token_piece(const celiums_bitnet_model * model, llama_token token) {
    const llama_vocab * vocab = llama_model_get_vocab(model->handle);
    char stack_buffer[256];
    int32_t size = llama_token_to_piece(vocab, token, stack_buffer, sizeof(stack_buffer), 0, true);
    if (size >= 0) {
        return std::string(stack_buffer, (size_t) size);
    }
    std::vector<char> buffer((size_t) -size);
    size = llama_token_to_piece(vocab, token, buffer.data(), (int32_t) buffer.size(), 0, true);
    return size < 0 ? std::string() : std::string(buffer.data(), (size_t) size);
}

bool ends_with_stop(const std::string & text, const celiums_bitnet_generation_options & options) {
    for (size_t index = 0; index < options.stop_sequence_count; ++index) {
        const char * stop = options.stop_sequences[index];
        if (!stop || stop[0] == '\0') {
            continue;
        }
        const size_t size = std::strlen(stop);
        if (text.size() >= size && text.compare(text.size() - size, size, stop) == 0) {
            return true;
        }
    }
    return false;
}

bool request_abort(void * user_data) {
    return static_cast<celiums_bitnet_request *>(user_data)->cancelled.load(std::memory_order_acquire);
}

llama_sampler * create_sampler(const celiums_bitnet_generation_options & options) {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) {
        return nullptr;
    }
    if (options.temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(options.top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(options.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed));
    }
    return sampler;
}

bool valid_generation_options(const celiums_bitnet_generation_options & options) {
    return options.max_tokens >= 0 && std::isfinite(options.temperature) && options.temperature >= 0.0f &&
        std::isfinite(options.top_p) && options.top_p > 0.0f && options.top_p <= 1.0f &&
        (options.stop_sequence_count == 0 || options.stop_sequences != nullptr);
}

} // namespace

const char * celiums_bitnet_version(void) {
    return CELIUMS_BITNET_RUNTIME_VERSION;
}

const char * celiums_bitnet_product_commit(void) {
    return CELIUMS_BITNET_PRODUCT_COMMIT;
}

const char * celiums_bitnet_engine_commit(void) {
    return CELIUMS_BITNET_ENGINE_COMMIT;
}

const char * celiums_bitnet_cpu_profile(void) {
    return CELIUMS_BITNET_RUNTIME_PROFILE;
}

const char * celiums_bitnet_status_string(celiums_bitnet_status status) {
    switch (status) {
        case CELIUMS_BITNET_STATUS_OK:                return "ok";
        case CELIUMS_BITNET_STATUS_INVALID_ARGUMENT:  return "invalid argument";
        case CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED: return "model load failed";
        case CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL:  return "buffer too small";
        case CELIUMS_BITNET_STATUS_INTERNAL_ERROR:    return "internal error";
        case CELIUMS_BITNET_STATUS_UNSUPPORTED_MODEL: return "unsupported strict model";
        case CELIUMS_BITNET_STATUS_CONTEXT_CREATE_FAILED: return "context create failed";
        case CELIUMS_BITNET_STATUS_DECODE_FAILED:          return "decode failed";
        case CELIUMS_BITNET_STATUS_CANCELLED:              return "cancelled";
        case CELIUMS_BITNET_STATUS_CONTEXT_FULL:           return "context full";
        case CELIUMS_BITNET_STATUS_CALLBACK_ABORTED:       return "stream callback aborted";
    }
    return "unknown status";
}

celiums_bitnet_runtime_options celiums_bitnet_runtime_default_options(void) {
    return { sizeof(celiums_bitnet_runtime_options), CELIUMS_BITNET_API_VERSION };
}

celiums_bitnet_model_options celiums_bitnet_model_default_options(void) {
    return {
        sizeof(celiums_bitnet_model_options), CELIUMS_BITNET_API_VERSION,
        true, false, false,
    };
}

celiums_bitnet_session_options celiums_bitnet_session_default_options(void) {
    return {
        sizeof(celiums_bitnet_session_options), CELIUMS_BITNET_API_VERSION,
        2048, 512, 512, 2, 2,
    };
}

celiums_bitnet_generation_options celiums_bitnet_generation_default_options(void) {
    return {
        sizeof(celiums_bitnet_generation_options), CELIUMS_BITNET_API_VERSION,
        128, 0.8f, 40, 0.95f, UINT32_MAX, nullptr, 0,
    };
}

celiums_bitnet_status celiums_bitnet_runtime_create(
        const celiums_bitnet_runtime_options * options,
        celiums_bitnet_runtime ** runtime) {
    if (!runtime || (options && !valid_header(
            options->struct_size, options->api_version, sizeof(celiums_bitnet_runtime_options)))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    *runtime = nullptr;
    auto * created = new (std::nothrow) celiums_bitnet_runtime;
    if (!created) {
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    {
        std::lock_guard<std::mutex> lock(backend_mutex);
        if (backend_users++ == 0) {
            llama_backend_init();
        }
    }
    *runtime = created;
    return CELIUMS_BITNET_STATUS_OK;
}

void celiums_bitnet_runtime_destroy(celiums_bitnet_runtime * runtime) {
    if (!runtime) {
        return;
    }
    runtime->active.store(false, std::memory_order_release);
    release_runtime(runtime);
}

celiums_bitnet_status celiums_bitnet_model_load(
        celiums_bitnet_runtime * runtime,
        const char * path,
        const celiums_bitnet_model_options * options,
        celiums_bitnet_model ** model) {
    if (!runtime || !runtime->active.load(std::memory_order_acquire) || !path || path[0] == '\0' || !model ||
            (options && !valid_header(
                options->struct_size, options->api_version, sizeof(celiums_bitnet_model_options)))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    *model = nullptr;
    const celiums_bitnet_model_options resolved = options ? *options : celiums_bitnet_model_default_options();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = resolved.use_mmap;
    params.use_mlock = resolved.use_mlock;
    params.check_tensors = resolved.check_tensors;
    llama_model * handle = llama_model_load_from_file(path, params);
    if (!handle) {
        return CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED;
    }
    char architecture[64];
    char file_type[32];
    if (llama_model_meta_val_str(handle, "general.architecture", architecture, sizeof(architecture)) < 0 ||
            llama_model_meta_val_str(handle, "general.file_type", file_type, sizeof(file_type)) < 0 ||
            std::strcmp(architecture, "bitnet-b1.58") != 0 || std::strcmp(file_type, "41") != 0) {
        llama_model_free(handle);
        return CELIUMS_BITNET_STATUS_UNSUPPORTED_MODEL;
    }
    auto * loaded = new (std::nothrow) celiums_bitnet_model { runtime, handle };
    if (!loaded) {
        llama_model_free(handle);
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    runtime->references.fetch_add(1, std::memory_order_relaxed);
    *model = loaded;
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_model_validate_strict(
        celiums_bitnet_runtime * runtime,
        const char * path,
        celiums_bitnet_model_info * info) {
    if (!info || !valid_header(info->struct_size, info->api_version, sizeof(celiums_bitnet_model_info))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    celiums_bitnet_model_options options = celiums_bitnet_model_default_options();
    options.check_tensors = true;
    celiums_bitnet_model * model = nullptr;
    const celiums_bitnet_status status = celiums_bitnet_model_load(runtime, path, &options, &model);
    if (status != CELIUMS_BITNET_STATUS_OK) {
        return status;
    }
    const celiums_bitnet_status info_status = celiums_bitnet_model_get_info(model, info);
    celiums_bitnet_model_destroy(model);
    return info_status;
}

void celiums_bitnet_model_destroy(celiums_bitnet_model * model) {
    if (!model) {
        return;
    }
    release_model(model);
}

celiums_bitnet_status celiums_bitnet_model_get_info(
        const celiums_bitnet_model * model,
        celiums_bitnet_model_info * info) {
    if (!model || !info || !valid_header(
            info->struct_size, info->api_version, sizeof(celiums_bitnet_model_info))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    info->size_bytes = llama_model_size(model->handle);
    info->parameter_count = llama_model_n_params(model->handle);
    info->context_length = llama_model_n_ctx_train(model->handle);
    info->embedding_length = llama_model_n_embd(model->handle);
    info->layer_count = llama_model_n_layer(model->handle);
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_model_get_description(
        const celiums_bitnet_model * model,
        char * buffer,
        size_t * buffer_size) {
    if (!model || !buffer_size) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    char description[512];
    const int32_t length = llama_model_desc(model->handle, description, sizeof(description));
    if (length < 0 || (size_t) length >= sizeof(description)) {
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    const size_t required = (size_t) length + 1;
    if (!buffer || *buffer_size < required) {
        *buffer_size = required;
        return CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(buffer, description, required);
    *buffer_size = required;
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_session_create(
        celiums_bitnet_model * model,
        const celiums_bitnet_session_options * options,
        celiums_bitnet_session ** session) {
    if (!model || !session || (options && !valid_header(
            options->struct_size, options->api_version, sizeof(celiums_bitnet_session_options)))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    *session = nullptr;
    const celiums_bitnet_session_options resolved = options ? *options : celiums_bitnet_session_default_options();
    if (resolved.context_size == 0 || resolved.batch_size == 0 || resolved.ubatch_size == 0 ||
            resolved.threads <= 0 || resolved.threads_batch <= 0) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    llama_context_params params = llama_context_default_params();
    params.n_ctx = resolved.context_size;
    params.n_batch = resolved.batch_size;
    params.n_ubatch = resolved.ubatch_size;
    params.n_threads = resolved.threads;
    params.n_threads_batch = resolved.threads_batch;
    params.no_perf = false;
    llama_context * context = llama_init_from_model(model->handle, params);
    if (!context) {
        return CELIUMS_BITNET_STATUS_CONTEXT_CREATE_FAILED;
    }
    auto * created = new (std::nothrow) celiums_bitnet_session { model, context };
    if (!created) {
        llama_free(context);
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    model->references.fetch_add(1, std::memory_order_relaxed);
    *session = created;
    return CELIUMS_BITNET_STATUS_OK;
}

void celiums_bitnet_session_destroy(celiums_bitnet_session * session) {
    if (!session) {
        return;
    }
    session->active.store(false, std::memory_order_release);
    release_session(session);
}

void celiums_bitnet_session_reset(celiums_bitnet_session * session) {
    if (!session) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    llama_memory_clear(llama_get_memory(session->context), true);
    llama_sampler_reset(session->sampler);
    session->position = 0;
    session->last_logits_index = -1;
}

int32_t celiums_bitnet_session_context_size(const celiums_bitnet_session * session) {
    return session ? (int32_t) llama_n_ctx(session->context) : 0;
}

int32_t celiums_bitnet_session_vocab_size(const celiums_bitnet_session * session) {
    return session ? llama_vocab_n_tokens(llama_model_get_vocab(session->model->handle)) : 0;
}

int32_t celiums_bitnet_session_position(const celiums_bitnet_session * session) {
    return session ? session->position : 0;
}

bool celiums_bitnet_model_token_is_eog(const celiums_bitnet_model * model, celiums_bitnet_token token) {
    return model && llama_vocab_is_eog(llama_model_get_vocab(model->handle), token);
}

celiums_bitnet_status celiums_bitnet_model_token_to_piece(
        const celiums_bitnet_model * model,
        celiums_bitnet_token token,
        char * piece,
        size_t * piece_size) {
    if (!model || !piece_size) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    const std::string result = token_piece(model, token);
    const size_t required = result.size() + 1;
    if (!piece || *piece_size < required) {
        *piece_size = required;
        return CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(piece, result.data(), result.size());
    piece[result.size()] = '\0';
    *piece_size = required;
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_tokenize(
        const celiums_bitnet_model * model,
        const char * text,
        bool add_special,
        bool parse_special,
        celiums_bitnet_token * tokens,
        size_t * token_count) {
    if (!token_count) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    celiums_bitnet_status status;
    const std::vector<llama_token> result = tokenize_text(model, text, add_special, parse_special, status);
    if (status != CELIUMS_BITNET_STATUS_OK) {
        return status;
    }
    if (!tokens || *token_count < result.size()) {
        *token_count = result.size();
        return CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL;
    }
    std::copy(result.begin(), result.end(), tokens);
    *token_count = result.size();
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_detokenize(
        const celiums_bitnet_model * model,
        const celiums_bitnet_token * tokens,
        size_t token_count,
        bool remove_special,
        bool unparse_special,
        char * text,
        size_t * text_size) {
    if (!model || (!tokens && token_count != 0) || !text_size ||
            token_count > (size_t) std::numeric_limits<int32_t>::max()) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model->handle);
    int32_t required = llama_detokenize(
        vocab, tokens, (int32_t) token_count, nullptr, 0, remove_special, unparse_special);
    if (required == INT32_MIN) {
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    required = required < 0 ? -required : required;
    const size_t required_size = (size_t) required + 1;
    if (!text || *text_size < required_size) {
        *text_size = required_size;
        return CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL;
    }
    const int32_t written = llama_detokenize(
        vocab, tokens, (int32_t) token_count, text, required, remove_special, unparse_special);
    if (written < 0) {
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    text[written] = '\0';
    *text_size = (size_t) written + 1;
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_session_prefill(
        celiums_bitnet_session * session,
        const celiums_bitnet_token * tokens,
        size_t token_count,
        bool output_logits) {
    if (!session) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    return decode_tokens(session, tokens, token_count, output_logits);
}

celiums_bitnet_status celiums_bitnet_session_decode(
        celiums_bitnet_session * session,
        celiums_bitnet_token token,
        bool output_logits) {
    return celiums_bitnet_session_prefill(session, &token, 1, output_logits);
}

celiums_bitnet_status celiums_bitnet_session_copy_logits(
        celiums_bitnet_session * session,
        float * logits,
        size_t * logits_count) {
    if (!session || !logits_count) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->last_logits_index == -2) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    const size_t count = (size_t) llama_vocab_n_tokens(llama_model_get_vocab(session->model->handle));
    if (!logits || *logits_count < count) {
        *logits_count = count;
        return CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL;
    }
    const float * source = llama_get_logits_ith(session->context, session->last_logits_index);
    if (!source) {
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(logits, source, count * sizeof(float));
    *logits_count = count;
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_session_sample(
        celiums_bitnet_session * session,
        const celiums_bitnet_generation_options * options,
        celiums_bitnet_token * token) {
    if (!session || !token || (options && !valid_header(
            options->struct_size, options->api_version, sizeof(celiums_bitnet_generation_options)))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    const celiums_bitnet_generation_options resolved = options ? *options : celiums_bitnet_generation_default_options();
    if (!valid_generation_options(resolved)) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->last_logits_index == -2) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    if (!session->sampler || session->sampler_temperature != resolved.temperature ||
            session->sampler_top_k != resolved.top_k || session->sampler_top_p != resolved.top_p ||
            session->sampler_seed != resolved.seed) {
        llama_sampler_free(session->sampler);
        session->sampler = create_sampler(resolved);
        session->sampler_temperature = resolved.temperature;
        session->sampler_top_k = resolved.top_k;
        session->sampler_top_p = resolved.top_p;
        session->sampler_seed = resolved.seed;
        if (!session->sampler) {
            return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
        }
    }
    *token = llama_sampler_sample(session->sampler, session->context, session->last_logits_index);
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_request_create(
        celiums_bitnet_session * session,
        celiums_bitnet_request ** request) {
    if (!session || !session->active.load(std::memory_order_acquire) || !request) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    *request = new (std::nothrow) celiums_bitnet_request { session };
    if (*request) {
        session->references.fetch_add(1, std::memory_order_relaxed);
    }
    return *request ? CELIUMS_BITNET_STATUS_OK : CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
}

void celiums_bitnet_request_cancel(celiums_bitnet_request * request) {
    if (request) {
        request->cancelled.store(true, std::memory_order_release);
    }
}

bool celiums_bitnet_request_is_cancelled(const celiums_bitnet_request * request) {
    return request && request->cancelled.load(std::memory_order_acquire);
}

void celiums_bitnet_request_destroy(celiums_bitnet_request * request) {
    if (!request || request->running.load(std::memory_order_acquire)) {
        return;
    }
    release_session(request->session);
    delete request;
}

celiums_bitnet_status celiums_bitnet_generate(
        celiums_bitnet_request * request,
        const char * prompt,
        const celiums_bitnet_generation_options * options,
        celiums_bitnet_stream_callback callback,
        void * user_data,
        celiums_bitnet_generation_result * result) {
    if (!request || !request->session->active.load(std::memory_order_acquire) || !prompt || !result || !valid_header(
            result->struct_size, result->api_version, sizeof(celiums_bitnet_generation_result)) ||
            (options && !valid_header(
                options->struct_size, options->api_version, sizeof(celiums_bitnet_generation_options)))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    bool expected = false;
    if (!request->running.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    struct running_guard {
        celiums_bitnet_request * request;
        ~running_guard() { request->running.store(false, std::memory_order_release); }
    } guard { request };

    const celiums_bitnet_generation_options resolved = options ? *options : celiums_bitnet_generation_default_options();
    if (!valid_generation_options(resolved)) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }

    result->generated_tokens = 0;
    result->stopped_by_eog = false;
    result->stopped_by_sequence = false;
    result->cancelled = false;
    request->cancelled.store(false, std::memory_order_release);

    celiums_bitnet_status status;
    std::vector<llama_token> tokens = tokenize_text(request->session->model, prompt, true, false, status);
    if (status != CELIUMS_BITNET_STATUS_OK || tokens.empty()) {
        return status == CELIUMS_BITNET_STATUS_OK ? CELIUMS_BITNET_STATUS_INVALID_ARGUMENT : status;
    }

    std::lock_guard<std::mutex> lock(request->session->mutex);
    llama_set_abort_callback(request->session->context, request_abort, request);
    struct abort_guard {
        llama_context * context;
        ~abort_guard() { llama_set_abort_callback(context, nullptr, nullptr); }
    } abort { request->session->context };
    llama_memory_clear(llama_get_memory(request->session->context), true);
    request->session->position = 0;
    request->session->last_logits_index = -1;
    status = decode_tokens(request->session, tokens.data(), tokens.size(), true);
    if (status != CELIUMS_BITNET_STATUS_OK) {
        return status;
    }

    llama_sampler * sampler = create_sampler(resolved);
    if (!sampler) {
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }

    std::string generated_text;
    const llama_vocab * vocab = llama_model_get_vocab(request->session->model->handle);
    for (int32_t index = 0; index < resolved.max_tokens; ++index) {
        if (request->cancelled.load(std::memory_order_acquire)) {
            result->cancelled = true;
            llama_sampler_free(sampler);
            return CELIUMS_BITNET_STATUS_CANCELLED;
        }
        const llama_token token = llama_sampler_sample(sampler, request->session->context, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            result->stopped_by_eog = true;
            break;
        }
        const std::string piece = token_piece(request->session->model, token);
        generated_text += piece;
        ++result->generated_tokens;
        if (callback && !callback(token, piece.data(), piece.size(), user_data)) {
            llama_sampler_free(sampler);
            return CELIUMS_BITNET_STATUS_CALLBACK_ABORTED;
        }
        if (ends_with_stop(generated_text, resolved)) {
            result->stopped_by_sequence = true;
            break;
        }
        status = decode_tokens(request->session, &token, 1, true);
        if (status != CELIUMS_BITNET_STATUS_OK) {
            llama_sampler_free(sampler);
            return status;
        }
    }
    llama_sampler_free(sampler);
    return CELIUMS_BITNET_STATUS_OK;
}
