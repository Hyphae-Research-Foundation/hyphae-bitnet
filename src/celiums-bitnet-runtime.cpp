#include "celiums/bitnet_runtime.h"

#include "llama.h"

#include "chat.h"

extern "C" void ggml_cpu_set_repack_enabled(int enabled);

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

struct celiums_bitnet_runtime {
    std::atomic<uint32_t> references { 1 };
    std::atomic<bool> active { true };
    uint64_t ram_budget_bytes = 0;
    std::atomic<uint64_t> reserved_bytes { 0 };
};

struct celiums_bitnet_model {
    celiums_bitnet_runtime * runtime;
    llama_model * handle;
    celiums_bitnet_model_family family;
    uint64_t reserved_bytes = 0;
    std::atomic<uint32_t> references { 1 };
    mutable std::mutex chat_mutex;
    mutable common_chat_templates_ptr chat_templates;
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
    int32_t last_logits_index = -2;
    uint64_t reserved_bytes = 0;
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

bool valid_family(celiums_bitnet_model_family family) {
    return family == CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S ||
        family == CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0;
}

bool matches_family(
        celiums_bitnet_model_family family,
        const char * architecture,
        const char * file_type) {
    switch (family) {
        case CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S:
            return std::strcmp(architecture, "bitnet-b1.58") == 0 && std::strcmp(file_type, "41") == 0;
        case CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0:
            return std::strcmp(architecture, "qwen35") == 0 && std::strcmp(file_type, "40") == 0;
        case CELIUMS_BITNET_MODEL_FAMILY_UNKNOWN:
            return false;
    }
    return false;
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

bool u64_add(uint64_t a, uint64_t b, uint64_t * out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return false;
    }
    *out = a + b;
    return true;
}

bool u64_mul(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

bool ram_try_reserve(celiums_bitnet_runtime * runtime, uint64_t bytes) {
    if (!runtime) {
        return false;
    }
    if (bytes == 0) {
        return true;
    }
    uint64_t current = runtime->reserved_bytes.load(std::memory_order_relaxed);
    while (true) {
        if (bytes > runtime->ram_budget_bytes || current > runtime->ram_budget_bytes - bytes) {
            return false;
        }
        if (runtime->reserved_bytes.compare_exchange_weak(
                current, current + bytes, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return true;
        }
    }
}

void ram_release(celiums_bitnet_runtime * runtime, uint64_t bytes) {
    if (!runtime || bytes == 0) {
        return;
    }
    runtime->reserved_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
}

void release_model(celiums_bitnet_model * model) {
    if (model->references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    ram_release(model->runtime, model->reserved_bytes);
    llama_model_free(model->handle);
    release_runtime(model->runtime);
    delete model;
}

void release_session(celiums_bitnet_session * session) {
    if (session->references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    ram_release(session->model->runtime, session->reserved_bytes);
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

const char * celiums_bitnet_engine_tree(void) {
    return CELIUMS_BITNET_ENGINE_TREE;
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
        case CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED:    return "ram budget exceeded";
    }
    return "unknown status";
}

const char * celiums_bitnet_model_family_string(celiums_bitnet_model_family family) {
    switch (family) {
        case CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S: return "bitnet-b1.58-i2_s";
        case CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0: return "bonsai-qwen35-q1_0";
        case CELIUMS_BITNET_MODEL_FAMILY_UNKNOWN: return "unknown";
    }
    return "unknown";
}

celiums_bitnet_runtime_options celiums_bitnet_runtime_default_options(void) {
    return { sizeof(celiums_bitnet_runtime_options), CELIUMS_BITNET_API_VERSION, 0 };
}

celiums_bitnet_model_options celiums_bitnet_model_default_options(void) {
    return {
        sizeof(celiums_bitnet_model_options), CELIUMS_BITNET_API_VERSION,
        true, false, false, true,
    };
}

celiums_bitnet_session_options celiums_bitnet_session_default_options(void) {
    return {
        sizeof(celiums_bitnet_session_options), CELIUMS_BITNET_API_VERSION,
        2048, 512, 512, 2, 2,
        1, 0, true,
    };
}

uint64_t celiums_bitnet_host_ram_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return 0;
    }
    return (uint64_t) status.ullTotalPhys;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) {
        return 0;
    }
    return (uint64_t) pages * (uint64_t) page;
#endif
}

namespace {

uint64_t packed_file_bytes(const char * path) {
    if (!path || path[0] == '\0') {
        return 0;
    }
#if defined(_WIN32)
    WIN32_FILE_ATTRIBUTE_DATA info;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &info)) {
        return 0;
    }
    ULARGE_INTEGER size;
    size.LowPart = info.nFileSizeLow;
    size.HighPart = info.nFileSizeHigh;
    return (uint64_t) size.QuadPart;
#else
    struct stat info;
    if (stat(path, &info) != 0 || info.st_size < 0) {
        return 0;
    }
    return (uint64_t) info.st_size;
#endif
}

uint64_t compute_layout_bytes(bool use_compute_layout, uint64_t packed_model_bytes) {
    if (!use_compute_layout || packed_model_bytes == 0) {
        return 0;
    }
#if defined(__aarch64__) || defined(__arm__)
    uint64_t expanded = 0;
    if (!u64_mul(packed_model_bytes, 8ull, &expanded)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return expanded;
#else
    return packed_model_bytes;
#endif
}

uint32_t model_meta_u32(const llama_model * model, const char * key) {
    char buffer[64];
    if (!model || !key || llama_model_meta_val_str(model, key, buffer, sizeof(buffer)) < 0) {
        return 0;
    }
    char * end = nullptr;
    const unsigned long value = std::strtoul(buffer, &end, 10);
    if (end == buffer) {
        return 0;
    }
    return (uint32_t) value;
}

/* Same K/V tensor geometry llama_kv_cache allocates: F16 GQA rows. */
uint64_t kv_cache_bytes(const llama_model * model, uint32_t n_ctx, uint32_t n_seq) {
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_head = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    if (n_embd <= 0 || n_layer <= 0 || n_head <= 0 || n_head_kv <= 0 || n_ctx == 0 || n_seq == 0) {
        return (uint64_t) n_seq * (uint64_t) n_ctx * 131072ull;
    }
    const uint64_t n_embd_head = (uint64_t) n_embd / (uint64_t) n_head;
    const uint64_t kv_per_cell = n_embd_head * (uint64_t) n_head_kv * 2ull * 2ull;
    return (uint64_t) n_seq * (uint64_t) n_ctx * (uint64_t) n_layer * kv_per_cell;
}

uint64_t recurrent_state_bytes(const llama_model * model, uint32_t n_seq) {
    char architecture[64];
    if (llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) < 0) {
        return 0;
    }
    char inner_key[96];
    char state_key[96];
    if (std::snprintf(inner_key, sizeof(inner_key), "%s.ssm.inner_size", architecture) < 0 ||
            std::snprintf(state_key, sizeof(state_key), "%s.ssm.state_size", architecture) < 0) {
        return 0;
    }
    const uint32_t inner = model_meta_u32(model, inner_key);
    const uint32_t state = model_meta_u32(model, state_key);
    const int32_t n_layer = llama_model_n_layer(model);
    if (inner == 0 || state == 0 || n_layer <= 0 || n_seq == 0) {
        return 0;
    }
    return (uint64_t) n_seq * (uint64_t) n_layer * (uint64_t) inner * (uint64_t) state * 4ull;
}

uint64_t compute_scratch_bytes(const llama_model * model, uint32_t ubatch) {
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_layer = llama_model_n_layer(model);
    if (n_embd <= 0 || n_layer <= 0 || ubatch == 0) {
        return 0;
    }
    return (uint64_t) ubatch * (uint64_t) n_embd * (uint64_t) n_layer * 4ull;
}

} // namespace

uint64_t celiums_bitnet_default_ram_budget_bytes(void) {
    const uint64_t host = celiums_bitnet_host_ram_bytes();
    if (host == 0) {
        return 256ull * 1024ull * 1024ull;
    }
    const uint64_t half = host / 2;
    const uint64_t ten_percent = host / 10;
    const uint64_t reserve = std::max<uint64_t>(4ull * 1024ull * 1024ull * 1024ull, ten_percent);
    if (host <= reserve) {
        return host / 2;
    }
    const uint64_t capped = host - reserve;
    const uint64_t ninety = (host / 10) * 9;
    return std::min(half, std::min(capped, ninety));
}

uint64_t celiums_bitnet_estimate_session_ram_bytes(
        const celiums_bitnet_session_options * options,
        uint64_t packed_model_bytes) {
    if (options && !valid_header(
            options->struct_size, options->api_version, sizeof(celiums_bitnet_session_options))) {
        return 0;
    }
    const celiums_bitnet_session_options resolved =
        options ? *options : celiums_bitnet_session_default_options();
    const uint32_t n_seq = resolved.n_seq == 0 ? 1u : resolved.n_seq;
    uint64_t per_seq = 0;
    uint64_t seq_bytes = 0;
    uint64_t total = 0;
    if (!u64_mul((uint64_t) resolved.context_size, 131072ull, &per_seq) ||
            !u64_add(per_seq, 1048576ull, &per_seq) ||
            !u64_mul((uint64_t) n_seq, per_seq, &seq_bytes) ||
            !u64_add(seq_bytes, compute_layout_bytes(resolved.use_compute_layout, packed_model_bytes), &total)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return total;
}

uint64_t session_incremental_bytes(
        const llama_model * handle,
        const celiums_bitnet_session_options & resolved) {
    const uint32_t n_seq = resolved.n_seq == 0 ? 1u : resolved.n_seq;
    uint64_t total = 0;
    if (!u64_add(total, kv_cache_bytes(handle, resolved.context_size, n_seq), &total) ||
            !u64_add(total, recurrent_state_bytes(handle, n_seq), &total) ||
            !u64_add(total, compute_scratch_bytes(handle, resolved.ubatch_size), &total)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return total;
}

uint64_t celiums_bitnet_estimate_session_ram_bytes_for_model(
        const celiums_bitnet_model * model,
        const celiums_bitnet_session_options * options) {
    if (options && !valid_header(
            options->struct_size, options->api_version, sizeof(celiums_bitnet_session_options))) {
        return 0;
    }
    const celiums_bitnet_session_options resolved =
        options ? *options : celiums_bitnet_session_default_options();
    if (!model || !model->handle) {
        return celiums_bitnet_estimate_session_ram_bytes(options, 0);
    }
    const uint64_t packed = llama_model_size(model->handle);
    uint64_t total = 0;
    if (!u64_add(session_incremental_bytes(model->handle, resolved),
            compute_layout_bytes(resolved.use_compute_layout, packed), &total)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return total;
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
    created->ram_budget_bytes = (options && options->ram_budget_bytes)
        ? options->ram_budget_bytes
        : celiums_bitnet_default_ram_budget_bytes();
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
    return celiums_bitnet_model_load_family(
        runtime, path, CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S, options, model);
}

celiums_bitnet_status celiums_bitnet_model_load_family(
        celiums_bitnet_runtime * runtime,
        const char * path,
        celiums_bitnet_model_family family,
        const celiums_bitnet_model_options * options,
        celiums_bitnet_model ** model) {
    if (!model) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    *model = nullptr;
    if (!runtime || !runtime->active.load(std::memory_order_acquire) || !path || path[0] == '\0' ||
            !valid_family(family) ||
            (options && !valid_header(
                options->struct_size, options->api_version, sizeof(celiums_bitnet_model_options)))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    const celiums_bitnet_model_options resolved = options ? *options : celiums_bitnet_model_default_options();
    const uint64_t packed = packed_file_bytes(path);
    if (packed == 0) {
        return CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED;
    }
    uint64_t needed = 0;
    if (!u64_add(packed, compute_layout_bytes(resolved.use_compute_layout, packed), &needed) ||
            !ram_try_reserve(runtime, needed)) {
        return CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED;
    }
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = resolved.use_mmap;
    params.use_mlock = resolved.use_mlock;
    params.check_tensors = resolved.check_tensors;
    llama_model * handle = nullptr;
    {
        std::lock_guard<std::mutex> lock(backend_mutex);
        ggml_cpu_set_repack_enabled(resolved.use_compute_layout ? 1 : 0);
        handle = llama_model_load_from_file(path, params);
    }
    if (!handle) {
        ram_release(runtime, needed);
        return CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED;
    }
    char architecture[64];
    char file_type[32];
    const int32_t architecture_size = llama_model_meta_val_str(
        handle, "general.architecture", architecture, sizeof(architecture));
    const int32_t file_type_size = llama_model_meta_val_str(
        handle, "general.file_type", file_type, sizeof(file_type));
    if (architecture_size < 0 || (size_t) architecture_size >= sizeof(architecture) ||
            file_type_size < 0 || (size_t) file_type_size >= sizeof(file_type) ||
            !matches_family(family, architecture, file_type)) {
        llama_model_free(handle);
        ram_release(runtime, needed);
        return CELIUMS_BITNET_STATUS_UNSUPPORTED_MODEL;
    }
    auto * loaded = new (std::nothrow) celiums_bitnet_model { runtime, handle, family };
    if (!loaded) {
        llama_model_free(handle);
        ram_release(runtime, needed);
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    loaded->reserved_bytes = needed;
    runtime->references.fetch_add(1, std::memory_order_relaxed);
    *model = loaded;
    return CELIUMS_BITNET_STATUS_OK;
}

celiums_bitnet_status celiums_bitnet_model_validate_strict(
        celiums_bitnet_runtime * runtime,
        const char * path,
        celiums_bitnet_model_info * info) {
    return celiums_bitnet_model_validate_family(
        runtime, path, CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S, info);
}

celiums_bitnet_status celiums_bitnet_model_validate_family(
        celiums_bitnet_runtime * runtime,
        const char * path,
        celiums_bitnet_model_family family,
        celiums_bitnet_model_info * info) {
    if (!info || !valid_header(info->struct_size, info->api_version, sizeof(celiums_bitnet_model_info))) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    celiums_bitnet_model_options options = celiums_bitnet_model_default_options();
    options.check_tensors = true;
    celiums_bitnet_model * model = nullptr;
    const celiums_bitnet_status status = celiums_bitnet_model_load_family(
        runtime, path, family, &options, &model);
    if (status != CELIUMS_BITNET_STATUS_OK) {
        return status;
    }
    const celiums_bitnet_status info_status = celiums_bitnet_model_get_info(model, info);
    celiums_bitnet_model_destroy(model);
    return info_status;
}

celiums_bitnet_model_family celiums_bitnet_model_get_family(const celiums_bitnet_model * model) {
    return model ? model->family : CELIUMS_BITNET_MODEL_FAMILY_UNKNOWN;
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

celiums_bitnet_status celiums_bitnet_model_apply_chat_template(
        const celiums_bitnet_model * model,
        const celiums_bitnet_chat_message * messages,
        size_t message_count,
        bool add_assistant,
        char * text,
        size_t * text_size) {
    if (!model || !messages || message_count == 0 || !text_size ||
            message_count > (size_t) std::numeric_limits<int32_t>::max()) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < message_count; ++index) {
        if (!messages[index].role || !messages[index].content) {
            return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
        }
    }
    try {
        std::lock_guard<std::mutex> lock(model->chat_mutex);
        if (!model->chat_templates) {
            model->chat_templates = common_chat_templates_init(model->handle, "");
        }
        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = add_assistant;
        inputs.use_jinja = true;
        inputs.messages.reserve(message_count);
        for (size_t index = 0; index < message_count; ++index) {
            common_chat_msg message;
            message.role = messages[index].role;
            message.content = messages[index].content;
            inputs.messages.push_back(std::move(message));
        }
        const common_chat_params params = common_chat_templates_apply(model->chat_templates.get(), inputs);
        const size_t required_size = params.prompt.size() + 1;
        if (!text || *text_size < required_size) {
            *text_size = required_size;
            return CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL;
        }
        std::memcpy(text, params.prompt.data(), params.prompt.size());
        text[params.prompt.size()] = '\0';
        *text_size = required_size;
        return CELIUMS_BITNET_STATUS_OK;
    } catch (const std::exception &) {
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
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
            resolved.threads <= 0 || resolved.threads_batch <= 0 || resolved.n_seq == 0) {
        return CELIUMS_BITNET_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t budget = resolved.ram_budget_bytes
        ? resolved.ram_budget_bytes
        : (model->runtime ? model->runtime->ram_budget_bytes : celiums_bitnet_default_ram_budget_bytes());
    const uint64_t quoted = celiums_bitnet_estimate_session_ram_bytes_for_model(model, &resolved);
    const uint64_t incremental = session_incremental_bytes(model->handle, resolved);
    if (quoted > budget || incremental == std::numeric_limits<uint64_t>::max() ||
            !ram_try_reserve(model->runtime, incremental)) {
        return CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED;
    }
    llama_context_params params = llama_context_default_params();
    params.n_ctx = resolved.context_size;
    params.n_batch = resolved.batch_size;
    params.n_ubatch = resolved.ubatch_size;
    params.n_threads = resolved.threads;
    params.n_threads_batch = resolved.threads_batch;
    params.n_seq_max = resolved.n_seq;
    params.no_perf = false;
    llama_context * context = llama_init_from_model(model->handle, params);
    if (!context) {
        ram_release(model->runtime, incremental);
        return CELIUMS_BITNET_STATUS_CONTEXT_CREATE_FAILED;
    }
    auto * created = new (std::nothrow) celiums_bitnet_session { model, context };
    if (!created) {
        llama_free(context);
        ram_release(model->runtime, incremental);
        return CELIUMS_BITNET_STATUS_INTERNAL_ERROR;
    }
    created->reserved_bytes = incremental;
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
    if (session->sampler) {
        llama_sampler_reset(session->sampler);
    }
    session->position = 0;
    session->last_logits_index = -2;
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
    if (request->cancelled.load(std::memory_order_acquire)) {
        result->cancelled = true;
        return CELIUMS_BITNET_STATUS_CANCELLED;
    }

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
    request->session->last_logits_index = -2;
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
