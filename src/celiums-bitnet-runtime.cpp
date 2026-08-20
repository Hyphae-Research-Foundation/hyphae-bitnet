#include "celiums/bitnet_runtime.h"

#include "llama.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

struct celiums_bitnet_runtime {
    std::atomic<uint32_t> references { 1 };
    std::atomic<bool> active { true };
};

struct celiums_bitnet_model {
    celiums_bitnet_runtime * runtime;
    llama_model * handle;
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
    params.vocab_only = true;
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
    if (resolved.check_tensors) {
        llama_model_free(handle);
        params.vocab_only = false;
        handle = llama_model_load_from_file(path, params);
        if (!handle) {
            return CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED;
        }
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
    llama_model_free(model->handle);
    release_runtime(model->runtime);
    delete model;
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
