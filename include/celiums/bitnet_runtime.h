#ifndef CELIUMS_BITNET_RUNTIME_H
#define CELIUMS_BITNET_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(CELIUMS_BITNET_RUNTIME_BUILD)
#    define CELIUMS_BITNET_API __declspec(dllexport)
#  else
#    define CELIUMS_BITNET_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define CELIUMS_BITNET_API __attribute__((visibility("default")))
#else
#  define CELIUMS_BITNET_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CELIUMS_BITNET_API_VERSION 1u

typedef struct celiums_bitnet_runtime celiums_bitnet_runtime;
typedef struct celiums_bitnet_model celiums_bitnet_model;
typedef struct celiums_bitnet_session celiums_bitnet_session;
typedef struct celiums_bitnet_request celiums_bitnet_request;
typedef int32_t celiums_bitnet_token;

typedef enum celiums_bitnet_status {
    CELIUMS_BITNET_STATUS_OK = 0,
    CELIUMS_BITNET_STATUS_INVALID_ARGUMENT = 1,
    CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED = 2,
    CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL = 3,
    CELIUMS_BITNET_STATUS_INTERNAL_ERROR = 4,
    CELIUMS_BITNET_STATUS_UNSUPPORTED_MODEL = 5,
    CELIUMS_BITNET_STATUS_CONTEXT_CREATE_FAILED = 6,
    CELIUMS_BITNET_STATUS_DECODE_FAILED = 7,
    CELIUMS_BITNET_STATUS_CANCELLED = 8,
    CELIUMS_BITNET_STATUS_CONTEXT_FULL = 9,
    CELIUMS_BITNET_STATUS_CALLBACK_ABORTED = 10,
    CELIUMS_BITNET_STATUS_RAM_BUDGET_EXCEEDED = 11,
} celiums_bitnet_status;

typedef enum celiums_bitnet_model_family {
    CELIUMS_BITNET_MODEL_FAMILY_UNKNOWN = 0,
    CELIUMS_BITNET_MODEL_FAMILY_BITNET_B158_I2S = 1,
    CELIUMS_BITNET_MODEL_FAMILY_BONSAI_QWEN35_Q1_0 = 2,
} celiums_bitnet_model_family;

typedef struct celiums_bitnet_runtime_options {
    size_t struct_size;
    uint32_t api_version;
    /* 0 = auto: half of host RAM, always leaving headroom so the host stays usable. */
    uint64_t ram_budget_bytes;
} celiums_bitnet_runtime_options;

typedef struct celiums_bitnet_model_options {
    size_t struct_size;
    uint32_t api_version;
    bool use_mmap;
    bool use_mlock;
    bool check_tensors;
    /* Materialize a compute-oriented in-RAM image (Q1 4x8 panels) when the ISA allows. */
    bool use_compute_layout;
} celiums_bitnet_model_options;

typedef struct celiums_bitnet_model_info {
    size_t struct_size;
    uint32_t api_version;
    uint64_t size_bytes;
    uint64_t parameter_count;
    int32_t context_length;
    int32_t embedding_length;
    int32_t layer_count;
} celiums_bitnet_model_info;

typedef struct celiums_bitnet_session_options {
    size_t struct_size;
    uint32_t api_version;
    uint32_t context_size;
    uint32_t batch_size;
    uint32_t ubatch_size;
    int32_t threads;
    int32_t threads_batch;
    /* Distinct decode states sharing one weight image. 1 = single sequence. */
    uint32_t n_seq;
    /* 0 = inherit the runtime budget. Extra KV/layout must fit or create fails. */
    uint64_t ram_budget_bytes;
    bool use_compute_layout;
} celiums_bitnet_session_options;

typedef struct celiums_bitnet_generation_options {
    size_t struct_size;
    uint32_t api_version;
    int32_t max_tokens;
    float temperature;
    int32_t top_k;
    float top_p;
    uint32_t seed;
    const char * const * stop_sequences;
    size_t stop_sequence_count;
} celiums_bitnet_generation_options;

typedef struct celiums_bitnet_generation_result {
    size_t struct_size;
    uint32_t api_version;
    int32_t generated_tokens;
    bool stopped_by_eog;
    bool stopped_by_sequence;
    bool cancelled;
} celiums_bitnet_generation_result;

typedef struct celiums_bitnet_chat_message {
    const char * role;
    const char * content;
} celiums_bitnet_chat_message;

typedef bool (*celiums_bitnet_stream_callback)(
    celiums_bitnet_token token,
    const char * piece,
    size_t piece_size,
    void * user_data);

CELIUMS_BITNET_API const char * celiums_bitnet_version(void);
CELIUMS_BITNET_API const char * celiums_bitnet_product_commit(void);
CELIUMS_BITNET_API const char * celiums_bitnet_engine_commit(void);
CELIUMS_BITNET_API const char * celiums_bitnet_engine_tree(void);
CELIUMS_BITNET_API const char * celiums_bitnet_cpu_profile(void);
CELIUMS_BITNET_API const char * celiums_bitnet_status_string(celiums_bitnet_status status);
CELIUMS_BITNET_API const char * celiums_bitnet_model_family_string(celiums_bitnet_model_family family);

CELIUMS_BITNET_API celiums_bitnet_runtime_options celiums_bitnet_runtime_default_options(void);
CELIUMS_BITNET_API celiums_bitnet_model_options celiums_bitnet_model_default_options(void);
CELIUMS_BITNET_API celiums_bitnet_session_options celiums_bitnet_session_default_options(void);
CELIUMS_BITNET_API celiums_bitnet_generation_options celiums_bitnet_generation_default_options(void);

/* Host MemTotal. Auto budget is half of this, never more than 90% and always
 * leaving at least 4 GiB (or 10%) free so serving stays possible. */
CELIUMS_BITNET_API uint64_t celiums_bitnet_host_ram_bytes(void);
CELIUMS_BITNET_API uint64_t celiums_bitnet_default_ram_budget_bytes(void);
CELIUMS_BITNET_API uint64_t celiums_bitnet_estimate_session_ram_bytes(
    const celiums_bitnet_session_options * options,
    uint64_t packed_model_bytes);
/* KV/RS/compute from the loaded model's allocator geometry, plus layout. */
CELIUMS_BITNET_API uint64_t celiums_bitnet_estimate_session_ram_bytes_for_model(
    const celiums_bitnet_model * model,
    const celiums_bitnet_session_options * options);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_runtime_create(
    const celiums_bitnet_runtime_options * options,
    celiums_bitnet_runtime ** runtime);
CELIUMS_BITNET_API void celiums_bitnet_runtime_destroy(celiums_bitnet_runtime * runtime);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_load(
    celiums_bitnet_runtime * runtime,
    const char * path,
    const celiums_bitnet_model_options * options,
    celiums_bitnet_model ** model);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_load_family(
    celiums_bitnet_runtime * runtime,
    const char * path,
    celiums_bitnet_model_family family,
    const celiums_bitnet_model_options * options,
    celiums_bitnet_model ** model);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_validate_strict(
    celiums_bitnet_runtime * runtime,
    const char * path,
    celiums_bitnet_model_info * info);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_validate_family(
    celiums_bitnet_runtime * runtime,
    const char * path,
    celiums_bitnet_model_family family,
    celiums_bitnet_model_info * info);
CELIUMS_BITNET_API void celiums_bitnet_model_destroy(celiums_bitnet_model * model);
CELIUMS_BITNET_API celiums_bitnet_model_family celiums_bitnet_model_get_family(
    const celiums_bitnet_model * model);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_get_info(
    const celiums_bitnet_model * model,
    celiums_bitnet_model_info * info);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_get_description(
    const celiums_bitnet_model * model,
    char * buffer,
    size_t * buffer_size);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_apply_chat_template(
    const celiums_bitnet_model * model,
    const celiums_bitnet_chat_message * messages,
    size_t message_count,
    bool add_assistant,
    char * text,
    size_t * text_size);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_session_create(
    celiums_bitnet_model * model,
    const celiums_bitnet_session_options * options,
    celiums_bitnet_session ** session);
CELIUMS_BITNET_API void celiums_bitnet_session_destroy(celiums_bitnet_session * session);
CELIUMS_BITNET_API void celiums_bitnet_session_reset(celiums_bitnet_session * session);
CELIUMS_BITNET_API int32_t celiums_bitnet_session_context_size(const celiums_bitnet_session * session);
CELIUMS_BITNET_API int32_t celiums_bitnet_session_vocab_size(const celiums_bitnet_session * session);
CELIUMS_BITNET_API int32_t celiums_bitnet_session_position(const celiums_bitnet_session * session);
CELIUMS_BITNET_API bool celiums_bitnet_model_token_is_eog(
    const celiums_bitnet_model * model,
    celiums_bitnet_token token);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_token_to_piece(
    const celiums_bitnet_model * model,
    celiums_bitnet_token token,
    char * piece,
    size_t * piece_size);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_tokenize(
    const celiums_bitnet_model * model,
    const char * text,
    bool add_special,
    bool parse_special,
    celiums_bitnet_token * tokens,
    size_t * token_count);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_detokenize(
    const celiums_bitnet_model * model,
    const celiums_bitnet_token * tokens,
    size_t token_count,
    bool remove_special,
    bool unparse_special,
    char * text,
    size_t * text_size);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_session_prefill(
    celiums_bitnet_session * session,
    const celiums_bitnet_token * tokens,
    size_t token_count,
    bool output_logits);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_session_decode(
    celiums_bitnet_session * session,
    celiums_bitnet_token token,
    bool output_logits);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_session_copy_logits(
    celiums_bitnet_session * session,
    float * logits,
    size_t * logits_count);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_session_sample(
    celiums_bitnet_session * session,
    const celiums_bitnet_generation_options * options,
    celiums_bitnet_token * token);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_request_create(
    celiums_bitnet_session * session,
    celiums_bitnet_request ** request);
CELIUMS_BITNET_API void celiums_bitnet_request_cancel(celiums_bitnet_request * request);
CELIUMS_BITNET_API bool celiums_bitnet_request_is_cancelled(const celiums_bitnet_request * request);
// Destroy requires no concurrent celiums_bitnet_generate call for this Request.
CELIUMS_BITNET_API void celiums_bitnet_request_destroy(celiums_bitnet_request * request);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_generate(
    celiums_bitnet_request * request,
    const char * prompt,
    const celiums_bitnet_generation_options * options,
    celiums_bitnet_stream_callback callback,
    void * user_data,
    celiums_bitnet_generation_result * result);

#ifdef __cplusplus
}
#endif

#endif
