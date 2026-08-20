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

typedef enum celiums_bitnet_status {
    CELIUMS_BITNET_STATUS_OK = 0,
    CELIUMS_BITNET_STATUS_INVALID_ARGUMENT = 1,
    CELIUMS_BITNET_STATUS_MODEL_LOAD_FAILED = 2,
    CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL = 3,
    CELIUMS_BITNET_STATUS_INTERNAL_ERROR = 4,
    CELIUMS_BITNET_STATUS_UNSUPPORTED_MODEL = 5,
} celiums_bitnet_status;

typedef struct celiums_bitnet_runtime_options {
    size_t struct_size;
    uint32_t api_version;
} celiums_bitnet_runtime_options;

typedef struct celiums_bitnet_model_options {
    size_t struct_size;
    uint32_t api_version;
    bool use_mmap;
    bool use_mlock;
    bool check_tensors;
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

CELIUMS_BITNET_API const char * celiums_bitnet_version(void);
CELIUMS_BITNET_API const char * celiums_bitnet_product_commit(void);
CELIUMS_BITNET_API const char * celiums_bitnet_engine_commit(void);
CELIUMS_BITNET_API const char * celiums_bitnet_cpu_profile(void);
CELIUMS_BITNET_API const char * celiums_bitnet_status_string(celiums_bitnet_status status);

CELIUMS_BITNET_API celiums_bitnet_runtime_options celiums_bitnet_runtime_default_options(void);
CELIUMS_BITNET_API celiums_bitnet_model_options celiums_bitnet_model_default_options(void);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_runtime_create(
    const celiums_bitnet_runtime_options * options,
    celiums_bitnet_runtime ** runtime);
CELIUMS_BITNET_API void celiums_bitnet_runtime_destroy(celiums_bitnet_runtime * runtime);

CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_load(
    celiums_bitnet_runtime * runtime,
    const char * path,
    const celiums_bitnet_model_options * options,
    celiums_bitnet_model ** model);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_validate_strict(
    celiums_bitnet_runtime * runtime,
    const char * path,
    celiums_bitnet_model_info * info);
CELIUMS_BITNET_API void celiums_bitnet_model_destroy(celiums_bitnet_model * model);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_get_info(
    const celiums_bitnet_model * model,
    celiums_bitnet_model_info * info);
CELIUMS_BITNET_API celiums_bitnet_status celiums_bitnet_model_get_description(
    const celiums_bitnet_model * model,
    char * buffer,
    size_t * buffer_size);

#ifdef __cplusplus
}
#endif

#endif
