#include "celiums/bitnet_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

int celiums_runtime_bench(int argc, char ** argv);
#ifdef CELIUMS_BITNET_RUNTIME_SERVER
int celiums_runtime_server(int argc, char ** argv);
#endif

namespace {

void print_help(const char * program) {
    printf("Celiums BitNet Runtime %s\n\n", celiums_bitnet_version());
    printf("Usage: %s <command> [options]\n\n", program);
    printf("Commands:\n");
    printf("  run       Run one-shot BitNet inference\n");
#ifdef CELIUMS_BITNET_RUNTIME_SERVER
    printf("  serve     Start the HTTP API server\n");
#endif
    printf("  bench     Benchmark prefill and decode\n");
    printf("  validate  Load a model and print validated metadata\n");
    printf("  version   Show runtime and engine versions\n");
    printf("  help      Show this help\n");
}

int print_version() {
    printf("Celiums BitNet Runtime %s\n", celiums_bitnet_version());
    printf("product commit: %s\n", celiums_bitnet_product_commit());
    printf("engine commit: %s\n", celiums_bitnet_engine_commit());
    printf("engine tree: %s\n", celiums_bitnet_engine_tree());
    printf("cpu profile: %s\n", celiums_bitnet_cpu_profile());
    printf("strict: true\n");
    return 0;
}

struct run_options {
    const char * model = nullptr;
    const char * prompt = nullptr;
    int32_t max_tokens = 128;
    int32_t context_size = 2048;
    int32_t batch_size = 512;
    int32_t ubatch_size = 512;
    int32_t threads = 2;
    int32_t threads_batch = 2;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float temperature = 0.8f;
    uint32_t seed = UINT32_MAX;
    std::vector<const char *> stop_sequences;
};

void print_run_help(const char * program) {
    printf("Usage: %s --model MODEL.gguf --prompt TEXT [options]\n", program);
    printf("  -n, --n-predict N        maximum generated tokens\n");
    printf("  -c, --ctx-size N         context size\n");
    printf("  -b, --batch-size N       logical batch size\n");
    printf("  -ub, --ubatch-size N     physical batch size\n");
    printf("  -t, --threads N          decode threads\n");
    printf("  -tb, --threads-batch N   prefill threads\n");
    printf("  --temp N                 temperature; 0 uses greedy sampling\n");
    printf("  --top-k N                top-k sampling\n");
    printf("  --top-p N                nucleus sampling\n");
    printf("  --seed N                 sampling seed\n");
    printf("  --stop TEXT              repeatable stop sequence\n");
}

int parse_int(const char * value) {
    size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (value[parsed] != '\0') {
        throw std::invalid_argument("invalid integer");
    }
    return result;
}

float parse_float(const char * value) {
    size_t parsed = 0;
    const float result = std::stof(value, &parsed);
    if (value[parsed] != '\0') {
        throw std::invalid_argument("invalid number");
    }
    return result;
}

bool stream_stdout(celiums_bitnet_token, const char * piece, size_t piece_size, void *) {
    if (piece_size > 0) {
        fwrite(piece, 1, piece_size, stdout);
        fflush(stdout);
    }
    return true;
}

int run_inference(int argc, char ** argv) {
    run_options options;
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if ((argument == "--help" || argument == "-h")) {
                print_run_help(argv[0]);
                return 0;
            }
            if (index + 1 >= argc) {
                fprintf(stderr, "celiums-bitnet run: argument '%s' requires a value\n", argument.c_str());
                return 2;
            }
            const char * value = argv[++index];
            if (argument == "--model" || argument == "-m") options.model = value;
            else if (argument == "--prompt" || argument == "-p") options.prompt = value;
            else if (argument == "--n-predict" || argument == "-n") options.max_tokens = parse_int(value);
            else if (argument == "--ctx-size" || argument == "-c") options.context_size = parse_int(value);
            else if (argument == "--batch-size" || argument == "-b") options.batch_size = parse_int(value);
            else if (argument == "--ubatch-size" || argument == "-ub") options.ubatch_size = parse_int(value);
            else if (argument == "--threads" || argument == "-t") options.threads = parse_int(value);
            else if (argument == "--threads-batch" || argument == "-tb") options.threads_batch = parse_int(value);
            else if (argument == "--temp" || argument == "--temperature") options.temperature = parse_float(value);
            else if (argument == "--top-k") options.top_k = parse_int(value);
            else if (argument == "--top-p") options.top_p = parse_float(value);
            else if (argument == "--seed") options.seed = (uint32_t) std::stoul(value);
            else if (argument == "--stop") options.stop_sequences.push_back(value);
            else {
                fprintf(stderr, "celiums-bitnet run: unknown argument '%s'\n", argument.c_str());
                return 2;
            }
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "celiums-bitnet run: %s\n", error.what());
        return 2;
    }
    if (!options.model || !options.prompt) {
        fprintf(stderr, "celiums-bitnet run: --model and --prompt are required\n");
        return 2;
    }

    celiums_bitnet_runtime * runtime = nullptr;
    celiums_bitnet_model * model = nullptr;
    celiums_bitnet_session * session = nullptr;
    celiums_bitnet_request * request = nullptr;
    celiums_bitnet_status status;

    celiums_bitnet_runtime_options runtime_options = celiums_bitnet_runtime_default_options();
    status = celiums_bitnet_runtime_create(&runtime_options, &runtime);
    if (status == CELIUMS_BITNET_STATUS_OK) {
        celiums_bitnet_model_options model_options = celiums_bitnet_model_default_options();
        status = celiums_bitnet_model_load(runtime, options.model, &model_options, &model);
    }
    if (status == CELIUMS_BITNET_STATUS_OK) {
        celiums_bitnet_session_options session_options = celiums_bitnet_session_default_options();
        session_options.context_size = options.context_size;
        session_options.batch_size = options.batch_size;
        session_options.ubatch_size = options.ubatch_size;
        session_options.threads = options.threads;
        session_options.threads_batch = options.threads_batch;
        status = celiums_bitnet_session_create(model, &session_options, &session);
    }
    if (status == CELIUMS_BITNET_STATUS_OK) {
        status = celiums_bitnet_request_create(session, &request);
    }

    celiums_bitnet_generation_result result = {};
    result.struct_size = sizeof(result);
    result.api_version = CELIUMS_BITNET_API_VERSION;
    if (status == CELIUMS_BITNET_STATUS_OK) {
        celiums_bitnet_generation_options generation = celiums_bitnet_generation_default_options();
        generation.max_tokens = options.max_tokens;
        generation.temperature = options.temperature;
        generation.top_k = options.top_k;
        generation.top_p = options.top_p;
        generation.seed = options.seed;
        generation.stop_sequences = options.stop_sequences.data();
        generation.stop_sequence_count = options.stop_sequences.size();
        status = celiums_bitnet_generate(
            request, options.prompt, &generation, stream_stdout, nullptr, &result);
    }
    printf("\n");
    if (status != CELIUMS_BITNET_STATUS_OK) {
        fprintf(stderr, "celiums-bitnet run: %s\n", celiums_bitnet_status_string(status));
    }
    celiums_bitnet_request_destroy(request);
    celiums_bitnet_session_destroy(session);
    celiums_bitnet_model_destroy(model);
    celiums_bitnet_runtime_destroy(runtime);
    return status == CELIUMS_BITNET_STATUS_OK ? 0 : 1;
}

#ifdef CELIUMS_BITNET_RUNTIME_SERVER
int serve(int argc, char ** argv) {
    return celiums_runtime_server(argc, argv);
}
#endif

int validate_model(int argc, char ** argv) {
    const char * path = nullptr;
    for (int index = 1; index < argc; ++index) {
        if ((strcmp(argv[index], "--model") == 0 || strcmp(argv[index], "-m") == 0) && index + 1 < argc) {
            path = argv[++index];
        } else if (strcmp(argv[index], "--help") == 0 || strcmp(argv[index], "-h") == 0) {
            printf("Usage: %s --model MODEL.gguf\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "celiums-bitnet validate: unknown argument '%s'\n", argv[index]);
            return 2;
        }
    }
    if (!path) {
        fprintf(stderr, "celiums-bitnet validate: --model is required\n");
        return 2;
    }

    celiums_bitnet_runtime * runtime = nullptr;
    celiums_bitnet_runtime_options runtime_options = celiums_bitnet_runtime_default_options();
    celiums_bitnet_status status = celiums_bitnet_runtime_create(&runtime_options, &runtime);
    celiums_bitnet_model_info info = {};
    info.struct_size = sizeof(info);
    info.api_version = CELIUMS_BITNET_API_VERSION;
    if (status == CELIUMS_BITNET_STATUS_OK) {
        status = celiums_bitnet_model_validate_strict(runtime, path, &info);
    }
    if (status != CELIUMS_BITNET_STATUS_OK) {
        fprintf(stderr, "celiums-bitnet validate: %s\n", celiums_bitnet_status_string(status));
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }

    if (status == CELIUMS_BITNET_STATUS_OK) {
        printf("model: %s\n", path);
        printf("size_bytes: %llu\n", (unsigned long long) info.size_bytes);
        printf("parameters: %llu\n", (unsigned long long) info.parameter_count);
        printf("context_length: %d\n", info.context_length);
        printf("embedding_length: %d\n", info.embedding_length);
        printf("layers: %d\n", info.layer_count);
    }
    celiums_bitnet_runtime_destroy(runtime);
    return status == CELIUMS_BITNET_STATUS_OK ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0) {
        print_help(argv[0]);
        return 0;
    }
    if (strcmp(argv[1], "version") == 0 || strcmp(argv[1], "--version") == 0) {
        return print_version();
    }
    if (strcmp(argv[1], "run") == 0) {
        return run_inference(argc - 1, argv + 1);
    }
    if (strcmp(argv[1], "bench") == 0) {
        return celiums_runtime_bench(argc - 1, argv + 1);
    }
    if (strcmp(argv[1], "validate") == 0) {
        return validate_model(argc - 1, argv + 1);
    }
#ifdef CELIUMS_BITNET_RUNTIME_SERVER
    if (strcmp(argv[1], "serve") == 0) {
        return serve(argc - 1, argv + 1);
    }
#endif
    fprintf(stderr, "celiums-bitnet: unknown command '%s'\n", argv[1]);
    return 2;
}
