#include "celiums/bitnet_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int llama_cli(int argc, char ** argv);
int llama_bench(int argc, char ** argv);
#ifdef CELIUMS_BITNET_RUNTIME_SERVER
int llama_server(int argc, char ** argv);
#endif

namespace {

void print_help(const char * program) {
    printf("Celiums BitNet Runtime %s\n\n", celiums_bitnet_version());
    printf("Usage: %s <command> [options]\n\n", program);
    printf("Commands:\n");
    printf("  run       Run interactive BitNet inference\n");
    printf("  serve     Start the HTTP API server\n");
    printf("  bench     Benchmark prefill and decode\n");
    printf("  validate  Load a model and print validated metadata\n");
    printf("  version   Show runtime and engine versions\n");
    printf("  help      Show this help\n");
}

int print_version() {
    printf("Celiums BitNet Runtime %s\n", celiums_bitnet_version());
    printf("product commit: %s\n", celiums_bitnet_product_commit());
    printf("engine commit: %s\n", celiums_bitnet_engine_commit());
    printf("cpu profile: %s\n", celiums_bitnet_cpu_profile());
    printf("strict: true\n");
    return 0;
}

bool is_loopback_host(const std::string & host) {
    return host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
}

int serve(int argc, char ** argv) {
    std::string host = "127.0.0.1";
    bool authenticated = std::getenv("LLAMA_API_KEY") != nullptr;
    bool allow_unauthenticated_remote = false;
    std::vector<char *> forwarded;
    forwarded.reserve(argc);
    forwarded.push_back(argv[0]);

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--allow-unauthenticated-remote") {
            allow_unauthenticated_remote = true;
            continue;
        }
        if (argument == "--host" && index + 1 < argc) {
            host = argv[index + 1];
        } else if (argument.rfind("--host=", 0) == 0) {
            host = argument.substr(strlen("--host="));
        } else if (argument == "--api-key" || argument == "--api-key-file" ||
                argument.rfind("--api-key=", 0) == 0 || argument.rfind("--api-key-file=", 0) == 0) {
            authenticated = true;
        }
        forwarded.push_back(argv[index]);
    }

    if (!is_loopback_host(host) && !authenticated && !allow_unauthenticated_remote) {
        fprintf(stderr,
                "celiums-bitnet serve: refusing unauthenticated remote host '%s'; "
                "use --api-key-file or --allow-unauthenticated-remote\n",
                host.c_str());
        return 2;
    }
#ifdef CELIUMS_BITNET_RUNTIME_SERVER
    return llama_server((int) forwarded.size(), forwarded.data());
#else
    fprintf(stderr, "celiums-bitnet: server support was not built\n");
    return 1;
#endif
}

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
        return llama_cli(argc - 1, argv + 1);
    }
    if (strcmp(argv[1], "bench") == 0) {
        return llama_bench(argc - 1, argv + 1);
    }
    if (strcmp(argv[1], "validate") == 0) {
        return validate_model(argc - 1, argv + 1);
    }
    if (strcmp(argv[1], "serve") == 0) {
        return serve(argc - 1, argv + 1);
    }
    fprintf(stderr, "celiums-bitnet: unknown command '%s'\n", argv[1]);
    return 2;
}
