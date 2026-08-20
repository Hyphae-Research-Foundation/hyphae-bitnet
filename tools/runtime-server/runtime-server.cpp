#include "celiums/bitnet_runtime.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct server_options {
    const char * model = nullptr;
    std::string host = "127.0.0.1";
    int port = 8080;
    int32_t context_size = 2048;
    int32_t batch_size = 512;
    int32_t ubatch_size = 512;
    int32_t threads = 2;
    int32_t threads_batch = 2;
    std::string api_key;
};

void usage(const char * program) {
    printf("Usage: %s --model MODEL.gguf [--host 127.0.0.1] [--port 8080]\n", program);
}

bool parse(int argc, char ** argv, server_options & result) {
    if (const char * api_key = std::getenv("CELIUMS_BITNET_API_KEY")) {
        result.api_key = api_key;
    }
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            return false;
        }
        if (index + 1 >= argc) return false;
        const char * value = argv[++index];
        if (argument == "--model" || argument == "-m") result.model = value;
        else if (argument == "--host") result.host = value;
        else if (argument == "--port") result.port = std::stoi(value);
        else if (argument == "--ctx-size" || argument == "-c") result.context_size = std::stoi(value);
        else if (argument == "--batch-size" || argument == "-b") result.batch_size = std::stoi(value);
        else if (argument == "--ubatch-size" || argument == "-ub") result.ubatch_size = std::stoi(value);
        else if (argument == "--threads" || argument == "-t") result.threads = std::stoi(value);
        else if (argument == "--threads-batch" || argument == "-tb") result.threads_batch = std::stoi(value);
        else if (argument == "--api-key") result.api_key = value;
        else return false;
    }
    return result.model && result.port > 0 && result.port <= 65535 && result.context_size > 0 &&
        result.batch_size > 0 && result.ubatch_size > 0 && result.threads > 0 && result.threads_batch > 0;
}

bool authorized(const httplib::Request & request, const std::string & api_key) {
    if (api_key.empty()) return true;
    std::string supplied = request.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (supplied.rfind(prefix, 0) == 0) supplied.erase(0, prefix.size());
    if (supplied.empty()) supplied = request.get_header_value("X-Api-Key");
    return supplied == api_key;
}

void set_json(httplib::Response & response, const json & body, int status = 200) {
    response.status = status;
    response.set_content(body.dump(), "application/json; charset=utf-8");
}

struct stream_state {
    json chunks = json::array();
};

bool collect_piece(celiums_bitnet_token, const char * piece, size_t size, void * user_data) {
    auto & state = *static_cast<stream_state *>(user_data);
    state.chunks.push_back(std::string(piece, size));
    return true;
}

} // namespace

int celiums_runtime_server(int argc, char ** argv) {
    server_options options;
    try {
        if (!parse(argc, argv, options)) {
            usage(argv[0]);
            return argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") ? 0 : 2;
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "celiums-bitnet serve: %s\n", error.what());
        return 2;
    }

    celiums_bitnet_runtime * runtime = nullptr;
    celiums_bitnet_model * model = nullptr;
    auto runtime_options = celiums_bitnet_runtime_default_options();
    auto status = celiums_bitnet_runtime_create(&runtime_options, &runtime);
    if (status == CELIUMS_BITNET_STATUS_OK) {
        auto model_options = celiums_bitnet_model_default_options();
        status = celiums_bitnet_model_load(runtime, options.model, &model_options, &model);
    }
    if (status != CELIUMS_BITNET_STATUS_OK) {
        fprintf(stderr, "celiums-bitnet serve: %s\n", celiums_bitnet_status_string(status));
        celiums_bitnet_runtime_destroy(runtime);
        return 1;
    }

    httplib::Server server;
    server.set_pre_routing_handler([&](const httplib::Request & request, httplib::Response & response) {
        response.set_header("X-Celiums-BitNet-Runtime", celiums_bitnet_version());
        if ((request.path == "/health" || request.path == "/v1/health" || request.path == "/v1/models") ||
                authorized(request, options.api_key)) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        set_json(response, {{"error", {{"message", "Invalid API Key"}, {"type", "authentication_error"}, {"code", 401}}}}, 401);
        return httplib::Server::HandlerResponse::Handled;
    });

    server.Get("/health", [](const httplib::Request &, httplib::Response & response) {
        set_json(response, {{"status", "ok"}});
    });
    server.Get("/v1/health", [](const httplib::Request &, httplib::Response & response) {
        set_json(response, {{"status", "ok"}});
    });
    server.Get("/v1/models", [&](const httplib::Request &, httplib::Response & response) {
        set_json(response, {{"object", "list"}, {"data", json::array({{
            {"id", "celiums-bitnet"}, {"object", "model"}, {"owned_by", "celiums"}
        }})}});
    });

    auto completion = [&](const httplib::Request & request, httplib::Response & response, bool chat) {
        try {
            const json body = json::parse(request.body);
            if (body.value("stream", false)) {
                set_json(response, {{"error", {{"message", "streaming HTTP responses are not implemented"}, {"type", "not_supported_error"}}}}, 400);
                return;
            }
            std::string prompt;
            if (chat) {
                for (const auto & message : body.at("messages")) {
                    const std::string role = message.value("role", "user");
                    const std::string content = message.value("content", "");
                    prompt += role + ": " + content + "\n";
                }
                prompt += "assistant: ";
            } else {
                prompt = body.value("prompt", "");
            }
            if (prompt.empty()) {
                set_json(response, {{"error", {{"message", "prompt is required"}, {"type", "invalid_request_error"}}}}, 400);
                return;
            }

            celiums_bitnet_session * session = nullptr;
            celiums_bitnet_request * generation_request = nullptr;
            auto session_options = celiums_bitnet_session_default_options();
            session_options.context_size = options.context_size;
            session_options.batch_size = options.batch_size;
            session_options.ubatch_size = options.ubatch_size;
            session_options.threads = options.threads;
            session_options.threads_batch = options.threads_batch;
            auto request_status = celiums_bitnet_session_create(model, &session_options, &session);
            if (request_status == CELIUMS_BITNET_STATUS_OK) {
                request_status = celiums_bitnet_request_create(session, &generation_request);
            }
            auto generation = celiums_bitnet_generation_default_options();
            generation.max_tokens = body.value("max_tokens", 128);
            generation.temperature = body.value("temperature", 0.8f);
            generation.top_k = body.value("top_k", 40);
            generation.top_p = body.value("top_p", 0.95f);
            generation.seed = body.value("seed", UINT32_MAX);
            stream_state stream;
            celiums_bitnet_generation_result generated = {};
            generated.struct_size = sizeof(generated);
            generated.api_version = CELIUMS_BITNET_API_VERSION;
            if (request_status == CELIUMS_BITNET_STATUS_OK) {
                request_status = celiums_bitnet_generate(
                    generation_request, prompt.c_str(), &generation, collect_piece, &stream, &generated);
            }
            std::string text;
            for (const auto & piece : stream.chunks) text += piece.get<std::string>();
            celiums_bitnet_request_destroy(generation_request);
            celiums_bitnet_session_destroy(session);
            if (request_status != CELIUMS_BITNET_STATUS_OK) {
                set_json(response, {{"error", {{"message", celiums_bitnet_status_string(request_status)}, {"type", "server_error"}}}}, 500);
                return;
            }
            if (chat) {
                set_json(response, {
                    {"id", "chatcmpl-celiums"}, {"object", "chat.completion"}, {"model", "celiums-bitnet"},
                    {"choices", json::array({{{"index", 0}, {"message", {{"role", "assistant"}, {"content", text}}}, {"finish_reason", "stop"}}})},
                    {"usage", {{"prompt_tokens", nullptr}, {"completion_tokens", generated.generated_tokens}, {"total_tokens", nullptr}}}
                });
            } else {
                set_json(response, {
                    {"id", "cmpl-celiums"}, {"object", "text_completion"}, {"model", "celiums-bitnet"},
                    {"choices", json::array({{{"index", 0}, {"text", text}, {"finish_reason", "stop"}}})},
                    {"usage", {{"prompt_tokens", nullptr}, {"completion_tokens", generated.generated_tokens}, {"total_tokens", nullptr}}}
                });
            }
        } catch (const std::exception & error) {
            set_json(response, {{"error", {{"message", error.what()}, {"type", "invalid_request_error"}}}}, 400);
        }
    };

    server.Post("/v1/completions", [&](const httplib::Request & request, httplib::Response & response) {
        completion(request, response, false);
    });
    server.Post("/v1/chat/completions", [&](const httplib::Request & request, httplib::Response & response) {
        completion(request, response, true);
    });

    fprintf(stderr, "Celiums BitNet Runtime listening on http://%s:%d\n", options.host.c_str(), options.port);
    const bool listened = server.listen(options.host, options.port);
    celiums_bitnet_model_destroy(model);
    celiums_bitnet_runtime_destroy(runtime);
    return listened ? 0 : 1;
}
