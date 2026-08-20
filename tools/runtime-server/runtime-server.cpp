#include "celiums/bitnet_runtime.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
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

struct server_metrics {
    std::atomic<uint64_t> requests { 0 };
    std::atomic<uint64_t> completions { 0 };
    std::atomic<uint64_t> failures { 0 };
    std::atomic<uint64_t> cancelled { 0 };
    std::atomic<uint64_t> generated_tokens { 0 };
    std::atomic<uint64_t> active { 0 };
};

struct generation_state {
    celiums_bitnet_session * session = nullptr;
    celiums_bitnet_request * request = nullptr;
    celiums_bitnet_generation_result result = {};
    celiums_bitnet_status status = CELIUMS_BITNET_STATUS_OK;
    std::string text;
};

struct callback_state {
    generation_state * generation;
    httplib::DataSink * sink = nullptr;
    std::function<bool()> connection_closed;
    std::string id;
    bool chat = false;
    bool sent_role = false;
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

void set_error(httplib::Response & response, const std::string & message, const std::string & type, int status) {
    set_json(response, {{"error", {{"message", message}, {"type", type}, {"code", status}}}}, status);
}

std::string finish_reason(const generation_state & state, int32_t max_tokens) {
    if (state.result.stopped_by_eog || state.result.stopped_by_sequence) return "stop";
    return state.result.generated_tokens >= max_tokens ? "length" : "stop";
}

bool collect_piece(celiums_bitnet_token, const char * piece, size_t size, void * user_data) {
    auto & state = *static_cast<callback_state *>(user_data);
    if ((state.connection_closed && state.connection_closed()) ||
            (state.sink && !state.sink->is_writable())) {
        celiums_bitnet_request_cancel(state.generation->request);
        return false;
    }
    if (!state.sink) {
        state.generation->text.append(piece, size);
        return true;
    }
    json choice;
    if (state.chat) {
        json delta = {{"content", std::string(piece, size)}};
        if (!state.sent_role) {
            delta["role"] = "assistant";
            state.sent_role = true;
        }
        choice = {{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}};
    } else {
        choice = {{"index", 0}, {"text", std::string(piece, size)}, {"finish_reason", nullptr}};
    }
    const json chunk = {
        {"id", state.id},
        {"object", state.chat ? "chat.completion.chunk" : "text_completion"},
        {"model", "celiums-bitnet"},
        {"choices", json::array({choice})},
    };
    const std::string frame = "data: " + chunk.dump() + "\n\n";
    if (!state.sink->write(frame.data(), frame.size())) {
        celiums_bitnet_request_cancel(state.generation->request);
        return false;
    }
    return true;
}

std::string prompt_from_body(const celiums_bitnet_model * model, const json & body, bool chat) {
    if (!chat) return body.value("prompt", "");
    const auto & input = body.at("messages");
    if (!input.is_array() || input.empty()) throw std::invalid_argument("messages must be a non-empty array");
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    roles.reserve(input.size());
    contents.reserve(input.size());
    for (const auto & message : input) {
        if (!message.is_object() || !message.contains("role") || !message.contains("content") ||
                !message["role"].is_string() || !message["content"].is_string()) {
            throw std::invalid_argument("each message requires string role and content fields");
        }
        roles.push_back(message["role"].get<std::string>());
        contents.push_back(message["content"].get<std::string>());
    }
    std::vector<celiums_bitnet_chat_message> messages(input.size());
    for (size_t index = 0; index < messages.size(); ++index) {
        messages[index] = { roles[index].c_str(), contents[index].c_str() };
    }
    size_t size = 0;
    auto status = celiums_bitnet_model_apply_chat_template(
        model, messages.data(), messages.size(), true, nullptr, &size);
    if (status != CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL) {
        throw std::runtime_error(celiums_bitnet_status_string(status));
    }
    std::vector<char> buffer(size);
    status = celiums_bitnet_model_apply_chat_template(
        model, messages.data(), messages.size(), true, buffer.data(), &size);
    if (status != CELIUMS_BITNET_STATUS_OK) {
        throw std::runtime_error(celiums_bitnet_status_string(status));
    }
    return std::string(buffer.data(), size - 1);
}

void create_generation(
        celiums_bitnet_model * model,
        const server_options & options,
        generation_state & state) {
    auto session_options = celiums_bitnet_session_default_options();
    session_options.context_size = options.context_size;
    session_options.batch_size = options.batch_size;
    session_options.ubatch_size = options.ubatch_size;
    session_options.threads = options.threads;
    session_options.threads_batch = options.threads_batch;
    state.status = celiums_bitnet_session_create(model, &session_options, &state.session);
    if (state.status == CELIUMS_BITNET_STATUS_OK) {
        state.status = celiums_bitnet_request_create(state.session, &state.request);
    }
    state.result.struct_size = sizeof(state.result);
    state.result.api_version = CELIUMS_BITNET_API_VERSION;
}

void destroy_generation(generation_state & state) {
    celiums_bitnet_request_destroy(state.request);
    celiums_bitnet_session_destroy(state.session);
    state.request = nullptr;
    state.session = nullptr;
}

std::string metrics_text(const server_metrics & metrics) {
    return
        "# TYPE celiums_bitnet_http_requests_total counter\n"
        "celiums_bitnet_http_requests_total " + std::to_string(metrics.requests.load()) + "\n"
        "# TYPE celiums_bitnet_completions_total counter\n"
        "celiums_bitnet_completions_total " + std::to_string(metrics.completions.load()) + "\n"
        "# TYPE celiums_bitnet_failures_total counter\n"
        "celiums_bitnet_failures_total " + std::to_string(metrics.failures.load()) + "\n"
        "# TYPE celiums_bitnet_cancelled_total counter\n"
        "celiums_bitnet_cancelled_total " + std::to_string(metrics.cancelled.load()) + "\n"
        "# TYPE celiums_bitnet_generated_tokens_total counter\n"
        "celiums_bitnet_generated_tokens_total " + std::to_string(metrics.generated_tokens.load()) + "\n"
        "# TYPE celiums_bitnet_active_requests gauge\n"
        "celiums_bitnet_active_requests " + std::to_string(metrics.active.load()) + "\n";
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

    server_metrics metrics;
    std::atomic<uint64_t> request_id { 0 };
    httplib::Server server;
    server.set_pre_routing_handler([&](const httplib::Request & request, httplib::Response & response) {
        response.set_header("X-Celiums-BitNet-Runtime", celiums_bitnet_version());
        if ((request.path == "/health" || request.path == "/v1/health" || request.path == "/v1/models") ||
                authorized(request, options.api_key)) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        set_error(response, "Invalid API Key", "authentication_error", 401);
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
    server.Get("/metrics", [&](const httplib::Request &, httplib::Response & response) {
        response.set_content(metrics_text(metrics), "text/plain; version=0.0.4; charset=utf-8");
    });

    auto completion = [&](const httplib::Request & request, httplib::Response & response, bool chat) {
        ++metrics.requests;
        try {
            const json body = json::parse(request.body);
            const std::string prompt = prompt_from_body(model, body, chat);
            if (prompt.empty()) {
                set_error(response, chat ? "messages are required" : "prompt is required", "invalid_request_error", 400);
                ++metrics.failures;
                return;
            }
            const int32_t max_tokens = body.value("max_tokens", 128);
            if (max_tokens < 0) throw std::invalid_argument("max_tokens must be nonnegative");
            auto generation = celiums_bitnet_generation_default_options();
            generation.max_tokens = max_tokens;
            generation.temperature = body.value("temperature", 0.8f);
            generation.top_k = body.value("top_k", 40);
            generation.top_p = body.value("top_p", 0.95f);
            generation.seed = body.value("seed", UINT32_MAX);
            const std::string id = (chat ? "chatcmpl-celiums-" : "cmpl-celiums-") +
                std::to_string(request_id.fetch_add(1));

            if (body.value("stream", false)) {
                response.set_header("Cache-Control", "no-cache");
                response.set_header("X-Accel-Buffering", "no");
                auto streamed = std::make_shared<bool>(false);
                response.set_chunked_content_provider(
                    "text/event-stream; charset=utf-8",
                    [&, streamed, prompt, generation, id, chat, connection_closed = request.is_connection_closed]
                    (size_t, httplib::DataSink & sink) mutable {
                        if (*streamed) {
                            sink.done();
                            return true;
                        }
                        *streamed = true;
                        generation_state state;
                        create_generation(model, options, state);
                        callback_state callback { &state, &sink, connection_closed, id, chat };
                        ++metrics.active;
                        if (state.status == CELIUMS_BITNET_STATUS_OK) {
                            state.status = celiums_bitnet_generate(
                                state.request, prompt.c_str(), &generation, collect_piece, &callback, &state.result);
                        }
                        --metrics.active;
                        metrics.generated_tokens += state.result.generated_tokens;
                        const bool disconnected = connection_closed() || !sink.is_writable();
                        if (state.status == CELIUMS_BITNET_STATUS_OK && !disconnected) {
                            const json choice = chat
                                ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason(state, max_tokens)}}
                                : json{{"index", 0}, {"text", ""}, {"finish_reason", finish_reason(state, max_tokens)}};
                            const json final_chunk = {
                                {"id", id},
                                {"object", chat ? "chat.completion.chunk" : "text_completion"},
                                {"model", "celiums-bitnet"},
                                {"choices", json::array({choice})},
                            };
                            const std::string final_frame = "data: " + final_chunk.dump() + "\n\ndata: [DONE]\n\n";
                            sink.write(final_frame.data(), final_frame.size());
                            ++metrics.completions;
                        } else if (disconnected || state.status == CELIUMS_BITNET_STATUS_CANCELLED ||
                                state.status == CELIUMS_BITNET_STATUS_CALLBACK_ABORTED) {
                            ++metrics.cancelled;
                        } else {
                            const json error = {{"error", {{"message", celiums_bitnet_status_string(state.status)}, {"type", "server_error"}}}};
                            const std::string error_frame = "data: " + error.dump() + "\n\ndata: [DONE]\n\n";
                            sink.write(error_frame.data(), error_frame.size());
                            ++metrics.failures;
                        }
                        destroy_generation(state);
                        sink.done();
                        return !disconnected;
                    });
                return;
            }

            generation_state state;
            create_generation(model, options, state);
            callback_state callback { &state, nullptr, request.is_connection_closed, id, chat };
            ++metrics.active;
            if (state.status == CELIUMS_BITNET_STATUS_OK) {
                state.status = celiums_bitnet_generate(
                    state.request, prompt.c_str(), &generation, collect_piece, &callback, &state.result);
            }
            --metrics.active;
            metrics.generated_tokens += state.result.generated_tokens;
            if (state.status != CELIUMS_BITNET_STATUS_OK) {
                const bool cancelled = state.status == CELIUMS_BITNET_STATUS_CANCELLED ||
                    state.status == CELIUMS_BITNET_STATUS_CALLBACK_ABORTED;
                if (cancelled) ++metrics.cancelled;
                else ++metrics.failures;
                set_error(response, celiums_bitnet_status_string(state.status),
                    cancelled ? "cancelled_error" : "server_error", cancelled ? 499 : 500);
                destroy_generation(state);
                return;
            }
            ++metrics.completions;
            if (chat) {
                set_json(response, {
                    {"id", id}, {"object", "chat.completion"}, {"model", "celiums-bitnet"},
                    {"choices", json::array({{{"index", 0}, {"message", {{"role", "assistant"}, {"content", state.text}}}, {"finish_reason", finish_reason(state, max_tokens)}}})},
                    {"usage", {{"prompt_tokens", nullptr}, {"completion_tokens", state.result.generated_tokens}, {"total_tokens", nullptr}}}
                });
            } else {
                set_json(response, {
                    {"id", id}, {"object", "text_completion"}, {"model", "celiums-bitnet"},
                    {"choices", json::array({{{"index", 0}, {"text", state.text}, {"finish_reason", finish_reason(state, max_tokens)}}})},
                    {"usage", {{"prompt_tokens", nullptr}, {"completion_tokens", state.result.generated_tokens}, {"total_tokens", nullptr}}}
                });
            }
            destroy_generation(state);
        } catch (const std::exception & error) {
            ++metrics.failures;
            set_error(response, error.what(), "invalid_request_error", 400);
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
