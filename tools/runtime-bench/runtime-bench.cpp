#include "celiums/bitnet_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct options {
    const char * model = nullptr;
    int32_t prompt_tokens = 128;
    int32_t generated_tokens = 128;
    int32_t batch = 128;
    int32_t ubatch = 128;
    int32_t threads = 2;
    int32_t repetitions = 5;
};

void usage(const char * program) {
    printf("Usage: %s --model MODEL.gguf [-p 128] [-n 128] [-t 2] [-r 5]\n", program);
}

bool parse(int argc, char ** argv, options & result) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            return false;
        }
        if (index + 1 >= argc) {
            return false;
        }
        const char * value = argv[++index];
        if (argument == "--model" || argument == "-m") result.model = value;
        else if (argument == "--n-prompt" || argument == "-p") result.prompt_tokens = std::stoi(value);
        else if (argument == "--n-gen" || argument == "-n") result.generated_tokens = std::stoi(value);
        else if (argument == "--batch-size" || argument == "-b") result.batch = std::stoi(value);
        else if (argument == "--ubatch-size" || argument == "-ub") result.ubatch = std::stoi(value);
        else if (argument == "--threads" || argument == "-t") result.threads = std::stoi(value);
        else if (argument == "--repetitions" || argument == "-r") result.repetitions = std::stoi(value);
        else return false;
    }
    return result.model && result.prompt_tokens >= 0 && result.generated_tokens >= 0 &&
        result.batch > 0 && result.ubatch > 0 && result.threads > 0 && result.repetitions > 0;
}

std::string make_prompt(int32_t target_tokens) {
    std::string prompt;
    for (int32_t index = 0; index < target_tokens; ++index) {
        prompt += index == 0 ? "test" : " test";
    }
    return prompt;
}

bool tokenize_prompt(
        const celiums_bitnet_model * model,
        int32_t target_tokens,
        std::vector<celiums_bitnet_token> & tokens) {
    std::string prompt = make_prompt(std::max(1, target_tokens));
    size_t token_count = 0;
    celiums_bitnet_status status = celiums_bitnet_tokenize(
        model, prompt.c_str(), true, false, nullptr, &token_count);
    if (status != CELIUMS_BITNET_STATUS_BUFFER_TOO_SMALL || token_count == 0) return false;
    tokens.resize(token_count);
    status = celiums_bitnet_tokenize(model, prompt.c_str(), true, false, tokens.data(), &token_count);
    if (status != CELIUMS_BITNET_STATUS_OK) return false;
    tokens.resize(token_count);
    if ((int32_t) tokens.size() > target_tokens) {
        tokens.resize((size_t) target_tokens);
    }
    while ((int32_t) tokens.size() < target_tokens) {
        prompt += " test";
        token_count = 0;
        celiums_bitnet_tokenize(model, prompt.c_str(), true, false, nullptr, &token_count);
        tokens.resize(token_count);
        status = celiums_bitnet_tokenize(model, prompt.c_str(), true, false, tokens.data(), &token_count);
        if (status != CELIUMS_BITNET_STATUS_OK) return false;
        tokens.resize(token_count);
        if ((int32_t) tokens.size() > target_tokens) tokens.resize((size_t) target_tokens);
    }
    return !tokens.empty();
}

double average(const std::vector<double> & values) {
    double sum = 0.0;
    for (double value : values) sum += value;
    return values.empty() ? 0.0 : sum / values.size();
}

} // namespace

int celiums_runtime_bench(int argc, char ** argv) {
    options benchmark;
    if (!parse(argc, argv, benchmark)) {
        usage(argv[0]);
        return argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") ? 0 : 2;
    }

    celiums_bitnet_runtime * runtime = nullptr;
    celiums_bitnet_model * model = nullptr;
    celiums_bitnet_session * session = nullptr;
    auto runtime_options = celiums_bitnet_runtime_default_options();
    auto status = celiums_bitnet_runtime_create(&runtime_options, &runtime);
    if (status == CELIUMS_BITNET_STATUS_OK) {
        auto model_options = celiums_bitnet_model_default_options();
        status = celiums_bitnet_model_load(runtime, benchmark.model, &model_options, &model);
    }
    if (status == CELIUMS_BITNET_STATUS_OK) {
        auto session_options = celiums_bitnet_session_default_options();
        session_options.context_size = std::max(512, benchmark.prompt_tokens + benchmark.generated_tokens + 16);
        session_options.batch_size = benchmark.batch;
        session_options.ubatch_size = benchmark.ubatch;
        session_options.threads = benchmark.threads;
        session_options.threads_batch = benchmark.threads;
        status = celiums_bitnet_session_create(model, &session_options, &session);
    }
    if (status != CELIUMS_BITNET_STATUS_OK) {
        fprintf(stderr, "runtime benchmark: %s\n", celiums_bitnet_status_string(status));
        return 1;
    }

    std::vector<celiums_bitnet_token> tokens;
    if (!tokenize_prompt(model, std::max(1, benchmark.prompt_tokens), tokens)) {
        fprintf(stderr, "runtime benchmark: tokenization failed\n");
        return 1;
    }

    std::vector<double> prompt_rates;
    std::vector<double> decode_rates;
    auto greedy = celiums_bitnet_generation_default_options();
    greedy.temperature = 0.0f;

    for (int32_t repetition = 0; repetition < benchmark.repetitions; ++repetition) {
        if (benchmark.prompt_tokens > 0) {
            celiums_bitnet_session_reset(session);
            const auto start = std::chrono::steady_clock::now();
            status = celiums_bitnet_session_prefill(session, tokens.data(), tokens.size(), true);
            const auto end = std::chrono::steady_clock::now();
            if (status != CELIUMS_BITNET_STATUS_OK) break;
            const double seconds = std::chrono::duration<double>(end - start).count();
            prompt_rates.push_back(tokens.size() / seconds);
        }
        if (benchmark.generated_tokens > 0) {
            celiums_bitnet_session_reset(session);
            status = celiums_bitnet_session_prefill(session, tokens.data(), tokens.size(), true);
            const auto start = std::chrono::steady_clock::now();
            int32_t decoded = 0;
            for (; decoded < benchmark.generated_tokens; ++decoded) {
                celiums_bitnet_token token;
                status = celiums_bitnet_session_sample(session, &greedy, &token);
                if (status != CELIUMS_BITNET_STATUS_OK || celiums_bitnet_model_token_is_eog(model, token)) break;
                status = celiums_bitnet_session_decode(session, token, true);
                if (status != CELIUMS_BITNET_STATUS_OK) break;
            }
            const auto end = std::chrono::steady_clock::now();
            if (status != CELIUMS_BITNET_STATUS_OK) break;
            const double seconds = std::chrono::duration<double>(end - start).count();
            decode_rates.push_back(decoded / seconds);
        }
    }

    if (status == CELIUMS_BITNET_STATUS_OK && !prompt_rates.empty()) {
        printf("{\"build_commit\":\"%s\",\"runtime_version\":\"%s\",\"test\":\"pp%d\","
               "\"n_prompt\":%d,\"n_gen\":0,\"n_batch\":%d,\"n_ubatch\":%d,"
               "\"n_threads\":%d,\"avg_ts\":%.6f}\n",
               celiums_bitnet_engine_commit(), celiums_bitnet_version(), benchmark.prompt_tokens,
               benchmark.prompt_tokens, benchmark.batch, benchmark.ubatch, benchmark.threads, average(prompt_rates));
    }
    if (status == CELIUMS_BITNET_STATUS_OK && !decode_rates.empty()) {
        printf("{\"build_commit\":\"%s\",\"runtime_version\":\"%s\",\"test\":\"tg%d\","
               "\"n_prompt\":0,\"n_gen\":%d,\"n_batch\":%d,\"n_ubatch\":%d,"
               "\"n_threads\":%d,\"avg_ts\":%.6f}\n",
               celiums_bitnet_engine_commit(), celiums_bitnet_version(), benchmark.generated_tokens,
               benchmark.generated_tokens, benchmark.batch, benchmark.ubatch, benchmark.threads, average(decode_rates));
    }

    celiums_bitnet_session_destroy(session);
    celiums_bitnet_model_destroy(model);
    celiums_bitnet_runtime_destroy(runtime);
    return status == CELIUMS_BITNET_STATUS_OK ? 0 : 1;
}
