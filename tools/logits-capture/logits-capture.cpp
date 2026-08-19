#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct probe_selector {
    std::string name;
    std::string op;
};

struct custom_options {
    std::vector<int32_t> positions;
    std::vector<probe_selector> probes;
};

struct probe_capture {
    std::string name;
    std::string op;
    ggml_type type;
    int32_t n_dims;
    std::array<int64_t, GGML_MAX_DIMS> ne;
    int32_t decode_start;
    int32_t decode_tokens;
    std::vector<uint8_t> data;
};

struct probe_state {
    std::vector<probe_selector> selectors;
    std::vector<probe_capture> captures;
    std::string error;
    int32_t decode_start = 0;
    int32_t decode_tokens = 0;
};

static probe_selector parse_probe_selector(const std::string & value) {
    const size_t separator = value.rfind('@');
    probe_selector selector;
    selector.name = value.substr(0, separator);
    if (separator != std::string::npos) {
        selector.op = value.substr(separator + 1);
    }
    if (selector.name.empty() || (separator != std::string::npos && selector.op.empty())) {
        throw std::invalid_argument("--capture-tensor requires NAME or NAME@OP");
    }
    return selector;
}

static custom_options parse_custom_options(int & argc, char ** argv) {
    custom_options options;
    int write = 1;
    for (int read = 1; read < argc; ++read) {
        const std::string arg = argv[read];
        if (arg.rfind("--capture-position=", 0) == 0) {
            options.positions.push_back(std::stoi(arg.substr(std::strlen("--capture-position="))));
            continue;
        }
        if (arg == "--capture-position") {
            if (++read >= argc) {
                throw std::invalid_argument("--capture-position requires an integer");
            }
            options.positions.push_back(std::stoi(argv[read]));
            continue;
        }
        if (arg.rfind("--capture-tensor=", 0) == 0) {
            options.probes.push_back(parse_probe_selector(arg.substr(std::strlen("--capture-tensor="))));
            continue;
        }
        if (arg == "--capture-tensor") {
            if (++read >= argc) {
                throw std::invalid_argument("--capture-tensor requires NAME or NAME@OP");
            }
            options.probes.push_back(parse_probe_selector(argv[read]));
            continue;
        }
        argv[write++] = argv[read];
    }
    argc = write;

    std::unordered_set<std::string> seen;
    options.probes.erase(std::remove_if(options.probes.begin(), options.probes.end(), [&](const probe_selector & probe) {
        return !seen.insert(probe.name + "@" + probe.op).second;
    }), options.probes.end());
    return options;
}

static std::vector<int32_t> normalize_positions(std::vector<int32_t> positions, int32_t n_tokens) {
    if (positions.empty()) {
        positions.push_back(n_tokens - 1);
    }
    for (int32_t & position : positions) {
        if (position < 0) {
            position += n_tokens;
        }
        if (position < 0 || position >= n_tokens) {
            throw std::invalid_argument("capture position is outside the tokenized prompt");
        }
    }
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    return positions;
}

static bool probe_matches(const probe_state & state, const ggml_tensor * tensor) {
    const std::string name = ggml_get_name(tensor);
    const std::string op = ggml_op_name(tensor->op);
    return std::any_of(state.selectors.begin(), state.selectors.end(), [&](const probe_selector & selector) {
        return selector.name == name && (selector.op.empty() || selector.op == op);
    });
}

static bool capture_probe_callback(ggml_tensor * tensor, bool ask, void * user_data) noexcept {
    auto & state = *static_cast<probe_state *>(user_data);
    if (!state.error.empty()) {
        return !ask;
    }
    const bool selected = probe_matches(state, tensor);
    if (ask) {
        return selected;
    }
    if (!selected) {
        return true;
    }

    try {
        if (!ggml_is_contiguous(tensor)) {
            throw std::runtime_error("selected probe is not contiguous: " + std::string(ggml_get_name(tensor)));
        }
        probe_capture capture;
        capture.name = ggml_get_name(tensor);
        capture.op = ggml_op_name(tensor->op);
        capture.type = tensor->type;
        capture.n_dims = ggml_n_dims(tensor);
        std::copy(std::begin(tensor->ne), std::end(tensor->ne), capture.ne.begin());
        capture.decode_start = state.decode_start;
        capture.decode_tokens = state.decode_tokens;
        capture.data.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, capture.data.data(), 0, capture.data.size());
        state.captures.push_back(std::move(capture));
    } catch (const std::exception & error) {
        state.error = error.what();
    } catch (...) {
        state.error = "unknown probe capture failure";
    }
    return true;
}

static std::vector<float> evaluate_logits(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        const std::vector<int32_t> & positions,
        int32_t n_vocab,
        probe_state * probes) {
    const int32_t n_batch = llama_n_batch(ctx);
    if (n_batch <= 0) {
        throw std::runtime_error("invalid context batch size");
    }

    std::vector<float> logits(positions.size() * (size_t) n_vocab);
    size_t selected = 0;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (int32_t start = 0; start < (int32_t) tokens.size(); start += n_batch) {
        const int32_t end = std::min<int32_t>(start + n_batch, tokens.size());
        const size_t selected_begin = selected;
        common_batch_clear(batch);
        for (int32_t position = start; position < end; ++position) {
            const bool want_logits = selected < positions.size() && positions[selected] == position;
            common_batch_add(batch, tokens[position], position, { 0 }, want_logits);
            if (want_logits) {
                ++selected;
            }
        }

        if (probes) {
            probes->decode_start = start;
            probes->decode_tokens = end - start;
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed");
        }
        if (probes && !probes->error.empty()) {
            llama_batch_free(batch);
            throw std::runtime_error(probes->error);
        }

        size_t output = selected_begin;
        for (int32_t local = 0; local < batch.n_tokens; ++local) {
            if (!batch.logits[local]) {
                continue;
            }
            const float * row = llama_get_logits_ith(ctx, local);
            if (!row) {
                llama_batch_free(batch);
                throw std::runtime_error("requested logits are unavailable");
            }
            std::memcpy(logits.data() + output * n_vocab, row, n_vocab * sizeof(float));
            ++output;
        }
    }

    llama_batch_free(batch);
    if (selected != positions.size()) {
        throw std::runtime_error("not all requested logits were captured");
    }
    return logits;
}

static void write_capture(
        const common_params & params,
        const llama_model * model,
        const llama_context * ctx,
        const std::vector<llama_token> & tokens,
        const std::vector<int32_t> & positions,
        const std::vector<float> & logits,
        int32_t n_vocab,
        const std::vector<probe_capture> & probes) {
    const size_t tensor_bytes =
        tokens.size() * sizeof(int32_t) +
        positions.size() * sizeof(int32_t) +
        logits.size() * sizeof(float) +
        std::accumulate(probes.begin(), probes.end(), (size_t) 0, [](size_t sum, const probe_capture & probe) {
            return sum + probe.data.size();
        });
    const size_t tensor_overhead = (3 + probes.size()) * ggml_tensor_overhead();
    ggml_context_ptr tensor_ctx(ggml_init({ tensor_bytes + tensor_overhead + 4096, nullptr, false }));
    if (!tensor_ctx) {
        throw std::runtime_error("failed to allocate capture tensors");
    }

    gguf_context_ptr capture(gguf_init_empty());
    gguf_set_val_str(capture.get(), "general.type", "celiums-logits-capture");
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.version", 2);
    gguf_set_val_str(capture.get(), "celiums.logits_capture.model.path", params.model.path.c_str());
    gguf_set_val_str(capture.get(), "celiums.logits_capture.prompt", params.prompt.c_str());
    gguf_set_val_bool(capture.get(), "celiums.logits_capture.add_special", true);
    gguf_set_val_bool(capture.get(), "celiums.logits_capture.parse_special", false);
    gguf_set_val_bool(capture.get(), "celiums.logits_capture.escape", params.escape);
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_vocab", n_vocab);
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_tokens", tokens.size());
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_positions", positions.size());
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_ctx", llama_n_ctx(ctx));
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_batch", llama_n_batch(ctx));
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_ubatch", llama_n_ubatch(ctx));
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_threads", params.cpuparams.n_threads);
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_threads_batch", params.cpuparams_batch.n_threads);
    gguf_set_val_u64(capture.get(), "celiums.logits_capture.model.size", llama_model_size(model));
    gguf_set_val_u64(capture.get(), "celiums.logits_capture.model.parameters", llama_model_n_params(model));
    gguf_set_val_str(capture.get(), "celiums.logits_capture.build.commit", llama_commit());
    gguf_set_val_i32(capture.get(), "celiums.logits_capture.build.number", llama_build_number());
    gguf_set_val_str(capture.get(), "celiums.logits_capture.build.compiler", llama_compiler());
    gguf_set_val_str(capture.get(), "celiums.logits_capture.build.target", llama_build_target());
    const std::string system_info = common_params_get_system_info(params);
    gguf_set_val_str(capture.get(), "celiums.logits_capture.system_info", system_info.c_str());
    gguf_set_val_u32(capture.get(), "celiums.logits_capture.n_probes", probes.size());

    ggml_tensor * tokens_tensor = ggml_new_tensor_1d(tensor_ctx.get(), GGML_TYPE_I32, tokens.size());
    ggml_set_name(tokens_tensor, "tokens");
    std::memcpy(tokens_tensor->data, tokens.data(), tokens.size() * sizeof(int32_t));
    gguf_add_tensor(capture.get(), tokens_tensor);

    ggml_tensor * positions_tensor = ggml_new_tensor_1d(tensor_ctx.get(), GGML_TYPE_I32, positions.size());
    ggml_set_name(positions_tensor, "positions");
    std::memcpy(positions_tensor->data, positions.data(), positions.size() * sizeof(int32_t));
    gguf_add_tensor(capture.get(), positions_tensor);

    ggml_tensor * logits_tensor = positions.size() == 1
        ? ggml_new_tensor_1d(tensor_ctx.get(), GGML_TYPE_F32, n_vocab)
        : ggml_new_tensor_2d(tensor_ctx.get(), GGML_TYPE_F32, n_vocab, positions.size());
    ggml_set_name(logits_tensor, "logits");
    std::memcpy(logits_tensor->data, logits.data(), logits.size() * sizeof(float));
    gguf_add_tensor(capture.get(), logits_tensor);

    for (size_t index = 0; index < probes.size(); ++index) {
        const probe_capture & probe = probes[index];
        char tensor_name[32];
        snprintf(tensor_name, sizeof(tensor_name), "probe.%04zu", index);
        ggml_tensor * probe_tensor = ggml_new_tensor(
            tensor_ctx.get(), probe.type, probe.n_dims, probe.ne.data());
        ggml_set_name(probe_tensor, tensor_name);
        std::memcpy(probe_tensor->data, probe.data.data(), probe.data.size());
        gguf_add_tensor(capture.get(), probe_tensor);

        const std::string prefix = "celiums.logits_capture." + std::string(tensor_name);
        gguf_set_val_str(capture.get(), (prefix + ".name").c_str(), probe.name.c_str());
        gguf_set_val_str(capture.get(), (prefix + ".op").c_str(), probe.op.c_str());
        gguf_set_val_str(capture.get(), (prefix + ".type").c_str(), ggml_type_name(probe.type));
        gguf_set_val_i32(capture.get(), (prefix + ".decode_start").c_str(), probe.decode_start);
        gguf_set_val_i32(capture.get(), (prefix + ".decode_tokens").c_str(), probe.decode_tokens);
    }

    if (!gguf_write_to_file(capture.get(), params.out_file.c_str(), false)) {
        throw std::runtime_error("failed to write logits capture");
    }
}

int main(int argc, char ** argv) {
    const char * stage = "argument parsing";
    bool backend_initialized = false;
    try {
        custom_options custom = parse_custom_options(argc, argv);
        probe_state probes;
        probes.selectors = std::move(custom.probes);
        common_params params;
        params.escape = false;
        params.warmup = false;
        params.fit_params = false;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_RESULTS)) {
            return 1;
        }
        if (params.out_file.empty()) {
            throw std::invalid_argument("--output is required");
        }
        const auto model_path = std::filesystem::weakly_canonical(params.model.path);
        const auto output_path = std::filesystem::weakly_canonical(params.out_file);
        if (model_path == output_path ||
                (std::filesystem::exists(output_path) && std::filesystem::equivalent(model_path, output_path))) {
            throw std::invalid_argument("--output must not alias --model");
        }
        params.sampling.backend_sampling = false;
        params.embedding = false;
        if (!probes.selectors.empty()) {
            params.cb_eval = capture_probe_callback;
            params.cb_eval_user_data = &probes;
        }

        stage = "backend initialization";
        llama_backend_init();
        backend_initialized = true;
        llama_numa_init(params.numa);
        {
            stage = "model initialization";
            common_init_result_ptr init = common_init_from_params(params);
            llama_model * model = init->model();
            llama_context * ctx = init->context();
            if (!model || !ctx) {
                throw std::runtime_error("failed to initialize model and context");
            }

            const llama_vocab * vocab = llama_model_get_vocab(model);
            stage = "tokenization";
            const std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true, false);
            if (tokens.empty()) {
                throw std::invalid_argument("the tokenized prompt is empty");
            }
            if (tokens.size() > llama_n_ctx(ctx)) {
                throw std::invalid_argument("the tokenized prompt exceeds the context size");
            }

            const std::vector<int32_t> positions = normalize_positions(
                custom.positions, tokens.size());
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            stage = "logits evaluation";
            const std::vector<float> logits = evaluate_logits(
                ctx, tokens, positions, n_vocab, probes.selectors.empty() ? nullptr : &probes);
            stage = "capture serialization";
            write_capture(params, model, ctx, tokens, positions, logits, n_vocab, probes.captures);
        }
        llama_backend_free();
        backend_initialized = false;
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "celiums-logits-capture: %s: %s\n", stage, error.what());
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
