#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/repack.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

enum class scenario {
    gap,
    overlap,
    replacement,
};

struct run_result {
    std::vector<float> first;
    std::vector<float> second;
};

static uint32_t hash_bytes(const std::vector<float> & values) {
    uint32_t hash = 2166136261u;
    const uint8_t * bytes = (const uint8_t *) values.data();
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash = (hash ^ bytes[i]) * 16777619u;
    }
    return hash;
}

static ggml_backend_buffer_t alloc_tensor(ggml_tensor * tensor, ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(tensor) + 64);
    GGML_ASSERT(buffer != nullptr);
    tensor->data = ggml_backend_buffer_get_base(buffer);
    tensor->buffer = buffer;
    GGML_ASSERT(ggml_backend_buffer_init_tensor(buffer, tensor) == GGML_STATUS_SUCCESS);
    return buffer;
}

static run_result run_graph(scenario which, int64_t n, bool use_plan, int threads) {
    constexpr int64_t k = 256;
    constexpr int64_t m = 8;
    const int n_projections = which == scenario::replacement ? 4 : 2;

    ggml_context * ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    std::vector<ggml_tensor *> weights;
    std::vector<ggml_tensor *> outputs;
    for (int i = 0; i < n_projections; ++i) {
        weights.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_Q1_0, k, m));
    }
    ggml_tensor * x0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * x1 = which == scenario::replacement ?
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n) : nullptr;
    ggml_tensor * gap_input = which == scenario::gap ?
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, n) : nullptr;

    for (int i = 0; i < n_projections; ++i) {
        ggml_tensor * source = which == scenario::replacement && i == 1 ? x1 : x0;
        outputs.push_back(ggml_mul_mat(ctx, weights[i], source));
    }

    ggml_tensor * result;
    if (which == scenario::replacement) {
        result = outputs[0];
        for (int i = 1; i < n_projections; ++i) {
            result = ggml_add(ctx, result, outputs[i]);
        }
    } else {
        result = ggml_add(ctx, outputs[0], outputs[1]);
    }

    std::vector<ggml_backend_buffer_t> buffers;
    const ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();
    for (ggml_tensor * weight : weights) {
        buffers.push_back(alloc_tensor(weight, repack_buft));
    }
    buffers.push_back(alloc_tensor(x0, ggml_backend_cpu_buffer_type()));
    if (x1 != nullptr) {
        buffers.push_back(alloc_tensor(x1, ggml_backend_cpu_buffer_type()));
    }
    if (gap_input != nullptr) {
        buffers.push_back(alloc_tensor(gap_input, ggml_backend_cpu_buffer_type()));
    }
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (tensor->data == nullptr && tensor->view_src == nullptr && tensor->op != GGML_OP_NONE) {
            buffers.push_back(alloc_tensor(tensor, ggml_backend_cpu_buffer_type()));
        }
    }

    std::vector<float> wf(k * m);
    std::vector<uint8_t> wq(ggml_nbytes(weights[0]));
    for (int weight = 0; weight < n_projections; ++weight) {
        for (int64_t i = 0; i < k * m; ++i) {
            wf[i] = ((i + 3 * weight) % 5 < 2 ? -1.0f : 1.0f) * (0.25f + 0.01f * (i % 13));
        }
        quantize_row_q1_0_ref(wf.data(), (block_q1_0 *) wq.data(), k * m);
        ggml_backend_tensor_set(weights[weight], wq.data(), 0, wq.size());
    }

    std::vector<float> xv0(k * n);
    std::vector<float> xv1(k * n);
    for (int64_t i = 0; i < k * n; ++i) {
        xv0[i] = std::sin((float) i * 0.17f) + 0.01f * (float) (i % 7);
        xv1[i] = std::cos((float) i * 0.11f) - 0.02f * (float) (i % 5);
    }
    ggml_backend_tensor_set(x0, xv0.data(), 0, ggml_nbytes(x0));
    if (x1 != nullptr) {
        ggml_backend_tensor_set(x1, xv1.data(), 0, ggml_nbytes(x1));
    }
    if (gap_input != nullptr) {
        std::vector<float> gap_values(m * n);
        for (int64_t i = 0; i < m * n; ++i) {
            gap_values[i] = 0.5f + 0.25f * (float) i;
        }
        ggml_backend_tensor_set(gap_input, gap_values.data(), 0, ggml_nbytes(gap_input));
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    if (which == scenario::gap || which == scenario::overlap) {
        int first_projection = -1;
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i] == outputs[0]) {
                first_projection = i;
                break;
            }
        }
        GGML_ASSERT(first_projection >= 0 && graph->n_nodes < graph->size);
        ggml_tensor * interposed;
        if (which == scenario::gap) {
            interposed = ggml_sqr(ctx, gap_input);
            buffers.push_back(alloc_tensor(interposed, ggml_backend_cpu_buffer_type()));
        } else {
            interposed = ggml_scale_inplace(ctx, x0, 0.5f);
            interposed->data = x0->data;
            interposed->buffer = x0->buffer;
        }
        for (int i = graph->n_nodes; i > first_projection + 1; --i) {
            graph->nodes[i] = graph->nodes[i - 1];
        }
        interposed->flags |= GGML_TENSOR_FLAG_COMPUTE;
        graph->nodes[first_projection + 1] = interposed;
        graph->n_nodes += 1;
        if (which == scenario::overlap) {
            GGML_ASSERT(interposed->data == x0->data && interposed->buffer == x0->buffer);
        }
    }

    std::vector<ggml_tensor *> eligible_sources;
    std::vector<int> eligible_nodes;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_Q1_0) {
            eligible_sources.push_back(node->src[1]);
            eligible_nodes.push_back(i);
        }
    }
    GGML_ASSERT((int) eligible_sources.size() == n_projections);
    if (which == scenario::gap || which == scenario::overlap) {
        GGML_ASSERT(eligible_sources[0] == x0 && eligible_sources[1] == x0);
        GGML_ASSERT(eligible_nodes[1] > eligible_nodes[0] + 1);
    } else {
        GGML_ASSERT(eligible_sources[0] == x0 && eligible_sources[1] == x1 &&
                    eligible_sources[2] == x0 && eligible_sources[3] == x0);
    }

    ggml_backend_t backend = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_backend_graph_plan_t backend_plan = nullptr;
    ggml_cplan cplan = {};
    std::vector<uint8_t> work;
    if (use_plan) {
        backend = ggml_backend_cpu_init();
        ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
        threadpool = ggml_threadpool_new(&tpp);
        ggml_backend_cpu_set_n_threads(backend, threads);
        ggml_backend_cpu_set_threadpool(backend, threadpool);
        backend_plan = ggml_backend_graph_plan_create(backend, graph);
    } else {
        cplan = ggml_graph_plan(graph, threads, nullptr);
        work.resize(cplan.work_size);
        cplan.work_data = work.data();
    }

    const ggml_status status = use_plan ? ggml_backend_graph_plan_compute(backend, backend_plan) :
                                          ggml_graph_compute(graph, &cplan);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    std::vector<float> first(ggml_nelements(result));
    ggml_backend_tensor_get(result, first.data(), 0, ggml_nbytes(result));

    for (float & value : xv0) {
        value = which == scenario::overlap ? value + 0.25f : value + 0.375f;
    }
    ggml_backend_tensor_set(x0, xv0.data(), 0, ggml_nbytes(x0));
    const ggml_status second_status = use_plan ? ggml_backend_graph_plan_compute(backend, backend_plan) :
                                                 ggml_graph_compute(graph, &cplan);
    GGML_ASSERT(second_status == GGML_STATUS_SUCCESS);
    std::vector<float> second(ggml_nelements(result));
    ggml_backend_tensor_get(result, second.data(), 0, ggml_nbytes(result));
    GGML_ASSERT(std::memcmp(first.data(), second.data(), ggml_nbytes(result)) != 0);

    if (use_plan) {
        ggml_backend_graph_plan_free(backend, backend_plan);
        ggml_backend_free(backend);
        ggml_threadpool_free(threadpool);
    }
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
        ggml_backend_buffer_free(*it);
    }
    ggml_free(ctx);
    return { std::move(first), std::move(second) };
}

static std::string run_child(const char * self, bool cache_enabled, bool debug, int repeats) {
    char command[4096];
    const int written = std::snprintf(command, sizeof(command),
        "GGML_Q1_ACT_CACHE=%d GGML_Q1_ACT_CACHE_DEBUG=%d '%s' --child %d 2>&1",
        cache_enabled ? 1 : 0, debug ? 1 : 0, self, repeats);
    GGML_ASSERT(written > 0 && (size_t) written < sizeof(command));

    FILE * pipe = popen(command, "r");
    GGML_ASSERT(pipe != nullptr);
    std::string output;
    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    GGML_ASSERT(pclose(pipe) == 0);
    return output;
}

static std::string hash_line(const std::string & output) {
    const size_t begin = output.find("q1-cache-hash ");
    GGML_ASSERT(begin != std::string::npos);
    const size_t end = output.find('\n', begin);
    return output.substr(begin, end - begin);
}

static void check_equal(const run_result & expected, const run_result & actual) {
    GGML_ASSERT(expected.first.size() == actual.first.size());
    GGML_ASSERT(std::memcmp(expected.first.data(), actual.first.data(), expected.first.size() * sizeof(float)) == 0);
    GGML_ASSERT(std::memcmp(expected.second.data(), actual.second.data(), expected.second.size() * sizeof(float)) == 0);
}

static size_t count_substring(const std::string & output, const char * needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = output.find(needle, pos)) != std::string::npos) {
        count += 1;
        pos += std::strlen(needle);
    }
    return count;
}

static run_result run_scenario(scenario which, int64_t n) {
    const run_result expected = run_graph(which, n, false, 1);
    check_equal(expected, run_graph(which, n, false, 8));
    check_equal(expected, run_graph(which, n, true, 1));
    check_equal(expected, run_graph(which, n, true, 8));
    return expected;
}

int main(int argc, char ** argv) {
    if (argc == 1) {
        const std::string cached = run_child(argv[0], true, true, 1);
        const std::string uncached = run_child(argv[0], false, true, 1);
        constexpr size_t expected_counter_lines = 3 * 4 * 2;
        GGML_ASSERT(count_substring(cached, "Q1 activation cache: hits=1 misses=1\n") == expected_counter_lines);
        GGML_ASSERT(count_substring(cached, "Q1 activation cache: hits=0 misses=2\n") == expected_counter_lines);
        GGML_ASSERT(count_substring(cached, "Q1 activation cache: hits=1 misses=3\n") == expected_counter_lines);
        GGML_ASSERT(uncached.find("Q1 activation cache:") == std::string::npos);
        GGML_ASSERT(hash_line(cached) == hash_line(uncached));
        std::printf("%s\n", hash_line(cached).c_str());
        std::printf("Q1 activation cache decode/GEMM gaps/replacement, cached/uncached, direct/plan bitwise: ok\n");
        return 0;
    }

    GGML_ASSERT(argc == 3 && std::strcmp(argv[1], "--child") == 0);
    const int repeats = std::atoi(argv[2]);
    GGML_ASSERT(repeats > 0);
    // Internal test seam for x86 hosts without the production AVX-512 Q1 trait.
    setenv("GGML_Q1_REPACK_GENERIC_TEST", "1", 1);
    const int64_t activation_rows[] = { 1, 4, 8 };
    run_result gap[3];
    run_result overlap[3];
    run_result replacement[3];
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int i = 0; i < 3; ++i) {
            gap[i] = run_scenario(scenario::gap, activation_rows[i]);
            overlap[i] = run_scenario(scenario::overlap, activation_rows[i]);
            replacement[i] = run_scenario(scenario::replacement, activation_rows[i]);
        }
    }

    std::printf("q1-cache-hash");
    for (int i = 0; i < 3; ++i) {
        std::printf(" n%lld=%08x/%08x,%08x/%08x,%08x/%08x",
                    (long long) activation_rows[i],
                    hash_bytes(gap[i].first), hash_bytes(gap[i].second),
                    hash_bytes(overlap[i].first), hash_bytes(overlap[i].second),
                    hash_bytes(replacement[i].first), hash_bytes(replacement[i].second));
    }
    std::printf("\n");
    return 0;
}
