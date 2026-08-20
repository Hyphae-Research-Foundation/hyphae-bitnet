#include "arg.h"
#include "common.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
    common_params params;
    std::vector<std::string> args = {
        "test-celiums-hybrid", "--model", "unused.gguf", "--celiums-hybrid-auto",
    };
    std::vector<char *> argv;
    for (auto & arg : args) {
        argv.push_back(arg.data());
    }

    assert(common_params_parse(argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads > 0);
    assert(params.cpuparams_batch.n_threads > 0);
    assert(params.cpuparams.n_threads <= params.cpuparams_batch.n_threads);
    assert(params.cpuparams.mask_valid);
    assert(params.cpuparams_batch.mask_valid);
    assert(params.cpuparams.strict_cpu);

    return 0;
}
