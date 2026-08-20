#include "celiums/bitnet_runtime.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    celiums_bitnet_runtime_options options = celiums_bitnet_runtime_default_options();
    celiums_bitnet_runtime * runtime = NULL;

    if (options.struct_size != sizeof(options) ||
            options.api_version != CELIUMS_BITNET_API_VERSION ||
            strcmp(celiums_bitnet_version(), CELIUMS_BITNET_TEST_VERSION) != 0 ||
            strlen(celiums_bitnet_product_commit()) != 9 ||
            celiums_bitnet_runtime_create(&options, &runtime) != CELIUMS_BITNET_STATUS_OK ||
            runtime == NULL) {
        fprintf(stderr, "Celiums Runtime C API smoke test failed\n");
        return 1;
    }

    celiums_bitnet_runtime_destroy(runtime);
    return 0;
}
