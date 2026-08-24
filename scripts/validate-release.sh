#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
model=${CELIUMS_BITNET_TEST_MODEL:-$root/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf}
oracle=${CELIUMS_BITNET_TEST_ORACLE:-$root/captures/runtime-reference.gguf}
oracle_sidecar=${CELIUMS_BITNET_TEST_ORACLE_SIDECAR:-$oracle.json}
model_sha256=${CELIUMS_BITNET_MODEL_SHA256:-e23b16fa81b890e8b65e676262b645e8ffa5ae1f6df89dadaf793246826bbd90}
jobs=${JOBS:-2}
release_root=${BUILD_ROOT:-/tmp/celiums-bitnet-release}
cmake=${CMAKE:-$(command -v cmake || true)}
python=${PYTHON:-$(command -v python3 || command -v python || true)}
if [[ -z "$cmake" ]]; then
  printf 'cmake is required (or set CMAKE)\n' >&2
  exit 2
fi
if [[ -z "$python" ]]; then
  printf 'python3 is required (or set PYTHON)\n' >&2
  exit 2
fi
if ! command -v cargo >/dev/null 2>&1 || ! command -v cargo-about >/dev/null 2>&1; then
  printf 'cargo and cargo-about 0.9.2 are required for the Celiums Runtime Gateway release gate\n' >&2
  exit 2
fi
if [[ $(cd tools/runtime-gateway && rustc --version | cut -d' ' -f2) != 1.97.1 || $(cargo-about --version | cut -d' ' -f2) != 0.9.2 ]]; then
  printf 'Rust 1.97.1 and cargo-about 0.9.2 are required for the release gate\n' >&2
  exit 2
fi
if [[ $(ldd --version 2>&1 | sed -n '1{s/.* \([0-9][0-9.]*\)$/\1/p;q}') != 2.39 ]]; then
  printf 'Official release validation requires a glibc 2.39 builder (Ubuntu 24.04 baseline)\n' >&2
  exit 2
fi
if [[ "$release_root" == / || "$release_root" == "$root" ]]; then
  printf 'BUILD_ROOT must not be the filesystem or repository root\n' >&2
  exit 2
fi
rm -rf "$release_root"
if [[ $(scripts/compute-engine-tree.sh) != $(tr -d '[:space:]' < cmake/ENGINE_TREE) ]]; then
  printf 'Vendored engine tree does not match cmake/ENGINE_TREE\n' >&2
  exit 2
fi
if [[ ! -f "$model" ]]; then
  printf 'Model-backed release tests cannot run; fixture not found: %s\n' "$model" >&2
  exit 2
fi
if [[ ! -f "$oracle" || ! -f "$oracle_sidecar" ]]; then
  printf 'Exactness oracle or sidecar not found: %s / %s\n' "$oracle" "$oracle_sidecar" >&2
  exit 2
fi

for profile in native avx2 scalar; do
  build=$release_root/build-$profile
  install=$release_root/install-$profile
  "$cmake" -S "$root" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DCELIUMS_BITNET_BUILD_SERVER=ON \
    -DCELIUMS_BITNET_BUILD_GATEWAY=ON \
    -DCELIUMS_BITNET_CPU_PROFILE="$profile" \
    -DCELIUMS_BITNET_TEST_MODEL="$model" \
    -DCELIUMS_BITNET_INSTALL_COMPAT=OFF \
    -DCELIUMS_BITNET_VERIFY_ENGINE_TREE=OFF \
    -DCMAKE_INSTALL_PREFIX="$install"
  "$cmake" --build "$build" --parallel "$jobs" --target \
    celiums-bitnet celiums-runtime-bench celiums-runtime-server \
    celiums-runtime-gateway-binaries \
    celiums-logits-capture test-celiums-runtime-api test-celiums-runtime-v0.3.0-client \
    test-celiums-runtime-session \
    test-quantize-fns test-i2s-mul-mat test-celiums-hybrid
  "${CTEST:-${cmake%/cmake}/ctest}" --test-dir "$build" --output-on-failure --no-tests=error \
    -R 'test-(celiums-runtime-api|celiums-runtime-v0.3.0-client|celiums-runtime-v0.3.0-model|celiums-runtime-session|quantize-fns|i2s-mul-mat|celiums-hybrid)$'
  "$cmake" --install "$build"
  CELIUMS_BITNET_TEST_BUILD_DIR="$build" \
  CELIUMS_BITNET_TEST_INSTALL_PREFIX="$install" \
  "$python" -m unittest tests.test_runtime_product -v
done

(
  cd tools/runtime-gateway
  CARGO_INCREMENTAL=0 cargo fmt --all -- --check
  CARGO_INCREMENTAL=0 cargo clippy --locked --all-targets -- -D warnings
  CARGO_INCREMENTAL=0 cargo test --locked --all-targets
  cargo about generate --locked about.hbs --output-file "$release_root/THIRD_PARTY_LICENSES.raw.html" --fail
  perl -pe 's/[ \t\r]+$//' "$release_root/THIRD_PARTY_LICENSES.raw.html" > "$release_root/THIRD_PARTY_LICENSES.html"
  cmp THIRD_PARTY_LICENSES.html "$release_root/THIRD_PARTY_LICENSES.html"
)

"$python" -m unittest \
  tests.test_i2s_conversion tests.test_logits_comparison tests.test_wrappers -v

CELIUMS_BITNET_TEST_BUILD_DIR="$release_root/build-native" \
CELIUMS_BITNET_TEST_MODEL="$model" \
"$python" -m unittest \
  tests.test_runtime_product.RuntimeProductTests.test_native_run_generates_expected_greedy_prefix \
  tests.test_runtime_product.RuntimeProductTests.test_native_benchmark_reports_prefill_and_decode \
  tests.test_runtime_product.RuntimeProductTests.test_native_server_openai_completion_and_authentication -v

"$python" utils/compare_runtime_logits.py \
  --library "$release_root/build-native/src/libceliums-bitnet-runtime.so" \
  --model "$model" \
  --reference "$oracle" \
  --reference-sidecar "$oracle_sidecar" \
  --expected-model-sha256 "$model_sha256" \
  --require-bitwise
