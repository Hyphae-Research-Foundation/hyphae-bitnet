#!/usr/bin/env bash
set -euo pipefail

profile=${1:-native}
build_dir=${2:-build-${profile}}
stage_dir=${3:-stage-${profile}}
archive=${4:-celiums-bitnet-runtime-0.2.0-linux-x86_64-${profile}.tar.gz}

cmake -S . -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCELIUMS_BITNET_BUILD_SERVER=ON \
  -DCELIUMS_BITNET_CPU_PROFILE="$profile" \
  -DCELIUMS_BITNET_INSTALL_COMPAT=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr

cmake --build "$build_dir" --parallel "${JOBS:-2}" --target \
  celiums-bitnet celiums-runtime-bench \
  test-celiums-runtime-api test-celiums-runtime-session

rm -rf "$stage_dir"
DESTDIR="$PWD/$stage_dir" cmake --install "$build_dir"
tar -C "$stage_dir" -czf "$archive" .
dirty_args=()
if [[ ${ALLOW_DIRTY:-0} == 1 ]]; then
  dirty_args+=(--allow-dirty)
fi
manifest_command=(
  python utils/release_manifest.py
  --profile "$profile"
  --compiler "$(cc --version | sed -n '1p')"
  --artifact "$archive"
  --output "${archive}.json"
)
manifest_command+=("${dirty_args[@]}")
"${manifest_command[@]}"
