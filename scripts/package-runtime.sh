#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
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
if [[ $(cd tools/runtime-gateway && rustc --version | cut -d' ' -f2) != 1.97.1 ]]; then
  printf 'Rust 1.97.1 is required for gateway release packaging\n' >&2
  exit 2
fi
if [[ $(ldd --version 2>&1 | sed -n '1{s/.* \([0-9][0-9.]*\)$/\1/p;q}') != 2.39 ]]; then
  printf 'Official Linux release packaging requires a glibc 2.39 builder (Ubuntu 24.04 baseline)\n' >&2
  exit 2
fi

profile=${1:-native}
build_dir=${2:-build-${profile}}
stage_dir=${3:-stage-${profile}}
version=$(tr -d '[:space:]' < VERSION)
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  printf 'Release packaging is supported only on Linux x86_64\n' >&2
  exit 2
fi
case "$profile" in
  native|avx2|scalar) ;;
  *) printf 'Unknown CPU profile: %s\n' "$profile" >&2; exit 2 ;;
esac
archive=${4:-hyphae-bitnet-runtime-${version}-linux-x86_64-${profile}.tar.gz}

build_dir=$(realpath -m "$build_dir")
stage_dir=$(realpath -m "$stage_dir")
archive=$(realpath -m "$archive")
case "$stage_dir" in
  /|"$root")
    printf 'Stage directory must be outside the repository: %s\n' "$stage_dir" >&2
    exit 2
    ;;
esac
if [[ "$build_dir" == "$root" || "$stage_dir" == "$root" || "$archive" == "$root" ]]; then
  printf 'Build, stage, and archive paths must not be the repository root\n' >&2
  exit 2
fi
if [[ "$build_dir" == / ]]; then
  printf 'Build directory must not be the filesystem root\n' >&2
  exit 2
fi
case "$(basename "$stage_dir")" in
  stage-*|*-stage) ;;
  *) printf 'Stage directory must have a stage-* or *-stage basename\n' >&2; exit 2 ;;
esac
if [[ -e "$build_dir" ]]; then
  printf 'Build directory must not already exist: %s\n' "$build_dir" >&2
  exit 2
fi
if [[ $(scripts/compute-engine-tree.sh) != $(tr -d '[:space:]' < cmake/ENGINE_TREE) ]]; then
  printf 'Vendored engine tree does not match cmake/ENGINE_TREE\n' >&2
  exit 2
fi

source_date_epoch=${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}

"$cmake" -S . -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCELIUMS_BITNET_BUILD_SERVER=ON \
  -DCELIUMS_BITNET_BUILD_GATEWAY=ON \
  -DCELIUMS_BITNET_CPU_PROFILE="$profile" \
  -DCELIUMS_BITNET_INSTALL_COMPAT=OFF \
  -DCELIUMS_BITNET_VERIFY_ENGINE_TREE=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr

"$cmake" --build "$build_dir" --parallel "${JOBS:-2}" --target \
  celiums-bitnet celiums-runtime-bench celiums-runtime-server \
  celiums-runtime-gateway-binaries test-celiums-runtime-api test-celiums-runtime-session

rm -rf "$stage_dir"
DESTDIR="$stage_dir" "$cmake" --install "$build_dir"
tar -C "$stage_dir" --sort=name --owner=0 --group=0 --numeric-owner \
  --mtime="@$source_date_epoch" -czf "$archive" .
dirty_args=()
if [[ ${ALLOW_DIRTY:-0} == 1 ]]; then
  dirty_args+=(--allow-dirty)
fi
manifest_command=(
  "$python" utils/release_manifest.py
  --profile "$profile"
  --compiler "$(sed -n 's/^CMAKE_C_COMPILER:FILEPATH=//p' "$build_dir/CMakeCache.txt")"
  --cxx-compiler "$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$build_dir/CMakeCache.txt")"
  --rustc "$(cd tools/runtime-gateway && rustc --version)"
  --cargo-lock tools/runtime-gateway/Cargo.lock
  --hyphae-commit 0471ae25b263fd506da1578068ec57429a6783de
  --artifact "$archive"
  --output "${archive}.json"
)
manifest_command+=("${dirty_args[@]}")
"${manifest_command[@]}"
