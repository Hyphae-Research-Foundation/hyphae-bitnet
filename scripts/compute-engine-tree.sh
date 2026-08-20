#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
temp_index=$(mktemp)
trap 'rm -f "$temp_index"' EXIT

GIT_INDEX_FILE="$temp_index" git -C "$root" read-tree HEAD
GIT_INDEX_FILE="$temp_index" git -C "$root" add -A -- 3rdparty/llama.cpp
tree=$(GIT_INDEX_FILE="$temp_index" git -C "$root" write-tree)
git -C "$root" ls-tree "$tree:3rdparty" llama.cpp | cut -d' ' -f3 | cut -f1
