#!/bin/bash
# Measure sequential DRAM bandwidth. Default: 8 GiB arrays, 1/24/48/96 threads.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GIB="${1:-8}"
REPS="${2:-5}"
OUT="${3:-}"
gcc -O3 -fopenmp -o /tmp/measure-dram "$ROOT/measure-dram.c"
THREADS_LIST="${THREADS:-1 24 48 96}"
{
  echo "# $(uname -a)"
  echo "# nproc=$(nproc)  mem=$(awk '/MemTotal/{printf "%.0fGiB", $2/1024/1024}' /proc/meminfo)"
  lscpu | grep -E 'Model name|CPU\(s\)|Thread|NUMA'
  echo
  for t in $THREADS_LIST; do
    echo "======== threads=$t ========"
    OMP_PROC_BIND=close OMP_PLACES=cores /tmp/measure-dram "$GIB" "$t" "$REPS"
    echo
  done
} | tee ${OUT:+"$OUT"}
