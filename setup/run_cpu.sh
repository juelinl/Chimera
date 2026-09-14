#!/usr/bin/env bash
# Run one query at a time with eight workers on one NUMA node.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
command -v numactl >/dev/null 2>&1 || {
    echo "numactl is required by this launcher; install it or run build-cpu/cpu_search directly." >&2
    exit 1
}
exec numactl --cpunodebind="${CHIMERA_CPU_NUMA_NODE:-0}"     --membind="${CHIMERA_CPU_NUMA_NODE:-0}"     "$repo_root/build-cpu/cpu_search" "$@"
