#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$repo_root" -B "$repo_root/build-cpu" \
    -DCHIMERA_BUILD_GPU=OFF -DCHIMERA_BUILD_CPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$repo_root/build-cpu" -j "${CHIMERA_BUILD_JOBS:-4}"
ctest --test-dir "$repo_root/build-cpu" --output-on-failure
