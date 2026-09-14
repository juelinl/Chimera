# Optimized CPU search

## Build and launch

Requires 64-bit Linux, AVX-512 F/BW/DQ/VL, CMake 3.24+, and a C++17 compiler
with OpenMP (tested with GCC 12.3). CUDA is not required. The build uses
Release and -march=native; rebuild when moving to a different CPU.

```bash
./setup/build_cpu.sh
./setup/run_cpu.sh --query data/queries.bin --gt data/groundtruth.tsv \
  --index indexes/example --k 100 --nprobe 128 \
  --k-rank-cluster 1800 --k-rank-all-tokens 300
```

The script builds cpu_search, cpu_bench, build_cpu_hnsw, and correctness tests.
The launcher requires numactl and defaults to eight workers, one worker per
OpenMP core place, and CPU/memory binding to NUMA node 0. Set
CHIMERA_CPU_NUMA_NODE to another permitted node and OMP_NUM_THREADS to change
the worker count. Choose a node with at least that many available physical
cores. Queries are processed serially; the workers cooperate within each query.
Index construction may use all CPUs and does not need these query constraints.

Direct use of build-cpu/cpu_search defaults to at most eight workers but does
not set affinity or memory policy. The library respects the caller's OpenMP
settings. The benchmark takes its thread count explicitly.

## Production defaults

- HNSW centroid routing: M=16, efConstruction=500, query ef=clamp(2*nprobe,64,768).
- Packed clustered 1-bit FastScan for candidate generation.
- sorted_doc_merge_fast (mode 1): partition-based parallel cross-query reduction
  for the standard 32-token query, partition-local top-k, and final top-k merge.
  Other query lengths retain the established fallback reducer.
- Full-document 1-bit scoring: CHIMERA_CPU_STAGE2_BATCH=1. Each kernel processes
  one query token, vectorizes score correction and MaxSim over document tokens,
  and masks document boundaries. Candidate documents are distributed to workers.
- Dynamic scheduling for extra-bit refinement; prefer doc_4bit_ex.bin when
  available, with decoding of doc_4bit.bin as a loading-time fallback.

These are the defaults of both cpu_search and cpu_bench. No stage-2 environment
variables are needed. For experiments only, CHIMERA_CPU_STAGE2_BATCH supports
0 (old scalar maxima), 2, 4, 8, 16, and 32. CHIMERA_CPU_STAGE2_KERNEL=visit
is an optional batch-1/2 prototype; the default is baseline. Existing environment
overrides take precedence, so unset them to return to the production defaults.

Larger fused query batches were slower in our LoTTE experiments; synthetic
tile improvements were workload dependent. These defaults select the validated
LoTTE path. Recall still depends on nprobe and the two candidate limits and
should be tuned for the desired operating point.

## Index compatibility

Use the established Chimera index directory with doclens.bin, ivf.bin,
cluster_1bit.bin, doc_1bit.bin, a compatible extra-bit payload, and
centroids.hnsw. GPU CAGRA graph bytes cannot be loaded as HNSW.

build_cpu_hnsw CENTROIDS_CPU OUTPUT THREADS constructs the centroid sidecar
from already rotated float32 centroids. Its input is three little-endian
uint64 values (magic 0x315550434d494843, centroid count, dimension=128),
followed by row-major float32 data. Write OUTPUT as indexes/example/centroids.hnsw.
The centroid ordering must match the IVF cluster IDs. This tool only builds
the routing sidecar; it does not construct a complete index from embeddings.

Queries use the README binary layout. The evaluation CLI expects 32 tokens per
query and 128-dimensional embeddings for the standard optimized pipeline.
Use cpu_search --help for all arguments.

## Profiling and verification

Pass --profile-eval-all-queries to cpu_search to report average stage timings.
The C++ search API is unprofiled by default; search_profiled exposes detailed
counters. Counters ending in partition_*_work_ms sum worker time and must not
be added to total query latency. stage1_reduce_cross_query_merge_ms is the
wall-clock partition reduction including local selection and final merge.

cpu_bench takes INDEX QUERY TRUTH WARMUP CONFIG THREADS REPEAT OUT.
CONFIG rows are: id nprobe k_rank_cluster k_rank_all_tokens 0 (native HNSW ef).
TRUTH contains query_id document_id rank with exactly 100 distinct reference
documents per query. It outputs latency/recall JSONL and OUT.profiles.jsonl.
CHIMERA_VALIDATE_THREADS=1 checks one-thread, requested-thread, and profiled
rankings during warmup.

CTest compares partition reduction against an independent dense oracle at
one/eight threads, and SIMD/fused scoring against the original FastScan plus
scalar score correction, including query tails and document boundaries.
Compile-time assertions protect the production defaults.

Optional synthetic research tools are enabled with
-DCHIMERA_CPU_MICROBENCH=ON. They are excluded from the default build.
