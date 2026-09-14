# CPU backend provenance

Restored from Chimera commit 63f2ca9, subtree Chimera/cpp. The default path uses
HNSW routing, packed FastScan, partitioned cross-query reduction, SIMD document
scoring with one query token at a time, and dynamic extra-bit refinement.

The CPU-specific RaBitQ and HNSW headers under dep retain their upstream notices.
Unchanged RaBitQ/Eigen headers are shared from cpp/third_party.
The optional microbenchmarks contain experimental fused kernels; they do not
change the production defaults. See ../../docs/cpu.md for build and usage.
