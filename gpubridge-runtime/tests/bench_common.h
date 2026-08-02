/*
 * bench_common.h — shared benchmark-harness plumbing for the Phase 5
 * cross-backend benchmark suite (spec5.md section 5), extracted from
 * test_matmul_benchmark.c's original median()/read_iters()/gpu_path_ms()
 * (spec3.md section 9) so the four new per-op benchmarks don't each
 * duplicate them a fifth time. Pure test-harness code — not part of
 * libgpubridge_runtime.a, compiled directly into each *_benchmark_test
 * executable (see CMakeLists.txt).
 */
#ifndef GPUBRIDGE_BENCH_COMMON_H
#define GPUBRIDGE_BENCH_COMMON_H

#include "gpubridge_runtime.h"

/* Default per-path sample count (spec3.md section 9.2 precedent),
 * overridable via GPUBRIDGE_BENCH_ITERS. Any unset, non-numeric, or <= 0
 * value falls back to this default rather than erroring. Named
 * GPUBRIDGE_BENCH_ITERS (not GPUBRIDGE_GEMM_BENCH_ITERS) since it now
 * covers every op's benchmark, not just matmul's — test_matmul_benchmark.c
 * keeps reading its own GPUBRIDGE_GEMM_BENCH_ITERS unchanged (spec5.md
 * section 4 non-goal 6: its vendor/naive alternating logic is untouched by
 * this phase), the two env vars are deliberately independent. */
#define GPUBRIDGE_BENCH_DEFAULT_ITERS 10
/* Independent warmup rounds per path, small and fixed, not configurable —
 * absorbs one-time costs (backend init, first-call overhead) outside the
 * timed loop (spec3.md section 9.2 step 5 precedent). */
#define GPUBRIDGE_BENCH_WARMUP_ROUNDS 3

/* Reads GPUBRIDGE_BENCH_ITERS, falling back to GPUBRIDGE_BENCH_DEFAULT_ITERS
 * on unset/invalid/non-positive values. */
int gpuBridgeBenchReadIters(void);

/* Sorts `samples` in place (destructive) and returns the median: the
 * average of the two middle elements for an even count, the single middle
 * element for an odd count. */
double gpuBridgeBenchMedian(double* samples, int count);

/* Nearest-rank percentile (ceil(p * count) - 1, 0-indexed) over an
 * ALREADY-SORTED array (call gpuBridgeBenchMedian() first, or sort
 * separately) — spec5.md section 5.1. p in [0, 1]; p=0.95 for p95,
 * p=0.99 for p99. No interpolation: at low iteration counts this
 * deliberately resolves to the same (largest) sample for both p95 and
 * p99 rather than implying more precision than the sample count
 * supports. */
double gpuBridgeBenchPercentile(const double* sorted_samples, int count, double p);

/* kernel_time_ms + synchronization_time_ms (spec3.md section 9.1):
 * OpenCL/Vulkan's kernel_time_ms alone only times the non-blocking
 * launch call — real completion time only shows up in
 * synchronization_time_ms after gpuBridgeDeviceSynchronize(). Summing
 * both gives the real wall-clock op cost on every backend (CPU's
 * synchronization_time_ms is always ~0 since its sync() is a trivial
 * no-op, so this degrades gracefully to just kernel_time_ms there). */
double gpuBridgeBenchGpuPathMs(const GpuBridgeProfileStats* s);

#endif /* GPUBRIDGE_BENCH_COMMON_H */
