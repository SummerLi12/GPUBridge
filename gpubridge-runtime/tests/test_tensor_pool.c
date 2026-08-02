/*
 * test_tensor_pool.c — Milestone 2.5 (spec2.5.md section 11.1): pooled
 * device-allocator correctness test.
 *
 * Three phases, all against a fixed f32[64,64] tensor shape so every
 * allocation in a phase shares one byte size:
 *
 *   1. Pooling enabled (default): repeated alloc/free of the same shape —
 *      the first allocation is always a pool miss, every subsequent one a
 *      hit. Asserts tensor_allocation_count/tensor_free_count/
 *      tensor_pool_hit_count via gpuBridgeGetProfileStats().
 *   2. GPUBRIDGE_POOL_DISABLE=1: the same loop must produce zero hits — every
 *      allocation bypasses the pool and calls malloc_device() directly.
 *   3. Pool-drain check: with pooling re-enabled, allocate+free once (so the
 *      pool holds exactly one free block), reselect the backend (which must
 *      call gpuBridgePoolDrain()), then allocate the same size again and
 *      assert it is a fresh miss — proving the drain actually emptied the
 *      free list rather than silently carrying it across the reselect.
 *
 * gpuBridgeSetBackend() with the currently-selected (concrete) backend kind
 * is used between phases both to reset the profile counters to zero (same
 * reset-on-backend-switch behavior gpuBridgeInit()/gpuBridgeSetBackend()
 * already have, spec1.5.md section 9.3) and, in phase 3, specifically to
 * exercise gpuBridgePoolDrain().
 *
 * Exit code doubles as the pass/fail signal for `ctest`: 0 on PASS, 1 on
 * FAIL or any runtime failure.
 */
#define _POSIX_C_SOURCE 200809L /* setenv() */

#include "gpubridge_runtime.h"

#include <stdio.h>
#include <stdlib.h>

#define N 10
#define ROWS 64
#define COLS 64

static int alloc_free_loop(size_t count, size_t rows, size_t cols)
{
    size_t shape[2] = { rows, cols };
    for (size_t i = 0; i < count; i++) {
        GpuBridgeTensorDesc desc;
        if (gpuBridgeTensorAlloc(&desc, GPUBRIDGE_TYPE_F32, 2, shape) != 0) {
            fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
        if (gpuBridgeTensorFree(&desc) != 0) {
            fprintf(stderr, "gpuBridgeTensorFree failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    if (gpuBridgeInit() != 0) {
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeBackendKind backend = gpuBridgeGetSelectedBackend();
    printf("GPUBridge tensor_pool test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(backend));

    bool all_pass = true;
    GpuBridgeProfileStats stats;

    /* Phase 1: pooling enabled. */
    if (alloc_free_loop(N, ROWS, COLS) != 0) {
        return 1;
    }
    if (gpuBridgeGetProfileStats(&stats) != 0) {
        fprintf(stderr, "gpuBridgeGetProfileStats failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    bool phase1_pass = stats.tensor_allocation_count == N
        && stats.tensor_free_count == N
        && stats.tensor_pool_hit_count == N - 1;
    printf("phase1_pooled_reuse: allocations=%zu frees=%zu hits=%zu %s\n",
        stats.tensor_allocation_count, stats.tensor_free_count,
        stats.tensor_pool_hit_count, phase1_pass ? "PASS" : "FAIL");
    all_pass = all_pass && phase1_pass;

    /* Phase 2: GPUBRIDGE_POOL_DISABLE=1 — reselecting the same backend resets
     * the profile counters (and drains the pool) for a clean phase. */
    setenv("GPUBRIDGE_POOL_DISABLE", "1", 1);
    if (gpuBridgeSetBackend(backend) != 0) {
        fprintf(stderr, "gpuBridgeSetBackend failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    if (alloc_free_loop(N, ROWS, COLS) != 0) {
        return 1;
    }
    if (gpuBridgeGetProfileStats(&stats) != 0) {
        fprintf(stderr, "gpuBridgeGetProfileStats failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    bool phase2_pass = stats.tensor_allocation_count == N
        && stats.tensor_pool_hit_count == 0;
    printf("phase2_pool_disabled: allocations=%zu hits=%zu %s\n",
        stats.tensor_allocation_count, stats.tensor_pool_hit_count,
        phase2_pass ? "PASS" : "FAIL");
    all_pass = all_pass && phase2_pass;

    /* Phase 3: pool-drain check. Re-enable pooling and reset counters. */
    setenv("GPUBRIDGE_POOL_DISABLE", "0", 1);
    if (gpuBridgeSetBackend(backend) != 0) {
        fprintf(stderr, "gpuBridgeSetBackend failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    /* One alloc/free leaves exactly one free block of this size in the pool. */
    if (alloc_free_loop(1, ROWS, COLS) != 0) {
        return 1;
    }
    /* Reselecting the same backend again must call gpuBridgePoolDrain()
     * (spec2.5.md section 7.3) before this happens. */
    if (gpuBridgeSetBackend(backend) != 0) {
        fprintf(stderr, "gpuBridgeSetBackend (drain) failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorDesc desc;
    size_t shape[2] = { ROWS, COLS };
    if (gpuBridgeTensorAlloc(&desc, GPUBRIDGE_TYPE_F32, 2, shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    if (gpuBridgeGetProfileStats(&stats) != 0) {
        fprintf(stderr, "gpuBridgeGetProfileStats failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    /* Must be a fresh miss: if the drain had failed to empty the free list,
     * this would silently be a hit against the block from before the
     * reselect. */
    bool phase3_pass = stats.tensor_pool_hit_count == 0;
    printf("phase3_pool_drain: hits_after_drain=%zu %s\n",
        stats.tensor_pool_hit_count, phase3_pass ? "PASS" : "FAIL");
    all_pass = all_pass && phase3_pass;
    gpuBridgeTensorFree(&desc);

    printf("result: %s\n", all_pass ? "PASS" : "FAIL");

    gpuBridgeShutdown();
    return all_pass ? 0 : 1;
}
