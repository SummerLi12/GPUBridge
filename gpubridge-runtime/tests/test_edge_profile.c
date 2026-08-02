/*
 * test_edge_profile.c — Phase 4 (spec4.md section 12): edge-inference
 * runtime tests that need a real backend (gpuBridgeInit()).
 *
 * Same shape as test_profile.c/test_tensor_pool.c: one main() sequence of
 * independent phases, each contributing to a single all_pass flag. Exit code
 * doubles as the pass/fail signal for `ctest`: 0 on PASS, 1 on FAIL or any
 * runtime failure.
 */
#include "gpubridge_runtime.h"

#include <stdio.h>
#include <string.h>

#define ROWS 8
#define COLS 8

int main(void)
{
    if (gpuBridgeInit() != 0) {
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeBackendKind backend = gpuBridgeGetSelectedBackend();
    printf("GPUBridge edge_profile test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(backend));

    bool all_pass = true;

    /* --- Device memory limit: nonzero on both backends --- */
    size_t mem_limit = 0;
    if (gpuBridgeGetDeviceMemoryLimit(&mem_limit) != 0) {
        fprintf(stderr, "gpuBridgeGetDeviceMemoryLimit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    bool mem_limit_pass = mem_limit > 0;
    printf("device_memory_limit_bytes: %zu %s\n", mem_limit, mem_limit_pass ? "PASS" : "FAIL");
    all_pass = all_pass && mem_limit_pass;

    /* --- Hardware tier: CPU_ONLY for cpu backend, not UNKNOWN for opencl --- */
    GpuBridgeHardwareTier tier;
    if (gpuBridgeGetHardwareTier(&tier) != 0) {
        fprintf(stderr, "gpuBridgeGetHardwareTier failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    bool tier_pass = (backend == GPUBRIDGE_BACKEND_CPU)
        ? (tier == GPUBRIDGE_HARDWARE_TIER_CPU_ONLY)
        : (tier != GPUBRIDGE_HARDWARE_TIER_UNKNOWN);
    printf("hardware_tier: %s %s\n", gpuBridgeHardwareTierToString(tier), tier_pass ? "PASS" : "FAIL");
    all_pass = all_pass && tier_pass;

    /* --- Quantized types are rejected by gpuBridgeTensorAlloc, distinctly --- */
    {
        GpuBridgeTensorDesc desc;
        size_t shape[2] = { ROWS, COLS };
        int rc = gpuBridgeTensorAlloc(&desc, GPUBRIDGE_TYPE_INT8_QUANTIZED, 2, shape);
        const char* err = gpuBridgeGetLastErrorString();
        bool quant_reject_pass = rc != 0 && strstr(err, "descriptive-only") != NULL;
        printf("quantized_alloc_rejected: rc=%d err=\"%s\" %s\n", rc, err,
            quant_reject_pass ? "PASS" : "FAIL");
        all_pass = all_pass && quant_reject_pass;
    }

    /* --- KV-cache alloc/free round-trip --- */
    {
        GpuBridgeTensorDesc desc;
        size_t shape[2] = { ROWS, COLS };
        if (gpuBridgeTensorAllocKvCache(&desc, GPUBRIDGE_TYPE_F32, 2, shape) != 0) {
            fprintf(stderr, "gpuBridgeTensorAllocKvCache failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
        bool kv_alloc_pass = desc.device_resident
            && desc.usage == GPUBRIDGE_TENSOR_USAGE_KV_CACHE;
        printf("kv_cache_alloc: usage=%d device_resident=%d %s\n",
            desc.usage, desc.device_resident, kv_alloc_pass ? "PASS" : "FAIL");
        all_pass = all_pass && kv_alloc_pass;

        GpuBridgeProfileStats stats;
        if (gpuBridgeGetProfileStats(&stats) != 0) {
            fprintf(stderr, "gpuBridgeGetProfileStats failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
        size_t expected_bytes = ROWS * COLS * sizeof(float);
        bool kv_stats_pass = stats.kv_cache_allocation_count == 1
            && stats.kv_cache_bytes == expected_bytes;
        printf("kv_cache_stats: allocations=%zu bytes=%zu (expected %zu) %s\n",
            stats.kv_cache_allocation_count, stats.kv_cache_bytes, expected_bytes,
            kv_stats_pass ? "PASS" : "FAIL");
        all_pass = all_pass && kv_stats_pass;

        if (gpuBridgeTensorFree(&desc) != 0) {
            fprintf(stderr, "gpuBridgeTensorFree failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
    }

    /* --- Model-load bracket around a real alloc+memcpy+free sequence --- */
    {
        if (gpuBridgeBeginModelLoad() != 0) {
            fprintf(stderr, "gpuBridgeBeginModelLoad failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }

        GpuBridgeTensorDesc desc;
        size_t shape[2] = { ROWS, COLS };
        if (gpuBridgeTensorAlloc(&desc, GPUBRIDGE_TYPE_F32, 2, shape) != 0) {
            fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
        float host_buf[ROWS * COLS];
        for (size_t i = 0; i < ROWS * COLS; i++) {
            host_buf[i] = (float)i;
        }
        if (gpuBridgeMemcpy(desc.device_ptr, host_buf, sizeof(host_buf),
                GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
            fprintf(stderr, "gpuBridgeMemcpy failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
        if (gpuBridgeTensorFree(&desc) != 0) {
            fprintf(stderr, "gpuBridgeTensorFree failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }

        if (gpuBridgeEndModelLoad() != 0) {
            fprintf(stderr, "gpuBridgeEndModelLoad failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }

        GpuBridgeProfileStats stats;
        if (gpuBridgeGetProfileStats(&stats) != 0) {
            fprintf(stderr, "gpuBridgeGetProfileStats failed: %s\n", gpuBridgeGetLastErrorString());
            return 1;
        }
        /* Not "> 0": a single fast operation can legitimately measure 0.00ms
         * (same reasoning test_profile.c already uses for timing fields). */
        bool model_load_pass = stats.model_load_time_ms >= 0.0;
        bool runtime_init_pass = stats.runtime_init_time_ms >= 0.0;
        printf("model_load_time_ms: %.4f %s\n", stats.model_load_time_ms,
            model_load_pass ? "PASS" : "FAIL");
        printf("runtime_init_time_ms: %.4f %s\n", stats.runtime_init_time_ms,
            runtime_init_pass ? "PASS" : "FAIL");
        all_pass = all_pass && model_load_pass && runtime_init_pass;
    }

    printf("result: %s\n", all_pass ? "PASS" : "FAIL");

    gpuBridgeShutdown();
    return all_pass ? 0 : 1;
}
