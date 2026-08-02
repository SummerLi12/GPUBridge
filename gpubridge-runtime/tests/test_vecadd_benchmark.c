/*
 * test_vecadd_benchmark.c — cross-backend vector_add_f32 benchmark
 * (spec5.md section 6.1), generalizing test_matmul_benchmark.c's
 * methodology (spec3.md section 9) to a single-path op (no vendor/naive
 * alternation — vector_add_f32 has exactly one code path per backend).
 *
 * Exit code: 0 iff the correctness check (test_vecadd.c's own tolerance,
 * reused verbatim) passed. GPUBRIDGE_BACKEND=vulkan skips cleanly
 * (GPUBRIDGE_TEST_SKIP 77) if unavailable, same precedent test_vecadd.c
 * itself established.
 */
#include "gpubridge_runtime.h"
#include "bench_common.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    /* 4,194,304 elements (spec5.md section 6.1) — larger than
     * test_vecadd.c's 1<<20 correctness-only size, so kernel time is
     * measurable above launch/sync noise. */
    const size_t n = 1 << 22;
    const int iters = gpuBridgeBenchReadIters();

    if (gpuBridgeInit() != 0) {
        const char* requested_backend = getenv("GPUBRIDGE_BACKEND");
        if (requested_backend != NULL && strcmp(requested_backend, "vulkan") == 0) {
            printf("SKIP: GPUBRIDGE_BACKEND=vulkan requested but unavailable on "
                "this machine (%s)\n", gpuBridgeGetLastErrorString());
            return GPUBRIDGE_TEST_SKIP;
        }
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    float* a = malloc(n * sizeof(float));
    float* b = malloc(n * sizeof(float));
    float* c = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(n - i) * 0.25f;
    }

    void* dev_a = NULL;
    void* dev_b = NULL;
    void* dev_c = NULL;
    if (gpuBridgeMalloc(&dev_a, n * sizeof(float)) != 0 ||
        gpuBridgeMalloc(&dev_b, n * sizeof(float)) != 0 ||
        gpuBridgeMalloc(&dev_c, n * sizeof(float)) != 0) {
        fprintf(stderr, "gpuBridgeMalloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    if (gpuBridgeMemcpy(dev_a, a, n * sizeof(float), GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(dev_b, b, n * sizeof(float), GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeKernelIR kernel;
    GpuBridgeBufferArg kernel_args[3];
    gpuBridgeIrInitVectorAddF32(&kernel, kernel_args, n);
    void* launch_args[3] = { dev_a, dev_b, dev_c };

    /* Clean quiescent baseline (spec5.md section 5 step 3). */
    gpuBridgeDeviceSynchronize();

    /* Independent warmup, untimed, discarded (step 4). */
    for (int i = 0; i < GPUBRIDGE_BENCH_WARMUP_ROUNDS; i++) {
        gpuBridgeLaunchKernel(&kernel, launch_args, 3);
        gpuBridgeDeviceSynchronize();
    }

    /* Timed loop (step 5). */
    double* samples = malloc((size_t)iters * sizeof(double));
    for (int round = 0; round < iters; round++) {
        GpuBridgeProfileStats before, after;
        gpuBridgeDeviceSynchronize();
        gpuBridgeGetProfileStats(&before);
        gpuBridgeLaunchKernel(&kernel, launch_args, 3);
        gpuBridgeDeviceSynchronize();
        gpuBridgeGetProfileStats(&after);
        samples[round] = gpuBridgeBenchGpuPathMs(&after) - gpuBridgeBenchGpuPathMs(&before);
    }
    double median_ms = gpuBridgeBenchMedian(samples, iters);
    double p95_ms = gpuBridgeBenchPercentile(samples, iters, 0.95);
    double p99_ms = gpuBridgeBenchPercentile(samples, iters, 0.99);
    free(samples);

    /* Correctness, untimed, after all sampling (step 6) — same tolerance
     * as test_vecadd.c. */
    if (gpuBridgeLaunchKernel(&kernel, launch_args, 3) != 0 ||
        gpuBridgeDeviceSynchronize() != 0 ||
        gpuBridgeMemcpy(c, dev_c, n * sizeof(float), GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "post-benchmark correctness launch failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
    double max_error = 0.0;
    for (size_t i = 0; i < n; i++) {
        double expected = (double)a[i] + (double)b[i];
        double error = fabs((double)c[i] - expected);
        if (error > max_error) { max_error = error; }
    }
    bool pass = max_error <= 1e-6;

    /* Report (spec5.md section 6.5/7). */
    char device_name[128];
    gpuBridgeGetSelectedDeviceName(device_name, sizeof(device_name));
    size_t mem_limit = 0;
    gpuBridgeGetDeviceMemoryLimit(&mem_limit);
    GpuBridgeHardwareTier tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;
    gpuBridgeGetHardwareTier(&tier);

    printf("GPUBridge vector_add_f32 benchmark\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("device: %s\n", device_name);
    printf("hardware_tier: %s\n", gpuBridgeHardwareTierToString(tier));
    printf("device_memory_limit_bytes: %zu\n", mem_limit);
    printf("elements: %zu\n", n);
    printf("iterations: %d%s\n", iters,
        getenv("GPUBRIDGE_BENCH_ITERS") != NULL ? "" : " (GPUBRIDGE_BENCH_ITERS unset, using default)");
    printf("median_ms: %.4f\n", median_ms);
    printf("p95_ms: %.4f\n", p95_ms);
    printf("p99_ms: %.4f\n", p99_ms);
    printf("throughput_ops_per_sec: %.2f\n",
        median_ms > 0.0 ? ((double)n / (median_ms / 1000.0)) : 0.0);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max error: %f\n", max_error);

    gpuBridgeFree(dev_a);
    gpuBridgeFree(dev_b);
    gpuBridgeFree(dev_c);
    free(a);
    free(b);
    free(c);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
