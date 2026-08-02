/*
 * test_reduce_sum_benchmark.c — cross-backend reduce_sum_f32 benchmark
 * (spec5.md section 6.2), same single-path shape as
 * test_vecadd_benchmark.c. Exit code: 0 iff correctness passed
 * (test_reduce_sum.c's own 1e-3 relative tolerance, reused verbatim).
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
    /* 1,048,576 elements (spec5.md section 6.2) — large enough to matter,
     * still small enough that the naive single-invocation-walks-the-
     * whole-array kernel (every backend) completes in reportable time. */
    const size_t n = 1 << 20;
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
    for (size_t i = 0; i < n; i++) {
        a[i] = (float)(i % 17) * 0.01f - 0.05f;
    }

    GpuBridgeTensorDesc a_dev, result_dev;
    const size_t a_shape[2] = { n, 0 };
    const size_t result_shape[2] = { 1, 0 };
    if (gpuBridgeTensorAlloc(&a_dev, GPUBRIDGE_TYPE_F32, 1, a_shape) != 0 ||
        gpuBridgeTensorAlloc(&result_dev, GPUBRIDGE_TYPE_F32, 1, result_shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    if (gpuBridgeMemcpy(a_dev.device_ptr, a, n * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR kernel;
    GpuBridgeTensorArg kernel_args[2];
    gpuBridgeIrInitReduceSumF32(&kernel, kernel_args, n);
    void* launch_args[2] = { a_dev.device_ptr, result_dev.device_ptr };

    gpuBridgeDeviceSynchronize();

    for (int i = 0; i < GPUBRIDGE_BENCH_WARMUP_ROUNDS; i++) {
        gpuBridgeLaunchTensorKernel(&kernel, launch_args, 2);
        gpuBridgeDeviceSynchronize();
    }

    double* samples = malloc((size_t)iters * sizeof(double));
    for (int round = 0; round < iters; round++) {
        GpuBridgeProfileStats before, after;
        gpuBridgeDeviceSynchronize();
        gpuBridgeGetProfileStats(&before);
        gpuBridgeLaunchTensorKernel(&kernel, launch_args, 2);
        gpuBridgeDeviceSynchronize();
        gpuBridgeGetProfileStats(&after);
        samples[round] = gpuBridgeBenchGpuPathMs(&after) - gpuBridgeBenchGpuPathMs(&before);
    }
    double median_ms = gpuBridgeBenchMedian(samples, iters);
    double p95_ms = gpuBridgeBenchPercentile(samples, iters, 0.95);
    double p99_ms = gpuBridgeBenchPercentile(samples, iters, 0.99);
    free(samples);

    float result = 0.0f;
    if (gpuBridgeLaunchTensorKernel(&kernel, launch_args, 2) != 0 ||
        gpuBridgeDeviceSynchronize() != 0 ||
        gpuBridgeMemcpy(&result, result_dev.device_ptr, sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "post-benchmark correctness launch failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
    double expected = 0.0;
    for (size_t i = 0; i < n; i++) {
        expected += (double)a[i];
    }
    double error = fabs((double)result - expected);
    double rel = error / fmax(1.0, fabs(expected));
    bool pass = rel <= 1e-3;

    char device_name[128];
    gpuBridgeGetSelectedDeviceName(device_name, sizeof(device_name));
    size_t mem_limit = 0;
    gpuBridgeGetDeviceMemoryLimit(&mem_limit);
    GpuBridgeHardwareTier tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;
    gpuBridgeGetHardwareTier(&tier);

    printf("GPUBridge reduce_sum_f32 benchmark\n");
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
    /* Reported for consistency with the other four benchmarks, but not a
     * meaningful rate here — reduce_sum's output is always exactly 1
     * element regardless of input size (spec5.md section 6.5). */
    printf("throughput_ops_per_sec: %.2f\n", median_ms > 0.0 ? (1.0 / (median_ms / 1000.0)) : 0.0);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max relative error: %f\n", rel);

    gpuBridgeTensorFree(&a_dev);
    gpuBridgeTensorFree(&result_dev);
    free(a);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
