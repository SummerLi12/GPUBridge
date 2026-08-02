/*
 * test_relu_benchmark.c — cross-backend relu_f32 benchmark (spec5.md
 * section 6.4), same single-path shape as test_vecadd_benchmark.c. Exit
 * code: 0 iff correctness passed (test_relu.c's own 1e-6 absolute
 * tolerance, reused verbatim).
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
    /* 4,194,304 elements (spec5.md section 6.4) — same order of magnitude
     * as vector_add_f32's benchmark, both simple flattened elementwise ops. */
    const size_t rows = 2048;
    const size_t cols = 2048;
    const size_t count = rows * cols;
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

    float* x = malloc(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        x[i] = (float)((long)(i % 23) - 11) * 0.1f;
    }

    GpuBridgeTensorDesc x_dev, y_dev;
    const size_t shape[2] = { rows, cols };
    if (gpuBridgeTensorAlloc(&x_dev, GPUBRIDGE_TYPE_F32, 2, shape) != 0 ||
        gpuBridgeTensorAlloc(&y_dev, GPUBRIDGE_TYPE_F32, 2, shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    if (gpuBridgeMemcpy(x_dev.device_ptr, x, count * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR kernel;
    GpuBridgeTensorArg kernel_args[2];
    gpuBridgeIrInitReluF32(&kernel, kernel_args, 2, shape);
    void* launch_args[2] = { x_dev.device_ptr, y_dev.device_ptr };

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

    float* y = malloc(count * sizeof(float));
    if (gpuBridgeLaunchTensorKernel(&kernel, launch_args, 2) != 0 ||
        gpuBridgeDeviceSynchronize() != 0 ||
        gpuBridgeMemcpy(y, y_dev.device_ptr, count * sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "post-benchmark correctness launch failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
    double max_error = 0.0;
    for (size_t i = 0; i < count; i++) {
        double expected = x[i] > 0.0f ? (double)x[i] : 0.0;
        double error = fabs((double)y[i] - expected);
        if (error > max_error) { max_error = error; }
    }
    bool pass = max_error <= 1e-6;

    char device_name[128];
    gpuBridgeGetSelectedDeviceName(device_name, sizeof(device_name));
    size_t mem_limit = 0;
    gpuBridgeGetDeviceMemoryLimit(&mem_limit);
    GpuBridgeHardwareTier tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;
    gpuBridgeGetHardwareTier(&tier);

    printf("GPUBridge relu_f32 benchmark\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("device: %s\n", device_name);
    printf("hardware_tier: %s\n", gpuBridgeHardwareTierToString(tier));
    printf("device_memory_limit_bytes: %zu\n", mem_limit);
    printf("shape: X[%zu,%zu]\n", rows, cols);
    printf("iterations: %d%s\n", iters,
        getenv("GPUBRIDGE_BENCH_ITERS") != NULL ? "" : " (GPUBRIDGE_BENCH_ITERS unset, using default)");
    printf("median_ms: %.4f\n", median_ms);
    printf("p95_ms: %.4f\n", p95_ms);
    printf("p99_ms: %.4f\n", p99_ms);
    printf("throughput_ops_per_sec: %.2f\n",
        median_ms > 0.0 ? ((double)count / (median_ms / 1000.0)) : 0.0);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max error: %f\n", max_error);

    gpuBridgeTensorFree(&x_dev);
    gpuBridgeTensorFree(&y_dev);
    free(x);
    free(y);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
