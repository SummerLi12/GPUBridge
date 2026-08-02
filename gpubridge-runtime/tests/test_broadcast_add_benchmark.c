/*
 * test_broadcast_add_benchmark.c — cross-backend broadcast_add_f32
 * benchmark (spec5.md section 6.3), same single-path shape as
 * test_vecadd_benchmark.c. Exit code: 0 iff correctness passed
 * (test_broadcast_add.c's own 1e-6 absolute tolerance, reused verbatim).
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
    /* rows=2048, cols=2048 (spec5.md section 6.3) — large enough for the
     * 2D dispatch shape (every backend) to show real parallelism. */
    const size_t rows = 2048;
    const size_t cols = 2048;
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

    float* y = malloc(rows * cols * sizeof(float));
    float* bias = malloc(cols * sizeof(float));
    for (size_t i = 0; i < rows * cols; i++) {
        y[i] = (float)(i % 11) * 0.3f - 1.0f;
    }
    for (size_t j = 0; j < cols; j++) {
        bias[j] = (float)j * 0.0005f;
    }

    GpuBridgeTensorDesc y_dev, bias_dev;
    const size_t y_shape[2] = { rows, cols };
    const size_t bias_shape[2] = { cols, 0 };
    if (gpuBridgeTensorAlloc(&y_dev, GPUBRIDGE_TYPE_F32, 2, y_shape) != 0 ||
        gpuBridgeTensorAlloc(&bias_dev, GPUBRIDGE_TYPE_F32, 1, bias_shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    /* Keep a pristine copy of y's original contents on the host: the op is
     * in-place, so each re-launch during warmup/timing/correctness would
     * otherwise keep adding bias on top of the previous launch's result. */
    float* y_original = malloc(rows * cols * sizeof(float));
    memcpy(y_original, y, rows * cols * sizeof(float));

    if (gpuBridgeMemcpy(bias_dev.device_ptr, bias, cols * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR kernel;
    GpuBridgeTensorArg kernel_args[2];
    gpuBridgeIrInitBroadcastAddF32(&kernel, kernel_args, rows, cols);
    void* launch_args[2] = { y_dev.device_ptr, bias_dev.device_ptr };

    /* re_upload_y(): resets y's device buffer to its pristine host state
     * before every launch, so an in-place op can be re-timed/re-warmed-up
     * repeatedly without each round adding to the last round's result. */
#define RE_UPLOAD_Y() \
    gpuBridgeMemcpy(y_dev.device_ptr, y_original, rows * cols * sizeof(float), \
        GPUBRIDGE_MEMCPY_HOST_TO_DEVICE)

    gpuBridgeDeviceSynchronize();

    for (int i = 0; i < GPUBRIDGE_BENCH_WARMUP_ROUNDS; i++) {
        RE_UPLOAD_Y();
        gpuBridgeLaunchTensorKernel(&kernel, launch_args, 2);
        gpuBridgeDeviceSynchronize();
    }

    double* samples = malloc((size_t)iters * sizeof(double));
    for (int round = 0; round < iters; round++) {
        GpuBridgeProfileStats before, after;
        RE_UPLOAD_Y();
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

    float* result = malloc(rows * cols * sizeof(float));
    if (RE_UPLOAD_Y() != 0 ||
        gpuBridgeLaunchTensorKernel(&kernel, launch_args, 2) != 0 ||
        gpuBridgeDeviceSynchronize() != 0 ||
        gpuBridgeMemcpy(result, y_dev.device_ptr, rows * cols * sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "post-benchmark correctness launch failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
#undef RE_UPLOAD_Y
    double max_error = 0.0;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double expected = (double)y_original[i * cols + j] + (double)bias[j];
            double error = fabs((double)result[i * cols + j] - expected);
            if (error > max_error) { max_error = error; }
        }
    }
    bool pass = max_error <= 1e-6;

    char device_name[128];
    gpuBridgeGetSelectedDeviceName(device_name, sizeof(device_name));
    size_t mem_limit = 0;
    gpuBridgeGetDeviceMemoryLimit(&mem_limit);
    GpuBridgeHardwareTier tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;
    gpuBridgeGetHardwareTier(&tier);

    printf("GPUBridge broadcast_add_f32 benchmark\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("device: %s\n", device_name);
    printf("hardware_tier: %s\n", gpuBridgeHardwareTierToString(tier));
    printf("device_memory_limit_bytes: %zu\n", mem_limit);
    printf("shape: Y[%zu,%zu] Bias[%zu]\n", rows, cols, cols);
    printf("iterations: %d%s\n", iters,
        getenv("GPUBRIDGE_BENCH_ITERS") != NULL ? "" : " (GPUBRIDGE_BENCH_ITERS unset, using default)");
    printf("median_ms: %.4f\n", median_ms);
    printf("p95_ms: %.4f\n", p95_ms);
    printf("p99_ms: %.4f\n", p99_ms);
    printf("throughput_ops_per_sec: %.2f\n",
        median_ms > 0.0 ? ((double)(rows * cols) / (median_ms / 1000.0)) : 0.0);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max error: %f\n", max_error);

    gpuBridgeTensorFree(&y_dev);
    gpuBridgeTensorFree(&bias_dev);
    free(y);
    free(y_original);
    free(bias);
    free(result);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
