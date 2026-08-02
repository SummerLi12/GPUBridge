/*
 * test_broadcast_add.c — isolated correctness test for broadcast_add_f32
 * (spec2.md section 7.3), same structure as test_matmul.c. Y[i,j] += b[j],
 * in-place on Y's device buffer.
 */
#include "gpubridge_runtime.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Vulkan tensor-op port (beyond spec4.5.md's own vector_add_f32-only scope
 * — see CLAUDE.md's "Phase 4.5" section): broadcast_add_vulkan needs the
 * same "explicitly-requested backend that isn't present on this machine
 * skips cleanly" treatment test_vecadd.c/test_matmul.c already established. */
#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    const size_t rows = 40;
    const size_t cols = 24;

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
        bias[j] = (float)j * 0.05f;
    }

    GpuBridgeTensorDesc y_dev, bias_dev;
    const size_t y_shape[2] = { rows, cols };
    const size_t bias_shape[2] = { cols, 0 };
    if (gpuBridgeTensorAlloc(&y_dev, GPUBRIDGE_TYPE_F32, 2, y_shape) != 0 ||
        gpuBridgeTensorAlloc(&bias_dev, GPUBRIDGE_TYPE_F32, 1, bias_shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeMemcpy(y_dev.device_ptr, y, rows * cols * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(bias_dev.device_ptr, bias, cols * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR kernel;
    GpuBridgeTensorArg kernel_args[2];
    gpuBridgeIrInitBroadcastAddF32(&kernel, kernel_args, rows, cols);

    void* launch_args[2] = { y_dev.device_ptr, bias_dev.device_ptr };
    if (gpuBridgeLaunchTensorKernel(&kernel, launch_args, 2) != 0) {
        fprintf(stderr, "gpuBridgeLaunchTensorKernel failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeDeviceSynchronize() != 0) {
        fprintf(stderr, "gpuBridgeDeviceSynchronize failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    float* result = malloc(rows * cols * sizeof(float));
    if (gpuBridgeMemcpy(result, y_dev.device_ptr, rows * cols * sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy D2H failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Tolerance from spec2.md 7.3: 1e-6 absolute. */
    double max_error = 0.0;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double expected = (double)y[i * cols + j] + (double)bias[j];
            double error = fabs((double)result[i * cols + j] - expected);
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    bool pass = max_error <= 1e-6;

    printf("GPUBridge broadcast_add_f32 test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("shape: Y[%zu,%zu] Bias[%zu] -> Y[%zu,%zu]\n", rows, cols, cols,
        rows, cols);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max error: %f\n", max_error);

    gpuBridgeTensorFree(&y_dev);
    gpuBridgeTensorFree(&bias_dev);
    free(y);
    free(bias);
    free(result);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
