/*
 * test_relu.c — isolated correctness test for relu_f32 (spec2.md section
 * 7.4), same structure as test_matmul.c. Uses a rank-2 tensor to also
 * exercise the flattened-elementwise-over-rank-2 path.
 */
#include "gpubridge_runtime.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Vulkan tensor-op port (beyond spec4.5.md's own vector_add_f32-only scope
 * — see CLAUDE.md's "Phase 4.5" section): relu_vulkan needs the same
 * "explicitly-requested backend that isn't present on this machine skips
 * cleanly" treatment test_vecadd.c/test_matmul.c already established. */
#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    const size_t rows = 30;
    const size_t cols = 40;
    const size_t count = rows * cols;

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

    float* y = malloc(count * sizeof(float));
    if (gpuBridgeMemcpy(y, y_dev.device_ptr, count * sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy D2H failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Tolerance from spec2.md 7.4: 1e-6 absolute (no accumulation). */
    double max_error = 0.0;
    for (size_t i = 0; i < count; i++) {
        double expected = x[i] > 0.0f ? (double)x[i] : 0.0;
        double error = fabs((double)y[i] - expected);
        if (error > max_error) {
            max_error = error;
        }
    }
    bool pass = max_error <= 1e-6;

    printf("GPUBridge relu_f32 test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("shape: X[%zu,%zu] -> Y[%zu,%zu]\n", rows, cols, rows, cols);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max error: %f\n", max_error);

    gpuBridgeTensorFree(&x_dev);
    gpuBridgeTensorFree(&y_dev);
    free(x);
    free(y);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
