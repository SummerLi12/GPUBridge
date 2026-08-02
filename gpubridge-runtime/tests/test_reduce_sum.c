/*
 * test_reduce_sum.c — isolated correctness test for reduce_sum_f32
 * (spec2.md section 7.2), same structure as test_matmul.c.
 */
#include "gpubridge_runtime.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Vulkan tensor-op port (beyond spec4.5.md's own vector_add_f32-only scope
 * — see CLAUDE.md's "Phase 4.5" section): reduce_sum_vulkan needs the same
 * "explicitly-requested backend that isn't present on this machine skips
 * cleanly" treatment test_vecadd.c/test_matmul.c already established. */
#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    const size_t n = 100000;

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

    float result = 0.0f;
    if (gpuBridgeMemcpy(&result, result_dev.device_ptr, sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy D2H failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Double-precision CPU reference (spec2.md 7.2). */
    double expected = 0.0;
    for (size_t i = 0; i < n; i++) {
        expected += (double)a[i];
    }
    double error = fabs((double)result - expected);
    double rel = error / fmax(1.0, fabs(expected));

    /* Tolerance from spec2.md 7.2: 1e-3 relative. */
    bool pass = rel <= 1e-3;

    printf("GPUBridge reduce_sum_f32 test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("shape: A[%zu] -> Result[1]\n", n);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max relative error: %f\n", rel);

    gpuBridgeTensorFree(&a_dev);
    gpuBridgeTensorFree(&result_dev);
    free(a);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
