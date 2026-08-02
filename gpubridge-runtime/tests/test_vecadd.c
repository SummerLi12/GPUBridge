/*
 * test_vecadd.c — the Milestone 0-1 correctness test (spec1.md sections
 * 14-16, exact output format and pass/fail criterion).
 *
 * Exercises the full public API end to end: gpuBridgeInit (which reads
 * GPUBRIDGE_BACKEND/GPUBRIDGE_VERBOSE from the environment) -> gpuBridgeMalloc x3 -> gpuBridgeMemcpy
 * host-to-device x2 -> gpuBridgeLaunchKernel -> gpuBridgeDeviceSynchronize -> gpuBridgeMemcpy
 * device-to-host -> compare against a host-computed reference -> gpuBridgeFree x3
 * -> gpuBridgeShutdown. Whichever backend gpuBridgeInit() resolved to (CPU or OpenCL) is
 * exercised identically through this same code path — that's the point of
 * the backend-neutral runtime.
 *
 * Exit code doubles as the pass/fail signal for `ctest` (see CMakeLists.txt's
 * add_test() entries): 0 on PASS, 1 on FAIL or any runtime failure.
 */
#include "gpubridge_runtime.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Phase 4.5 addition (spec4.5.md section 15): mirrors test_matmul.c's
 * existing GPUBRIDGE_BACKEND=opencl skip check, applied here for
 * GPUBRIDGE_BACKEND=vulkan for the first time — no vecadd_opencl ctest entry
 * ever needed this before. */
#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    /* 1,048,576 elements — the exact size used in spec1.md's worked
     * diagnostic example (section 11) and its expected test output
     * (section 14), so this test's output can be compared directly against
     * the spec text. */
    const size_t n = 1 << 20;

    /* gpuBridgeInit() reads GPUBRIDGE_BACKEND/GPUBRIDGE_VERBOSE from the environment and
     * resolves/initializes a backend. A failure here (e.g. GPUBRIDGE_BACKEND=sycl,
     * or an unrecognized GPUBRIDGE_BACKEND value) must be reported clearly and
     * exit without crashing — this is exactly how the "wrong backend
     * selection fails clearly" success criterion (spec1.md section 16.8)
     * is demonstrated. */
    if (gpuBridgeInit() != 0) {
        const char* requested_backend = getenv("GPUBRIDGE_BACKEND");
        if (requested_backend != NULL && strcasecmp(requested_backend, "vulkan") == 0) {
            printf("SKIP: GPUBRIDGE_BACKEND=vulkan requested but unavailable on "
                "this machine (no Vulkan-capable device, or libvulkan.so.1 not installed)\n");
            return GPUBRIDGE_TEST_SKIP;
        }
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Two known host input arrays and one host output array. The exact
     * fill formulas don't matter for correctness (any deterministic values
     * work, since we check c[i] against a[i]+b[i] computed from the same
     * arrays, not against a fixed expected constant) — chosen simply to
     * avoid degenerate all-zero or all-equal inputs. */
    float* a = malloc(n * sizeof(float));
    float* b = malloc(n * sizeof(float));
    float* c = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(n - i) * 0.25f;
    }

    /* Device-resident buffers for a, b, c. Each is an opaque handle — a
     * plain heap pointer if the CPU backend was selected, or a cl_mem cast
     * to void* if OpenCL was selected; this code never needs to know which. */
    void* dev_a = NULL;
    void* dev_b = NULL;
    void* dev_c = NULL;
    if (gpuBridgeMalloc(&dev_a, n * sizeof(float)) != 0 ||
        gpuBridgeMalloc(&dev_b, n * sizeof(float)) != 0 ||
        gpuBridgeMalloc(&dev_c, n * sizeof(float)) != 0) {
        fprintf(stderr, "gpuBridgeMalloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Upload both inputs. Blocking (per the GpuBridgeBackend contract), so it's
     * safe to launch the kernel immediately afterward. */
    if (gpuBridgeMemcpy(dev_a, a, n * sizeof(float), GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(dev_b, b, n * sizeof(float), GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Build the GPUBridgeIR description of this one kernel launch (see gpubridge_ir.h) —
     * kernel_args backs kernel.args and must stay alive for as long as
     * `kernel` is used below. */
    GpuBridgeKernelIR kernel;
    GpuBridgeBufferArg kernel_args[3];
    gpuBridgeIrInitVectorAddF32(&kernel, kernel_args, n);

    /* The actual device pointers passed to gpuBridgeLaunchKernel, in the same
     * order gpuBridgeIrInitVectorAddF32() described them (a, b, c). This is also
     * where GPUBRIDGE_VERBOSE=1 diagnostics get printed, as a side effect inside
     * gpuBridgeLaunchKernel(). */
    void* launch_args[3] = { dev_a, dev_b, dev_c };
    if (gpuBridgeLaunchKernel(&kernel, launch_args, 3) != 0) {
        fprintf(stderr, "gpuBridgeLaunchKernel failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Not strictly required for correctness here (copy_to_host is itself
     * blocking), but demonstrates the API and matches the natural flow of
     * "launch, then wait, then read results" callers are expected to use. */
    if (gpuBridgeDeviceSynchronize() != 0) {
        fprintf(stderr, "gpuBridgeDeviceSynchronize failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Download the result back into the host array c. */
    if (gpuBridgeMemcpy(c, dev_c, n * sizeof(float), GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy D2H failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Compare every element against a host-computed reference. Accumulate
     * in double to avoid the comparison itself introducing float rounding
     * error that could mask (or manufacture) a mismatch. */
    double max_error = 0.0;
    for (size_t i = 0; i < n; i++) {
        double expected = (double)a[i] + (double)b[i];
        double error = fabs((double)c[i] - expected);
        if (error > max_error) {
            max_error = error;
        }
    }

    /* Tolerance from spec1.md section 15: abs(c[i] - (a[i]+b[i])) <= 1e-6
     * for every element. */
    bool pass = max_error <= 1e-6;

    /* Exact output shape from spec1.md section 14's expected result block. */
    printf("GPUBridge vector_add_f32 test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("elements: %zu\n", n);
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
