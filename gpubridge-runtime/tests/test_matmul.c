/*
 * test_matmul.c — isolated correctness test for matmul_f32 (spec2.md
 * section 7.1), following test_vecadd.c's init->...->shutdown structure but
 * exercising the Milestone 2 tensor API (gpuBridgeTensorAlloc/gpuBridgeLaunchTensorKernel)
 * instead of gpuBridgeMalloc/gpuBridgeLaunchKernel.
 *
 * Exit code doubles as ctest's pass/fail signal (see CMakeLists.txt's
 * matmul_cpu/matmul_auto add_test() entries): 0 on PASS, 1 otherwise.
 */
#include "gpubridge_runtime.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Milestone 3 (spec3.md section 10.1): CTest's conventional "this test was
 * skipped, not failed" exit code (paired with each relevant add_test()'s
 * SKIP_RETURN_CODE 77 property in CMakeLists.txt). */
#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    const size_t m = 64;
    const size_t k = 128;
    const size_t n = 32;

    if (gpuBridgeInit() != 0) {
        /* Milestone 3 (spec3.md section 10.1): matmul_opencl_vendor/
         * matmul_opencl_novendor explicitly request GPUBRIDGE_BACKEND=opencl;
         * on a machine with no OpenCL device, gpuBridgeInit() fails here — that's
         * "OpenCL unavailable on this machine", not a real test failure, so
         * skip cleanly instead. Every other requested backend (cpu/auto) is
         * expected to always succeed, so an init failure there remains a
         * genuine failure. Vulkan tensor-op port (beyond spec4.5.md's own
         * scope, CLAUDE.md's "Phase 4.5" section): matmul_vulkan gets the
         * same treatment, mirroring test_vecadd.c's existing
         * GPUBRIDGE_BACKEND=vulkan skip precedent. */
        const char* requested_backend = getenv("GPUBRIDGE_BACKEND");
        if (requested_backend != NULL &&
            (strcmp(requested_backend, "opencl") == 0 ||
             strcmp(requested_backend, "vulkan") == 0)) {
            printf("SKIP: GPUBRIDGE_BACKEND=%s requested but unavailable on "
                "this machine (%s)\n", requested_backend, gpuBridgeGetLastErrorString());
            return GPUBRIDGE_TEST_SKIP;
        }
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* Milestone 3 (spec3.md section 10.1): matmul_cpu_vendor/
     * matmul_opencl_vendor set GPUBRIDGE_TEST_REQUIRE_VENDOR=1 to prove the
     * vendor GEMM path is genuinely exercised, not just possibly-exercised.
     * Skip cleanly if the active backend's vendor library never loaded,
     * rather than silently passing-by-omission. Test-only env var, not read
     * anywhere in the runtime library itself. */
    const char* require_vendor = getenv("GPUBRIDGE_TEST_REQUIRE_VENDOR");
    if (require_vendor != NULL && strcmp(require_vendor, "1") == 0) {
        GpuBridgeBackendCaps caps;
        if (gpuBridgeGetBackendCaps(&caps) != 0 || !caps.vendor_gemm_loaded) {
            printf("SKIP: GPUBRIDGE_TEST_REQUIRE_VENDOR=1 but no vendor GEMM "
                "library is loaded on backend %s\n",
                gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
            gpuBridgeShutdown();
            return GPUBRIDGE_TEST_SKIP;
        }
    }

    /* GPUBRIDGE_TEST_VERBOSE=1: test-only env var (same precedent as
     * GPUBRIDGE_TEST_REQUIRE_VENDOR above, not read anywhere in the runtime
     * library itself) that prints every input value, a fully expanded
     * dot-product for C[0,0] as a worked example, and a per-output-element
     * expected/actual/error/PASS-FAIL line — so the correctness logic and
     * how the final PASS is reached are both visible, not just the summary. */
    const char* verbose_str = getenv("GPUBRIDGE_TEST_VERBOSE");
    bool verbose = verbose_str != NULL && strcmp(verbose_str, "1") == 0;

    float* a = malloc(m * k * sizeof(float));
    float* b = malloc(k * n * sizeof(float));
    float* c = malloc(m * n * sizeof(float));
    for (size_t i = 0; i < m * k; i++) {
        a[i] = (float)(i % 13) * 0.1f - 0.6f;
    }
    for (size_t i = 0; i < k * n; i++) {
        b[i] = (float)(i % 7) * 0.2f - 0.6f;
    }

    if (verbose) {
        printf("\n--- input generation ---\n");
        printf("A[%zu,%zu]: a[i] = (i %% 13) * 0.1 - 0.6\n", m, k);
        for (size_t i = 0; i < m * k; i++) {
            printf("  a[%zu] = %f\n", i, a[i]);
        }
        printf("B[%zu,%zu]: b[i] = (i %% 7) * 0.2 - 0.6\n", k, n);
        for (size_t i = 0; i < k * n; i++) {
            printf("  b[%zu] = %f\n", i, b[i]);
        }
    }

    GpuBridgeTensorDesc a_dev, b_dev, c_dev;
    const size_t a_shape[2] = { m, k };
    const size_t b_shape[2] = { k, n };
    const size_t c_shape[2] = { m, n };
    if (gpuBridgeTensorAlloc(&a_dev, GPUBRIDGE_TYPE_F32, 2, a_shape) != 0 ||
        gpuBridgeTensorAlloc(&b_dev, GPUBRIDGE_TYPE_F32, 2, b_shape) != 0 ||
        gpuBridgeTensorAlloc(&c_dev, GPUBRIDGE_TYPE_F32, 2, c_shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeMemcpy(a_dev.device_ptr, a, m * k * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(b_dev.device_ptr, b, k * n * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR kernel;
    GpuBridgeTensorArg kernel_args[3];
    gpuBridgeIrInitMatmulF32(&kernel, kernel_args, m, k, n);

    void* launch_args[3] = { a_dev.device_ptr, b_dev.device_ptr,
        c_dev.device_ptr };
    if (gpuBridgeLaunchTensorKernel(&kernel, launch_args, 3) != 0) {
        fprintf(stderr, "gpuBridgeLaunchTensorKernel failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeDeviceSynchronize() != 0) {
        fprintf(stderr, "gpuBridgeDeviceSynchronize failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeMemcpy(c, c_dev.device_ptr, m * n * sizeof(float),
            GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy D2H failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    if (verbose) {
        /* Worked example: fully expand the dot product for C[0,0] so the
         * exact formula behind every other element (just i,j substituted)
         * is visible once, instead of implied. */
        printf("\n--- worked example: C[0,0] = sum_p A[0,p] * B[p,0] ---\n");
        double c00 = 0.0;
        for (size_t p = 0; p < k; p++) {
            double term = (double)a[0 * k + p] * (double)b[p * n + 0];
            c00 += term;
            printf("  p=%zu: A[0,%zu]=%f * B[%zu,0]=%f = %f  (running sum = %f)\n",
                p, p, a[0 * k + p], p, b[p * n + 0], term, c00);
        }
        printf("expected C[0,0] = %f, device-computed C[0,0] = %f\n\n", c00,
            (double)c[0]);
    }

    /* Double-precision CPU reference (spec2.md 7.1). */
    double max_rel_error = 0.0;
    size_t max_i = 0, max_j = 0;
    if (verbose) {
        printf("--- per-element expected vs. actual (tolerance: rel <= 1e-3) ---\n");
    }
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double expected = 0.0;
            for (size_t p = 0; p < k; p++) {
                expected += (double)a[i * k + p] * (double)b[p * n + j];
            }
            double actual = (double)c[i * n + j];
            double error = fabs(actual - expected);
            double rel = error / fmax(1.0, fabs(expected));
            if (verbose) {
                printf("  C[%zu,%zu]: expected=%f actual=%f abs_error=%f "
                    "rel_error=%f %s\n",
                    i, j, expected, actual, error, rel,
                    rel <= 1e-3 ? "PASS" : "FAIL");
            }
            if (rel > max_rel_error) {
                max_rel_error = rel;
                max_i = i;
                max_j = j;
            }
        }
    }

    /* Tolerance from spec2.md 7.1: abs(C-ref) <= 1e-3 * max(1, abs(ref)). */
    bool pass = max_rel_error <= 1e-3;

    printf("\nGPUBridge matmul_f32 test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("shape: A[%zu,%zu] B[%zu,%zu] -> C[%zu,%zu]\n", m, k, k, n, m, n);
    printf("result: %s\n", pass ? "PASS" : "FAIL");
    printf("max relative error: %f\n", max_rel_error);
    if (verbose) {
        printf("max relative error occurred at C[%zu,%zu]\n", max_i, max_j);
        printf("tolerance rule: PASS requires max_rel_error <= 1e-3, where "
            "rel_error = |actual - expected| / max(1, |expected|)\n");
    }

    gpuBridgeTensorFree(&a_dev);
    gpuBridgeTensorFree(&b_dev);
    gpuBridgeTensorFree(&c_dev);
    free(a);
    free(b);
    free(c);
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
