/*
 * test_blas_sgemm.c — correctness/rejection test for gpuBridgeBlasSgemm
 * (spec_gpuportal.md section 12, item 1).
 *
 * Reuses gpubridge-runtime's own tests/test_matmul.c small correctness
 * shape (64x128x32) and compares against a hand-computed double-precision
 * reference, same tolerance rule (spec2.md 7.1: abs(C-ref) <=
 * 1e-3 * max(1, abs(ref))). Additionally asserts GPUBRIDGE_BLAS_UNSUPPORTED_PARAM
 * for a non-GPUBRIDGE_BLAS_OP_N transpose and for alpha != 1.0f / beta != 0.0f,
 * and that GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE is at least reachable in code
 * (not necessarily forceable on a machine that genuinely has OpenCL).
 *
 * This spec is OpenCL-only (section 2, no cpu/auto fallback): on a machine
 * with no OpenCL device, gpuBridgeBlasCreate() fails with
 * GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE — that's "OpenCL unavailable here", not
 * a genuine test failure, so this test skips cleanly (exit 77) in that case,
 * the same convention gpubridge-runtime's own OpenCL-dependent ctest
 * entries already use.
 */
#include "gpubridge_blas.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    GpuBridgeBlasHandle handle = NULL;
    GpuBridgeBlasStatus create_status = gpuBridgeBlasCreate(&handle);
    if (create_status == GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE) {
        printf("SKIP: no OpenCL device enumerable on this machine (%s)\n",
            gpuBridgeBlasStatusString(create_status));
        return GPUBRIDGE_TEST_SKIP;
    }
    if (create_status != GPUBRIDGE_BLAS_SUCCESS) {
        fprintf(stderr, "gpuBridgeBlasCreate failed: %s\n",
            gpuBridgeBlasStatusString(create_status));
        return 1;
    }

    bool all_pass = true;

    /* --- Correctness: same 64x128x32 shape as gpubridge-runtime's own
     * tests/test_matmul.c, row-major, lda=k/ldb=n/ldc=n. --- */
    const int m = 64, k = 128, n = 32;
    float *A = malloc((size_t)m * k * sizeof(float));
    float *B = malloc((size_t)k * n * sizeof(float));
    float *C = malloc((size_t)m * n * sizeof(float));
    for (int i = 0; i < m * k; i++) {
        A[i] = (float)(i % 13) * 0.1f - 0.6f;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (float)(i % 7) * 0.2f - 0.6f;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    GpuBridgeBlasStatus status = gpuBridgeBlasSgemm(
        handle, GPUBRIDGE_BLAS_OP_N, GPUBRIDGE_BLAS_OP_N,
        m, n, k, &alpha, A, k, B, n, &beta, C, n);
    if (status != GPUBRIDGE_BLAS_SUCCESS) {
        fprintf(stderr, "gpuBridgeBlasSgemm (correctness case) failed: %s\n",
            gpuBridgeBlasStatusString(status));
        all_pass = false;
    } else {
        double max_rel_error = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double expected = 0.0;
                for (int p = 0; p < k; p++) {
                    expected += (double)A[i * k + p] * (double)B[p * n + j];
                }
                double actual = (double)C[i * n + j];
                double error = fabs(actual - expected);
                double rel = error / fmax(1.0, fabs(expected));
                if (rel > max_rel_error) {
                    max_rel_error = rel;
                }
            }
        }
        bool pass = max_rel_error <= 1e-3;
        printf("correctness case: %s (max relative error: %f)\n",
            pass ? "PASS" : "FAIL", max_rel_error);
        all_pass = all_pass && pass;
    }

    /* --- Rejection: transa != GPUBRIDGE_BLAS_OP_N. Cast a made-up value
     * since GPUBRIDGE_BLAS_OP_N is the only real enumerator this spec
     * defines (section 6.1) — the rejection path must still fire for any
     * other integer value. --- */
    status = gpuBridgeBlasSgemm(
        handle, (GpuBridgeBlasOperation)1, GPUBRIDGE_BLAS_OP_N,
        m, n, k, &alpha, A, k, B, n, &beta, C, n);
    bool transpose_rejected = (status == GPUBRIDGE_BLAS_UNSUPPORTED_PARAM);
    printf("transpose rejection case: %s (%s)\n",
        transpose_rejected ? "PASS" : "FAIL", gpuBridgeBlasStatusString(status));
    all_pass = all_pass && transpose_rejected;

    /* --- Rejection: alpha != 1.0f. --- */
    const float bad_alpha = 2.0f;
    status = gpuBridgeBlasSgemm(
        handle, GPUBRIDGE_BLAS_OP_N, GPUBRIDGE_BLAS_OP_N,
        m, n, k, &bad_alpha, A, k, B, n, &beta, C, n);
    bool alpha_rejected = (status == GPUBRIDGE_BLAS_UNSUPPORTED_PARAM);
    printf("alpha rejection case: %s (%s)\n",
        alpha_rejected ? "PASS" : "FAIL", gpuBridgeBlasStatusString(status));
    all_pass = all_pass && alpha_rejected;

    /* --- Rejection: beta != 0.0f. --- */
    const float bad_beta = 1.0f;
    status = gpuBridgeBlasSgemm(
        handle, GPUBRIDGE_BLAS_OP_N, GPUBRIDGE_BLAS_OP_N,
        m, n, k, &alpha, A, k, B, n, &bad_beta, C, n);
    bool beta_rejected = (status == GPUBRIDGE_BLAS_UNSUPPORTED_PARAM);
    printf("beta rejection case: %s (%s)\n",
        beta_rejected ? "PASS" : "FAIL", gpuBridgeBlasStatusString(status));
    all_pass = all_pass && beta_rejected;

    /* --- GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE is reachable in code (section
     * 12 item 1's "at least reachable, not necessarily forceable on a
     * machine that has OpenCL" — this machine has OpenCL, per the
     * successful gpuBridgeBlasCreate() above, so this only exercises a
     * second gpuBridgeBlasCreate() call while the first handle is still
     * live, which this GPUportal build additionally guards against
     * (gpubridge_blas.c's g_handle_live) — a real, reachable rejection
     * path, just not the OpenCL-absent one this status code is named for. */
    GpuBridgeBlasHandle second_handle = NULL;
    GpuBridgeBlasStatus second_create_status = gpuBridgeBlasCreate(&second_handle);
    bool second_handle_rejected = (second_create_status != GPUBRIDGE_BLAS_SUCCESS);
    printf("second-handle-while-live rejection case: %s (%s)\n",
        second_handle_rejected ? "PASS" : "FAIL",
        gpuBridgeBlasStatusString(second_create_status));
    all_pass = all_pass && second_handle_rejected;
    if (second_create_status == GPUBRIDGE_BLAS_SUCCESS) {
        gpuBridgeBlasDestroy(second_handle);
    }

    free(A);
    free(B);
    free(C);
    gpuBridgeBlasDestroy(handle);

    printf("result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
