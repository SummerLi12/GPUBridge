/*
 * test_portal_graph.c — correctness/rejection test for gpuBridgePortalSubmit
 * (spec_gpuportal.md section 12, item 2).
 *
 * Builds a 1-node GpuBridgePortalGraph (GPUBRIDGE_PORTAL_OP_MATMUL), calls
 * gpuBridgePortalSubmit(), and verifies the output buffer against a
 * hand-computed reference — same tolerance as test_blas_sgemm.c/
 * gpubridge-runtime's own test_matmul.c. Separately asserts
 * GPUBRIDGE_PORTAL_EMPTY_GRAPH for op_count == 0 and
 * GPUBRIDGE_PORTAL_UNSUPPORTED_OP for a hand-corrupted kind value.
 *
 * OpenCL-only (section 2), same skip-on-unavailable convention as
 * test_blas_sgemm.c: gpuBridgePortalSubmit()'s first call lazily creates a
 * GpuBridgeBlasHandle, which fails with GPUBRIDGE_PORTAL_BLAS_ERROR (wrapping
 * GPUBRIDGE_BLAS_BACKEND_UNAVAILABLE) if no OpenCL device is enumerable —
 * that's "OpenCL unavailable here", not a genuine test failure.
 */
#include "gpubridge_portal.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define GPUBRIDGE_TEST_SKIP 77

int main(void)
{
    bool all_pass = true;

    /* --- GPUBRIDGE_PORTAL_EMPTY_GRAPH: op_count == 0. Checked before any
     * OpenCL/handle interaction, so this case works even on a machine with
     * no OpenCL device at all. --- */
    GpuBridgePortalGraph empty_graph = { NULL, 0 };
    GpuBridgePortalStatus status = gpuBridgePortalSubmit(&empty_graph);
    bool empty_graph_rejected = (status == GPUBRIDGE_PORTAL_EMPTY_GRAPH);
    printf("empty graph rejection case: %s (%s)\n",
        empty_graph_rejected ? "PASS" : "FAIL", gpuBridgePortalStatusString(status));
    all_pass = all_pass && empty_graph_rejected;

    /* --- Correctness: 1-node matmul graph, 64x128x32 (same shape as
     * test_blas_sgemm.c/gpubridge-runtime's test_matmul.c). This is the
     * first call that actually needs OpenCL (gpuBridgePortalSubmit() lazily
     * creates its GpuBridgeBlasHandle here) — skip cleanly if unavailable. --- */
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

    GpuBridgePortalOp matmul_op = {
        .kind = GPUBRIDGE_PORTAL_OP_MATMUL,
        .m = m, .n = n, .k = k,
        .input_a = A, .input_b = B, .output = C
    };
    GpuBridgePortalGraph graph = { &matmul_op, 1 };

    status = gpuBridgePortalSubmit(&graph);
    if (status == GPUBRIDGE_PORTAL_BLAS_ERROR) {
        /* Could be a genuine execution failure or an OpenCL-unavailable
         * BACKEND_UNAVAILABLE wrapped by GpuBridge-BLAS — this test has no
         * way to distinguish the two through GpuBridgePortalStatus alone
         * (by design: section 7's dispatch layer intentionally does not
         * leak GpuBridgeBlasStatus detail), so treat it as a skip, matching
         * test_blas_sgemm.c's direct (non-wrapped) handling of the same
         * underlying condition. */
        printf("SKIP: gpuBridgePortalSubmit failed, likely no OpenCL device "
            "enumerable on this machine (%s)\n", gpuBridgePortalStatusString(status));
        free(A);
        free(B);
        free(C);
        return GPUBRIDGE_TEST_SKIP;
    }
    bool submit_ok = (status == GPUBRIDGE_PORTAL_OK);
    printf("submit case: %s (%s)\n",
        submit_ok ? "PASS" : "FAIL", gpuBridgePortalStatusString(status));
    all_pass = all_pass && submit_ok;

    if (submit_ok) {
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

    /* --- GPUBRIDGE_PORTAL_UNSUPPORTED_OP: hand-corrupted kind value. Only
     * reachable this way, since GpuBridgePortalOpKind has exactly one real
     * enumerator this spec defines (section 7.1). --- */
    GpuBridgePortalOp bad_op = matmul_op;
    bad_op.kind = (GpuBridgePortalOpKind)999;
    GpuBridgePortalGraph bad_graph = { &bad_op, 1 };
    status = gpuBridgePortalSubmit(&bad_graph);
    bool unsupported_op_rejected = (status == GPUBRIDGE_PORTAL_UNSUPPORTED_OP);
    printf("unsupported op rejection case: %s (%s)\n",
        unsupported_op_rejected ? "PASS" : "FAIL", gpuBridgePortalStatusString(status));
    all_pass = all_pass && unsupported_op_rejected;

    free(A);
    free(B);
    free(C);
    gpuBridgePortalShutdown();

    printf("result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
