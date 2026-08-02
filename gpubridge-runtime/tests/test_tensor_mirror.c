/*
 * test_tensor_mirror.c — Milestone 2.5 (spec2.5.md section 11.2):
 * GpuBridgeTensorMirror lazy-transfer / staleness-tracking correctness test.
 *
 * Sequence:
 *   1. Bind a host array to a freshly allocated device tensor.
 *   2. gpuBridgeTensorEnsureDeviceCurrent() — must issue exactly one transfer
 *      (host_to_device_bytes grows by `bytes`).
 *   3. Call it again with no intervening MarkHostModified — must issue no
 *      transfer at all (host_to_device_bytes delta == 0), proving the
 *      already-current case is a real no-op.
 *   4. gpuBridgeTensorMarkHostModified() then EnsureDeviceCurrent() again —
 *      must issue a transfer again (delta == bytes).
 *
 * gpuBridgeGetProfileStats()'s host_to_device_bytes counter (spec1.5.md
 * section 9) is the observable proxy for "was a gpuBridgeMemcpy actually
 * issued" — this test never inspects gpubridge_runtime.c internals directly.
 *
 * Exit code doubles as the pass/fail signal for `ctest`: 0 on PASS, 1 on
 * FAIL or any runtime failure.
 */
#include "gpubridge_runtime.h"

#include <stdio.h>
#include <stdlib.h>

static int get_h2d_bytes(size_t* out)
{
    GpuBridgeProfileStats stats;
    if (gpuBridgeGetProfileStats(&stats) != 0) {
        fprintf(stderr, "gpuBridgeGetProfileStats failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    *out = stats.host_to_device_bytes;
    return 0;
}

int main(void)
{
    const size_t n = 64;
    const size_t bytes = n * sizeof(float);

    if (gpuBridgeInit() != 0) {
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    printf("GPUBridge tensor_mirror test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));

    float* host = malloc(bytes);
    for (size_t i = 0; i < n; i++) {
        host[i] = (float)i;
    }

    GpuBridgeTensorDesc desc;
    size_t shape[2] = { n, 0 };
    if (gpuBridgeTensorAlloc(&desc, GPUBRIDGE_TYPE_F32, 1, shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorMirror mirror;
    if (gpuBridgeTensorMirrorCreate(&mirror, host, &desc, bytes) != 0) {
        fprintf(stderr, "gpuBridgeTensorMirrorCreate failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    bool all_pass = true;
    size_t before, after;

    /* Step 1: first EnsureDeviceCurrent -- device_valid starts false, so
     * exactly one transfer must be issued. */
    if (get_h2d_bytes(&before) != 0) return 1;
    if (gpuBridgeTensorEnsureDeviceCurrent(&mirror) != 0) {
        fprintf(stderr, "gpuBridgeTensorEnsureDeviceCurrent failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
    if (get_h2d_bytes(&after) != 0) return 1;
    bool step1_pass = (after - before) == bytes;
    printf("step1_initial_transfer: delta=%zu expected=%zu %s\n",
        after - before, bytes, step1_pass ? "PASS" : "FAIL");
    all_pass = all_pass && step1_pass;

    /* Step 2: second EnsureDeviceCurrent, no intervening MarkHostModified --
     * device_valid is already true, so this must be a real no-op. */
    before = after;
    if (gpuBridgeTensorEnsureDeviceCurrent(&mirror) != 0) {
        fprintf(stderr, "gpuBridgeTensorEnsureDeviceCurrent failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
    if (get_h2d_bytes(&after) != 0) return 1;
    bool step2_pass = (after - before) == 0;
    printf("step2_skipped_redundant_transfer: delta=%zu expected=0 %s\n",
        after - before, step2_pass ? "PASS" : "FAIL");
    all_pass = all_pass && step2_pass;

    /* Step 3: mark host modified, then EnsureDeviceCurrent again -- must
     * issue a transfer again. */
    before = after;
    if (gpuBridgeTensorMarkHostModified(&mirror) != 0) {
        fprintf(stderr, "gpuBridgeTensorMarkHostModified failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
    if (gpuBridgeTensorEnsureDeviceCurrent(&mirror) != 0) {
        fprintf(stderr, "gpuBridgeTensorEnsureDeviceCurrent failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }
    if (get_h2d_bytes(&after) != 0) return 1;
    bool step3_pass = (after - before) == bytes;
    printf("step3_transfer_after_mark_modified: delta=%zu expected=%zu %s\n",
        after - before, bytes, step3_pass ? "PASS" : "FAIL");
    all_pass = all_pass && step3_pass;

    printf("result: %s\n", all_pass ? "PASS" : "FAIL");

    gpuBridgeTensorMirrorDestroy(&mirror);
    free(host);
    gpuBridgeShutdown();

    return all_pass ? 0 : 1;
}
