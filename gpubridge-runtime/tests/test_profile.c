/*
 * test_profile.c — Milestone 1.5 profiling correctness test (spec1.5.md
 * section 15, items 3-4).
 *
 * Same init->malloc->memcpy->launch->sync->memcpy sequence as
 * tests/test_vecadd.c, but instead of checking vector_add_f32's numeric
 * result, this test checks that gpuBridgeGetProfileStats() actually observed
 * the work that happened: non-zero transfer bytes in both directions and a
 * non-zero synchronization count. Timing fields are only asserted >= 0.0
 * (not > 0.0) — a single fast operation can legitimately measure 0.00ms at
 * clock_gettime's precision, so a strict > 0 assertion would be flaky.
 *
 * gpuBridgeGetProfileStats() is called *before* gpuBridgeShutdown(), which
 * prints (only if GPUBRIDGE_PROFILE=1, set by this test's ctest registration)
 * and then clears the counters (spec1.5.md section 9.3) — so the "GPUBridge
 * profile:" block below is a side effect of shutdown, not something this
 * test prints itself.
 *
 * Exit code doubles as the pass/fail signal for `ctest`: 0 on PASS, 1 on
 * FAIL or any runtime failure.
 */
#include "gpubridge_runtime.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const size_t n = 1 << 20;

    if (gpuBridgeInit() != 0) {
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    float* a = malloc(n * sizeof(float));
    float* b = malloc(n * sizeof(float));
    float* c = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(n - i) * 0.25f;
    }

    void* dev_a = NULL;
    void* dev_b = NULL;
    void* dev_c = NULL;
    if (gpuBridgeMalloc(&dev_a, n * sizeof(float)) != 0 ||
        gpuBridgeMalloc(&dev_b, n * sizeof(float)) != 0 ||
        gpuBridgeMalloc(&dev_c, n * sizeof(float)) != 0) {
        fprintf(stderr, "gpuBridgeMalloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeMemcpy(dev_a, a, n * sizeof(float), GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(dev_b, b, n * sizeof(float), GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeKernelIR kernel;
    GpuBridgeBufferArg kernel_args[3];
    gpuBridgeIrInitVectorAddF32(&kernel, kernel_args, n);

    void* launch_args[3] = { dev_a, dev_b, dev_c };
    if (gpuBridgeLaunchKernel(&kernel, launch_args, 3) != 0) {
        fprintf(stderr, "gpuBridgeLaunchKernel failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeDeviceSynchronize() != 0) {
        fprintf(stderr, "gpuBridgeDeviceSynchronize failed: %s\n",
            gpuBridgeGetLastErrorString());
        return 1;
    }

    if (gpuBridgeMemcpy(c, dev_c, n * sizeof(float), GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy D2H failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeProfileStats stats;
    if (gpuBridgeGetProfileStats(&stats) != 0) {
        fprintf(stderr, "gpuBridgeGetProfileStats failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    bool pass = stats.host_to_device_bytes > 0
        && stats.device_to_host_bytes > 0
        && stats.synchronization_count > 0
        && stats.host_to_device_time_ms >= 0.0
        && stats.kernel_time_ms >= 0.0
        && stats.device_to_host_time_ms >= 0.0;

    printf("GPUBridge profile_stats test\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(gpuBridgeGetSelectedBackend()));
    printf("host_to_device_bytes: %zu\n", stats.host_to_device_bytes);
    printf("device_to_host_bytes: %zu\n", stats.device_to_host_bytes);
    printf("synchronization_count: %d\n", stats.synchronization_count);
    printf("result: %s\n", pass ? "PASS" : "FAIL");

    gpuBridgeFree(dev_a);
    gpuBridgeFree(dev_b);
    gpuBridgeFree(dev_c);
    free(a);
    free(b);
    free(c);
    /* Prints the GPUBridge profile: block (only if GPUBRIDGE_PROFILE=1) and
     * clears the counters, per spec1.5.md section 9.3. */
    gpuBridgeShutdown();

    return pass ? 0 : 1;
}
