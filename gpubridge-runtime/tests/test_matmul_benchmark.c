/*
 * test_matmul_benchmark.c — fair, clean-timing-scope GEMM benchmark
 * (spec3.md section 9), comparing the optional vendor GEMM path
 * (CLBlast/OpenBLAS) against this project's hand-written kernel/naive loop
 * at a scale (512x512x512) large enough for the difference to matter,
 * unlike test_matmul.c's small 64x128x32 correctness-only shape.
 *
 * Exit code: 0 iff every correctness check that actually ran (naive
 * always; vendor only if a vendor library is loaded) passed. A missing
 * vendor library is NOT a test failure — it is an honestly-reported
 * absence (spec3.md section 9.4), consistent with
 * gpubridge_performance_spec.md section 3.10's "never silently fail into a
 * slow or incorrect path... fallback must be explicit in diagnostics."
 *
 * The raw speedup number itself is never a pass/fail gate (spec3.md
 * section 9.2's rationale: hardware/library-version-dependent, would make
 * the test flaky across machines).
 */
/* Needed for setenv/unsetenv (POSIX.1-2001) under -std=c11, same reason
 * gpubridge_runtime.c defines _POSIX_C_SOURCE for clock_gettime. */
#define _POSIX_C_SOURCE 200809L

#include "gpubridge_runtime.h"
#include "bench_common.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef GPUBRIDGE_BUILD_TYPE
#define GPUBRIDGE_BUILD_TYPE "unspecified"
#endif

/* This benchmark's own vendor/naive alternating sample counts still use
 * their original name, GPUBRIDGE_GEMM_BENCH_ITERS — deliberately kept
 * separate from bench_common.h's GPUBRIDGE_BENCH_ITERS (spec5.md section 4
 * non-goal 6 / section 5): this file's alternating-path logic has no
 * equivalent in the other four *_benchmark_test files, so it keeps its own
 * env var rather than being forced to share one that doesn't quite mean
 * the same thing everywhere. median()/read_iters()'s bodies moved to
 * bench_common.c verbatim (gpuBridgeBenchMedian/gpuBridgeBenchReadIters);
 * this file's read_iters() wraps the GEMM-specific env var name around the
 * same fallback-on-invalid logic. */
static int read_iters(void)
{
    const char* val = getenv("GPUBRIDGE_GEMM_BENCH_ITERS");
    if (val == NULL) {
        return GPUBRIDGE_BENCH_DEFAULT_ITERS;
    }
    int parsed = atoi(val);
    return parsed > 0 ? parsed : GPUBRIDGE_BENCH_DEFAULT_ITERS;
}

/* One matmul_f32 launch through the same GpuBridgeTensorKernelIR every time —
 * whether it actually runs the vendor path or the naive path is decided
 * entirely by GPUBRIDGE_VENDOR_GEMM_DISABLE at the moment of the call
 * (checked internally by the backend's launch_matmul_f32 dispatch), not by
 * anything this function does. Returns 0 on success, matching every
 * gpuBridge* function's convention. */
static int launch_once(const GpuBridgeTensorKernelIR* kernel, void* a_dev,
    void* b_dev, void* c_dev)
{
    void* launch_args[3] = { a_dev, b_dev, c_dev };
    return gpuBridgeLaunchTensorKernel(kernel, launch_args, 3);
}

int main(void)
{
    const size_t n_dim = 512;
    const int iters = read_iters();

    /* Captured once, before this benchmark's own setenv/unsetenv logic
     * takes ownership of GPUBRIDGE_VENDOR_GEMM_DISABLE (spec3.md section
     * 9.2) — "enabled" (section 8.1's term) describes the ambient
     * environment this process was launched with, not whatever transient
     * state the alternating loop happens to leave it in. */
    const char* ambient_disable = getenv("GPUBRIDGE_VENDOR_GEMM_DISABLE");
    bool ambient_vendor_disabled = ambient_disable != NULL &&
        strcmp(ambient_disable, "1") == 0;

    if (gpuBridgeInit() != 0) {
        fprintf(stderr, "gpuBridgeInit failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeBackendCaps caps;
    if (gpuBridgeGetBackendCaps(&caps) != 0) {
        fprintf(stderr, "gpuBridgeGetBackendCaps failed: %s\n", gpuBridgeGetLastErrorString());
        gpuBridgeShutdown();
        return 1;
    }
    bool vendor_loaded = caps.vendor_gemm_loaded;

    GpuBridgeBackendKind selected = gpuBridgeGetSelectedBackend();
    const char* vendor_name =
        (selected == GPUBRIDGE_BACKEND_OPENCL) ? "clblast" :
        (selected == GPUBRIDGE_BACKEND_CPU)    ? "openblas" : "vendor";

    /* --- Step 2: untimed setup --- */
    float* a = malloc(n_dim * n_dim * sizeof(float));
    float* b = malloc(n_dim * n_dim * sizeof(float));
    float* c = malloc(n_dim * n_dim * sizeof(float));
    for (size_t i = 0; i < n_dim * n_dim; i++) {
        a[i] = (float)(i % 13) * 0.1f - 0.6f;
        b[i] = (float)(i % 7) * 0.2f - 0.6f;
    }

    GpuBridgeTensorDesc a_dev, b_dev, c_dev;
    const size_t shape[2] = { n_dim, n_dim };
    if (gpuBridgeTensorAlloc(&a_dev, GPUBRIDGE_TYPE_F32, 2, shape) != 0 ||
        gpuBridgeTensorAlloc(&b_dev, GPUBRIDGE_TYPE_F32, 2, shape) != 0 ||
        gpuBridgeTensorAlloc(&c_dev, GPUBRIDGE_TYPE_F32, 2, shape) != 0) {
        fprintf(stderr, "gpuBridgeTensorAlloc failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }
    if (gpuBridgeMemcpy(a_dev.device_ptr, a, n_dim * n_dim * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0 ||
        gpuBridgeMemcpy(b_dev.device_ptr, b, n_dim * n_dim * sizeof(float),
            GPUBRIDGE_MEMCPY_HOST_TO_DEVICE) != 0) {
        fprintf(stderr, "gpuBridgeMemcpy H2D failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    GpuBridgeTensorKernelIR kernel;
    GpuBridgeTensorArg kernel_args[3];
    gpuBridgeIrInitMatmulF32(&kernel, kernel_args, n_dim, n_dim, n_dim);

    /* --- Step 3: clean quiescent baseline --- */
    if (gpuBridgeDeviceSynchronize() != 0) {
        fprintf(stderr, "gpuBridgeDeviceSynchronize failed: %s\n", gpuBridgeGetLastErrorString());
        return 1;
    }

    /* --- Step 5: independent warmup, untimed, discarded --- */
    if (vendor_loaded) {
        for (int i = 0; i < GPUBRIDGE_BENCH_WARMUP_ROUNDS; i++) {
            launch_once(&kernel, a_dev.device_ptr, b_dev.device_ptr, c_dev.device_ptr);
            gpuBridgeDeviceSynchronize();
        }
    }
    setenv("GPUBRIDGE_VENDOR_GEMM_DISABLE", "1", 1);
    for (int i = 0; i < GPUBRIDGE_BENCH_WARMUP_ROUNDS; i++) {
        launch_once(&kernel, a_dev.device_ptr, b_dev.device_ptr, c_dev.device_ptr);
        gpuBridgeDeviceSynchronize();
    }
    unsetenv("GPUBRIDGE_VENDOR_GEMM_DISABLE");

    /* --- Step 6: alternating timed loop --- */
    double* vendor_samples = vendor_loaded ? malloc((size_t)iters * sizeof(double)) : NULL;
    double* naive_samples = malloc((size_t)iters * sizeof(double));

    for (int round = 0; round < iters; round++) {
        GpuBridgeProfileStats before, after;

        if (vendor_loaded) {
            gpuBridgeDeviceSynchronize();
            gpuBridgeGetProfileStats(&before);
            launch_once(&kernel, a_dev.device_ptr, b_dev.device_ptr, c_dev.device_ptr);
            gpuBridgeDeviceSynchronize();
            gpuBridgeGetProfileStats(&after);
            vendor_samples[round] = gpuBridgeBenchGpuPathMs(&after) - gpuBridgeBenchGpuPathMs(&before);
        }

        setenv("GPUBRIDGE_VENDOR_GEMM_DISABLE", "1", 1);
        gpuBridgeDeviceSynchronize();
        gpuBridgeGetProfileStats(&before);
        launch_once(&kernel, a_dev.device_ptr, b_dev.device_ptr, c_dev.device_ptr);
        gpuBridgeDeviceSynchronize();
        gpuBridgeGetProfileStats(&after);
        naive_samples[round] = gpuBridgeBenchGpuPathMs(&after) - gpuBridgeBenchGpuPathMs(&before);
        unsetenv("GPUBRIDGE_VENDOR_GEMM_DISABLE");
    }

    /* gpuBridgeBenchMedian() sorts in place, so p95/p99 (spec5.md section 7)
     * are computed from the same now-sorted arrays afterward — order
     * matters: percentile must run after median's sort, not before. */
    double naive_median_ms = gpuBridgeBenchMedian(naive_samples, iters);
    double naive_p95_ms = gpuBridgeBenchPercentile(naive_samples, iters, 0.95);
    double naive_p99_ms = gpuBridgeBenchPercentile(naive_samples, iters, 0.99);
    double vendor_median_ms = vendor_loaded ? gpuBridgeBenchMedian(vendor_samples, iters) : 0.0;
    double vendor_p95_ms = vendor_loaded ? gpuBridgeBenchPercentile(vendor_samples, iters, 0.95) : 0.0;
    double vendor_p99_ms = vendor_loaded ? gpuBridgeBenchPercentile(vendor_samples, iters, 0.99) : 0.0;
    free(naive_samples);
    free(vendor_samples);

    /* --- Step 8: correctness, untimed, after all sampling --- */
    bool correctness_naive_pass = false;
    double correctness_naive_err = 0.0;
    {
        setenv("GPUBRIDGE_VENDOR_GEMM_DISABLE", "1", 1);
        if (launch_once(&kernel, a_dev.device_ptr, b_dev.device_ptr, c_dev.device_ptr) == 0 &&
            gpuBridgeDeviceSynchronize() == 0 &&
            gpuBridgeMemcpy(c, c_dev.device_ptr, n_dim * n_dim * sizeof(float),
                GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) == 0) {
            double max_rel_error = 0.0;
            for (size_t i = 0; i < n_dim; i++) {
                for (size_t j = 0; j < n_dim; j++) {
                    double expected = 0.0;
                    for (size_t p = 0; p < n_dim; p++) {
                        expected += (double)a[i * n_dim + p] * (double)b[p * n_dim + j];
                    }
                    double actual = (double)c[i * n_dim + j];
                    double rel = fabs(actual - expected) / fmax(1.0, fabs(expected));
                    if (rel > max_rel_error) { max_rel_error = rel; }
                }
            }
            correctness_naive_err = max_rel_error;
            correctness_naive_pass = max_rel_error <= 1e-3;
        }
        unsetenv("GPUBRIDGE_VENDOR_GEMM_DISABLE");
    }

    bool correctness_vendor_pass = false;
    double correctness_vendor_err = 0.0;
    if (vendor_loaded) {
        if (launch_once(&kernel, a_dev.device_ptr, b_dev.device_ptr, c_dev.device_ptr) == 0 &&
            gpuBridgeDeviceSynchronize() == 0 &&
            gpuBridgeMemcpy(c, c_dev.device_ptr, n_dim * n_dim * sizeof(float),
                GPUBRIDGE_MEMCPY_DEVICE_TO_HOST) == 0) {
            double max_rel_error = 0.0;
            for (size_t i = 0; i < n_dim; i++) {
                for (size_t j = 0; j < n_dim; j++) {
                    double expected = 0.0;
                    for (size_t p = 0; p < n_dim; p++) {
                        expected += (double)a[i * n_dim + p] * (double)b[p * n_dim + j];
                    }
                    double actual = (double)c[i * n_dim + j];
                    double rel = fabs(actual - expected) / fmax(1.0, fabs(expected));
                    if (rel > max_rel_error) { max_rel_error = rel; }
                }
            }
            correctness_vendor_err = max_rel_error;
            correctness_vendor_pass = max_rel_error <= 1e-3;
        }
    }

    /* --- Step 9: report (spec3.md section 9.4, extended spec5.md section
     * 6.5/7 with device/hardware-tier/tail-latency/throughput) --- */
    char device_name[128];
    gpuBridgeGetSelectedDeviceName(device_name, sizeof(device_name));
    size_t mem_limit = 0;
    gpuBridgeGetDeviceMemoryLimit(&mem_limit);
    GpuBridgeHardwareTier tier = GPUBRIDGE_HARDWARE_TIER_UNKNOWN;
    gpuBridgeGetHardwareTier(&tier);

    const char* iters_source = getenv("GPUBRIDGE_GEMM_BENCH_ITERS") != NULL
        ? "" : " (GPUBRIDGE_GEMM_BENCH_ITERS unset, using default)";

    printf("GPUBridge matmul_f32 GEMM benchmark\n");
    printf("backend: %s\n", gpuBridgeBackendKindToString(selected));
    printf("device: %s\n", device_name);
    /* heuristic, device-name-based — see gpuBridgeClassifyHardwareTierFromDeviceName() */
    printf("hardware_tier: %s\n", gpuBridgeHardwareTierToString(tier));
    printf("device_memory_limit_bytes: %zu\n", mem_limit);
    printf("shape: A[%zu,%zu] B[%zu,%zu] -> C[%zu,%zu]\n", n_dim, n_dim, n_dim, n_dim, n_dim, n_dim);
    printf("iterations: %d%s\n", iters, iters_source);
    if (iters < 100) {
        printf("  (raise GPUBRIDGE_GEMM_BENCH_ITERS for more p95/p99 resolution — "
            "at %d samples, p95 and p99 both resolve to the largest sample)\n", iters);
    }
    printf("build type: %s\n", GPUBRIDGE_BUILD_TYPE);
    if (selected == GPUBRIDGE_BACKEND_OPENCL) {
        printf("timing metric: kernel_time_ms + synchronization_time_ms (host-side, OpenCL)\n");
    } else {
        printf("timing metric: kernel_time_ms (host-side, direct call)\n");
    }
    printf("vendor library: %s - loaded: %s, enabled: %s\n", vendor_name,
        vendor_loaded ? "yes" : "no",
        (vendor_loaded && !ambient_vendor_disabled) ? "yes" : "no");
    if (selected == GPUBRIDGE_BACKEND_CPU && vendor_loaded) {
        const char* threads_env = getenv("OPENBLAS_NUM_THREADS");
        if (threads_env != NULL) {
            printf("OpenBLAS threads: %s (OPENBLAS_NUM_THREADS=%s)\n", threads_env, threads_env);
        } else {
            printf("OpenBLAS threads: default/unknown (OPENBLAS_NUM_THREADS not set)\n");
        }
    }
    printf("naive_median_ms: %.2f\n", naive_median_ms);
    printf("naive_p95_ms: %.2f\n", naive_p95_ms);
    printf("naive_p99_ms: %.2f\n", naive_p99_ms);
    /* throughput_ops_per_sec: output-element rate, not FLOPs — same
     * definition gpubridge_diagnostics.c's Phase 4 avg_kernel_latency_ms/
     * throughput_ops_per_sec derivation already uses (spec5.md section
     * 6.5), reused here instead of introducing a second throughput
     * definition. */
    printf("throughput_ops_per_sec: %.2f\n",
        naive_median_ms > 0.0 ? ((double)(n_dim * n_dim) / (naive_median_ms / 1000.0)) : 0.0);
    if (vendor_loaded) {
        printf("vendor_median_ms: %.2f\n", vendor_median_ms);
        printf("vendor_p95_ms: %.2f\n", vendor_p95_ms);
        printf("vendor_p99_ms: %.2f\n", vendor_p99_ms);
        if (vendor_median_ms > 0.0) {
            printf("speedup: %.2fx\n", naive_median_ms / vendor_median_ms);
        }
    } else {
        printf("vendor_median_ms: N/A - no vendor GEMM library found on this machine\n");
    }
    printf("correctness (naive): %s (max relative error: %f)\n",
        correctness_naive_pass ? "PASS" : "FAIL", correctness_naive_err);
    if (vendor_loaded) {
        printf("correctness (vendor): %s (max relative error: %f)\n",
            correctness_vendor_pass ? "PASS" : "FAIL", correctness_vendor_err);
    }

    gpuBridgeTensorFree(&a_dev);
    gpuBridgeTensorFree(&b_dev);
    gpuBridgeTensorFree(&c_dev);
    free(a);
    free(b);
    free(c);
    gpuBridgeShutdown();

    /* Step 9 (success criteria): 0 iff every correctness check that
     * actually ran passed. A missing vendor library is not a failure. */
    bool overall_pass = correctness_naive_pass && (!vendor_loaded || correctness_vendor_pass);
    return overall_pass ? 0 : 1;
}
