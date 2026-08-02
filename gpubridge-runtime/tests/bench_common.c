/*
 * bench_common.c — see bench_common.h. Logic moved verbatim from
 * test_matmul_benchmark.c's original static helpers (spec3.md section 9),
 * plus one new function (gpuBridgeBenchPercentile) for spec5.md section 7's
 * tail-latency reporting.
 */
#include "bench_common.h"

#include <math.h>
#include <stdlib.h>

int gpuBridgeBenchReadIters(void)
{
    const char* val = getenv("GPUBRIDGE_BENCH_ITERS");
    if (val == NULL) {
        return GPUBRIDGE_BENCH_DEFAULT_ITERS;
    }
    int parsed = atoi(val);
    return parsed > 0 ? parsed : GPUBRIDGE_BENCH_DEFAULT_ITERS;
}

static int compare_double(const void* a, const void* b)
{
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) { return -1; }
    if (da > db) { return 1; }
    return 0;
}

double gpuBridgeBenchMedian(double* samples, int count)
{
    qsort(samples, (size_t)count, sizeof(double), compare_double);
    if (count % 2 == 0) {
        return (samples[count / 2 - 1] + samples[count / 2]) / 2.0;
    }
    return samples[count / 2];
}

double gpuBridgeBenchPercentile(const double* sorted_samples, int count, double p)
{
    if (count <= 0) {
        return 0.0;
    }
    int rank = (int)ceil(p * (double)count) - 1;
    if (rank < 0) { rank = 0; }
    if (rank >= count) { rank = count - 1; }
    return sorted_samples[rank];
}

double gpuBridgeBenchGpuPathMs(const GpuBridgeProfileStats* s)
{
    return s->kernel_time_ms + s->synchronization_time_ms;
}
