/*
 * gpubridge_scheduling.h — Phase 4 (spec4.md section 8): a batching/scheduling
 * policy skeleton.
 *
 * A new leaf header, deliberately not appended to gpubridge_memory.h — that
 * header's own comment scopes it to "memory-strategy vocabulary, backend
 * capability reporting, and profiling stats." Scheduling is a distinct
 * concern, and this project already gives each concern its own leaf header
 * (gpubridge_ir.h, gpubridge_backend.h, gpubridge_memory.h) — this header
 * matches that granularity rather than bloating an existing one.
 */
#ifndef GPUBRIDGE_SCHEDULING_H
#define GPUBRIDGE_SCHEDULING_H

#include <stdbool.h>

typedef enum {
    GPUBRIDGE_SCHEDULING_LATENCY_FIRST,
    GPUBRIDGE_SCHEDULING_THROUGHPUT_FIRST
} GpuBridgeSchedulingPolicy;

/* Pure decision function, mirroring gpuBridgeChooseMemoryStrategy's exact
 * shape: no side effects, no backend interaction. Not wired into any real
 * launch path this phase (spec4.md section 4 non-goal 7) — no
 * batching/request queue exists yet to consult it. See gpubridge_scheduling.c
 * for why its thresholds are provisional, not measurement-derived. */
GpuBridgeSchedulingPolicy gpuBridgeChooseSchedulingPolicy(
    double latency_budget_ms, int queue_depth, bool interactive);

const char* gpuBridgeSchedulingPolicyToString(GpuBridgeSchedulingPolicy policy);

#endif /* GPUBRIDGE_SCHEDULING_H */
