/*
 * gpubridge_scheduling.c — implements gpuBridgeChooseSchedulingPolicy/
 * gpuBridgeSchedulingPolicyToString (spec4.md section 8).
 *
 * Pure function: no runtime state, no gpuBridgeInit() dependency, same
 * footprint as gpubridge_memory.c's gpuBridgeChooseMemoryStrategy.
 */
#include "gpubridge_scheduling.h"

/* Thresholds below (queue_depth >= 4, latency_budget_ms <= 100.0) are
 * provisional defaults, not values derived from any measurement — no
 * batching scheduler or request queue exists yet to generate the
 * latency/throughput data that would justify a specific number (spec4.md
 * section 4 non-goal 4). Phase 5's benchmark suite is the first place real
 * workload data could revise them; until then, treat them as
 * reasonable-guess placeholders that the table test (tests/test_edge_vocabulary.c)
 * locks in as *current* behavior, not as validated policy. */
GpuBridgeSchedulingPolicy gpuBridgeChooseSchedulingPolicy(
    double latency_budget_ms, int queue_depth, bool interactive)
{
    if (interactive) {
        return GPUBRIDGE_SCHEDULING_LATENCY_FIRST;
    }
    if (queue_depth >= 4) {
        return GPUBRIDGE_SCHEDULING_THROUGHPUT_FIRST;
    }
    if (latency_budget_ms > 0.0 && latency_budget_ms <= 100.0) {
        return GPUBRIDGE_SCHEDULING_LATENCY_FIRST;
    }
    return GPUBRIDGE_SCHEDULING_THROUGHPUT_FIRST;
}

const char* gpuBridgeSchedulingPolicyToString(GpuBridgeSchedulingPolicy policy)
{
    switch (policy) {
    case GPUBRIDGE_SCHEDULING_LATENCY_FIRST:    return "latency_first";
    case GPUBRIDGE_SCHEDULING_THROUGHPUT_FIRST: return "throughput_first";
    default:                                     return "unknown";
    }
}
