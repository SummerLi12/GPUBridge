/*
 * test_memory_strategy.c — Milestone 1.5 unit test for
 * gpuBridgeChooseMemoryStrategy (spec1.5.md section 8).
 *
 * Pure function test: no gpuBridgeInit() / backend involved at all, since
 * gpuBridgeChooseMemoryStrategy takes a caller-constructed GpuBridgeBackendCaps and
 * returns a value with no side effects. Each case below exercises one
 * branch of the decision table in order.
 *
 * Exit code doubles as the pass/fail signal for `ctest`, same convention as
 * every other test in this project: 0 on PASS (all cases), 1 on FAIL.
 */
#include "gpubridge_runtime.h"

#include <stdio.h>

typedef struct {
    const char* name;
    size_t bytes;
    int reuse_count;
    bool latency_sensitive;
    GpuBridgeBackendCaps caps;
    GpuBridgeMemoryStrategy expected;
} Case;

int main(void)
{
    Case cases[] = {
        /* latency_sensitive + shared available -> SHARED, first branch wins
         * even though device_local is also available. */
        {
            .name = "latency_sensitive_prefers_shared",
            .bytes = 4096, .reuse_count = 5, .latency_sensitive = true,
            .caps = { .supports_shared_memory = true, .supports_device_local_memory = true },
            .expected = GPUBRIDGE_MEMORY_SHARED,
        },
        /* Not latency-sensitive, but reuse_count <= 1 + shared available ->
         * SHARED (second branch). */
        {
            .name = "single_use_prefers_shared",
            .bytes = 4096, .reuse_count = 1, .latency_sensitive = false,
            .caps = { .supports_shared_memory = true },
            .expected = GPUBRIDGE_MEMORY_SHARED,
        },
        /* Reused, not latency-sensitive, no shared memory, but device-local
         * available -> DEVICE_LOCAL (third branch). This is the OpenCL
         * backend's actual caps shape (spec1.5.md section 12). */
        {
            .name = "reused_falls_to_device_local",
            .bytes = 4096, .reuse_count = 5, .latency_sensitive = false,
            .caps = { .supports_device_local_memory = true, .supports_async_kernel_launch = true },
            .expected = GPUBRIDGE_MEMORY_DEVICE_LOCAL,
        },
        /* Nothing but host-pinned available -> HOST_PINNED (fourth branch). */
        {
            .name = "falls_to_host_pinned",
            .bytes = 4096, .reuse_count = 5, .latency_sensitive = false,
            .caps = { .supports_host_pinned_memory = true },
            .expected = GPUBRIDGE_MEMORY_HOST_PINNED,
        },
        /* All-false caps -> HOST_ONLY (final fallback). This is the CPU
         * backend's actual caps shape (spec1.5.md section 12). */
        {
            .name = "all_false_falls_to_host_only",
            .bytes = 4096, .reuse_count = 5, .latency_sensitive = false,
            .caps = { 0 },
            .expected = GPUBRIDGE_MEMORY_HOST_ONLY,
        },
    };
    size_t num_cases = sizeof(cases) / sizeof(cases[0]);

    bool all_pass = true;
    printf("GPUBridge memory_strategy test\n");
    for (size_t i = 0; i < num_cases; i++) {
        GpuBridgeMemoryStrategy actual = gpuBridgeChooseMemoryStrategy(
            cases[i].bytes, cases[i].reuse_count, cases[i].latency_sensitive,
            &cases[i].caps);
        bool pass = actual == cases[i].expected;
        all_pass = all_pass && pass;
        printf("  %-32s expected=%-12s actual=%-12s %s\n",
            cases[i].name,
            gpuBridgeMemoryStrategyToString(cases[i].expected),
            gpuBridgeMemoryStrategyToString(actual),
            pass ? "PASS" : "FAIL");
    }

    printf("result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
