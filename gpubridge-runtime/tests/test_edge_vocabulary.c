/*
 * test_edge_vocabulary.c — Phase 4 (spec4.md section 12): pure-function unit
 * test for the edge-inference vocabulary that needs no backend at all.
 *
 * Three independent checks, none of which call gpuBridgeInit():
 *   1. gpuBridgeChooseSchedulingPolicy() — table-driven, same Case-array
 *      shape test_memory_strategy.c already uses.
 *   2. gpuBridgeClassifyHardwareTierFromDeviceName() — table-driven,
 *      including the unrecognized-string default-to-DISCRETE_GPU case.
 *   3. gpuBridgeIrDescribeQuantizedTensor() — pure data-shaping, asserts
 *      device_resident/device_ptr/quant fields land correctly.
 *
 * Exit code doubles as the pass/fail signal for `ctest`: 0 on PASS, 1 on FAIL.
 */
#include "gpubridge_runtime.h"
#include "gpubridge_scheduling.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    const char* name;
    double latency_budget_ms;
    int queue_depth;
    bool interactive;
    GpuBridgeSchedulingPolicy expected;
} SchedulingCase;

typedef struct {
    const char* name;
    const char* device_name;
    GpuBridgeHardwareTier expected;
} HardwareTierCase;

static bool test_scheduling_policy(void)
{
    SchedulingCase cases[] = {
        { "interactive_wins_over_everything", 500.0, 10, true, GPUBRIDGE_SCHEDULING_LATENCY_FIRST },
        { "deep_queue_prefers_throughput", 50.0, 4, false, GPUBRIDGE_SCHEDULING_THROUGHPUT_FIRST },
        { "tight_budget_prefers_latency", 80.0, 1, false, GPUBRIDGE_SCHEDULING_LATENCY_FIRST },
        { "loose_budget_falls_to_throughput", 500.0, 1, false, GPUBRIDGE_SCHEDULING_THROUGHPUT_FIRST },
        { "zero_budget_falls_to_throughput", 0.0, 0, false, GPUBRIDGE_SCHEDULING_THROUGHPUT_FIRST },
    };
    size_t num_cases = sizeof(cases) / sizeof(cases[0]);

    bool all_pass = true;
    printf("scheduling_policy:\n");
    for (size_t i = 0; i < num_cases; i++) {
        GpuBridgeSchedulingPolicy actual = gpuBridgeChooseSchedulingPolicy(
            cases[i].latency_budget_ms, cases[i].queue_depth, cases[i].interactive);
        bool pass = actual == cases[i].expected;
        all_pass = all_pass && pass;
        printf("  %-32s expected=%-16s actual=%-16s %s\n",
            cases[i].name,
            gpuBridgeSchedulingPolicyToString(cases[i].expected),
            gpuBridgeSchedulingPolicyToString(actual),
            pass ? "PASS" : "FAIL");
    }
    return all_pass;
}

static bool test_hardware_tier_classification(void)
{
    HardwareTierCase cases[] = {
        { "discrete_nvidia", "NVIDIA TITAN Xp", GPUBRIDGE_HARDWARE_TIER_DISCRETE_GPU },
        { "integrated_intel_uhd", "Intel(R) UHD Graphics 630", GPUBRIDGE_HARDWARE_TIER_INTEGRATED_GPU },
        { "integrated_amd_apu", "AMD Radeon(TM) Graphics", GPUBRIDGE_HARDWARE_TIER_INTEGRATED_GPU },
        { "unrecognized_defaults_discrete", "Some Unknown Vendor Accelerator X9",
            GPUBRIDGE_HARDWARE_TIER_DISCRETE_GPU },
    };
    size_t num_cases = sizeof(cases) / sizeof(cases[0]);

    bool all_pass = true;
    printf("hardware_tier_classification:\n");
    for (size_t i = 0; i < num_cases; i++) {
        GpuBridgeHardwareTier actual =
            gpuBridgeClassifyHardwareTierFromDeviceName(cases[i].device_name);
        bool pass = actual == cases[i].expected;
        all_pass = all_pass && pass;
        printf("  %-32s device=%-32s expected=%-16s actual=%-16s %s\n",
            cases[i].name, cases[i].device_name,
            gpuBridgeHardwareTierToString(cases[i].expected),
            gpuBridgeHardwareTierToString(actual),
            pass ? "PASS" : "FAIL");
    }
    return all_pass;
}

static bool test_describe_quantized_tensor(void)
{
    GpuBridgeTensorDesc desc;
    size_t shape[2] = { 4096, 4096 };
    GpuBridgeQuantDesc quant = { .scale = 0.01f, .zero_point = 0, .block_size = 32 };

    gpuBridgeIrDescribeQuantizedTensor(&desc, GPUBRIDGE_TYPE_INT8_QUANTIZED, 2, shape, quant);

    bool pass = desc.type == GPUBRIDGE_TYPE_INT8_QUANTIZED
        && desc.rank == 2
        && desc.shape[0] == shape[0] && desc.shape[1] == shape[1]
        && desc.device_resident == false
        && desc.device_ptr == NULL
        && desc.quant.scale == quant.scale
        && desc.quant.zero_point == quant.zero_point
        && desc.quant.block_size == quant.block_size;

    printf("describe_quantized_tensor: device_resident=%d device_ptr=%p scale=%.4f %s\n",
        desc.device_resident, desc.device_ptr, (double)desc.quant.scale,
        pass ? "PASS" : "FAIL");
    return pass;
}

int main(void)
{
    printf("GPUBridge edge_vocabulary test\n");

    bool all_pass = true;
    all_pass = test_scheduling_policy() && all_pass;
    all_pass = test_hardware_tier_classification() && all_pass;
    all_pass = test_describe_quantized_tensor() && all_pass;

    printf("result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
