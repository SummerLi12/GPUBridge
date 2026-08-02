/*
 * gpubridge_memory.c — implements the pure functions declared in
 * gpubridge_memory.h (spec1.5.md sections 6 and 8; Phase 4 section 9 below).
 *
 * Every function here is pure: no runtime state, no gpuBridgeInit()
 * dependency — mirrors gpubridge_ir.c's footprint.
 */
#include "gpubridge_memory.h"

#include <ctype.h>
#include <string.h>

GpuBridgeMemoryStrategy gpuBridgeChooseMemoryStrategy(
    size_t bytes,
    int reuse_count,
    bool latency_sensitive,
    const GpuBridgeBackendCaps* caps
)
{
    (void)bytes; /* not referenced by the current decision rule; see spec1.5.md section 8 */

    if (latency_sensitive && caps->supports_shared_memory) {
        return GPUBRIDGE_MEMORY_SHARED;
    }
    if (reuse_count <= 1 && caps->supports_shared_memory) {
        return GPUBRIDGE_MEMORY_SHARED;
    }
    if (caps->supports_device_local_memory) {
        return GPUBRIDGE_MEMORY_DEVICE_LOCAL;
    }
    if (caps->supports_host_pinned_memory) {
        return GPUBRIDGE_MEMORY_HOST_PINNED;
    }
    return GPUBRIDGE_MEMORY_HOST_ONLY;
}

const char* gpuBridgeMemoryStrategyToString(GpuBridgeMemoryStrategy strategy)
{
    switch (strategy) {
    case GPUBRIDGE_MEMORY_AUTO:         return "auto";
    case GPUBRIDGE_MEMORY_DEVICE_LOCAL: return "device_local";
    case GPUBRIDGE_MEMORY_SHARED:       return "shared";
    case GPUBRIDGE_MEMORY_HOST_PINNED:  return "host_pinned";
    case GPUBRIDGE_MEMORY_HOST_ONLY:    return "host_only";
    default:                            return "unknown";
    }
}

/* --- Phase 4 addition (spec4.md section 9) --- */

/* Case-insensitive substring search — a small local helper rather than GNU
 * strcasestr(), to avoid a new feature-test-macro dependency (spec4.md
 * section 9's exact rationale). */
static bool contains_ci(const char* haystack, const char* needle)
{
    size_t hlen = strlen(haystack);
    size_t nlen = strlen(needle);
    if (nlen == 0 || nlen > hlen) {
        return false;
    }
    for (size_t i = 0; i + nlen <= hlen; i++) {
        size_t j = 0;
        for (; j < nlen; j++) {
            if (tolower((unsigned char)haystack[i + j]) != tolower((unsigned char)needle[j])) {
                break;
            }
        }
        if (j == nlen) {
            return true;
        }
    }
    return false;
}

GpuBridgeHardwareTier gpuBridgeClassifyHardwareTierFromDeviceName(const char* device_name)
{
    if (device_name == NULL) {
        return GPUBRIDGE_HARDWARE_TIER_DISCRETE_GPU;
    }
    /* Common Intel/AMD integrated-GPU OpenCL device-name substrings
     * (spec4.md section 9's exact table). Any unmatched name — including a
     * real integrated GPU with an unusual string — falls to DISCRETE_GPU. */
    if (contains_ci(device_name, "UHD Graphics") ||
        contains_ci(device_name, "Iris") ||
        contains_ci(device_name, "HD Graphics") ||
        contains_ci(device_name, "Radeon(TM) Graphics")) {
        return GPUBRIDGE_HARDWARE_TIER_INTEGRATED_GPU;
    }
    return GPUBRIDGE_HARDWARE_TIER_DISCRETE_GPU;
}

const char* gpuBridgeHardwareTierToString(GpuBridgeHardwareTier tier)
{
    switch (tier) {
    case GPUBRIDGE_HARDWARE_TIER_UNKNOWN:        return "unknown";
    case GPUBRIDGE_HARDWARE_TIER_CPU_ONLY:       return "cpu_only";
    case GPUBRIDGE_HARDWARE_TIER_INTEGRATED_GPU: return "integrated_gpu";
    case GPUBRIDGE_HARDWARE_TIER_DISCRETE_GPU:   return "discrete_gpu";
    default:                                     return "unknown";
    }
}
