/*
 * gpubridge_selector.c — turns a requested GpuBridgeBackendKind into an actual backend.
 *
 * This is the single place that implements spec1.md section 10's backend
 * selection policy. Every other file either asks this module for a
 * decision (gpubridge_runtime.c) or only reports the decision that was already
 * made (gpubridge_diagnostics.c) — none of them duplicate this policy.
 */
#include "gpubridge_internal.h"

#include <stdio.h>

bool gpuBridgeSelectBackend(GpuBridgeBackendKind requested, GpuBridgeSelection* out,
                      char* err_buf, size_t err_buf_size)
{
    /* gpuBridgeBackendCpuGet()/gpuBridgeBackendOpenclGet() return static, always-valid
     * pointers (see backends/cpu/gpubridge_backend_cpu.c and
     * backends/opencl/gpubridge_backend_opencl.c) — there is nothing to allocate
     * or free here. */
    const GpuBridgeBackend* cpu = gpuBridgeBackendCpuGet();
    const GpuBridgeBackend* opencl = gpuBridgeBackendOpenclGet();
    const GpuBridgeBackend* vulkan = gpuBridgeBackendVulkanGet();

    /* Always probe every backend's availability, regardless of what was
     * requested — diagnostics (GPUBRIDGE_VERBOSE=1) must report the availability
     * of *all* backends even when, say, GPUBRIDGE_BACKEND=cpu forced a choice
     * that ignores whether OpenCL happens to be present. */
    out->cpu_available = cpu->available();
    out->opencl_available = opencl->available();
    out->sycl_available = false; /* not implemented in this milestone */
    out->vulkan_available = vulkan->available(); /* new, Phase 4.5 */
    out->selected = NULL;
    out->selected_kind = requested;

    switch (requested) {
    case GPUBRIDGE_BACKEND_CPU:
        /* CPU is always available, so this can never fail. */
        out->selected = cpu;
        out->selected_kind = GPUBRIDGE_BACKEND_CPU;
        return true;

    case GPUBRIDGE_BACKEND_OPENCL:
        /* Explicit request for a backend that isn't there: fail loudly
         * with a specific, actionable message rather than silently
         * substituting a different backend (spec1.md success criterion #8). */
        if (!out->opencl_available) {
            snprintf(err_buf, err_buf_size,
                "OpenCL backend requested via GPUBRIDGE_BACKEND=opencl but no "
                "OpenCL platform/GPU device was found");
            return false;
        }
        out->selected = opencl;
        out->selected_kind = GPUBRIDGE_BACKEND_OPENCL;
        return true;

    case GPUBRIDGE_BACKEND_SYCL:
        /* SYCL is out of scope for this milestone (spec1.md allows either
         * OpenCL or SYCL as the first GPU backend; this build implements
         * OpenCL only). Always fails, with a message that says so rather
         * than a generic "unavailable". */
        snprintf(err_buf, err_buf_size,
            "SYCL backend requested via GPUBRIDGE_BACKEND=sycl but SYCL support "
            "is not implemented in this build");
        return false;

    case GPUBRIDGE_BACKEND_VULKAN:
        /* Same "fail loudly, never silently substitute" shape as the
         * existing GPUBRIDGE_BACKEND_OPENCL case above (spec4.5.md section 5). */
        if (!out->vulkan_available) {
            snprintf(err_buf, err_buf_size,
                "Vulkan backend requested via GPUBRIDGE_BACKEND=vulkan but no "
                "Vulkan-capable device with a compute queue family was found "
                "(or libvulkan.so.1 is not installed)");
            return false;
        }
        out->selected = vulkan;
        out->selected_kind = GPUBRIDGE_BACKEND_VULKAN;
        return true;

    case GPUBRIDGE_BACKEND_AUTO:
    default:
        /* spec1.md section 10: auto prefers a GPU backend if one is
         * available, otherwise falls back to CPU. CPU is always
         * available, so this branch can never fail. Deliberately does NOT
         * consider Vulkan a candidate (spec4.5.md section 5's fourth
         * principle) — this prototype backend only covers one of five ops,
         * so a caller must request it explicitly via GPUBRIDGE_BACKEND=vulkan. */
        if (out->opencl_available) {
            out->selected = opencl;
            out->selected_kind = GPUBRIDGE_BACKEND_OPENCL;
        } else {
            out->selected = cpu;
            out->selected_kind = GPUBRIDGE_BACKEND_CPU;
        }
        return true;
    }
}
