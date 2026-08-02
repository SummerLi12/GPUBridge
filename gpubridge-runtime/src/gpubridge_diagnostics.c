/*
 * gpubridge_diagnostics.c — prints the GPUBRIDGE_VERBOSE=1 diagnostic block.
 *
 * Implements the exact output shape from spec1.md section 11. This is
 * called from gpubridge_runtime.c's gpuBridgeLaunchKernel(), which is the only point in
 * the runtime where both the resolved backend selection (known since
 * gpuBridgeInit()) and the operation/element-count being launched (only known at
 * launch time) are available simultaneously.
 */
#include "gpubridge_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

/* Shared by gpuBridgePrintDiagnostics/gpuBridgePrintTensorDiagnostics: the "GPUBridge runtime:"
 * header through "selected backend:" is identical for every op kind (spec1
 * section 11 / spec2 section 12 use the exact same preamble); only the last
 * one or two lines ("elements:" vs. "operation:"+"shapes:") differ. Callers
 * must have already checked gpuBridgeEnvVerbose(). */
static void print_diagnostics_preamble(GpuBridgeBackendKind requested,
                                        const GpuBridgeSelection* selection)
{
    printf("GPUBridge runtime:\n");
    /* What the user actually asked for via GPUBRIDGE_BACKEND (e.g. "auto"), not
     * what it resolved to — see the "selected backend" line below for that. */
    printf("  requested backend: %s\n", gpuBridgeBackendKindToString(requested));
    /* CPU is always available in this milestone; still printed explicitly
     * to match spec1.md's exact diagnostic block shape. */
    printf("  CPU backend: %s\n",
        selection->cpu_available ? "available" : "unavailable");
    printf("  OpenCL backend: %s\n",
        selection->opencl_available ? "available" : "unavailable");
    /* SYCL is hard-coded unavailable: it is not implemented in this
     * milestone, so there is nothing to probe. */
    printf("  SYCL backend: %s\n",
        selection->sycl_available ? "available" : "unavailable");
    /* New, Phase 4.5 (spec4.5.md section 14) — appended after SYCL, matching
     * the enum's declaration order in gpubridge_runtime.h. */
    printf("  Vulkan backend: %s\n",
        selection->vulkan_available ? "available" : "unavailable");
    /* The backend that was actually chosen by gpubridge_selector.c — this is what
     * differs from "requested backend" when GPUBRIDGE_BACKEND=auto resolves to a
     * concrete backend. */
    printf("  selected backend: %s\n",
        gpuBridgeBackendKindToString(selection->selected_kind));
}

/*
 * --- Milestone 1.5 addition (spec1.5.md section 11.1) ---
 *
 * Shared by gpuBridgePrintDiagnostics/gpuBridgePrintTensorDiagnostics, called
 * after their operation-specific line(s). Queries the selected backend's
 * get_caps() and prints 5 lines: the memory strategy gpuBridgeChooseMemoryStrategy
 * would pick for an illustrative single-use allocation (reuse_count=1,
 * latency_sensitive=false — no real per-allocation context exists at this
 * generic diagnostics call site), plus 4 of the 7 capability flags, matching
 * optmization.md section 6's exact worked-example line set. No-op if
 * get_caps() reports failure (never happens for the CPU/OpenCL backends
 * implemented so far, but the vtable contract allows it).
 */
static void print_memory_and_caps(const GpuBridgeSelection* selection)
{
    GpuBridgeBackendCaps caps;
    if (!selection->selected->get_caps(&caps)) {
        return;
    }

    GpuBridgeMemoryStrategy strategy = gpuBridgeChooseMemoryStrategy(0, 1, false, &caps);
    printf("  memory strategy: %s\n", gpuBridgeMemoryStrategyToString(strategy));
    printf("  shared memory supported: %s\n", caps.supports_shared_memory ? "yes" : "no");
    printf("  host pinned memory supported: %s\n", caps.supports_host_pinned_memory ? "yes" : "no");
    printf("  async copy supported: %s\n", caps.supports_async_copy ? "yes" : "no");
    printf("  profiling timestamps supported: %s\n", caps.supports_profiling_timestamps ? "yes" : "no");
    /* Milestone 3 (spec3.md section 8.3) */
    printf("  vendor GEMM library loaded: %s\n", caps.vendor_gemm_loaded ? "yes" : "no");
}

void gpuBridgePrintDiagnostics(GpuBridgeBackendKind requested,
                         const GpuBridgeSelection* selection,
                         const char* operation, size_t elements)
{
    /* Silently do nothing unless the user opted in with GPUBRIDGE_VERBOSE=1 — this
     * function must be safe to call unconditionally on every kernel launch. */
    if (!gpuBridgeEnvVerbose()) {
        return;
    }

    print_diagnostics_preamble(requested, selection);
    printf("  operation: %s\n", operation);
    printf("  elements: %zu\n", elements);
    print_memory_and_caps(selection);
}

/* Renders one GpuBridgeTensorArg as "Name[dim0,dim1]" (or "Name[dim0]" for rank 1)
 * into buf, capitalizing the arg's first letter to match spec2.md section
 * 12's worked example ("A[64,128]", not "a[64,128]"). */
static void format_tensor_arg(const GpuBridgeTensorArg* arg, char* buf,
                               size_t buf_size)
{
    char name[32];
    snprintf(name, sizeof(name), "%s", arg->name);
    name[0] = (char)toupper((unsigned char)name[0]);

    if (arg->rank == 1) {
        snprintf(buf, buf_size, "%s[%zu]", name, arg->shape[0]);
    } else {
        snprintf(buf, buf_size, "%s[%zu,%zu]", name, arg->shape[0],
            arg->shape[1]);
    }
}

void gpuBridgePrintTensorDiagnostics(GpuBridgeBackendKind requested,
                               const GpuBridgeSelection* selection,
                               const char* operation,
                               const GpuBridgeTensorArg* args, size_t arg_count)
{
    if (!gpuBridgeEnvVerbose()) {
        return;
    }

    print_diagnostics_preamble(requested, selection);
    printf("  operation: %s\n", operation);

    /* Build "A[..] B[..] -> C[..]": every is_input arg (in argument order)
     * before the arrow, every is_output arg after it. An arg that is both
     * (broadcast_add_f32's in-place `y`) appears on both sides, which is the
     * accurate description of what actually happened to it. */
    char shapes[256];
    size_t pos = 0;
    shapes[0] = '\0';

    bool first = true;
    for (size_t i = 0; i < arg_count; i++) {
        if (!args[i].is_input) {
            continue;
        }
        char part[64];
        format_tensor_arg(&args[i], part, sizeof(part));
        pos += (size_t)snprintf(shapes + pos, sizeof(shapes) - pos, "%s%s",
            first ? "" : " ", part);
        first = false;
    }
    pos += (size_t)snprintf(shapes + pos, sizeof(shapes) - pos, " ->");
    for (size_t i = 0; i < arg_count; i++) {
        if (!args[i].is_output) {
            continue;
        }
        char part[64];
        format_tensor_arg(&args[i], part, sizeof(part));
        pos += (size_t)snprintf(shapes + pos, sizeof(shapes) - pos, " %s",
            part);
    }

    printf("  shapes: %s\n", shapes);

    /* --- Milestone 3 addition (spec3.md section 8.3) ---
     * Printed only for matmul_f32, before the launch actually runs — worded
     * as *selected*, never as an implicit success claim (spec3.md section
     * 8.1's "selected" vs. "used" distinction). If the selected vendor path
     * then fails at runtime, the dispatch code in
     * gpubridge_backend_opencl.c prints a separate, post-launch
     * "gemm result: failed (...)" line — "selected" and "result" are never
     * conflated into one claim. On success, no additional line is printed
     * (silence == success, this project's existing convention). */
    if (strcmp(operation, "matmul_f32") == 0) {
        GpuBridgeBackendCaps caps;
        if (selection->selected->get_caps(&caps)) {
            const char* vendor_name =
                (selection->selected_kind == GPUBRIDGE_BACKEND_OPENCL) ? "clblast" :
                (selection->selected_kind == GPUBRIDGE_BACKEND_CPU)    ? "openblas" : "vendor";

            if (gpuBridgeEnvVendorGemmDisable()) {
                printf("  gemm path selected: naive (GPUBRIDGE_VENDOR_GEMM_DISABLE=1)\n");
            } else if (caps.vendor_gemm_loaded) {
                printf("  gemm path selected: %s\n", vendor_name);
            } else {
                printf("  gemm path selected: naive (no vendor library loaded)\n");
            }
        }
    }

    print_memory_and_caps(selection);
}

static const char* scalar_type_to_enum_name(GpuBridgeScalarType type)
{
    switch (type) {
    case GPUBRIDGE_TYPE_F32:            return "GPUBRIDGE_TYPE_F32";
    case GPUBRIDGE_TYPE_INT8_QUANTIZED:  return "GPUBRIDGE_TYPE_INT8_QUANTIZED";
    case GPUBRIDGE_TYPE_INT4_QUANTIZED:  return "GPUBRIDGE_TYPE_INT4_QUANTIZED";
    default:                             return "GPUBRIDGE_TYPE_UNKNOWN";
    }
}

static const char* op_kind_to_enum_name(GpuBridgeOpKind op_kind)
{
    switch (op_kind) {
    case GPUBRIDGE_OP_VECTOR_ADD:    return "GPUBRIDGE_OP_VECTOR_ADD";
    case GPUBRIDGE_OP_MATMUL:        return "GPUBRIDGE_OP_MATMUL";
    case GPUBRIDGE_OP_REDUCE_SUM:    return "GPUBRIDGE_OP_REDUCE_SUM";
    case GPUBRIDGE_OP_BROADCAST_ADD: return "GPUBRIDGE_OP_BROADCAST_ADD";
    case GPUBRIDGE_OP_RELU:          return "GPUBRIDGE_OP_RELU";
    default:                         return "GPUBRIDGE_OP_UNKNOWN";
    }
}

/* matmul_f32's args are always (a: in [m,k], b: in [k,n], c: out [m,n]) —
 * gpuBridgeIrInitMatmulF32()'s fixed argument order (gpubridge_ir.c). No
 * other op has a named m/k/n-style dimension convention worth annotating,
 * so this is deliberately matmul-only; every other op's shape line below
 * prints with no inline comment. */
static const char* matmul_arg_dim_note(size_t index)
{
    switch (index) {
    case 0:  return "[m, k]";
    case 1:  return "[k, n]";
    case 2:  return "[m, n]";
    default: return NULL;
    }
}

void gpuBridgePrintTensorIR(const GpuBridgeTensorKernelIR* kernel)
{
    if (!gpuBridgeEnvVerbose()) {
        return;
    }

    printf("Low level tensor IR in runtime\n");
    for (size_t i = 0; i < kernel->arg_count; i++) {
        const GpuBridgeTensorArg* arg = &kernel->args[i];
        const char* dim_note = (kernel->op_kind == GPUBRIDGE_OP_MATMUL)
            ? matmul_arg_dim_note(i) : NULL;

        printf("kernel_args[%zu] (\"%s\"):\n", i, arg->name);
        printf("name         = \"%s\"\n", arg->name);
        printf("type         = %s\n", scalar_type_to_enum_name(arg->type));
        printf("rank         = %d\n", arg->rank);
        if (arg->rank == 1) {
            printf("shape        = {%zu}\n", arg->shape[0]);
        } else if (dim_note != NULL) {
            printf("shape        = {%zu, %zu}      // %s\n",
                arg->shape[0], arg->shape[1], dim_note);
        } else {
            printf("shape        = {%zu, %zu}\n", arg->shape[0], arg->shape[1]);
        }
        printf("is_input     = %s\n", arg->is_input ? "true" : "false");
        printf("is_output    = %s\n", arg->is_output ? "true" : "false");
        printf("\n");
    }

    printf("kernel (the GpuBridgeTensorKernelIR itself):\n");
    printf("kernel_name  = \"%s\"\n", kernel->kernel_name);
    printf("op_kind      = %s\n", op_kind_to_enum_name(kernel->op_kind));
    printf("args         = kernel_args      // pointer to the %zu-element array above\n",
        kernel->arg_count);
    printf("arg_count    = %zu\n", kernel->arg_count);
    if (kernel->op_kind == GPUBRIDGE_OP_MATMUL) {
        printf("global_size  = {%zu, %zu}            // {m, n} — one GPU thread per output cell\n",
            kernel->global_size[0], kernel->global_size[1]);
    } else {
        printf("global_size  = {%zu, %zu}\n", kernel->global_size[0], kernel->global_size[1]);
    }
}

/*
 * --- Milestone 1.5 addition (spec1.5.md section 11.2) ---
 *
 * Prints the GPUBRIDGE_PROFILE=1 "GPUBridge profile:" block. Called once by
 * gpuBridgeShutdown() (not per-launch), so `stats` is a snapshot of the whole
 * session's cumulative counters (spec1.5.md section 9.3). total_gpu_path_time_ms
 * is computed here, not stored in GpuBridgeProfileStats.
 */
void gpuBridgePrintProfile(const GpuBridgeProfileStats* stats, double total_runtime_ms,
    GpuBridgeBackendKind selected_backend, const char* device_name,
    size_t device_memory_limit_bytes, GpuBridgeHardwareTier hardware_tier)
{
    if (!gpuBridgeEnvProfile()) {
        return;
    }

    double total_gpu_path_time_ms = stats->host_to_device_time_ms
        + stats->kernel_time_ms
        + stats->device_to_host_time_ms;

    /* --- gpubridge_performance_spec.md section 5 gap-fix (2026-07-14) ---
     * cpu_orchestration_time_ms is total_runtime_ms minus every bucket of
     * time already known to be GPU-path or sync-wait time — i.e. time spent
     * somewhere other than waiting on the backend (validation, dispatch,
     * bookkeeping, the test program's own non-runtime work). Floored at 0:
     * CLOCK_MONOTONIC jitter across the many small timed sections above
     * could otherwise make this go very slightly negative. */
    double cpu_orchestration_time_ms = total_runtime_ms - total_gpu_path_time_ms
        - stats->synchronization_time_ms;
    if (cpu_orchestration_time_ms < 0.0) {
        cpu_orchestration_time_ms = 0.0;
    }

    /* --- Phase 4 additions (spec4.md section 9.2, section 4's own example
     * profile format) --- closes a real, pre-existing gap: the performance
     * spec's own example profile already calls for "selected backend:"/
     * "device:" as the first two lines, which this function never printed
     * before this phase. */
    printf("GPUBridge profile:\n");
    printf("  selected_backend: %s\n", gpuBridgeBackendKindToString(selected_backend));
    printf("  device: %s\n", device_name);
    printf("  device_memory_limit_bytes: %zu\n", device_memory_limit_bytes);
    /* Labeling requirement (spec4.md section 9), not just an internal
     * comment: hardware_tier is a device-name-substring heuristic with a
     * real false-negative risk (gpuBridgeClassifyHardwareTierFromDeviceName's
     * own comment), never a measured hardware fact — this line makes that
     * caveat visible to anyone reading raw profile output, not just source.
     * Starts with '#', so bench_report.py's "key: value" parser silently
     * skips it (no accidental new profile field). */
    printf("  # heuristic, device-name-based — see gpuBridgeClassifyHardwareTierFromDeviceName()\n");
    printf("  hardware_tier: %s\n", gpuBridgeHardwareTierToString(hardware_tier));
    printf("  total_runtime_ms: %.2f\n", total_runtime_ms);
    printf("  runtime_init_time_ms: %.2f\n", stats->runtime_init_time_ms);
    printf("  model_load_time_ms: %.2f\n", stats->model_load_time_ms);
    printf("  cpu_orchestration_time_ms: %.2f\n", cpu_orchestration_time_ms);
    printf("  host_to_device_count: %zu\n", stats->host_to_device_count);
    printf("  host_to_device_bytes: %zu\n", stats->host_to_device_bytes);
    printf("  host_to_device_time_ms: %.2f\n", stats->host_to_device_time_ms);
    printf("  kernel_launch_count: %zu\n", stats->kernel_launch_count);
    printf("  kernel_time_ms: %.2f\n", stats->kernel_time_ms);
    /* --- Phase 4 additions (spec4.md section 8) ---
     * Both derived at print time from data already collected — no new
     * GpuBridgeProfileStats field needed (spec4.md section 8's own reasoning:
     * inventing a counter with nothing to feed it would be hollow vocabulary
     * since no real batching exists yet). Guarded against divide-by-zero. */
    double avg_kernel_latency_ms = stats->kernel_launch_count > 0
        ? stats->kernel_time_ms / (double)stats->kernel_launch_count
        : 0.0;
    double throughput_ops_per_sec = total_runtime_ms > 0.0
        ? (double)stats->kernel_launch_count / (total_runtime_ms / 1000.0)
        : 0.0;
    printf("  avg_kernel_latency_ms: %.2f\n", avg_kernel_latency_ms);
    printf("  throughput_ops_per_sec: %.2f\n", throughput_ops_per_sec);
    printf("  device_to_host_count: %zu\n", stats->device_to_host_count);
    printf("  device_to_host_bytes: %zu\n", stats->device_to_host_bytes);
    printf("  device_to_host_time_ms: %.2f\n", stats->device_to_host_time_ms);
    printf("  total_gpu_path_time_ms: %.2f\n", total_gpu_path_time_ms);
    printf("  synchronization_count: %d\n", stats->synchronization_count);
    printf("  synchronization_time_ms: %.2f\n", stats->synchronization_time_ms);

    /* --- Milestone 2.5 additions (spec2.5.md section 9.1) ---
     * pool_hit_rate is computed here at print time, not stored in
     * GpuBridgeProfileStats, same treatment as total_gpu_path_time_ms above.
     * Guarded against tensor_allocation_count == 0 (no tensor work this
     * session) to avoid a divide-by-zero. */
    double pool_hit_rate = stats->tensor_allocation_count > 0
        ? (double)stats->tensor_pool_hit_count / (double)stats->tensor_allocation_count
        : 0.0;
    printf("  tensor_allocations: %zu\n", stats->tensor_allocation_count);
    printf("  tensor_alloc_time_ms: %.2f\n", stats->tensor_alloc_time_ms);
    printf("  tensor_frees: %zu\n", stats->tensor_free_count);
    printf("  tensor_free_time_ms: %.2f\n", stats->tensor_free_time_ms);
    printf("  pool_hit_rate: %.2f\n", pool_hit_rate);
    printf("  peak_device_bytes: %zu\n", stats->peak_device_bytes);
    /* --- Phase 4 addition (spec4.md section 6.1) --- */
    printf("  kv_cache_allocations: %zu\n", stats->kv_cache_allocation_count);
    printf("  kv_cache_bytes: %zu\n", stats->kv_cache_bytes);
}
