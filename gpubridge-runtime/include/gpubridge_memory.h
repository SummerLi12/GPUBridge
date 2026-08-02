/*
 * gpubridge_memory.h — Milestone 1.5 (spec1.5.md): memory-strategy vocabulary,
 * backend capability reporting, and profiling stats.
 *
 * This header is deliberately a leaf: it depends on nothing but
 * <stdbool.h>/<stddef.h>, and is included independently by both
 * gpubridge_backend.h (which needs GpuBridgeBackendCaps for the get_caps
 * vtable slot) and gpubridge_runtime.h (which needs all three types for its
 * public accessors). Nothing here requires gpuBridgeInit() to have been
 * called; gpuBridgeChooseMemoryStrategy/gpuBridgeMemoryStrategyToString are
 * pure functions of their arguments.
 */
#ifndef GPUBRIDGE_MEMORY_H
#define GPUBRIDGE_MEMORY_H

#include <stdbool.h>
#include <stddef.h>

/* Memory placement intent, not a guarantee — see gpuBridgeChooseMemoryStrategy
 * (spec1.5.md section 8) for how a strategy is picked from a GpuBridgeBackendCaps. */
typedef enum {
    GPUBRIDGE_MEMORY_AUTO,
    GPUBRIDGE_MEMORY_DEVICE_LOCAL,
    GPUBRIDGE_MEMORY_SHARED,
    GPUBRIDGE_MEMORY_HOST_PINNED,
    GPUBRIDGE_MEMORY_HOST_ONLY
} GpuBridgeMemoryStrategy;

/* What a backend's current implementation actually supports (spec1.5.md
 * section 7) — honest reporting of present behavior, not hardware capability
 * probing. See spec1.5.md section 12 for the exact value each backend
 * reports and why. */
typedef struct {
    bool supports_device_local_memory;
    bool supports_shared_memory;
    bool supports_host_pinned_memory;
    bool supports_async_copy;
    bool supports_async_kernel_launch;
    bool supports_events;
    bool supports_profiling_timestamps;
    /* --- Milestone 3 addition (spec3.md section 8.2) ---
     * Whether this backend's optional vendor GEMM library (CLBlast for
     * OpenCL, OpenBLAS for CPU) was successfully dlopen'd/dlsym'd —
     * "loaded" only (spec3.md section 8.1's terminology): says nothing
     * about GPUBRIDGE_VENDOR_GEMM_DISABLE ("enabled") or whether a specific
     * launch actually succeeded ("used"). */
    bool vendor_gemm_loaded;
} GpuBridgeBackendCaps;

/* Cumulative profiling counters for the current runtime session (spec1.5.md
 * section 9) — reset on gpuBridgeInit()/gpuBridgeSetBackend(), printed and
 * cleared by gpuBridgeShutdown(). total_gpu_path_time_ms is not stored here;
 * it is computed at print time as the sum of the three *_time_ms fields. */
typedef struct {
    size_t host_to_device_bytes;
    size_t device_to_host_bytes;
    double host_to_device_time_ms;
    double kernel_time_ms;
    double device_to_host_time_ms;
    int    synchronization_count;
    /* --- Milestone 2.5 additions (spec2.5.md section 4 item 4) ---
     * tensor_allocation_count/tensor_free_count count every
     * gpuBridgeTensorAlloc/gpuBridgeTensorFree call that actually succeeded,
     * regardless of whether the pool or GPUBRIDGE_POOL_DISABLE=1 handled it.
     * tensor_pool_hit_count counts only the allocations the pool actually
     * satisfied from a previously-freed block. pool_hit_rate
     * (tensor_pool_hit_count / tensor_allocation_count) is not stored here;
     * it is computed at print time, same as total_gpu_path_time_ms above.
     * peak_device_bytes is a session-lifetime high-water mark (bytes ever
     * obtained from the backend's malloc_device across the session, which
     * never decreases mid-session since the pool never truly frees a block
     * until a drain) — not a live "currently allocated" figure; see
     * spec2.5.md section 5 non-goal #10. */
    size_t tensor_allocation_count;
    size_t tensor_free_count;
    size_t tensor_pool_hit_count;
    size_t peak_device_bytes;
    /* --- gpubridge_performance_spec.md section 5 compliance gap-fix (2026-07-14) ---
     * Five counters this checklist's areas 1/2/3/4/5 required but the struct
     * above never tracked: transfer *counts* (area 1 wanted count alongside
     * the existing bytes/time), synchronization *time* (area 2 only had a
     * count), a kernel *launch count* (area 3 had none at all), and tensor
     * allocation/free *time* (area 4 only had counts/hit-rate/peak-bytes).
     * total_runtime_ms/cpu_orchestration_time_ms (area 5) are deliberately
     * NOT stored here — total_runtime_ms depends on a session-start
     * timestamp gpuBridgeInit() captures separately in gpubridge_runtime.c, and
     * cpu_orchestration_time_ms is derived from it at print time, the same
     * "not stored, computed at print time" treatment total_gpu_path_time_ms
     * and pool_hit_rate already get. */
    size_t host_to_device_count;
    size_t device_to_host_count;
    double synchronization_time_ms;
    size_t kernel_launch_count;
    double tensor_alloc_time_ms;
    double tensor_free_time_ms;
    /* --- Phase 4 additions (spec4.md sections 5, 6) ---
     * runtime_init_time_ms: the runtime's own cold-start cost, a documented
     * subset of cpu_orchestration_time_ms (already folded into that bucket
     * above; this is a labeled breakout, not an additional deduction) —
     * bracketed around gpuBridgeInit()/gpuBridgeSetBackend()'s select_and_init()
     * call in gpubridge_runtime.c.
     * model_load_time_ms: cumulative across possibly-multiple
     * gpuBridgeBeginModelLoad()/gpuBridgeEndModelLoad() brackets in one session
     * (e.g. multiple weight shards) — deliberately OVERLAPS with
     * host_to_device_time_ms/tensor_alloc_time_ms if the bracketed work
     * includes real device transfers/allocations, the same "not an exclusive
     * bucket" relationship total_runtime_ms already has to every other
     * *_time_ms field.
     * kv_cache_allocation_count/kv_cache_bytes: incremented only by
     * gpuBridgeTensorAllocKvCache() (section 6.1), never by ordinary
     * gpuBridgeTensorAlloc() calls. kv_cache_bytes is cumulative/monotonic,
     * same treatment as peak_device_bytes above. */
    double runtime_init_time_ms;
    double model_load_time_ms;
    size_t kv_cache_allocation_count;
    size_t kv_cache_bytes;
} GpuBridgeProfileStats;

/* --- Phase 4 addition (spec4.md section 9) ---
 * Best-effort hardware-tier classification — NOT a measured hardware fact.
 * See gpuBridgeClassifyHardwareTierFromDeviceName()'s own comment below for
 * why this is a heuristic with a real, documented false-negative risk.
 * Every place this value reaches a human (GPUBRIDGE_PROFILE=1 output in
 * particular) must carry that caveat rather than presenting it as
 * authoritative. */
typedef enum {
    GPUBRIDGE_HARDWARE_TIER_UNKNOWN,
    GPUBRIDGE_HARDWARE_TIER_CPU_ONLY,
    GPUBRIDGE_HARDWARE_TIER_INTEGRATED_GPU,
    GPUBRIDGE_HARDWARE_TIER_DISCRETE_GPU
} GpuBridgeHardwareTier;

/* Best-effort, NOT authoritative — OpenCL has no portable CL_DEVICE_TYPE-
 * level distinction between integrated and discrete GPUs (CL_DEVICE_TYPE
 * only separates CPU/GPU/ACCELERATOR). This is a device-name substring
 * heuristic covering common Intel/AMD integrated-GPU naming only; any
 * unmatched name (including real integrated GPUs with unusual strings)
 * defaults to DISCRETE_GPU — a real, documented false-negative risk, not
 * hidden behind false precision. Pure function: string in, enum out,
 * independently unit-testable without any backend. */
GpuBridgeHardwareTier gpuBridgeClassifyHardwareTierFromDeviceName(const char* device_name);
const char* gpuBridgeHardwareTierToString(GpuBridgeHardwareTier tier);

/* Pure decision function, copied verbatim from optmization.md section 7.
 * `bytes` is not referenced by any branch of the current rule (kept for API
 * stability; a future milestone's rule may use it). */
GpuBridgeMemoryStrategy gpuBridgeChooseMemoryStrategy(
    size_t bytes,
    int reuse_count,
    bool latency_sensitive,
    const GpuBridgeBackendCaps* caps
);

const char* gpuBridgeMemoryStrategyToString(GpuBridgeMemoryStrategy strategy);

#endif /* GPUBRIDGE_MEMORY_H */
