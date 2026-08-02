/*
 * gpubridge_ir.h — GPUBridgeIR: the minimal backend-neutral kernel representation.
 *
 * This is the "IR" (intermediate representation) layer described in
 * spec1.md section 7. It exists so that a logical operation (right now,
 * only vector_add_f32) can be described once, independent of which backend
 * (CPU or OpenCL) will actually execute it. The runtime (gpubridge_runtime.c) reads
 * a GpuBridgeKernelIR and dispatches to whichever GpuBridgeBackend (gpubridge_backend.h) was
 * selected — the IR itself has no knowledge of CPU or OpenCL.
 *
 * This milestone's IR is intentionally tiny: it only needs to describe a
 * single data-parallel elementwise kernel. Later milestones (see
 * milestone.md section 7) grow this into a much richer tensor-op IR with
 * ranks, shapes, GEMM, convolution, autodiff, etc. None of that exists yet.
 */
#ifndef GPUBRIDGE_IR_H
#define GPUBRIDGE_IR_H

#include <stdbool.h>
#include <stddef.h>

/* Milestone 2.5 (spec2.5.md section 6): GpuBridgeTensorDesc gains a
 * GpuBridgeMemoryStrategy field, so this leaf header is needed here too —
 * already included independently by gpubridge_backend.h/gpubridge_runtime.h
 * since spec1.5.md, no new dependency cycle is introduced. */
#include "gpubridge_memory.h"

/* Scalar element type carried by a GpuBridgeBufferArg. GPUBRIDGE_TYPE_F32 was the
 * only type through Milestone 3; more (f64, bf16, fp16, i32, ...) are added
 * in later milestones.
 *
 * --- Phase 4 additions (spec4.md section 7) ---
 * GPUBRIDGE_TYPE_INT8_QUANTIZED/GPUBRIDGE_TYPE_INT4_QUANTIZED are metadata
 * only — no dequant kernel, no int8/int4 GEMM exists yet (spec4.md section 4
 * non-goal 2). Deliberately generic names, not GGUF's "Q8_0"/"Q4_0" —
 * Phase 5.5's investigation may still change what layout this project
 * actually adopts, and this spec should not presume that outcome.
 * gpuBridgeTensorAlloc()/gpuBridgeTensorAllocKvCache() (gpubridge_runtime.c)
 * reject both values outright (spec4.md section 7) — see
 * gpuBridgeIrDescribeQuantizedTensor() below for how a caller describes one
 * instead. */
typedef enum {
    GPUBRIDGE_TYPE_F32,
    GPUBRIDGE_TYPE_INT8_QUANTIZED,
    GPUBRIDGE_TYPE_INT4_QUANTIZED
} GpuBridgeScalarType;

/* --- Phase 4 addition (spec4.md section 7) ---
 * Parallel descriptor struct: an enum alone cannot carry a scale/zero-point.
 * Only meaningful when a GpuBridgeTensorDesc's `type` is one of the two
 * quantized values above; zeroed/unused otherwise. */
typedef struct {
    float scale;      /* real_value = quantized_value * scale */
    int   zero_point; /* 0 for the symmetric case both enum values assume */
    int   block_size; /* elements per quantization block; 0 = per-tensor (single block) */
} GpuBridgeQuantDesc;

/* --- Phase 4 addition (spec4.md section 6) ---
 * Which role a GpuBridgeTensorDesc plays. GENERIC (the implicit default, = 0,
 * for every existing tensor) covers everything through Milestone 3;
 * KV_CACHE is only ever set by gpuBridgeTensorAllocKvCache()
 * (gpubridge_runtime.c), which also bypasses the tensor pool on both
 * allocation and free — see that function's own comment for why a new
 * allocating entry point was needed instead of a post-hoc mutator. */
typedef enum {
    GPUBRIDGE_TENSOR_USAGE_GENERIC,
    GPUBRIDGE_TENSOR_USAGE_KV_CACHE
} GpuBridgeTensorUsage;

/* Which logical operation a GpuBridgeKernelIR/GpuBridgeTensorKernelIR describes.
 * GPUBRIDGE_OP_VECTOR_ADD is the only op gpuBridgeLaunchKernel() accepts (unchanged from
 * spec1.md). The four new values (spec2.md section 10) are only accepted by
 * the new gpuBridgeLaunchTensorKernel() entry point — kept as one enum rather than
 * two so a single GpuBridgeOpKind value always identifies one op regardless of
 * which IR struct/launch function carries it. */
typedef enum {
    GPUBRIDGE_OP_VECTOR_ADD,
    GPUBRIDGE_OP_MATMUL,
    GPUBRIDGE_OP_REDUCE_SUM,
    GPUBRIDGE_OP_BROADCAST_ADD,
    GPUBRIDGE_OP_RELU
} GpuBridgeOpKind;

/*
 * Describes one input or output buffer argument to a kernel: its name (for
 * diagnostics), element type, element count, and whether it is read
 * (is_input), written (is_output), or both. For vector_add_f32, `a` and `b`
 * are inputs and `c` is output.
 */
typedef struct {
    const char* name;
    GpuBridgeScalarType type;
    size_t length;
    bool is_input;
    bool is_output;
} GpuBridgeBufferArg;

/*
 * A full description of one kernel launch: which logical kernel to run
 * (kernel_name / op_kind), its argument list, and how many parallel
 * "lanes" to launch (global_size — one thread/work-item per output
 * element, matching the OpenCL kernel's get_global_id(0) indexing and the
 * CPU backend's plain for-loop bound).
 */
typedef struct {
    const char* kernel_name;
    GpuBridgeOpKind op_kind;
    GpuBridgeBufferArg* args;
    size_t arg_count;
    size_t global_size;
} GpuBridgeKernelIR;

/*
 * Not part of spec1.md's literal GPUBridgeIR shape (section 7) — a small builder to
 * avoid duplicating GpuBridgeKernelIR/GpuBridgeBufferArg construction between callers.
 * out_args must point to storage for at least 3 GpuBridgeBufferArg (a, b, c).
 *
 * Fills out_args[0..2] as (a: input, b: input, c: output), each of length n,
 * and points *out at them with kernel_name="vector_add_f32",
 * op_kind=GPUBRIDGE_OP_VECTOR_ADD, global_size=n. Pure data-shaping — no allocation,
 * no backend interaction.
 */
void gpuBridgeIrInitVectorAddF32(GpuBridgeKernelIR* out, GpuBridgeBufferArg out_args[3], size_t n);

/*
 * --- Milestone 2 additions (spec2.md sections 6, 10) ---
 *
 * Everything below is new; GpuBridgeScalarType, GpuBridgeOpKind's GPUBRIDGE_OP_VECTOR_ADD value,
 * GpuBridgeBufferArg, GpuBridgeKernelIR, and gpuBridgeIrInitVectorAddF32() above are all
 * unchanged and remain fully regression-safe.
 */

/*
 * GpuBridgeTensorDesc — device-resident tensor descriptor (spec2.md section 6,
 * verbatim). Deliberately a subset of milestone.md section 9's
 * training-oriented GpuBridgeTensorDesc (no alpha/beta scaling, no multi-device
 * placement, no ownership/copy-on-write metadata) — only what matmul_f32,
 * reduce_sum_f32, and broadcast_add_f32 need to describe shapes.
 *
 * `device_ptr` is filled in by gpuBridgeTensorAlloc() (an opaque handle from the
 * active backend's malloc_device(), exactly like gpuBridgeMalloc()'s return value)
 * and is only ever valid once `device_resident` is true.
 */
typedef struct {
    GpuBridgeScalarType type;
    int rank;              /* 1 or 2 */
    size_t shape[2];       /* shape[1] unused when rank == 1 */
    void* device_ptr;      /* backend-owned device memory */
    bool device_resident;  /* true once allocated via gpuBridgeTensorAlloc */
    /* --- Milestone 2.5 addition (spec2.5.md section 6) --- */
    GpuBridgeMemoryStrategy memory_strategy; /* set by gpuBridgeTensorAlloc */
    /* --- Phase 4 additions (spec4.md section 6) ---
     * Both are explicitly assigned by tensor_alloc_impl() (gpubridge_runtime.c)
     * for every successful allocation — gpuBridgeTensorAlloc() does not
     * zero-initialize *desc before writing to it, so leaving these to a
     * struct-literal zero-default would carry uninitialized garbage. */
    GpuBridgeTensorUsage usage; /* GENERIC unless allocated via gpuBridgeTensorAllocKvCache() */
    GpuBridgeQuantDesc quant;   /* zeroed/unused unless type is one of the two quantized scalar types */
} GpuBridgeTensorDesc;

/*
 * GpuBridgeTensorArg — one input or output tensor argument to a tensor-op kernel,
 * the GpuBridgeTensorKernelIR analogue of GpuBridgeBufferArg. GpuBridgeBufferArg itself is
 * unchanged and still used for vector_add_f32; this is a new, parallel type
 * because tensor ops need rank/shape rather than a flat element count.
 */
typedef struct {
    const char* name;
    GpuBridgeScalarType type;
    int rank;               /* 1 or 2 */
    size_t shape[2];
    bool is_input;
    bool is_output;
} GpuBridgeTensorArg;

/*
 * GpuBridgeTensorKernelIR — one tensor-op kernel launch description, parallel to
 * GpuBridgeKernelIR (which remains unchanged and still describes vector_add_f32
 * only). global_size[0] is the only entry used for rank-1-shaped launches
 * (reduce_sum_f32, and relu_f32 when its input is rank 1); both entries are
 * used for 2D launches (matmul_f32, broadcast_add_f32, and relu_f32 when its
 * input is rank 2).
 */
typedef struct {
    const char* kernel_name;
    GpuBridgeOpKind op_kind;
    GpuBridgeTensorArg* args;
    size_t arg_count;
    size_t global_size[2];
} GpuBridgeTensorKernelIR;

/*
 * One builder per new op (spec2.md section 10), following
 * gpuBridgeIrInitVectorAddF32()'s exact pattern: pure data-shaping, no allocation,
 * no backend interaction. out_args must point to storage for at least as
 * many GpuBridgeTensorArg as the op takes (3 for matmul, 2 for the other three).
 *
 * gpuBridgeIrInitMatmulF32:       args = (a: in [m,k], b: in [k,n], c: out [m,n]),
 *                          global_size = {m, n} (one lane per output elem).
 * gpuBridgeIrInitReduceSumF32:    args = (a: in [n], result: out [1]),
 *                          global_size = {n, 0} (rank-1 op).
 * gpuBridgeIrInitBroadcastAddF32: args = (y: in+out [rows,cols], b: in [cols]) —
 *                          y is both read and written, matching section 7.3's
 *                          in-place-on-X's-buffer semantics; global_size =
 *                          {rows, cols}.
 * gpuBridgeIrInitReluF32:         args = (x: in, y: out), same rank/shape as given;
 *                          global_size = {shape[0], shape[1]} for rank 2, or
 *                          {shape[0], 0} for rank 1 (flattened elsewhere by
 *                          the launch path, since relu is purely elementwise).
 */
void gpuBridgeIrInitMatmulF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[3],
    size_t m, size_t k, size_t n);
void gpuBridgeIrInitReduceSumF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[2],
    size_t n);
void gpuBridgeIrInitBroadcastAddF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[2],
    size_t rows, size_t cols);
void gpuBridgeIrInitReluF32(GpuBridgeTensorKernelIR* out, GpuBridgeTensorArg out_args[2],
    int rank, const size_t shape[2]);

/*
 * --- Milestone 2.5 addition (spec2.5.md section 8) ---
 *
 * GpuBridgeTensorMirror — host/device staleness tracking for one tensor,
 * closing spec2.md section 8.4's explicit deferral ("GpuBridgeArrayMirror-style
 * host_valid/device_valid staleness tracking... for a later milestone").
 * Deliberately a new type, not a further change to GpuBridgeTensorDesc above
 * (which only gained the one memory_strategy field) — every existing
 * spec2.md builder/test stays untouched.
 *
 * Named GpuBridgeTensorMirror, not optmization.md section 9's literal
 * GpuBridgeArrayMirror — consistent with this project's GpuBridgeTensor*
 * naming convention (GpuBridgeTensorDesc/GpuBridgeTensorArg/GpuBridgeTensorKernelIR),
 * a documented naming deviation (spec2.5.md section 8.1).
 *
 * `host_ptr` is caller-owned (never allocated or freed by the mirror API);
 * `tensor` must already be device_resident (e.g. freshly returned by
 * gpuBridgeTensorAlloc) when gpuBridgeTensorMirrorCreate() binds it. The six
 * functions operating on this type are declared in gpubridge_runtime.h,
 * alongside gpuBridgeTensorAlloc/gpuBridgeTensorFree, since they call into the
 * runtime (gpuBridgeMemcpy/gpuBridgeTensorFree) rather than being pure data
 * shaping like the IR builders above.
 */
typedef struct {
    void* host_ptr;
    GpuBridgeTensorDesc tensor;
    size_t bytes;
    bool host_valid;
    bool device_valid;
} GpuBridgeTensorMirror;

/*
 * --- Phase 2 addition (milestone.md section 22, "Common GPUBridge GPU/Tensor
 * IR" deliverable 4: "IR validator") ---
 *
 * Prior to this, gpuBridgeLaunchKernel()/gpuBridgeLaunchTensorKernel() (gpubridge_runtime.c)
 * each did their own ad-hoc arg-count checks inline and nothing else — no
 * type, shape, or rank consistency was ever checked. These two functions
 * consolidate that into one reusable, independently testable validator per
 * IR type, and add real shape-consistency checks the inline checks never
 * had (e.g. matmul's inner dimensions actually matching, bias length
 * matching y's column count, relu's x/y shapes matching).
 *
 * Both follow gpubridge_selector.c's gpuBridgeSelectBackend() error-reporting
 * convention: return true/false, and on false write a human-readable reason
 * into the caller-owned err_buf (err_buf_size bytes). `arg_count` is the
 * actual device-pointer-array length the caller is about to pass to
 * gpuBridgeLaunchKernel/gpuBridgeLaunchTensorKernel, checked against the IR's own
 * kernel->arg_count for consistency.
 *
 * gpuBridgeIrValidateKernel accepts only GPUBRIDGE_OP_VECTOR_ADD (the only op
 * gpuBridgeLaunchKernel understands); gpuBridgeIrValidateTensorKernel accepts the
 * four tensor ops (GPUBRIDGE_OP_MATMUL/_REDUCE_SUM/_BROADCAST_ADD/_RELU) and
 * rejects GPUBRIDGE_OP_VECTOR_ADD or any other value, since that's not a tensor
 * op — this mirrors the existing gpuBridgeLaunchKernel/gpuBridgeLaunchTensorKernel
 * split.
 */
bool gpuBridgeIrValidateKernel(const GpuBridgeKernelIR* kernel, size_t arg_count,
    char* err_buf, size_t err_buf_size);
bool gpuBridgeIrValidateTensorKernel(const GpuBridgeTensorKernelIR* kernel, size_t arg_count,
    char* err_buf, size_t err_buf_size);

/*
 * --- Phase 4 addition (spec4.md section 7) ---
 *
 * The only way to describe a quantized tensor's shape/scale/zero-point,
 * since gpuBridgeTensorAlloc()/gpuBridgeTensorAllocKvCache() reject
 * GPUBRIDGE_TYPE_INT8_QUANTIZED/GPUBRIDGE_TYPE_INT4_QUANTIZED outright (no
 * dequant kernel or int8/int4 GEMM exists yet to consume one). Pure
 * data-shaping, no allocation, no backend interaction, same shape as every
 * gpuBridgeIrInit*F32 builder above: device_resident is always false and
 * device_ptr is always NULL — this describes intent only. No consumer reads
 * this yet (Phase 4 vocabulary).
 */
void gpuBridgeIrDescribeQuantizedTensor(GpuBridgeTensorDesc* out,
    GpuBridgeScalarType quant_type, int rank, const size_t shape[2],
    GpuBridgeQuantDesc quant);

#endif /* GPUBRIDGE_IR_H */
