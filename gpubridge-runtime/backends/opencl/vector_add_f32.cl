/*
 * vector_add_f32.cl — the OpenCL kernel for c[i] = a[i] + b[i].
 *
 * This is the single source of truth for the GPU kernel text (spec1.md
 * section 13, Option A, verbatim). CMake embeds this exact file's bytes
 * into the binary at configure time (see CMakeLists.txt and
 * cmake/vector_add_f32_cl.h.in) rather than duplicating this text as a
 * hand-written C string literal — edit this file, not any generated header.
 */
__kernel void vector_add_f32(
    __global const float* a,
    __global const float* b,
    __global float* c,
    int n
) {
    /* One work-item per output element (see gpubridge_backend_opencl.c's
     * opencl_launch_vector_add_f32, which sets global_size = n). The bounds
     * check guards against a work-item count that doesn't divide evenly
     * into the driver-chosen local work-group size padding the global range
     * up past n. */
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
