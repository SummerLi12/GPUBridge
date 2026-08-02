/*
 * relu_f32.cl — y[i] = max(0, x[i]).
 *
 * One work-item per element, flattened global size (rows * cols for rank 2,
 * or just n for rank 1) — see spec2.md section 7.4. Elementwise with no
 * cross-element structure, so the rank of the underlying tensor doesn't
 * matter here.
 */
__kernel void relu_f32(
    __global const float* x,
    __global float* y,
    int count
) {
    int i = get_global_id(0);
    if (i < count) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}
