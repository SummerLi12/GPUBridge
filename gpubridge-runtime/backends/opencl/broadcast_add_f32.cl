/*
 * broadcast_add_f32.cl — Y[i,j] += bias[j] for i in [0,rows), j in [0,cols).
 *
 * One work-item per output element, 2D global size (rows, cols) (spec2.md
 * section 7.3). In-place on y's buffer, matching the `Y += b` style in
 * milestone.md section 7.5's dense_forward_f32 example.
 */
__kernel void broadcast_add_f32(
    __global float* y,
    __global const float* bias,
    int rows,
    int cols
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < rows && j < cols) {
        y[i * cols + j] += bias[j];
    }
}
