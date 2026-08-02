/*
 * reduce_sum_f32.cl — result[0] = sum_i( a[i] ) for i in [0, n).
 *
 * Naive single-work-group sequential accumulation (spec2.md section 7.2):
 * correctness only, no tree/parallel reduction, no multi-pass — performance
 * is explicitly out of scope for this milestone (section 5 item 12).
 * Launched with a single work-item; that one item walks the whole array.
 */
__kernel void reduce_sum_f32(
    __global const float* a,
    __global float* result,
    int n
) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    result[0] = sum;
}
