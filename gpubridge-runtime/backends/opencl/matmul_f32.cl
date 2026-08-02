/*
 * matmul_f32.cl — C[m,n] = sum_k( A[m,k] * B[k,n] ).
 *
 * One work-item per output element C[row,col] (spec2.md section 7.1) — no
 * tiling, no local/shared memory, matching this milestone's naive-kernel
 * performance non-goal (section 5 item 12) and vector_add_f32.cl's
 * one-thread-per-element precedent.
 */
__kernel void matmul_f32(
    __global const float* a,
    __global const float* b,
    __global float* c,
    int m,
    int k,
    int n
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += a[row * k + p] * b[p * n + col];
        }
        c[row * n + col] = sum;
    }
}
