// MKL BF16 GEMM with PRE-PACKED weights — tests the #1 BRGEMM/TPP lesson:
// pack the stationary operand (B / weights) into AMX-tile layout ONCE and reuse,
// so the timed inner loop only streams A and accumulates (mirrors inference where
// weights are packed at load time). Compares against per-call packing overhead.
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static MKL_BF16 f2bf16(float f) {   // round-to-nearest-even top 16 bits
    unsigned int x; __builtin_memcpy(&x, &f, 4);
    unsigned int r = (x >> 16) & 1; x += 0x7fff + r; return (MKL_BF16)(x >> 16);
}
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 8192;
    int N = argc > 2 ? atoi(argv[2]) : 8192;
    int K = argc > 3 ? atoi(argv[3]) : 8192;
    int iters = argc > 4 ? atoi(argv[4]) : 50;

    MKL_BF16 *A = mkl_malloc((size_t)M * K * 2, 64);
    MKL_BF16 *B = mkl_malloc((size_t)K * N * 2, 64);
    float *C = mkl_malloc((size_t)M * N * 4, 64);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = f2bf16((float)(rand() % 7 - 3) * 0.1f);
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = f2bf16((float)(rand() % 7 - 3) * 0.1f);

    // --- Pre-pack B (weights) ONCE into AMX-tile layout ---
    size_t bsz = cblas_gemm_bf16bf16f32_pack_get_size(CblasBMatrix, M, N, K);
    MKL_BF16 *Bp = mkl_malloc(bsz, 64);
    cblas_gemm_bf16bf16f32_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, M, N, K, B, N, Bp);

    // warmup
    for (int i = 0; i < 5; i++)
        cblas_gemm_bf16bf16f32_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
            M, N, K, 1.0f, A, K, Bp, N, 0.0f, C, N);

    // timed: compute-only, weights already packed (the inference case)
    double t0 = now();
    for (int i = 0; i < iters; i++)
        cblas_gemm_bf16bf16f32_compute(CblasRowMajor, CblasNoTrans, CblasPacked,
            M, N, K, 1.0f, A, K, Bp, N, 0.0f, C, N);
    double dt = (now() - t0) / iters;
    double tf = 2.0 * M * N * K / dt / 1e12;

    printf("{\"impl\": \"mkl_bf16_amx_prepacked\", \"M\": %d, \"N\": %d, \"K\": %d, "
           "\"threads\": %d, \"tflops\": %.1f}\n", M, N, K, mkl_get_max_threads(), tf);

    mkl_free(A); mkl_free(B); mkl_free(C); mkl_free(Bp);
    return 0;
}
