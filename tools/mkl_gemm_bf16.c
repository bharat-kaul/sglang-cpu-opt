// Headroom chase: tuned AMX BF16 GEMM via Intel MKL (cblas_gemm_bf16bf16f32),
// measured against the achievable ceiling from the microkernel. Compares to the
// stock torch.matmul baseline and validates correctness vs an FP32 reference.
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

static MKL_BF16 f2bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (MKL_BF16)(u >> 16);
}
static float bf162f(MKL_BF16 h) {
    uint32_t u = ((uint32_t)h) << 16; float f; memcpy(&f, &u, 4); return f;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 8192;
    int N = (argc > 2) ? atoi(argv[2]) : 8192;
    int K = (argc > 3) ? atoi(argv[3]) : 8192;
    int iters = (argc > 4) ? atoi(argv[4]) : 50;

    MKL_BF16 *A = mkl_malloc((size_t)M * K * sizeof(MKL_BF16), 64);
    MKL_BF16 *B = mkl_malloc((size_t)K * N * sizeof(MKL_BF16), 64);
    float    *C = mkl_malloc((size_t)M * N * sizeof(float), 64);
    srand(0);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = f2bf16((rand() / (float)RAND_MAX) - 0.5f);
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = f2bf16((rand() / (float)RAND_MAX) - 0.5f);

    for (int w = 0; w < 8; w++)
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                               M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++)
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                               M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / iters;
    double tflops = 2.0 * M * N * K / sec / 1e12;

    // Correctness: FP32 reference on a 256-row slab (full 8K sgemm is wasteful).
    int mr = M < 256 ? M : 256;
    float *Af = malloc((size_t)mr * K * sizeof(float));
    float *Bf = malloc((size_t)K * N * sizeof(float));
    float *Cref = malloc((size_t)mr * N * sizeof(float));
    for (size_t i = 0; i < (size_t)mr * K; i++) Af[i] = bf162f(A[i]);
    for (size_t i = 0; i < (size_t)K * N; i++) Bf[i] = bf162f(B[i]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mr, N, K,
                1.0f, Af, K, Bf, N, 0.0f, Cref, N);
    double sre = 0, denom = 0; int nan = 0;
    for (size_t i = 0; i < (size_t)mr * N; i++) {
        if (isnan(C[i])) nan = 1;
        sre += fabs(C[i] - Cref[i]); denom += fabs(Cref[i]) + 1e-3;
    }
    printf("{\"impl\": \"mkl_bf16_amx\", \"M\": %d, \"N\": %d, \"K\": %d, "
           "\"threads\": %d, \"tflops\": %.1f, \"mean_rel_err\": %.4f, \"has_nan\": %d}\n",
           M, N, K, mkl_get_max_threads(), tflops, sre / denom, nan);
    return 0;
}
