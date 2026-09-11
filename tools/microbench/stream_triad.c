// STREAM triad — establishes *achievable* DRAM bandwidth vs the peak from the
// profile (channels * MT/s * 8B). a[i] = b[i] + s*c[i] moves 24 bytes/elem
// (2 reads + 1 write). Arrays sized well beyond LLC to hit DRAM.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    size_t N = (argc > 1) ? atol(argv[1]) : (size_t)1 << 28;  // 256M doubles ~2GB/array
    int reps = (argc > 2) ? atoi(argv[2]) : 20;
    double *a = aligned_alloc(64, N * sizeof(double));
    double *b = aligned_alloc(64, N * sizeof(double));
    double *c = aligned_alloc(64, N * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }

    const double s = 3.0;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) a[i] = b[i] + s * c[i];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (sec < best) best = sec;
    }
    double gbs = 3.0 * N * sizeof(double) / best / 1e9;  // 3 arrays touched
    volatile double sink = a[N - 1]; (void)sink;
    int nt = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    { nt = omp_get_num_threads(); }
    #endif
    printf("{\"threads\": %d, \"achievable_bw_gbs\": %.1f}\n", nt, gbs);
    return 0;
}
