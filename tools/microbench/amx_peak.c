// AMX BF16 compute-peak microkernel — establishes the *achievable* TMUL ceiling.
// Keeps all operands resident in tile registers and issues back-to-back
// TDPBF16PS into 4 independent accumulators (hides latency behind throughput),
// so the measured rate is pure compute with zero memory traffic. This empirically
// resolves the per-core FLOP/cycle constant instead of assuming it.
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

typedef struct {
    uint8_t palette;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tilecfg;

static int enable_amx(void) {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
}

static double per_thread_tflops(long iters) {
    tilecfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette = 1;
    for (int t = 0; t < 8; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
    _tile_loadconfig(&cfg);

    // tmm4,5 = A tiles; tmm6,7 = B tiles (VNNI2); tmm0..3 = accumulators.
    __attribute__((aligned(64))) int16_t A[2][16 * 32];
    __attribute__((aligned(64))) int16_t B[2][16 * 32];
    __attribute__((aligned(64))) float   C[4][16 * 16];
    for (int i = 0; i < 16 * 32; i++) { A[0][i] = A[1][i] = 0x3f80 >> 4; B[0][i] = B[1][i] = 0x3f80 >> 4; }

    _tile_loadd(4, A[0], 64); _tile_loadd(5, A[1], 64);
    _tile_loadd(6, B[0], 64); _tile_loadd(7, B[1], 64);
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < iters; i++) {
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    _tile_stored(0, C[0], 64);  // prevent dead-code elimination
    volatile float sink = C[0][0]; (void)sink;

    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double flop = (double)iters * 4.0 * 16384.0;  // 4 TDPBF16PS * (16*16*32*2) FLOP
    return flop / sec / 1e12;
}

int main(int argc, char **argv) {
    long iters = (argc > 1) ? atol(argv[1]) : 20000000L;
    if (enable_amx()) { fprintf(stderr, "AMX enable failed\n"); return 1; }

    double total = 0.0;
    int nthreads = 1;
    #pragma omp parallel reduction(+:total)
    {
        if (enable_amx() == 0) {
            double tf = per_thread_tflops(iters);
            total += tf;
            #ifdef _OPENMP
            #pragma omp single
            nthreads = omp_get_num_threads();
            #endif
        }
    }
    printf("{\"threads\": %d, \"achievable_compute_tflops\": %.1f, \"per_core_tflops\": %.3f}\n",
           nthreads, total, total / nthreads);
    return 0;
}
