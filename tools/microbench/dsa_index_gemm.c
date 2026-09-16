// Baseline microbench for the DSA index_gemm_kernel_nn (P@V gather-accumulate).
// Faithful to the SGLang kernel's compute+memory pattern: C[M,N] += sum_k A[m,k] *
// B[indices[k], n], B in bf16 gathered from a large pool (random access), FP32 FMA.
// Compares the current 4-pass path (BLOCK_M=4 re-gathers B M/4 times) vs a
// single-pass M=16 (gather B once) to test the roofline claim that the lever is
// B-traffic reduction, not AMX compute. Reports GFLOP/s + effective B GB/s.
//
// build: gcc -O3 -march=native dsa_index_gemm.c -o dsa_index_gemm
// run:   ./dsa_index_gemm <M> <K> <N> <POOL_tokens> <iters> <mode 0=4pass|1=1pass>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline float bf16_to_f32(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static double now() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16;
  int K = argc > 2 ? atoi(argv[2]) : 512;    // index_topk
  int N = argc > 3 ? atoi(argv[3]) : 128;    // head_dim
  long POOL = argc > 4 ? atol(argv[4]) : 262144; // KV pool tokens (random gather)
  int iters = argc > 5 ? atoi(argv[5]) : 2000;
  int mode = argc > 6 ? atoi(argv[6]) : 1;   // 0=4pass(current M=16), 1=single-pass
  int BLOCK_M = (mode == 0) ? 4 : M;

  float* A = aligned_alloc(64, (size_t)M * K * sizeof(float));
  uint16_t* B = aligned_alloc(64, (size_t)POOL * N * sizeof(uint16_t));
  float* C = aligned_alloc(64, (size_t)M * N * sizeof(float));
  int* idx = aligned_alloc(64, (size_t)K * sizeof(int));
  for (long i = 0; i < (long)M * K; ++i) A[i] = (float)(i % 7) * 0.01f;
  for (long i = 0; i < (long)POOL * N; ++i) B[i] = (uint16_t)(0x3f80 + (i & 0x3f)); // ~1.0
  srand(1);

  double t0 = now();
  volatile float sink = 0;
  for (int it = 0; it < iters; ++it) {
    for (int k = 0; k < K; ++k) idx[k] = rand() % POOL;   // fresh random gather each iter
    memset(C, 0, (size_t)M * N * sizeof(float));
    // mimic the kernel's M-blocking: each M-block loops k and gathers B[idx[k]] once,
    // reusing it across the BLOCK_M rows -> B is re-gathered ceil(M/BLOCK_M) times.
    for (int mb = 0; mb < M; mb += BLOCK_M) {
      int mend = mb + BLOCK_M < M ? mb + BLOCK_M : M;
      for (int k = 0; k < K; ++k) {
        const uint16_t* brow = B + (size_t)idx[k] * N;
        for (int m = mb; m < mend; ++m) {
          float a = A[(size_t)m * K + k];
          float* c = C + (size_t)m * N;
          for (int n = 0; n < N; ++n) c[n] += a * bf16_to_f32(brow[n]);
        }
      }
    }
    sink += C[0];
  }
  double t = now() - t0;
  (void)sink;

  double flops = 2.0 * M * N * K * iters;
  int passes = (M + BLOCK_M - 1) / BLOCK_M;
  double b_bytes = (double)passes * K * N * 2.0 * iters; // bf16 B, re-gathered per pass
  printf("{\"mode\":\"%s\",\"M\":%d,\"K\":%d,\"N\":%d,\"POOL\":%ld,\"passes\":%d,"
         "\"sec\":%.4f,\"gflops\":%.2f,\"B_GBps\":%.1f,\"ai_flop_per_byte\":%.2f}\n",
         mode == 0 ? "4pass" : "1pass", M, K, N, POOL, passes, t,
         flops / t / 1e9, b_bytes / t / 1e9, flops / (passes * (double)K * N * 2.0 + (double)M*K*4 + (double)M*N*4));
  free(A); free(B); free(C); free(idx);
  return 0;
}
