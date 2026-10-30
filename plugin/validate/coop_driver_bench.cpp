// C++ per-layer DRIVER demonstrator for the DSV4 monolith: proves that sequencing REAL cooperative kernels
// (AVX-512 combine+rmsnorm, the inc-3/4 kernel) inside ONE omp region per layer beats running them as separate
// per-op fork-join calls — the fork-join saving the monolith is built to capture, measured on realistic work.
//
// Models a layer as K cooperative "glue" ops (combine+rmsnorm) over 43 layers, two ways:
//   (A) per-op fork-join : each op is its own `#pragma omp parallel for`      (today's structure)
//   (B) one region/layer : one `#pragma omp parallel`, each op an `omp for`   (the driver)
// Build: g++ -O3 -march=native -mavx512bf16 -fopenmp coop_driver_bench.cpp -o bench
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <immintrin.h>

static constexpr int HC = 4;

// one row of combine (sum_k pre[k]*x[k*H+h]) + rmsnorm, AVX-512, fp32 in/out for simplicity.
static inline void combine_rmsnorm_row(const float* x, const float* pre, const float* w,
                                       float* out, float* ytmp, int H, float eps) {
  __m512 vss = _mm512_setzero_ps();
  for (int h = 0; h < H; h += 16) {
    __m512 acc = _mm512_setzero_ps();
    for (int k = 0; k < HC; ++k)
      acc = _mm512_fmadd_ps(_mm512_set1_ps(pre[k]), _mm512_loadu_ps(x + k * H + h), acc);
    _mm512_storeu_ps(ytmp + h, acc);
    vss = _mm512_fmadd_ps(acc, acc, vss);
  }
  float inv = 1.f / std::sqrt(_mm512_reduce_add_ps(vss) / H + eps);
  __m512 vinv = _mm512_set1_ps(inv);
  for (int h = 0; h < H; h += 16)
    _mm512_storeu_ps(out + h, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(ytmp + h), vinv),
                                            _mm512_loadu_ps(w + h)));
}

int main(int argc, char** argv) {
  int T = argc > 1 ? atoi(argv[1]) : 32;
  int K = argc > 2 ? atoi(argv[2]) : 24;     // cooperative glue ops per layer
  int M = argc > 3 ? atoi(argv[3]) : 32;     // rows (batch)
  int H = argc > 4 ? atoi(argv[4]) : 4096;
  int NLAYERS = 43;
  omp_set_num_threads(T);
  std::vector<float> x(size_t(M) * HC * H, 0.5f), w(H, 1.0f), pre(size_t(M) * HC, 0.25f);
  std::vector<float> out(size_t(M) * H * K);
  int nth = omp_get_max_threads();
  std::vector<std::vector<float>> scratch(nth, std::vector<float>(H));

  auto op_row = [&](int op, int m) {
    combine_rmsnorm_row(x.data() + size_t(m) * HC * H, pre.data() + size_t(m) * HC, w.data(),
                        out.data() + (size_t(op) * M + m) * H, scratch[omp_get_thread_num()].data(), H, 1e-6f);
  };

  auto run_A = [&]() {  // per-op fork-join
    for (int l = 0; l < NLAYERS; ++l)
      for (int op = 0; op < K; ++op) {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) op_row(op, m);
      }
  };
  auto run_B = [&]() {  // one cooperative region per layer
    for (int l = 0; l < NLAYERS; ++l) {
      #pragma omp parallel
      for (int op = 0; op < K; ++op) {
        #pragma omp for schedule(static)
        for (int m = 0; m < M; ++m) op_row(op, m);
      }
    }
  };

  auto bench = [&](const char* name, auto fn) {
    double best = 1e30;
    for (int r = 0; r < 12; ++r) {
      auto t0 = std::chrono::high_resolution_clock::now();
      fn();
      auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    printf("  %-24s  %.3f ms/step  (sink=%.1f)\n", name, best * 1e3, out[0]);
    return best;
  };
  printf("T=%d  K=%d glue-ops/layer  M=%d  H=%d  layers=%d  (real combine+rmsnorm work)\n", T, K, M, H, NLAYERS);
  double a = bench("(A) per-op fork-join", run_A);
  double b = bench("(B) one region/layer", run_B);
  printf("  => DRIVER speedup = %.2fx\n", a / b);
  return 0;
}
