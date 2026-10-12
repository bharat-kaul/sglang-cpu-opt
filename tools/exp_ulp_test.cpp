// ULP characterization of exp512_ps (the inlined Cephes exp used by the DSA kernels) vs a correctly-rounded
// fp64 reference, over the kernels' actual argument range (softmax feeds exp(x-max) <= 0, so x in [-87.3,0]).
// Industry-standard acceptance: faithfully rounded ~1 ULP. Build: g++ -O3 -march=native tools/exp_ulp_test.cpp
#include "../plugin/kernels/dsa_pilot/avx512_exp.h"
#include <cmath>
#include <cstdio>

int main() {
#if !defined(__AVX512F__)
  std::printf("no AVX512 on build host -> exp512_ps compiles to scalar fallback; run on an AVX-512 node\n");
  return 0;
#else
  double max_ulp = 0.0, x_at_max = 0.0;
  long n_half = 0, n_one = 0, n_two = 0, total = 0;
  float buf[16], out[16];
  for (double x = -87.3; x <= 1e-4; x += 1e-4) {
    for (int j = 0; j < 16; ++j) buf[j] = (float)(x + j * 1e-6);
    _mm512_storeu_ps(out, exp512_ps(_mm512_loadu_ps(buf)));
    for (int j = 0; j < 16; ++j) {
      float xf = buf[j];
      float t = (float)std::exp((double)xf);          // correctly-rounded-to-fp32 reference
      float c = out[j];
      float u = std::nextafterf(t, INFINITY) - t;     // 1 ULP(fp32) at t
      double ulp = (u > 0.f) ? std::fabs((double)c - (double)t) / (double)u : 0.0;
      if (ulp > max_ulp) { max_ulp = ulp; x_at_max = xf; }
      n_half += (ulp > 0.5);
      n_one += (ulp > 1.0);
      n_two += (ulp > 2.0);
      ++total;
    }
  }
  std::printf("exp512_ps ULP vs fp64-exact, x in [-87.3,0], N=%ld:\n", total);
  std::printf("  max_ulp = %.3f   frac>0.5ULP = %.3f%%   frac>1ULP = %.4f%%   frac>2ULP = %.4f%%   (worst x=%.5f)\n",
              max_ulp, 100.0 * n_half / total, 100.0 * n_one / total, 100.0 * n_two / total, x_at_max);
  std::printf("  verdict: %s (industry bar for a hot vector exp = ~1 ULP faithfully rounded)\n",
              max_ulp <= 2.0 ? "within ~1-2 ULP, OK" : "LOOSER than 2 ULP, investigate");
  return 0;
#endif
}
