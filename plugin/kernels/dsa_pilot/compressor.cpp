// DSv4 DSA compressor (softmax-pool) — Phase-A kernel (pilot, authored FRESH).
// op: out[n,d] = sum_r softmax_r(score[n,r,d] + ape[r,d]) * kv[n,r,d]   (per-channel pool over window R).
// Reference: intel_cpu_models/dsa_compressor_cpu.compress_softmax_pool (torch oracle).
// Optimization: ONE streaming pass with online softmax, vectorized over the channel dim D and
// parallel over N rows — the weight tensor w[N,R,D] is never materialized (no DRAM temporary).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>

// kv,score:[N,R,D] (fp32); ape:[R,D] (fp32) -> out:[N,D] (fp32). Semantics == compress_softmax_pool.
torch::Tensor compressor_softmax_pool(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  TORCH_CHECK(kv.dim() == 3 && score.dim() == 3 && ape.dim() == 2, "kv/score must be [N,R,D], ape [R,D]");
  TORCH_CHECK(score.size(0) == kv.size(0) && score.size(1) == kv.size(1) && score.size(2) == kv.size(2),
              "score shape must equal kv shape");
  TORCH_CHECK(ape.size(0) == kv.size(1) && ape.size(1) == kv.size(2), "ape must be [R,D] matching kv");
  TORCH_CHECK(kv.device().is_cpu() && score.device().is_cpu() && ape.device().is_cpu(), "CPU tensors only");
  auto kvc = kv.to(torch::kFloat32).contiguous();
  auto scc = score.to(torch::kFloat32).contiguous();
  auto apc = ape.to(torch::kFloat32).contiguous();
  const int64_t N = kvc.size(0), R = kvc.size(1), D = kvc.size(2);
  auto out = torch::empty({N, D}, torch::kFloat32);
  const float* kp = kvc.data_ptr<float>();
  const float* sp = scc.data_ptr<float>();
  const float* ap = apc.data_ptr<float>();
  float* op = out.data_ptr<float>();

  // Parallelize over N * D-tiles (the softmax is independent per channel) so all cores are used even at
  // small N (N-only parallelism left cores idle and REGRESSED vs torch at M=1). DT >= 16 (one AVX-512 vec).
  const int64_t nthr = at::get_num_threads();
  int64_t dpt = std::max<int64_t>(1, (2 * nthr + N - 1) / N);        // d-tiles per n (target ~2*threads tasks)
  int64_t DT = std::max<int64_t>(16, (D + dpt - 1) / dpt);
  DT = (DT + 15) / 16 * 16;                                          // round to an AVX-512 width
  const int64_t ndt = (D + DT - 1) / DT;

  at::parallel_for(0, N * ndt, 0, [&](int64_t t0, int64_t t1) {
    std::vector<float> m(DT), l(DT), acc(DT);
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t n = t / ndt, dt = t % ndt, d0 = dt * DT;
      const int64_t dd = std::min(DT, D - d0);
      for (int64_t j = 0; j < dd; ++j) { m[j] = -INFINITY; l[j] = 0.f; acc[j] = 0.f; }
      for (int64_t r = 0; r < R; ++r) {
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        const float* krow = kp + (n * R + r) * D + d0;
        #pragma omp simd
        for (int64_t j = 0; j < dd; ++j) {
          // F1: a MASKED position (score==-inf) contributes nothing (corr=0 while running max is -inf; e=0).
          float x = srow[j] + arow[j];
          float mnew = x > m[j] ? x : m[j];
          float corr = (m[j] == -INFINITY) ? 0.f : std::exp(m[j] - mnew);
          float e = (x == -INFINITY) ? 0.f : std::exp(x - mnew);
          l[j] = l[j] * corr + e;
          acc[j] = acc[j] * corr + e * krow[j];
          m[j] = mnew;
        }
      }
      float* orow = op + n * D + d0;
      #pragma omp simd
      for (int64_t j = 0; j < dd; ++j) orow[j] = (l[j] > 0.f) ? acc[j] / l[j] : 0.f;  // all-masked -> 0
    }
  });
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compressor_softmax_pool", &compressor_softmax_pool,
        "DSA compressor softmax-pool (streaming online-softmax, no w[N,R,D] temporary)");
}
