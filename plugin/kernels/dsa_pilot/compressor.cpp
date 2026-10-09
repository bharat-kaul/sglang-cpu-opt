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

  at::parallel_for(0, N, 0, [&](int64_t n0, int64_t n1) {
    std::vector<float> m(D), l(D), acc(D);
    for (int64_t n = n0; n < n1; ++n) {
      for (int64_t d = 0; d < D; ++d) { m[d] = -INFINITY; l[d] = 0.f; acc[d] = 0.f; }
      for (int64_t r = 0; r < R; ++r) {
        const float* srow = sp + (n * R + r) * D;
        const float* arow = ap + r * D;
        const float* krow = kp + (n * R + r) * D;
        #pragma omp simd
        for (int64_t d = 0; d < D; ++d) {
          // F1: a MASKED position (score==-inf, e.g. ratio-4 overlap padding) contributes nothing.
          // corr=0 while the running max is still -inf (nothing accumulated); e=0 for a -inf input.
          // This avoids the indeterminate exp(-inf - -inf)=NaN that poisoned the pool.
          float x = srow[d] + arow[d];
          float mnew = x > m[d] ? x : m[d];
          float corr = (m[d] == -INFINITY) ? 0.f : std::exp(m[d] - mnew);
          float e = (x == -INFINITY) ? 0.f : std::exp(x - mnew);
          l[d] = l[d] * corr + e;
          acc[d] = acc[d] * corr + e * krow[d];
          m[d] = mnew;
        }
      }
      float* orow = op + n * D;
      #pragma omp simd
      for (int64_t d = 0; d < D; ++d) orow[d] = (l[d] > 0.f) ? acc[d] / l[d] : 0.f;  // all-masked -> 0
    }
  });
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compressor_softmax_pool", &compressor_softmax_pool,
        "DSA compressor softmax-pool (streaming online-softmax, no w[N,R,D] temporary)");
}
