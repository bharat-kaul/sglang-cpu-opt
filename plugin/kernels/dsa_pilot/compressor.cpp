// DSv4 DSA compressor (softmax-pool) — Phase-A kernel (pilot, authored FRESH).
// op: out[n,d] = sum_r softmax_r(score[n,r,d] + ape[r,d]) * kv[n,r,d]   (per-channel pool over window R).
// Reference: intel_cpu_models/dsa_compressor_cpu.compress_softmax_pool (torch oracle).
// Optimization: ONE streaming pass with online softmax, vectorized over the channel dim D and
// parallel over N rows — the weight tensor w[N,R,D] is never materialized (no DRAM temporary).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
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
  if (N == 0 || D == 0) return out;                          // empty batch/channels -> [N,D] (avoid /N below)
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
          // C1: ONE exp per update. The surviving factor is exp(min-max) (<=0 arg); the other is exactly
          // exp(0)=1, so it is hardcoded -> bit-identical to the two-exp form. The masked/first-valid/both
          // -inf lanes force the surviving arg to -inf (exp->0) so no unguarded inf subtraction reaches exp.
          float x = srow[j] + arow[j];
          float mj = m[j];
          bool up = x > mj;
          float mnew = up ? x : mj;
          bool surv_masked = up ? (mj == -INFINITY) : (x == -INFINITY);
          float arg = surv_masked ? -INFINITY : (up ? (mj - x) : (x - mj));
          float t = std::exp(arg);
          float corr = up ? t : 1.f;
          float e = up ? 1.f : t;
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


// C2 (review b51d1eb): VECTORIZED one-exp online-softmax. Same algorithm/state/loop order as the shipped
// kernel, but the per-channel update runs over at::vec lanes with a VECTOR exp. NOT bit-exact to the scalar
// path (vector exp != scalar libm), so it is SCREENED by F4, not asserted equal. Measured vs the C1 baseline.
torch::Tensor compressor_softmax_pool_vexp(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
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
  if (N == 0 || D == 0) return out;
  const float* kp = kvc.data_ptr<float>();
  const float* sp = scc.data_ptr<float>();
  const float* ap = apc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int64_t nthr = at::get_num_threads();
  int64_t dpt = std::max<int64_t>(1, (2 * nthr + N - 1) / N);
  int64_t DT = std::max<int64_t>(16, (D + dpt - 1) / dpt);
  DT = (DT + 15) / 16 * 16;
  const int64_t ndt = (D + DT - 1) / DT;
  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t VW = Vec::size();
  const Vec vninf(-INFINITY), vone(1.f);

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
        int64_t j = 0;
        for (; j + VW <= dd; j += VW) {
          Vec x = Vec::loadu(srow + j) + Vec::loadu(arow + j);
          Vec mj = Vec::loadu(m.data() + j);
          Vec up = (x > mj);                                      // lane mask
          Vec mnew = at::vec::maximum(x, mj);
          Vec survm = Vec::blendv(x == vninf, mj == vninf, up);   // up? (mj==-inf) : (x==-inf)
          Vec arg = Vec::blendv(x - mj, mj - x, up);              // up? (mj-x) : (x-mj)
          arg = Vec::blendv(arg, vninf, survm);                   // masked/first-valid -> exp(-inf)=0
          Vec tv = arg.exp();
          Vec corr = Vec::blendv(vone, tv, up);                   // up? t : 1
          Vec e = Vec::blendv(tv, vone, up);                      // up? 1 : t
          Vec lv = Vec::loadu(l.data() + j) * corr + e;
          Vec av = Vec::loadu(acc.data() + j) * corr + e * Vec::loadu(krow + j);
          lv.store(l.data() + j); av.store(acc.data() + j); mnew.store(m.data() + j);
        }
        for (; j < dd; ++j) {                                     // scalar tail (same C1 math)
          float x = srow[j] + arow[j], mj = m[j];
          bool up = x > mj; float mnew = up ? x : mj;
          bool survm = up ? (mj == -INFINITY) : (x == -INFINITY);
          float arg = survm ? -INFINITY : (up ? (mj - x) : (x - mj));
          float tt = std::exp(arg);
          float corr = up ? tt : 1.f, e = up ? 1.f : tt;
          l[j] = l[j] * corr + e; acc[j] = acc[j] * corr + e * krow[j]; m[j] = mnew;
        }
      }
      float* orow = op + n * D + d0;
      for (int64_t j = 0; j < dd; ++j) orow[j] = (l[j] > 0.f) ? acc[j] / l[j] : 0.f;
    }
  });
  return out;
}


// C3 (review b51d1eb): STABLE TWO-PASS pooling. Pass 1 = per-channel max over R; pass 2 = one exp per element
// (x - max), accumulate denom + weighted numerator, divide once. No running-max rescale of the accumulator,
// but an extra score/APE traversal. NOT bit-exact (different reduction) -> F4-screened. Measured vs C1.
torch::Tensor compressor_softmax_pool_multipass(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
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
  if (N == 0 || D == 0) return out;
  const float* kp = kvc.data_ptr<float>();
  const float* sp = scc.data_ptr<float>();
  const float* ap = apc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int64_t nthr = at::get_num_threads();
  int64_t dpt = std::max<int64_t>(1, (2 * nthr + N - 1) / N);
  int64_t DT = std::max<int64_t>(16, (D + dpt - 1) / dpt);
  DT = (DT + 15) / 16 * 16;
  const int64_t ndt = (D + DT - 1) / DT;

  at::parallel_for(0, N * ndt, 0, [&](int64_t t0, int64_t t1) {
    std::vector<float> m(DT), l(DT), acc(DT);
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t n = t / ndt, dt = t % ndt, d0 = dt * DT;
      const int64_t dd = std::min(DT, D - d0);
      for (int64_t j = 0; j < dd; ++j) { m[j] = -INFINITY; l[j] = 0.f; acc[j] = 0.f; }
      for (int64_t r = 0; r < R; ++r) {                           // pass 1: per-channel max over R
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        #pragma omp simd
        for (int64_t j = 0; j < dd; ++j) {
          float x = srow[j] + arow[j];
          m[j] = x > m[j] ? x : m[j];
        }
      }
      for (int64_t r = 0; r < R; ++r) {                           // pass 2: one exp/elem, accumulate
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        const float* krow = kp + (n * R + r) * D + d0;
        #pragma omp simd
        for (int64_t j = 0; j < dd; ++j) {
          float x = srow[j] + arow[j];
          float e = (x == -INFINITY) ? 0.f : std::exp(x - m[j]);   // m[j]==-inf (all masked) -> x==-inf -> 0
          l[j] += e;
          acc[j] += e * krow[j];
        }
      }
      float* orow = op + n * D + d0;
      #pragma omp simd
      for (int64_t j = 0; j < dd; ++j) orow[j] = (l[j] > 0.f) ? acc[j] / l[j] : 0.f;
    }
  });
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compressor_softmax_pool", &compressor_softmax_pool,
        "DSA compressor softmax-pool (streaming online-softmax, no w[N,R,D] temporary)");
  m.def("compressor_softmax_pool_vexp", &compressor_softmax_pool_vexp,
        "C2 variant: vectorized one-exp online-softmax (ATen vec exp; NOT bit-exact to scalar libm)");
  m.def("compressor_softmax_pool_multipass", &compressor_softmax_pool_multipass,
        "C3 variant: stable two-pass pooling (max over R, one exp/elem, divide once)");
}
