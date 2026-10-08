// DSv4 DSA sparse attend (top-k KV) — Phase-A kernel (pilot, authored FRESH).
// op: for each (n,h): out = softmax(q.k * scale) @ v  over the K gathered (top-k) keys.
// Reference: intel_cpu_models/dsa_sparse_attention_cpu.sparse_attention (torch oracle).
// Optimization: flash-style ONLINE softmax over the K keys — scores never materialized,
// one streaming pass per (n,h), parallel over N*H. (Donor MLA flash is the integration
// alternative; this standalone kernel is the Lane-B reference-authored path.)
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>

// q:[N,H,D]; k:[N,H,K,D]; v:[N,H,K,Dv] (fp32) -> out:[N,H,Dv] (fp32). Semantics == sparse_attention.
torch::Tensor sparse_attend(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale_) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.dim() == 4, "bad dims");
  auto qc = q.to(torch::kFloat32).contiguous();
  auto kc = k.to(torch::kFloat32).contiguous();
  auto vc = v.to(torch::kFloat32).contiguous();
  const int64_t N = qc.size(0), H = qc.size(1), D = qc.size(2);
  const int64_t K = kc.size(2), Dv = vc.size(3);
  const float scale = scale_ > 0 ? (float)scale_ : (float)(1.0 / std::sqrt((double)D));
  auto out = torch::empty({N, H, Dv}, torch::kFloat32);
  const float* qp = qc.data_ptr<float>();
  const float* kp = kc.data_ptr<float>();
  const float* vp = vc.data_ptr<float>();
  float* op = out.data_ptr<float>();

  at::parallel_for(0, N * H, 0, [&](int64_t i0, int64_t i1) {
    std::vector<float> acc(Dv);
    for (int64_t i = i0; i < i1; ++i) {
      const float* qrow = qp + i * D;              // [D]
      const float* kbase = kp + i * K * D;         // [K,D]
      const float* vbase = vp + i * K * Dv;        // [K,Dv]
      float m = -INFINITY, l = 0.f;
      for (int64_t d = 0; d < Dv; ++d) acc[d] = 0.f;
      for (int64_t kk = 0; kk < K; ++kk) {
        const float* krow = kbase + kk * D;
        float s = 0.f;
        #pragma omp simd reduction(+ : s)
        for (int64_t d = 0; d < D; ++d) s += qrow[d] * krow[d];
        s *= scale;
        float mnew = s > m ? s : m;
        float corr = std::exp(m - mnew);
        float e = std::exp(s - mnew);
        l = l * corr + e;
        const float* vrow = vbase + kk * Dv;
        #pragma omp simd
        for (int64_t d = 0; d < Dv; ++d) acc[d] = acc[d] * corr + e * vrow[d];
        m = mnew;
      }
      float inv = 1.f / l;
      float* orow = op + i * Dv;
      #pragma omp simd
      for (int64_t d = 0; d < Dv; ++d) orow[d] = acc[d] * inv;
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_attend", &sparse_attend,
        "DSA sparse attention over top-k KV (flash-style online softmax, no score materialize)");
}
