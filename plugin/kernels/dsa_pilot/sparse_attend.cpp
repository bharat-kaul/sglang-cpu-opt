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

// pass 2: bf16 AMX batched GEMM (q.k and w.v via at::bmm on AMX) — for larger M where the
// compute-bound GEMMs dominate and the scalar flash only ties torch's fp32 BLAS. Softmax in fp32.
torch::Tensor sparse_attend_amx(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale_) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.dim() == 4, "bad dims");
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2);
  const int64_t K = k.size(2), Dv = v.size(3);
  const double scale = scale_ > 0 ? scale_ : 1.0 / std::sqrt((double)D);
  auto qb = q.to(torch::kBFloat16).reshape({N * H, 1, D});
  auto kb = k.to(torch::kBFloat16).reshape({N * H, K, D});
  auto vb = v.to(torch::kBFloat16).reshape({N * H, K, Dv});
  auto scores = at::bmm(qb, kb.transpose(1, 2)).to(torch::kFloat32) * scale;  // [N*H,1,K]
  auto w = at::softmax(scores, -1).to(torch::kBFloat16);
  auto out = at::bmm(w, vb).to(torch::kFloat32);                               // [N*H,1,Dv]
  return out.reshape({N, H, Dv});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_attend", &sparse_attend,
        "DSA sparse attention over top-k KV (flash-style online softmax, no score materialize)");
  m.def("sparse_attend_amx", &sparse_attend_amx,
        "DSA sparse attention via bf16 AMX bmm (q.k, w.v batched GEMM; for larger M)");
}
