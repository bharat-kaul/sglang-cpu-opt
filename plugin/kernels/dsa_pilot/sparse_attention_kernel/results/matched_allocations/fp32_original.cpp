#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstdlib>
#include "avx512_exp.h"
torch::Tensor sparse_attend_fp32bmm_hx(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && sink.device().is_cpu(), "CPU tensors only");
  TORCH_CHECK(std::isfinite(scale_), "scale must be finite (pass a finite scale<=0 for the default D**-0.5)");
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1, "bad dims");
  TORCH_CHECK(q.size(0) == kv.size(0), "q/kv batch N mismatch");
  TORCH_CHECK(q.size(2) == kv.size(2), "q/kv head_dim mismatch");
  TORCH_CHECK(sink.size(0) == q.size(1), "sink must be [H]");
  auto qc = q.to(torch::kFloat32).contiguous();
  auto kc = kv.to(torch::kFloat32).contiguous();
  auto sc = sink.to(torch::kFloat32).contiguous();
  const int64_t N = qc.size(0), H = qc.size(1), D = qc.size(2), K = kc.size(1);
  const float scale = scale_ > 0 ? (float)scale_ : (float)(1.0 / std::sqrt((double)D));
  auto scores = at::bmm(qc, kc.transpose(1, 2));                               // [N,H,K] fp32 (donor GEMM)
  const bool _skip_sm = std::getenv("SPARSE_SKIP_SOFTMAX") != nullptr;    // diagnostic: isolate softmax cost
  const bool _skip_vg = std::getenv("SPARSE_SKIP_VALUE_GEMM") != nullptr; // diagnostic: isolate value-GEMM cost
  const float* sp = sc.data_ptr<float>();
  float* sdp = scores.data_ptr<float>();
  if (!_skip_sm) at::parallel_for(0, N * H, 0, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const int64_t h = i % H;
      float* row = sdp + i * K;
      float m = sp[h];
      for (int64_t kk = 0; kk < K; ++kk) { float s = row[kk] * scale; row[kk] = s; if (s > m) m = s; }
      float l = std::exp(sp[h] - m);                        // sink weight (scalar, one per row)
      int64_t kk = 0;
#if defined(__AVX512F__)
      const __m512 vm = _mm512_set1_ps(m);
      __m512 vacc = _mm512_setzero_ps();
      for (; kk + 16 <= K; kk += 16) {
        __m512 e = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(row + kk), vm));    // inlined Cephes exp
        _mm512_storeu_ps(row + kk, e);
        vacc = _mm512_add_ps(vacc, e);
      }
      l += _mm512_reduce_add_ps(vacc);
#endif
      for (; kk < K; ++kk) { float e = std::exp(row[kk] - m); row[kk] = e; l += e; }   // scalar tail
      float inv = 1.f / l;
      #pragma omp simd
      for (int64_t kk2 = 0; kk2 < K; ++kk2) row[kk2] *= inv;
    }
  });
  if (_skip_vg) return torch::empty({N, H, D}, scores.options());             // diagnostic: skip value GEMM
  return at::bmm(scores, kc);                                                  // [N,H,D] fp32 (donor GEMM)
}
