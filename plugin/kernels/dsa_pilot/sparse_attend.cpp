// DSv4 DSA sparse attend (top-k KV) — Phase-A kernel (pilot, authored per PUBLISHED model.py).
// MLA sparse attention: q[N,H,D] attends over K gathered latent KV[N,K,D] (num_key_value_heads=1 =>
// MQA: ONE kv latent shared across H=64 heads; k === v = the latent, D = head_dim = 512) with a
// per-head learned softmax SINK (attn_sink[H]) that absorbs probability but contributes no value.
// Ref: DeepSeek-V4-Flash inference/model.py Attention.sparse_attn(q, kv, attn_sink, topk, scale).
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>

// q:[N,H,D]; kv:[N,K,D] (shared k==v latent); sink:[H] -> out:[N,H,D]. scale default = D**-0.5.
torch::Tensor sparse_attend(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1, "bad dims");
  TORCH_CHECK(q.size(2) == kv.size(2), "q/kv head_dim mismatch");
  TORCH_CHECK(sink.size(0) == q.size(1), "sink must be [H]");
  auto qc = q.to(torch::kFloat32).contiguous();
  auto kc = kv.to(torch::kFloat32).contiguous();
  auto sc = sink.to(torch::kFloat32).contiguous();
  const int64_t N = qc.size(0), H = qc.size(1), D = qc.size(2), K = kc.size(1);
  const float scale = scale_ > 0 ? (float)scale_ : (float)(1.0 / std::sqrt((double)D));
  auto out = torch::empty({N, H, D}, torch::kFloat32);
  const float* qp = qc.data_ptr<float>();
  const float* kp = kc.data_ptr<float>();
  const float* sp = sc.data_ptr<float>();
  float* op = out.data_ptr<float>();

  at::parallel_for(0, N * H, 0, [&](int64_t i0, int64_t i1) {
    std::vector<float> acc(D);
    for (int64_t i = i0; i < i1; ++i) {
      const int64_t n = i / H, h = i % H;
      const float* qrow = qp + (n * H + h) * D;
      const float* kvbase = kp + n * K * D;          // latent shared across heads (MQA)
      float m = sp[h], l = 1.f;                        // sink: logit sp[h], weight exp(0)=1, no value
      for (int64_t d = 0; d < D; ++d) acc[d] = 0.f;
      for (int64_t kk = 0; kk < K; ++kk) {
        const float* kvr = kvbase + kk * D;
        float s = 0.f;
        #pragma omp simd reduction(+ : s)
        for (int64_t d = 0; d < D; ++d) s += qrow[d] * kvr[d];
        s *= scale;
        float mnew = s > m ? s : m;
        float corr = std::exp(m - mnew), e = std::exp(s - mnew);
        l = l * corr + e;
        #pragma omp simd
        for (int64_t d = 0; d < D; ++d) acc[d] = acc[d] * corr + e * kvr[d];
        m = mnew;
      }
      float inv = 1.f / l;
      float* orow = op + (n * H + h) * D;
      #pragma omp simd
      for (int64_t d = 0; d < D; ++d) orow[d] = acc[d] * inv;
    }
  });
  return out;
}

// bf16 AMX batched GEMM: score=bmm(q,kv^T) then softmax(+sink) then out=bmm(w,kv). At real dims
// (H=64 is the GEMM M, D=512, K=512) the two matmuls are AMX-tileable, unlike per-query M=1.
torch::Tensor sparse_attend_amx(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1, "bad dims");
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2), K = kv.size(1);
  const double scale = scale_ > 0 ? scale_ : 1.0 / std::sqrt((double)D);
  auto qb = q.to(torch::kBFloat16).contiguous();
  auto kb = kv.to(torch::kBFloat16).contiguous();
  auto sk = sink.to(torch::kFloat32).view({1, H, 1});
  auto scores = at::bmm(qb, kb.transpose(1, 2)).to(torch::kFloat32) * scale;   // [N,H,K]
  auto m = at::maximum(std::get<0>(scores.max(-1, true)), sk);                 // [N,H,1]
  auto e = (scores - m).exp();                                                 // [N,H,K]
  auto denom = e.sum(-1, true) + (sk - m).exp();                               // + sink term
  auto w = (e / denom).to(torch::kBFloat16);
  return at::bmm(w, kb).to(torch::kFloat32);                                   // [N,H,D]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_attend", &sparse_attend, "DSA MLA sparse attend (MQA flash, per-head sink; scalar online-softmax)");
  m.def("sparse_attend_amx", &sparse_attend_amx, "DSA MLA sparse attend via bf16 AMX bmm (+sink)");
}
