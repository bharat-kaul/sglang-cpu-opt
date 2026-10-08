// DSv4 DSA lightning-indexer logits — Phase-A fused kernel (pilot, authored FRESH).
// op: logits[n,s] = sum_h relu( q[n,h,:] . kv[n,s,:] ) * weight[n,h]
// Oracle-guided (xpu fp8_paged_mqa_logits_triton + roofline): keep the contraction in the
// library GEMM (bf16 AMX), fuse ONLY the irreducible epilogue (relu * weight * sum over H).
// Scores are computed per-n as [S,H] so they stay L2-resident -> the epilogue reads them
// from cache, removing the [N,H,S] DRAM round-trip that makes a non-fused port BW-bound.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

// q:[N,H,D] kv:[N,S,D] weight:[N,H] -> logits:[N,S] (fp32), semantics == indexer_logits ref.
torch::Tensor indexer_logits_fused(torch::Tensor q, torch::Tensor kv, torch::Tensor weight) {
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && weight.dim() == 2, "bad dims");
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2), S = kv.size(1);
  auto qb = q.to(torch::kBFloat16).contiguous();
  auto kvb = kv.to(torch::kBFloat16).contiguous();
  auto wf = weight.to(torch::kFloat32).contiguous();
  auto logits = torch::empty({N, S}, torch::kFloat32);

  // scores[N,S,H] = bmm(kvb [N,S,D], qb^T [N,D,H])  — one batched bf16 AMX GEMM.
  auto scores = at::bmm(kvb, qb.transpose(1, 2)).contiguous();   // [N,S,H] bf16
  const at::BFloat16* sp = scores.data_ptr<at::BFloat16>();
  const float* wp = wf.data_ptr<float>();
  float* lp = logits.data_ptr<float>();
  // fused epilogue: one pass over N*S, inner H contiguous, bf16 read (no relu/mul/sum temps).
  at::parallel_for(0, N * S, 4096, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const at::BFloat16* row = sp + i * H;
      const float* w = wp + (i / S) * H;
      float acc = 0.f;
      #pragma omp simd reduction(+ : acc)
      for (int64_t h = 0; h < H; ++h) {
        float v = static_cast<float>(row[h]);
        acc += (v > 0.f ? v : 0.f) * w[h];
      }
      lp[i] = acc;
    }
  });
  return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer_logits_fused", &indexer_logits_fused, "DSA indexer logits (fused epilogue)");
}
