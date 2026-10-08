// DSv4 DSA indexer top-k — Phase-A kernel (pilot). Select the top-k KV indices per query.
// Sparse attention only needs the SET of top-k (gather is permutation-invariant), so use an
// O(S) selection (std::nth_element) instead of a sort. Parallel over N rows.
// INTEGRATION: best-of = this custom path for M=1 and priority M=16/32/64; torch.topk FALLBACK at
// M=8 (custom 0.42x there — 8 row-tasks underutilize 64 threads; at the limit, no GEMM primitive).
// Logged in plugin/validate/results/torch_fallbacks.json (surfaced for review at phase end).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <numeric>
#include <vector>

// logits:[N,S] fp32 -> indices:[N,k] int64 (UNSORTED top-k; set matches torch.topk).
torch::Tensor indexer_topk(torch::Tensor logits, int64_t k) {
  auto lc = logits.contiguous();
  const int64_t N = lc.size(0), S = lc.size(1);
  k = std::min<int64_t>(k, S);
  auto out = torch::empty({N, k}, torch::kLong);
  const float* L = lc.data_ptr<float>();
  int64_t* O = out.data_ptr<int64_t>();
  at::parallel_for(0, N, 1, [&](int64_t a, int64_t b) {
    std::vector<int32_t> ord(S);
    for (int64_t n = a; n < b; ++n) {
      const float* lp = L + n * S;
      std::iota(ord.begin(), ord.end(), 0);
      std::nth_element(ord.begin(), ord.begin() + k, ord.end(),
                       [lp](int32_t i, int32_t j) { return lp[i] > lp[j]; });
      int64_t* op = O + n * k;
      for (int64_t t = 0; t < k; ++t) op[t] = ord[t];
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer_topk", &indexer_topk, "DSA indexer top-k (parallel nth_element, unsorted set)");
}
