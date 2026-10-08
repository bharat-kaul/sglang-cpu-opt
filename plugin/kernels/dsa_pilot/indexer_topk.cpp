// DSv4 DSA indexer top-k — Phase-A kernel (pilot). Select the top-k KV indices per query.
// Sparse attention only needs the SET of top-k (gather is permutation-invariant), so use an
// O(S) selection (std::nth_element) instead of a sort.
// best-of dispatcher (ALL C/C++, no python/torch fallback):
//   large N  -> row-parallel nth_element (one row per thread; saturates the machine).
//   small N  -> CHUNKED 2-pass: split each row into C chunks, local top-k per (row,chunk) in
//               parallel, then merge the C*k candidates -> exact top-k. This parallelizes WITHIN
//               a row so M=1/8 use all cores instead of 1/8 of them (closes the small-M gap that
//               previously routed to torch.topk — now a single consistent C/C++ path).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <numeric>
#include <vector>

static inline void topk_row_parallel(const float* L, int64_t N, int64_t S, int64_t k, int64_t* O) {
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
}

// Chunked 2-pass: the global top-k of a row is a subset of the union of per-chunk top-k, so exact.
static inline void topk_chunked(const float* L, int64_t N, int64_t S, int64_t k, int64_t* O) {
  const int64_t nthr = at::get_num_threads();
  int64_t chunks = std::max<int64_t>(1, nthr / std::max<int64_t>(N, 1));
  chunks = std::min<int64_t>(chunks, std::max<int64_t>(1, S / std::max<int64_t>(k, 1)));
  if (chunks <= 1) { topk_row_parallel(L, N, S, k, O); return; }
  std::vector<int32_t> cand(N * chunks * k);
  std::vector<int64_t> cnt(N * chunks);
  at::parallel_for(0, N * chunks, 1, [&](int64_t a, int64_t b) {
    for (int64_t t = a; t < b; ++t) {
      int64_t n = t / chunks, c = t % chunks;
      int64_t s0 = c * S / chunks, s1 = (c + 1) * S / chunks, len = s1 - s0;
      int64_t kk = std::min<int64_t>(k, len);
      const float* lp = L + n * S;
      std::vector<int32_t> ord(len);
      std::iota(ord.begin(), ord.end(), (int32_t)s0);
      std::nth_element(ord.begin(), ord.begin() + kk, ord.end(),
                       [lp](int32_t i, int32_t j) { return lp[i] > lp[j]; });
      int32_t* cp = cand.data() + (n * chunks + c) * k;
      for (int64_t i = 0; i < kk; ++i) cp[i] = ord[i];
      cnt[n * chunks + c] = kk;
    }
  });
  at::parallel_for(0, N, 1, [&](int64_t a, int64_t b) {
    std::vector<int32_t> all;
    for (int64_t n = a; n < b; ++n) {
      all.clear();
      all.reserve(chunks * k);
      for (int64_t c = 0; c < chunks; ++c) {
        int32_t* cp = cand.data() + (n * chunks + c) * k;
        all.insert(all.end(), cp, cp + cnt[n * chunks + c]);
      }
      const float* lp = L + n * S;
      int64_t kk = std::min<int64_t>(k, (int64_t)all.size());
      std::nth_element(all.begin(), all.begin() + kk, all.end(),
                       [lp](int32_t i, int32_t j) { return lp[i] > lp[j]; });
      int64_t* op = O + n * k;
      for (int64_t t = 0; t < kk; ++t) op[t] = all[t];
    }
  });
}

// logits:[N,S] fp32 -> indices:[N,k] int64 (UNSORTED top-k; set matches torch.topk).
torch::Tensor indexer_topk(torch::Tensor logits, int64_t k) {
  auto lc = logits.contiguous();
  const int64_t N = lc.size(0), S = lc.size(1);
  k = std::min<int64_t>(k, S);
  auto out = torch::empty({N, k}, torch::kLong);
  const float* L = lc.data_ptr<float>();
  int64_t* O = out.data_ptr<int64_t>();
  if (N < at::get_num_threads()) topk_chunked(L, N, S, k, O);  // small N -> within-row parallelism
  else topk_row_parallel(L, N, S, k, O);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer_topk", &indexer_topk, "DSA indexer top-k (best-of: row-parallel / chunked 2-pass; all C/C++)");
}
