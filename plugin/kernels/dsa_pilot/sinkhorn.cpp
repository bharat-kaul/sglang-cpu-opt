// DSv4 MHC sinkhorn (hash-cluster split) — Phase-A kernel (pilot, authored FRESH).
// Reference: sglang.kernels.ops.layernorm.mhc._hc_split_sinkhorn_torch (torch oracle).
// Per row (b*s rows): pre=sigmoid(.)+eps, post=2*sigmoid(.), and a hc x hc comb matrix driven
// through 20 Sinkhorn normalization iters. Optimization: fuse ALL iters per row with the tiny
// hc x hc matrix kept register/L1-resident, parallel over rows (no per-iter global tensor pass).
// NOTE: eps placement matches the oracle EXACTLY (initial row-norm adds eps to the result;
// subsequent iters add eps to the denominator) so parity holds.
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>

// mixes:[B,S,(2+hc)*hc]; hc_scale:[3]; hc_base:[(2+hc)*hc] -> (pre[B,S,hc], post[B,S,hc], comb[B,S,hc,hc]).
std::vector<torch::Tensor> mhc_sinkhorn(torch::Tensor mixes, torch::Tensor hc_scale,
                                        torch::Tensor hc_base, int64_t hc,
                                        int64_t iters, double eps) {
  auto mf = mixes.to(torch::kFloat32).contiguous();
  auto sf = hc_scale.to(torch::kFloat32).contiguous();
  auto bf = hc_base.to(torch::kFloat32).contiguous();
  const int64_t B = mf.size(0), S = mf.size(1), W = mf.size(2);  // W = (2+hc)*hc
  const int64_t rows = B * S;
  TORCH_CHECK(W == (2 + hc) * hc, "mixes width != (2+hc)*hc");
  auto pre = torch::empty({B, S, hc}, torch::kFloat32);
  auto post = torch::empty({B, S, hc}, torch::kFloat32);
  auto comb = torch::empty({B, S, hc, hc}, torch::kFloat32);
  const float* mp = mf.data_ptr<float>();
  const float* sc = sf.data_ptr<float>();
  const float* ba = bf.data_ptr<float>();
  float* prep = pre.data_ptr<float>();
  float* postp = post.data_ptr<float>();
  float* combp = comb.data_ptr<float>();
  const float e = (float)eps;

  at::parallel_for(0, rows, 0, [&](int64_t r0, int64_t r1) {
    std::vector<float> C(hc * hc), rs(hc), cs(hc);
    for (int64_t r = r0; r < r1; ++r) {
      const float* row = mp + r * W;
      for (int64_t h = 0; h < hc; ++h) {
        prep[r * hc + h] = 1.f / (1.f + std::exp(-(row[h] * sc[0] + ba[h]))) + e;
        postp[r * hc + h] = 2.f / (1.f + std::exp(-(row[hc + h] * sc[1] + ba[hc + h])));
      }
      // comb = flat[2hc:]*scale2 + base2hc, reshaped [hc,hc]
      for (int64_t i = 0; i < hc; ++i)
        for (int64_t j = 0; j < hc; ++j) {
          int64_t idx = 2 * hc + i * hc + j;
          C[i * hc + j] = row[idx] * sc[2] + ba[idx];  // comb = flat[2hc:]*scale2 + base[2hc:]
        }
      // initial: row softmax over j (stabilized), then (val/sum_j)+eps, then /(sum_i+eps)
      for (int64_t i = 0; i < hc; ++i) {
        float mx = -INFINITY;
        for (int64_t j = 0; j < hc; ++j) mx = std::max(mx, C[i * hc + j]);
        float s = 0.f;
        for (int64_t j = 0; j < hc; ++j) { C[i * hc + j] = std::exp(C[i * hc + j] - mx); s += C[i * hc + j]; }
        for (int64_t j = 0; j < hc; ++j) C[i * hc + j] = C[i * hc + j] / s + e;
      }
      for (int64_t j = 0; j < hc; ++j) { cs[j] = 0.f; for (int64_t i = 0; i < hc; ++i) cs[j] += C[i * hc + j]; }
      for (int64_t i = 0; i < hc; ++i)
        for (int64_t j = 0; j < hc; ++j) C[i * hc + j] = C[i * hc + j] / (cs[j] + e);
      // iters-1 Sinkhorn steps: normalize over j then over i (eps in denominator)
      for (int64_t it = 0; it < iters - 1; ++it) {
        for (int64_t i = 0; i < hc; ++i) { rs[i] = 0.f; for (int64_t j = 0; j < hc; ++j) rs[i] += C[i * hc + j]; }
        for (int64_t i = 0; i < hc; ++i)
          for (int64_t j = 0; j < hc; ++j) C[i * hc + j] = C[i * hc + j] / (rs[i] + e);
        for (int64_t j = 0; j < hc; ++j) { cs[j] = 0.f; for (int64_t i = 0; i < hc; ++i) cs[j] += C[i * hc + j]; }
        for (int64_t i = 0; i < hc; ++i)
          for (int64_t j = 0; j < hc; ++j) C[i * hc + j] = C[i * hc + j] / (cs[j] + e);
      }
      float* co = combp + r * hc * hc;
      for (int64_t k = 0; k < hc * hc; ++k) co[k] = C[k];
    }
  });
  return {pre, post, comb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_sinkhorn", &mhc_sinkhorn,
        "MHC hash-split sinkhorn (fused per-row iters, hc x hc resident)");
}
