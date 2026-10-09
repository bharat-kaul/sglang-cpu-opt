// DSv4 MHC combine/assign (hash-cluster weighted combine) — Phase-A kernel (pilot, authored FRESH).
// op: y[m,h] = sum_k pre[m,k] * x_flat[m, k*H + h]   (weighted sum over hc clusters).
// Reference: intel_cpu_models _cpu_hc_combine == torch.einsum("mk,mkh->mh", pre, x.reshape(m,hc,H)).
// Optimization: one fused pass, vectorized over H, parallel over M — no [m,hc,H] broadcast temporary.
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

// x_flat:[M, hc*H] (fp32); pre:[M, hc] (fp32) -> y:[M, H] (fp32). Semantics == _cpu_hc_combine.
// Tiled accumulate-once: per h-tile, accumulate over hc into an L1 buffer, read x once, write y once
// (the naive per-k += pass re-reads/re-writes the whole y row hc times).
torch::Tensor mhc_combine(torch::Tensor x_flat, torch::Tensor pre, int64_t hc) {
  TORCH_CHECK(x_flat.dim() == 2 && pre.dim() == 2, "x_flat must be [M,hc*H], pre [M,hc]");
  TORCH_CHECK(hc > 0, "hc must be > 0");
  TORCH_CHECK(x_flat.size(0) == pre.size(0) && pre.size(1) == hc, "pre must be [M,hc] matching x_flat rows");
  TORCH_CHECK(x_flat.size(1) % hc == 0, "x_flat width must be divisible by hc");
  TORCH_CHECK(x_flat.device().is_cpu() && pre.device().is_cpu(), "CPU tensors only");
  auto xf = x_flat.to(torch::kFloat32).contiguous();
  auto pf = pre.to(torch::kFloat32).contiguous();
  const int64_t M = xf.size(0), HCH = xf.size(1), H = HCH / hc;
  auto out = torch::empty({M, H}, torch::kFloat32);
  const float* xp = xf.data_ptr<float>();
  const float* pp = pf.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int64_t TILE = 1024;  // 4 KB fp32, L1-resident
  at::parallel_for(0, M, 0, [&](int64_t m0, int64_t m1) {
    std::vector<float> acc(TILE);
    for (int64_t m = m0; m < m1; ++m) {
      const float* xm = xp + m * HCH;
      const float* pm = pp + m * hc;
      float* y = op + m * H;
      for (int64_t h0 = 0; h0 < H; h0 += TILE) {
        int64_t hb = std::min<int64_t>(TILE, H - h0);
        for (int64_t i = 0; i < hb; ++i) acc[i] = 0.f;
        for (int64_t k = 0; k < hc; ++k) {
          const float p = pm[k];
          const float* xk = xm + k * H + h0;
          #pragma omp simd
          for (int64_t i = 0; i < hb; ++i) acc[i] += p * xk[i];
        }
        #pragma omp simd
        for (int64_t i = 0; i < hb; ++i) y[h0 + i] = acc[i];
      }
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_combine", &mhc_combine, "MHC hash-cluster combine (fused weighted sum, no broadcast temp)");
}
