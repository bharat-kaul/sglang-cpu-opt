// DSv4 MHC combine/assign (hash-cluster weighted combine) — Phase-A kernel (pilot, authored FRESH).
// op: y[m,h] = sum_k pre[m,k] * x_flat[m, k*H + h]   (weighted sum over hc clusters).
// Reference: intel_cpu_models _cpu_hc_combine == torch.einsum("mk,mkh->mh", pre, x.reshape(m,hc,H)).
// Optimization: one fused pass, vectorized over H, parallel over M — no [m,hc,H] broadcast temporary.
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

// x_flat:[M, hc*H] (fp32); pre:[M, hc] (fp32) -> y:[M, H] (fp32). Semantics == _cpu_hc_combine.
torch::Tensor mhc_combine(torch::Tensor x_flat, torch::Tensor pre, int64_t hc) {
  TORCH_CHECK(x_flat.dim() == 2 && pre.dim() == 2, "bad dims");
  auto xf = x_flat.to(torch::kFloat32).contiguous();
  auto pf = pre.to(torch::kFloat32).contiguous();
  const int64_t M = xf.size(0), HCH = xf.size(1), H = HCH / hc;
  auto out = torch::zeros({M, H}, torch::kFloat32);
  const float* xp = xf.data_ptr<float>();
  const float* pp = pf.data_ptr<float>();
  float* op = out.data_ptr<float>();
  at::parallel_for(0, M, 0, [&](int64_t m0, int64_t m1) {
    for (int64_t m = m0; m < m1; ++m) {
      float* y = op + m * H;
      for (int64_t k = 0; k < hc; ++k) {
        const float p = pp[m * hc + k];
        const float* xk = xp + m * HCH + k * H;
        #pragma omp simd
        for (int64_t h = 0; h < H; ++h) y[h] += p * xk[h];
      }
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_combine", &mhc_combine, "MHC hash-cluster combine (fused weighted sum, no broadcast temp)");
}
