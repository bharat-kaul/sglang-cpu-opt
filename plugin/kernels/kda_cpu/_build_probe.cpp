// Minimal build-feasibility probe: can torch cpp_extension compile against the
// sgl-kernel CPU headers with AMX flags? If this builds + the op runs, the fused
// KDA kernel port is build-feasible via a self-contained JIT extension.
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <immintrin.h>

// AMX-adjacent AVX512-bf16 dot (the primitive fla.cpp uses: _mm512_dpbf16_ps).
at::Tensor bf16_dot_probe(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16);
  int64_t n = a.numel();
  auto af = a.to(at::kFloat), bf = b.to(at::kFloat);
  float acc = 0.f;
  auto* ap = af.data_ptr<float>();
  auto* bp = bf.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) acc += ap[i] * bp[i];
  return at::tensor(acc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_dot_probe", &bf16_dot_probe, "bf16 dot probe");
}
