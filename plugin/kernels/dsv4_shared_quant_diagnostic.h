#pragma once
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <vector>
#include "dsv4_shared_clamp.h"

inline bool& dsv4_shared_quantize_ref() {
  static bool enabled = false;
  return enabled;
}

template <typename scalar_t>
void dsv4_shared_quantize_activation(
    scalar_t* out, const scalar_t* input, int64_t rows, int64_t cols,
    at::Float8_e4m3fn* quantized = nullptr, float* scales = nullptr) {
  TORCH_CHECK(cols % 128 == 0, "shared quant diagnostic requires groups of 128");
  at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      for (int64_t group = 0; group < cols / 128; ++group) {
        const int64_t offset = row * cols + group * 128;
        float amax = 1e-10f;
        for (int64_t i = 0; i < 128; ++i) {
          const float value = float(input[offset + i]);
          TORCH_CHECK(std::isfinite(value), "nonfinite shared quant diagnostic input");
          amax = std::max(amax, std::abs(value));
        }
        const float scale = amax * (1.0f / 448.0f);
        // CPU reciprocal rounding is not CUDA's approximate reciprocal. Native
        // GPU operand comparisons measure the resulting FP8 tie differences.
        const float multiplier = (1.0f / amax) * 448.0f;
        if (scales) scales[row * (cols / 128) + group] = scale;
        for (int64_t i = 0; i < 128; ++i) {
          const float scaled = std::min(448.0f, std::max(-448.0f, float(input[offset + i]) * multiplier));
          const at::Float8_e4m3fn q(scaled);
          if (quantized) quantized[offset + i] = q;
          out[offset + i] = scalar_t(float(q) * scale);
        }
      }
    }
  });
}

inline std::vector<at::Tensor> dsv4_shared_quantize_probe(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kBFloat16 &&
              input.dim() == 2 && input.is_contiguous(), "expected contiguous CPU BF16 matrix");
  auto out = at::empty_like(input);
  auto q = at::empty_like(input, input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({input.size(0), input.size(1) / 128}, input.options().dtype(at::kFloat));
  dsv4_shared_quantize_activation(
      out.data_ptr<at::BFloat16>(), input.data_ptr<at::BFloat16>(), input.size(0), input.size(1),
      q.data_ptr<at::Float8_e4m3fn>(), scales.data_ptr<float>());
  return {out, q, scales};
}
