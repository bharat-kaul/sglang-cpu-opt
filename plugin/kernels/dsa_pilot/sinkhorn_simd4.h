#pragma once

#if !defined(__AVX512F__)
#error "SINK_SIMD_EXPERIMENT requires AVX-512"
#endif
#include <immintrin.h>

static inline __m512 sinkhorn_step4(__m512 c, __m512 e) {
  const __m512 zero = _mm512_setzero_ps();
  // Broadcast in the original reduction order; do not replace division by reciprocal.
  __m512 sum = _mm512_add_ps(zero, _mm512_permute_ps(c, 0x00));
  sum = _mm512_add_ps(sum, _mm512_permute_ps(c, 0x55));
  sum = _mm512_add_ps(sum, _mm512_permute_ps(c, 0xaa));
  sum = _mm512_add_ps(sum, _mm512_permute_ps(c, 0xff));
  c = _mm512_div_ps(c, _mm512_add_ps(sum, e));
  sum = _mm512_add_ps(zero, _mm512_shuffle_f32x4(c, c, 0x00));
  sum = _mm512_add_ps(sum, _mm512_shuffle_f32x4(c, c, 0x55));
  sum = _mm512_add_ps(sum, _mm512_shuffle_f32x4(c, c, 0xaa));
  sum = _mm512_add_ps(sum, _mm512_shuffle_f32x4(c, c, 0xff));
  return _mm512_div_ps(c, _mm512_add_ps(sum, e));
}

template <int Streams>
static void sinkhorn_simd_blocks(float* matrices, int64_t begin, int64_t end,
                                 int64_t iters, float eps) {
  const __m512 e = _mm512_set1_ps(eps);
  int64_t r = begin;
  for (; r <= end - Streams; r += Streams) {
    __m512 c[Streams];
#pragma GCC unroll 8
    for (int s = 0; s < Streams; ++s) c[s] = _mm512_loadu_ps(matrices + (r + s) * 16);
    for (int64_t it = 0; it < iters - 1; ++it) {
#pragma GCC unroll 8
      for (int s = 0; s < Streams; ++s) c[s] = sinkhorn_step4(c[s], e);
    }
#pragma GCC unroll 8
    for (int s = 0; s < Streams; ++s) _mm512_storeu_ps(matrices + (r + s) * 16, c[s]);
  }
  if constexpr (Streams > 1) sinkhorn_simd_blocks<1>(matrices, r, end, iters, eps);
}

static std::vector<torch::Tensor> mhc_sinkhorn_simd4(
    torch::Tensor mixes, torch::Tensor hc_scale, torch::Tensor hc_base,
    int64_t hc, int64_t iters, double eps, int64_t streams) {
  TORCH_CHECK(streams == 1 || streams == 4 || streams == 8, "streams must be 1, 4 or 8");
  if (hc != 4 || iters <= 1)
    return mhc_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps);
  // Reuse donor initialization verbatim; the final comb output doubles as the L1 staging buffer.
  auto out = mhc_sinkhorn(mixes, hc_scale, hc_base, hc, 1, eps);
  const int64_t rows = out[2].numel() / 16;
  float* matrices = out[2].data_ptr<float>();
  int64_t grain = 512;
  if (std::getenv("SINK_PARALLEL")) grain = 1;
  if (std::getenv("SINK_SERIAL")) grain = rows;
  if (const char* g = std::getenv("SINK_GRAIN")) grain = std::max<int64_t>(1, atol(g));
  at::parallel_for(0, rows, grain, [&](int64_t begin, int64_t end) {
    if (streams == 1) sinkhorn_simd_blocks<1>(matrices, begin, end, iters, (float)eps);
    else if (streams == 4) sinkhorn_simd_blocks<4>(matrices, begin, end, iters, (float)eps);
    else sinkhorn_simd_blocks<8>(matrices, begin, end, iters, (float)eps);
  });
  return out;
}
