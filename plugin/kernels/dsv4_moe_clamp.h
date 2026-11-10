#pragma once
#include "moe.h"
#include <cmath>

template <typename scalar_t>
inline void dsv4_silu_clamp_mul(
    scalar_t* out, const scalar_t* gate, const scalar_t* up, int64_t size, float limit) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec hi(limit), lo(-limit);
  for (int64_t d = 0; d < size; d += bVec::size()) {
    auto [g0, g1] = load_float_vec2(gate + d);
    auto [u0, u1] = load_float_vec2(up + d);
    g0 = at::vec::minimum(g0, hi);
    g1 = at::vec::minimum(g1, hi);
    u0 = at::vec::minimum(hi, at::vec::maximum(lo, u0));
    u1 = at::vec::minimum(hi, at::vec::maximum(lo, u1));
    convert_from_float_ext<scalar_t>(fast_silu(g0) * u0, fast_silu(g1) * u1).store(out + d);
  }
}
