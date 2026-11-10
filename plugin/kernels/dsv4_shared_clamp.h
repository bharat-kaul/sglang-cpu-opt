#pragma once
#include "dsv4_moe_clamp.h"
// Runtime clamp toggle for the PRIVATE shared-expert diagnostic donor (reviewer step 1-2).
// value < 0  => OFF: the shared kernel uses the native silu_and_mul_stub (must reproduce production
//               bit-exact — the reviewer's "inactive clamp reproduces original" requirement).
// value > 0  => clamp the shared SwiGLU at this limit (dsv4_silu_clamp_mul), matching the GPU semantics.
// `inline` gives a SINGLE shared instance across translation units, so the pybind setter (compiled into
// moe.cpp) and the activation read (moe_fp8.cpp) refer to the same variable.
inline float& dsv4_shared_clamp_limit_ref() {
  static float v = -1.0f;
  return v;
}
