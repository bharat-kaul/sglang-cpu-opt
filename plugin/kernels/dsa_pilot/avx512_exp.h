// Shared hand-inlined AVX-512 exp for the DSA pilot kernels (SKILL Stage-4). Removing the non-inlined
// at::vec/Sleef/libmvec exp call boundary was the whole compressor win (job 385022: 80.7us -> 36.9us).
// exp512_ps: Cephes single-precision, range-reduce + degree-6 FMA poly + 2^n bit-scale (~1 ulp). Clamp so
// -inf -> ~0 (no nan into the poly). On non-AVX512 hosts this header is a no-op (guarded by __AVX512F__).
#pragma once
#if defined(__AVX512F__)
#include <immintrin.h>

static inline __m512 exp512_ps(__m512 x) {
  const __m512 LOG2EF = _mm512_set1_ps(1.44269504088896341f);
  const __m512 C1 = _mm512_set1_ps(0.693359375f), C2 = _mm512_set1_ps(-2.12194440e-4f);
  x = _mm512_max_ps(x, _mm512_set1_ps(-87.3365478f));
  x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626f));
  __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, LOG2EF), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, C1, x);
  r = _mm512_fnmadd_ps(n, C2, r);
  __m512 y = _mm512_set1_ps(1.9875691500E-4f);
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
  __m512 z = _mm512_mul_ps(r, r);
  y = _mm512_fmadd_ps(y, z, _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
  __m512i pw = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(pw));
}
#endif
