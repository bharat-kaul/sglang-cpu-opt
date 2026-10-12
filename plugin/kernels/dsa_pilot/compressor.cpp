// DSv4 DSA compressor (softmax-pool) — Phase-A kernel (pilot, authored FRESH).
// op: out[n,d] = sum_r softmax_r(score[n,r,d] + ape[r,d]) * kv[n,r,d]   (per-channel pool over window R).
// Reference: intel_cpu_models/dsa_compressor_cpu.compress_softmax_pool (torch oracle).
// Optimization: ONE streaming pass with online softmax, vectorized over the channel dim D and
// parallel over N rows — the weight tensor w[N,R,D] is never materialized (no DRAM temporary).
//
// VECTORIZATION FINDING (RAJAPerf hpc-kernel-opt SKILL retry, EMR jobs 385013/385022): the shipped scalar
// C1 path tops out at ~27% of the nominal BW roof NOT because it is bandwidth-bound but because its per-lane
// std::exp stays SCALAR. The op is actually EXP-COMPUTE bound and (at decode M) CACHE-RESIDENT, so the DRAM
// roofline is the wrong ceiling. THROUGHPUT PATH = compressor_softmax_pool_fp32intrin (D1): a hand-inlined
// AVX-512 Cephes exp (exp512_ps, FMA, ~1ulp), FP32 throughout. M32 36.9us = 4.8x over C1, 2.2x over the
// at::vec Sleef path, accuracy IDENTICAL to scalar (rel-err 2.8e-7); needs NO special build flags.
// GROUNDED: model.py Compressor.forward runs the pool in FP32 ("compression need fp32"); the corpus bf16->
// fp32-exp->bf16 pattern is for sparse_attn/indexer, NOT the compressor. A bf16 pool (D3) was MEASURED a
// NON-WIN: same speed as D1 (exp must be fp32 -> no throughput gain) but rel-err 6.4e-3. C1 stays the
// bit-exact oracle; D1 is the perf path (F4-screened, equally faithful to the fp32 reference).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <vector>

// kv,score:[N,R,D] (fp32); ape:[R,D] (fp32) -> out:[N,D] (fp32). Semantics == compress_softmax_pool.
torch::Tensor compressor_softmax_pool(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  TORCH_CHECK(kv.dim() == 3 && score.dim() == 3 && ape.dim() == 2, "kv/score must be [N,R,D], ape [R,D]");
  TORCH_CHECK(score.size(0) == kv.size(0) && score.size(1) == kv.size(1) && score.size(2) == kv.size(2),
              "score shape must equal kv shape");
  TORCH_CHECK(ape.size(0) == kv.size(1) && ape.size(1) == kv.size(2), "ape must be [R,D] matching kv");
  TORCH_CHECK(kv.device().is_cpu() && score.device().is_cpu() && ape.device().is_cpu(), "CPU tensors only");
  auto kvc = kv.to(torch::kFloat32).contiguous();
  auto scc = score.to(torch::kFloat32).contiguous();
  auto apc = ape.to(torch::kFloat32).contiguous();
  const int64_t N = kvc.size(0), R = kvc.size(1), D = kvc.size(2);
  auto out = torch::empty({N, D}, torch::kFloat32);
  if (N == 0 || D == 0) return out;                          // empty batch/channels -> [N,D] (avoid /N below)
  const float* kp = kvc.data_ptr<float>();
  const float* sp = scc.data_ptr<float>();
  const float* ap = apc.data_ptr<float>();
  float* op = out.data_ptr<float>();

  // Parallelize over N * D-tiles (the softmax is independent per channel) so all cores are used even at
  // small N (N-only parallelism left cores idle and REGRESSED vs torch at M=1). DT >= 16 (one AVX-512 vec).
  const int64_t nthr = at::get_num_threads();
  int64_t dpt = std::max<int64_t>(1, (2 * nthr + N - 1) / N);        // d-tiles per n (target ~2*threads tasks)
  int64_t DT = std::max<int64_t>(16, (D + dpt - 1) / dpt);
  DT = (DT + 15) / 16 * 16;                                          // round to an AVX-512 width
  const int64_t ndt = (D + DT - 1) / DT;

  at::parallel_for(0, N * ndt, 0, [&](int64_t t0, int64_t t1) {
    std::vector<float> m(DT), l(DT), acc(DT);
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t n = t / ndt, dt = t % ndt, d0 = dt * DT;
      const int64_t dd = std::min(DT, D - d0);
      for (int64_t j = 0; j < dd; ++j) { m[j] = -INFINITY; l[j] = 0.f; acc[j] = 0.f; }
      for (int64_t r = 0; r < R; ++r) {
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        const float* krow = kp + (n * R + r) * D + d0;
        #pragma omp simd
        for (int64_t j = 0; j < dd; ++j) {
          // F1: a MASKED position (score==-inf) contributes nothing (corr=0 while running max is -inf; e=0).
          // C1: ONE exp per update. The surviving factor is exp(min-max) (<=0 arg); the other is exactly
          // exp(0)=1, so it is hardcoded -> bit-identical to the two-exp form. The masked/first-valid/both
          // -inf lanes force the surviving arg to -inf (exp->0) so no unguarded inf subtraction reaches exp.
          float x = srow[j] + arow[j];
          float mj = m[j];
          bool up = x > mj;
          float mnew = up ? x : mj;
          bool surv_masked = up ? (mj == -INFINITY) : (x == -INFINITY);
          float arg = surv_masked ? -INFINITY : (up ? (mj - x) : (x - mj));
          float t = std::exp(arg);
          float corr = up ? t : 1.f;
          float e = up ? 1.f : t;
          l[j] = l[j] * corr + e;
          acc[j] = acc[j] * corr + e * krow[j];
          m[j] = mnew;
        }
      }
      float* orow = op + n * D + d0;
      #pragma omp simd
      for (int64_t j = 0; j < dd; ++j) orow[j] = (l[j] > 0.f) ? acc[j] / l[j] : 0.f;  // all-masked -> 0
    }
  });
  return out;
}


// C2 (review b51d1eb): VECTORIZED one-exp online-softmax. Same algorithm/state/loop order as the shipped
// kernel, but the per-channel update runs over at::vec lanes with a VECTOR exp. NOT bit-exact to the scalar
// path (vector exp != scalar libm), so it is SCREENED by F4, not asserted equal. Measured vs the C1 baseline.
torch::Tensor compressor_softmax_pool_vexp(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  TORCH_CHECK(kv.dim() == 3 && score.dim() == 3 && ape.dim() == 2, "kv/score must be [N,R,D], ape [R,D]");
  TORCH_CHECK(score.size(0) == kv.size(0) && score.size(1) == kv.size(1) && score.size(2) == kv.size(2),
              "score shape must equal kv shape");
  TORCH_CHECK(ape.size(0) == kv.size(1) && ape.size(1) == kv.size(2), "ape must be [R,D] matching kv");
  TORCH_CHECK(kv.device().is_cpu() && score.device().is_cpu() && ape.device().is_cpu(), "CPU tensors only");
  auto kvc = kv.to(torch::kFloat32).contiguous();
  auto scc = score.to(torch::kFloat32).contiguous();
  auto apc = ape.to(torch::kFloat32).contiguous();
  const int64_t N = kvc.size(0), R = kvc.size(1), D = kvc.size(2);
  auto out = torch::empty({N, D}, torch::kFloat32);
  if (N == 0 || D == 0) return out;
  const float* kp = kvc.data_ptr<float>();
  const float* sp = scc.data_ptr<float>();
  const float* ap = apc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int64_t nthr = at::get_num_threads();
  int64_t dpt = std::max<int64_t>(1, (2 * nthr + N - 1) / N);
  int64_t DT = std::max<int64_t>(16, (D + dpt - 1) / dpt);
  DT = (DT + 15) / 16 * 16;
  const int64_t ndt = (D + DT - 1) / DT;
  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t VW = Vec::size();
  const Vec vninf(-INFINITY), vone(1.f);

  at::parallel_for(0, N * ndt, 0, [&](int64_t t0, int64_t t1) {
    std::vector<float> m(DT), l(DT), acc(DT);
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t n = t / ndt, dt = t % ndt, d0 = dt * DT;
      const int64_t dd = std::min(DT, D - d0);
      for (int64_t j = 0; j < dd; ++j) { m[j] = -INFINITY; l[j] = 0.f; acc[j] = 0.f; }
      for (int64_t r = 0; r < R; ++r) {
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        const float* krow = kp + (n * R + r) * D + d0;
        int64_t j = 0;
        for (; j + VW <= dd; j += VW) {
          Vec x = Vec::loadu(srow + j) + Vec::loadu(arow + j);
          Vec mj = Vec::loadu(m.data() + j);
          Vec up = (x > mj);                                      // lane mask
          Vec mnew = at::vec::maximum(x, mj);
          Vec survm = Vec::blendv(x == vninf, mj == vninf, up);   // up? (mj==-inf) : (x==-inf)
          Vec arg = Vec::blendv(x - mj, mj - x, up);              // up? (mj-x) : (x-mj)
          arg = Vec::blendv(arg, vninf, survm);                   // masked/first-valid -> exp(-inf)=0
          Vec tv = arg.exp();
          Vec corr = Vec::blendv(vone, tv, up);                   // up? t : 1
          Vec e = Vec::blendv(tv, vone, up);                      // up? 1 : t
          Vec lv = Vec::loadu(l.data() + j) * corr + e;
          Vec av = Vec::loadu(acc.data() + j) * corr + e * Vec::loadu(krow + j);
          lv.store(l.data() + j); av.store(acc.data() + j); mnew.store(m.data() + j);
        }
        for (; j < dd; ++j) {                                     // scalar tail (same C1 math)
          float x = srow[j] + arow[j], mj = m[j];
          bool up = x > mj; float mnew = up ? x : mj;
          bool survm = up ? (mj == -INFINITY) : (x == -INFINITY);
          float arg = survm ? -INFINITY : (up ? (mj - x) : (x - mj));
          float tt = std::exp(arg);
          float corr = up ? tt : 1.f, e = up ? 1.f : tt;
          l[j] = l[j] * corr + e; acc[j] = acc[j] * corr + e * krow[j]; m[j] = mnew;
        }
      }
      float* orow = op + n * D + d0;
      for (int64_t j = 0; j < dd; ++j) orow[j] = (l[j] > 0.f) ? acc[j] / l[j] : 0.f;
    }
  });
  return out;
}


// C3 (review b51d1eb): STABLE TWO-PASS pooling. Pass 1 = per-channel max over R; pass 2 = one exp per element
// (x - max), accumulate denom + weighted numerator, divide once. No running-max rescale of the accumulator,
// but an extra score/APE traversal. NOT bit-exact (different reduction) -> F4-screened. Measured vs C1.
torch::Tensor compressor_softmax_pool_multipass(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  TORCH_CHECK(kv.dim() == 3 && score.dim() == 3 && ape.dim() == 2, "kv/score must be [N,R,D], ape [R,D]");
  TORCH_CHECK(score.size(0) == kv.size(0) && score.size(1) == kv.size(1) && score.size(2) == kv.size(2),
              "score shape must equal kv shape");
  TORCH_CHECK(ape.size(0) == kv.size(1) && ape.size(1) == kv.size(2), "ape must be [R,D] matching kv");
  TORCH_CHECK(kv.device().is_cpu() && score.device().is_cpu() && ape.device().is_cpu(), "CPU tensors only");
  auto kvc = kv.to(torch::kFloat32).contiguous();
  auto scc = score.to(torch::kFloat32).contiguous();
  auto apc = ape.to(torch::kFloat32).contiguous();
  const int64_t N = kvc.size(0), R = kvc.size(1), D = kvc.size(2);
  auto out = torch::empty({N, D}, torch::kFloat32);
  if (N == 0 || D == 0) return out;
  const float* kp = kvc.data_ptr<float>();
  const float* sp = scc.data_ptr<float>();
  const float* ap = apc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int64_t nthr = at::get_num_threads();
  int64_t dpt = std::max<int64_t>(1, (2 * nthr + N - 1) / N);
  int64_t DT = std::max<int64_t>(16, (D + dpt - 1) / dpt);
  DT = (DT + 15) / 16 * 16;
  const int64_t ndt = (D + DT - 1) / DT;

  at::parallel_for(0, N * ndt, 0, [&](int64_t t0, int64_t t1) {
    std::vector<float> m(DT), l(DT), acc(DT);
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t n = t / ndt, dt = t % ndt, d0 = dt * DT;
      const int64_t dd = std::min(DT, D - d0);
      for (int64_t j = 0; j < dd; ++j) { m[j] = -INFINITY; l[j] = 0.f; acc[j] = 0.f; }
      for (int64_t r = 0; r < R; ++r) {                           // pass 1: per-channel max over R
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        #pragma omp simd
        for (int64_t j = 0; j < dd; ++j) {
          float x = srow[j] + arow[j];
          m[j] = x > m[j] ? x : m[j];
        }
      }
      for (int64_t r = 0; r < R; ++r) {                           // pass 2: one exp/elem, accumulate
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        const float* krow = kp + (n * R + r) * D + d0;
        #pragma omp simd
        for (int64_t j = 0; j < dd; ++j) {
          float x = srow[j] + arow[j];
          float e = (x == -INFINITY) ? 0.f : std::exp(x - m[j]);   // m[j]==-inf (all masked) -> x==-inf -> 0
          l[j] += e;
          acc[j] += e * krow[j];
        }
      }
      float* orow = op + n * D + d0;
      #pragma omp simd
      for (int64_t j = 0; j < dd; ++j) orow[j] = (l[j] > 0.f) ? acc[j] / l[j] : 0.f;
    }
  });
  return out;
}


// ============================ D-variants: hand-written AVX-512 intrinsics (SKILL Stage-4) ============
// Goal: stop relying on the compiler/Sleef for the exp (the measured bottleneck) and test the user's
// bf16-run / fp32-exp-island idea. KEY CONSTRAINT grounded in the corpus: model.py Compressor.forward
// runs the pool in FP32 ("compression need fp32"); even the bf16 sparse_attn_kernel keeps T.exp in FP32.
// So exp is ALWAYS fp32 -> bf16 cannot speed the exp; it only changes operand precision (accuracy) and
// read bytes. These variants MEASURE that. Accumulators stay FP32 (ACCUMULATE-IN-FP32 rule).
#if defined(__AVX512F__)
#include <immintrin.h>

// Cephes single-precision exp, AVX-512 + FMA (~1 ulp). Range-reduce x = n*ln2 + r, exp(r) by degree-6
// minimax, scale by 2^n via integer exponent. Clamp so -inf -> ~0 (no nan into the poly).
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

// Reduced degree-4 Taylor exp(r) on |r|<=ln2/2 (~1e-4 rel): fewer FMAs than Cephes. Same range reduction.
static inline __m512 fexp512_ps(__m512 x) {
  const __m512 LOG2EF = _mm512_set1_ps(1.44269504088896341f);
  const __m512 C1 = _mm512_set1_ps(0.693359375f), C2 = _mm512_set1_ps(-2.12194440e-4f);
  x = _mm512_max_ps(x, _mm512_set1_ps(-87.3365478f));
  __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, LOG2EF), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, C1, x);
  r = _mm512_fnmadd_ps(n, C2, r);
  __m512 y = _mm512_set1_ps(1.0f / 24.0f);
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f / 6.0f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(0.5f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.0f));
  __m512i pw = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(pw));
}

// Round fp32 lane -> bf16 precision and back (RNE), to SIMULATE bf16 operands/intermediates on fp32 input.
static inline __m512 bf16_round(__m512 v) {
  __m512i b = _mm512_castps_si512(v);
  __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(b, 16), _mm512_set1_epi32(1));
  __m512i rnd = _mm512_add_epi32(_mm512_add_epi32(b, lsb), _mm512_set1_epi32(0x7FFF));
  return _mm512_castsi512_ps(_mm512_and_si512(rnd, _mm512_set1_epi32(0xFFFF0000u)));
}

// Shared intrinsic core: online-softmax over R, 16-lane, FP32 accumulate. EXP selects exp512/fexp512;
// BF16IO rounds score+ape, the running max, and kv to bf16 each step (exp island + accumulate stay fp32).
template <bool BF16IO, bool FAST>
static torch::Tensor compressor_core_intrin(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  TORCH_CHECK(kv.dim() == 3 && score.dim() == 3 && ape.dim() == 2, "kv/score [N,R,D], ape [R,D]");
  auto kvc = kv.to(torch::kFloat32).contiguous();
  auto scc = score.to(torch::kFloat32).contiguous();
  auto apc = ape.to(torch::kFloat32).contiguous();
  const int64_t N = kvc.size(0), R = kvc.size(1), D = kvc.size(2);
  auto out = torch::empty({N, D}, torch::kFloat32);
  if (N == 0 || D == 0) return out;
  const float* kp = kvc.data_ptr<float>();
  const float* sp = scc.data_ptr<float>();
  const float* ap = apc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int64_t nthr = at::get_num_threads();
  int64_t dpt = std::max<int64_t>(1, (2 * nthr + N - 1) / N);
  int64_t DT = std::max<int64_t>(16, (D + dpt - 1) / dpt);
  DT = (DT + 15) / 16 * 16;
  const int64_t ndt = (D + DT - 1) / DT;
  const __m512 vninf = _mm512_set1_ps(-INFINITY), vone = _mm512_set1_ps(1.f);

  at::parallel_for(0, N * ndt, 0, [&](int64_t t0, int64_t t1) {
    std::vector<float> m(DT), l(DT), acc(DT);
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t n = t / ndt, dt = t % ndt, d0 = dt * DT;
      const int64_t dd = std::min(DT, D - d0);
      for (int64_t j = 0; j < dd; ++j) { m[j] = -INFINITY; l[j] = 0.f; acc[j] = 0.f; }
      for (int64_t r = 0; r < R; ++r) {
        const float* srow = sp + (n * R + r) * D + d0;
        const float* arow = ap + r * D + d0;
        const float* krow = kp + (n * R + r) * D + d0;
        int64_t j = 0;
        for (; j + 16 <= dd; j += 16) {
          __m512 x = _mm512_add_ps(_mm512_loadu_ps(srow + j), _mm512_loadu_ps(arow + j));
          __m512 kk = _mm512_loadu_ps(krow + j);
          if (BF16IO) { x = bf16_round(x); kk = bf16_round(kk); }
          __m512 mj = _mm512_loadu_ps(m.data() + j);
          __mmask16 up = _mm512_cmp_ps_mask(x, mj, _CMP_GT_OQ);
          __mmask16 mjn = _mm512_cmp_ps_mask(mj, vninf, _CMP_EQ_OQ);
          __mmask16 xn = _mm512_cmp_ps_mask(x, vninf, _CMP_EQ_OQ);
          __mmask16 survm = _kor_mask16(_kand_mask16(up, mjn), _kandn_mask16(up, xn));
          __m512 mnew = _mm512_max_ps(x, mj);
          __m512 arg = _mm512_mask_blend_ps(up, _mm512_sub_ps(x, mj), _mm512_sub_ps(mj, x));
          arg = _mm512_mask_blend_ps(survm, arg, vninf);
          __m512 tv = FAST ? fexp512_ps(arg) : exp512_ps(arg);
          __m512 corr = _mm512_mask_blend_ps(up, vone, tv);     // up? t : 1
          __m512 e = _mm512_mask_blend_ps(up, tv, vone);        // up? 1 : t
          __m512 lv = _mm512_fmadd_ps(_mm512_loadu_ps(l.data() + j), corr, e);
          __m512 av = _mm512_fmadd_ps(_mm512_loadu_ps(acc.data() + j), corr, _mm512_mul_ps(e, kk));
          _mm512_storeu_ps(l.data() + j, lv);
          _mm512_storeu_ps(acc.data() + j, av);
          _mm512_storeu_ps(m.data() + j, mnew);
        }
        for (; j < dd; ++j) {                                   // scalar tail (fp32 libm exp)
          float x = srow[j] + arow[j], mj = m[j];
          bool up = x > mj; float mnew = up ? x : mj;
          bool sm = up ? (mj == -INFINITY) : (x == -INFINITY);
          float arg = sm ? -INFINITY : (up ? (mj - x) : (x - mj));
          float tt = std::exp(arg), corr = up ? tt : 1.f, e = up ? 1.f : tt;
          l[j] = l[j] * corr + e; acc[j] = acc[j] * corr + e * krow[j]; m[j] = mnew;
        }
      }
      float* orow = op + n * D + d0;
      for (int64_t j = 0; j < dd; ++j)
        orow[j] = (m[j] != -INFINITY && l[j] > 0.f) ? acc[j] / l[j] : 0.f;   // all-masked (m stays -inf) -> 0, matching C1
    }
  });
  return out;
}

torch::Tensor compressor_softmax_pool_fp32intrin(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  return compressor_core_intrin<false, false>(kv, score, ape);   // D1: hand Cephes exp, fp32
}
torch::Tensor compressor_softmax_pool_fastexp(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  return compressor_core_intrin<false, true>(kv, score, ape);    // D2: reduced degree-4 exp, fp32
}
torch::Tensor compressor_softmax_pool_bf16exp(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) {
  return compressor_core_intrin<true, true>(kv, score, ape);     // D3: bf16 operands, fp32 exp island
}
#else   // portable fallback (non-AVX512 dev host): alias to the scalar bit-exact path so it still builds.
torch::Tensor compressor_softmax_pool_fp32intrin(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) { return compressor_softmax_pool(kv, score, ape); }
torch::Tensor compressor_softmax_pool_fastexp(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) { return compressor_softmax_pool(kv, score, ape); }
torch::Tensor compressor_softmax_pool_bf16exp(torch::Tensor kv, torch::Tensor score, torch::Tensor ape) { return compressor_softmax_pool(kv, score, ape); }
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compressor_softmax_pool", &compressor_softmax_pool,
        "DSA compressor softmax-pool (streaming online-softmax, no w[N,R,D] temporary)");
  m.def("compressor_softmax_pool_vexp", &compressor_softmax_pool_vexp,
        "C2 THROUGHPUT path: 512-bit at::vec one-exp online-softmax. Build with -mprefer-vector-width=512 "
        "-DCPU_CAPABILITY_AVX512 -> 2.2x over C1 scalar (NOT bit-exact to scalar libm; F4-screened)");
  m.def("compressor_softmax_pool_multipass", &compressor_softmax_pool_multipass,
        "C3 variant: stable two-pass pooling (max over R, one exp/elem, divide once)");
  m.def("compressor_softmax_pool_fp32intrin", &compressor_softmax_pool_fp32intrin,
        "D1: hand-written AVX-512 Cephes exp (~1ulp), fp32 operands + fp32 accumulate");
  m.def("compressor_softmax_pool_fastexp", &compressor_softmax_pool_fastexp,
        "D2: hand-written AVX-512 reduced degree-4 exp (~1e-4), fp32 operands + fp32 accumulate");
  m.def("compressor_softmax_pool_bf16exp", &compressor_softmax_pool_bf16exp,
        "D3: bf16 operands/intermediates (RNE-rounded) + fp32 exp island + fp32 accumulate (user's bf16 ask)");
}
