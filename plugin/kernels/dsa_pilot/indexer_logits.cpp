// DSv4 DSA lightning-indexer logits — Phase-A fused kernel (pilot, authored FRESH).
// op: logits[n,s] = sum_h relu( q[n,h,:] . kv[n,s,:] ) * weight[n,h]
// Oracle-guided (xpu fp8_paged_mqa_logits_triton + roofline): keep the contraction in the
// library GEMM (bf16 AMX), fuse ONLY the irreducible epilogue (relu * weight * sum over H).
// Scores are computed per-n as [S,H] so they stay L2-resident -> the epilogue reads them
// from cache, removing the [N,H,S] DRAM round-trip that makes a non-fused port BW-bound.
//
// RESULTS vs MACHINE-PEAK roofline (EMR: BW 358.4 GB/s, AMX bf16 124.6 TF @1.9GHz; op is
// BW-bound at every M). set-match cos 1.000000. The INTEGRATION entry uses the single tiled path for all M
// (F2); the separate bmm 'fused' export is BF16-score-rounded and is NOT the integration path.
// NOTE: the per-M speedup/plateau table previously here was SUPERSEDED and is retired. Current
// measured latency is the validated SLURM 384414 sweep (plugin/validate/results/perf_sweep.json),
// joined in dsv4_roofline_vs_measured.py. No plateau/ROI-exhaustion claim is made in source: the
// earlier "primary lever EXHAUSTED / residual gap NOT closable by kernel work" verdict was RETRACTED
// (F7) — useful byte-throughput is still far below reference BW, so ROI is OPEN with ranked hypotheses
// (indexer conversion/pack/reduction split, larger-M attribution), NOT a proven wall.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// q:[N,H,D] kv:[N,S,D] weight:[N,H] -> logits:[N,S] (fp32), semantics == indexer_logits ref.
// NUMERICAL CONTRACT (F2): the INTEGRATION entry (indexer_logits) uses the TILED path for ALL M, whose
// scores stay FP32-accumulated, so the M-dispatch boundary does not change numerics. This 'fused' variant
// is a SEPARATE export only (not reachable from the dispatcher): it computes scores via a BF16 bmm whose
// output is BF16-ROUNDED before the FP32 upcast, so its scores are NOT precision-identical to the tiled
// path (observed max logit diff ~0.045 on the FP4-grid seed-1 input). Do not claim shared score precision.
torch::Tensor indexer_logits_fused(torch::Tensor q, torch::Tensor kv, torch::Tensor weight) {
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && weight.dim() == 2, "q/kv must be [N,H,D]/[N,S,D], weight [N,H]");
  TORCH_CHECK(q.size(0) == kv.size(0) && q.size(2) == kv.size(2), "q/kv batch and head_dim must match");
  TORCH_CHECK(weight.size(0) == q.size(0) && weight.size(1) == q.size(1), "weight must be [N,H]");
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && weight.device().is_cpu(), "CPU tensors only");
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2), S = kv.size(1);
  auto qb = q.to(torch::kBFloat16).contiguous();
  auto kvb = kv.to(torch::kBFloat16).contiguous();
  auto wf = weight.to(torch::kFloat32).contiguous();
  auto logits = torch::empty({N, S}, torch::kFloat32);

  // scores[N,S,H] = bmm(kvb [N,S,D], qb^T [N,D,H]) in bf16, upcast to FP32. NOTE: the bmm output is
  // BF16-ROUNDED before this upcast, so these scores are NOT precision-identical to the tiled path (the
  // integration entry uses tiled only, F2, to avoid this boundary difference).
  auto scores = at::bmm(kvb, qb.transpose(1, 2)).to(torch::kFloat32).contiguous();   // [N,S,H] fp32
  const float* sp = scores.data_ptr<float>();
  const float* wp = wf.data_ptr<float>();
  float* lp = logits.data_ptr<float>();
  // fused epilogue: one pass over N*S, inner H contiguous, fp32 relu/mul/sum (no temps).
  at::parallel_for(0, N * S, 4096, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const float* row = sp + i * H;
      const float* w = wp + (i / S) * H;
      float acc = 0.f;
      #pragma omp simd reduction(+ : acc)
      for (int64_t h = 0; h < H; ++h) {
        float v = row[h];
        acc += (v > 0.f ? v : 0.f) * w[h];
      }
      lp[i] = acc;
    }
  });
  return logits;
}

// ---- tiled brgemm + per-tile fused epilogue: scores never leave L1 (no DRAM round-trip) ----
// Per n: pack q[n]^T [D,H] to VNNI once; loop S in Sb-row tiles; brgemm each tile's scores
// [Sb,H] into an L1 stack buffer; apply relu*weight*sum over H in place -> logits. kv streamed once.
torch::Tensor indexer_logits_tiled(torch::Tensor q, torch::Tensor kv, torch::Tensor weight) {
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && weight.dim() == 2, "q/kv must be [N,H,D]/[N,S,D], weight [N,H]");
  TORCH_CHECK(q.size(0) == kv.size(0) && q.size(2) == kv.size(2), "q/kv batch and head_dim must match");
  TORCH_CHECK(weight.size(0) == q.size(0) && weight.size(1) == q.size(1), "weight must be [N,H]");
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && weight.device().is_cpu(), "CPU tensors only");
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2), S = kv.size(1);
  if (N == 0) return torch::empty({0, S}, torch::kFloat32);   // empty batch -> [0,S] (avoid /N below)
  const bool _brk = std::getenv("INDEXER_BREAKDOWN") != nullptr;   // env-gated stage timers (diagnostic)
  auto _now = [] { return std::chrono::high_resolution_clock::now(); };
  auto _t0 = _now();
  // BW lever (free): the brgemm already rounds kv->bf16 to compute, so a bf16 KV cache is numerically
  // IDENTICAL to what this kernel consumes while HALVING the dominant kv read. If kv arrives bf16, read
  // it directly (no fp32 round-trip, no per-tile convert); else keep fp32 and convert per-tile in-cache.
  const bool kv_bf16 = (kv.scalar_type() == torch::kBFloat16);
  auto qb = q.to(torch::kBFloat16).contiguous();
  auto kvf = kv_bf16 ? torch::Tensor() : kv.to(torch::kFloat32).contiguous();
  auto kvb = kv_bf16 ? kv.contiguous() : torch::Tensor();
  auto wf = weight.to(torch::kFloat32).contiguous();
  auto logits = torch::empty({N, S}, torch::kFloat32);
  auto _t1 = _now();
  const bool vnni = at::native::cpublas::could_pack(torch::kBFloat16);
  // Adaptive tile: target ~4*threads tasks so each brgemm M is large (AMX-efficient) yet enough
  // parallelism remains at small N. Sb multiple of 16, clamped [16,256] (Cbuf <= 64KB, L1/L2).
  const int64_t nthr = at::get_num_threads();
  int64_t tpn = std::max<int64_t>(1, (4 * nthr + N - 1) / N);        // tiles per n
  int64_t Sb = std::max<int64_t>(16, ((S + tpn - 1) / tpn + 15) / 16 * 16);
  Sb = std::min<int64_t>(Sb, 1024);               // larger tiles -> bigger brgemm M (AMX util)
  const int64_t ntiles = (S + Sb - 1) / Sb;

  // Pack B = q[n]^T [D,H] (VNNI) once per n; reused by every S-tile of that n.
  auto qpack = torch::empty({N, D * H}, torch::kBFloat16);
  at::parallel_for(0, N, 0, [&](int64_t n0, int64_t n1) {   // parallel over n (serial pack was a large-M bottleneck)
    for (int64_t n = n0; n < n1; ++n) {
      auto qT = qb[n].transpose(0, 1).contiguous();   // [D,H]
      if (vnni) {
        at::native::cpublas::pack(D, H, H, H, torch::kBFloat16, torch::kBFloat16,
                                  qT.data_ptr<at::BFloat16>(), qpack[n].data_ptr<at::BFloat16>());
      } else {
        std::memcpy(qpack[n].data_ptr<at::BFloat16>(), qT.data_ptr<at::BFloat16>(),
                    D * H * sizeof(at::BFloat16));
      }
    }
  });
  auto _t2 = _now();

  at::parallel_for(0, N * ntiles, 1, [&](int64_t a, int64_t b) {
    std::vector<float> Cbuf(Sb * H);            // heap C tile (L2-resident), per chunk
    std::vector<at::BFloat16> Abuf(kv_bf16 ? 0 : Sb * D);   // I1: staging only needed to convert FP32 KV
    std::vector<float> wbf(H);                  // I3: bf16-rounded weights, hoisted (invariant across rows)
    float* C = Cbuf.data();
    for (int64_t it = a; it < b; ++it) {
      const int64_t n = it / ntiles, ti = it % ntiles, s0 = ti * Sb;
      const int64_t sb = std::min(Sb, S - s0);
      const at::BFloat16* A;
      if (kv_bf16) {
        A = kvb[n].data_ptr<at::BFloat16>() + s0 * D;                   // bf16 kv read directly (half traffic)
      } else {
        const float* Afp = kvf[n].data_ptr<float>() + s0 * D;          // [sb,D] FP32 tile
        at::BFloat16* Ab = Abuf.data();
        const int64_t ne = sb * D;                                     // convert tile -> bf16 in cache
        int64_t i = 0;
        constexpr int64_t VW = at::vec::Vectorized<float>::size();
        for (; i + 2 * VW <= ne; i += 2 * VW) {
          auto v0 = at::vec::Vectorized<float>::loadu(Afp + i);
          auto v1 = at::vec::Vectorized<float>::loadu(Afp + i + VW);
          at::vec::convert_float_bfloat16(v0, v1).store(Ab + i);
        }
        for (; i < ne; ++i) Ab[i] = static_cast<at::BFloat16>(Afp[i]);
        A = Ab;
      }
      const at::BFloat16* B = qpack[n].data_ptr<at::BFloat16>();          // packed [D,H]
      at::native::cpublas::brgemm(sb, H, D, D, H, H, /*add_C=*/false, A, B, C, vnni);
      const float* w = wf[n].data_ptr<float>();
      float* lp = logits[n].data_ptr<float>() + s0;
      for (int64_t h = 0; h < H; ++h)
        wbf[h] = static_cast<float>(static_cast<at::BFloat16>(w[h]));         // I3: bf16(weight) once per task
      // Published stage boundaries (model.py L420-421): bf16 einsum output, bf16 relu, * bf16 weights,
      // bf16 reduce over heads -> bf16 logits (topk input). Round each stage to bf16; FP32-accumulate the
      // head sum (matches torch bf16 .sum) then round the logit to bf16. Store bf16-exact value in fp32.
      for (int64_t r = 0; r < sb; ++r) {
        const float* row = C + r * H;
        float acc = 0.f;
        for (int64_t h = 0; h < H; ++h) {
          float v = static_cast<float>(static_cast<at::BFloat16>(row[h]));     // bf16 einsum-output boundary
          float rv = v > 0.f ? v : 0.f;                                        // relu in bf16 domain
          acc += static_cast<float>(static_cast<at::BFloat16>(rv * wbf[h]));   // bf16(relu*weight) product
        }
        lp[r] = static_cast<float>(static_cast<at::BFloat16>(acc));            // bf16 logit (topk input)
      }
    }
    at::native::cpublas::brgemm_release(vnni);
  });
  if (_brk) {
    auto _t3 = _now();
    auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    std::fprintf(stderr, "[indexer_breakdown N=%ld S=%ld Sb=%ld ntiles=%ld] convert=%.1fus pack=%.1fus compute=%.1fus\n",
                 (long)N, (long)S, (long)Sb, (long)ntiles, us(_t0, _t1), us(_t1, _t2), us(_t2, _t3));
  }
  return logits;
}

// M=1 (N=1) low-overhead path: one brgemm over the whole S (fp32-output -> scores IDENTICAL to the tiled
// path's bf16-input/fp32-accumulate), then a parallel FP32 relu*weight*sum epilogue over S. Avoids the
// many-tiny-tile dispatch + per-tile overhead that made N=1 overhead-bound.
torch::Tensor indexer_logits_m1(torch::Tensor q, torch::Tensor kv, torch::Tensor weight) {
  const int64_t H = q.size(1), D = q.size(2), S = kv.size(1);
  auto qb = q.to(torch::kBFloat16).contiguous();
  auto kvb = kv.to(torch::kBFloat16).contiguous();
  auto wf = weight.to(torch::kFloat32).contiguous();
  auto logits = torch::empty({1, S}, torch::kFloat32);
  const bool vnni = at::native::cpublas::could_pack(torch::kBFloat16);
  auto qT = qb[0].transpose(0, 1).contiguous();                 // [D,H]
  auto qpack = torch::empty({D * H}, torch::kBFloat16);
  if (vnni) {
    at::native::cpublas::pack(D, H, H, H, torch::kBFloat16, torch::kBFloat16,
                              qT.data_ptr<at::BFloat16>(), qpack.data_ptr<at::BFloat16>());
  } else {
    std::memcpy(qpack.data_ptr<at::BFloat16>(), qT.data_ptr<at::BFloat16>(), D * H * sizeof(at::BFloat16));
  }
  std::vector<float> C(S * H);                                   // [S,H] fp32 scores (L2-resident)
  at::native::cpublas::brgemm(S, H, D, D, H, H, /*add_C=*/false, kvb[0].data_ptr<at::BFloat16>(),
                              qpack.data_ptr<at::BFloat16>(), C.data(), vnni);
  at::native::cpublas::brgemm_release(vnni);
  const float* w = wf.data_ptr<float>();
  float* lp = logits.data_ptr<float>();
  at::parallel_for(0, S, 0, [&](int64_t s0, int64_t s1) {        // parallel epilogue over S
    for (int64_t s = s0; s < s1; ++s) {
      const float* row = C.data() + s * H;
      float acc = 0.f;                                          // bf16 stage boundaries (model.py L420-421)
      for (int64_t h = 0; h < H; ++h) {
        float v = static_cast<float>(static_cast<at::BFloat16>(row[h]));   // bf16 einsum-output boundary
        float rv = v > 0.f ? v : 0.f;                                      // relu in bf16 domain
        float wb = static_cast<float>(static_cast<at::BFloat16>(w[h]));    // bf16 (signed) weight
        acc += static_cast<float>(static_cast<at::BFloat16>(rv * wb));     // bf16(relu*weight) product
      }
      lp[s] = static_cast<float>(static_cast<at::BFloat16>(acc));          // bf16 logit (topk input)
    }
  });
  return logits;
}

// Integration entry point: the SINGLE tiled path (brgemm + L1 fused epilogue) for ALL M (F2 — no per-M
// dispatch, so the numerical contract does not change at a boundary).
torch::Tensor indexer_logits(torch::Tensor q, torch::Tensor kv, torch::Tensor weight) {
  // Validate the COMMON contract HERE so the N==1 specialization cannot bypass it (P1-F2): both the m1 and
  // tiled paths read raw pointers assuming these hold.
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && weight.dim() == 2, "q/kv must be [N,H,D]/[N,S,D], weight [N,H]");
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && weight.device().is_cpu(), "CPU tensors only");
  TORCH_CHECK(q.size(0) == kv.size(0), "q/kv batch N mismatch");
  TORCH_CHECK(q.size(2) == kv.size(2), "q/kv head_dim mismatch");
  TORCH_CHECK(weight.size(0) == q.size(0) && weight.size(1) == q.size(1), "weight must be [N,H]");
  if (q.size(0) == 0) return torch::empty({0, kv.size(1)}, torch::kFloat32);   // empty batch -> [0,S]
  // F2: scores round to the SAME bf16 stage boundaries on BOTH paths (tiled brgemm; m1 single brgemm) ->
  // IDENTICAL numerics across the N==1 boundary. m1 avoids the tiny-tile overhead that bound N=1.
  if (q.size(0) == 1) return indexer_logits_m1(q, kv, weight);
  return indexer_logits_tiled(q, kv, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer_logits", &indexer_logits, "DSA indexer logits (single tiled path, integration entry)");
  m.def("indexer_logits_fused", &indexer_logits_fused, "DSA indexer logits (fused epilogue)");
  m.def("indexer_logits_tiled", &indexer_logits_tiled, "DSA indexer logits (tiled brgemm + fused epilogue, no DRAM scores)");
}
