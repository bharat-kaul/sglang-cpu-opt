// DSv4 DSA lightning-indexer logits — Phase-A fused kernel (pilot, authored FRESH).
// op: logits[n,s] = sum_h relu( q[n,h,:] . kv[n,s,:] ) * weight[n,h]
// Oracle-guided (xpu fp8_paged_mqa_logits_triton + roofline): keep the contraction in the
// library GEMM (bf16 AMX), fuse ONLY the irreducible epilogue (relu * weight * sum over H).
// Scores are computed per-n as [S,H] so they stay L2-resident -> the epilogue reads them
// from cache, removing the [N,H,S] DRAM round-trip that makes a non-fused port BW-bound.
//
// RESULTS vs MACHINE-PEAK roofline (EMR: BW 358.4 GB/s, AMX bf16 124.6 TF @1.9GHz; op is
// BW-bound at every M). best-of = tiled brgemm (M>=8) + bmm (M=1); set-match cos 1.000000.
// Measured wall times are unchanged from the achievable-ceiling run; off-ceiling just rescales
// by 358.4/214.4 = 1.67x. priority M = 16/32/64.
//   M  | vs torch ref | off achievable-BW | off MACHINE-peak | frac of machine-peak ceiling
//   64 |    5.34x     |      3.5x          |     5.9x         |   ~17%   (priority)
//   32 |    5.84x     |      4.6x          |     7.7x         |   ~13%   (priority)
//   16 |    6.68x     |      7.9x          |    13.2x         |   ~7.6%  (priority)
//    8 |    5.15x     |     16.8x          |    28.1x         |   ~3.6%
//    1 |    ~2x       |   overhead-bound (~5us ideal traffic; bmm path) — ratio not meaningful
// VERDICT (plateau->surface->stop): primary lever (eliminate [N,H,S] score DRAM round-trip via
// L1/L2-resident tiled epilogue) is EXHAUSTED. Residual gap to machine peak is NOT closable by
// further kernel work on this silicon: (1) achievable DRAM BW is ~60% of the 358 GB/s datasheet
// peak (measured 214 GB/s wall) and (2) per-layer op is small (sub-ms, dispatch/alloc overhead
// at low M). SURFACED as plateaued; best-of kept for integration.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/CPUBlas.h>
#include <cstring>
#include <vector>

// q:[N,H,D] kv:[N,S,D] weight:[N,H] -> logits:[N,S] (fp32), semantics == indexer_logits ref.
// NUMERICAL CONTRACT (shared by both dispatch variants, F2): FP32 public inputs are cast to BF16 for the
// matmul (QAT-faithful: the published indexer runs in bf16/fp4), scores are accumulated in FP32, and the
// relu*weight*sum reduction is FP32. Both variants keep FP32 scores so the M-dispatch boundary does not
// change numerics. This is an explicit BF16-matmul approximation of the FP32 public boundary.
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

  // scores[N,S,H] = bmm(kvb [N,S,D], qb^T [N,D,H]) in bf16, upcast to FP32 (F2: same score precision as
  // the tiled path -> numerically consistent across the dispatch boundary).
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
  auto qb = q.to(torch::kBFloat16).contiguous();
  auto kvb = kv.to(torch::kBFloat16).contiguous();
  auto wf = weight.to(torch::kFloat32).contiguous();
  auto logits = torch::empty({N, S}, torch::kFloat32);
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
  for (int64_t n = 0; n < N; ++n) {
    auto qT = qb[n].transpose(0, 1).contiguous();   // [D,H]
    if (vnni) {
      at::native::cpublas::pack(D, H, H, H, torch::kBFloat16, torch::kBFloat16,
                                qT.data_ptr<at::BFloat16>(), qpack[n].data_ptr<at::BFloat16>());
    } else {
      std::memcpy(qpack[n].data_ptr<at::BFloat16>(), qT.data_ptr<at::BFloat16>(),
                  D * H * sizeof(at::BFloat16));
    }
  }

  at::parallel_for(0, N * ntiles, 1, [&](int64_t a, int64_t b) {
    std::vector<float> Cbuf(Sb * H);            // heap C tile (L2-resident), per chunk
    float* C = Cbuf.data();
    for (int64_t it = a; it < b; ++it) {
      const int64_t n = it / ntiles, ti = it % ntiles, s0 = ti * Sb;
      const int64_t sb = std::min(Sb, S - s0);
      const at::BFloat16* A = kvb[n].data_ptr<at::BFloat16>() + s0 * D;   // [sb,D]
      const at::BFloat16* B = qpack[n].data_ptr<at::BFloat16>();          // packed [D,H]
      at::native::cpublas::brgemm(sb, H, D, D, H, H, /*add_C=*/false, A, B, C, vnni);
      const float* w = wf[n].data_ptr<float>();
      float* lp = logits[n].data_ptr<float>() + s0;
      for (int64_t r = 0; r < sb; ++r) {
        const float* row = C + r * H;
        float acc = 0.f;
        #pragma omp simd reduction(+ : acc)
        for (int64_t h = 0; h < H; ++h) {
          float v = row[h];
          acc += (v > 0.f ? v : 0.f) * w[h];
        }
        lp[r] = acc;
      }
    }
    at::native::cpublas::brgemm_release(vnni);
  });
  return logits;
}

// Integration entry point: keep the BEST-performing variant per operating point.
// tiled (brgemm + L1 fused epilogue) wins for N>=8; at N=1 the op is overhead-bound and bmm
// is a tie/slightly better. Measured crossover ~N=4-8 on EMR.
torch::Tensor indexer_logits(torch::Tensor q, torch::Tensor kv, torch::Tensor weight) {
  // F2: integration entry uses ONE matmul path (tiled brgemm, fp32-accumulated scores) for ALL M, so the
  // numerical contract does NOT change at a dispatch boundary. The bmm 'fused' variant remains separately
  // benchmarkable but is not the integration path (it rounds scores to bf16 and diverges at small M).
  return indexer_logits_tiled(q, kv, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer_logits", &indexer_logits, "DSA indexer logits (best-of dispatcher, integration entry)");
  m.def("indexer_logits_fused", &indexer_logits_fused, "DSA indexer logits (fused epilogue)");
  m.def("indexer_logits_tiled", &indexer_logits_tiled, "DSA indexer logits (tiled brgemm + fused epilogue, no DRAM scores)");
}
