// DSv4 DSA sparse attend (top-k KV) — Phase-A kernel (pilot, authored per PUBLISHED model.py).
// MLA sparse attention: q[N,H,D] attends over K gathered latent KV[N,K,D] (num_key_value_heads=1 =>
// MQA: ONE kv latent shared across H=64 heads; k === v = the latent, D = head_dim = 512) with a
// per-head learned softmax SINK (attn_sink[H]) that absorbs probability but contributes no value.
// Ref: DeepSeek-V4-Flash inference/model.py Attention.sparse_attn(q, kv, attn_sink, topk, scale).
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <cmath>
#include <vector>

// q:[N,H,D]; kv:[N,K,D] (shared k==v latent); sink:[H] -> out:[N,H,D]. scale default = D**-0.5.
torch::Tensor sparse_attend(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && sink.device().is_cpu(), "CPU tensors only");
  TORCH_CHECK(std::isfinite(scale_), "scale must be finite (pass a finite scale<=0 for the default D**-0.5)");
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1, "bad dims");
  TORCH_CHECK(q.size(0) == kv.size(0), "q/kv batch N mismatch");
  TORCH_CHECK(q.size(2) == kv.size(2), "q/kv head_dim mismatch");
  TORCH_CHECK(sink.size(0) == q.size(1), "sink must be [H]");
  auto qc = q.to(torch::kFloat32).contiguous();
  auto kc = kv.to(torch::kFloat32).contiguous();
  auto sc = sink.to(torch::kFloat32).contiguous();
  const int64_t N = qc.size(0), H = qc.size(1), D = qc.size(2), K = kc.size(1);
  const float scale = scale_ > 0 ? (float)scale_ : (float)(1.0 / std::sqrt((double)D));
  auto out = torch::empty({N, H, D}, torch::kFloat32);
  const float* qp = qc.data_ptr<float>();
  const float* kp = kc.data_ptr<float>();
  const float* sp = sc.data_ptr<float>();
  float* op = out.data_ptr<float>();

  at::parallel_for(0, N * H, 0, [&](int64_t i0, int64_t i1) {
    std::vector<float> acc(D);
    for (int64_t i = i0; i < i1; ++i) {
      const int64_t n = i / H, h = i % H;
      const float* qrow = qp + (n * H + h) * D;
      const float* kvbase = kp + n * K * D;          // latent shared across heads (MQA)
      float m = sp[h], l = 1.f;                        // sink: logit sp[h], weight exp(0)=1, no value
      for (int64_t d = 0; d < D; ++d) acc[d] = 0.f;
      for (int64_t kk = 0; kk < K; ++kk) {
        const float* kvr = kvbase + kk * D;
        float s = 0.f;
        #pragma omp simd reduction(+ : s)
        for (int64_t d = 0; d < D; ++d) s += qrow[d] * kvr[d];
        s *= scale;
        float mnew = s > m ? s : m;
        float corr = std::exp(m - mnew), e = std::exp(s - mnew);
        l = l * corr + e;
        #pragma omp simd
        for (int64_t d = 0; d < D; ++d) acc[d] = acc[d] * corr + e * kvr[d];
        m = mnew;
      }
      float inv = 1.f / l;
      float* orow = op + (n * H + h) * D;
      #pragma omp simd
      for (int64_t d = 0; d < D; ++d) orow[d] = acc[d] * inv;
    }
  });
  return out;
}

// fp32 bmm DONOR path (OneDNN/MKL via at::bmm) + FUSED softmax/sink epilogue. For N>=2 the two
// matmuls are GEMM-bound; MKL bmm sits near fp32 roofline (~64% peak) where the scalar online-
// softmax loses badly. We reuse the donor for the GEMMs but fuse the softmax (max/exp/sum/div +
// sink) into ONE parallel pass over [N,H,K] in-place, avoiding the ref's 4 extra full temporaries
// (m,e,denom,w). Bit-conformant with the fp32 oracle (no bf16), so it holds the no-regression floor.
torch::Tensor sparse_attend_fp32bmm(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && sink.device().is_cpu(), "CPU tensors only");
  TORCH_CHECK(std::isfinite(scale_), "scale must be finite (pass a finite scale<=0 for the default D**-0.5)");
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1, "bad dims");
  TORCH_CHECK(q.size(0) == kv.size(0), "q/kv batch N mismatch");
  TORCH_CHECK(q.size(2) == kv.size(2), "q/kv head_dim mismatch");
  TORCH_CHECK(sink.size(0) == q.size(1), "sink must be [H]");
  auto qc = q.to(torch::kFloat32).contiguous();
  auto kc = kv.to(torch::kFloat32).contiguous();
  auto sc = sink.to(torch::kFloat32).contiguous();
  const int64_t N = qc.size(0), H = qc.size(1), D = qc.size(2), K = kc.size(1);
  const float scale = scale_ > 0 ? (float)scale_ : (float)(1.0 / std::sqrt((double)D));
  // scores[N,H,K] = q @ kv^T  (donor GEMM)
  auto scores = at::bmm(qc, kc.transpose(1, 2));                               // [N,H,K] fp32
  const float* sp = sc.data_ptr<float>();
  float* sdp = scores.data_ptr<float>();
  // fused in-place softmax(+per-head sink) over K: row r = (n,h) -> scores[r, :]*scale
  at::parallel_for(0, N * H, 0, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const int64_t h = i % H;
      float* row = sdp + i * K;
      float m = sp[h];                                  // sink logit participates in the max
      for (int64_t kk = 0; kk < K; ++kk) { float s = row[kk] * scale; row[kk] = s; if (s > m) m = s; }
      float l = std::exp(sp[h] - m);                    // sink weight (no value contribution)
      for (int64_t kk = 0; kk < K; ++kk) { float e = std::exp(row[kk] - m); row[kk] = e; l += e; }
      float inv = 1.f / l;
      #pragma omp simd
      for (int64_t kk = 0; kk < K; ++kk) row[kk] *= inv;
    }
  });
  // out[N,H,D] = w @ kv  (donor GEMM)
  return at::bmm(scores, kc);                                                  // [N,H,D] fp32
}

// Best-of dispatcher: N==1 => scalar online-softmax (avoids bmm launch overhead, 3.38x torch,
// bit-exact); N>=2 => fp32 bmm donor + fused softmax (GEMM-bound regime, holds floor vs torch).
// The bf16 AMX path is NOT wired here: it is non-conformant to the fp32 oracle (cos~0.9999,
// degrading with M) and did not beat torch at any M; it remains experimental (sparse_attend_amx).
torch::Tensor sparse_attend_bestof(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  if (q.size(0) == 1) return sparse_attend(q, kv, sink, scale_);
  return sparse_attend_fp32bmm(q, kv, sink, scale_);
}

// bf16 AMX batched GEMM: score=bmm(q,kv^T) then softmax(+sink) then out=bmm(w,kv). At real dims
// (H=64 is the GEMM M, D=512, K=512) the two matmuls are AMX-tileable, unlike per-query M=1.
torch::Tensor sparse_attend_amx(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && sink.device().is_cpu(), "CPU tensors only");
  TORCH_CHECK(std::isfinite(scale_), "scale must be finite (pass a finite scale<=0 for the default D**-0.5)");
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1, "bad dims");
  TORCH_CHECK(q.size(0) == kv.size(0), "q/kv batch N mismatch");
  TORCH_CHECK(q.size(2) == kv.size(2), "q/kv head_dim mismatch");
  TORCH_CHECK(sink.size(0) == q.size(1), "sink must be [H]");
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2), K = kv.size(1);
  const double scale = scale_ > 0 ? scale_ : 1.0 / std::sqrt((double)D);
  auto qb = q.to(torch::kBFloat16).contiguous();
  auto kb = kv.to(torch::kBFloat16).contiguous();
  auto sk = sink.to(torch::kFloat32).view({1, H, 1});
  auto scores = at::bmm(qb, kb.transpose(1, 2)).to(torch::kFloat32) * scale;   // [N,H,K]
  auto m = at::maximum(std::get<0>(scores.max(-1, true)), sk);                 // [N,H,1]
  auto e = (scores - m).exp();                                                 // [N,H,K]
  auto denom = e.sum(-1, true) + (sk - m).exp();                               // + sink term
  auto w = (e / denom).to(torch::kBFloat16);
  return at::bmm(w, kb).to(torch::kFloat32);                                   // [N,H,D]
}
// S1 (review b51d1eb): SOURCE-FAITHFUL 64-block flash attention with BF16-input/FP32-output GEMMs via the
// cpublas brgemm facility (the same library capability the indexer uses). Per n: run all H heads as a GEMM
// dim; stream K in 64-blocks; per block do a bf16 score GEMM (fp32 accumulate), an online-softmax update
// with a BF16 cast of the UNNORMALIZED exp'd weights before the value GEMM (bf16, fp32 accumulate), per-block
// fp32 rescale of the running sum/accumulator; add the sink AFTER the block loop with the FINAL running max;
// BF16 output. Matches sparse_ref.ora_sparse_blockwise op/rounding order. NOT bit-exact to the fp32 donor
// (bf16 operands) -> F4-SCREENED. EXPERIMENTAL: measured vs the shipped fp32-bmm donor; a loss is a valid
// disposition (many small packed GEMMs per request may outweigh the bf16 AMX contraction).
// Both GEMMs are framed with N=64 (the AMX pack tile width caps N): the value step computes the TRANSPOSED
// accumulator accT[D,H] = kvblkT[D,blk] @ wblkT[blk,H] (N=H=64), reusing the transposed KV tile as score-B
// and value-A.
torch::Tensor sparse_attend_blockbf16(torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale_) {
  TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() && sink.device().is_cpu(), "CPU tensors only");
  TORCH_CHECK(std::isfinite(scale_), "scale must be finite (pass a finite scale<=0 for the default D**-0.5)");
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1, "bad dims");
  TORCH_CHECK(q.size(0) == kv.size(0), "q/kv batch N mismatch");
  TORCH_CHECK(q.size(2) == kv.size(2), "q/kv head_dim mismatch");
  TORCH_CHECK(sink.size(0) == q.size(1), "sink must be [H]");
  const int64_t N = q.size(0), H = q.size(1), D = q.size(2), K = kv.size(1);
  const float scale = scale_ > 0 ? (float)scale_ : (float)(1.0 / std::sqrt((double)D));
  constexpr int64_t BLK = 64;
  auto qb = q.to(torch::kBFloat16).contiguous();                 // [N,H,D] bf16 (score A)
  auto kvb = kv.to(torch::kBFloat16).contiguous();               // [N,K,D] bf16 (source for the transposed tile)
  auto out = torch::empty({N, H, D}, torch::kFloat32);
  const at::BFloat16* qp = qb.data_ptr<at::BFloat16>();
  const at::BFloat16* kp = kvb.data_ptr<at::BFloat16>();
  auto skc = sink.to(torch::kFloat32).contiguous();
  const float* skp = skc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const bool vnni = at::native::cpublas::could_pack(torch::kBFloat16);

  at::parallel_for(0, N, 0, [&](int64_t n0, int64_t n1) {
    std::vector<at::BFloat16> kvtile(D * BLK), spack(D * BLK), wpack(BLK * H), wblkT(BLK * H);
    std::vector<float> sblk(H * BLK), accT(D * H), m(H), l(H), resc(H);
    for (int64_t n = n0; n < n1; ++n) {
      const at::BFloat16* qn = qp + n * H * D;
      const at::BFloat16* kn = kp + n * K * D;
      for (int64_t h = 0; h < H; ++h) { m[h] = -INFINITY; l[h] = 0.f; }
      std::fill(accT.begin(), accT.end(), 0.f);
      for (int64_t s0 = 0; s0 < K; s0 += BLK) {
        const int64_t blk = std::min(BLK, K - s0);
        // transpose kvblk[blk,D] -> kvtile[D,blk] (contiguous), used as score-B source AND value-A
        for (int64_t j = 0; j < blk; ++j) {
          const at::BFloat16* krow = kn + (s0 + j) * D;
          for (int64_t d = 0; d < D; ++d) kvtile[d * blk + j] = krow[d];
        }
        // score GEMM: sblk[H,blk] = qn[H,D] @ kvtile[D,blk] (N=blk<=64), bf16 operands -> fp32 C
        const at::BFloat16* Bs = kvtile.data();
        int64_t ldbs = blk;
        if (vnni) { at::native::cpublas::pack(D, blk, blk, blk, torch::kBFloat16, torch::kBFloat16, kvtile.data(), spack.data()); Bs = spack.data(); }
        at::native::cpublas::brgemm(H, blk, D, D, ldbs, blk, /*add_C=*/false, qn, Bs, sblk.data(), vnni);
        // online softmax update over this block (fp32 state); sink added AFTER the loop
        for (int64_t h = 0; h < H; ++h) {
          float* srow = sblk.data() + h * blk;
          float mprev = m[h], mx = mprev;
          for (int64_t j = 0; j < blk; ++j) { srow[j] *= scale; if (srow[j] > mx) mx = srow[j]; }
          float mnew = mx;
          float rc = (mprev == -INFINITY) ? 0.f : std::exp(mprev - mnew);
          resc[h] = rc;
          float ls = l[h] * rc;
          for (int64_t j = 0; j < blk; ++j) {
            float e = std::exp(srow[j] - mnew);
            ls += e;
            wblkT[j * H + h] = static_cast<at::BFloat16>(e);       // wblkT[blk,H]: BF16 unnormalized exp
          }
          l[h] = ls; m[h] = mnew;
        }
        for (int64_t d = 0; d < D; ++d) {                         // rescale running accT[D,H] by resc[h]
          float* arow = accT.data() + d * H;
          for (int64_t h = 0; h < H; ++h) arow[h] *= resc[h];
        }
        // value GEMM: accT[D,H] += kvtile[D,blk] @ wblkT[blk,H] (N=H<=64), bf16 operands -> fp32 C (add_C)
        const at::BFloat16* Bv = wblkT.data();
        int64_t ldbv = H;
        if (vnni) { at::native::cpublas::pack(blk, H, H, H, torch::kBFloat16, torch::kBFloat16, wblkT.data(), wpack.data()); Bv = wpack.data(); }
        at::native::cpublas::brgemm(D, H, blk, blk, ldbv, H, /*add_C=*/true, kvtile.data(), Bv, accT.data(), vnni);
      }
      if (vnni) at::native::cpublas::brgemm_release(vnni);
      for (int64_t h = 0; h < H; ++h) {
        float denom = l[h] + std::exp(skp[h] - m[h]);             // sink post-loop with FINAL running max
        float inv = 1.f / denom;
        float* orow = op + (n * H + h) * D;
        for (int64_t d = 0; d < D; ++d)
          orow[d] = static_cast<float>(static_cast<at::BFloat16>(accT[d * H + h] * inv));   // bf16 output (transposed read)
      }
    }
  });
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_attend", &sparse_attend, "DSA MLA sparse attend (MQA flash, per-head sink; scalar online-softmax)");
  m.def("sparse_attend_fp32bmm", &sparse_attend_fp32bmm, "DSA MLA sparse attend via fp32 bmm donor + fused softmax (+sink)");
  m.def("sparse_attend_bestof", &sparse_attend_bestof, "DSA MLA sparse attend best-of dispatch (N==1 scalar, else fp32 bmm donor)");
  m.def("sparse_attend_amx", &sparse_attend_amx, "DSA MLA sparse attend via bf16 AMX bmm (+sink) [EXPERIMENTAL, non-conformant]");
  m.def("sparse_attend_blockbf16", &sparse_attend_blockbf16, "S1: source-faithful 64-block bf16-in/fp32-out brgemm flash (+sink) [EXPERIMENTAL]");
}
