---
name: meso-scale-optimization
description: "MESO scale of the multi-scale CPU performance tree (do after macro). Goal: for each KERNEL-DOMINANT op, get its DATA MOVEMENT optimal — cache-tiling + BRGEMM resident-C accumulator across the K reduction + one-time weight prepacking into VNNI/AMX-tile layout + operand-byte reduction (low-precision STORAGE) — to the STREAMED-achievable roofline (not the resident compute peak). Includes the arithmetic-intensity-vs-ridge classification that decides whether an op is compute-bound (fusion/kernel-worthy, grows with M) or BW-bound (operating-point-only). Load after macro-scale; route to cache-blocking-tiling, weight-prepacking-brgemm, kernel-authoring, compute-from-native-precision, roofline-validation."
---

# Meso-scale optimization — get each kernel's DATA MOVEMENT right

Middle scale. The question: **for this dominant kernel, are operands streamed with maximum
reuse at the right arithmetic intensity, so it hits the STREAMED-achievable roofline?**

## The two-ceiling rule (the core meso insight)
The resident microkernel's compute peak (e.g. ~350 TF/socket) is NOT the achievable ceiling for a
STREAMED GEMM — perf depends on whether operands are streamed or resident. The achievable target is
the **streamed-BW roofline** `min(compute_peak, BW × AI)`. The building block that reaches it is the
**batch-reduce GEMM (BRGEMM)**: a register-resident C accumulator with A and B streamed past it
across the whole K reduction (`C += Σ_r A_r·B_r`). In SGLang, `M>4` already routes to
`at::native::cpublas::brgemm()` (oneDNN, dispatches `avx512_core_amx`) — **BRGEMM is the BASELINE,
not an untried optimization. Reuse it; do not hand-roll a loop-nest.** (LIBXSMM/TPP — Heinecke SC'19
— is the source of truth; its wins are end-to-end: fused epilogues, persisted layouts, small/odd
shapes, batched throughput.) See `weight-prepacking-brgemm`, `kernel-authoring`.

## ⛔ CLASSIFY EACH OP: arithmetic intensity vs the ridge (decides the whole strategy)
Compute the **ridge AI** = `peak_FLOPs / peak_BW` on the TARGET machine with the REAL measured BW
(not a torch-clone-understated BW — use uPP stream_triad; e.g. EMR 11.6 TF/s ÷ 226 GB/s ≈ ridge 51).
For each op compute `AI = FLOP / bytes_moved`:
- **AI > ridge ⇒ COMPUTE-bound**, and AI GROWS with batch M → a fused AMX kernel wins and the win
  grows with M (DSA indexer AI≈63 → 7.2× at M=32). THIS is where kernel/fusion effort pays.
- **AI < ridge ⇒ BW-bound** → the lever is operand-byte reduction + killing intermediates (fused
  streaming), NOT more compute; the realistic win is bounded (DSA compressor AI≈1 → ~2–3×), and
  batching does little for its efficiency. Reconsider a C++ port here — RoI is small.
Mislabeling (BW-bound auto-labels from understated BW) sends you to the wrong lever — always use the
measured BW. See `roofline-validation`, `establish-achievable-performance`.

## The meso levers
1. **Cache-blocking / tiling** (`cache-blocking-tiling`): `BLOCK_M`/`BLOCK_N` (e.g. 32×32), loop order
   that keeps the C accumulator resident across K; block so A-tile + B-tile + C-tile fit the target
   cache level. The 1-4-16 / BRGEMM split by M.
2. **Weight prepacking** (`weight-prepacking-brgemm`): one-time pack weights into persisted VNNI/AMX
   -tile layout so the GEMM never repacks per-call and reads contiguous tiles. Usually the largest
   single win for a lone large GEMM; it closes the gap to the streamed-achievable ceiling.
3. **Operand-byte reduction = data-movement lever** (`compute-from-native-precision`): low-precision
   STORAGE (fp8/int4 weights) halves/quarters bytes moved — helps BW-bound AND feed-limited
   compute-bound ops; upcast in the microkernel load, never a separate dequant pass to DRAM.
4. **Contiguous layout invariant**: feed the kernel contiguous [.,.,D]; never a `transpose`-induced
   non-contiguous view that forces a repack copy inside the loop (a measured indexer tax).

## Exit criterion
Each dominant kernel at its streamed-achievable (BW or AMX-compute) roofline for its regime, with the
remaining gap attributed (and if it's the inner-loop/epilogue → descend to `micro-scale`).

## Pitfalls this scale exists to prevent
- Chasing the resident compute peak for a streamed GEMM (wrong ceiling).
- Re-implementing a loop-nest when oneDNN BRGEMM is already the baseline.
- Authoring a C++ kernel for a BW-bound, tiny-AI op (small RoI) — the AI-vs-ridge check catches it.
