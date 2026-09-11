---
name: peer-relative-roofline
description: "The performance gate for model enablement. Proves the newly-enabled model's performance TWICE: (1) ABSOLUTE — each hot op's % of the achievable roofline ceiling (via roofline-validation), and (2) PEER-RELATIVE — the same op runs the IDENTICAL kernel that already powers an enabled donor model (Llama/Qwen-MoE/DeepSeek), so it must reach that donor's measured efficiency within tolerance on matching shapes. Also compares normalized end-to-end throughput to the peer. A pass-absolute/fail-relative split localizes a wiring bug (missing prepack, AVX-512 fallback, NUMA thrash) rather than a kernel limit. Use after accuracy-oracle passes."
---

# Peer-Relative Roofline

Absolute roofline alone is necessary but not sufficient for an enablement claim.
Because we reuse kernels that ALREADY hit their roofline on enabled models, a
correct enablement must REPRODUCE that efficiency. This gate adds the missing
half: measure the new model against the donor that shares its kernel.

## Why two proofs
| Proof | Answers | Catches |
|-------|---------|---------|
| Absolute (roofline-validation) | is this kernel near the silicon ceiling? | a genuinely slow kernel / wrong ceiling |
| Peer-relative (this skill) | did OUR wiring preserve the donor's efficiency? | missing prepack, AVX-512 fallback, NUMA thrash, under-threading |

They are independent: a kernel can be near-peak in isolation yet lose 30% in our
model because we forgot to prepack. Only the peer comparison exposes that.

## Inputs
- The node's `achievable` ceilings (`establish-achievable-performance`).
- Per-op donor efficiencies from `kernel-capability-registry`
  (`assets/registry.yaml`) + `assets/peer-baselines.yaml` — each MUST be measured
  on THIS node (else re-run the donor's roofline first).
- Measured per-op throughput for the new model at matching shapes.

## Procedure
1. **Confirm dispatch.** `ONEDNN_VERBOSE=1` (or profile) shows the intended `amx`
   kernel for each hot op — not a silent AVX-512 fallback. A dispatch miss is an
   automatic FAIL regardless of the number.
2. **Absolute.** For each hot op run `roofline-validation` → `eff_abs = achieved /
   applicable_achievable_ceiling`.
3. **Peer-relative.** Pick the donor op at the SAME shape family (M,N,K / heads /
   expert dims). `eff_rel = eff_abs / donor_efficiency`. Matching shapes is
   essential — compare a dense GEMM to the donor's dense GEMM at the same dims, a
   grouped MoE GEMM to the donor's MoE GEMM.
4. **End-to-end.** Compare normalized serving throughput to a peer model of
   comparable shape: tokens/s per active GFLOP (prefill and decode separately, at
   matched batch/seq). This catches losses that hide between ops (scheduling,
   layout conversions, comms).
5. Aggregate into the verdict table (per op: eff_abs, donor_eff, eff_rel, dispatch
   ok?) + the end-to-end normalized ratios.

## Verdict rules
Per hot op, PASS requires BOTH:
- `eff_abs ≥ 0.70` of the applicable achievable ceiling (single socket; adjust for
  documented fallbacks) — use the ceiling for THAT precision: BF16 gates against
  the bf16 achievable, INT8 (w8a8) against the INT8 ceiling (~2x,
  `amx_int8_ops_per_cycle_per_core`), AND
- `eff_rel ≥ 0.90` vs a donor running the identical kernel AT THE SAME PRECISION
  (bf16 model vs bf16 donor, int8 model vs int8 donor).
End-to-end PASS: normalized prefill AND decode throughput ≥ 0.90 of the peer.
- Any op `eff_rel < 0.90` while `eff_abs` is high → **WIRING BUG**: check prepack
  first, then ISA dispatch, then NUMA/thread binding — return to `cpu-model-wiring`.
- Any op `eff_abs < 0.70` with `eff_rel ≈ 1.0` → the donor is ALSO below the
  ceiling: this is a kernel-optimization opportunity (route to `cpu-optimizer`),
  not an enablement defect — record it, do not block the enablement on it.
- `eff_rel > 1.0` materially → suspect a stale/cross-node donor baseline; re-measure
  the donor before claiming a win.

## Gate
Emit the per-op table + end-to-end ratios with the ceiling and donor assumptions
(freq, FLOP/cycle, donor model + shape) alongside every number so the claim is
auditable. Enablement is performance-proven only when both proofs pass for every
hot op and end-to-end.

## Why this is the right bar for the throughput thesis
The thesis is "existing hand-optimized kernels deliver performance on new models
automatically." The honest test of that is: does the new model reach the SAME
efficiency those kernels already deliver on shipped models? Peer-relative is
exactly that test — it turns "we hit some roofline %" into "we hit the same % the
proven models do, on the same silicon, with the same kernels."
