# Future task scope: B=32 decode MoE apply — raise bandwidth utilization in the native `moe.cpp`

**Status:** SCOPED, not started. Gated behind `kernel-feasibility-gate` (preconditions below). This is the one
MoE kernel lever that survived wave-3; it is bounded and shape-specific, and it touches the INSTALLED SGLang
`moe.cpp` (upstream), not the external plugin. Do NOT author until the preconditions pass.

## 1. Problem statement (measured, wave-3)

At the warm operating point (single EMR socket, `GOMP_SPINCOUNT` warm, M-adaptive threads), the native MXFP4
`fused_experts_cpu` apply is **memory-bound** and **near its sensible ceiling at B=1 and B=64 but under-utilizes
bandwidth specifically at B=32**:

| decode shape | active experts | rows/expert | apply @32t | achieved GB/s | % of single-domain 329 GB/s |
|---|---|---|---|---|---|
| B=1  (M=1)  | 6   | 1.0 | 0.42 ms | 179 | 54% (latency-bound, fine) |
| **B=32 (M=32)** | **56** | **~2** | **7.05 ms** | **100** | **30% ← the dip** |
| B=64 (M=64) | 203 | ~3.4 | 15.9 ms | 160 | 49% |

**It is NOT thread-starved** (measured: B=32 @24/32/48 threads = 75/100/104 GB/s — plateaus past 32). More
threads do not help; the limit is intrinsic to the shape.

## 2. Root-cause hypothesis (grounded in `moe.cpp`) — UPDATED by the feasibility microbench (job 394934)

**⚠ The original "scattered access is the ceiling" hypothesis was REFUTED by the gate.** A pure AVX-512 stream of
B=32's EXACT 56-expert byte pattern achieves **235 GB/s** (T=32) / **240–255 GB/s** (T=48) — the actual MoE kernel
reaches only **100–104 GB/s**. So the scattered access is NOT the limit; there is **~2.3× recoverable headroom at
B=32** (and ~1.6× at B=64: 160 vs 252). SW prefetch and multi-expert interleave did NOT raise the pure stream
(235→237; HW prefetcher already saturates), so those levers are OUT.

**Revised root cause:** the kernel runs at 100 GB/s because its **compute does not hide in the shadow of the
weight stream** (rule 0's corollary). The gap (235 achievable stream vs 100 achieved) is the per-expert
**fp4→bf16 dequant + GEMV + A-row gather (`copy_stub`) + thread imbalance** (56 experts awkward for 32–48
threads), NOT memory bandwidth or access pattern. At ~2 rows/expert the GEMV is tiny, but the dequant processes
the FULL ~12.6 MB expert weight regardless of m_size — so if the dequant is not vectorized/overlapped enough to
clear under the 235 GB/s stream, the op runs at the (slower) dequant/compute rate. `use_brgemm` is correctly
false (vector path); this is a compute-hiding + thread-balance problem, not a BRGEMM problem.

**Pending gate step (do next, before authoring):** isolate the fp4→bf16 dequant + tiny-GEMV compute rate to
confirm the kernel is dequant/compute-bound (not stream-bound) and bound the achievable (can the compute be made
to hide under 235 GB/s, or is dequant itself the new floor?).

## 3. Candidate levers (to be screened by microbench BEFORE authoring) — REVISED post-gate

**Refuted by the gate (do NOT pursue):** cross-expert SW prefetch and multi-expert interleave — the pure stream
already hits 235 GB/s and these did not raise it (HW prefetcher saturates). The access pattern is not the limit.

**Revised levers (make the compute hide under the 235 GB/s stream):**
1. **Faster / vectorized fp4→bf16 dequant** — the dominant per-expert compute; ensure it is AMX/AVX-512-vectorized
   and fused into the GEMV epilogue (compute-from-native-precision), so dequant clears under the weight stream.
2. **Thread-balance by weight-bytes, not expert-count** — 56 experts across 32–48 threads leaves tail threads
   idle; assign experts so threads finish together (B=64's 203 experts already balance well → 1.6× the BW).
3. **Amortize/hoist per-expert setup** (tile/scale/gather) out of the inner path.
- NOT levers (ruled out): prefetch/interleave (gate-refuted), force BRGEMM/AMX tiles (underfilled at ~2 rows),
  more threads (plateaus past 32), the monolith (unrelated).

## 4. Feasibility-gate preconditions (MUST pass before authoring — per `kernel-feasibility-gate`)

1. **Counterfactual ROI bound (already estimated, confirm):** the MAX recoverable is B=32 reaching B=64's
   efficiency (~160 GB/s): apply 7.05 → ~4.4 ms = **~1.6× on the apply = ~1.4× on decode@B=32** (MoE = 72% of the
   warm step). Confirm with a counterfactual (e.g. a prefetch-only prototype) that the bound is reachable before
   committing to the full change.
2. **Microbench the closest path:** isolate `fused_experts_cpu` at the B=32 shape with real packed weights +
   perf counters (LLC-miss, mem-stall, BW); confirm the bottleneck is memory-level-parallelism / BW-saturation
   (not compute, not the `copy_stub` gather). If the counters say compute/gather, re-aim.
3. **Amdahl check at the TARGET operating point(s):** B=32 is ONE point. Confirm B=32 (and nearby mid-batch) is a
   priority operating point for the KPI before spending on a shape-specific win; B=1 and B=64 are already near
   their ceilings and would not benefit.

## 5. Cost / risk

- **Touches the INSTALLED `sglang/kernels/.../cpu/moe.cpp`, not the plugin** — an upstream change (rebuild the
  native kernel; coordinate with the SGLang CPU kernel owners for upstreaming). Heavier than any plugin work done
  so far; the external-plugin thesis does not cover it.
- **Bounded win (~1.4× decode at B=32 only).** Shape-specific; lower priority than the banked config win (~4–5×).
- **Correctness:** parity-gate the new path (fail-closed `parity_gate.py`) vs the current kernel on matched
  inputs across the occupancy distribution; bit-faithful MXFP4.

## 6. Validation plan (when/if authored)

1. Microbench the isolated apply, B∈{8,16,32,48,64}, before/after, achieved GB/s vs the 329 GB/s single-domain
   ceiling; target B=32 toward B=64's utilization.
2. Parity: fail-closed gate on the executed path, matched inputs, across the real per-expert occupancy.
3. E2E: worker-step decode measurement at the warm operating point (correct `/(output_len-1)` normalization),
   B=32 primary, confirm the apply win propagates (MoE is 72% => ~1.4× ceiling) and does not regress B=1/B=64.
4. Re-attribute after the change (Amdahl): attention/dense/glue rise as the MoE shrinks.

## 7. Recommendation — UPDATED post-gate: PURSUE (headroom confirmed real)

The feasibility gate **refuted the defer rationale**: B=32's scattered access is NOT the ceiling (pure stream
235 GB/s vs kernel 100) → **~2.3× recoverable at B=32** (~1.4× on decode@B=32, MoE=72%), and ~1.6× at B=64. The
lever is making the per-expert fp4-dequant+GEMV compute hide under the 235 GB/s stream + balancing experts across
threads — a compute/overlap/balance problem, not access-pattern. **One gate step remains** (§4.2 revised: isolate
the dequant+GEMV compute rate to confirm dequant-bound vs imbalance-bound and set the achievable). If that
confirms, author in the native `moe.cpp` (upstream). Still bounded (~1.4× decode@B=32) and upstream-touching, but
now a CONFIRMED-real win worth banking for absolute-best-performance, not a speculative one.

Evidence: wave-3 jobs 394867/868/907/908/917/918; occupancy census job 394490;
`docs/dsv4-decode-operating-point.md`; machine BW `plugin/validate/results/machine_constants_emr.json`
(numa.local_bw 328.9 GB/s).
