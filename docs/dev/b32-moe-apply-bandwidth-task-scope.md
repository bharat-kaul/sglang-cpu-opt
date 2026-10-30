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

## 2. Root-cause hypothesis (grounded in `moe.cpp`)

`fused_experts_kernel_impl` parallelizes via `parallel_2d(MB, NB, ...)` over M-blocks (token groups, `BLOCK_M`)
× N-blocks, and each active expert contributes ≈ 1 M-block at B=32 (m_size ≈ 1–2 << BLOCK_M). Consequences at
B=32 (56 experts):
- **Low memory-level parallelism.** 56 experts => 56 M-blocks; across 32–48 threads that is ~1–2 blocks/thread,
  and each block streams ONE expert's full packed weights (~12.6 MB: `w1 [2N,K]` + `w2 [N,K]`) from a distinct,
  strided region of the `[E, …]` weight tensor. Few concurrent weight streams + strided access => DRAM BW is
  under-saturated (B=64's 203 blocks keep far more requests in flight => 1.6× the BW).
- **Thread imbalance.** 56 (or 57/58) active experts is an awkward count for 32–48 threads => tail threads idle.
- **Per-expert fixed overhead poorly amortized.** For each M-block: gather `m_size` A-rows (`copy_stub`), tile/
  scale setup, dequant-scale load — all paid per expert, amortized over only ~2 rows of real work.
- (`use_brgemm` is globally false here since `avg_M = M*topk/E ≈ 1`; the vector path is correct — this is NOT a
  "force BRGEMM" task. See the occupancy census: 56% of B=32 experts have 1 row.)

Net: at B=32 the kernel does not keep enough independent weight streams in flight to saturate the memory system,
and pays per-expert setup on tiny row counts.

## 3. Candidate levers (to be screened by microbench BEFORE authoring)

1. **Cross-expert software prefetch / double-buffering** — prefetch expert k+1's packed weights while the GEMV
   for expert k runs (raise memory-level parallelism without more threads). Highest-leverage, lowest-risk.
2. **Multiple experts in flight per thread** — interleave 2–4 experts' weight streams per thread so the memory
   subsystem sees more concurrent requests (saturate BW at low rows/expert).
3. **Block/expert-to-thread balancing** — assign experts to threads by weight-bytes (not count) so 32–48 threads
   finish together; split nothing (rows too few), just balance the stream assignment.
4. **Amortize per-expert setup** — hoist dequant-scale/tile setup out of the per-block inner path where possible.
- NOT a lever (ruled out wave-3): forcing BRGEMM/AMX (tile underfilled at ~2 rows); adding threads (plateaus);
  the monolith (unrelated — this is intra-kernel).

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

## 7. Recommendation

**Defer** unless B=32/mid-batch is a declared priority operating point. The decode campaign's banked result is the
warm + M-adaptive-threads config (~4–5× over naive). This MoE lever is a real but **bounded ~1.4×-at-B=32**
follow-on that requires upstream `moe.cpp` work; schedule it only after the feasibility-gate preconditions (§4)
confirm the operating-point priority and the memory-level-parallelism root cause with perf counters.

Evidence: wave-3 jobs 394867/868/907/908/917/918; occupancy census job 394490;
`docs/dsv4-decode-operating-point.md`; machine BW `plugin/validate/results/machine_constants_emr.json`
(numa.local_bw 328.9 GB/s).
