---
name: roofline-validation
description: "Use when validating whether a measured CPU kernel (GEMM, attention, MLP) achieves its performance ceiling. Computes AMX/AVX compute peak and memory-bandwidth peak from a hardware profile, the ridge point, arithmetic intensity, and reports achieved % of roofline with a pass/fail verdict. Pairs with cpu-gemm-amx-bf16."
---

# Roofline Validation

Turn a raw throughput number into a defensible verdict using the
[canonical acceptance policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).
Report idealized bounds separately from calibrated operation-specific targets.
For whole-model charts, enforce the
[matched end-to-end target contract](../../../../docs/methodology-playbook.md#matched-end-to-end-target-publication):
separate targets for each phase and deployment, matching per-replica work and shared resources.
A missing target is **pending**, not a historical line copied into the current chart.
Analytical achievable estimates are predictions from workload and hardware, not measured model
ceilings. They require no full-model run; short targeted microbenchmarks can refine assumptions.
Publish sensitivity ranges and label ratios as "measured / analytical estimate", not certified
efficiency. Do not use unmatched or hardware-only denominators as whole-model targets.

## Peak vs achievable (measure the ceiling, don't assume it)
Before judging any kernel, establish the achievable ceilings on the target
silicon with microbenchmarks (run once per hardware profile, cache in its
`achievable` block):
- **Compute**: `tools/microbench/amx_peak.c` — resident-tile TDPBF16PS loop, zero
  memory traffic → achievable AMX compute TFLOP/s (also fixes FLOP/cycle empirically).
- **Memory**: `tools/microbench/stream_triad.c` → achievable DRAM GB/s.
Example (GNR 6980P): resident compute probe = 350.8 TF/socket, STREAM = 630.7 GB/s;
paper peak compute is ~351 TF/socket and 844.8 GB/s — so AMX compute is ~100%
utilized in that probe while STREAM reaches ~75% of theoretical bandwidth.
Neither observation alone certifies a full kernel's achievable rate.

## Inputs
- A hardware profile (`hardware-profiles/<part>.yaml`) with its `achievable` block.
- Measured `achieved_tflops` (or GFLOP/s) for the kernel.
- Problem dims `M,N,K` and the actual DRAM bytes moved (or an upper bound).

## Procedure
1. `peak_compute = cores * flops_per_cycle_per_core * freq` (sanity only).
2. Record the resident probe as a reference bound; justify the operation-specific
   compute target at matching shape, dtype, locality, threads and cache conditions.
3. Record theoretical and measured bandwidth for the relevant memory level separately.
4. `ridge = compute_rate / bandwidth` (FLOP/byte), naming which bound/target is used.
5. `AI = FLOPs / bytes_moved`; `floor_time = max(FLOPs/compute_rate, bytes/bandwidth)`.
   Include non-GEMM work and additional costs explicitly where this model is incomplete.
6. `efficiency = target_time / measured_time`, equivalently measured throughput /
   target throughput. Do not silently compare mismatched roofline definitions.

## Verdict rules
- For a justified single-socket large-square-GEMM target, `efficiency >= 0.70` → threshold **PASS**.
  This does not close standalone optimization without the canonical opportunity ledger.
- `0.70 > efficiency >= 0.5` → **MARGINAL**: revisit blocking/threads/NUMA.
- `efficiency < 0.5` → threshold **FAIL**; investigate the target and implementation,
  rather than assigning a cause from the ratio.
- Non-finite/nonpositive inputs → **INVALID EVIDENCE**.
- `efficiency > 1.0` → **TARGET/MODEL REVIEW REQUIRED**: check cache residency,
  work/byte counts, timing, frequency and whether a provisional baseline was surpassed.
  Do not clamp or silently certify.

## A low % is a tested hypothesis, not a verdict (a bound is not a wall)
A sub-ceiling efficiency is **not** automatically a failure *or* a wall — it is a hypothesis to test before
you either keep optimizing or declare "at ceiling":
1. **Wrong ceiling first (operative, not nominal).** If the op is selection / dispatch / recurrence /
   cache-resident / small-GEMM, the byte-or-FLOP roofline is **not** the operative bound. Compute the
   **operative floor** (what is achievable given the op's regime) and report % of *that*. An op reading at the
   bare `memcpy`/`sum` bandwidth, or a small-batch GEMM at the library-microkernel floor, is **at its real
   ceiling** even at a low % of the nominal roof — the low % is a diagnostic, not owed work. Decompose per
   component (data-movement against **operative** BW, compute against calibrated shape/dtype targets,
   overhead measured and minimized); the roofline-% of
   the *total* just names the dominant component.
2. **Config sweep before "at ceiling."** A measured bound prompts a **sweep** over tiles / grain /
   effective-threads (via `grain_size`/task-count — **never per-forward `set_num_threads`**, an affinity
   cliff). If the % rises, the bound was *config* (fix it; ship the per-M optimum). If the sweep moves nothing,
   retain a bounded "best demonstrated" claim; a finite sweep does not prove a fundamental limit.
3. **Knob before closing.** Where compute dominates, test the **precision** lever (bf16/fp8 — may win *or*
   lose; publish either); where measurement localizes the bottleneck **inside the donor's inner loop**,
   consider **authoring your own contraction**. Closing on assertion is the anti-pattern; a measured loss is a
   first-class, published disposition.

## Anti-pitfalls
- Never use max turbo for the compute ceiling; AMX throttles the all-core clock.
- Confirm the intended ISA actually ran (e.g. `ONEDNN_VERBOSE=1` showing
  `amx` kernels) before trusting a high number.
- Report the ceiling assumptions (freq, FLOP/cycle) alongside the % so the
  claim is auditable.
