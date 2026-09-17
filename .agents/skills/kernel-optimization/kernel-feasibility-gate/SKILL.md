---
name: kernel-feasibility-gate
description: "MANDATORY go/no-go gate that runs BEFORE authoring any new kernel (new-kernel leg). Enforces two audits with the user in the loop: (1) a ROOFLINE MODEL walk-through — FLOPs, real byte traffic (stream vs gather), arithmetic intensity, the CORRECT ceilings for the actual compute dtype/ISA (never assume AMX), ridge, and regime (compute vs memory bound); and (2) a BASELINE MICROBENCHMARK of the closest existing kernel/path on the target node with perf counters, to confirm the true bottleneck and set the achievable target, plus an Amdahl check of the kernel's share of end-to-end time. Blocks kernel-authoring until the roofline is user-reviewed and the baseline is measured. Prevents building the wrong kernel (e.g. applying a compute lever to a memory-bound op)."
---

# Kernel Feasibility Gate (roofline-first + baseline microbench)

The most expensive mistake in the new-kernel leg is writing a fast kernel that
doesn't move the bottleneck. This gate runs BEFORE `kernel-authoring` and refuses
to let authoring start until the math and a measurement agree — with the user
signing off on the roofline. Origin lesson: the DSA `index_gemm_kernel_nn` M=16
"AMX" plan was memory/gather-bound FP32 FMA, so AMX (a compute lever) + staging
would have been slower — the roofline caught it before a line was written.

## Trigger
A candidate new/PARTIAL kernel from `coverage-gate` or `fusion-analysis`, before
any implementation — AND only after the model tiers have ranked its op-class ABOVE
the ROI threshold: `model-roofline-analysis` (analytical) then confirmed by
`model-profile-hotspots` (measured share AND headroom below its roofline). This gate
is the kernel tier: it confirms per-kernel what the model tiers flagged as worth doing.

## Step 1 — Roofline model (WALK THE USER THROUGH IT)
Read the donor/reference kernel to get the EXACT compute + access pattern, then:
1. **FLOPs** = the real math (e.g. `2·M·N·K`).
2. **Bytes** = the real traffic, distinguishing **streamed** vs **gathered** (random
   access wastes cache lines) vs **resident/reused** operands; count re-streaming
   caused by the current blocking (e.g. `BLOCK_M=4` re-reads B `M/4` times).
3. **Arithmetic intensity** = FLOPs / bytes (do it for BOTH the current blocking and
   the proposed one — the delta is often the real lever).
4. **Ceilings for the ACTUAL compute** — match the dtype/ISA the kernel really uses.
   Take them from **uArch Performance Probe `machine_constants.json`** (`compute_peak` per
   dtype, `memory` DRAM + per-domain BW, `roofline_ridge`) — measured + self-validated on the
   target node, not assumed. FP32 FMA ≈ 64 FLOP/cyc/core; bf16 dpbf16 ~2×; AMX ~16×; memory =
   achievable (gather-effective, not paper STREAM). **Never assume the AMX/bf16 ceiling for an
   FP32 or memory-bound kernel.** (If no `machine_constants.json` exists for this node, run uPP
   first — an un-characterized node is a gate failure.)
5. **Ridge + regime** — compute-bound (AI≥ridge) or memory/gather-bound (AI<ridge). Use the
   measured `roofline_ridge` from uPP for this dtype.
6. **Lever check** — does the intended optimization address the ACTUAL bottleneck?
   Compute lever (AMX) on a memory-bound op = red flag. Traffic-reduction levers:
   single-pass, convert-once, cache-block, prepack, AND **reduce operand precision**.
7. **Precision as a data-movement lever (ALWAYS evaluate, both regimes).**
   - **Memory/BW-bound op:** cutting the streamed/gathered operand's STORAGE precision
     (bf16→fp8/int8, or int4) directly halves/quarters its bytes → usually the primary
     lever; it raises effective throughput even with unchanged compute math.
   - **Compute-bound op:** still consider it — (a) low-precision STORAGE + upconvert in
     the microkernel cuts operand-FEED traffic (helps when the "compute-bound" op is
     actually operand-feed-limited, i.e. weight-streamed inference GEMM), and (b)
     low-precision COMPUTE (int8/AMX) raises the compute ceiling itself (~2x).
   - **Caveats:** count the up-convert/dequant cost — it can BECOME the bottleneck
     (the DSA fp8→bf16→fp32 path is a live example); and every precision drop spends
     accuracy budget → gate with a per-precision accuracy check (`accuracy-oracle` /
     `quantization-amx-int8`). This is the SAME lever the corpus already uses (fp8/int4
     weights stored packed, upcast at load — see kernel-authoring reduced-precision cheatsheet).
Present this to the user as a short table + verdict. Do not proceed silently.

## Step 2 — Baseline microbenchmark (measure, don't trust the model alone)
On the TARGET node, measure the closest existing kernel/path with perf counters:
1. Reproduce the op's shape + access pattern (or call the existing kernel directly).
2. `perf stat` for cycles, IPC, LLC/L2 misses, and DRAM bytes → classify the
   bottleneck: **DRAM-BW / L2-BW / port-issue / conversion**. This disambiguates
   what the roofline could only bracket, and picks the design (e.g. AVX-512
   single-pass vs AMX-tiles).
3. Record the measured **achievable** for this op class as the authoring target.

## Step 3 — Amdahl check
Measure/estimate the kernel's share of end-to-end (e.g. decode) time. A 3× kernel
that is 3% of decode is ~2% end-to-end. Cheap kernels with tiny shares → **SKIP**.

## Step 4 — Go / No-Go (record as an audit artifact)
AUTHOR only if ALL hold: the intended lever addresses the measured bottleneck; the
projected achievable gain × Amdahl share clears the agreed threshold; and the user
approved the roofline. Otherwise **SKIP** (record why) or revise the design. Emit
`{op, roofline table, measured baseline + bottleneck, Amdahl share, target TF/s,
decision}` — the certificate/PR references it.

## Gate
`kernel-authoring` does NOT begin until this artifact exists with (a) a
user-reviewed roofline and (b) a measured baseline on the target node. No kernel is
built on the model alone; no kernel is built without confirming it moves the
bottleneck and matters end-to-end.
