---
name: kernel-feasibility-gate
description: "Use before standalone kernel optimization to review the roofline and measure a source-faithful baseline on the target node. Mandatory functional bring-up may precede a runnable-model profile. ROI prioritizes work but never exempts a required kernel from native C/C++ optimization; user review remains required."
---

# Kernel Feasibility Gate (roofline-first + baseline microbench)

Follow the [canonical acceptance policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).
Before optimization, require a user-reviewed roofline and a measured source-faithful baseline.
A missing functional path may be authored for bring-up first; it is not optimized acceptance.
Origin lesson: the DSA `index_gemm_kernel_nn` M=16 AMX staging proposal targeted the wrong
resource. Measure whether the proposed lever addresses the bottleneck.

> **⛔ PER-KERNEL PHASE-1 TARGET + NO-REGRESSION FLOOR (the acceptance bar for "kernel done").** A
> kernel is Phase-1 done ONLY when, across the FULL M-sweep AND the KV-cache knob (S / topk / page
> length): (a) it is optimized as close to its roofline as the regime allows (publish per-op
> roofline-target-vs-measured), AND (b) it meets a HARD **no-regression floor vs the torch fallback at
> EVERY M** — faster, or at worst a TIE. Small M (M=1/M=8) MAY not beat torch (acceptable), but it must
> NEVER be slower; if it would be, DISPATCH to a validated native C/C++ library path at that M
> so the floor always holds. Correctness is held by the per-op acceptance gate (tie_eps=0) throughout.

## Trigger
Every new/PARTIAL kernel from `coverage-gate` and each optimization/fusion proposal.
Use analytical ranking first and a representative model profile once runnable.
If the model cannot run, measure an isolated source-conformed reference or minimal
bring-up implementation; report end-to-end share as unknown until measured.
Neither low ROI nor missing full-model profiling waives standalone optimization.

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
   AMX can help a memory-bound op if compute/feeding overhead prevents bandwidth saturation;
   verify the effect rather than requiring or rejecting AMX by regime alone. Traffic-reduction levers:
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
     `quantization-amx-int8`). Keep weights packed in production, with bounded per-tile
     conversion inside the kernel. Lossless re-encoding does not waive kernel conformance.
Present this to the user as a short table + verdict. Do not proceed silently.

## Step 2 — Baseline microbenchmark (measure, don't trust the model alone)
On the TARGET node, measure the closest existing kernel/path with perf counters:
0. **First get the `overhead-attribution` bucket for this op** (kernel/torch/framework). If it
   is tagged **kernel but far below its roofline floor** (like Flash `fused_experts` at
   ~1.9 s/call, ~1000× above floor), the answer is **kernel-ISOLATION** (threads / weight
   prepack / dtype path / verify ISA dispatched) first. If native implementation costs remain,
   evaluate co-design/fusion; an existing C/C++ kernel is not automatically optimized.
1. Reproduce the op's shape + access pattern (or call the existing kernel directly).
2. `perf stat` for cycles, IPC, LLC/L2 misses, and DRAM bytes → classify the
   bottleneck: **DRAM-BW / L2-BW / port-issue / conversion**. This disambiguates
   what the roofline could only bracket, and picks the design (e.g. AVX-512
   single-pass vs AMX-tiles).
3. Record the measured **achievable** for this op class as the authoring target.

## Step 3 — Amdahl check
Measure/estimate the kernel's share of end-to-end (e.g. decode) time. A 3× kernel
that is 3% of decode is ~2% end-to-end. Use this to order work, not to skip a required kernel.

## Step 4 — Go / No-Go (record as an audit artifact)
AUTHOR the optimization when the intended lever addresses the measured bottleneck and
the user approved the roofline. Reject a particular ineffective design with evidence or
revise it; do not remove the kernel from scope. Complete the canonical opportunity ledger,
including fusion, even for low-share kernels. Emit
`{op, roofline table, measured baseline + bottleneck, Amdahl share, target TF/s,
decision}` — the certificate/PR references it.

## Gate
Optimization does not begin until this artifact has (a) a user-reviewed roofline and
(b) a measured baseline on the target node. A minimal functional path can precede
this gate solely to establish that baseline. It must subsequently pass the same
standalone native-code, correctness, optimization and review requirements.
