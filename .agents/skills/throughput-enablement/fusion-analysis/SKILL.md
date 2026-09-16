---
name: fusion-analysis
description: "Graph-level operator-fusion analysis — the BW/AI performance lever that sits BETWEEN model-op-decomposition and coverage-gate. Scans a model's op graph, identifies fusion opportunities (vertical producer→consumer, horizontal sibling branches, epilogue/prologue), quantifies the benefit on the roofline (DRAM round-trips removed, effective arithmetic-intensity lift), and maps each viable fusion to a DONOR fused-kernel (covered) or a NEW fused-kernel task (→ kernel-authoring) or skip. Uses external implementations of the SAME model — vLLM, TensorRT-LLM, FlashInfer, the authors' reference kernels, and day-0 blogs/PRs — as an idea source + numerical cross-check (code-backed = oracle; claim-only = hint), with the caveat that fusion economics are not 1:1 across hardware/frameworks. Load after decomposition, before coverage."
---

# Fusion Analysis (graph-level BW/AI lever)

Kernel optimization tunes ONE op; fusion changes the GRAPH. For memory-bound
decode, fusing adjacent ops to raise effective arithmetic intensity (AI) often
beats micro-optimizing a single kernel. This skill makes fusion an explicit,
measured stage — not an afterthought — and routes each opportunity to a donor
fused-kernel or a new-kernel task.

## Pipeline position
`model-op-decomposition → fusion-analysis → coverage-gate → (cpu-model-wiring |
kernel-authoring)`. Consumes the normalized op graph + the roofline ceilings;
emits a **fusion plan** (per fusion: kind, benefit, mapping).

## Fusion taxonomy
- **Vertical (producer→consumer):** GEMM+bias+activation, norm→projection,
  RoPE→attention, quant→GEMM, residual-add+norm. Fuse into the producer epilogue
  or consumer prologue to skip the intermediate DRAM store+load.
- **Horizontal (sibling branches sharing an input):** q/k/v → fused QKV,
  gate/up → fused gate_up. One weight-packed GEMM instead of N.
- **Epilogue/prologue element-wise:** `SiLU(C0)*C1` (MoE), dequant, KV-quant-on-write,
  softmax-scale — emitted in the GEMM store/load step.

## Pattern library (grounded in the SGLang CPU corpus)
| Candidate | Fuse into | Donor already does it |
|---|---|---|
| gate/up + SiLU + mul | MoE W1 store step | `moe.cpp` (SiLU(C0)*C1) |
| q/k/v projection | fused QKV GEMM | `qkv_proj.cpp` |
| RMSNorm + affine (weight/bias) | norm streaming loop | `norm.cpp` (two-pass FP32) |
| activation + elementwise mul | one streaming kernel | `activation.cpp` |
| KV quant (fp8/int8) + cache write | write kernel | `kvcache.cpp` |
| int8/fp8 dequant + scale | GEMM epilogue | `gemm_{int8,fp8}.cpp` |
Anything NOT in this table with a real benefit is a **new-fused-kernel** candidate.

## Benefit model (roofline-gated — never fuse on faith)
For each candidate estimate:
- `bytes_saved` = size of the removed intermediate × (1 store + 1 load).
- `AI_before → AI_after` = FLOPs / (bytes after fusion); does it cross the ridge
  (memory-bound → compute-bound)?
- **Gate: fuse only if** it raises AI on a **memory-bound** op AND the fused
  intermediate/working set stays **L2-resident**. Do NOT fuse when the intermediate
  is position-indexed (RoPE cache), reused by an unrelated op, or spills L2 — the
  corpus keeps GEMMs separate exactly there (`qkv_proj.cpp` staged, not monolithic).
- Compute-bound GEMMs gain little from fusing an element-wise epilogue (already
  bandwidth-cheap) — prioritize the memory-bound tail.

## External-reference cross-check (idea source + numerical oracle)
Do NOT restrict to SGLang. When a model is hot, many groups race to showcase it and
fusion levers surface all over the ecosystem. Use ANY external reference as an AID:
1. **Where to look (broad):** the model's SGLang CUDA path; **vLLM**; TensorRT-LLM;
   LMDeploy; MLC-LLM; FlashInfer / FlashAttention; TransformerEngine / Apex fused ops;
   the model authors' reference kernels + **technical report / model card**; and day-0
   **performance write-ups, blogs, and GitHub PRs** racing to enable the model.
2. **Two kinds of evidence, treated differently:**
   - **Runnable fused code** (any framework/HW) → the fusion is semantically valid; use
     its output as a **numerical oracle** to cross-verify the CPU port.
   - **A performance CLAIM without code** (blog, tweet, benchmark table) → a HINT to
     investigate, NOT a verified fusion. Confirm it exists in real code first, then
     re-derive the benefit on the CPU roofline. Claims can be marketing.
3. **Absence anywhere is NOT a reason to skip** — CPU is more bandwidth-bound, so a
   CPU-only fusion may still pay; create your own plan and prove it on the roofline.
4. **Never port 1:1:** fusion economics differ across hardware (shared-mem/warp/
   tensor-core vs AMX+L2+OpenMP) AND across frameworks (graph capture, launch cost).
   External references tell you *what is fusible and correct*; the CPU roofline tells
   you *whether it pays here*.

## Mapping / routing (the output)
Per viable fusion, emit one of:
- **COVERED** — a donor fused-kernel already implements it → mark for `coverage-gate`.
- **NEW-FUSED-KERNEL** — beneficial, no donor → hand a task to `kernel-authoring`
  (donor to adapt + the 4-layer plan; the epilogue/gather-fuse patterns apply).
- **SKIP** — benefit below threshold or violates the fusion-boundary gate; record why.

## Procedure
1. Take the op graph; enumerate adjacent op-groups by the taxonomy.
2. For each, estimate bytes_saved + AI lift; apply the roofline gate.
3. Cross-reference external implementations + performance claims (vLLM, TRT-LLM,
   FlashInfer, authors' report, blogs, PRs): code → numerical oracle; claim → hint.
4. Route: COVERED / NEW-FUSED-KERNEL / SKIP.
5. Emit the fusion plan; feed COVERED to coverage-gate, NEW to kernel-authoring.

## Gate
A fusion is in the plan only with a quantified benefit (bytes_saved + AI lift) that
clears the roofline gate, a resident-working-set check, and — when a GPU reference
exists — a numerical cross-check target. No hand-wave fusions; each carries its
measured justification so the certificate can audit it.
