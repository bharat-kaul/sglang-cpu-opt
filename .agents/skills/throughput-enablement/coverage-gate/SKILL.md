---
name: coverage-gate
description: "Use after model-op-decomposition + kernel-capability-registry to decide whether a model is enablable on CPU with EXISTING kernels only. Matches every op in the graph against the registry's capability contracts and returns COVERED / COVERED-WITH-FALLBACK / GAP per op. A single GAP (a novel op with no CPU kernel, or a contract mismatch) exits the throughput leg and routes that op to the cpu-optimizer (new-kernel) leg. This gate is what makes the throughput-thesis claim honest — it refuses to dense-approximate a novel op just to make the demo pass."
---

# Coverage Gate

The fork in the workflow. Decides, deterministically and auditably, whether a
model belongs to the throughput leg (all ops covered) or the new-kernel leg (any
gap). Being strict here is the whole point: an over-eager "covered" that silently
approximates a novel op produces a model that fails accuracy or defeats its own
design.

## Inputs
- The normalized op graph (`model-op-decomposition`).
- The `enablement-scope-discovery` closure manifest (all kernel families across every
  subsystem + the runtime substrate, classified) — this is what makes the gate see the
  gaps a curated op-scan misses (a second family like MHC, the infra layer).
- The loaded `kernel-capability-registry` (`assets/registry.yaml`) for this node.

## Matching procedure
For each op in the graph:
1. Find the registry entry with the same canonical `op`. None → **GAP**.
2. Check EVERY `provides` field against the op signature:
   - dtype ∈ provided dtypes,
   - GEMM dims satisfy `divisibility` (`OC%16==0`, `IC%32==0`) — else GAP unless a
     padding fallback is registered,
   - attention_kind / mask ∈ provided,
   - pos scaling / routing ∈ provided.
   Any unsatisfied field → **GAP** (record which field failed).
3. If satisfied via a documented slower path (e.g. AVX-512 fallback where AMX ISA
   is absent, or padding to divisibility) → **COVERED-WITH-FALLBACK** (record the
   expected efficiency haircut so `peer-relative-roofline` uses the right bar).
4. Else → **COVERED**.
Also cross-check the graph against `known_gaps`: any match is an immediate GAP with
its `leg: new-kernel` routing.

## Output contract
A per-op verdict table + one overall verdict:
- **ALL COVERED** (fallbacks allowed) → proceed to `cpu-model-wiring`.
- **ANY GAP** → STOP the throughput leg. Emit a scaffolding task for each gap op
  (op name, failed contract field, suggested donor kernel to adapt, target
  shapes) and hand it to the `cpu-optimizer` agent. The model re-enters this gate
  once the new kernel registers a contract.

## Worked outcomes
- **OLMo 2 / OLMoE / Arcee** → ALL COVERED: dense_gemm, gqa_attention, rms_norm
  (incl. QK-norm), rope, swiglu, and (OLMoE) moe_grouped_gemm all have donors.
  Proceed to wiring — the hours-scale path.
- **DeepSeek Flash v4.1** → GAP on `dsa_sparse_attention` (+ `fp8_per_token_group_
  quant_cpu`): no CPU kernel. Route the indexer/compressor to `cpu-optimizer`; the
  rest of the model (MLA, MoE) is already covered, so the scaffolding task is
  tightly bounded to the one missing kernel. **BUT** — empirical bring-up later
  surfaced an ENTIRE gap category the op-scan could not see (KV-pool layout +
  attention-backend compat guard); see below. The compute-op verdict was right; the
  *scope* was not complete until the model was actually stood up.

## Static coverage is necessary but NOT sufficient — the infra layer
The op-scan matches COMPUTE ops (GEMM, attention, norm, MoE, …). But a model does
not run on compute ops alone; it runs on a **runtime substrate** the op graph does
not contain, and that substrate has its own CPU gaps:
- **KV-cache / memory-pool layout** (packed byte layouts, store dtype, paged/SWA
  pools, per-arch accessors) — e.g. DSV4's 584 B/token uint8 fp8+bf16 packed pool.
- **Attention-backend selection + compatibility guards** — a backend the arch forces
  (e.g. `dsv4`) vs. a guard that demands another (e.g. "fp8 KV ⇒ intel_amx only").
- **Quantization plumbing** — weight_scale_inv containers, block-quant asserts, KV
  quant methods that exist only for CUDA/one backend.
- **Device predicates / config schema** — `is_cpu()` gates (`SGLANG_USE_CPU_ENGINE`),
  a framework-native config schema that differs from the model's dataclass.
These are **invisible to op-decomposition** and only surface when you INSTANTIATE +
LOAD + build the KV pool + run a forward. So the gate's verdict must be paired with
an empirical **make-it-work bring-up** (see `cpu-model-wiring`), and the enablement
SCOPE = compute-op gaps (this table) **PLUS** infra gaps (found only by bring-up).
Record infra gaps alongside the op table; they often dominate the make-it-work cost.

## Gate
Emit the verdict table with, for every op, the matched kernel + donor OR the
failed contract field. A verdict without per-op evidence is not acceptable — the
certificate depends on this table.

## Pitfalls
- "Same family, must be fine" is how gaps slip through. Match on the op SIGNATURE,
  not the model name.
- A fallback is COVERED-WITH-FALLBACK, not COVERED — it carries a performance
  expectation change that the perf gate must know about.
- Divisibility failures are common on odd head_dims / intermediate sizes; prefer a
  registered padding fallback over silently skipping the check.
- **Declaring scope from the op table alone.** The op-scan cannot see infra gaps
  (KV-pool/backend/quant/device-gate). "Bounded to N kernels" is provisional until
  an end-to-end bring-up confirms nothing else blocks execution. Gaps hide behind
  gaps — each fix reveals the next — so the real scope is known only after the model
  runs, not after the scan.
- **Assuming ONE novel kernel family.** A model can carry SEVERAL independent novel
  families in DIFFERENT subsystems — scan the WHOLE forward (attention, norm/layernorm,
  routing, MoE), not just the headline op. DeepSeek-V4 has TWO: DSA sparse attention
  (compressor/indexer/sparse-prefill) AND MHC hash-clustering in the norm path
  (mhc_pre / hc_split_sinkhorn / hc_combine / mhc_post). A scan that stops at the
  attention decomposition misses MHC entirely. Each family often has a partial torch
  outer op but accelerator-only INNER sub-ops — enumerate to the sub-op leaf.
