---
name: model-enablement-playbook
description: "Master index + decision flow for enabling a NEW model on Intel Xeon CPU in SGLang via the external plugin (throughput-thesis leg — all ops covered by existing AMX kernels). Routes a model config through an ordered, composable set of skills — decompose -> coverage-gate -> wire -> accuracy-oracle -> peer-relative-roofline -> certificate — each gated. Use this FIRST for a model-enablement task; it decides WHICH skills to load and in WHAT order, and when to hand off to the cpu-optimizer (new-kernel) leg."
---

# Model Enablement Playbook (orchestrator index)

Entry point the enablement workflow loads first. It does not itself enable a
model; it **routes** a model config through a progressive, composable skill
library, loading each only when its trigger fires and gating every step. It is the
model-level analogue of `cpu-optimization-playbook` (which operates at the kernel
level). Enablement reuses kernels; optimization writes/improves them.

## The skill library (progressive order)

| # | Skill | Load when | Gate |
|---|-------|-----------|------|
| 0 | `establish-achievable-performance` | ALWAYS first, once per node | sets ceilings |
| 1 | `model-op-decomposition` | new model config in hand | complete op graph |
| 1b | `fusion-analysis` | after decomposition | fusion plan (BW/AI, roofline-gated) |
| 1c | `model-roofline-analysis` | after fusion-analysis | per-phase Amdahl-ranked high-ROI plan |
| 1c2 | `enablement-scope-discovery` | after roofline, BEFORE first run | complete kernel-family + infra scope (dependency-closure) |
| 1d | `model-profile-hotspots` | model runnable on node | measured RoI-ranked kernel hotlist |
| 2 | `kernel-capability-registry` | after decomposition | contracts loaded |
| 3 | `coverage-gate` | graph + registry ready | 100% covered, else HAND OFF |
| 4 | `cpu-model-wiring` | coverage clean | loads + forward pass on CPU |
| 5 | `accuracy-oracle` | model runs on CPU | parity + task within tol |
| 6 | `peer-relative-roofline` | accuracy passed | ≥ absolute AND peer bar |
| 7 | `enablement-certificate` | all gates green | PR-ready report + diff |

Each skill is self-contained (trigger, inputs, procedure, gate) so new models
reuse them unmodified, and new capabilities are added as new skill folders.

## Decision flow

```
0. establish-achievable-performance  -> compute/stream/mem ceilings for THIS node
1. model-op-decomposition            -> normalized op graph
1b. fusion-analysis                   -> fusion plan: COVERED (donor fused-kernel) /
                                        NEW-FUSED-KERNEL (-> kernel-authoring) / SKIP;
                                        cross-checked vs external impls/claims
                                        (vLLM, TRT-LLM, FlashInfer, authors, blogs)
1c. model-roofline-analysis           -> per-phase (prefill/decode) roofline; Amdahl-rank
                                        op-classes; high-ROI plan (memory-bound->precision,
                                        compute-bound->AMX); gates entry to the kernel tier
1c2. enablement-scope-discovery       -> dependency-closure walk of the ACTUAL forward + backend
                                        + KV/memory-pool code to the sub-op leaf; enumerate ALL
                                        kernel families (attention+norm+routing+...) AND the
                                        runtime substrate; classify each (routing/port/authoring);
                                        honest scope BEFORE the first run (no reactive cascade)
1d. model-profile-hotspots            -> RUN the model (random weights ok); per-kernel measured
                                        time vs kernel roofline floor; RoI = share*(1-efficiency);
                                        measured hotlist wins over analytical ranking
2. kernel-capability-registry        -> load op->kernel contracts + peer donors
3. coverage-gate:
     all ops covered      -> continue
     any GAP (novel op)   -> route to cpu-optimizer (new-kernel leg); STOP here
4. cpu-model-wiring (plugin only)    -> prepack + intel_amx attn + FusedMoE CPU
5. accuracy-oracle                   -> per-layer parity, then task score
6. peer-relative-roofline            -> per-op % vs achievable AND vs peer model;
                                        end-to-end tok/s vs peer; loop on the gap
7. enablement-certificate            -> machine-checkable pass/fail + reviewable diff
```

## Composition rule

Run gates in order; a red gate blocks the next. Coverage is the fork: a genuine
gap exits this leg to the new-kernel loop rather than degrading the model. Record
the winning op→kernel mapping + measured efficiencies into the registry's
learned-patterns so the next model starts from a known-good wiring.

## Why two performance proofs (skill 6)

We reuse kernels that ALREADY hit their roofline on enabled models. So a correct
enablement must reproduce that efficiency. Absolute roofline catches "this kernel
is slow on this hardware"; peer-relative catches "this kernel is fine but OUR
wiring lost performance" (missing prepack, AVX-512 fallback where AMX was
expected, NUMA thrash). Both must pass — see `peer-relative-roofline`.

## Worked example

`olmo2` (dense GQA + QK-norm) is the first end-to-end worked enablement: all ops
map to Llama-family donor kernels; it is the template for a new dense model.
DeepSeek Flash v4.1 is deliberately NOT here — its DSA indexer is a coverage GAP
(no CPU kernel; see `coverage-gate`), so it belongs to the new-kernel leg.
