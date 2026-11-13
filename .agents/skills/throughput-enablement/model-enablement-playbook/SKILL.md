---
name: model-enablement-playbook
description: "Master index + decision flow for enabling a NEW model on Intel Xeon CPU in SGLang via the external plugin (throughput-thesis leg — all ops covered by existing AMX kernels). Routes a model config through an ordered, composable set of skills — FRAME-THE-KPI (ask the user the target operating point S/B/G + headline metric, throughput-vs-latency, FIRST) -> decompose -> coverage-gate -> wire -> accuracy-oracle -> peer-relative-roofline -> certificate — each gated. Use this FIRST for a model-enablement task; it decides WHICH skills to load and in WHAT order, and when to hand off to the cpu-optimizer (new-kernel) leg."
---

# Model Enablement Playbook (orchestrator index)

Entry point the enablement workflow loads first. It does not itself enable a
model; it **routes** a model config through a progressive, composable skill
library, loading each only when its trigger fires and gating every step. It is the
model-level analogue of `cpu-optimization-playbook` (which operates at the kernel
level). Enablement reuses kernels; optimization writes/improves them.

## The skill library (progressive order)

Use the [canonical acceptance policy](../../../../docs/methodology-playbook.md#canonical-acceptance-policy).
Reference bring-up may precede optimization; production completion requires every standalone
kernel's native C/C++ acceptance, regardless of its end-to-end share.

At **each** phase exit, run the evidence-bound adversarial self-audit and
[executable prefix gate](../enablement-certificate/SKILL.md#executable-phase-exits-and-final-certificate).
Keep the Xeon request-batch **1..64** goal explicit (B1 latency plus declared throughput points).
Missing evidence or an unresolved finding blocks progression; self-audit never replaces human approval.

Use the [local recovery runner](../enablement-certificate/SKILL.md#local-failure-restart-and-approval-handling)
for automated phase replay: no blind retry loops, no duplicate owner, no reuse of partial PASS.
Pause for missing/rejected approval. Changed evidence requires renewed approval, not an edited
journal. Validate Slurm-specific recovery during the model acceptance rehearsal.

When the reference needs TP>1, use the
[shared deterministic checkpoint oracle](../accuracy-oracle/SKILL.md#tp-consistent-shared-checkpoint-path).
Do not use rank-local dummy RNG or compare only shared capture keys. Require matched checkpoint,
request/numerical settings and complete layer/logit evidence before accepting model parity.

| # | Skill | Load when | Gate |
|---|-------|-----------|------|
| 0a | **FRAME THE KPI (see below)** | ALWAYS first, before ceilings | user-confirmed operating point + headline metric |
| 0b | **[Startup infrastructure gate](../../../../docs/methodology-playbook.md#mandatory-startup-infrastructure-gate)** | before selecting resources or launching probes | timestamped CPU/GPU/scheduler/storage/software inventory + budgeted execution plan |
| 0 | `establish-achievable-performance` | after KPI and infrastructure preflight | sets ceilings FOR THAT operating point on allocated, verified resources |
| 1 | `model-op-decomposition` | new model config in hand | complete op graph |
| 1b | `fusion-analysis` | after decomposition | fusion plan (BW/AI, roofline-gated) |
| 1c | `model-roofline-analysis` | after fusion-analysis | per-phase Amdahl-ranked high-ROI plan |
| 1c2 | `enablement-scope-discovery` | after roofline, BEFORE first run | complete kernel-family + infra scope (dependency-closure) |
| 2 | `kernel-capability-registry` | after decomposition | contracts loaded |
| 3 | `coverage-gate` | graph + registry ready | enumerate native coverage and functional gaps; hand gaps to new-kernel leg |
| 4 | `cpu-model-wiring` / `cpu-serving-integration` | scope known | source-faithful reference bring-up; temporary fallback is not completion |
| 5a | `accuracy-oracle` | reference path runnable | bring-up parity, then full-shape/depth parity before optimization |
| 5b | `model-profile-hotspots` | representative model runnable | measured priority; no low-ROI exemptions |
| 5c | `cpu-optimization-playbook` | baseline available | every standalone kernel accepted, including fusion exploration; human review |
| 5d | `cpu-serving-integration` / `accuracy-oracle` | standalone gate reviewed | native integration, actual dispatch and real-weight numerical/task checks |
| 6 | `peer-relative-roofline` | accuracy passed | ≥ absolute AND peer bar |
| 7 | `enablement-certificate` | all gates green | PR-ready report + diff |

Each skill is self-contained (trigger, inputs, procedure, gate) so new models
reuse them unmodified, and new capabilities are added as new skill folders.

## Residual-gap checkpoint (after the model runs correctly end to end)

Do not let the cheap bring-up ladder become an indefinite single-example debug
loop. When a remaining task gap is small, a reference changes, or an intervention
improves tensor agreement without a broad answer gain, route back to
`accuracy-oracle` for a **fresh matched full-task evaluation** and use
`high-information-runs` for a staggered, checkpointed node wave.

Before the next diagnostic load, state its answer-level hypothesis, original
batch/control set, run budget and stop condition. If the budget produces no
repeatable net answer improvement, stop precision sweeps and re-anchor the task
metric. Do not require one historical reference answer to reproduce before
allowing a full accuracy measurement; reference stability is required for
**local causal attribution**, not for reporting a new protocol-bound task score.

If the matched full run has no aggregate deficit, close that task-accuracy
campaign without declaring numerical equivalence, default promotion or a green
enablement certificate. Further per-op correctness obligations remain separate.
See [the worked accuracy closure](../../../../docs/dev/DSv4-accuracy-closure.md).

## Step 0a — FRAME THE PERFORMANCE KPI (surface this to the user FIRST, do not assume)

Performance has no single number: an autoregressive model runs **prefill** (M = S·B, compute-bound,
sets time-to-first-token) and **decode** (M = batch B, memory/dispatch-bound, dominates wall time for
any real output length and sets per-output-token latency). The "sweet spot" M, the ceiling, the kernel
priorities, and even which win counts all DEPEND on the target. So before `establish-achievable-performance`,
**ASK THE USER** (use the ask-questions tool) which KPI defines success, and fix the operating point from it:

- **Throughput serving** (many concurrent requests) → batch **B large** (e.g. 32/64) → decode runs at
  M=B, off the M=1 cliff, near the kernels' sweet spot → headline = **aggregate decode tok/s**.
- **Low-latency single stream** → **B=1** → decode M=1 is the reality (cannot be moved to the sweet
  spot; the job is to make M=1 good) → headline = **single-stream tok/s / per-token latency**.
- **Both** (recommended default) → headline the throughput point, but ALWAYS carry the B=1 number as the
  honest latency floor so a decode-M=1 regression can't hide behind a batched win.

The operating point is the triple **(input length S, batch M=B, output length G)**, chosen to match the
target deployment — NOT an arbitrary M. S must be realistic (it is the attention/indexer KV length that
drives sparse-attention cost; a toy S makes those shapes non-representative). Record the chosen
(S, B, G) + headline metric and feed it to EVERY downstream step: the ceiling (0), the roofline phase
split (1c), the profile (1d), and the progressive wall-time ledger. Report the headline tok/s at the
operating point AND keep the prefill/decode decomposition of the SAME run as the attribution/regression
view. The dense per-M kernel sweep stays at the standalone/roofline level + the one final full-depth
confirm; the E2E ledger tracks the chosen operating point.

> **STANDING DISCIPLINE — load `shared/adversarial-self-audit` before submitting ANY gate output or
> review-response (every gated step above).** Fix the vulnerability CLASS not the shown instance;
> validate by FALSIFICATION (red-team unseen sibling cases through the FULL entry path), not
> confirmation; and apply provenance to your OWN claims (re-verify every "closed/all/validated/cited"
> against the artifact + real published source before writing it). This prevents the multi-round
> review FAIL spiral (patching the last counterexample while the vulnerability class stays open).

## Decision flow

```
0a. FRAME THE KPI (ask the user)     -> (S, B, G) operating point + headline metric (throughput vs latency)
0b. startup infrastructure gate      -> live resource inventory, per-rank budgets, CPU/GPU placement,
                                        safe shard/load concurrency; refresh before each submit
0. establish-achievable-performance  -> compute/stream/mem ceilings for THIS allocated, verified node
1. model-op-decomposition            -> normalized op graph + UPFRONT DATA-TYPE AUDIT
                                        (each op's ACTUAL stored ckpt dtype -> best target-HW
                                        compute dtype; flag missing-native-compute dequant paths,
                                        low-bit-ckpt INFLATION traps, and suboptimal fp32/torch ops)
1b. fusion-analysis                   -> fusion plan: COVERED (donor fused-kernel) /
                                        NEW-FUSED-KERNEL (-> kernel-authoring) / SKIP;
                                        cross-checked vs external impls/claims
                                        (vLLM, TRT-LLM, FlashInfer, authors, blogs)
1c. model-roofline-analysis           -> per-phase (prefill/decode) roofline; Amdahl-rank
                                        op-classes; high-ROI plan (memory-bound->precision,
                                        compute-bound->AMX); orders the complete kernel queue
1c2. enablement-scope-discovery       -> dependency-closure walk of the ACTUAL forward + backend
                                        + KV/memory-pool code to the sub-op leaf; enumerate ALL
                                        kernel families (attention+norm+routing+...) AND the
                                        runtime substrate; classify each (routing/port/authoring);
                                        honest scope BEFORE the first run (no reactive cascade)
2. kernel-capability-registry        -> load op->kernel contracts + peer donors
3. coverage-gate:
     all ops covered      -> reference bring-up
     any GAP (novel op)   -> new-kernel leg establishes source-faithful functional path;
                            no performance acceptance while coverage is incomplete
4. cpu-model-wiring / cpu-serving-integration (plugin only)
                                     -> reference bring-up; temporary fallbacks explicitly labeled
5a. accuracy-oracle                  -> tiny bring-up parity, then full-shape/depth parity;
                                        deterministic weights must match across reference/target
5b. model-profile-hotspots           -> representative profiling orders all required kernel work;
                                        dummy routing must be validated; recheck on real weights
5c. cpu-optimization-playbook         -> every standalone kernel native and optimized; measured
                                        alternatives including fusion; no-regression; human review
5d. cpu-serving-integration          -> integrate accepted kernels, prove dispatch, validate
                                        fused regions; accuracy-oracle real-weight/task finale
6. peer-relative-roofline            -> per-op % vs achievable AND vs peer model;
                                        end-to-end tok/s vs peer; loop on the gap
7. enablement-certificate            -> machine-checkable pass/fail + reviewable diff
```

## Parallelize the long pole (artifact acquisition ∥ weight-independent work)

Downloading the real checkpoint (often ~TB) is the longest single wait, but ALMOST
NOTHING on the critical path needs the weight BYTES — so never idle on it. The moment
a model is named, FORK two tracks:
- **Background:** kick off the full weight pull (resumable) and let it run unattended.
- **Foreground (weight-independent, start immediately):** steps 1–4 above —
  op-decomposition, fusion/roofline analysis, scope-discovery, coverage-gate — plus the
  reference-wiring bring-up on a TINY arch-faithful config with DUMMY weights
  (`cpu-model-wiring`) and reference-first authoring of any GAP op. All of these read the
  model CLASS + CONFIG, not the trained weights.
- **Metadata-first unblocks the dtype audit WITHOUT the full download OR a token-gated
  full pull.** Fetch ONLY the small files first — `config.json` + the safetensors INDEX
  (`model.safetensors.index.json`) and, if needed, a single shard's HEADER (the leading
  JSON length-prefix). That is enough to (a) census every weight family's stored dtype +
  shape → run `model-op-decomposition` §2b in full, (b) finalize the scope/coverage, and
  (c) confirm repo access/gating early — all before a byte of tensor data lands. Real case
  (GLM-5.3 Flash): a metadata-only fetch completed the entire dtype audit + scope and
  proved the repo ungated, so the weeks-of-bytes download never blocked design work.
Only the FINAL accuracy/perf-with-real-weights runs (accuracy-oracle Layer 1–2, perf
ladder stage 3) actually consume the downloaded bytes; by the time they're reached the
download has long finished in the background. Weights are a dependency of VALIDATION, not
of DESIGN — serialize nothing behind them.

## Composition rule

Run gates in order; a red gate blocks the next. Coverage is the fork: a genuine
gap exits this leg to the new-kernel loop rather than degrading the model. Record
the winning op→kernel mapping + measured efficiencies into the registry's
learned-patterns so the next model starts from a known-good wiring.

## PUBLISH-SYNC — collateral update is automatic, part of "done" (do NOT wait to be asked)

Any step that changes a PUBLISHED fact — a skill/playbook/memory edit, a perf
pass, a new/revised measurement, a gap-ledger change, a kernel landing — must, in
the SAME unit of work and BEFORE the step is called done, sync the published
collateral: (a) PROSE (README / THESIS / docs pages), (b) CHARTS regenerated FROM
the committed data (`make_*.py` → re-render PNG, cache-bust the filename `_vN`) with
embedding + caption updated, (c) the GAP LEDGER / opportunities / roofline tables,
(d) COMMIT + PUSH. "Done" is not "the code/number/skill changed" — "done" is "the
published artifacts reflect it and are pushed." Charts are generated-from-data,
never hand-edited. The user should never have to prompt "now update GitHub."

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
