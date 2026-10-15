# CPU Enablement & Kernel-Optimization Playbook — Methodology

How a new model is brought up and made fast on Intel Xeon (EMR/GNR) CPUs: a **waterfall** of phases that
starts with discovery + provenance and ends with a running, accurate, roofline-validated model — orchestrated
by a **spine-leaf** set of skills, where **every phase is closed by a gate** so the output of one phase is
*trusted* before the next begins.

> This page documents the *method*. The executable rules live in `.agents/skills/`; a worked application is
> [kernel-pilot-performance.md](kernel-pilot-performance.md) (the DSv4-Flash DSA standalone-kernel results).
> The **Phase-1 "how-to"** — the staged way to optimize one kernel — is the
> [Standalone Kernel Optimization Playbook](standalone-kernel-optimization-playbook.md).

---

## 1. Spine-leaf structure

The method is **not** a monolith. A thin **spine** (two orchestrators) *routes* a target through a library of
**leaf** skills, loading each only when its trigger condition is met and gating every step against the
measured roofline. The spine decides *which* technique and *in what order*; the leaves *do* the work.

![Spine-leaf structure: two orchestrators route a target through leaf skills, each roofline-gated](spine-leaf-slide.png)

<details><summary>Same structure as a Mermaid source graph</summary>

```mermaid
flowchart TD
    subgraph SPINE["SPINE — orchestrators (routing + phase order + gates)"]
      MEP["model-enablement-playbook<br/>(throughput leg: wire a model from EXISTING kernels)"]
      COP["cpu-optimization-playbook<br/>(new-kernel leg: write / optimize a kernel)"]
    end
    subgraph DISCOVERY["LEAVES — discovery & decomposition"]
      D1["enablement-scope-discovery"]
      D2["model-op-decomposition"]
      D3["fusion-analysis"]
      D4["model-profile-hotspots"]
      D5["model-roofline-analysis"]
    end
    subgraph AUTHOR["LEAVES — author & optimize (ordered, roofline-gated)"]
      A0["establish-achievable-performance"]
      A1["kernel-authoring (donor-first)"]
      A2["openmp-parallelization"]
      A3["cache-blocking-tiling"]
      A4["amx-vectorization / cpu-gemm-amx-bf16"]
      A5["weight-prepacking-brgemm"]
      A6["quantization-amx-int8"]
      A7["inter-kernel-fusion"]
      A8["sub-numa-clustering / runtime-config-tuning"]
    end
    subgraph GATES["LEAVES — gates (integrity)"]
      G1["kernel-feasibility-gate"]
      G2["coverage-gate"]
      G3["roofline-validation"]
      G4["accuracy-oracle / f4_acceptance"]
      G5["adversarial-self-audit"]
      G6["enablement-certificate"]
    end
    MEP --> DISCOVERY
    COP --> AUTHOR
    DISCOVERY -->|genuine gap| COP
    AUTHOR --> GATES
    GATES -->|pass| MEP
```

</details>

- **`cpu-optimization-playbook`** — the *new-kernel* leg: write/optimize a kernel; routes it through
  establish-achievable → parallelize → tile → vectorize/AMX → prepack/BRGEMM → quantize → fuse, each gated by
  the roofline.
- **`model-enablement-playbook`** — the *throughput* leg: wire a whole model from existing kernels; its
  `coverage-gate` hands any genuine kernel gap to the new-kernel leg via the `cpu-optimizer` agent.
- **`shared/`** leaves (roofline-validation, establish-achievable-performance, perf-proxy, runtime-config-tuning,
  overhead-attribution, adversarial-self-audit, compute-from-native-precision) are used by both legs.

---

## 2. The waterfall — phases, in order

Each phase has a **purpose** (what it accomplishes), a **method** (how), and a **closing gate** (what makes
its output trusted). The order is strict: a later phase may not start until the prior gate passes.

```mermaid
flowchart LR
    P0["Phase 0<br/>Discovery &amp; Provenance"] --> P1["Phase 1<br/>All kernels in optimized C/C++<br/>(standalone, roofline, parallel wave)"]
    P1 --> RV{"⛔ Phase-1 exit<br/>PAUSE FOR REVIEW"}
    RV --> P2["Phase 2<br/>North-Star end-to-end<br/>WALL TIME"]
    P2 --> P3["Phase 3<br/>Accuracy sign-off<br/>(correctness → perf → task)"]
    P3 --> CERT["Enablement certificate<br/>(runs + accurate + roofline-vs-measured)"]
```

### Phase 0 — Discovery & Provenance

- **Accomplishes:** a trustworthy map of *what* must be built — the model's **dataflow graph** (not just an op
  *list*): every op **and fused op** as a **node**, plus the **edges** (producer→consumer), the **operands
  shared across nodes** (e.g. a KV cache or a compressed representation read by several ops), the **per-edge
  dtype/layout flow** (what each op actually *receives* from its producer), each node's **scaling dimension**
  (what makes its cost grow — batch / seq / top-k / experts), and the **architectural *intent*** of each
  sub-graph (e.g. a cheap low-precision *filter* feeding a precise *consumer*). Plus per-op shapes/dtypes, the
  knobs (batch/token **M**, KV-cache **S**/topk/page), the dynamic dispatch branches, and a **provenance
  binding** of every future kernel to the published source op at a pinned revision.
- **Method:** `enablement-scope-discovery` + `model-op-decomposition` + `fusion-analysis` read the *actual*
  model/reference (not assumptions); shapes are captured from a real run, never config-inferred. The
  **dataflow graph (nodes + edges + shared operands + intent) is verified against the EXTERNAL reference
  architecture** (the paper / official repo / reference kernels), not inferred from config or our own port — a
  composite mechanism (e.g. a sparse-attention sub-graph: compress → index → select → attend) is only understood
  once its stages and their hand-offs are confirmed against the source design.
  `model-profile-hotspots` + `model-roofline-analysis` rank the work by ROI and bound class.
- **Gate (trust):** **graph-/shape-provenance gate** — a kernel target is admitted only with captured shapes +
  dtypes + the source-op contract (storage/compute/accumulator dtype, intermediate rounding, mask/sink,
  output boundary, state) **and its place in the externally-verified dataflow graph** (what produces its
  inputs, what consumes its outputs, which operands it shares, what dimension it scales with). *A
  config-inferred shape — or a kernel scoped in isolation from its verified graph context — is flagged
  UNVERIFIED and cannot carry a perf claim.*

### Phase 1 — All kernels in optimized C/C++ (the hard prerequisite)

- **Accomplishes:** every op/fused-op has an **authored C/C++ kernel**, each optimized across its knobs
  (full M-sweep + KV) **toward roofline**, with a published per-kernel roofline-target-vs-measured record.
- **Standing practice (exhaust in-phase + publish):** every legitimate **kernel-authoring** optimization for a
  kernel is **attempted in this phase** — not deferred, not left as a "could try." Attempt it, **measure it
  under the streaming methodology, and publish the result on the standalone-kernel page** — including a
  **measured loss**, which is a valid, first-class disposition (it closes the lever honestly so it is not
  re-proposed). *Example: sparse-attend's bf16-AMX lever was attempted in 3 configs, all measured slower than
  the fp32 donor, and that loss is published.* Only levers realized in the **serving path** (memory-traffic /
  dtype / layout) are handed to Phase 2; everything that lives **inside the kernel** is exhausted here.
- **Method (donor-first, parallel wave):** reuse an existing optimized primitive wherever one exists
  (OneDNN, ATen cpublas BRGEMM + VNNI pack, `at::bmm`, MKL, LIBXSMM) and author only the irreducible glue;
  author a net-new kernel only when no donor covers the contract (`kernel-authoring`). The per-kernel
  optimizations are **independent work items → fan them out in parallel**, ranked by ROI. Apply techniques
  in order (parallelize → tile → vectorize/AMX → prepack → quantize → fuse), **re-running `roofline-validation`
  after each step** and classifying the regime (BW / compute / selection / dispatch / cache-resident) before
  assuming headroom. The staged how-to is the
  [Standalone Kernel Optimization Playbook](standalone-kernel-optimization-playbook.md).
- **Gates (trust), all required per kernel:**
  - **`kernel-feasibility-gate`** (before authoring a net-new kernel): user-reviewed roofline + measured
    baseline → go/no-go.
  - **No-regression floor** vs the same-contract torch fallback at *every* M (faster or at worst a tie; else
    dispatch to the library path at that M).
  - **Fusion gate** — no avoidable intermediate DRAM round-trip (record the irreducible residual + why).
  - **Precision/BW gate** — store/read at the lowest precision the *reference* tolerates; fp32-accumulate;
    a precision that changes the reference dtype is a *conformance decision*, not a free lever.
  - **`accuracy-oracle` / `f4_acceptance`** — per-op acceptance: output contract (shape/dtype/device),
    finiteness, run-to-run **determinism**, selection membership (`tie_eps=0`), and a *screened* continuous
    error vs a **source-conformed** oracle. (Thresholds are the human reviewer's to ratify → verdict is
    PARTIAL, never a silent PASS.)
  - **Transcendental rule** — a replaced `exp`/`erf`/`tanh` uses an industry-standard minimax, carries a
    **measured ULP** (≈1 ULP bar for exp, vs fp64 over the real range), guards the edge cases the approximation
    changes, and **keeps both** the exact-`libm` (bit-exact oracle) and approx variants for the E2E A/B.
  - **Standalone-kernel performance PAGE** (like [kernel-pilot-performance.md](kernel-pilot-performance.md)):
    per-kernel roofline + baseline + optimization + analysis, **streaming (cold-cache) headline latency**
    (flush ≥2× LLC; warm only as a labeled secondary), `objdump %zmm` ISA check + stated flags, honest
    regime verdicts ("at ceiling" is a valid outcome **only after a config sweep shows the bound doesn't
    move** — see [§4 rule 7](#4-per-kernel-performance-gap-ledger-model-agnostic)), replicated with the noisy column flagged.
  - **In-phase exhaustion + gap ledger** — every legitimate kernel-authoring lever is attempted, measured, and
    its outcome (win *or* measured loss) published; every kernel exits Phase 1 with a
    [gap-ledger](#4-per-kernel-performance-gap-ledger-model-agnostic) row whose owning phase is assigned (no
    un-triaged gap, no deferred in-kernel lever).

### ⛔ Phase-1 exit — PAUSE FOR REVIEW (the integrity checkpoint)

- **Accomplishes:** prevents a mis-aimed or under-validated Phase 1 from contaminating the whole E2E phase.
- **Method:** run the **adversarial self-audit** *yourself* first — conform every oracle/reference to the
  authoritative published source, validate by **falsification** (red-team unseen sibling cases through the
  *full* entry path, not confirmation), and apply provenance to your *own* claims (re-verify every
  "closed/all/validated" against the artifact + real source).
- **Gate (trust):** **human review ratifies policy/tolerances** — the executor never silently widens a
  tolerance, drops a case, swaps a comparator, or relabels uncertainty as parity. Do **not** auto-advance.

### Phase 2 — North-Star end-to-end wall time

- **Accomplishes:** the actual deployment metric — end-to-end tok/s at a chosen operating point — with each
  optimization's wall-time contribution attributed.
- **Method:**
  1. **Frame the KPI first (ask the user):** fix the operating point **(input S, batch B=M, output G)** +
     headline metric (throughput serving vs B=1 latency floor; default both).
  2. **Preserve the kosher baseline:** freeze the golden fingerprint, **tag** the baseline commit, make every
     optimization **env-gated default-OFF** so gate-off reproduces the baseline byte-for-byte. Clear the known
     systemic pathologies on the *baseline* (threads bound once at init, one rank per NUMA domain, no
     per-forward `set_num_threads`).
  3. **Integrate the authored kernels progressively — on the truncated `perf-proxy`.** The depth-reduced,
     full-width proxy is the integration *vehicle* (queue-free, ~`full/proxy`× faster): wire each authored
     kernel **one at a time**, env-gated default-OFF, with a per-toggle dummy-weight fingerprint gate, and
     record each wall-time delta **broken down into kernel / framework / dispatch** (via `overhead-attribution`)
     so every delta is bisectable to one op and the residual is *named*, not lumped. Predict full depth during
     iteration with the **two-depth linear model** `t(L) = t_fixed + L·t_layer` *per regime* (prefill/decode) —
     never naive `t_proxy/layers × full_layers`.
  4. **Carry forward the integration-class levers** surfaced during the standalone-kernel phase — memory-traffic,
     dtype, and data-layout wins that the kernel already supports but that are realized only in the serving path
     (e.g. a **bf16 KV cache**: numerically identical to what a bf16-rounding GEMM consumes, yet halves the
     dominant read and **amortizes across every op that reads that cache**, with the most leverage at decode
     M=1). Log each at discovery as a tracked Phase-2 lever (mechanism / amortization / expected delta /
     fingerprint-invariant gate / status), so it is optimized here instead of rediscovered later.
  5. **Address framework / dispatch overhead last** (after the kernels are in). With the authored kernels
     wired, the residual is the per-op **dispatch/framework tax** — OpenMP barriers, allocations, thread-pool
     rebuilds, lost affinity — attacked with **fewer/bigger parallel regions, bind-once-at-init, one rank per
     NUMA domain, prepack** (`runtime-config-tuning`). Doing it *after* kernel integration keeps the per-kernel
     attribution clean: a global config pathology otherwise masquerades as uniform per-op cost.
  6. **Full-depth dummy-weight run — measure & attribute.** Once proxy iteration converges, run the
     **full-depth (all-layer) dummy-weight** model to *measure* the real wall time and **attribute it across
     components** (per-kernel / framework / dispatch, via the per-op pivot). This both **validates the two-depth
     extrapolation** and exposes any residual depth-accumulating overhead (allocator / thread-pool /
     cache-residency) to bisect. Only now is the headline tok/s final — then the real-weight accuracy sign-off
     (Phase 3).
- **MoE proxy mode-collapse guard (when the model is MoE).** A depth-reduced **dummy-weight** proxy's random
  router routes **degenerately** — it collapses the whole batch onto the *same* few experts, not the real
  learned distribution — which **invalidates both the wall-time and the correctness check** until fixed:
  batched-MoE decode cost tracks the **distinct-expert working set** (which grows with batch), so a collapsed
  router under-counts expert memory traffic and looks **falsely fast**, and a single-expert path hides the
  grouped-GEMM gather/scatter routing bugs. **Keep the real router/gate weights** (tiny — load them even with
  dummy *expert* weights), drive **varied tokens** (never one repeated prompt), match the real top-k/grouping,
  and **gate on the measured distinct-expert-per-batch histogram matching the real `E(distinct | B)`** (not 1,
  not all-N) before trusting any MoE perf or parity from the proxy. (See `perf-proxy`.)
- **The ceiling is the model-level throughput roofline (target measured tok/s ≥ 70–75% at *every* M).**
  Decode is memory-bound: `tok/s_roof(M) = M·BW / (W_active + M·KV_tok)`, where `W_active` = weight bytes
  read per step (**MoE-aware: active experts only**, at the checkpoint dtype) and `KV_tok` = KV bytes/token
  (KV dtype). Prefill is compute-bound → a separate `FLOPs/peak` roofline, so **split the ceiling by phase.**
  The **TPOT** (time-per-output-token) roofline is the same ceiling inverted per request:
  `TPOT_floor(M) = (W_active/M + KV_tok) / BW` — so batching amortizes `W_active` across `M` tokens (TPOT
  falls as M rises until KV_tok/BW dominates), and a **prefill TTFT** floor = `prefill_FLOPs / peak`. Both the
  wall-time (prefill FLOPs-bound) and TPOT (decode BW-bound) rooflines are rooted in **native-compute peak + BW**
  (gap-ledger rule 13) — low-bit experts/KV cut the *bytes* (`W_active`, `KV_tok`), they do not raise a compute
  ceiling the HW lacks. Plot `tok/s_roof(M)` as the **ceiling line** on the E2E progression chart; the
  measured-vs-roofline gap is **integration efficiency**, attributed by the per-op pivot (e.g. MoE ~47%,
  authored DSA+MHC ~25–30%). The roofline rewards memory-traffic levers directly — **bf16-KV halves `KV_tok`;
  fp4/fp8 experts shrink `W_active`; both raise the ceiling** — and at M=1 the ceiling is both LOW (weights read
  for one token) and hard to reach (overhead), so batching raises both the ceiling and the efficiency (DSv4
  ~26% at M=1 vs ~75% batched). The 70–75% headroom budgets non-overlap, dispatch gaps, and the authored-op
  slice (not pure weight-GEMM). **Execution loop:** integrate one kernel → measure wall-time + TPOT vs the
  roofline → attribute the gap (kernel / framework / **dispatch & parallelization overhead**) → clear the
  overhead (bind-once, fewer/bigger parallel regions, one rank per NUMA domain, prepack) → re-measure, so
  **every integration step updates measured-vs-roofline** until the 70–75% target holds at every M.
- **Gate (trust):** per-toggle **dummy-weight fingerprint** stays invariant (perf transform, not a numeric
  change); the full-depth run validates the extrapolation; the roofline+pivot report pair is emitted.

### Phase 3 — Accuracy sign-off (correctness → perf → task, in that order)

- **Accomplishes:** proves the fast CPU build is *numerically faithful*, then *task-accurate*.
- **Method:** per-layer **parity** vs the reference oracle (real weights, one short prefill) is the first
  correctness gate; on the dummy proxy the guard is the optimized-vs-reference A/B on the same input; **task
  accuracy** (real weights, full generation) is last. The task sign-off has two legitimate routes —
  **(A) reproduce the published card number** or **(B) CPU-vs-GPU equivalence on a short benchmark** — surfaced
  as a **user decision** (cost vs external-validation tradeoff), never silently chosen.
- **Gate (trust):** golden per-op parity + full-model re-diff; an optimization becomes the default only after
  it passes both. `enablement-certificate` records runs-end-to-end + accuracy-parity + roofline-vs-measured.

---

## 3. The gate ladder (why each phase output is trusted)

| phase | closing gate | what it guarantees |
|---|---|---|
| 0 Discovery | shape-provenance gate | shapes/dtypes/contract are *captured from source*, not assumed |
| 1 Author (per kernel) | feasibility → no-regression floor → fusion → precision → `f4_acceptance` → page | correct (contract/finite/deterministic/selection), never slower than torch, no wasted DRAM traffic, published roofline record |
| 1 exit | adversarial self-audit + **human review** | falsified not just confirmed; tolerances ratified by a human, not the executor |
| 2 E2E | kosher-baseline freeze/tag/env-gate + per-toggle fingerprint + two-depth validation | every speedup is bisectable to one op; baseline is one `git checkout` away; extrapolation is validated |
| 3 Accuracy | golden parity + full re-diff + task sign-off (user-chosen route) | numerically faithful *and* task-accurate vs a trusted reference |
| cycle exit | roofline+pivot report pair + standalone-kernel page + certificate | "done" = runs + accurate + roofline-vs-measured, published |

> **Definition of done includes publishing.** Any phase whose output changes a published fact is not complete
> until the collateral is synced per the **PUBLISH-SYNC invariant** below — prose + charts (regenerated from
> data, cache-busted) + gap ledger, committed and pushed. A measured win that isn't on the page didn't ship.

**Cross-cutting invariants** (hold in every phase): PROVENANCE always (no claim without its bound evidence);
DONOR-FIRST *for the compute inner loop* — never re-implement a library GEMM, **but own the composition/fusion
when the op is dispatch- or materialization-bound** (donor-first is not a reason to ship a donor whose per-call
dispatch dominates — see gap-ledger rule 14); MEASURE don't assume (regime, ULP, streaming number,
ISA via `objdump`); and `adversarial-self-audit` before submitting any gate/record/review-response — fix the
vulnerability **class**, not the shown instance.

**PUBLISH-SYNC invariant (collateral update is part of the work, not a follow-up).** Any change that alters a
**published fact** — a skill/playbook edit, a performance-optimization pass, a new or revised measurement, a
gap-ledger change, a kernel landing — **automatically** triggers, *in the same unit of work and before the task
is called done*, a sync of the published collateral: **(a) prose** (README / THESIS / docs pages that state the
fact), **(b) charts** regenerated **from the committed data** (`make_*.py` → re-render PNG, cache-bust the
filename) with their embedding + caption updated, **(c) the gap ledger / opportunities / roofline tables**, and
**(d) commit + push**. "Done" is not "the code or number changed"; "done" is "the published artifacts reflect it
and are pushed." Charts are **generated-from-data** (edit the JSON / re-run the bench, re-run the generator) —
never hand-edited pixels. No separate prompt is required to publish.


---

## 4. Per-kernel performance-gap ledger (model-agnostic)

The standing mechanism that turns "roofline vs measured" into a tracked to-do list for **any** model. Every
standalone kernel gets **one row**; a kernel is not "done" until its gap is *classified* and, if recoverable,
*assigned an owning phase*. It is populated at **Phase-1 exit**, carried through Phase 2/3, and **published on
the standalone-kernel page** so the gap analysis ships alongside the speedup — not just the win.

**Ledger row (one per kernel):**

| field | meaning |
|---|---|
| roofline floor | analytic `max(bytes/BW, FLOPs/peak)` vs the **nominal** peak (not an achievable ceiling) |
| measured | promoted-variant **streaming** latency (LLC-flushed), 3-replicate median |
| % roof | floor / measured — the honest ceiling check |
| regime / bound | BW · compute · cache-resident · selection · dispatch · recurrence |
| **gap class** | **RECOVERABLE** (roofline *is* the operative bound and measured is below it) · **WRONG-CEILING** (selection/dispatch/recurrence — a byte/FLOP roofline is not the bound; low % is diagnostic, not owed) · **AT-CEILING** (meets the roof, or cache-resident so the nominal roof under-counts) |
| root cause | *why* perf is lost — attributed, not guessed (which bytes, which overhead, which serial dependency) |
| opportunity | the concrete lever + **expected effect** (e.g. "bf16-KV halves the dominant read → floor 50→25 µs, ~1.5×") |
| **owning phase** | where the fix is realized (see mapping below) |
| status | open · in-flight · closed · owed-to-Phase-2 |

**Owning-phase mapping** (every RECOVERABLE gap gets exactly one):

- **Phase 1 — kernel authoring:** the lever lives *inside the kernel* — compute dtype (bf16/AMX), fusion,
  vectorization, transcendental, tiling. *(e.g. sparse-attend bf16-AMX GEMM.)*
- **Phase 2 — E2E integration:** the lever is realized only in the **serving path** — memory-traffic / cache
  dtype / data layout / batching. The kernel may already support it; the caller must supply it. These are
  **mirrored into the Phase-2 carry-forward lever backlog** (default-OFF, env-gated, dummy-weight
  fingerprint-invariant). *(e.g. bf16 KV cache, amortized across every KV-reading op.)*
- **Phase 3 — accuracy:** a numerically non-identical candidate (a tolerance-gated bf16/fp8 path) that is
  **ratified** by the task-accuracy gate before promotion. *(e.g. the sparse bf16 candidate.)*
- **Closed — regime ceiling:** WRONG-CEILING or AT-CEILING kernels — no owed work; the low/high % is honest.

**Rules.**
1. **Every kernel gets a row** — nothing ships without one.
2. **No `RECOVERABLE` without attribution.** Prove the gap with stage timers or an isolation microbench that
   separates a *real bandwidth shortfall* from *fixed convert/pack/dispatch overhead*. A low roofline-% alone
   does **not** prove the gap is recoverable.
3. **`WRONG-CEILING` is a valid result, not a failure.** A selection / dispatch / recurrence op at 2% of a byte
   roofline is *at its real ceiling* — the byte roofline is simply the wrong yardstick.
4. **Mirror every `RECOVERABLE` gap to its owning queue** (Phase-2 backlog or the author queue) so it isn't
   rediscovered later.
5. **The ledger is the per-kernel source of truth**, published on the standalone-kernel page.
6. **Predicted ≠ measured.** The *opportunity* column is a **hypothesis** — label it unvalidated, and never
   publish a predicted speedup as a result. Validate cheapest-first: component model → isolation microbench →
   minimal variant → full. A one-component roofline over-predicts when the op has other first-order costs the
   lever doesn't move (the sparse-attend bf16 miss) — so an **expensive** change needs the isolation microbench
   *before* you commit; a **cheap** one, just measure it. *(See standalone
   [Stage 1.5](standalone-kernel-optimization-playbook.md#stage-15--validate-a-levers-upside-before-committing-cost-scaled).)*

**7. A bound is a tested hypothesis, not a wall.** Before you write `AT-CEILING` or `WRONG-CEILING`, do two things:
   - **Sweep the config** — tiles / grain / effective-threads (via `grain_size` or task-count; **never**
     per-forward `set_num_threads`, which loses thread affinity). If the bound **moves**, it was config → fix it
     and ship the per-M optimum. If it **doesn't** move, it's algorithmic/hardware → the ceiling is earned.
   - **Test the one knob attribution points to** — the **precision** lever where it hits the dominant component
     (bf16/fp8 can win *or* lose; publish either), and **author your own inner GEMM** only if measurement puts
     the bottleneck *inside* the donor (rare — otherwise route around it and publish the gap).

   Closing on assertion is the anti-pattern: sinkhorn's low-batch win (adaptive serial dispatch, +1.15–2.09×,
   bit-exact) was found because the knob was *tested*; three other ceilings were *confirmed* because the knob
   measured a loss.

**8. Target each component, not the total %.** Split the latency into **data-movement / compute / dispatch** and
   give each its own target: data-movement ≥ 75% of the **operative** bandwidth (DRAM if the working set spills
   cache, **cache bandwidth if it fits** — not nominal DRAM); compute ≥ its ISA/dtype peak (AMX or AVX-512);
   overhead → 0. The roofline-% of the *total* is only a **diagnostic** of which component dominates — not the
   target — and the nominal roofline can be the **wrong ceiling** (an op at 24% of the nominal DRAM roof but
   ~52% of its measured *operative* floor is already near its real ceiling).

**9. Hunt data reuse where tiling allows it — predict the traffic cut, then measure it.** When you shard into
   tiles (rule 7), look for an operand that is **re-read from DRAM** and block/order the loop so it is read
   **once** and reused from cache. This raises the op's effective arithmetic intensity (fewer DRAM bytes per
   FLOP) and moves it right on the roofline — **capped at the compute ridge** (reuse turns a BW-bound op
   compute-bound, then stops). **State the predicted floor first:** reuse factor = total reads ÷ unique bytes,
   so the floor drops from `redundant_bytes/BW` to `unique_bytes/BW`. Then size the tile **jointly** for reuse
   + cache-fit + core-fill (rule 7), re-assess per M (small M is often already cache-resident → overhead is the
   lever; large M must block to keep reuse before eviction), own the **composition** not the donor's inner loop
   (donor-first), and **validate on traffic** — confirm the operand is read from DRAM ≈ once (achieved bytes ≈
   unique bytes); a latency win *without* a measured traffic drop means the reuse didn't bite.

**10. The best ISA is per‑(target, shape, M) — probe it, sweep it, dispatch adaptively (and conformance‑gate the switch).**
   The structure is arch‑invariant (author once); the **inner compute** (AMX‑bf16/int8 · AVX‑512‑bf16/fp16/fp32 ·
   AVX2) is arch‑specific and chosen by **measurement** — AMX is **not** always best (it underfills narrow or
   small‑M GEMMs where AVX‑512 wins). The mechanism:
   - **Probe** the target's CPUID flags (`uarch-perf-probe`) — never infer the ISA from the part name (EMR‑8580
     has AMX‑bf16 *and* AVX‑512‑bf16 *and* fp16; a client like Panther Lake has **no AMX** → AVX2/AVX‑VNNI).
   - **Isolation M‑sweep** each candidate ISA at the real GEMM shape, **find the crossover M**, and **dispatch
     the per‑M optimum** (like adaptive tiling). The donor dispatches its own ISA; your authored glue owns its
     ISA and needs an AVX2/scalar fallback for AMX/AVX‑512‑less targets. Add a path by *need* or *measured
     upside*, not speculatively.
   - **Conformance‑gate the switch** (ISA paths aren't bit‑exact): validate each to the oracle on **realistic**
     distributions. A precision change (fp32 vs the model's bf16 boundary) is Phase‑3‑gated; a dtype proxy
     (fp16 for a bf16 roofline) is **perf‑only**, fenced from accuracy. *Worked example — indexer score GEMM:
     AVX‑512 beats AMX **2.6× at M=1 → tie at M=32 → AMX wins M=64** (AMX underfills the K=128/N=64 shape), so
     the win lands at the decode latency floor; but the fast path is **fp32** (a precision change — open,
     Phase‑3‑gated), the conformance‑safe bf16 single‑bmm measured **slower**, and the shipped bf16 tiled
     brgemm stays the best bit‑exact path. Counter‑example — **sparse attend**: fat GEMMs where AMX is already
     the right, well‑utilized ISA, so an AVX‑512 swap can't help; the rule correctly excludes it.*
   - **AMX‑underfill at low M is often a CALL‑SHAPE artifact, not an ISA verdict — reframe before you switch.**
     Measured (wave 385951): a *batched* `at::bmm` of the score GEMM is catastrophic at batch‑1 on AMX (12× slow)
     and AVX‑512‑bf16 "rescues" it — but the **shipped single‑brgemm `m1` path already avoids it** by framing the
     long sequence dim S as the GEMM's **M‑dimension** (so AMX fills), and then AMX *wins* at M≥8 while the forced
     AVX‑512‑bf16 *loses*. So: when AMX underfills, first ask *can I reshape the GEMM so AMX fills* (batch→M,
     coalesce tiles) — a conformance‑safe win — before concluding "switch ISA." The ISA switch is the last resort,
     for shapes that genuinely can't be reframed.

**11. Optimize the graph, not the node — and enumerate the decision space it seeds.** A kernel is a **node** in
   the model's dataflow graph (discovered + externally verified in Phase 0 — see the graph-provenance gate),
   **not an island**. Before optimizing, read its graph context: what **produces** its inputs (upstream
   dtype/layout/residency), what **consumes** its outputs, which operands are **shared**, which **dimension it
   scales with**, and the architectural **intent**. Then build a **decision-space ledger** for the node whose
   axes are *seeded by the graph*:
   - **local axes:** donor · ISA (AMX/AVX-512/AVX2) · precision/dtype · algorithm · layout/packing · fusion;
   - **graph axes** (only visible from the graph): the **true input dtype/layout the pipeline supplies** (a
     standalone fp32/contiguous assumption may be an *artifact* — the real producer may emit bf16/fp8),
     **cross-kernel fusion** of producer→consumer pairs (kill the intermediate DRAM round-trip),
     **shared-operand residency** (read once, reused across nodes), and **pipeline-position priority**
     (optimize the node that scales with the hard dimension, not the one with the worst *local* %).

   **Dispose of every axis explicitly** — *shipped · measured-loss · N/A because X · **UNTESTED (named)***. An
   unexplored alternative is **named as untested, never silently absent** — that is how a blind spot (an
   alternative no one put on the list) is prevented. And **carry Phase-0 discovery findings forward** as
   pre-seeded rows (e.g. *"the reference already ships an AVX-512 kernel here"* — a fact our coverage report
   held but our kernel authoring ignored). *Worked: the DSA sub-graph (compressor→indexer→top-k→sparse-attn) —
   the **indexer** scales with seq-length (the long-context bottleneck) while **sparse** is bounded by top-k;
   sparse's fp32-KV was a **standalone artifact** (the pipeline supplies bf16/fp8); and the indexer's AVX-512
   win was **pre-stated in Phase-0** but not carried forward.*

**12. Benchmark the reference's *actual* algorithm/dtype/fusion — not a locally-plausible reformulation.** A
   kernel's ceiling is the ceiling of the **algorithm the reference uses**, measured at the **dtype and fusion
   structure it uses** — not of whatever formulation is convenient to write on CPU. External verification
   (Phase 0) must therefore capture, from the reference source, three things beyond the op-list/shapes/edges:
   the **algorithm** (e.g. FlashAttention-style fused gather+score+softmax+value vs a dense gather-then-two-GEMMs),
   the **dtype flow** (e.g. fp8/bf16 *stored* KV with fused dequant vs fp32), and the **fusion boundaries** (what
   is kept resident / never materialized to DRAM). Reconstructing a *different* formulation and measuring **its**
   ceiling measures the **wrong wall** — and worse, a negative result obtained under that reformulation (e.g. a
   bf16 GEMM that "lost") can be an **artifact of the reformulation** (a fp32→bf16 cast the reference never pays
   on its natively-bf16 operand), which then wrongly **closes** a real opportunity. *Worked: DSA sparse-attend —
   our dense-fp32 reformulation measured "compute-bound, at-ceiling," but the DeepSeek reference (`inference/
   kernel.py` `sparse_attn_kernel`) is **flash-fused, bf16-KV-resident, nothing materialized**.* **The reference
   is the right *algorithm* — not a guaranteed CPU win.** A GPU reference's algorithm/dtype/fusion is a
   **hypothesis to measure on the target HW**, never a result: fusion/precision that wins on a GPU can **lose on
   a CPU** because the cost model differs (cheap GPU shared-mem residency + warp GEMMs vs expensive CPU per-block
   AMX-underfill + pack/dispatch). *Measured (wave job 385941): the bf16-flash sparse path **lost** on CPU (flash
   0.33–0.76×, bf16-native dense 1.18×@M32 only) — the fp32 donor **is** the CPU ceiling, now known by direct
   bf16-native + flash measurement, not inference. The reference check was still essential: it turned an
   assumption into a falsifiable test.* So: reconstruct the reference algorithm, **measure it on the target**,
   and let the measurement — not the reference's provenance — decide ship/close.

**13. The roofline compute ceiling is the NATIVE-dtype peak; a sub-native low-bit dtype is a DATA-MOVEMENT
   optimization, not a ceiling.** The roofline has exactly two HW ceilings: the **BW slope** and the **compute
   peak of the dtypes the HW computes *natively*** (e.g. AMX bf16/int8/fp32 — *not* fp8/fp4 on GNR/EMR, which
   have no tile). A dtype the HW lacks a compute unit for is **never a compute-roofline point**: it is used for
   **storage/movement** only — store low-bit, **upcast to the nearest native dtype** (bf16), compute native. Its
   benefit is therefore **data-movement (fewer DRAM bytes) only**, realized **only when the op is BW-bound** (not
   dispatch- or compute-bound). Keep the roofline at **native-compute + BW**, compute `%roof` against it, and put
   the low-bit down-conversion in the **optimization/upside ledger** as a traffic cut — **do not relabel a
   storage optimization as a lower roofline** (that implies a compute capability the HW lacks and over-credits
   the win). *Worked: the indexer "fp8 ceiling" was this error — fp8 has no AMX tile, so it is a BW-only lever
   that helps at long context but buys ~0 at S=1024 where the op is dispatch-bound (wave 385941: 0.1–3.7% of the
   BW roof). Same principle as the MXFP4 MoE: fp4 **moved**, bf16 **computed**.*

**14. Donor-first has an EXIT CONDITION: own the *composition* when the op is dispatch/materialization-bound, not
   the donor's inner loop.** Donor-first forbids re-implementing a **library GEMM inner loop** you can't beat —
   that is a statement about **compute FLOPS**. It does **not** oblige you to accept the donor's **call shape**.
   When the regime is **dispatch- or materialization-bound** (low M / small-op / decode), the donor's cost is
   its **per-call primitive dispatch + the intermediate tensors it forces you to materialize** (e.g. two separate
   `at::bmm` calls + a `[N,H,K]` scores round-trip + a separate softmax pass), which can **exceed** the GEMM win
   it buys. The lever is then **FUSION / owning the composition**: one pass, intermediates kept **resident**
   (L1/L2), producer→consumer fused (softmax into the value step; logits into top-k), **persistent** packed
   micro-kernels instead of per-call primitive creation — *keeping the donor's math* where it is actually used.
   **Decision rule:** attribute first (rule 8) — if **data-movement ≈ 0 and compute ≈ 0 but latency ≫ roofline**
   (tiny `%roof` that is **flat** vs problem size), the op is **dispatch-bound → the donor is the wrong *shape*,
   author the fusion** (not the inner GEMM). Conversely if the donor's inner GEMM **is** the bottleneck (high
   `%roof`, compute-bound), keep it. *Don't get stuck shipping a donor whose per-call overhead dominates just
   because "donor-first."* Still **measure** the own-fused path against the donor before committing (rule 6) —
   over-blocking can reintroduce the very dispatch cost you set out to remove (the DSA sparse `blockbf16` flash
   lost by splitting into 64-token blocks). *Worked: DSA sparse/indexer are dispatch-bound at decode — the fp32
   donor forces 2 dispatches + a materialized score matrix; the lever is a fused one-pass kernel, not a faster
   GEMM. The ISA knob (AVX-512 vs AMX, rule 10) is the sibling: AMX tile-setup is pure dispatch overhead at low
   M, so the no-tile ISA wins there.*
