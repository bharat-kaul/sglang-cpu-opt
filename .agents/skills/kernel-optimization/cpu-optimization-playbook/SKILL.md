---
name: cpu-optimization-playbook
description: "Master index + decision flow for optimizing an SGLang model op on Intel Xeon CPUs. Routes a target op (GEMM, attention, MLP, norm, MoE) through an ordered, composable set of per-technique skills — establish-achievable → parallelize → tile → vectorize/AMX → prepack/BRGEMM → quantize — each gated by the roofline. Use this first; it decides WHICH technique skills to load and in WHAT order."
---

# CPU Optimization Playbook (orchestrator index)

This is the entry point the agentic workflow loads first for any CPU optimization
task. It does not itself optimize; it **routes** the op through a progressive,
composable library of per-technique skills, loading each only when its trigger
condition is met and gating every step against the measured roofline.

> **⛔⛔ USER-MANDATED CAMPAIGN REQUIREMENTS — NON-NEGOTIABLE (do NOT relax, defer, or dilute; any
> model-agnostic / adapter refactor MUST preserve these verbatim as campaign policy).** These are hard
> gates the user has explicitly required for this enablement; treat each as a blocking prerequisite, not a
> recommendation:
> 1. **SCOPE / SHAPE DISCOVERY FIRST.** Discover the model's real op graph, per-op tensor SHAPES/dtypes, the
>    knobs (batch/token M, KV-cache S/topk/page), and dynamic dispatch branches from the ACTUAL
>    model/reference — not assumed. Shapes drive roofline, donor choice, and coverage.
> 2. **PROVENANCE, ALWAYS.** Every authored kernel maps to the PUBLISHED source op at a PINNED revision
>    (contract: storage/compute/accumulator dtype, intermediate rounding, normalization/block order, masks,
>    output boundary, state). Every measurement/record binds source+build+config+input+job identity. No
>    claim without its evidence; conform the reference to the source, never validate against your own
>    reconstruction (`shared/adversarial-self-audit`).
> 3. **IDENTIFY OPS *AND* FUSED OPS.** Enumerate every op and every FUSED op in the graph
>    (`model-op-decomposition` + `fusion-analysis`); a fused op is a first-class kernel target. Coverage is
>    over the full op/fused-op set, declared independently of what happened to be tested.
> 4. **ALL KERNELS IN OPTIMIZED C/C++ *BEFORE* END-TO-END (the hard phase gate).** Every op / fused-op has a
>    C/C++ kernel, each optimized across its knobs (M-sweep + KV) toward roofline, with a no-regression
>    floor vs the SAME-CONTRACT torch fallback (replicated, order-varied, identity-bound) and correctness
>    held by the per-op gate (`tie_eps=0` selection / source-conformed continuous). 100% authored-C/C++
>    coverage + recorded perf-vs-roofline is the gate; ONLY THEN the North-Star end-to-end wall-time phase.
> 5. **PAUSE FOR REVIEW at the Phase-1 exit**, after the pre-submit self-audit (below). Do NOT auto-advance
>    to end-to-end integration. Policy/tolerance ratification is the human reviewer's, never the executor's
>    (never silently widen a tolerance, drop a case, swap a comparator, or relabel uncertainty as parity).
>
> The sections below (phase order, donor-first, fusion/precision gates, pre-review falsification pass) are
> the mechanized enforcement of these mandates. A model-agnostic redesign may re-house them into adapters,
> but the REQUIREMENTS stay as the DSv4 campaign policy and the executable gates must still enforce them.

> **STANDING DISCIPLINE — load `shared/adversarial-self-audit` before submitting ANY gate/validator/
> record/review-response.** Fix the vulnerability CLASS not the shown instance; validate by
> FALSIFICATION (red-team unseen sibling cases through the FULL entry path), not confirmation; test
> adversarial cases end-to-end; and apply provenance to your OWN claims (re-verify every "closed/all/
> validated/cited" against the artifact + real source before writing it). Stops the multi-round review
> FAIL spiral (patching the last counterexample while the vulnerability class stays open).

> **⛔ KERNEL-OPTIMIZATION META-LEARNINGS (distilled from the DSv4 compressor retry + full kernel-set wave;
> GENERALIZABLE to any model's standalone-kernel phase — apply BEFORE and DURING per-op work).**
> 1. **CLASSIFY EACH KERNEL'S REGIME BEFORE ASSUMING HEADROOM.** The same campaign gave a 4.8× on one kernel
>    (compressor — exp-compute-bound AND unvectorized) and ~0 on three others (selection / dispatch / cache-
>    resident, already at ceiling). "At ceiling" is a VALID, required outcome — report it, do not invent
>    headroom the regime forbids. Measure each kernel's regime (BW / compute / selection / dispatch / cache);
>    never extrapolate one kernel's win to the set.
> 2. **A ROOFLINE-% / "bound" LABEL CAN BE WRONG — question a surprising one.** The compressor was labeled
>    "27% of roofline / BW-bound" but was actually exp-compute-bound AND cache-resident (17 MB fits L3), so the
>    nominal DRAM roofline was the WRONG ceiling. Two cheap checks catch it: (a) WORKING-SET vs LLC (set < L3 ⇒
>    not DRAM-bound); (b) PHYSICS sanity (measured BW ABOVE DRAM peak ⇒ cache-resident, not a win). Beating your
>    own "floor" means the floor was computed against the wrong level of the memory hierarchy.
> 3. **STREAMING (cold-cache) IS THE RELIABLE STANDALONE NUMBER.** Warm buffer reuse re-reads an L3-resident
>    set from cache and flatters the result (compressor warm 37µs vs cold 62µs = 77% DRAM roof). Flush ≥2× LLC
>    before every timed call; quote warm only as a labeled secondary (the fused-pipeline, GEMM-just-wrote case).
> 4. **VERIFY THE BUILD/ISA, NOT JUST THE SOURCE.** `objdump %zmm` the built `.so`: gcc defaults to 256-bit and
>    `at::vec` falls back to AVX2 in a JIT extension unless `-mprefer-vector-width=512 -DCPU_CAPABILITY_AVX512`.
>    A "vectorized" variant that was actually built 256-bit was wrongly rejected once — measure the real binary.
> 5. **TRANSCENDENTAL REPLACEMENT: inline it, bound it, keep both.** Replacing a `libm`/`at::vec`/Sleef `exp`
>    CALL with an INLINED minimax (Cephes) removed the call boundary and was the whole compressor win (2.2×).
>    But: use an INDUSTRY-STANDARD polynomial; MEASURE the ULP vs an fp64 reference over the REAL arg range
>    (~1 ULP bar for exp — 0.5 ULP / correctly-rounded is impractical and not met by the reference stack either,
>    GPU `ex2.approx` ≈ 2 ULP); GUARD the edge cases the clamp breaks (`exp(-inf) → ~1e-38`, not 0 → all-masked
>    guard); and KEEP BOTH the exact-`libm` (bit-exact oracle) and approx variants for the end-to-end accuracy
>    A/B — never delete the exact path.
> 6. **GROUND A PRECISION/ALGORITHM CHANGE IN THE AUTHORITATIVE REFERENCE BEFORE ADOPTING IT.** "Run the pool in
>    bf16" looked plausible, but the real `model.py` says `x = x.float()  # compression need fp32` and there is
>    NO GPU compressor kernel; the bf16→fp32-exp pattern is the ATTENTION/indexer pattern, NOT the compressor.
>    A pattern from one op does NOT transfer to another — read the source op, don't assume. (bf16 also MEASURED
>    a non-win here: same speed, error 4 orders worse — because `exp` must be fp32 regardless.)
> 7. **ONE CONSISTENT SOURCE FOR THE PUBLISHED PAGE.** Never mix numbers from different jobs / build-configs /
>    methodologies in one comparison (the page once showed 176µs in §1 and 62µs in §4). Regenerate the whole
>    artifact from ONE job; label single-run; REPLICATE and flag the noisiest column (torch baselines under
>    streaming allocate temporaries → run ≥3× and take the cross-run median).
> 8. **AN APPROXIMATION'S CORRECTNESS IS THE GATE + EDGE CASES — not "faster and looks accurate on random
>    data."** The hand-`exp` was trusted only after the F4 hard gates (contract/finiteness/DETERMINISM/mask),
>    the all-masked guard, AND the measured ULP. Random-input accuracy HID the masked edge case until it was
>    inspected by hand — always red-team the edge the approximation changes.
> These are mechanized by: `kernel-authoring` (co-design rules 1–2 = streaming + objdump; transcendental/ULP
> rule), the CYCLE-EXIT standalone-kernel-page deliverable (below), and `shared/adversarial-self-audit`.

> **North-star deliverable (both legs).** Enablement is "done" ONLY when the new model
> RUNS end-to-end on CPU, is ACCURACY-parity vs a trusted reference (accuracy-oracle,
> real weights), AND ships a published ROOFLINE-TARGET-vs-MEASURED perf artifact at a
> labeled machine config. Authoring/tuning a kernel is a means; the proof is the running,
> accurate, perf-vs-roofline model in the enablement-certificate. `cpu-serving-integration`
> is the step that turns authored kernels into that running model.

> **⛔ MANDATORY PHASE ORDER (do PHASE 1 fully before the North-Star wall-time phase).** The North
> Star is end-to-end WALL TIME — but it is gated behind a hard prerequisite:
>
> **PHASE 1 — ALL-KERNELS-IN-OPTIMIZED-C/C++ (a REQUIREMENT, not a nicety).**
> - EVERY op / fused-op in the model's op graph MUST have a C/C++ kernel (no op left on a torch/
>   Python fallback). "A torch ref exists" is NOT coverage; authored-C/C++ coverage is the gate
>   (`coverage-gate` must be 100% in C/C++ before Phase 2).
> - Each kernel is OPTIMIZED AGAINST ITS KNOBS — the batch/token count **M** AND the **KV-cache**
>   context (S / topk / page length) — **across the full M-sweep**, toward **as close to roofline as
>   possible** (`roofline-validation` per op, per M). Publish per-kernel roofline-target-vs-measured.
> - **NO-REGRESSION FLOOR vs the torch fallback (hard):** at EVERY M the C/C++ kernel must be **≥ the
>   torch fallback** (faster, or at worst a TIE). At small M (e.g. M=1/M=8) a kernel may not beat
>   torch — that is acceptable — but it must **never be slower**. A kernel that regresses vs torch at
>   any M is NOT done; dispatch it (best-of / fall through to the library path at that M) so the floor
>   holds. Correctness is held throughout by the per-op acceptance gate (`accuracy-oracle` /
>   `f4_acceptance.py`, tie_eps=0) — a speedup that changes tokens is worthless.
> - RANK the Phase-1 order by ROI (standalone median × call-count, e.g. `f7_roi_ledger.py`); optimize
>   the heaviest / farthest-from-roofline kernel first. This is STANDALONE kernel work (microbench +
>   roofline), the only phase where per-op roofline — not wall time — is the target.
>
> **PHASE 2 — NORTH-STAR WALL TIME (only after Phase 1 is complete + reviewed).**
> - **FRAME THE KPI FIRST (ask the user):** fix the operating point **(input S, batch B=M, output G)** +
>   headline metric from the target regime — throughput serving (B large, decode off the M=1 cliff) vs
>   low-latency single stream (B=1), default BOTH (headline throughput, carry B=1 as the latency floor).
>   See `model-enablement-playbook` step 0a. M is the decode batch, NOT a free dial; S must be realistic
>   (it is the attention/indexer KV length). Report the headline tok/s AND the prefill/decode split of the
>   SAME run (prefill M=S·B compute-bound; decode M=B dominates wall time) for attribution + regression-safety.
> - Baseline = the all-C/C++ model with every optimization gate OFF (the kosher baseline), with the
>   **systemic config already correct** (threads bound ONCE at init, one rank per NUMA domain, NO
>   per-forward `set_num_threads` — clear the known affinity/NUMA pathologies on the BASELINE, not later).
>   Then **progressively integrate the kernels/optimizations ONE AT A TIME** (each env-gated default-OFF,
>   with a dummy-weight output-fingerprint gate per toggle), measuring WALL TIME and RECORDING the
>   improvement per toggle (`cpu-serving-integration`).
> - Do this on the **depth-reduced, full-width DUMMY-weight proxy** (`perf-proxy`) and extrapolate with the
>   **two-depth linear model** `t(L) = t_fixed + L·t_layer` — calibrate from TWO proxy depths (e.g. proxy4 +
>   proxy8: `t_layer=(t8−t4)/4`, `t_fixed=t4−4·t_layer`) PER REGIME (prefill/decode). NOT naive
>   `t_proxy/layers × full_layers` — that wrongly multiplies the per-forward fixed overhead (warmup/embed/
>   lm_head/framework) by the full depth. Known systemic overheads are cleared up front; the ONE full-depth
>   dummy run then VALIDATES the extrapolation and exposes any RESIDUAL depth-accumulating overhead
>   (allocator/thread-pool/cache-residency) to bisect — that is what the full-depth run is for, not to
>   discover the known pathologies.
> - When all kernels are integrated, measure the FULL-MODEL (dummy-weight) wall time **including the
>   systemic-config pathology pass** (`runtime-config-tuning`: threads/affinity/NUMA/prepack), optimize
>   it, and record the wall-time improvement. CONFIRM on the full model at the end; real weights only
>   for the accuracy + no-perf-regression finale. The dense per-M kernel sweep stays at the standalone/
>   roofline level + this one full-depth confirm; the E2E ledger tracks the chosen operating point.
>
> Do NOT jump to Phase-2 wall-time optimization while any op is still on a torch fallback or any
> kernel is below its no-regression floor — fix Phase 1 first.
>
> **BW-BOUND ≠ "SKIP AMX."** A low-M (CPU-serving) op being bandwidth-bound sets its *ceiling* (DRAM BW); it
> does NOT excuse a torch fallback or an untiled path. The op only *reaches* `Bytes/BW` when AMX compute HIDES
> under the stream (`t = max(FLOPs/P_compute, Bytes/BW)`) — a torch `P_compute` makes the compute term dominate
> (DSv4 routed expert @M=32: 38 µs stream, AMX = 13 µs hidden, **torch = 805 µs = 21× above the BW floor**). So
> every hot GEMM/op is an authored **C/C++ AMX/AVX-512** kernel (BRGEMM/grouped-BRGEMM), **tiled + double-buffered**
> (AMX consumes the resident tile while the next streams) and **fusion-folded** (cut the `Bytes` term) regardless
> of regime; the regime only changes the *streaming + thread-count* strategy, **adaptive in M** (M=1 → ≤4–8
> threads, GEMV+prefetch; M→64 → all cores, grouped BRGEMM). See the methodology playbook "BW-bound is the ceiling,
> not a license to skip AMX" rule + `docs/dsv4-gemm-regime-analysis.md`.
>
> **DONOR-FIRST (do NOT hand-roll what a library already provides).** Wherever an existing optimized
> primitive exists — **OneDNN** (eltwise/softmax/reorder), ATen **cpublas** brgemm + VNNI `pack` (oneDNN/
> MKL-backed), `at::bmm`, MKL, LIBXSMM/TPP — USE IT and author only the irreducible missing GLUE (fused
> epilogue, dispatch, tiling, gather, online-softmax composition). Never re-implement a matmul/softmax a
> library gives you. Author a net-new kernel ONLY when no donor primitive covers the contract
> (`kernel-authoring` + `assets/donor-kernel-map.md`); even then, build it from the nearest donor.
>
> **RUN PHASE 1 AS A PARALLEL WAVE.** The per-kernel standalone optimizations are INDEPENDENT work items
> (distinct source, distinct roofline) — fan them out in parallel (analysis + author + measure), ranked by
> ROI (`f7_roi_ledger.py`), and keep each ready. Integration (Phase 2) is the serial, one-by-one step.
>
> **⛔ FUSION GATE (every kernel, before it is "done").** A BW-bound op that writes an intermediate to
> DRAM and re-reads it has DOUBLED its traffic for nothing. For each kernel prove there is NO avoidable
> intermediate round-trip: keep the contraction in the donor GEMM but FUSE the epilogue (relu/scale/
> weight/softmax/reduction) so intermediates stay L1/L2-resident and never touch DRAM; compose multi-step
> math as ONE streaming pass (online-softmax, accumulate-once) so no full weight/score/prob tensor is
> materialized. Record, per kernel, which intermediates are fused-away vs. which residual round-trip
> remains and WHY it is irreducible (e.g. keeping it would forfeit a near-roofline library GEMM). Also
> flag CROSS-OP (adjacent-kernel) fusion candidates for Phase-2 integration (they need the caller context).
>
> **⛔ PRECISION / BANDWIDTH GATE (every kernel, before it is "done").** Exploit precision to cut the DRAM
> wall on BW-bound ops: STORE/READ the heavy operands (KV cache, activations, weights) in the LOWEST
> precision the reference tolerates, UPCONVERT in-cache for compute (bf16 AMX / fp16 with FP32 ACCUMULATE),
> and DOWNCONVERT on write-out. Two rules make this safe and free:
>   1. **Free-BW rule:** if the kernel's compute ALREADY rounds an operand (e.g. a bf16 AMX GEMM rounds the
>      fp32 KV it reads), then STORING that operand at the compute precision is numerically IDENTICAL to
>      what the kernel already consumes while HALVING its dominant read — take it (gated only on the
>      cache/operand format, a caller/integration knob). Never read fp32 only to immediately round to bf16.
>   2. **Accumulate-in-fp32 rule (SOURCE-STAGE semantics govern):** keep reductions/softmax/normalization
>      accumulators in FP32 for numerical stability even when inputs/outputs are low-precision, and downconvert
>      only the final store — UNLESS the published op rounds at INTERMEDIATE stage boundaries (e.g. the DSv4
>      indexer rounds to bf16 after the einsum, after relu, after the weight-multiply, and after the head
>      reduction). When it does, MATCH those stage boundaries, not a final-only downconvert: the source op's
>      per-stage dtype/rounding is the acceptance contract (conform the reference to it; a final-only
>      downconvert would diverge and change selections).
> Precision that changes the reference dtype (e.g. bf16 where the oracle is fp32) is a CONFORMANCE DECISION,
> not a free lever: the gate is the REFERENCE precision — if model.py runs the op in bf16/fp8, fp32 is
> over-precise and the low-precision ISA path (bf16 AMX, 124 TF vs 7.78 TF fp32) is the FAITHFUL one;
> surface it for the oracle-dtype decision rather than shipping a non-conformant "win". Keep selection/
> cutoff-sensitive ops (topk) and tiny non-BW-bound ops (sinkhorn/combine) in FP32.
>
> **⛔ PHASE-1 EXIT = PAUSE FOR REVIEW (do not auto-advance).** Phase 1 is done only when ALL kernels are
> optimized AND their performance-vs-roofline is RECORDED ACROSS THE SWEEPS (and the no-regression floor
> holds at every M). At that point STOP and surface for review. Proceed to Phase-2 proxy integration
> (fix correctness in-engine + extrapolate wall-time via progressive one-by-one integration) ONLY after
> the review completes.
>
> **BEFORE surfacing, run the `shared/adversarial-self-audit` pre-submit pass (BOTH clusters) — make the
> reviewer's three moves yourself first:** (1) conform every oracle/reference to the AUTHORITATIVE published
> op / real kernel at each stage (dtype / accumulation / block structure / output precision), not your own
> reconstruction; (2) test the ADVERSARIAL domain (signed weights, real shapes/K-values, sentinel/empty,
> every dispatch branch — coverage must EQUAL the claimed sweep, a benchmark at M is not an acceptance gate
> at M); (3) attach a CLAIM-EVIDENCE LEDGER (every perf/correctness claim → job+number → estimator
> [≥3-trial median+spread, never a single sequential mean] → the datum that would falsify it) and strike
> every inflated word ("bit-exact" w/o `torch.equal`, "noise floor"/"ratified" w/o a source-faithful ref,
> "every M"/"floor" w/ any sub-1.0 datum, "only/resolved/exhausted" w/o a falsification attempt). Bind each
> run to source/build/affinity identity and make harnesses fail-closed (pipefail; tested to FAIL on a
> known-bad run). A finding the reviewer could raise from the published source, the adversarial domain, or a
> claim-vs-evidence word check is one you were required to find here.

> **⛔ CYCLE-EXIT GATE — a perf-optimization cycle is NOT done until the published artifact PAIR
> exists (auto-emit it as the CLOSING STEP, do not wait to be asked).** The moment the final
> measured per-op profile is in hand, emit BOTH, from that profile, at the labeled machine config:
> (1) the **roofline-target-vs-measured** report+chart (`roofline_vs_measured.py`), AND (2) the
> **time-attribution pivot** companion (`time_attribution_pivot.py`, where the wall-clock goes per
> phase, summing to 100% with an explicit unattributed slice) — then ADD the model's bullet to the
> README "Roofline target vs measured" section (mirroring the existing entries) and publish. Both
> files are the deliverable; one without the other is incomplete. **A baseline→optimized→roofline
> "journey"/summary chart is NOT a substitute** — it answers "how much faster did we get," the
> required pair answers "how close to the ceiling, and where does the wall-time go" (the RoI view
> every published result links). *Anti-pattern that actually happened (GLM-5.3 Flash): the cycle
> ended with only the journey chart, which LOOKED like "charts done," so the DSv4-format roofline+pivot
> pair + the README results-section bullet were silently skipped and had to be back-filled.* Treat
> "generated a chart" as NOT satisfying this gate until the roofline report, the pivot report, and the
> README bullet all exist. If the hot op is recurrence/small-op bound (dispatch-limited, not a dense
> GEMM), the FLOP roofline is the WRONG reachable target — publish it but CAPTION the realistic
> engineering floor (e.g. fused-kernel) so "1% of achievable" is not misread as failure.

> **⛔ STANDALONE-KERNEL PERFORMANCE PAGE (required Phase-1 deliverable — produce one like
> `docs/kernel-pilot-performance.md`).** For the authored-kernel set, publish ONE reviewer-facing page that,
> for EVERY kernel, carries: (a) the **roofline** (byte/FLOP contract → floor vs nominal peak, with the bound
> class named: BW / compute / selection / dispatch / cache-resident); (b) the **baseline** (the torch/library
> op it replaces and WHY it is slow); (c) the **optimization + analysis** (what was authored, the measured
> speedup, % of roofline, and the regime verdict — "at ceiling" is a valid, required outcome, not a gap).
> NON-NEGOTIABLE methodology for the page:
> - **STREAMING (cold-cache) latency is the headline number** — flush ≥2× LLC before every timed call. Warm
>   reuse flatters L3-resident ops (compressor warm 37µs/cache-resident vs cold 62µs=77% DRAM roof). Quote
>   warm only as a labeled secondary (the fused-pipeline case).
> - **Classify by `objdump %zmm`** and state the build flags (gcc defaults 256-bit; at::vec needs
>   `-DCPU_CAPABILITY_AVX512`). The flagset the production kernels build with must be the one measured.
> - **Transcendental approximations carry a measured ULP** (vs fp64 over the real range; ~1 ULP bar for exp)
>   and KEEP BOTH exact(libm)+approx variants for the E2E A/B — state both on the page.
> - **Honest regime verdicts:** nominal peak is not an achievable ceiling; selection/dispatch/cache-bound
>   kernels show low %roof as a DIAGNOSTIC; don't claim headroom that the regime forbids. Single-run numbers
>   are labeled single-run; torch baselines under streaming are the noisiest column (say so).
> This page is a SEPARATE, standing deliverable from the roofline+pivot report pair above; all three plus the
> README bullet are required before the cycle is "done."

> **When the RUN is the bottleneck (long load / slow decode / cluster queue) — load the
> run-efficiency skills (apply to EVERY model, both legs):** `high-information-runs` (make each
> expensive run a multi-angle probe; and ALWAYS launch `scripts/await_job.sh` in the async terminal
> after submitting a long job so completion AUTO-WAKES the agent — never fire-and-forget then wait
> for a human to ask "is it done?"); `perf-proxy` (depth-reduced, full-width, DUMMY-weight perf
> ladder — perf iteration only, never accuracy). For TASK ACCURACY specifically, `accuracy-oracle`
> carries the node-parallel + chunked + checkpointed harness, the generate-once → score-OFFLINE
> decouple, and the certify-vs-trusted-reference rule (a bare score certifies nothing).

> **PRESERVE THE KOSHER BASELINE (do this BEFORE the first optimization).** The parity-checked,
> unoptimized CPU build is the reference for the whole perf phase — it must stay runnable and
> untouched. Three guards, all required: (1) **FREEZE the golden fingerprint** (the per-op tensors +
> logits from the unoptimized real-weight build) — the immutable correctness reference; (2) **TAG the
> baseline commit** (e.g. `git tag <model>-baseline-parity`) so the exact baseline CODE is recoverable,
> not just a floating SHA; (3) **make every optimization ENV-GATED default-OFF** (the repo convention —
> DSV4 ships ~47 `INTEL_CPU_*` gates) so gate-off reproduces the baseline byte-for-byte in the SAME tree,
> and develop the opt work on a **separate branch** merged back only after the full golden-parity +
> no-regression confirm. An optimization becomes the DEFAULT (gate flipped on / merged) ONLY after it
> passes the per-op golden-check AND the full-model re-diff. This is how "incrementally optimize +
> integrate, correct by construction" stays honest: the baseline is always one `git checkout <tag>` or
> one gate-off away, so any regression is bisectable to the single op that caused it.

> **Reference-wiring-FIRST.** Before optimizing, wire the full serving path with
> fallback/reference kernels and make it RUN + CORRECT on a **tiny architecturally-faithful
> config** (real arch switches, tiny dims, dummy weights — runs in seconds, same code paths).
> This front-loads the wiring/infra breaks (TP/NUMA, config, allocator, metadata) that static
> analysis misses and gives a running reference to A/B every later optimization against.
> Correctness backbone first, speed on top. See `cpu-serving-integration`.
>
> **ORDERING (correctness → perf → task).** Per-layer PARITY vs the reference baseline is the
> correctness gate FIRST (vs the GPU/HF oracle, real weights, one short prefill — cheap), then it
> GUARDS every optimization: on the dummy-weight `perf-proxy` the guard is the optimized-vs-reference
> A/B on the same input (valid on dummy — it tests the kernel transform, not accuracy), re-confirmed
> per-layer on the full OPTIMIZED model. TASK accuracy (real weights, full generation) is LAST. See
> `accuracy-oracle` CORRECTNESS ORDERING + the `perf-proxy` ladder.

> **TASK-ACCURACY SIGN-OFF — SURFACE THE METHOD AS A USER DECISION (don't silently pick).** The final
> task-accuracy step has two legitimate routes; PRESENT BOTH to the user and let them choose (cost vs
> external-validation tradeoff): **(A) Reproduce the published card number** — exact card protocol
> (shots, `reasoning_effort`/thinking budget, long generations, sampling). Gold-standard external
> validation, but for a frontier agentic/multimodal model this can be days of CPU compute or out of
> scope for the text path (and the card may not even report a cheap task like gsm8k). **(B) CPU-vs-GPU
> equivalence on a SHORT benchmark** — run the IDENTICAL harness (gsm8k) on the CPU build and the GPU
> reference and show they match (aggregate within <1 pt, high per-question agreement, SYMMETRIC
> disagreements, 0 invalid). Cheap, feasible, and a direct correctness proof of the CPU build vs the
> same-model reference — though it certifies *equivalence to the reference*, not an absolute leaderboard
> number. Recommend (B) when (A) is infeasible, but STATE the tradeoff and let the user decide. See
> `accuracy-oracle` (the CPU-vs-GPU cross-check + the PARITY-vs-CAPABILITY distinction).

> Two legs, one plugin. This playbook is the **new-kernel** leg (write/optimize a
> kernel). `model-enablement-playbook` is the **throughput** leg (wire a new model
> from EXISTING kernels). The enablement `coverage-gate` hands a genuine gap to
> THIS leg via the `cpu-optimizer` agent; when the new kernel passes its roofline
> it registers a capability contract and the model re-enters enablement.

> WRITING vs optimizing. To AUTHOR a net-new kernel (a coverage GAP with no CPU
> implementation, e.g. the DSA indexer), load **`kernel-authoring`** FIRST — it maps
> the op to the nearest donor kernel in the SGLang corpus and the 4-layer adaptation
> (control / inner-op / packing / epilogue), grounded in LIBXSMM/TPP + oneDNN. Then
> the technique skills below TUNE the drafted kernel.

> **One law, recurring at every level — FIT THE WORK TO THE RESOURCE, up front, from a
> MEASURED constant.** Each technique below is the same decision at a different level of the
> hierarchy: match the unit of work to the resource's capacity/shape BEFORE tuning, or overhead
> dominates. Decide these up front from `uarch-perf-probe` constants, not after:
> | Level | Fit the work so… | Resource constant (uPP) | Mismatch failure |
> |---|---|---|---|
> | Threads (grain) | work/thread ≥ sync/spawn floor | thread-scaling knee | inverse scaling (MoE 0.1ms@4 vs 2432ms@60) |
> | Cache (tile) | working set ≤ L2/core, reused panel resident | cache-ladder sizes | L2 spill → DRAM re-streaming |
> | SIMD/AMX (layout) | data in VNNI/tile order, dims divisible | tile shape / ISA | gather/stride stalls, padding waste |
> | Bandwidth (precision) | operand bytes ≤ BW can feed | per-domain BW, ridge | BW-starved compute unit |
> These are two-sided (too small wastes the unit; too big overruns it) — sweep the curve, it can
> be non-monotonic. `overhead-attribution` tells you WHICH level is the current bottleneck.

## The skill library (progressive order)

| # | Technique skill | Load when | Gates on |
|---|-----------------|-----------|----------|
| G | `kernel-feasibility-gate` | BEFORE authoring a net-new kernel | user-reviewed roofline + measured baseline; go/no-go |
| A | `kernel-authoring` | writing a NET-NEW kernel (not just tuning) | draft matches FP32 ref |
| S | `cpu-serving-integration` | novel kernels authored+validated; make the model actually RUN | model runs end-to-end + accuracy + roofline-vs-measured |
| 0 | `establish-achievable-performance` | ALWAYS first, once per hardware profile | sets the ceilings |
| D | `sub-numa-clustering` | after 0, BEFORE the tile/vectorize loop; multi-socket / SNC node | per-domain scaling + capacity fit |
| 1 | `roofline-validation` | after every implementation step | % of achievable |
| 2 | `openmp-parallelization` | op runs on >1 core | scaling efficiency |
| 3 | `cache-blocking-tiling` | working set > L2; streamed operands | L2/L1 residency |
| 4 | `amx-vectorization` | dtype∈{bf16,fp16,int8} and `amx_*` present | FLOP/cyc vs peak |
| 5 | `weight-prepacking-brgemm` | stationary operand (weights) reused across calls | close gap to achievable |
| 6 | `quantization-amx-int8` | accuracy budget allows lower precision | INT8 AMX peak |

Each skill is self-contained (capability contract, trigger, procedure, gate) so
new ops reuse them without modification, and new techniques are added as new
skill folders without touching the others.

## Decision flow

```
0. establish-achievable-performance  -> per-DOMAIN compute_peak + streamed_gemm + mem_bw ceilings
0.5 sub-numa-clustering -> FIX the SNC/NUMA domain to optimize WITHIN (capacity fit, tp=#SNC,
    one rank/domain, cpu+mem co-bind, first-touch). Steps below tune ONE domain, then replicate;
    the roofline ceiling is the EFFECTIVE (tp_used/n_domains)x full node, not the ideal node.
1. classify op + arithmetic intensity (roofline-validation: compute- vs memory-bound)
2. if memory-bound  -> cache-blocking-tiling, then re-roofline
   if compute-bound -> amx-vectorization (right ISA/tiling), then re-roofline
3. parallelize (openmp-parallelization): threads, affinity, NUMA/SNC
4. if weights reused (inference) -> weight-prepacking-brgemm  (usually the biggest
   single win for a lone large GEMM; see that skill's measured evidence)
5. if accuracy budget allows -> quantization-amx-int8
6. roofline-validation after each step; loop on the gap; record learned pattern
```

## Composition rule

Apply techniques in the order above and **re-run `roofline-validation` after each**.
Also **re-check numerical parity against the persistent reference oracle after each
step** (see `kernel-authoring` 1b): a slow-but-correct reference is authored/kept as
the correctness fallback, and any optimized variant that drifts from it is a
regression, not a speedup. Speed is only valid on top of correctness.
Stop when efficiency ≥ target (70% of the *streamed achievable* ceiling, not the
resident compute peak — see `establish-achievable-performance`). Record which
technique closed the gap in the skill's learned-patterns so the next op starts
from the winning combination.

## Worked composition

`cpu-gemm-amx-bf16` is a complete worked example that composes skills 0–5 for a
dense BF16 GEMM. Use it as the template when optimizing a new dense linear layer.
