# Standalone Kernel Optimization Playbook (CPU / Intel Xeon)

How to take **one operator** from its torch/library baseline to **as close to the roofline as its regime
allows** — a staged, *measure-after-every-step* approach. This is the **Phase-1 "how"** of the
[methodology](methodology-playbook.md): it expands the leaf skills the `cpu-optimization-playbook` spine routes
through, in the order it routes them.

> Distilled from our own spine-leaf skills (`cpu-optimization-playbook`, `kernel-authoring`, and the
> `kernel-optimization/` + `shared/` leaves) — the generally-useful HPC-kernel ideas are already encoded there.
> Worked results for the six DSv4 DSA kernels are on [kernel-pilot-performance.md](kernel-pilot-performance.md).

**Rules of the road** (apply to every stage):
- **Work the stages in order.** Each is more complex and *less portable* than the last; only advance when the
  earlier stage is exhausted or clearly cannot help. **Re-run the roofline after every stage.**
- **Keep the reference runnable.** The torch/library op the kernel replaces is the baseline *and* the
  correctness oracle — never delete it; A/B every change against it.
- **Measure correctly or not at all** (Stage 0): streaming/cold-cache, right build flags, `objdump`-verified
  ISA, replicated. A speedup measured wrong is worse than none.
- **Target per component, not the total %.** Decompose each kernel into **data-movement + compute +
  dispatch/overhead** by measured attribution and target *each*: data-movement ≥ 75% of the **operative** BW
  (DRAM if the working set exceeds cache; **cache BW if it fits** — not nominal DRAM), compute ≥ its ISA/dtype
  peak, overhead → 0. The roofline-% of the *total* latency is a diagnostic of which component dominates, not
  the target — and the nominal roofline can be the **wrong ceiling** (cache-resident / dispatch-bound ops).
- **A bound is a tested hypothesis, not a wall.** A measured bound prompts a **config sweep** (tiles / grain /
  effective-threads, via `grain_size`/task-count — **never per-forward `set_num_threads`**, an affinity cliff):
  it either *moves* the bound (ship the per-M optimum) or returns negative. "At ceiling" is valid **only after**
  the sweep shows no move *and* the attribution-motivated knob — **precision** where it hits the dominant
  component, or **authoring our own donor** inner loop when measurement localizes the bottleneck there — has
  been measured (a win *or* a published loss). Closing on assertion is the anti-pattern.
- **Lead with M=1, then sweep M=2–64.** M=1 has the highest overhead-to-work ratio, so it exposes the fixed-cost
  pathologies (alloc, dispatch, per-call pack) most clearly, and **techniques carry** (overhead-killing, fusion,
  access-pattern fixes generalize) while **code/config/ISA do not** (the BW ceiling cache→DRAM, the compute ISA
  AVX↔AMX, and the per-M config are earned fresh — AVX-512 often beats AMX at M=1 where tiles underfill). Gate:
  **best-case per M, bit-exact throughout, no regression at the throughput point M=B.**
- **The best ISA is per-(target, shape, M) — probe it, sweep it, dispatch adaptively.** AMX is *not* always best:
  it underfills **narrow or small-M GEMMs** where AVX-512 wins. **Probe** the target's CPUID flags
  (`uarch-perf-probe`; never infer the ISA from the part name — EMR has AMX **and** AVX-512-bf16/fp16; a client
  like Panther Lake has **no AMX**), **M-sweep** each candidate ISA at the real GEMM shape, **find the crossover**,
  and **dispatch the per-M optimum**. The donor dispatches its own ISA; your authored glue owns its ISA and needs
  an AVX2/scalar fallback for AMX/AVX-512-less targets. **Conformance-gate the switch** (ISA paths aren't
  bit-exact; a precision change like fp32-vs-bf16 is Phase-3-gated on *realistic* distributions; a dtype proxy
  like fp16-for-bf16 is perf-only). *Worked: indexer score GEMM — AVX-512 wins 2.6×@M1→AMX@M64 (narrow K=128/N=64);
  counter-example: sparse's fat GEMMs are already AMX-optimal, so the rule excludes them.*
- **Optimize the graph, not the node.** Your kernel is a **node** in the model's dataflow graph (discovered +
  externally verified in Phase 0). Before you trust a standalone result, check its graph context: the **real
  input dtype/layout the pipeline supplies** (a standalone fp32/contiguous assumption may be an *artifact*),
  the **producer→consumer fusion** available, the **shared operand** you can read once, and whether this node
  is even the **pipeline's bottleneck** (optimize the one that scales with the hard dimension). Enumerate the
  decision space the graph seeds (local axes: donor/ISA/precision/algorithm/layout/fusion; graph axes:
  input-dtype/cross-kernel-fusion/shared-residency/pipeline-priority) and **dispose of every axis explicitly**
  — an untested alternative is *named*, never silently absent. See
  [methodology §4 rule 11](methodology-playbook.md#4-per-kernel-performance-gap-ledger-model-agnostic).
- **Tiling has two faces — overhead *and* reuse.** Sharding into tiles fills cores and cuts per-call overhead
  (fewer/bigger tiles at low M), but a tile also lets you **reuse a loaded operand from cache instead of
  re-reading it from DRAM**. Where an operand is re-read (BW-bound, reuse factor > 1), block/order the loop so
  it is read **once** and reused — this raises effective arithmetic intensity and moves the op right on the
  roofline, **capped at the compute ridge** (reuse turns a BW-bound op compute-bound, then stops). **Predict**
  the floor first (reuse factor = total reads ÷ unique bytes → floor `redundant_bytes/BW` → `unique_bytes/BW`),
  size the tile jointly for reuse + cache-fit + core-fill, re-assess per M, own the *composition* not the
  donor's inner loop, and **validate on measured DRAM traffic** (operand read ≈ once), not latency alone. See
  Stage 3 and `cache-blocking-tiling`.

---

## The stages at a glance

```mermaid
flowchart LR
    S0["0 · Baseline &amp; measurement"] --> S1["1 · Classify (roofline)"]
    S1 --> S15["1.5 · Validate upside\n(cost-scaled ladder)"]
    S15 --> S2["2 · Donor-first"]
    S2 --> S3["3 · Parallelize &amp; tile"]
    S3 --> S4["4 · Fuse (kill DRAM round-trips)"]
    S4 --> S5["5 · Precision / BW"]
    S5 --> S6["6 · Vectorize / AMX / intrinsics"]
    S6 --> S7["7 · Prepack / BRGEMM"]
    S7 --> S8["8 · Validate &amp; publish"]
    S1 -.regime decides which stages help.-> S4
```

| stage | purpose | our skill(s) | gate that makes it trusted |
|---|---|---|---|
| 0 Baseline & measurement | a trustworthy number to optimize against | `establish-achievable-performance`, `high-information-runs` | streaming + flags + `objdump` ISA + sensitive correctness |
| 1 Classify (roofline) | name the bound; pick the right lever | `roofline-validation`, `model-roofline-analysis` | floor computed at the **right** hierarchy level |
| 1.5 Validate upside | prove the lever *wins* before committing (cost-scaled) | `kernel-feasibility-gate`, `overhead-attribution` | whole-op component model; predicted≠measured; Tier-1 microbench before an expensive pass |
| 2 Donor-first | never re-implement a library primitive | `kernel-authoring` (+ `assets/donor-kernel-map.md`) | donor used where it exists; author only the glue |
| 3 Parallelize & tile | fill the cores; keep the working set in cache | `openmp-parallelization`, `cache-blocking-tiling`, `sub-numa-clustering` | all cores used at small M; one rank/NUMA domain |
| 4 Fuse | eliminate avoidable DRAM round-trips | `inter-kernel-fusion` + **FUSION GATE** | no materialized intermediate; residual justified |
| 5 Precision / BW | move fewer bytes at the reference's tolerance | `compute-from-native-precision`, `quantization-amx-int8` + **PRECISION/BW GATE** | fp32-accumulate; dtype change is a *conformance* decision |
| 6 Vectorize / AMX / intrinsics | use the full ISA; inline transcendentals | `amx-vectorization`, `cpu-gemm-amx-bf16` + **TRANSCENDENTAL RULE** | 512-bit confirmed; ULP measured; both exact+approx kept |
| 7 Prepack / BRGEMM | amortize layout; AMX-efficient contraction | `weight-prepacking-brgemm` | pack once; brgemm dispatched (`ONEDNN_VERBOSE`) |
| 8 Validate & publish | prove correct + record vs roofline | `roofline-validation`, `accuracy-oracle`/`f4_acceptance` | no-regression floor; F4 gates; the kernel page |

---

## Stage 0 — Baseline & measurement setup

**Purpose.** Produce a *reliable, reproducible* number to optimize against — and the reference to A/B against.

**How (our skills + concrete moves):**
- **Pin the baseline:** the torch/library op the kernel replaces, at the kernel's exact arithmetic contract
  (same shapes/dtypes/mask/output boundary). This is both the perf baseline and the correctness oracle.
- **Build flags (standard):** `-O3 -march=native -fopenmp -mprefer-vector-width=512 -DCPU_CAPABILITY_AVX512`.
  GCC defaults to 256-bit and ATen `at::vec` falls back to AVX2 in a JIT extension without the last two —
  `objdump -d <so> | grep -c %zmm` to **confirm the binary is actually 512-bit** (a mis-built "vectorized"
  variant was once wrongly rejected).
- **Measure STREAMING (cold cache), not warm.** Flush ≥2× the LLC before every timed call so inputs stream
  from DRAM. Warm buffer reuse lets an L3-resident working set re-read from cache and *flatters* the result
  (e.g. 37µs warm vs 62µs cold). Quote warm only as a labeled secondary (the fused-pipeline case).
- **Sensitive correctness:** random **and** heavy-tailed **and** edge inputs (masked `−∞`, all-masked, tails
  where `N mod VL ≠ 0`); a **mutation test** (perturb one input → output must change); `f4_acceptance`.
- **Problem size vs LLC:** compute the working set and compare to L3 — pick sizes deliberately above/below it
  and state which (a win that collapses once data leaves the LLC is the memory-bound signature).
- **Replicate:** ≥3 runs, cross-run median; flag the noisy column (torch baselines allocate temporaries).

**Gate:** the measurement obeys first-principles ordering and the right methodology — else fix the instrument
before optimizing.

## Stage 1 — Classify the kernel (roofline)

**Purpose.** Name the bound so you apply the *right* lever; most of the win comes from choosing correctly here.

**How:** compute the byte/FLOP contract → `floor = max(bytes/BW, FLOPs/peak)` vs the nominal peak; the
arithmetic intensity vs the ridge `peak/BW` tells BW- vs compute-bound. Then classify the **regime** and route:

| regime | signal | lever (stages that help) |
|---|---|---|
| **memory-bound @ DRAM peak** | achieved ≈ 80–100% DRAM BW | reduce traffic: **fuse** (4), **precision** (5), recompute vs load |
| **memory-bound below peak** | low BW, strided/indirect access | fix access pattern: **unit-stride/tile** (3), gather-not-scatter |
| **compute-bound** | AI > ridge | less work first (**fuse**/algebra), then **ISA/AMX** (6), **prepack** (7) |
| **latency / dispatch-bound** | tiny per-item work, many ops | **batch**, fully unroll, **fuse adjacent ops** (4); FLOP roofline is the WRONG ceiling |
| **cache-resident** | measured BW **above** DRAM peak | say so; DRAM tricks (streaming stores) *hurt* here |

> **A surprising roofline-% can be WRONG — question it.** One kernel read "27% of roofline / BW-bound" but was
> actually *exp-compute-bound AND cache-resident* (17MB fits L3). Two cheap checks: (a) working set vs LLC;
> (b) measured BW *above* DRAM peak ⇒ cache-resident, not a win. Beating your own "floor" means it was computed
> against the wrong level of the memory hierarchy.

**Gate:** the floor is computed at the hierarchy level the data actually comes from; the regime is named, not
assumed — and "already at the ceiling" is a *valid* outcome (don't invent headroom the regime forbids).

## Stage 1.5 — Validate a lever's upside before committing (cost-scaled)

Classifying the bound (Stage 1) tells you *which* lever; it does **not** prove the lever will *win*. A roofline
that models **one component** (e.g. a GEMM's FLOPs) over-predicts when the op has other first-order costs the
lever doesn't touch — or *worsens*. *(Real case: sparse-attend's bf16-AMX looked like 2.6–4× from the GEMM
compute peak, but the op's value-GEMM, the fp32↔bf16 conversion casts, and small-M dispatch are first-order;
three bf16 configs all measured **slower** than the fp32 donor.)* Two rules and a cost-scaled ladder:

- **Rule 1 — model the WHOLE op, not one component.** Before claiming upside, decompose the op into its
  components (GEMM₁, GEMM₂, convert, softmax, reduce, dispatch), apply the lever to each, and confirm it moves
  the **dominant** component *and does not inflate the others*. A lever aimed at a non-dominant component (the
  GEMM when the op is already at 60% of roof) cannot deliver a large win however big its *local* speedup.
- **Rule 2 — predicted ≠ measured.** A roofline upside is a **hypothesis**, not a result. Label it "predicted,
  unvalidated" until a measurement promotes it — never publish a predicted multiple as an outcome.

**The ladder (cheapest-first; stop as soon as a tier kills the lever):**
1. **Tier 0 — analytical (free):** the whole-op component model above.
2. **Tier 1 — isolation microbench (~minutes):** measure the lever's *target primitive* in isolation at the
   **real shapes** (e.g. bf16 vs fp32 `at::bmm`, both GEMMs). Catches "the primitive doesn't actually get
   faster" (wrong ISA dispatch, small-M pathology) before any fused kernel is authored.
3. **Tier 2 — minimal variant (~1 job):** the simplest end-to-end form, measured **whole-call incl.
   conversions**. Catches interaction effects an isolated test can't (cache residency, dispatch, materialization).
4. **Tier 3 — full:** optimized variant + tolerance + replicate.

**When to run which — the ladder's cost scales with the *full pass's* cost.** **Cheap pass** (hours — most
standalone kernels): skip the formal gate, *measurement is the smoke test* — but still do Tier 0 (free) and obey
Rule 2. **Expensive pass** (multi-day port, net-new kernel from scratch, scarce-HW time): **Tier 1 is mandatory**
before committing. **Emergent levers** (cross-op fusion, cache-residency, dispatch restructuring) have **no
faithful cheap proxy** — the integrated measurement *is* the test; don't fake a Tier-1 for them.

**Gate:** no upside is published as a *result* without a measurement (Rule 2); for an expensive pass, Tier 1
passes before the full authoring is committed.

## Stage 2 — Donor-first

**Purpose.** Never re-implement a matmul/softmax a tuned library already provides.

**How:** where an optimized primitive exists — OneDNN (eltwise/softmax/reorder), ATen **cpublas BRGEMM** +
VNNI `pack`, `at::bmm`, MKL, LIBXSMM/TPP — **use it** and author only the irreducible glue (fused epilogue,
dispatch, tiling, gather, online-softmax composition). Author a net-new kernel only when no donor covers the
contract (`kernel-authoring` + `assets/donor-kernel-map.md`); even then build from the nearest donor. *In the
DSv4 set, indexer-logits and sparse keep their GEMMs on the oneDNN/`at::bmm` donor and author only the epilogue.*

**Gate:** the donor is used wherever it exists; the authored code is the missing glue, not a re-rolled library.

## Stage 3 — Parallelize & tile

**Purpose.** Use every core even at small batch, and keep each task's working set in L1/L2.

**How:** choose the parallel-region granularity so **all cores are filled even at M=1** — parallelize over the
*independent* axis (e.g. the compressor parallelizes over N×D-tiles, not N alone, so small batch still saturates
64 cores). Tile the reduction so the per-task working set is L1/L2-resident; serial-threshold tiny-total-work
cases (avoid thread-team spin-up that *regresses* at M=1). Pin threads once at init; one rank per NUMA domain
(`sub-numa-clustering`); first-touch each rank's data into its own socket.

**Tiling has two faces — pick the tile for both.** The *count* of tiles trades off overhead (fewer/bigger tiles
cut per-call dispatch at low M; see the M=1 rule), while the *size* of a tile sets **data reuse**: a tile held
in cache can be reused across the inner loop instead of re-read from DRAM. Where an operand is re-read
(BW-bound, reuse factor > 1), **block and order the loop so it is read once and reused** — e.g. the indexer
packs `q^T` once per request and reuses it across every S-tile, and keeps the scores L2-resident so the
epilogue reads them from cache (no DRAM round-trip). **Predict** the gain first (reuse factor = total reads ÷
unique bytes → the BW-floor drops from `redundant_bytes/BW` to `unique_bytes/BW`, **capped at the compute
ridge** — reuse turns a BW-bound op compute-bound, then stops), then **validate on DRAM traffic** (achieved
bytes ≈ unique operand bytes, not reuse× that — the `cache-blocking-tiling` gate). Re-assess per M: at small M
the working set is often already cache-resident (reuse is free; the lever is overhead), at large M it spills
cache so you must block to keep reuse before eviction.

**Gate:** cores filled at small M; no small-M regression vs the serial/torch path.

## Stage 4 — Fuse (kill avoidable DRAM round-trips)

**Purpose.** A BW-bound op that writes an intermediate to DRAM and re-reads it has *doubled* its traffic for
nothing.

**How (`inter-kernel-fusion` + the FUSION GATE):** keep the contraction in the donor GEMM but **fuse the
epilogue** (relu/scale/weight/softmax/reduction) so intermediates stay L1/L2-resident and never touch DRAM;
compose multi-step reductions as **one streaming pass** — online-softmax (running max/denom/accumulator,
never materialize the weight tensor), accumulate-once tiling (read x once, write y once). *This was the
compressor's first and biggest structural win: ~6 passes + a materialized `w[N,R,D]` DRAM temporary → one pass.*

**Gate (FUSION GATE):** prove there is **no avoidable intermediate round-trip**; record the irreducible
residual and *why* (e.g. keeping it would forfeit a near-roofline library GEMM).

## Stage 5 — Precision / BW

**Purpose.** Move fewer bytes by storing/reading at the lowest precision the *reference* tolerates.

**How (`compute-from-native-precision` + PRECISION/BW GATE):**
- **FREE-BW rule:** if the kernel's own compute *already rounds* an operand (a bf16 GEMM rounds the fp32 KV it
  reads), then storing that operand at compute precision is **numerically identical** to what the kernel
  consumes while halving the dominant read — take it (a caller/cache-format knob). *E.g. indexer bf16-KV is a
  bit-exact free win.*
- **ACCUMULATE-IN-FP32:** reductions/softmax/normalize accumulators stay fp32 for stability; downconvert only
  the final store.
- **Conformance, not free:** a precision that changes the *reference* dtype (bf16 where the oracle is fp32) is
  a **conformance decision**, not a free lever — gate on the reference precision. *Ground it in the source:
  "run the compressor pool in bf16" contradicts `model.py` (`x = x.float()  # compression need fp32`) and
  measured a non-win (same speed, 4-orders-worse error, since `exp` must be fp32 regardless).*

**Gate (PRECISION/BW GATE):** fp32-accumulate preserved; any reference-dtype change is surfaced as a
conformance decision, not shipped as a silent "win"; measure before assuming (a precision lever only helps a
*read-BW-bound* op).

## Stage 6 — Vectorize / AMX / intrinsics

**Purpose.** Use the full ISA; stop relying on the compiler for the hot transcendental.

**How (`amx-vectorization`, `cpu-gemm-amx-bf16` + TRANSCENDENTAL RULE):** confirm 512-bit (`objdump %zmm`);
force FMA; register-block the accumulators (stay under 32 zmm to avoid spills); unit-stride, SIMD-lane-innermost
inner loops. For a transcendental-bound kernel, **hand-inline the `exp`/`erf`/`tanh`** (an industry-standard
minimax — Cephes/Sleef-class range-reduce + FMA poly + `2^n` bit-scale) to remove the non-inlined `at::vec`/
Sleef **call boundary** — that was the whole compressor Stage-6 win (at::vec Sleef 80.7µs → inlined Cephes
36.9µs warm, 2.2×). **Bound and keep both:**
- **Measure the ULP** vs an fp64 reference over the *real* argument range — **≈1 ULP bar for `exp`** (0.5 ULP /
  correctly-rounded is impractical for a hot vector kernel and isn't met by the reference stack either: GPU
  `ex2.approx` ≈ 2 ULP). A reduced-degree "fast" variant (several ULP) is acceptable only if the downstream
  (bf16/fp4 quant) demonstrably swamps it AND it is gated.
- **Guard the edge cases the approximation changes** (`exp(−inf) → ~1e-38`, not 0 → an all-masked guard).
- **Keep BOTH** the exact-`libm` path (bit-exact oracle) and the approx for the end-to-end accuracy A/B —
  never delete the exact path.

**Gate (TRANSCENDENTAL RULE):** industry-standard poly, measured ULP within bar, edge cases guarded, both
variants retained, F4-screened until the E2E task metric confirms.

## Stage 7 — Prepack / BRGEMM

**Purpose.** Amortize weight layout and feed the AMX contraction efficiently.

**How (`weight-prepacking-brgemm`):** VNNI-pack the reused operand **once** at load (bf16 VNNI2 `[K/2,N,2]`,
int8 VNNI4 `[K/4,N,4]` + per-col INT32 compensation for the u8×s8 trick); loop tiles so the brgemm M is large
(AMX-efficient) while enough parallelism remains at small N. For gather-GEMM, stage gathered rows into a
contiguous VNNI temp then brgemm (AMX has no gather) — but only when `M ≥ TILE_M` and the staged copy amortizes.

**Gate:** pack is done once (amortized); the intended ISA actually dispatched (`ONEDNN_VERBOSE`, throughput
magnitude — never trust capability flags alone).

## Stage 8 — Validate & publish

**Purpose.** Prove the optimized kernel is correct and record how close to the ceiling it got.

**How:** re-run `roofline-validation` after the final stage; enforce the **no-regression floor** vs the
torch fallback at *every* M (faster or at worst a tie; else dispatch to the library path at that M); run
`f4_acceptance` (output contract, finiteness, run-to-run **determinism**, selection `tie_eps=0`, screened
continuous vs a source-conformed oracle). Then publish the per-kernel record on the
[standalone-kernel performance page](kernel-pilot-performance.md): roofline floor + baseline + optimization +
analysis, streaming headline latency, the lever(s) applied, and the honest regime verdict. **Emit one
[performance-gap ledger](methodology-playbook.md#4-per-kernel-performance-gap-ledger-model-agnostic) row per
kernel** — classify the roofline-vs-measured gap (RECOVERABLE / WRONG-CEILING / AT-CEILING), attribute the
root cause, state the opportunity + expected effect, and assign the **owning phase** (kernel-authoring here,
or hand off to the Phase-2 integration backlog / Phase-3 accuracy ratification). A kernel is not "done" until
its gap is classified and any recoverable gap is mirrored to its owning queue. **Also publish a "what didn't
work, and why" record** (the attempted lever, its measured number, the root cause) alongside the wins — the
*why* of a rejection is as valuable as a win: it stops the same dead end being re-proposed.

**Gate:** no-regression floor holds; F4 hard gates pass; the published page exists (a chart alone is not the
deliverable — the roofline+pivot report pair + this page are); **every kernel has a gap-ledger row with an
owning phase** (no un-triaged gap).

---

## What's different from a generic HPC-kernel guide

This playbook keeps the universal spine (measure → classify → algorithm/layout → intrinsics → validate) but is
distilled around **our** CPU-serving reality, which adds first-class stages a generic guide omits:
**donor-first** (Stage 2 — use the tuned library, don't re-roll it), the **fusion gate** (Stage 4), the
**precision/BW gate with the free-BW + conformance rules** (Stage 5), the **transcendental ULP + keep-both
rule** (Stage 6), and **correctness as a hard gate** throughout (`f4_acceptance`, `tie_eps=0`, source-conformed
oracle) rather than a closing checksum. Data-layout (AoS→SoA) folds into Stages 3/6 — our kernels are already
unit-stride SoA, so it is usually a non-issue, but check it when a loop will not vectorize.
