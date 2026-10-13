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

---

## The stages at a glance

```mermaid
flowchart LR
    S0["0 · Baseline &amp; measurement"] --> S1["1 · Classify (roofline)"]
    S1 --> S2["2 · Donor-first"]
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
its gap is classified and any recoverable gap is mirrored to its owning queue.

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
