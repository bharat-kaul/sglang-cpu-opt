# Response to the independent review — DSv4 decode fusion

**Author:** implementation side. **Verdict: I substantially AGREE.** The review found real,
verified errors in my analysis and tooling, not nitpicks. I accept the decision — **a
measurement-and-prototype phase, NOT the multi-week monolithic-layer build** — and I extend one
finding to a consequence the review did not state (it invalidates a *published* conclusion, §6).

I independently reproduced the review's key technical claims before responding:

## 1. Agreements — verified, with the check I ran

| # | Review finding | My verification | Verdict |
|---|---|---|---|
| a | The 29% is an **unattributed layer-local residual**, not a measured recoverable fusion benefit; "CONFIRMED recoverable" should be a hypothesis | The layer-scaling fit + 29% residual ARE valid (separate-process jobs, env set before Engine construction). But recoverability/mechanism is unestablished — I over-claimed. | **AGREE** |
| b | The C++ **outer per-layer loop is OUTSIDE `layer.total`**, so removing it can't be credited against the residual inside it | Correct — `_timed("layer.total")` wraps `Layer.forward`; my build-plan increment-1 targets the wrong thing. | **AGREE** |
| c | `hc_pre+hc_post` are **timed** work (12.4%), not residual; MHC fusion can't be double-counted as residual | Correct — they're in the TIMED buckets, not the `layer.total − Σtimed` gap. I conflated them. | **AGREE** |
| d | The incremental A/Bs have a **process-boundary control error**: `sgl.Engine` runs the model in a worker **subprocess**, so per-rep `os.environ` toggles in the client never reach it | Verified: `sglang.srt.entrypoints.engine` uses `mp.Process` / `run_scheduler_process` (subprocess). My `mask_ab_interleaved.py` and `validate_mhc_wire_increment1.py` set the toggle env **after** Engine construction → the worker never saw it → **both arms ran identically**. | **AGREE — invalid** |
| e | `diff_fullcap.py` is a **diagnostic printer, not a fail-closed gate** | Verified: **zero `sys.exit`** in the file → it always exits 0 even on NaN/opposite tensors. | **AGREE** |
| f | FULLCAP **doesn't cover the B=32 decode target** | Verified: capture gated on `2<=T<=16` (prefill) or `T==1` (decode); B=32 decode has T=32 → never captured. The `dc//64` label is also wrong for 43 layers. | **AGREE** |
| g | A single outer `omp parallel` is **not transparent reuse** — ATen checks `!at::in_parallel_region()` and will **serialize** (or race/duplicate); level-4 is an architecture experiment, not a medium-risk wrapper | Correct and I underweighted it; my "wrap in one region" hand-waved this. | **AGREE** |
| h | Full-socket BW is **aspirational for an 8-thread config**; use config-matched BW and `max(bytes/BW, work/compute)`, don't label the whole gap-to-ideal as overhead | Correct — my "20× above BW floor" used 329 GB/s (full socket); at 8 threads ~250 GB/s, and part of the residual is copies/allocs/metadata, not pure park/wake. | **AGREE** |

## 2. What SURVIVES (still-valid evidence)
- The **29% residual and its per-layer scaling** (`decode_step_ms = 47.9 + 75.8·N`) — valid (separate-process jobs).
- The **thread sweep** (decode worsens with more cores) — valid: it used a **control FILE**
  (`INTEL_CPU_DSV4_DECODE_THREADS_FILE`), which the worker re-reads from disk each forward, so it DOES
  cross the process boundary (unlike `os.environ`). This is the correct pattern the review endorses.
- The **per-op attribution** (job 393902) — valid (single fixed-env process).
- The **offline mask-vectorization fuzz test** (4000 bit-exact cases) — valid (pure function, no engine).

## 3. What is now UNKNOWN (previously claimed, evidence invalidated)
- **MHC sinkhorn/combine wiring = "+0.1%"** → invalid (process-boundary). Effect unknown; likely small by
  Amdahl (26 µs ops), but must be re-measured with a worker-side control.
- **Mask vectorization = "0% e2e"** → invalid (same bug, in `mask_ab_interleaved.py`). **See §6.**

## 4. Where I add nuance (not push-back)
- The **"small per-op swaps are ~0%"** conclusion is *probably* still true by magnitude (sinkhorn is 26 µs of
  a ~2.4 s step), but the **A/B that "proved" it was invalid** — I agree it must be re-established validly, and
  I withdraw it as *evidence* even though the Amdahl expectation stands.
- My own review-request §4/§5 already flagged the level-1-vs-level-2 uncertainty and the Amdahl caveat; the
  review is right that the **build plan doc contradicted that by saying "CONFIRMED."** I will make the docs
  consistent (hypothesis, not confirmed).

## 5. I have no substantive disagreement
Every point I could check is correct. Attempting to defend the invalidated A/Bs or the "confirmed 29%"
would contradict the evidence. This is a case to accept, not litigate.

## 6. Consequence the review did not state — a PUBLISHED conclusion is affected (class-level fix)
The `os.environ`-toggle-across-a-subprocess bug is a **class**, not two instances. It also invalidates the
**earlier mask A/B (job 393518)** whose "0% e2e" conclusion is **published on `main`** in
`docs/e2e-wall-time.md` (it drove "mask lever refuted" + "withdraw the ~275 prefill achievable" + a
methodology "LAW"). That conclusion now rests on an invalid measurement. **Action:** correct the published
docs to mark the mask-A/B result UNVALIDATED (the *bit-exactness* stands; the *perf* claim does not), and
re-run every `os.environ`-interleaved A/B with a **worker-side control** (control file / per-op CK file /
worker-local replay). The thread-sweep and separate-process jobs are unaffected.

## 7. Adopted plan (per the review's roadmap)
1. **Fix experimental controls:** replace all client-side `os.environ` interleaving with worker-side control
   files (+ branch counters proving the arm changed; reject fallback-contaminated runs).
2. **Build a real fail-closed parity gate:** nonzero exit on any cos/rel-err/shape/dtype/finiteness failure;
   exact request/token/rank coordinates, no duplicate-key overwrite; **cover B=32 decode + multi-step +
   compression boundaries**; exact equality target for a pure control-flow port.
3. **Reconcile the 29% with an identity-bound nested timeline** of one real decode layer (native calls, norms/
   rope/MHC, copies, alloc, cache/metadata, thread activity), retaining call counts + nesting; measure probe
   perturbation. Decompose into dispatch / standalone-region / inter-big-kernel park-wake.
4. **Config-matched roofline:** B=1 and throughput batches separately; measured socket memory traffic +
   useful-byte rate; BW measured at the SAME 8-thread/affinity/placement config; `max(bytes/BW, work/rate)`.
5. **Investigate MoE inflation (33%, the comparable Amdahl lever) in parallel** with ONE small glue prototype
   (original vs C++-driver-same-ops vs one genuinely-fused glue segment, identical operands, worker-confirmed).
6. **Escalate to persistent-team / monolithic only with measured justification** from (5).

Net: the north star is unchanged — faster correct decode + higher useful socket-BW efficiency — but the
next investment is **measurement repair + attribution**, chosen by demonstrated gain vs risk, not a prior
commitment to monolithic fusion. Thank you for the rigor; it caught a real measurement-validity class bug.
