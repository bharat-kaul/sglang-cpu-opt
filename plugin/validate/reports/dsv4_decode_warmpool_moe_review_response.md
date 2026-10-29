# Response to the warm-pool / MoE review (B ≤ 64) — AGREE on all points

**Re:** [independent review](dsv4_decode_warmpool_moe_independent_review.md) of the
[pivot assessment](dsv4_decode_warmpool_moe_pivot_assessment.md). I independently reproduced every load-bearing
claim and **agree with all six**. The review is accepted in full; the pivot (MoE first, defer the monolith)
stands with corrected numbers and a corrected mechanism. Below: what I verified, what I'm fixing, and the
adopted plan.

## Verifications (reproduced, not just accepted)

1. **§1 Absolute rates were wrong — a real code bug.** `validate_decode_threads_clean.py` (and the MHC A/B
   driver) normalized `(total_k − total_1)` by `output_len` instead of `output_len − 1`. Token 1 comes from
   prefill/extend, so that difference spans **k−1** decode steps. Reproduced from the saved samples + `tpref`:
   | B | cold (mine → corrected) | warm (mine → corrected) |
   |---|---|---|
   | 1 | 502 → **602 ms**, 1.66 tok/s | 202 → **242 ms**, 4.12 tok/s |
   | 32 | 2690 → **3228 ms**, 9.91 tok/s | 1230 → **1476 ms**, 21.68 tok/s |
   The attribution's **direct worker-step** timing (warm B=32 ≈ 1469 ms) matches the corrected 1476 — so the
   worker-step number is authoritative, exactly as the review says. **Relative win unchanged: 2.48× / 2.19×.**
   *Fix landed:* both drivers now divide by `output_len − 1`; going forward I will prefer worker-step timings
   with real output counts and add order-balanced **process** replication (six tokens in one process ≠ six
   replications).

2. **§2 Prefill regression is real, not hypothetical.** Raw attr JSONs: prefill accumulates **14.60 s cold vs
   72.70 s warm (~5×)**. Both arms used the sticky cap, so this is not sticky alone — the warm spin and/or the
   sticky residue (decode leaves threads at 8, so the *next* prefill inherits 8) is implicated. This refutes my
   "strictly better on a dedicated socket" wording. *Adopted:* measure **first-prefill / later-prefill /
   sustained-decode / mixed** separately, identify the loaded OpenMP runtime, and compare park vs finite-spin vs
   infinite-spin, recording **effective worker** thread counts/affinity — not the client's. Also: sticky was a
   measurement device, not a deployment config; deployment must restore full threads for prefill.

3. **§3 Native MXFP4 ≠ AMX — I inferred ISA from a name, which the review rightly forbids.** Confirmed in the
   installed source: `moe.cpp:437` `avg_M = M*topk/E`, `use_brgemm = can_use_brgemm<scalar_t>(avg_M)`, MXFP4 at
   `moe.cpp:834`, and `gemm.h` `can_use_brgemm<uint8_t>` / `<Float8_e4m3fn>` both `return M > 4`. With top-6 /
   256 experts, `avg_M = B·6/256 ≤ 1.5` for all B ≤ 64 → **use_brgemm is false throughout → the non-BRGEMM
   vector path runs, not AMX.** So "underfilled AMX tiles" is withdrawn. *Adopted:* instrument the actual branch
   (vector vs BRGEMM) + per-expert occupancy from the loaded binary before claiming a cause.

4. **§4 "44 ms/apply" was misnormalized; the 6–10× floor is withdrawn pending a proper rebuild.** Warmup +
   benchmark each run 11 decode steps → `22.93 s / (2·11·43) ≈ 24.2 ms/apply`, not 44 (I divided by 12·43). The
   "6–10× above the weight-streaming floor" rested on that number **and** the wrong ISA, and a BW gap doesn't
   uniquely implicate dispatch anyway (conversion throughput, vector compute, memory concurrency, repeated reads
   all contribute). *Adopted:* rebuild the floor from recorded call counts, per-layer active-expert counts,
   packed-weights-plus-scales traffic, config-matched BW, and the compute ceiling of the **executed (vector)**
   path.

5. **§5 No 12 % hard ceiling.** Correct on both counts: the attribution nests `mhc.sinkhorn`/`mhc.combine` inside
   `mhc.hc_pre` (double count; removing it nudges warm residual 12.28 → 12.92 %), and fusion **can** cut
   in-operator costs (scratch alloc, conversion, intermediate writes, internal parallel-region transitions), so
   "MoE is one kernel ⇒ irreducible" is too strong. *Reframed:* defer the monolith because **MoE dominates
   measured layer time, so targeted MoE work has higher initial priority** — not because of a hard ceiling.

## What I'm NOT changing

The direction stands and the review endorses it: **support the MoE pivot, defer the broad layer rewrite.** The
relative warm-pool decode win (~2.2×) is real and worker-step-confirmed; it is a strong **scoped** finding, to be
promoted to a default only after the phase-separated prefill/decode/mixed measurement and parity checks.

## Adopted next experiment (the review's "next decisive experiment")

One combined campaign, scoped to one EMR socket, TP=1, B ∈ {1,2,4,8,16,32,64}, representative context lengths:
1. **Worker-step timing** with correct output accounting (not prefill subtraction); retain raw samples, effective
   worker thread counts/affinity/memory placement, source/build identity; order-balanced process replication.
2. **Warm thread-count screening including counts > 8** (8 is a *cold* optimum; not established warm) — separately
   for prefill, sustained decode, and mixed; park vs finite-spin vs infinite-spin.
3. **Actual vector/BRGEMM branch counters + per-expert occupancy distribution** across layers/steps (not the
   first observed sample; B=32 already shows rows/expert 1/2.1/10, not uniformly 1–6).
4. **Matched-input tests of the existing vector and BRGEMM paths** on the observed expert groups → occupancy-based
   dispatch (vector microkernel for one/few rows; BRGEMM/AMX only where tile+conversion setup amortizes; per-group
   choice, never a diluted global average; never merge unrelated experts' rows into one GEMM).
5. **Re-attribute after each change** (Amdahl: 2× MoE → 1.56× overall, 3× → 1.92×, 4× → 2.17×). As MoE improves or
   context grows, attention / dense projections / residual glue rise in priority.

TP > 1 is explicitly out of scope for the single-socket objective.
