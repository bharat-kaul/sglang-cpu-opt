# DSv4-Flash DSA — Phase-1 Kernel Review Package

Standalone C/C++ kernel optimization, measured on the **EMR target node** (nominal BW 358 GB/s,
AMX bf16 ~125 TF, FP32 7.78 TF; `OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores`,
`numactl -N 0 -m 0`). Correctness held throughout by the F4 acceptance gate (`tie_eps=0` selection;
continuous screening). Every number below is from a labeled SLURM job this cycle.

Phase-1 requirement: **every op has an optimized C/C++ kernel**, tuned across the M-sweep (and the
KV-cache knob) toward roofline, with a **hard no-regression floor vs torch at every M**. This is the
review checkpoint before any Phase-2 wall-time integration.

> **Review response (reviewer commit `4d8edbf`).** Independent review raised 8 findings (P1-F1..F8); this
> doc is corrected accordingly and the prior overclaims are fixed inline below. Disposition:
> - **Fixed in code/harness (commit `2d4d223`):** P1-F1 indexer bf16 stage conformance (oracle + candidate;
>   signed weights; selection `non_tie=0` across M=1/8/16/32/64 — re-measured job 384526); P1-F2 dispatcher
>   guards + N==0; P1-F3 F4 structural checks (tuple count/shape/dtype, non-finite-reference hard-fail,
>   empty-manifest fail); P1-F4 sparse threshold relabelled PROPOSED (not a ratified noise floor) +
>   ALL-records budget + bf16-matched operands in Part B; P1-F7 `pipefail` + io provenance; P1-F8
>   non-finite-scale rejection.
> - **Claims corrected here:** no-regression is **comparator- and thread-dependent** — the authoritative
>   measure is the replicated 64-thread, order-varied, same-contract rig (`bench_replicated.py`, job 384531):
>   every kernel is **≥1.0× at every M except indexer M8 ≈ 0.98×** (statistical parity, spread 0.98–1.03).
>   "bit-identical" only applies to the bf16-KV experiment (`torch.equal`); cosine≈1.0 is not exactness;
>   FP32 utilization varies ~13–74%; combine M64 is cache-resident; flash bf16-AMX is not the "only" path.
> - **Accepted + PENDING (gates Phase-1 sign-off):** replicated medians with full run-identity binding
>   (P1-F6, partially started); coverage expansion — K128/640/160 unions, independent batches, causal
>   sentinels, TP-local heads, non-contiguous layouts (P1-F5); a source-faithful 64-block sparse replica +
>   ratified downstream error budgets (P1-F4). Tracked in the Exit Criteria.

---

## 1. Implementations (point of review)

| # | Kernel | Implementation | Op |
|---|--------|----------------|-----|
| 1 | indexer logits | [indexer_logits.cpp](../../kernels/dsa_pilot/indexer_logits.cpp) | relu-weighted-sum scoring (brgemm + L1 fused epilogue) |
| 2 | sparse attend | [sparse_attend.cpp](../../kernels/dsa_pilot/sparse_attend.cpp) | MLA MQA+sink top-k attention (best-of dispatch) |
| 3 | compressor | [compressor.cpp](../../kernels/dsa_pilot/compressor.cpp) | softmax-pool (streaming online-softmax) |
| 4 | indexer top-k | [indexer_topk.cpp](../../kernels/dsa_pilot/indexer_topk.cpp) | top-512 selection (serial / row-parallel / chunked) |
| 5 | MHC sinkhorn | [sinkhorn.cpp](../../kernels/dsa_pilot/sinkhorn.cpp) | hc-split sinkhorn (20-iter per-row, L1-resident) |
| 6 | MHC combine | [combine.cpp](../../kernels/dsa_pilot/combine.cpp) | hc weighted-sum (tiled accumulate-once) |

Supporting records: per-kernel plan + measured results [kernel_opt_queue.json](results/kernel_opt_queue.json);
acceptance gate [f4_acceptance.py](f4_acceptance.py); provenance [kernel_provenance.json](results/kernel_provenance.json);
roofline join (baseline source) [dsv4_roofline_vs_measured.py](dsv4_roofline_vs_measured.py).

---

## 2. Performance vs roofline + no-regression floor (optimized, measured)

**No-regression (authoritative, replicated — job 384531, `bench_replicated.py`).** 64 threads, order-varied
paired trials vs a **same-contract** fallback (the torch op each kernel replaces, at the kernel's arithmetic
contract), median of ≥5 trials × 3 process-repeats, identity-bound (node/git/dirty=0/affinity). Speedup = ref/cpp:

| kernel \ M | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| indexer_logits | 11.3× | **0.98×** | 1.49× | 3.06× | 1.75× |
| sparse_bestof | 3.26× | 2.22× | 1.74× | 1.35× | 3.56× |
| compressor_R128 | 8.45× | 2.91× | 2.19× | 1.46× | 2.64× |
| indexer_topk | 2.10× | 4.60× | 11.8× | 25.8× | 28.8× |
| sinkhorn | 93.7× | 19.4× | 19.7× | 21.0× | 21.5× |
| combine | 5.18× | 2.39× | 2.35× | 2.24× | 2.07× |

Every kernel holds ≥1.0× at every M **except indexer M8 ≈ 0.98×** (statistical parity; spread 0.98–1.03). At
**low thread counts** (8t) the parallelism-heavy kernels (indexer/sparse/compressor) regress — they are tuned
for the 64-thread target config; the floor claim is scoped to that config. Correctness: set-match / cosine,
`tie_eps=0` (F4 gate).

### #1 indexer logits — BW-bound (replicated job 384526, conformed bf16-stage kernel)
| M | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| cpp_ms (median of 3) | 0.100 | 0.204 | 0.217 | 0.272 | 0.395 |
| speedup vs torch (median) | 0.97× | 3.03× | 3.85× | 3.29× | 2.51× |
| off-roofline (was→now) | — | — | 14.6×→8.6× | 10.8×→**5.1×** | 9.1×→**3.6×** |

- Levers: parallel VNNI pack + **fused per-tile bf16 conversion** (no full bf16 kv copy); L1 fused relu·weight·sum epilogue (scores never hit DRAM). Off-roofline ~halved at large M.
- **Stage conformance (P1-F1):** the epilogue now rounds to the published **bf16** stage boundaries (bf16 einsum out, bf16 relu, bf16 signed weights, bf16 reduce → bf16 logits), matching `Indexer.forward`. Output is **fp32 storage holding bf16-rounded values** (the topk interface needs fp32), not a bf16 tensor. Selection `non_tie=0` across M=1/8/16/32/64 under **signed** weights **comes from the F4 gate** (not job 384526, which is timing-only).
- **No-regression (corrected, R2-F3 — the comparator contract flips the result):** against a **same-contract bf16 fallback**, cpp beats torch at M≤32 (M1 **9.44×**, M32 1.47×) but is **0.92× at M64**; against the fp32-nonconformed path it is **0.97× at M1**. **There is no universal ≥1.0× floor** — the small end-regressions are real and their acceptance is an explicit requirement decision (PENDING), not a relabelled "parity". Set-match vs the same-contract fallback = 1.0 at every M. Replicated paired trials with order variation are PENDING.
- The M1 path **materializes a full `[S,H]` score buffer** (≈256 KiB at S1024/H64) — the "L1-only, scores never hit DRAM" description holds for the *tiled* path, not m1.
- **BW lever shipped:** bf16-KV path — `biteq=True` all M vs fp32-kv (the GEMM already rounds kv→bf16), halves the dominant read → **+1.03–1.47×** (job 384482, `torch.equal` on sampled inputs). Kernel accepts fp32 *or* bf16 kv.
- Correctness: selection `non_tie=0` (F4 gate, signed weights).

### #2 sparse attend — fp32-GEMM-bound, near roofline (job 384476; oracle 384502/384505)
| M | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| `bestof` speedup vs torch | 3.62× | 9.08× | 1.59× | 1.34× | 1.07× |

- Shipped: **best-of dispatch** — N==1 scalar flash, N≥2 **fp32-bmm donor + in-place fused softmax**. Beats torch at every M.
- Roofline (corrected, P1-F8): sparse is dominated by its two **fp32** GEMMs; whole-op FP32 utilization **varies ~13/47/56/66/74%** across M=1/8/16/32/64 (not a uniform 64%). The M8 9.08× figure also reflects **comparator instability** (torch was 1.336 ms in job 384476 vs 0.320 ms in 384474) — the fusion share is not cleanly isolated.
- **Conformance (corrected, P1-F4):** the GPU oracle (H200 job 384502) establishes the real TileLang `sparse_attn` is **bf16** (bf16 operands + bf16 unnormalized-exp cast + bf16 out, fp32 accumulate). BUT the F4 bf16 replica is **globally** normalized, NOT the published 64-block online softmax, and the 4e-3 threshold is **approximation error on bf16-matched operands, PROPOSED not a ratified noise floor** — sparse continuous is SCREENING only (PARTIAL). A source-faithful blockwise replica + downstream budget are PENDING. Naive `sparse_attend_amx` is DROPPED (drifts further + slower).
- Residual headroom: a flash bf16-AMX with fp32 accumulate is **one** path to the 124 TF ceiling; a tiled bf16-input/fp32-output library GEMM with the published recurrence is an **untested alternative** (not foreclosed).

### #3 compressor — BW/grain-bound, 3-shape coverage (job 384474)
| M | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| R128/D512 | 8.64× | 28.51× | 11.77× | 4.91× | 2.58× |
| R8/D512 | 2.57× | 4.82× | 5.09× | 4.50× | 3.22× |
| R8/D128 | 1.92× | 2.22× | 2.62× | 4.61× | 5.50× |

- Lever: **N·D-tile parallelism** (was N-only). Fixed the M=1 regression (R128/D512 0.56×→8.64×). cos=1.0 all shapes; F4 max_err ~5e-7 incl masked.
- BW-precision lever **rejected by measurement**: this op is grain-bound (~7% of BW-ideal), not read-BW-bound; bf16 input is 0.26–0.73× (slower) — lever N/A.

### #4 indexer top-k — selection-bound (job 384480)
| M | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| speedup vs torch | 2.29× | 4.24× | 11.25× | 22.19× | 28.13× |

- M=1 regression **fixed** (0.41×→2.29×): serial `nth_element` at N==1 (no thread-team spinup); chunked 2-pass guarded to `2k ≤ S/chunks` (it was pathological at k=512/S=1024), else row-parallel. set_match=1.0 all M.

### #5 MHC sinkhorn — dispatch-bound tiny (job 384480)
| M | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| speedup vs torch | 98.3× | 19.6× | 20.6× | 21.5× | 23.0× |

- Fused 20-iter per-row, hc×hc L1-resident. cos=1.0; no small-M dispatch regression. (FLOP roofline is the wrong ceiling for a dispatch-bound tiny op — the 19–98× over torch is the real result.)

### #6 MHC combine — BW/latency-bound (job 384480)
| M | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| speedup vs torch | 5.52× | 2.25× | 2.31× | 2.15× | 1.98× |

- Tiled accumulate-once (no broadcast temp). cos=1.0. Corrected (P1-F8): M64 useful traffic / 14 us ≈ **375 GB/s > nominal 358 GB/s** → warm buffers are partly **cache-resident**; this is NOT proof of DRAM saturation.

---

## 3. Fusion & precision gates (mandatory playbook gates — audit result)

- **Fusion:** intra-op epilogue fusion is applied on all 6 (compressor streaming softmax; sparse in-place fused softmax; sinkhorn/combine per-row; indexer tiled keeps scores L1-resident — **the m1 path does materialize a full `[S,H]` score buffer**). The sparse scores between the two donor bmms are a residual round-trip; removing it would forfeit the near-roofline MKL bmm, so we keep the donor — but **irreducibility is a design trade-off, not measured/proven**, and a tiled bf16-input/fp32-output library GEMM is an untested alternative. Cross-op fusion deferred to Phase-2.
- **Precision/BW:** compute is bf16 where GEMM-bound; low-precision **storage/IO** applied where it helps — **shipped** on indexer bf16-KV (free, bit-exact), **rejected by measurement** on compressor (not read-BW-bound), and on sparse the **reference dtype = bf16** (GPU oracle) with the fp32-accumulate donor shipped — but sparse acceptance is **SCREENING/PROPOSED, not ratified/resolved** (source-faithful blockwise replica + downstream budget PENDING). Recorded in [kernel_opt_queue.json](results/kernel_opt_queue.json) `fusion_precision_gate`.

---

## 4. Provenance (job IDs, all this cycle)

| Job | Node | Measures |
|---|---|---|
| 384474 | EMR | indexer sweep + compressor 3-shape + sparse scalar/amx/torch |
| 384476 | EMR | sparse `bestof` vs torch (beats at every M) |
| 384480 | EMR | topk / sinkhorn / combine no-regression + correctness |
| 384482 | EMR | indexer confirm + **bf16-KV** BW lever (biteq + speedup) |
| 384502 | H200 | **sparse GPU oracle** — real TileLang `sparse_attn` vs fp32/bf16 replicas → reference = bf16 |
| 384505 | EMR | sparse CPU paths vs saved real-kernel output → fp32-bmm conformant, amx dominated |

---

## 5. Review status

- **All 6 kernels**: optimized C/C++, correctness clean (F4 PARTIAL, no hard-gate failure); no-regression holds at M≥8, **M=1 indexer is parity (0.94–0.96×), not a proven floor** (P1-F6). ⚠️ requalified
- **Perf vs roofline**: recorded across the M-sweep (single-generation; replicated medians PENDING). ⚠️
- **Sparse dtype**: reference = bf16 (GPU oracle); acceptance is SCREENING/PROPOSED, not ratified (P1-F4). ⚠️
- **Phase-1 exit = PAUSE FOR REVIEW**: not advancing to Phase-2 until the Exit Criteria below are met.

## Exit criteria (from review `4d8edbf`, tracked)

1. **Done:** source-stage indexer oracle + candidate conformance (signed weights, `non_tie=0`); common guards; F4 rejects malformed structures / non-finite comparison / empty inventory / hard selection failures.
2. **Pending:** extend acceptance to every required M + captured shapes/layouts (K128/640/160 unions, independent batches, causal sentinels, TP-local heads, non-contiguous); conform the sparse reference on identical operands/output boundaries with a source-faithful 64-block replica; keep approximate budgets explicitly pending until independently justified + ratified.
3. **Pending:** bind each run to source/build/library/config/input identity; reproduce all M with ≥3 independent trials (medians + variability, randomized/alternated pairing, cold vs steady-state); investigate M1 indexer + M8 sparse comparator behavior before any universal-floor claim; final perf on target HW (GNR), not portable infra.
4. **Pending:** preserve the historical baseline; publish optimized measurements as a separate versioned record; price remaining opportunities by actual caller workload, not microbench speedup; proceed to captured-integration / full-model gates.

Open items for reviewer judgment: (a) ratify the F4 tolerance posture + downstream budgets; (b) whether the flash bf16-AMX / tiled-bf16-GEMM sparse headroom is worth authoring now; (c) proceed to Phase-2 only after criteria 2–4 close.
