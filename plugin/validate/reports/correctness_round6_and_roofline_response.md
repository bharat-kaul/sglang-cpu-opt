# Response: Correctness Round-6 (`7e142bd`) + Performance-vs-Roofline M-Sweep

Two deliverables in one response: (1) the fixes to the round-6 correctness assessment, and (2) a fresh performance-vs-roofline M-sweep for **all** shipped kernels on the current build. Commits: `72271b2` (R5-F1/F2), `aaccef6`/`b73d763` (roofline bench + results), `0dd57cd` (meta-learnings). No promotion is requested; F4 stays **PARTIAL** and `promotion_gate.py` stays **BLOCKED**.

## Part 1 — Correctness round-6 fixes (`7e142bd`)

Both findings are fixed in code with persistent intended-reason controls.

### R5-F1 (High) — replay accepted non-finite oracle evidence
The replay now rejects **invalid** evidence, not just wrong shape/scale:
- `validate_archive` rejects any **non-finite** (`NaN`/`inf`) `gpu_out` / `q` / `kv` / `sink` record tensor ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L63)) — sparse attention has no `-inf` mask domain in these tensors, so all must be finite.
- `metrics()` rejects a non-finite **reference**, a non-finite **candidate output**, and non-finite **cos/mae** instead of printing `NaN` as if valid ([test_sparse_cpu_vs_gpu.py](plugin/validate/test_sparse_cpu_vs_gpu.py#L79)).
- **Controls** (replay selftest, now 19): non-finite `gpu_out`/`q`/`kv`/`sink` each rejected; a **finite positive control**; `metrics()` rejecting a non-finite reference and a non-finite candidate. A NaN oracle output is treated as invalid evidence, not an approximation exceeding a tolerance.

### R5-F2 (Medium) — joint-coordinate check validated a different build than evaluated
The required-coordinate shape check is now **bound to the actual evaluated tuple**:
- `run_case` validates `c["_coord_verify"](inp)` on the **same** `inp` it passes to candidate and reference, on **every** seed ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L306)); `run()` only **binds** the verify and checks declaration presence (no separate preflight build) ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L551)).
- **Control** (F4 selftest): a case whose evaluated tuple (R64) differs from its declared coordinate (R128) hard-fails for the **"builder drift"** reason ([f4_acceptance.py](plugin/validate/f4_acceptance.py#L846)).

**Verification:** replay selftest OK (incl. 7 new R5-F1 controls); F4 selftest OK (incl. the R5-F2 drift control); full gate **PARTIAL**, 240 evaluations, no hard failures. The scoped closures the assessment confirmed (R4-F1/F2/F3, persistent replay selftest, manifest claim-drift) are unchanged. The budget methodology remains a requirements checklist for an **authorized** experiment, not a ratified number — accepted as such.

## Part 2 — Performance vs roofline, full M-sweep (all shipped kernels)

Fresh measurement on the current build (includes the bit-exact **I1** indexer and **C1** compressor wins), EMR `8592+` DDR5-5600, `OMP_NUM_THREADS=64 OMP_PROC_BIND=close`, `numactl -N0 -m0`, 3-process-replicate medians. Floor = `max(bytes/BW, FLOPs/peak)` from each kernel's byte/FLOP contract against the **datasheet** peak (BW 358.4 GB/s, AMX bf16 124.5 TF, FP32 AVX-512 7.78 TF). Full data: [roofline_sweep.json](plugin/validate/results/roofline_sweep.json).

**achieved% = floor / measured** (higher = closer to the ceiling):

| Kernel | regime | M1 | M8 | M16 | M32 | M64 |
|---|---|---:|---:|---:|---:|---:|
| indexer_logits / tiled | BW | 1.6% | 6.2% | 12.2% | 19.1% | **25.3%** |
| indexer_topk | selection | 0.6% | 1.5% | 3.1% | 6.0% | 10.6% |
| compressor / r128d512 | BW | 11.1% | 24.2% | 26.5% | 25.2% | 26.2% |
| compressor / r8d512 | BW | 0.9% | 4.5% | 7.5% | 12.7% | 17.6% |
| compressor / r8d128 | BW | 0.3% | 1.5% | 2.7% | 5.1% | 8.9% |
| sparse / bestof | compute | 12.9% | 46.1% | 57.0% | 66.3% | **71.8%** |
| sinkhorn | dispatch | ~0% | ~0% | 0.1% | 0.1% | 0.2% |
| combine | BW | 7.2% | 13.2% | 27.0% | 53.8% | **97.5%** |

Measured median latency (µs) is in the JSON (e.g. indexer_logits M64 = 395.6µs vs a 100.25µs floor; sparse/bestof M64 = 768.5µs vs 551.9µs).

**Honest reading (not a target claim):**
- **sparse / bestof** reaches **71.8%** of the FP32 ceiling at M64 — the fp32-bmm MKL donor is genuinely compute-bound and near its roof; this is why the S1 bf16-blockwise experiment (standalone review) could not beat it.
- **combine** reaches **97.5%** at M64 — but this is **cache-resident** throughput, not DRAM saturation; a datasheet-BW floor is not the right ceiling here (measurement-physics caveat).
- **indexer_logits** climbs 1.6%→25.3%: small-M is convert/pack/launch-overhead bound (BW-bound but far from the byte floor), large-M amortizes — consistent with the standalone attribution.
- **indexer_topk** and **sinkhorn** are **selection/dispatch-bound**; a byte-only or dense-FLOP floor is the **wrong** ceiling for them, so the low % is a diagnostic, not a recoverable gap.
- **The datasheet peak is a ceiling, not achievable.** The measured-reference on this cluster is ~277 GB/s BW and ~47 TF AMX (per [emr.json](plugin/validate/platforms/emr.json)); achieved-vs-**achievable** is materially higher than achieved-vs-datasheet. The residual gap is surfaced, not assumed recoverable.

These are standalone microbench numbers at the kernels' call contracts; they are **not** an end-to-end or current-target certificate, and no dispatch/E2E tuning is implied.

## Part 3 — Meta-learnings encoded (model-agnostic)

The recurring review classes across this arc are encoded as a third cluster in the model-agnostic [adversarial-self-audit skill](.agents/skills/shared/adversarial-self-audit/SKILL.md) (`0dd57cd`), so future gates self-audit before review: (11) validate evidence on **both** sides and reject **invalid** (not just imprecise) inputs with a legal-vs-invalid policy + positive control; (12) bind the shape check to the **evaluated** artifact, not a preliminary sample; (13) bind evidence to the expected **content hash + canonical run-id** and validate coordinate **consistency** + the un-removable **joint coordinate**, not mere presence or a producer flag; (14) **closure ≠ promotion** — keep arithmetic acceptance separate from speed, prove bit-exact by diff not cosine, **attribute before changing** (neutral ⇒ revert), classify the regime first, and record a measured loss/neutral as a valid disposition.

## Status
F4 **PARTIAL** (shipped C1 + bit-exact I1; selftest OK), promotion **BLOCKED**, thresholds UNRATIFIED. No native kernel source changed in Part 1. The performance numbers are standalone microbench vs datasheet-peak roofline with the achievable-gap caveat; no promotion, budget ratification, or phase-dependency change is requested here.
