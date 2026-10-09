# DSv4 Roofline — Author Response to Re-review (F1–F5)

Responds to [dsv4_flash_roofline_gate_rereview.md](dsv4_flash_roofline_gate_rereview.md).
Reference centralized in one constant `REF_REVISION = 60d8d70770c6776ff598c94bb586a859a38244f1`
(typo fixed, used by generator, join, and tracker). Self-test now asserts reference conformance
**and** reporting contracts, and gates report emission.

Run: `python plugin/validate/dsv4_roofline_p2.py --selftest` (all PASS, fail-closed).

Verified corrections (R1/R3 and the partial R2/R4/R6/R7 items) are preserved.

## Disposition

### F1 — Single-point measurements populated every batch column — FIXED
- Introduced explicit row KINDS: `gemm` (ideal, all M), `measured` (ONE observed coordinate), and
  `unmodeled` (declared not-modeled).
- `measured` rows (`topk`, `sinkhorn`, `combine`) now render ONLY at their observed `M=32`; every
  other M shows `N/A`. Each carries a benchmark **source link** (`bench_topk.py`,
  `bench_sinkhorn.py`, `bench_combine.py`) and node/dtype/ctx provenance — not a model SHA.
- `q_norm`, `RoPE`, `kv_norm`, `hash-route`, `embed` had no identifiable observation → reclassified
  `unmodeled`; they render `n/m` and are surfaced in the tracker instead of being populated as
  measured constants.
- Self-test asserts: a measured row renders a number only at its M and `N/A` elsewhere; carries a
  source; an unmodeled row renders a number at NO M.

### F2 — Pooling omitted the score stream / misstated state precision — FIXED
- `pool` now reads BOTH FP32 state streams (KV + score, `win·D` each) + the shared FP32 positional
  APE (`win·D`, once per call) and writes an FP32 output (`D`/request). State dtype = FP32 (reference
  compressor state). This matches the `bench_compressor.py` contract (`kv, score, ape`) and gives
  17,104,896 bytes at M=32, R=128, D=512.
- Compressor projection GEMMs now declare FP32 state output (`act="fp32"`); checkpoint-BF16 weights
  no longer imply BF16 outputs/state.
- The main compressor's remaining per-token state-write cadence + dtype-conversion/normalization
  postprocessing is listed as EXPLICITLY-UNMODELED in the tracker.
- Self-test asserts the pool byte formula (2× FP32 `win·D` + out + APE) and FP32 state dtype.

### F3 — FP32 labels used the BF16 compute model — FIXED
- Added `COMPUTE_PEAK`: `bf16/fp8/fp4` → AMX 124.5 TF; `fp32` → a justified nominal AVX-512 FP32
  ceiling (7.78 TF, derived in emr.json: 64c × 2 FMA × 16 lanes × 2 × 1.9 GHz). `prec` now selects
  the compute resource and the ridge; FP32 MHC rows and crossings use the FP32 ceiling.
- `hc_fn` operand dtype is FP32 (`hc_pre` casts the flattened input); its activation bytes are FP32.
- Self-test asserts FP32 and BF16 use different peaks and that a compute-bound FP32 GEMM is strictly
  slower than the BF16 equivalent.

### F4 — Join added incorrect operands / omitted required ones — FIXED
Each row now derives from its benchmark's input/output contract; totals asserted fail-closed:
- **Indexer logits**: removed the `wq_b` matrix term (wrong boundary) and the shared `64·4`
  head-weights; query is a BF16 projected activation, head-weights are per-request `M·64·4` →
  **9,052,160 B**.
- **Compressor**: restored the shared APE read and both FP32 streams → **17,104,896 B**.
- **Sinkhorn**: added `scale`/`base` reads and `pre`/`post`/`comb` writes → **6,252 B**.
- **Combine**: per-request `pre[M,4]` weights (not a shared `[4,4]`) and reduction adds included →
  **2,621,952 B**, FLOPs `M·H·(4+3)` = **917,504**.
- Removed causal conclusions ("overhead dominance", "confirms donor routing"); the join now reports
  observations + distance only, deferring causation and donor validation to implementation review.

### F5 — Tracker failures bypassed the publication gate — FIXED
- `load_tracker()` is fail-closed: raises on missing/malformed tracker, enforces `REF_REVISION` and
  valid dispositions; report emission aborts if it fails.
- Self-test loads+validates the tracker and asserts it rejects an invalid disposition.
- Centralized `REF_REVISION` (fixed the 41→40-char typo) across generator, join, and tracker.
- Marked the stale `dsv4_roofline_vs_measured.txt` HISTORICAL; the canonical join artifact is
  `dsv4_roofline_vs_measured_emr.txt`.
- The reusable-skill occupancy-formula alignment remains a tracked follow-up for the playbook phase
  (not a roofline blocker), per the agreed sequencing.
