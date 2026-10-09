# DSv4 Roofline — Author Response to Gate Review (R1–R7)

Responds to [dsv4_flash_roofline_gate_review.md](dsv4_flash_roofline_gate_review.md).
Reference pinned to HF revision `60d8d70770c6776ff598c94bb586a859a38244f1` (config.json +
inference/model.py + checkpoint-header dtypes). Every fix is now asserted by a
REFERENCE-CONFORMANCE self-test that gates report emission.

Run: `python plugin/validate/dsv4_roofline_p2.py --selftest` (13 checks, all PASS).

## Disposition

### R1 — Attention graph & main-layer counts — FIXED
- Layer counts now derived from the pinned `compress_ratios[:43]` = `[0,0]+[4,128]*20+[4,0]` →
  `Counter{0:2, 4:21, 128:20}` (N_SWA=2, N_IDX=21, N_128=20). Self-test asserts this exactly.
- Attention is now ONE sparse op per layer over window(128)+compressed sources. Per-layer
  positions: ratio-0→128, ratio-4→128+min(512,1024)=640, ratio-128→128+32=160. Summed over the
  43 main layers `SUM_POS=16896` (asserted). Removed the prior full-S=4096 pass + separate
  top-512 pass.
- `attn.flop(M=1)=4·64·16896·512 = 2,214,592,512` (asserted). `attn.bytes(M=32)=734,003,200`
  → 2.048 ms at nominal BW (matches the review counterexample).

### R2 — Compressor subgraph — FIXED
- Every-token projections now itemized as GEMM rows: main r4 4096→2048 ×21, main r128 4096→1024
  ×20, indexer r4 4096→512 ×21 (BF16 storage). Capacity main=0.520 GB, indexer=0.088 GB (both
  asserted).
- Softmax-pool now amortized by boundary cadence: main calls/step = 21/4 + 20/128 = 5.40625,
  indexer = 21/4. Ratio-4 overlap pools 2·ratio=8 positions (D=512 main / D=128 indexer);
  ratio-128 pools 128.

### R3 — Shared experts are FP8 — FIXED
- Shared expert dtype switched FP4→FP8 (checkpoint F8_E4M3 + F8_E8M0 scale). Capacity 1.082 GB
  (asserted). M=1 time now 3.02 ms (was 1.61). Routed experts remain MXFP4.

### R4 — MHC hc_fn dim & dtype — FIXED
- `hc_fn` K = hc_mult·H = 16384 (not NH·HD=32768), N=(2+hc)·hc=24; dtype FP32 (checkpoint
  hc_attn_fn F32). `hc_fn.flop(M=1)=67,633,152` (asserted). hc_post/hc_head FP32; their
  norm/affine/sigmoid-gating/reduction vector work is now EXPLICITLY-UNMODELED and surfaced.

### R5 — `measured()` provenance & join byte accounting — FIXED
- `measured()` rows now carry single-point provenance (node/revision/dtype/batch/ctx) and are
  labeled "single-point, not a batch sweep" in the report and tracker.
- Join byte accounting corrected: indexer adds head-weight read (IDX_NH·4) and query-weight read;
  top-k adds selection-output traffic (M·512 int32); compressor output is per-request (M·D) not
  per-window; sinkhorn/combine add output writes; combine adds comb-weight read.

### R6 — emr.json stale fields — FIXED
- `host` no longer claims pcl-spr10; ISA/topology scan node relabeled `isa_scanned_on`.
- `measured_how` now describes pcl-sprh02 DDR5-5600 (277 GB/s, 47 TF); removed the stale
  214.4/40.3 pcl-spr10 figures and the "floor = recoverable headroom" framing.
- AMX peak corrected to 124.5184 TF; 1.9 GHz relabeled `amx_bf16_ghz_ref` (nominal base clock,
  not a measured all-core AMX frequency).

### R7 — Self-test was self-consistency, not conformance — FIXED (root cause)
- Self-test now asserts REFERENCE CONFORMANCE against the pinned config/graph/checkpoint dtypes
  (layer counts, SUM_POS, attention & hc_fn FLOPs, shared=FP8, routed=FP4, hc_fn=FP32, compressor
  capacity). Report emission is GATED on these passing.
- The report now consumes `results/roofline_open_items.json` and prints the EXPLICITLY-UNMODELED
  quantities so they are visible at review.
- The `.agents/.../model-roofline-analysis` SKILL occupancy-formula alignment is deferred to the
  playbook-encoding phase per the agreed sequencing (roofline-pass first).
