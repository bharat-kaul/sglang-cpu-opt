---
name: cpu-serving-integration
description: "Use in the new-kernel leg AFTER the novel compute kernels are authored + parity-validated (kernel-authoring) and BEFORE the enablement-certificate. Captures how to wire authored CPU kernels into the model's SERVING RUNTIME so it actually runs end-to-end: the attention BACKEND forward, the paged KV-cache (exact packed byte layout + pack/unpack), the forward-batch metadata, and absorbed-attention specifics. Distinguishes the (smaller) novel-compute authoring from the (larger) serving-runtime port, and ends at the workflow's north-star PROOF: model runs + accuracy vs a trusted reference + roofline-target-vs-measured published. Learnings distilled from the DeepSeek-V4 DSA/MLA CPU bring-up."
---

# CPU Serving Integration (make the model actually run)

Authoring the novel kernels (kernel-authoring) proves the math; it does NOT make the
model serve. A modern arch's attention runs through a **backend forward** that calls a
serving kernel (e.g. `flash_mla_with_kvcache` / `flash_mla_sparse_fwd`) over a **paged
KV cache**, driven by per-query indices + metadata. Getting a running model means
porting THAT layer to CPU — a larger, distinct effort from the compute kernels it calls.

## Pipeline position
`kernel-authoring (novel compute, parity-validated) → cpu-serving-integration →
accuracy-oracle → roofline-vs-measured → enablement-certificate`. This is the "make it
serve" phase; its gate is the workflow's proof deliverable.

## The serving-runtime surface (what to port)
1. **Attention backend forward.** The arch's backend (`<arch>_backend.py`) `forward`
   calls the CUDA serving kernel over the paged cache. Reimplement it in torch/CPU,
   REUSING the already-authored compute kernels. Know the attention form:
   - **MLA-absorbed** (DeepSeek): value == key (one latent), so
     `out = softmax(q·kᵀ·scale ⊕ per_head_sink)·k` — the per-head attention *sink* is a
     virtual key column dropped after softmax. `head_dim_v == head_dim`.
   - GQA/MHA: standard, k and v distinct.
2. **Paged KV cache pack/unpack.** The cache is a **packed byte buffer**, not plain
   tensors — you MUST reproduce the exact byte layout to write AND read it. DSV4 example:
   584 B/token = 448 fp8 nope + 64 bf16 rope (128 B) + 7 ue8m0 tile-scales + 1 pad, with
   nope/rope in the token region and scales in a separate per-page region (see
   `index_buf_accessor._set_k_and_s_torch` — it has an in-tree torch writer; write the
   matching **unpack**). Per-64-tile fp8 uses pow2 (ue8m0) scales: `val * 2^(scale-127)`.
3. **Metadata prep.** positions, compressed-attn metadata, page indices, topk lengths —
   several are Triton with torch fallbacks (route via `enablement-scope-discovery`).
4. **Gather.** Per query, gather its attended tokens via the page indices (+ compressed
   "extra" indices), unpack to bf16, attend.

## Reference-wiring-FIRST (the correctness backbone)
**Build the end-to-end serving path with fallback/reference kernels and make it RUN +
CORRECT *before* optimizing anything.** Per-kernel parity (kernel-authoring 1b) proves a
kernel's *math* but NOT its *integration*: a wrong paged byte layout, page-index, sink,
or metadata field passes unit parity and still yields garbage end-to-end. Only a running
reference serving path catches those. This ordering also surfaces the whole INFRA layer
(NUMA/TP, config adjustment, allocator, KV pool) that static analysis misses — reactively,
one break at a time, is how DSV4 revealed ~24 breaks; doing it deliberately front-loads them.

**So the phase order is:** roofline + `enablement-scope-discovery` → **reference-wiring
bring-up (this)** → author/optimize each kernel against the now-running reference →
real-weights accuracy + perf. The reference path is your oracle for every later optimization.

### The tiny-faithful config (makes it practical at any model size)
The user's fair objection — "the real model is 100s of GB, you can't iterate on it" — is
solved by a **tiny but architecturally-faithful config**: copy EVERY architecture *switch*
from the real config (attention form e.g. MLA, sparse/DSA on, hash/clustering layers on,
quant scheme e.g. fp8 block-quant, `head_dim`/rope/compress-ratios, `num_key_value_heads`)
but shrink the *dimensions* (2 layers, tiny hidden/experts) and use `--load-format dummy`.
It runs in **seconds on a login node**, exercises the **same code paths**, and hits the
**same wiring/infra breaks** (DSV4: reproduced the TP `num_key_value_heads` config bug and
the MoE/attention breaks on tiny, not on the 806 GB model). Debug the plumbing here; spend
expensive full-scale load cycles only on the final accuracy/perf run.
- **Faithful means the switches, not the sizes.** A tiny config with `num_hash_layers=0`
  or DSA bypassed is NOT faithful — it silently skips whole subsystems. Match every flag.
- **Keep optimized kernels behind a flag** so the reference path stays runnable, then
  **A/B-diff reference-vs-optimized on the same input** = an end-to-end correctness
  regression harness (stronger than unit parity).
- **What tiny-faithful + dummy weights does NOT catch:** (a) accuracy — needs REAL
  weights + `accuracy-oracle`; (b) weight-shape/quant-packing artifacts that differ from
  real (DSV4 dummy fp8 gave a spurious `packed_w1 260 vs 512` that real weights resolve —
  do NOT chase dummy-only shape bugs); (c) some scale-only bugs (memory/NUMA/TP capacity).
  Keep TWO run types: **structural/wiring** (tiny, dummy, cheap, frequent) and
  **accuracy+scale** (real weights, expensive, final).

## Procedure
1. Read the backend `forward` to get the interface (q/k/v shapes, compress_ratio, sink,
   which caches + index tensors) and the serving kernel's contract.
2. Get the AUTHORITATIVE attention math from the arch's **pure-torch test reference**
   (`test/kits/.../<arch>_attention.py`) — it is the oracle for the absorbed form + sink.
3. Port `forward` to torch: store KV (packed) → read caches → unpack → per-query gather →
   attention (reusing authored kernels) → return. Iterate on a TINY same-arch config.
4. Validate: (a) runs end-to-end; (b) `accuracy-oracle` — per-layer/logits parity + task
   vs a trusted reference (HF / GPU path) on REAL weights; (c) `model-profile-hotspots` →
   `roofline_vs_measured.py` for the perf proof (same machine config, labeled).

## Learnings (DeepSeek-V4, transferable)
- **Wiring/infra ≫ kernels in break-count.** The bring-up cleared ~24 breaks; almost all
  were plumbing (Triton→torch ports, config/TP, allocator, metadata, `input_ids`
  threading), not novel math. Front-load them with reference-wiring-first on tiny-faithful.
- **CPU multi-NUMA = TP across NUMA.** SGLang CPU sizes each rank's memory as
  `total/n_numa` and binds it to one NUMA node, so a model bigger than one NUMA node's RAM
  MUST be TP-sharded (tp = #NUMA, or a divisor of the head count). A tp=1 process OOMs at
  the single-node limit regardless of total RAM. On a node with more NUMA nodes than ranks,
  no explicit bind is needed; when ranks > NUMA nodes, set `SGLANG_CPU_OMP_THREADS_BIND`.
- **MLA breaks the generic unaligned-CPU-TP head padding.** `adjust_config_with_unaligned_cpu_tp`
  fires when `total_kv_heads % tp != 0`; MLA's single replicated latent KV head (1) always
  trips it and the GQA-oriented pad rewrites `num_key_value_heads` (1→N), violating the
  model's `num_key_value_heads == 1` invariant. Patch the pad size to 1 for MLA. General
  lesson: framework TP/config helpers assume GQA — verify they no-op for MLA/novel attention.
- **A novel-feature backend has no dense fallback** — DSA sparse attention IS the dsv4
  forward; you cannot bypass it to a dense path. Verify a fallback exists before
  planning any bypass (also in `model-profile-hotspots`).
- **Compute authoring ≪ serving port.** The novel DSA math (compressor/indexer/sparse)
  was a handful of small parity-validated kernels; the paged flash-MLA serving forward
  (unpack + absorbed attention + sparse gather + plan/state-pool) is the larger effort.
  Scope them separately in `enablement-scope-discovery`.
- **The paged cache is a packed byte layout** with a separate scale region — reproduce it
  exactly (there is usually an in-tree torch accessor for the write; mirror it for read).
- **v==k + sink** is the MLA-absorbed shortcut that makes the torch attention simple.

## Gate — the workflow's PROOF deliverable
Enablement is only "done" when the model **RUNS end-to-end on CPU**, is **accuracy-parity
vs a trusted reference** (accuracy-oracle, real weights), AND has a **published
roofline-target-vs-measured** perf artifact at a labeled machine config. Running without
accuracy, or accuracy without the perf-vs-roofline proof, is not done — the certificate
requires all three (enablement-certificate). That triple IS the thesis proof.
