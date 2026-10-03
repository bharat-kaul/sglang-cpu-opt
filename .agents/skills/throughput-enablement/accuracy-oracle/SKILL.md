---
name: accuracy-oracle
description: "Use after cpu-model-wiring to prove the CPU-enabled model is numerically correct. ORDERING (critical): on UNOPTIMIZED code do NOT gate correctness on full-model generation or a task harness (too slow — watchdog); the feasible correctness gate is the per-layer fingerprint parity vs the GPU/HF baseline on ONE short prefill. CONFIG LADDER (do not mistake the proxy for the gate): a TINY arch-faithful config is the BRING-UP proxy (op-TYPE coverage) ONLY — the AUTHORITATIVE correctness gate is FULL-MODEL DUMMY parity (ALL layers at REAL shapes, deterministic-dummy weights so NO real-weight load, MULTI-GPU reference when the full model exceeds one GPU), established BEFORE optimizing and reused to regression-guard EACH optimization; REAL weights only for the dtype-bridge + task finale. Optimize AFTER correctness; regression-guard with the SAME fast parity; full generation + task accuracy come LAST once fast. LAYERED checks so a failure localizes: (0) a cheap real-prompt COHERENCE smoke test first ('it ran' on random/dummy inputs is NOT 'it's correct'; catches gross zeroed/no-op/routing/layout bugs and silently-stubbed forward paths), (0.5) a per-kernel LOW-BIT PARITY gate for every dtype bridge (fp4/mxfp4/fp8/int4 weights on a bf16/fp8 kernel) — kernel output vs an independent torch dequant oracle from the same packed bytes, because 'a bf16/fp8 kernel exists' is coverage, not correctness, (1) per-layer / logits parity vs a trusted reference (HF or the GPU SGLang path) under tolerance, and (2) end-to-end task accuracy via the in-repo harnesses (gsm8k, mmlu, hellaswag) within N points. Includes a correct-by-construction ISOLATION technique to bisect MoE vs shared-path bugs. Blocks enablement on any regression; a per-layer diff pinpoints the offending op for the wiring step to fix."
---

# Accuracy Oracle

Correctness is the non-negotiable gate. A plausible-but-wrong model is worse than
an unenabled one, so prove parity numerically AND on task, and make failures
localizable.

## ⛔ CORRECTNESS ORDERING (read FIRST — do not gate correctness on a full unoptimized run)
On freshly-wired, UNOPTIMIZED code the forward is SLOW (unfused torch refs, no AMX/kernel path,
no batching) — a 512+-token prefill can take >14 min and full generation / a task harness (gsm8k)
will blow the watchdog. So **do NOT use full-model generation or task accuracy as the first
correctness check.** The feasible, decisive correctness gate on unoptimized code is the **per-layer
fingerprint parity against the GPU/HF baseline (Layer 1), which needs only ONE short-prompt
PREFILL** (e.g. 8 tokens × all layers — minutes, not hours; the pathological cost is long-context
prefill + slow decode, neither of which a tiny-prefill capture incurs). Sequence it:
1. **Bring-up** (`cpu-model-wiring`) → the model completes ONE forward (make-it-work).
2. **CORRECTNESS = per-layer fingerprint diff (Layer 1), NOW, on unoptimized code.** Capture CPU
   per-`(pass,op,layer,rank)` tensors on a single short prefill and cosine+magnitude-diff them
   against the pre-frozen GPU fingerprint (stood up in parallel at day-0 — see the GPU-oracle
   section). First divergent layer = the bug. This proves the KERNELS/wiring are right.
3. **THEN optimize** kernels + framework overhead (the perf ladder). Nothing slow runs before
   correctness is established this cheap way.
4. **Regression guard THROUGH optimization = the SAME fast parity**, not a slow full run — and it
   RUNS ON THE DUMMY-WEIGHT PERF PROXY: an optimized kernel is A/B'd against the reference impl it
   replaces on the SAME input (full-tensor cos+mag), which is VALID on dummy weights because it tests
   the KERNEL TRANSFORM, not accuracy. Plus a periodic per-layer re-diff vs the frozen reference
   (T1/T2 below) + a short-prompt cached-golden next-token parity — seconds each. Net flow: parity is
   established FIRST (DETERMINISTIC-DUMMY cross-engine diff, short prefill — see the WEIGHT-AXIS note),
   then CARRIED through the entire dummy-proxy optimization campaign, then RE-CONFIRMED per-layer on the
   full OPTIMIZED model, and only at the END promoted to a REAL-WEIGHT parity + TASK-accuracy run.

### ⛔ WEIGHT AXIS (dummy vs real) — orthogonal to the optimization axis; get this right or you pay a needless full-weight load
The correctness ladder has TWO axes. The optimization axis (unoptimized → optimized) is above. The
WEIGHT axis decides which checkpoint each parity stage runs on, and the DEFAULT for every EARLY and
OPTIMIZATION-phase correctness check is **DETERMINISTIC DUMMY weights**, NOT the real checkpoint:
- **Stages 1–4 (bring-up, first parity, optimization regression-guard, re-confirm): DETERMINISTIC-DUMMY.**
  Run BOTH the CPU build and the GPU/HF reference on the SAME tiny-but-arch-faithful config with the
  SAME deterministic dummy init, and diff per-layer. This is cheap (no 100s-of-GB load, fits one GPU,
  seconds-to-minutes), isolates the WIRING + kernel math, and does NOT drag in the dtype bridge. It is
  the right correctness tool for the entire make-it-work + make-it-fast campaign.
- **Stage 5 (final): REAL weights.** The real checkpoint is required ONLY for the things dummy cannot
  prove: the fp4/fp8/mxfp4 **dtype bridge** (dummy drops quant → scales=1.0), the FULL layer depth,
  and end-to-end TASK accuracy. Capture/freeze the real-weight GPU fingerprint day-0 (it is independent
  of the CPU port) and ARCHIVE it for this final gate — but do NOT make it the FIRST parity.
- **DETERMINISM IS THE ENABLER (the trap that makes naive dummy 'useless').** `load_format=dummy`
  initializes weights with a generator seeded on the PARAM'S DEVICE (`torch.Generator(device=...)`),
  so CPU and CUDA RNG streams DIFFER and the stream can shift across torch versions (CPU engine vs GPU
  container) — naive dummy weights are NOT bit-identical across the two engines, which is why an
  earlier version of this skill wrongly concluded 'dummy is useless for parity'. The FIX, not the
  avoidance: patch `initialize_dummy_weights` on BOTH sides to fill each param from **numpy**
  (MT19937, platform/version/device-independent) with a per-param seed derived from the param NAME
  (both sides tp=1, full params → identical). Then dummy weights ARE bit-identical and dummy parity is
  valid. Reusable asset: `plugin/_dummy_determinism.py` (gated `DETERMINISTIC_DUMMY=1`), wired into
  both the CPU plugin and the GPU reference hook. [PROVEN on GLM-5.3 Flash: GPU-dummy vs CPU-dummy
  8-layer prefill diff = cos 1.000001 / rel_maxerr 0.0 (BIT-EXACT) on every per-layer hidden +
  MHC residual, logits cos 0.999996 — deterministic-dummy cross-engine parity works and is cheap.]

### ⛔ CONFIG AXIS (tiny-faithful → full-model dummy → real) — a THIRD axis; the tiny config is a BRING-UP proxy, NOT the final correctness gate
Shrinking the config is the parity analog of the truncated-model perf proxy, and it has the SAME ladder:
- **TINY ARCH-FAITHFUL config (bring-up proxy).** A depth- AND width-reduced config that keeps ONE of
  every op TYPE (each attention kind, dense + MoE, norm/rope/residual variants) + the real tokenizer.
  Purpose: clear the wiring-break ladder in SECONDS, not minutes on a scarce big-mem node (GLM: ~18
  breaks). tp=1 both sides so the deterministic dummy is trivially bit-identical. It proves each op TYPE
  executes + matches — it is NOT proof that ALL layers / REAL shapes are correct (shrunk hidden/heads
  hide tile-size, real-expert-routing, and depth-interaction bugs).
- **FULL-MODEL DUMMY (the AUTHORITATIVE all-layer correctness gate — do this BEFORE optimizing).** Run
  the REAL config (all layers, real hidden/heads/experts, real shapes) with DETERMINISTIC DUMMY weights
  on BOTH sides and diff EVERY layer. No real-weight load, no dtype bridge — just the full wiring+kernels
  at real shape and depth. THIS is the gate that says "every layer is correct" before the optimization
  campaign, and the SAME fast parity then regression-guards each optimization (optimize → per-layer
  re-diff → integrate). Mirrors the DSV4 methodology.
  - **The one extra requirement: the GPU reference needs MULTI-GPU (the full model >> one GPU), so the
    deterministic dummy must be TP-CONSISTENT** — each GPU rank's weight SHARD must equal the matching
    slice of the CPU's (tp=1) full name-seeded weight. `_dummy_determinism.py` currently fills the LOCAL
    `param.shape` (correct only at tp=1). To go full-model: either (a) make the fill TP-aware (generate
    the GLOBAL name-seeded array per param, copy the rank's shard using the param's sharding metadata —
    output_dim/input_dim, tp_rank, tp_size), or (b) generate a deterministic dummy CHECKPOINT once and
    load it on both sides (the real loader shards it identically — sidesteps TP-aware fill). Do NOT
    default to a single-GPU tiny run when multi-GPU (e.g. H200 ×8) is available — that under-scopes the
    gate to op-type coverage instead of all-layer correctness.
- **REAL weights (final):** the dtype bridge + task accuracy, per the WEIGHT AXIS above.
Net: tiny-dummy (bring-up, op-type) → full-model-dummy (all-layer correctness gate + opt regression
guard, multi-GPU ref) → real (bridge + task). The middle rung is the one that proves the whole model.

### ⛔ CAPACITY BUDGET — size the FULL memory footprint BEFORE launching ANY full-model parity/capture run
Do this UP FRONT (part of scope/infra planning), not via load-to-OOM. A wrong guess costs a chain of
~15-min load-to-failure round-trips (GLM real-weight capture burned 3: mem_frac 0.15 → state-cache
NEGATIVE, 0.9 → OOM-killed, then context-cap 8192 + mem_frac 0.5 fit). Before the run, COMPUTE
resident + load-PEAK memory and confirm it fits the planned node(s):
- **Weights (and the dtype bridge DOUBLES or more).** fp8/fp4 ckpt dequanted to bf16 at load ≈ 2× the
  on-disk size; add an AMX-prepack transient copy → load PEAK can be ~2–3× the on-disk low-bit bytes
  (GLM: 306 GB fp8 → ~600 GB bf16 resident, higher peak). Multi-GPU: divide by TP and check PER-GPU.
- **KV pool (attention)** = `context_length × max_total_tokens × per-token-KV`. A model's NATIVE context
  (GLM ~1M) over-reserves massively — this is usually the biggest single pool.
- **State cache (mamba / linear-attn, e.g. KDA)** = `max_running_requests × per-req-state` — LARGE for
  linear-attention; a capture needs only `max_running=1`.
- **Activations** for the prefill (small for a short capture prompt).
- **REDUCTION LEVERS (apply before flagging a roadblock):** cap `context_length` (shrinks the KV pool —
  the biggest lever for long-context models), `MAX_RUNNING=1` (shrinks the state cache), set
  `mem_fraction` correctly (**CPU `mem_fraction_static` = TOTAL engine budget model+pools, NOT a
  GPU-style KV-only fraction** — too low starves the state cache, too high OOMs; GPU side it IS KV-only),
  pick a bigger-RAM node, or TP-shard across MORE GPUs and size per-GPU.
- **If it STILL doesn't fit after all levers → FLAG A ROADBLOCK with options:** keep weights low-bit
  in-memory (skip the bf16-bridge doubling — W8A16/W4A16 dequant-in-GEMM), stream/checkpoint the capture
  layer-by-layer, reduce depth for a first pass, or acquire more/bigger nodes. Surface the number + the
  options; do NOT just launch and hope.
- **Anti-pattern this kills (cost: a needless 300–600GB load):** reflexively standing up the FIRST
  CPU-vs-GPU parity on the REAL checkpoint because 'dummy differs run-to-run'. Make dummy deterministic
  and the early parity is cheap; reserve real weights for the dtype-bridge + task-accuracy FINALE.
5. **Full-model coherence (Layer 0 generation) + task accuracy (Layer 2, gsm8k) come LAST**, once
   the model is fast enough for generation to be feasible — they are the final end-to-end proof,
   NOT the per-change correctness loop. Report any perf number as UNVALIDATED until Layer 1 passes.
The anti-pattern this kills: "load full weights + run gsm8k to check correctness" on unoptimized
code — infeasible (watchdog) AND unnecessary (the fingerprint already localized correctness to the op).

## Reference
- Primary: the HF `modeling_*` forward in fp32/bf16 on the same inputs.
- Secondary: the GPU SGLang path for the same model (catches SGLang-specific
  wiring differences vs pure HF).
- For OLMo 2 / OLMoE, also cross-check the models' published eval numbers.

## Layer 0 — coherence smoke test (do FIRST; catches gross bugs for ~free)
**"It ran" is NOT "it's correct."** A benchmark on RANDOM/DUMMY inputs (`bench_one_batch` with
`--load-format dummy` or random token ids) exercises shapes and TIMING but can pass while the
model emits garbage — random inputs have no "right" output to check against, and a fast
non-erroring run proves nothing about correctness. Before any parity work, greedily generate a
few tokens from a handful of REAL prompts and READ the text:
- Coherent English/code → proceed to Layer 1.
- One token repeated (`"Didži Didži…"`), all-same-token, or gibberish → a GROSS bug (a zeroed/
  no-op op, wrong expert routing, a scrambled layout). Stop and localize BEFORE measuring parity.

**SMOKE A LONG PROMPT AND A BATCH, not just a short one — token-count-gated CUDA paths hide otherwise.**
A short smoke prompt exercises only the small-M path. Many models DISPATCH BY TOKEN COUNT and route
large-M / long-context inputs to a different kernel — often a CUDA-only one (`deep_gemm`, TF32 prenorm,
chunked prefill, a big-M GEMM branch) that isn't wired on CPU and crashes with `NameError`/`ImportError`
ONLY above the threshold. Real case (this repo): MHC `hc_pre` used `deep_gemm.tf32_hc_prenorm_gemm` once
`x.shape[0] >= 1024`; the short "Paris" prompt passed, gsm8k's ~1200-token 8-shot prompts crashed. So the
coherence smoke MUST include one prompt LONGER than every dispatch threshold (grep the forward for
`>= *_MIN_TOKENS`, `deep_gemm`, extend-vs-decode branches) plus a multi-prompt batch — before the task harness.

**Watch for silently no-op'ing / stubbed forward paths — the #1 gross-bug source.** A WIP branch
that returns EMPTY/ZERO when a feature isn't wired does not error; it silently produces plausible-
but-wrong output. Real case (this repo): the decode sparse-attention selection was stubbed
("later increment / stash empty for now") → empty indices → the attention gathered nothing →
`out=0` → attention became a no-op → the model emitted a single repeated token. Any
"stub/empty-for-now" branch on the forward path is a correctness landmine: give it a correct
**fallback** (e.g. dense attention over all valid KV when the sparse selection is empty), never a
silent zero. Grep the wired forward for `empty`/`zeros(0`/`return None`/`pass  # TODO`.

## Isolation — localize the bug with a correct-by-construction variant
When output is garbage, bisect by swapping ONE component for a slow-but-obviously-correct
reference that shares the rest of the pipeline: e.g. run the MoE via a plain torch
dequant+matmul reference (same weights) while keeping the real attention/norm/rope. Coherent →
the bug is in the swapped component; still garbage → it's in the SHARED path (attention/rope/
norm). Raise the framework **watchdog timeout** first (`watchdog_timeout`) — a deliberately slow
reference otherwise gets killed as a false "hang" (exit 137 with a watchdog stack).

## Multi-vector differential capture (when you have 2-3 suspects, resolve them in ONE run)
Single-swap isolation needs one run PER suspect. When the model load dominates (~12 min) the
bottleneck is NUMBER OF RUNS, so instead **capture every candidate op's tensors in ONE pass on both
the reference and the target, then diff offline** — one run dispositions all suspects. This is the
correctness analog of `high-information-runs`.
- **Build an op-level reference bundle.** Run the trusted path (GPU SGLang / HF) once with a hook
  that saves each op boundary's INPUT and OUTPUT (embedding, each attn sub-op q/k/v/scores/out, MoE
  out, norms, logits), keyed by `(layer, op, n_tokens, tp_rank)`. This bundle IS the oracle.
- **Op-level teacher forcing kills cascades.** A wrong op makes every downstream op look wrong. Feed
  each target op the reference's ground-truth INPUT (not the target's own upstream output), so each
  op is judged on clean input. First op whose OUTPUT diverges (given matching input) is THE bug — no
  cascade, no ambiguity.
- **Instrument all suspects at once with a disposition matrix.** Before submitting, list the 2-3
  suspects and the tap that distinguishes each; one run collects all. E.g. q-path bug: tap `q_lora`,
  `wq_b` out (nope+rope), and post-norm+rope out → `qlora` differs ⇒ wq_a/q_norm; `in` differs ⇒
  wq_b; `out_rope` differs ⇒ rope kernel — three suspects, one run.

### Diff pitfalls that cost real runs (do these or the run is wasted)
- **COSINE IS MAGNITUDE-BLIND — always pair it with relative max-abs-err / magnitude ratio.** A
  systematic scale drift (e.g. a CPU fp8 dequant path ~8% hot) passes `cos≈1.0` yet compounds across
  layers into garbage. Real case: "fp8 proj parity cos=1.0" hid the actual bug for a long time.
- **Coarse fingerprints (mean/max/first-4) hide permutations, direction flips, and scale.** Save full
  tensors; compare with cosine AND max-abs-err AND magnitude ratio.
- **Match sequences or the diff is meaningless.** The reference (GPU) does CUDA-graph/warmup passes,
  padding (e.g. 256-tok), and **TP-shards** head-dim tensors across ranks; the target (CPU tp=1) does
  not. Filter warmup by token-count/magnitude gates, key by `n_tokens`, and **rank-tag head-sharded
  tensors** (q, attn-core-out) so rank0=head0 lines up with the target's head 0. Unsharded quantities
  (MLA `wkv` latent, the final all-reduced attn output) compare directly.
- **Validate your tap indexing.** `x[0, -4:]` on a `[T, H, D]`-flattened tensor grabs the LAST head,
  not head 0 (and under TP that's a different head on each side) — a silent wrong-head compare wastes
  a whole run. Print shapes; assert the slice hits the head/dim you mean.
- **A self-check reference shares your blind spot.** A naive-torch twin only catches bugs the twin
  doesn't ALSO make (a from-scratch port checked against another from-scratch port passes while both
  are wrong). The independent GPU/HF oracle is the spec.

### Fine-grained full-tensor GPU-oracle localization (the fast path — build this FIRST, it collapses days to hours)
Proven end-to-end on DeepSeek-V4 CPU. When output is garbage and op-by-op elimination is slow, stand up
a **full-tensor per-layer diff against the GPU oracle** and drill hierarchically. This is the single
highest-leverage tool; reach for it before hand-bisecting.

**⛔ LAUNCH THE GPU REFERENCE CAPTURE IN PARALLEL AT BRING-UP START — day-0, not after you get stuck.**
The GPU reference side is INDEPENDENT of the CPU port: it only needs the real checkpoint + the capture
hook on the native-GPU path. So the moment CPU bring-up begins, retarget the capture hook to the new
the model's decoder layer and FIRE the GPU reference-forward job on the GPU partition, co-running with
the CPU bring-up on the farm. It saves the frozen per-(pass,op,layer,rank) fingerprint to disk so it's
ALREADY WAITING when the CPU build first produces a matching forward — turning the parity diff into an
offline step with zero extra wall-time. Do NOT defer it as "blocked on the CPU running first" (a mistake
made on GLM-5.3): only the DIFF needs both sides; the GPU DUMP does not need the CPU at all.
**WHICH WEIGHTS for the oracle — follow the WEIGHT AXIS above:** the EARLY parity oracle is a
DETERMINISTIC-DUMMY capture on the tiny arch-faithful config (cheap, bit-identical to the CPU dummy
build via `_dummy_determinism.py` — NOT the old 'dummy is useless, must be real' rule); the REAL-weight
GPU fingerprint is captured + archived day-0 too but is the FINAL-stage gate (dtype bridge + full depth +
task accuracy), not the first parity. Retargeting the hook to a new arch (swap the decoder-layer class +
sub-op taps) is a cheap per-model step — budget it as part of bring-up setup.

**⛔ GPU-ORACLE INFRA IS A RECURRING TIME SINK — use the proven recipe, and keep it OFF the CPU-correctness
critical path.** Most GPU-reference breaks are *plumbing*, not CPU bugs (podman store, image pull,
cuda-graph capture asserts, DSA kpool constraints, KV OOM, dtype checks) — the CPU build was already
parity-proven, so don't read GPU-ref churn as a CPU regression. Standardize ONCE:
- **Podman store (HPC):** a PERSISTENT per-model store on /scratch pulled once + reused (`podman --root
  /scratch/.../podman-<model>-store --runroot /tmp/rr-$JOBID --cgroup-manager=cgroupfs --storage-driver
  overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs --storage-opt
  overlay.ignore_chown_errors=true`), `image exists || pull` (**pre-warm** it on the login node in the
  background — the pull is ~7-9 min fuse-overlayfs/NFS unpack), `--userns=host`, per-job runroot on local
  /tmp, clean ONLY the runroot on EXIT. A *shared* store accumulates stale container locks from --rm runs
  that can't delete busy NFS `.nfs` files; some nodes have no writable `/run/user/$UID`; a CLI `--root`
  does NOT inherit storage.conf overlay opts. Run GPU-ref jobs serially against the per-model store.
- **Save INSIDE the forward**, not at exit: sglang runs the model in a scheduler subprocess that is
  hard-killed on shutdown → atexit/LogitsProcessor saves there never fire and the driver's buffer is
  empty. `torch.save` per capture in the forward hook (self-diagnosing: log class-found + hidden shape +
  pid). Key off the SEQUENCE dim `shape[-2]` (GPU hidden is 3D `[B,T,H]`, CPU 2D `[T,H]`); normalize to
  `[T,H]`. Verify the fingerprint where torch EXISTS (offline / in-container), not on the torch-less host.
- **Tiny-config vs GPU-kernel constraints:** shrinking a config for cheap CPU bring-up can VIOLATE GPU
  kernel asserts the CPU path no-ops (GLM DSA: `index_n_heads` H%4==0 & N%8==0; decode kpool
  `group_topk=index_topk//index_kpool ∈ {128,160,192,224,256,512,2048}`; 1M `context_len` + high
  mem_fraction OOMs the decode flash-attn transient). For params the CPU ignores but the GPU constrains,
  keep real(ish) values and LOWER mem_fraction for the GPU run; at tiny ctx top-k=all=dense so parity
  holds. Jump to a known-good value — don't bump one divisibility step per GPU round-trip.
1. **Reusable capture hook on BOTH sides, env-gated, saving FULL fp32 tensors (never fp[:4]).** One hook
   file per side (CPU plugin + a GPU-reference hook) that, when its env flag is set, wraps the decoder
   layer + key sub-ops and `torch.save`s a dict keyed `{pass}.{op}.L{layer}.r{rank}` where pass ∈
   {prefill `pf`, first-decode `dc0`}, op ∈ {layer-output, attn-block, moe, attn-core, inverse-rope, …}.
   Trigger the save in the LogitsProcessor wrapper (also captures logits). Gate prefill vs decode by
   token-count (T in prompt-range vs T==1 after prefill). Keep the GPU hook a PURE capture (no
   forward-altering A/B in the same run — that contaminates the oracle).
2. **Cosine + magnitude PER LAYER, CPU vs GPU.** Find the FIRST layer whose output cosine drops. A
   severe drop AT layer 0 (not a slow compounding decay) means one op in that layer is grossly wrong —
   not precision. (Compounding ~0.99→lower across many layers ⇒ a scale/precision drift instead.)
3. **Drill into the first diverging layer by sub-op, same run's taps.** Split attention-block vs MoE
   vs MHC-residual; then within the bad block split the core (q-in, attn-core-out) vs the projection
   (inverse-rope, wo_a/o_proj). The first sub-op whose OUTPUT diverges given a MATCHING input is THE op.
   Real result: q cos 1.000 → attn-core-out cos 1.000 → inverse-rope cos 1.000 → o_proj cos 0.02 ⇒ the
   bug is wo_a alone.
4. **GPU-TP alignment gotchas (or the diff lies):** the oracle runs TP>1 and (a) pads per-rank heads to
   64 with `new_empty` GARBAGE (nonzero → breaks norm-threshold "valid head" detection; use the known
   n_local_heads, not a magnitude test); (b) each rank fills heads `0:n_local` with its GLOBAL block
   `r*n_local:(r+1)*n_local` → compare CPU[r*n_local:(r+1)*n_local] vs GPU[0:n_local]; (c) the single
   saved file is written by whichever rank runs logits LAST (varies run-to-run) → strip the `.rN`
   suffix and read the rank off the key. Replicated tensors (MLA latent, post-all-reduce block output)
   compare directly regardless of rank.

### Bug class: a weight consumed by a HAND-WRITTEN einsum/matmul that bypasses the linear method `.apply`
The most-likely gross bug once the *math* checks out. On CPU (and ROCm) the linear method's
`process_weights_after_loading` **VNNI/AMX-prepacks** (or B-preshuffles) every weight so its own AMX
GEMM (`.apply`, `weight_packed_linear`, is_vnni) can read it. But some ops read `self.W.weight`
DIRECTLY and run a bespoke batched GEMM/einsum that expects the **plain row-major logical** layout
(e.g. DeepSeek-V4 `wo_a` absorb: `einsum("tgd,grd->tgr")`). The prepack silently permutes the weight →
that op reads packed bytes as logical → output is **orthogonal to truth (cos≈0) with ~preserved
magnitude** (permutation keeps the norm) → whole-model garbage.
- **Diagnostic (nails it in one dump):** save the RUNTIME weight the op actually consumes; compare to
  `dequant(checkpoint weight, scale)` in logical layout. `abs-mean matches` (scale/dequant correct) +
  `cosine≈0` + `sorted-values match (pure permutation)` ⇒ prepack/layout, NOT scale. If magnitude were
  also off ⇒ a missing/!wrong scale instead.
- **Find all such sites up front:** grep the wired forward for weights used outside `.apply` — bespoke
  `einsum`/`bmm`/`torch.matmul(x, layer.weight...)`, grouped/absorb GEMMs, MoE hand-kernels.
- **Correctness fix:** keep that weight row-major (skip the prepack) — mirror any existing opt-out the
  model already has for another backend (DeepSeek-V4 sets `skip_aiter_bpreshuffle=True` on `wo_a` for
  ROCm; the CPU path just lacked the equivalent). Identify the weight reliably (tag it in the module
  `__init__`; do NOT rely on a marker that's only set on a code path your platform doesn't take — on
  CPU `wo_a` was UNQUANTIZED bf16 so it went through `UnquantizedLinearMethod`, not the fp8 method, and
  had no fp8-only marker). **Perf caveat:** row-major + generic einsum is NOT AMX-accelerated — leave a
  roofline-phase TODO to route it through the packed AMX GEMM per group instead of un-packing.

## Layer 0.5 — low-bit kernel parity gate (per dtype bridge)
**Every `dtype_bridge` from the dtype audit (`model-op-decomposition` §2b) gets its own numeric
gate — coverage ("a bf16/fp8 kernel exists") is NOT correctness.** When a kernel consumes weights
whose STORED dtype differs from its compute dtype (fp4/mxfp4/fp8/int4 experts on a bf16/fp8 AMX
GEMM), a dequant+repack bridge (scale decode, nibble order, group/block layout, VNNI prepack,
SwiGLU gate/up split) sits in between and fails SILENTLY (plausible garbage, not a crash). Prove
it before trusting end-to-end output:
- **Independent oracle from the SAME packed bytes.** Compare the real kernel output to a torch
  dequant+compute reference derived from the checkpoint's own packed weights + scales — decode the
  low-bit values per the KERNEL's exact convention (read the kernel source, e.g. csrc/cpu/vec.h
  `cvt_mxfp4_e2m1_bf16`: standard OCP e2m1 LUT, low-nibble-first, per-32 e8m0 scale = 2^(byte−127)),
  NOT via a different quant path that may use another scale layout. Assert cosine ≥ 0.99 and small
  relative error (bf16 accumulation vs fp32 reference → tolerate ~1e-2, but a wiring bug blows past
  it by orders of magnitude).
- **Two complementary forms.** (1) A STANDALONE synthetic test (random packed weights → prepack →
  kernel vs oracle) proves the kernel + packing math, is CI-able, and needs no model load — but must
  run on real ISA hardware (AMX). (2) An IN-SITU probe gated by an env flag stashes the first layer's
  raw packed weights before they're replaced by the prepacked ones, then on the first real forward
  compares kernel vs oracle on the REAL checkpoint + REAL activations — this catches checkpoint-
  wiring bugs the synthetic test cannot (e.g. the loader's float32-pow2 scale layout, real SwiGLU
  order). Worked example in this repo: `plugin/validate/test_mxfp4_moe_cpu.py` (standalone) +
  `INTEL_CPU_DSV4_MOE_PARITY=1` (`[MXFP4 PARITY]` in-situ), mirroring the DSA `test_dsa_*` gates.
This gate localizes a bridge bug to the exact kernel; end-to-end coherence (Layer 0) or task
accuracy (Layer 2) would only tell you "something is wrong" somewhere in a 43-layer forward.

## Layer 1 — numerical parity (localizes)
1. Feed identical token inputs to reference and CPU model.
2. Capture per-decoder-layer hidden states + final logits.
3. Metrics per capture point: cosine similarity and max-abs / relative error.
4. Tolerance (bf16 CPU vs bf16 ref): cosine ≥ 0.999 on hidden states, top-1 logit
   agreement 100% and top-5 KL small on a fixed prompt set. Tighten if the
   reference is fp32.
5. The FIRST layer whose error exceeds tolerance identifies the mis-wired op —
   hand that op back to `cpu-model-wiring` (usually a prepack/layout or a
   norm/rope placement bug).

## Layer 2 — task accuracy (end-to-end truth)
Run the in-repo harnesses on CPU vs reference:
- `benchmark/gsm8k` (reasoning), `benchmark/mmlu` (knowledge), `benchmark/hellaswag`
  (commonsense).
- Pass if each score is within N points of the reference (default N=1.0 absolute,
  or within run-to-run noise), on a fixed decode config (greedy or fixed seed).

### Certify against a trusted reference — a bare task number certifies NOTHING
A standalone score (e.g. "61%") proves nothing alone; it must be tallied against a TRUSTED reference,
APPLES-TO-APPLES. Reference hierarchy + the two checks it enables:
- **Published number at the SAME setting** (model card / tech report). MATCH THE PROTOCOL EXACTLY — shots,
  prompt style (CoT vs short-answer), `max_new_tokens`, stop strings, answer extraction. A protocol mismatch
  ALONE can cost ~30 points and masquerade as a correctness bug: here DSV4-Flash-**Base** scored ~61% under
  our 256-token / `"\n\n"`-stop harness vs the card's **90.8% gsm8k 8-shot**, because the short cap + `\n\n`
  stop TRUNCATED the chain-of-thought before the final answer (gsm8k 8-shot is conventionally CoT → needs
  ~512 tok, stop only on `"Question"`). Replicate the published protocol before concluding anything.
- **Base models need CLEAN CoT exemplars, not raw dataset answers (applies to ANY model on the task).**
  Feeding the raw gsm8k `answer` field (with `<<calc>>` annotations + a bare `#### N`) made this base model
  DEGENERATE after answering — unrelated math, repeated `</s>`, "Confidence: 100" — which both deflates the
  score AND corrupts last-number parsing (measured ~78% full-run vs the clean-exemplar protocol that produces
  the published number). Strip dataset-specific markup and end every exemplar with an explicit marker the
  model will imitate ("The answer is N."), then extract THAT marker. `task_gsm8k._clean_answer` +
  `extract_final_answer` do this generically — reuse them for any model, don't re-hand-roll parsing.
- **GPU oracle running YOUR harness** = the tightest certification; it separates two questions:
  - **CPU vs GPU, same harness** → certifies the CPU IMPLEMENTATION (greedy → expect near-identical; strongest
    form = TOKEN-LEVEL parity on the task prompts, which also catches the long-context / long-generation
    divergence that short-prompt per-token parity misses).
  - **GPU vs published** → certifies the HARNESS PROTOCOL is faithful.
  So if CPU==GPU but both < published → the gap is YOUR harness (fixable, not a bug); if GPU==published but
  CPU<GPU → a real CPU bug.
- **Diagnose a gap CHEAPLY first (measure-first):** dump a few real generations (gold vs full text vs
  extracted) to distinguish TRUNCATION vs PARSE vs WRONG-REASONING before building a GPU env — only reach for
  the GPU oracle if the dump implicates the CPU forward. Worked example: `diag_gsm8k_dump.py`.
- Do NOT publish a task number as "certified" (into README / enablement-certificate) until it is tied to a
  reference this way.

### Standard-harness-on-CPU (lm-eval) — the GPU-free way to isolate HARNESS vs MODEL
When the GPU oracle is blocked or you want a cheaper decomposition, run the **community-standard harness**
(EleutherAI `lm-evaluation-harness`) against your CPU server and compare to published. This moves the only
variable to the HARNESS: **standard-harness-on-CPU ≈ published → your custom harness was the gap (CPU correct);
≈ your-harness number → the model genuinely scores that under a standard protocol and the published figure is
the author's own eval setup.** Recipe (sglang CPU server on a port + `lm_eval --model local-completions`):
- **PRE-FLIGHT the client glue BEFORE paying the model load** (a server reload here is ~9 min; don't burn it to
  discover a missing dep). Construct the lm-eval model object with NO server (`create_from_arg_string`) and load
  the task index — it exercises every import/arg. Glue that bit here, in order: `pip install lm-eval[api]`
  (tenacity/aiohttp), `transformers` must be installed (api_models imports it even with `tokenized_requests=False`),
  and pass `tokenizer=<LOCAL model dir>` or it tries to fetch the HF repo named by `model=` and 401s.
- **Client `timeout` defaults to 300s — far too short for slow CPU decode.** Each gsm8k CoT (~256 tok) can take
  **~7 min/question** on an unoptimized CPU forward; 16 concurrent long gens blow the 300s client timeout →
  `aiohttp asyncio.TimeoutError` → **zero results after a full model load.** Set `timeout=3600` and
  `num_concurrent` LOW (2 — CPU is the bottleneck, high concurrency just thrashes the detokenizer and inflates
  per-request latency).
- **Size the sneak preview by the real per-question cost.** At ~7 min/question, `--limit 8` ≈ ~50 min, `--limit 40`
  is hours. Start with a SMALL `--limit` for the directional answer; only scale up once it's landing.
- **Make the run OBSERVABLE — never `lm_eval ... | tail -50`.** `tail` buffers to EOF, so the tqdm progress bar
  AND the results table are invisible until the job exits (you can't tell slow from hung). Stream unbuffered
  (`stdbuf -oL -eL lm_eval ... 2>&1`, no pipe) and add `--log_samples` for an incremental per-sample dump. Read
  live progress from the LOG (`tr '\r' '\n' | grep 'Requesting API'` → `X/N`), not from the server.
- **NEVER probe a saturated CPU server for a "preview".** A `/v1/completions` probe queues behind the eval's
  own requests, adds contention, and SLOWS the actual run (it also wedged the detokenizer here). `/health` is the
  only safe poke (instant). For a real partial score, launch a SEPARATE small `--limit` job, don't poke the
  running one.

### GPU oracle on rootless-podman HPC (when you DO build it) — the infra, not the model, is the long pole
Standing up the GPU reference via a rootless container is a chain of real blockers (each distinct, they compound):
no `/etc/subuid` range → image unpack can't lchown (set overlay `ignore_chown_errors=true`); run-time userns
(`--userns=host --cgroup-manager=cgroupfs`); `--mem-fraction` default may be CPU-tuned (raise for GPU KV); native
MXFP4/fp8 deep_gemm JIT needs a real-fs `TMPDIR=/dev/shm` (fuse-overlayfs breaks the cicc→ptxas handoff); the
image store is NOT concurrency-safe (give each job a PRIVATE `--root` on big storage, and CLI `--root` does NOT
inherit storage.conf overlay opts so repeat them as `--storage-opt`). Full verified recipe lives in repo memory
(`/memories/repo/dsv4-accuracy-perf-caveats.md`, "GPU ORACLE" section) — reuse it, don't re-derive. KEY JUDGMENT:
the GPU oracle is COMPLEMENTARY; if per-layer bit-parity already certifies the forward AND the standard-harness-
on-CPU test isolates the gap, you do NOT need to brute-force the GPU path — stop and use the evidence you have.

### Decouple GENERATE (expensive, model) from SCORE (cheap, pure) — test the diagnosis for free
The model generation costs minutes/run; **stop-truncation + answer-extraction are pure string ops
(microseconds)**. So never re-run the engine to test a scoring/protocol idea:
- **Generate RAW once** on a small sample (long `max_new`, minimal stop e.g. only `["Question"]`) and SAVE
  `(text, gold)` (`--gens-out`). Then **grid-search `{stop-set} × {extractor}` OFFLINE** to LOCK the protocol
  (`score_gsm8k.py`) — milliseconds, no cluster. Here each engine diag was ~25 min; the offline grid is instant.
- **Make the full/sharded run ALSO save generations** (`--gens-out`) so any later extractor/stop improvement
  **re-scores the saved corpus for FREE** — you never re-generate to answer a *scoring* question; you re-run
  the model only to answer a *generation* question (different weights/kernel/sampling).
- This is the accuracy-harness analogue of the perf loop's cached-golden guardrail: pay the expensive step
  once, iterate the cheap step offline. Lock the protocol on the cheap grid BEFORE the long pipelined run.
- Worked example (DSV4-Flash): the 61%→? fix (drop `"\n\n"` stop; extract first `####`/"answer is"/`\boxed{}`
  instead of the last number of a base-model ramble) was unit-tested offline (475→460) before any full re-run.

## Fast guardrail for the PERF-OPTIMIZATION loop (re-verify accuracy cheaply after every kernel change)
During perf work you re-check accuracy constantly — do NOT pay full task generation each time. FIRST
MEASURE where the wall-time goes (load vs generate): in this repo the model LOAD was ~8 min (149 GB/46
shards over NFS) but the ~50-min sink was gsm8k GENERATION (long 8-shot prompts × 160 tokens on CPU). So:
- **Regression guardrail = short-prompt cached-golden parity, not the task harness.** Capture a golden
  greedy-completion set ONCE from the known-good build on a dozen SHORT prompts × a few tokens, then after
  each perf change compare CPU vs golden (exact-match, no HF reload, no long generation) → seconds of
  compute. A mis-wired/regressed kernel diverges immediately. (Worked example: `accuracy_parity.py
  --golden-out`/`--golden-in`, `run_parity_fast.sbatch`.) Keep short prompts under any token-count
  dispatch threshold so the fast check and the real path agree.
- **Cut repeat LOAD time — but NOT via tmpfs for a big model.** Staging weights to `/dev/shm` is RAM-backed,
  so for a model that is a large fraction of node RAM (149 GB on a 1 TB box) the tmpfs copy competes with
  the model's own loaded/dequantized footprint + KV reservation and OOM-kills the forward (learned the hard
  way). Only tmpfs-stage when the model is a SMALL fraction of RAM. Otherwise load from NFS directly (~8 min)
  and cut the recurring cost elsewhere: cache the post-dequant/prepack weights, or keep a warm engine for a
  fixed build. Also give the guardrail a raised `watchdog_timeout` (default 300 s kills a slow first forward).
- **Reserve the full task harness (gsm8k/mmlu) for PERIODIC checks**, not every iteration; raise
  `watchdog_timeout` (default 300 s kills slow CPU long-prompt forwards) and cap `chunked_prefill_size`.
- **Run the task harness CHUNKED + CHECKPOINTED (never one blocking generate over all N).** A single
  `engine.generate(all_prompts)` writes the result ONLY at the end, so a slurm wall / watchdog kill yields
  NOTHING — and on slow CPU decode a full 1319-Q gsm8k run can exceed the wall (measured: 3h+ with no
  completion, killed at the 4h limit, zero output). Instead generate in CHUNKS (e.g. 64 Q), accumulate
  correct/invalid, and write a PARTIAL json after EACH chunk (atomic temp+rename) + print one progress line
  per chunk. Two payoffs: (1) any kill still leaves a usable running accuracy; (2) you can tell SLOW from
  HUNG — the engine at `log_level="warning"` is silent during generation, so without per-chunk prints a
  live run and a deadlock look identical. Also run a smaller SUBSET first (~300–330 Q is statistically fine
  for a certificate) so it finishes inside the wall. Worked example: `task_gsm8k_chunked.py` +
  `run_gsm8k_dsv4_fallback.sbatch`.
- **Parallelize the task run across NODES — eval questions are INDEPENDENT (embarrassingly parallel).**
  Shard the question set across M nodes with a slurm JOB ARRAY (`--array=0-(M-1)`, each task takes
  `--shard-index $SLURM_ARRAY_TASK_ID --num-shards $SLURM_ARRAY_TASK_COUNT`), run each shard
  chunked+checkpointed, then SUM correct/done across the shard jsons (`combine_gsm8k_shards.py`) ->
  ~M× faster wall-clock, and partial shards still count. Generation (not load) is the long pole on CPU
  (measured: 1319-Q single-node gen ran 3h+ and still walled; the same split 8 ways finishes in minutes),
  so node-parallelism is the real lever. Worked example: `run_gsm8k_dsv4_sharded.sbatch` +
  `combine_gsm8k_shards.py`.
- **SNEAK-PREVIEW + EARLY-ABORT the full run (fail-fast).** Because each shard is chunk-checkpointed, `combine`
  on the partial jsons gives a running accuracy within ~1 chunk (~150–300 Q of a 3 h run). Read it against an
  EXPECTED floor (prior known-good number / the published target) and `scancel` + fix if it's clearly below —
  don't burn the full wall on a regressed harness. Likewise gate the full run on a CHEAP pre-check (small
  diag), but make that pre-check itself chunk-checkpointed so it gives a per-question sneak preview, not an
  all-or-nothing signal at completion. (See `high-information-runs` FAIL-FAST.)
- **MEASURE load contention before "fixing" it — don't pre-optimize the fan-out.** The obvious worry is
  M nodes each reading the big model (149 GB) over NFS at once = thundering herd. MEASURE it first: here
  8× concurrent load was **~2.5 min/node — FASTER than the 7 min cold single-node load**, because the NFS
  server's page cache was WARM from a prior load (reads served from cache, not disk). So staggering would
  only have ADDED a start-time tail (shard k waits k·GAP) to fix a problem that wasn't there. Only stagger
  (`sleep $((SLURM_ARRAY_TASK_ID * GAP))`, env-gated, default 0) or pre-warm the cache (one read before the
  array) if a COLD or genuinely bandwidth-limited load actually shows up — confirm from the shard logs'
  "loading shards" timing before reaching for it.

### Three-tier equivalence validation for the optimize loop (defer task-accuracy to the end)
**Guiding principle:** if the optimized build is EQUIVALENT to the reference baseline (end-to-end, within
tolerance), the optimizations are FUNCTIONALLY TRANSPARENT — accuracy is identical to the baseline, so one
task-accuracy run at the end on the optimized build == the run you'd have done on the baseline. The tiers
below are just the CHEAP PROOF of equivalence that lets you skip a full accuracy run per optimization.
When a kernel is REPLACED by an optimized one, you don't need full task accuracy per change — you need to
prove the optimized kernel is EQUIVALENT to the reference it replaces. Layer the checks by cost:
- **T1 (every change, fast): in-situ A/B equivalence.** Build each optimization behind an env flag
  (reference impl vs optimized). In ONE forward, run both on the SAME real activations and compare
  **cosine AND magnitude-ratio AND relative-max-abs error, full-tensor** (never cosine-alone — it's
  magnitude-blind — never `fp[:4]`). This is an exact equivalence test on real data: no second run, no
  input mismatch. Set a tolerance that separates legitimate low-precision drift (bf16 opt vs fp32 ref →
  ~cos ≥ 0.999, rel-err ≤ 1e-2) from a bug. Use REAL activations — data-dependent ops (MoE routing,
  sparse/topk selection) take different paths on random vs real input.
- **T2 (periodic): full-model per-LAYER cosine vs the reference baseline.** Per-op equivalence does NOT
  guarantee end-to-end — small drifts COMPOUND across depth (per-op cos 0.999 × L layers can collapse).
  Re-establish the current known-good baseline's per-layer tensors as the golden, and diff the integrated
  build against it every few optimizations.
- **T3 (once, at the end): full task accuracy** (gsm8k/mmlu) + ideally full per-layer cosine vs the
  INDEPENDENT oracle (GPU/HF), not just the CPU baseline. Critical nuance: a coherence-validated baseline
  is NOT task-accuracy-validated ("it says Paris" ≠ "it's accurate"); cosine-preserving the baseline only
  proves "no regression from baseline," so the baseline's own absolute accuracy stays unconfirmed until T3.
This defers the expensive task-accuracy run to the end and lets you optimize+validate in parallel, while
T1/T2 still catch regressions immediately. It is the layered oracle (Layers 0.5/1/2) applied to the opt loop.

## Procedure
1. Run Layer 1 on a small fixed prompt set; block on the first out-of-tol layer.
2. Once parity holds, run Layer 2 harnesses; block on any score regression.
3. Record both results (per-layer max error + task deltas) for the certificate.

## Gate
PASS only if per-layer parity within tolerance AND every task score within N
points. Any regression blocks — never ship on "looks close". Report the tolerances
used so the verdict is auditable.

## Pitfalls
- **A fast, non-erroring run proves nothing about correctness.** A dummy-weight or random-token
  bench that hits the target latency can still be numerically garbage — always run Layer 0 on
  REAL prompts before believing a perf number. Report perf numbers as "unvalidated" until the
  accuracy gate passes.
- Comparing against a differently-configured reference (rope scaling, dtype,
  sampling) manufactures fake diffs — pin dtype, rope params, and sampling first.
- Passing task accuracy while per-layer parity fails can happen (error cancels) —
  keep BOTH; parity is what proves the KERNELS are right, task is what proves the
  MODEL is right.
- Validate EACH precision the workflow generated as its own config with its own
  tolerance: BF16 against the tight parity bar; INT8 (w8a8) against a looser INT8
  budget (a small task-score drop is expected from per-channel weight + dynamic-
  token quant) — never judge INT8 against the bf16 bar, and never ship INT8
  without its own accuracy pass.
