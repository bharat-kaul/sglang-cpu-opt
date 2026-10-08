#!/usr/bin/env python3
"""Generate the DSv4 tree-pilot methodology playbook as a downloadable Word doc."""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

NAVY = RGBColor(0x1F, 0x37, 0x64)
BLUE = RGBColor(0x2E, 0x5B, 0xA8)
GRAY = RGBColor(0x55, 0x55, 0x55)
GREEN = RGBColor(0x1E, 0x7A, 0x3C)
OUT = "/data/nfs_home/bkaul/dsv4_pilot_playbook.docx"

d = Document()
base = d.styles["Normal"]
base.font.name = "Calibri"
base.font.size = Pt(10.5)


def h1(t):
    p = d.add_heading(t, level=1)
    for r in p.runs:
        r.font.color.rgb = NAVY
        r.font.size = Pt(15)
    return p


def h2(t):
    p = d.add_heading(t, level=2)
    for r in p.runs:
        r.font.color.rgb = BLUE
        r.font.size = Pt(12)
    return p


def para(t, color=None, bold=False, size=10.5, after=4):
    p = d.add_paragraph()
    r = p.add_run(t)
    r.bold = bold
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    p.paragraph_format.space_after = Pt(after)
    return p


def bullet(t, lvl=0, bold_lead=None):
    p = d.add_paragraph(style="List Bullet" if lvl == 0 else "List Bullet 2")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.color.rgb = NAVY
        p.add_run(t)
    else:
        p.add_run(t)
    p.paragraph_format.space_after = Pt(2)
    return p


def num(t, bold_lead=None):
    p = d.add_paragraph(style="List Number")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.color.rgb = NAVY
        p.add_run(t)
    else:
        p.add_run(t)
    p.paragraph_format.space_after = Pt(2)
    return p


# ---- title ----
t = d.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Model- & Platform-Agnostic Enablement + Optimization Playbook")
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = NAVY
s = d.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s.add_run("A reusable methodology for enabling AND optimizing any model on any accelerator "
              "(CPU shown; the same loop applies to other CPUs and to GPUs)")
r.font.size = Pt(12)
r.font.color.rgb = GRAY
dt = d.add_paragraph()
dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = dt.add_run("Platform and model constants are INPUTS, not assumptions. The concrete pilot instantiation "
               "(DeepSeek-V4-Flash on Intel Xeon / EMR) lives in the per-platform spec + roofline artifacts, not in this method.")
r.font.size = Pt(9)
r.font.color.rgb = GRAY

# ---- overview ----
h1("Overview")
para("Goal: from a bare model config + checkpoint, enable AND optimize the model end-to-end on a target "
     "accelerator, driving final wall-time toward the sum of the per-op roofline floors — platform/runtime "
     "overhead squeezed out and every kept kernel at its ceiling. Nothing here is specific to one model or "
     "one chip: the platform constants, the op set, and the native-kernel library are INPUTS.", GRAY)
para("Two theses, one plugin (wire into the serving stack via its external/plugin mechanism — no fork):", bold=True)
bullet("ops already covered by the target's native high-throughput kernels (e.g. AMX/oneDNN on CPU, "
       "Tensor-Core/cuBLAS/Triton on GPU) are enabled by wiring + validation only (capability inheritance: "
       "dtype, prepack/layout, fused-MoE).", bold_lead="Thesis 1 (covered) — ")
bullet("genuinely novel ops get an AI-written, roofline-tuned native kernel (C/C++ for CPU, CUDA/Triton "
       "for GPU), human-gated.", bold_lead="Thesis 2 (novel) — ")
para("Scope discovery is the router (which ops are covered vs novel); the spine-leaf performance tree "
     "\u2014 the macro/meso/micro per-op roofline loop \u2014 is the engine. One pipeline, re-applied per "
     "(model, platform).", GRAY)
para("Deliverable contract (what the pipeline guarantees):", bold=True)
bullet("Every op / fused-op is AUTHORED as a native kernel \u2014 a C/C++ torch extension on CPU "
       "(CUDA/Triton on GPU). Nothing hot stays in eager python/torch.", bold_lead="1 \u2014 ")
bullet("Each authored kernel is TESTED standalone (parity + roofline) and ONLY THEN integrated into the "
       "serving stack's model behind the cosine-fingerprint gate \u2014 wired as a torch op, no fork "
       "(the pilot: SGLang external model package / PyTorch).", bold_lead="2 \u2014 ")
bullet("All end-to-end model runs execute through the serving-stack engine (the pilot: SGLang / "
       "sgl.Engine), not a bespoke loop \u2014 so wall-time reflects the real inference path.", bold_lead="3 \u2014 ")

# ---- governing two-phase ----
h1("Governing principle — two phases")
para("Per-op completeness FIRST, wall-time/ordering SECOND (a deliberate inversion of ROI-skip).", bold=True)
h2("Phase A — per-op peaking (ALL ops, including fused ops)")
num("Discover every op in the layer, including fused ops — a fused op is a first-class unit with its own roofline.")
num("Per op/fused-op: roofline + binding resource (BW / compute / latency) on the target platform (measured constants).")
num("Arithmetic-intensity improvability: can the binding resource be moved by a lever (increase M/batch, fusion)? Record the achievable ceiling.")
num("Implement the op in the target's native kernel language (C/C++ on CPU, CUDA/Triton on GPU) and optimize to a reasonable extent, driving it to its ceiling (BW peak / compute peak / latency floor).")
num("If the gap to roofline cannot be bridged by known techniques → surface to the user for review.")
num("Every op done this way — none skipped. Completion gate = every op/fused-op at its ceiling.")
h2("Phase B — wall-time + ordering (only after Phase A completes)")
num("Switch to wall-time; fix platform/runtime pathology first (CPU: OpenMP spin-wait/affinity, NUMA bind, allocator; GPU: launch/stream config, occupancy, memory pool; plus dispatch & batching) until kernel-domination clears the bar.")
num("Apply Amdahl ordering — prioritize/aggregate high-share work. Ordering is sequencing, not exclusion.")
para("Bounding valves that keep Phase A finite: \u201creasonable extent\u201d + \u201csurface if can\u2019t bridge.\u201d "
     "Latency-bound \u201cpeak\u201d = the dispatch/overhead floor (fuse/batch launches), not an AI target.", GRAY)

# ---- pipeline phases ----
h1("Pipeline phases (in order)")

h2("Phase 0 — Platform scan + intake + capacity/precision pre-flight (Gate 0)")
bullet("Run the uArch probe on the target device → measured machine constants: per-memory-domain BW, "
       "per-dtype compute peak, ridge = peak/BW, topology (CPU: NUMA/SNC; GPU: devices/interconnect), "
       "ISA/occupancy. No hardcoded constants — swap the spec file to retarget.")
bullet("Data-type audit from the real checkpoint (metadata-only first): {op, stored dtype, target "
       "compute dtype, kernel, inflation?, dtype-bridge, parity gate}. Never inflate a native low-bit "
       "checkpoint; sub-tile dtypes are moved low-bit and computed in the native compute dtype via fused dequant.")
bullet("Capacity pre-flight: resident weights + activations + KV/state vs per-domain memory floor → "
       "parallelism map (tp/EP; EP-first for MoE). On CPU tp>1 hurts latency-bound decode; on GPU pick "
       "tp/EP by memory fit + interconnect.")

h2("Phase 1 — Op scope discovery + decomposition (Gate 1)")
bullet("Enumerate every op (walk each layer index — heterogeneity is easy to miss); 100% compute "
       "coverage; anything unclassifiable is UNKNOWN → treated as a gap.")
bullet("Dependency-closure for non-compute substrate (KV pool, backends, novel non-attention families).")
bullet("Scope discovery assigns each op a lane: A = covered by a native kernel on the target (donor); "
       "B = missing on the target but a reference impl exists elsewhere (numerical oracle); "
       "C = missing, author from the math spec.")

h2("Phase 2 — Fusion pass → roofline (per op, target platform)")
bullet("Fusion analysis first (vertical / horizontal / epilogue). For each candidate: consult the "
       "GPU/XPU implementation where available — runnable fused code is a numerical ORACLE; a "
       "performance claim without code is a HINT only — OR derive the benefit MATHEMATICALLY on the "
       "roofline (bytes saved + AI before→after + L2-resident gate) where no reference exists. Absence "
       "of a GPU reference is not a reason to skip (CPU is more BW-bound, so a CPU-only fusion may pay).")
bullet("Map each fusion: COVERED (donor fused kernel) / NEW-C++ (→ kernel authoring) / SKIP (record why).")
bullet("Per-op and per-fused-op roofline: FLOPs, weight+activation+KV bytes, arithmetic intensity, "
       "binding resource, achievable ceiling, and the native-kernel implementation lane. Phase-split prefill vs decode.")

h2("Phase A / Phase B — execute per the governing principle above")

# ---- progressive wall-time tracking ----
h1("Progressive wall-time tracking (publication ledger)")
para("Each optimization is proven end-to-end, not just in a microbench. We keep a running "
     "wall-time ledger from the unoptimized baseline through every integrated op, so the final "
     "publication shows a monotonic, correctness-gated speed-up curve.", bold=True)
para("Use the CHEAPEST representative proxy at every altitude, wherever feasible, and recover the "
     "full-scale number ANALYTICALLY \u2014 validated by ONE full run at the end: (a) op optimization uses the "
     "isolated-op microbench at real op dims (and reduced context S where the op scales trivially in S); "
     "(b) wall-time uses the depth-reduced model with linear depth-scaling to full N. Proxies move us "
     "faster; the single full-depth run at phase end confirms the analytical scaling held.", GRAY)
para("Ledger artifact: plugin/validate/results/wall_time_progress.json (baseline row + one row per "
     "op, filled after each integration).", GRAY)
h2("Parallel authoring, serial integration, per-pass record")
para("Ops are independent work items (distinct kernel, distinct microbench), so AUTHOR THEM IN PARALLEL "
     "(one op per idle node) and QUEUE the finished best-of versions. INTEGRATION stays serial: dequeue "
     "one op at a time, cosine-gate, re-run the wall-time ledger. Only the integrated re-profile is serial.", GRAY)
bullet("Per op, record EVERY optimization pass (before/after): approach, off-machine-peak and vs-reference "
       "per M, correctness (cosine / set-match), kept?, commit. The final KEPT pass is the best-of, reported "
       "against the roofline (at-ceiling, or surfaced-as-plateaued with the residual-gap cause).")
bullet("Per-op pass ledgers: plugin/validate/results/op_passes/<op>.json (schema in _schema.json).")
bullet("Torch/donor fallback registry: every point where the kept best-of falls back to torch (or a "
       "donor) instead of the custom path is recorded with its reason + limit justification "
       "(plugin/validate/results/torch_fallbacks.json). A fallback is legitimate only if strictly best-of "
       "(no regression at that point) OR at the measured limit with no lever left. The WHOLE registry is "
       "SURFACED FOR REVIEW at the end of the optimization + integration phase.")
h2("The loop (repeat per op, in Phase-A order)")
num("Baseline wall time FIRST: run the model end-to-end (depth proxy, deterministic-dummy weights) and "
    "record the steady-state median latency at each M (step 0 = the reference; never time the cold step).",
    bold_lead="0 \u2014 ")
num("Optimize the op standalone to its machine-peak ceiling (author \u2192 measure \u2192 iterate \u2192 keep "
    "best-of), then surface-and-stop when levers are exhausted.", bold_lead="1 \u2014 ")
num("Integrate the kept kernel into the model behind a COSINE FINGERPRINT gate: per-layer + per-op "
    "cosine vs a trusted reference (eager / GPU / HF) on the SAME dummy weights must stay \u2265 0.9999 "
    "(set-match for top-k ops). The first layer/op below threshold localizes a break \u2014 do not proceed until it passes.",
    bold_lead="2 \u2014 ")
num("Re-run the model end-to-end (dummy weights, same M sweep) and record the new wall time + the delta "
    "vs the previous row and the cumulative speed-up vs baseline.", bold_lead="3 \u2014 ")
num("Repeat for the next op. The ledger accumulates to the final published curve: baseline \u2192 "
    "op-by-op \u2192 fully-optimized.", bold_lead="4 \u2014 ")
h2("Depth-proxy timing + analytical scaling to full depth")
para("Perf runs use DUMMY weights (no real load) AND a depth-reduced proxy. When the layer stack is a "
     "repeated archetype (homogeneous transformer \u2014 the common case), end-to-end wall time is LINEAR in "
     "depth, t(L) = t_fixed + L\u00b7t_layer, so a few-layer proxy is representative (pilot: 4 layers vs the "
     "full N). For HETEROGENEOUS stacks, include each distinct archetype at least once and scale per-archetype.", GRAY)
bullet("Calibrate ONCE at baseline from two proxy depths p and q: t_layer = (t_q \u2212 t_p)/(q \u2212 p) and "
       "t_fixed = t_p \u2212 p\u00b7t_layer, where t_fixed is the non-layer cost (embed + output head + "
       "framework/sampling). (pilot p=4, q=8.)")
bullet("Each iteration records BOTH: (a) the MEASURED proxy wall time, and (b) the full-depth wall time "
       "derived ANALYTICALLY \u2014 t_layer(iter) = (t_proxy(iter) \u2212 t_fixed)/p, predicted full = "
       "t_fixed + N\u00b7t_layer(iter). Report proxy and full-scaled speed-up curves side by side "
       "(they differ because the fixed overhead does not scale).")
bullet("t_fixed is config-invariant across per-layer optimizations; recalibrate only if a NON-layer op "
       "(e.g. the output head) is optimized.")
bullet("END VALIDATION: after all ops are integrated, do ONE full-depth dummy run (all N layers) and "
       "compare the MEASURED full-depth wall time against the analytically predicted value \u2014 small error "
       "confirms the whole depth-scaled progression.")
h2("Framework / dispatch / OpenMP overhead \u2014 the proxy studies these too (Phase B on the proxy)")
para("The depth proxy is the right instrument for system overhead, not just op math: the depth-independent "
     "t_fixed term IS the framework/dispatch/sampling cost, and per-op dispatch + OpenMP-barrier overhead "
     "lives in t_layer and is incurred every layer \u2014 both are fully present at depth 4. Global OMP "
     "pathologies are depth-invariant, so the proxy reproduces them at a fraction of the cost.", GRAY)
num("Finish ALL ops/fused-ops to their roofline ceiling and integrate them into the proxy (end of Phase A); "
    "record the fully-integrated proxy + full-scaled wall time.", bold_lead="Order \u2014 ")
num("THEN attack framework/dispatch/OMP overhead on that integrated proxy (this is Phase B). Bit-exact "
    "gate: an overhead fix must not change tokens.", bold_lead="")
num("Show the wall-time improvement from the overhead work (proxy + full-scaled), SURFACE it, and only "
    "THEN launch the single full-depth run.", bold_lead="")
bullet("Bind threads ONCE at init \u2014 never per-forward torch.set_num_threads (it rebuilds the OpenMP pool "
       "and LOSES CPU affinity \u2192 unpinned contention inflates the whole forward uniformly; invisible per-op).")
bullet("Collapse many tiny torch ops into fewer/bigger parallel regions \u2014 at N threads each tiny op pays "
       "a barrier, so ~100 ops/layer is a ~10-20x dispatch tax; fuse into one parallel region.")
bullet("Clear the usual systemic levers: OMP spin-wait (OMP_WAIT_POLICY), allocator (tcmalloc/jemalloc), "
       "NUMA first-touch into the rank's own domain, thread-cap, CPU-freq/cold-start; remove dtype-cast and "
       "dispatch churn on the hot path. A global-config pathology masquerades as uniform per-op cost \u2014 "
       "clear it before trusting any per-op ranking.")
h2("MoE mode-collapse \u2014 the input tensor matters")
para("The dummy-weight run must exercise a REPRESENTATIVE set of routed experts. With tiny random gate "
     "weights, a non-diverse input (e.g. one hidden vector broadcast across the batch) collapses routing "
     "to the same top-k experts every token \u2014 only ~topk of the E experts run, so the measured MoE "
     "wall time is unrepresentatively low and the roofline's expert traffic is never incurred.", GRAY)
bullet("Build the input with plugin/validate/moe_balanced_input.build_verified_input(M, gate_weight): "
       "per-token i.i.d. hidden states (NOT a broadcast), falling back to round-robin gate-row alignment.")
bullet("VERIFY (not assume) dispersion: realized distinct-experts must track the uniform expectation "
       "E\u00b7(1\u2212(1\u22121/E)^(topk\u00b7M)) and no expert may absorb an outsized share \u2014 a collapse gate "
       "asserts this before the timing is trusted. (pilot E=256: checked M=8\u219243, 16\u219281, 32\u2192138, "
       "64\u2192209 distinct vs expected 43.8/80.2/135.3/199; the broadcast trap collapses to 6 and is rejected.)")
bullet("Instrument the real router during the E2E run to confirm the same spread in-engine (per-forward "
       "distinct-expert count), so the proxy input is validated against the actual model path.")

h2("Phase 6 — Validate, certify, deliver")
bullet("Validate-by-run: confirm the predicted class/ceiling in-engine (dummy weights for perf, full "
       "depth, target backend). Mismatch → the paper model omitted a cost; update it.")
bullet("Accuracy finale: real-weight per-token parity vs a trusted reference → task accuracy "
       "(e.g. gsm8k) → reference-vs-target cross-check sign-off.")
bullet("Provenance record (op→kernel→donor + parity gates) and a peer-relative check where an already-"
       "shipped model shares the same kernels (efficiency within tolerance); ship as a plugin override (no fork).")

# ---- correctness toolkit ----
h1("Correctness debugging toolkit (localize cheaply before real-weight runs)")
bullet("Deterministic-dummy weights: (a) skip the real checkpoint load for fast perf iteration; "
       "(b) make the target dummy bit-identical to the reference dummy so a per-layer diff is valid "
       "without real weights.")
bullet("Layer-by-layer cosine fingerprint: capture a per-layer reference, run the target with the same dummy "
       "weights, diff per-layer hidden states by cosine similarity; the first layer below ~0.999 localizes the break.")
bullet("Per-op fingerprint (shape + norm + cosine/checksum) to pinpoint the op within the bad layer.")
bullet("Isolation switches to bisect novel ops (e.g. bypass sparse selection → dense, force-dense MoE, swap "
       "kernel ↔ reference).")
bullet("Reduces-to-identity checks (e.g. sparse attend → exact dense at top-k=all) and low-bit dtype-bridge "
       "parity vs an independent dequant oracle.")
bullet("Escalation order: L0 coherence → per-layer cosine (dummy, no load) → per-op fingerprint + "
       "isolation → component/reduces-to-identity parity → real-weight per-token parity → task accuracy "
       "→ reference-vs-target sign-off.")

# ---- shape-capture audit ----
h1("Shape & semantics provenance — ground kernels in the PUBLISHED reference (human-auditable)")
para("A green parity microbench is SELF-CONSISTENT only: it validates op MATH at shapes the AUTHOR chose. "
     "Shape-generic ops (softmax-pool, flash-attention) pass cos=1.0 at ANY dims, so a hallucinated shape "
     "OR a missing semantic (an overlap window, an attention sink, MQA-vs-MHA) still passes. Ground every "
     "kernel's shapes AND semantics in the model author's PUBLISHED reference, not in author-chosen shapes "
     "and not in inference from the config alone.", bold=True)
h2("The method (model-agnostic, in order)")
num("PIN the published reference model definition (the author's own model.py / modeling_*.py) at a fixed "
    "revision. This is the architecture a human reviewer already trusts \u2014 the audit anchor.", bold_lead="1 \u2014 ")
num("Derive, per op, a CONTRACT from it: shapes as f(config, per-layer variant, phase, TP, batch) AND the "
    "SEMANTICS (fusions, gating, sinks, overlap windows, MQA/MHA, two-of-a-kind modules). Cite the source line.",
    bold_lead="2 \u2014 ")
num("BIND numeric constants from the real checkpoint config.json \u2014 the reference's defaults are often a "
    "TOY example (e.g. fewer layers/experts). CROSS-CHECK config vs reference defaults and reconcile any "
    "divergence; this also audits a regenerated/proxy config.", bold_lead="3 \u2014 ")
num("SWEEP the contract over the REAL variation axes \u2014 per-layer variants (e.g. per-layer compression "
    "ratios), prefill vs decode, TP, batch M \u2014 and assert each shape is VALID and EQUALS the kernel's "
    "contract (equality, not mere internal validity).", bold_lead="4 \u2014 ")
num("Port the published forward as the NUMERICAL ORACLE (exact parity, not a re-derived reference) so a "
    "SEMANTIC error, not just a shape error, is caught.", bold_lead="5 \u2014 ")
num("RUNTIME-CAPTURE (module-level now, full-forward at integration) to CONFIRM the serving stack's actual "
    "layout matches the contract \u2014 the serving path may repack/absorb/shard differently from the reference.",
    bold_lead="6 \u2014 ")
para("Why it survives human audit: every shape and semantic traces to a line in the PUBLISHED model + the "
     "real config; the sweep, validity and equality checks are mechanical; the oracle gives exact numbers. "
     "The reviewer checks FACTS and CITATIONS, not the agent's reading.", GRAY)
para("Anti-patterns (each is a silent-failure trap): trusting a parity microbench at author-chosen shapes; "
     "inferring dims from config alone (misses semantics \u2014 overlap, sink, MQA); treating the reference's "
     "toy defaults as the real constants; and the cardinal sin \u2014 RE-GUESSING a shape after finding one "
     "wrong instead of capturing it from the published model.", GRAY)

# ---- gates ----
h1("Gates")
for g in ["G0 — capacity / precision (every launch)",
          "G1 — coverage / lane (routes Thesis 1 vs 2 per op)",
          "Fusion roofline gate (bytes saved + AI lift + cache-resident; reference oracle where available)",
          "Shape-provenance gate (every kernel's shapes CAPTURED from the real forward + human-signed manifest, before trust/perf-claims)",
          "Phase-A ceiling-or-surface (every op at its ceiling, else surface to user)",
          "G2 — validate-by-run (confirm the paper model in-engine)",
          "G2.5 — system-pathology (kernel-domination ≥ bar before per-op work)",
          "Accuracy gate · Human-gate (novel kernels)"]:
    bullet(g)

# ---- platform card (example instantiation) ----
h1("Platform card — an INPUT, filled per target by the probe")
para("The method assumes NO specific values. Below is the pilot's instantiation (produced by the uArch "
     "probe); retarget by swapping the per-platform spec file. Every number here is measured, not assumed.", GRAY)
tbl = d.add_table(rows=1, cols=2)
tbl.style = "Light Grid Accent 1"
tbl.rows[0].cells[0].paragraphs[0].add_run("Property (pilot example)").bold = True
tbl.rows[0].cells[1].paragraphs[0].add_run("Value (DeepSeek-V4-Flash on Intel Xeon EMR, measured 2026-10-07)").bold = True
for k, v in [
    ("CPU", "Intel Xeon Platinum 8592+ (Emerald Rapids)"),
    ("Topology", "2 sockets × 64 cores; 1 NUMA domain per socket (SNC off)"),
    ("Per-domain RAM", "~503 GB (capacity floor)"),
    ("Memory BW (roofline ceiling)", "358.4 GB/s machine peak (8ch × DDR5-5600 × 8B); measured floor 214.4 GB/s = 60% of peak"),
    ("AMX bf16 (roofline ceiling)", "124.6 TFLOP/s machine peak (64c × 1024 FLOP/cyc × 1.9 GHz base); measured floor 40.3 TFLOP/s = 32% of peak"),
    ("Ridge (AI*)", "~348 FLOP/byte (machine peak)"),
    ("Optimization basis", "Optimize every op toward MACHINE PEAK; when levers are exhausted and it plateaus below peak, surface the residual gap and stop that op"),
    ("Implication", "Decode weight-streaming GEMMs are BW-bound across M∈{1,8,16,32,64}; only attention/sparse-attend are compute-bound"),
]:
    row = tbl.add_row().cells
    row[0].paragraphs[0].add_run(k)
    row[1].paragraphs[0].add_run(v)

# ---- method note ----
h1("Method note — all-ops-to-ceiling before wall-time ordering")
para("This playbook requires EVERY op at its roofline ceiling (Phase A) BEFORE wall-time / Amdahl "
     "ordering (Phase B), rather than ROI-skipping low-share ops up front. The inversion is deliberate: "
     "a cheap op can hide a systemic pathology, and completeness-first makes the Phase-B wall-time "
     "attribution trustworthy. Applies unchanged across models and platforms.", GRAY)

d.save(OUT)
print("SAVED:", OUT)
