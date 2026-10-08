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
r = t.add_run("CPU Model Enablement + Optimization Playbook")
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = NAVY
s = d.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s.add_run("Scale-stratified methodology — DeepSeek-V4-Flash tree pilot (EMR)")
r.font.size = Pt(12)
r.font.color.rgb = GRAY
dt = d.add_paragraph()
dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = dt.add_run("Reference build — 2026-10-07")
r.font.size = Pt(9)
r.font.color.rgb = GRAY

# ---- overview ----
h1("Overview")
para("Goal: from a bare model config + checkpoint, run CPU enablement AND optimization end-to-end "
     "for any model, with final wall-time driven toward the sum of kernel roofline times — framework "
     "overhead squeezed out and every kept kernel at its ceiling.", GRAY)
para("Two theses, one plugin (SGLang external model package, override by __name__, no fork):", bold=True)
bullet("models whose ops are fully covered by existing AMX/CPU kernels are enabled by wiring + "
       "validation only (capability inheritance: bf16/int8, AMX, prepack/VNNI, FusedMoE).",
       bold_lead="Thesis 1 — ")
bullet("genuinely novel ops get an AI-written, roofline-tuned C/C++ kernel, human-gated.",
       bold_lead="Thesis 2 — ")
para("Scope-discovery (coverage-gate) is the router; the spine-leaf performance tree "
     "(macro/meso/micro roofline loop) is the engine. One pipeline.", GRAY)

# ---- governing two-phase ----
h1("Governing principle — two phases")
para("Per-op completeness FIRST, wall-time/ordering SECOND (a deliberate inversion of ROI-skip).", bold=True)
h2("Phase A — per-op peaking (ALL ops, including fused ops)")
num("Discover every op in the layer, including fused ops — a fused op is a first-class unit with its own roofline.")
num("Per op/fused-op: roofline + binding resource (BW / compute / latency) on the target platform (measured constants).")
num("Arithmetic-intensity improvability: can the binding resource be moved by a lever (increase M/batch, fusion)? Record the achievable ceiling.")
num("Implement the op in C/C++ and optimize to a reasonable extent, driving it to its ceiling (BW peak / compute peak / latency floor).")
num("If the gap to roofline cannot be bridged by known techniques → surface to the user for review.")
num("Every op done this way — none skipped. Completion gate = every op/fused-op at its ceiling.")
h2("Phase B — wall-time + ordering (only after Phase A completes)")
num("Switch to wall-time; fix system pathology first (OpenMP spin-wait, thread-cap, NUMA bind, allocator, dispatch, batching) until kernel-domination clears the bar.")
num("Apply Amdahl ordering — prioritize/aggregate high-share work. Ordering is sequencing, not exclusion.")
para("Bounding valves that keep Phase A finite: \u201creasonable extent\u201d + \u201csurface if can\u2019t bridge.\u201d "
     "Latency-bound \u201cpeak\u201d = the dispatch/overhead floor (fuse/batch launches), not an AI target.", GRAY)

# ---- pipeline phases ----
h1("Pipeline phases (in order)")

h2("Phase 0 — Platform scan + intake + capacity/precision pre-flight (Gate 0)")
bullet("Run the uArch probe on the target node → measured machine_constants: per-NUMA-domain memory "
       "BW, per-dtype compute peak, ridge = peak/BW, NUMA/SNC, ISA. No hardcoded constants.")
bullet("Data-type audit from the real checkpoint (metadata-only first): {op, stored dtype, target "
       "compute dtype, kernel, inflation?, dtype-bridge, parity gate}. Never inflate a native low-bit "
       "checkpoint; sub-tile dtypes are moved low-bit and computed bf16 via fused dequant.")
bullet("Capacity pre-flight: resident weights + activations + KV/state vs per-domain RAM floor → tp/EP "
       "(EP-first for MoE; CPU tp>1 hurts latency-bound decode).")

h2("Phase 1 — Op scope discovery + decomposition (Gate 1)")
bullet("Enumerate every op (walk each layer index — heterogeneity is easy to miss); 100% compute "
       "coverage; anything unclassifiable is UNKNOWN → treated as a gap.")
bullet("Dependency-closure for non-compute substrate (KV pool, backends, novel non-attention families).")
bullet("Coverage-gate assigns each op a lane: A = covered on CPU (donor); B = missing on CPU but "
       "GPU/XPU reference exists (oracle); C = missing, author from math spec.")

h2("Phase 2 — Fusion pass → roofline (per op, target platform)")
bullet("Fusion analysis first (vertical / horizontal / epilogue). For each candidate: consult the "
       "GPU/XPU implementation where available — runnable fused code is a numerical ORACLE; a "
       "performance claim without code is a HINT only — OR derive the benefit MATHEMATICALLY on the "
       "roofline (bytes saved + AI before→after + L2-resident gate) where no reference exists. Absence "
       "of a GPU reference is not a reason to skip (CPU is more BW-bound, so a CPU-only fusion may pay).")
bullet("Map each fusion: COVERED (donor fused kernel) / NEW-C++ (→ kernel authoring) / SKIP (record why).")
bullet("Per-op and per-fused-op roofline: FLOPs, weight+activation+KV bytes, arithmetic intensity, "
       "binding resource, achievable ceiling, and the C/C++ implementation lane. Phase-split prefill vs decode.")

h2("Phase A / Phase B — execute per the governing principle above")

h2("Phase 6 — Validate, certify, deliver")
bullet("Validate-by-run (Gate 2): confirm the predicted class/ceiling in-engine (dummy weights for "
       "perf, full depth, target ISA). Mismatch → the paper model omitted a cost; update it.")
bullet("Accuracy oracle finale: real-weight per-token parity vs HF → task accuracy (gsm8k) → "
       "GPU-vs-CPU cross-check sign-off.")
bullet("Enablement certificate (op→kernel→donor provenance + parity gates); peer-relative roofline "
       "(efficiency vs a shipped donor ≥ 0.90); ship as plugin override (no fork).")

# ---- correctness toolkit ----
h1("Correctness debugging toolkit (localize cheaply before real-weight runs)")
bullet("Deterministic-dummy weights: (a) skip the real checkpoint load for fast perf iteration; "
       "(b) make CPU dummy bit-identical to the GPU/torch dummy reference so a per-layer diff is valid "
       "without real weights.")
bullet("Layer-by-layer cosine fingerprint: capture a per-layer reference, run CPU with the same dummy "
       "weights, diff per-layer hidden states by cosine similarity; the first layer below ~0.999 localizes the break.")
bullet("Per-op fingerprint (shape + norm + cosine/checksum) to pinpoint the op within the bad layer.")
bullet("Isolation switches to bisect novel ops (ignore DSA selection → dense, force-dense MoE, swap "
       "kernel ↔ torch reference).")
bullet("Reduces-to-identity checks (sparse attend → exact dense at top-k=all) and low-bit dtype-bridge "
       "parity vs an independent torch dequant oracle.")
bullet("Escalation order: L0 coherence → per-layer cosine (dummy, no load) → per-op fingerprint + "
       "isolation → component/reduces-to-identity parity → real-weight per-token parity → task accuracy "
       "→ GPU-vs-CPU sign-off.")

# ---- gates ----
h1("Gates")
for g in ["G0 — capacity / precision (every launch)",
          "G1 — coverage / lane (routes Thesis 1 vs 2 per op)",
          "Fusion roofline gate (bytes saved + AI lift + L2-resident; GPU oracle where available)",
          "Phase-A ceiling-or-surface (every op at its ceiling, else surface to user)",
          "G2 — validate-by-run (confirm the paper model in-engine)",
          "G2.5 — system-pathology (kernel-domination ≥ bar before per-op work)",
          "Accuracy-oracle gate · Human-gate (novel kernels)"]:
    bullet(g)

# ---- EMR platform card ----
h1("Pilot platform — EMR (measured)")
tbl = d.add_table(rows=1, cols=2)
tbl.style = "Light Grid Accent 1"
tbl.rows[0].cells[0].paragraphs[0].add_run("Property").bold = True
tbl.rows[0].cells[1].paragraphs[0].add_run("Value (measured 2026-10-07, pcl-spr10)").bold = True
for k, v in [
    ("CPU", "Intel Xeon Platinum 8592+ (Emerald Rapids)"),
    ("Topology", "2 sockets × 64 cores; 1 NUMA domain per socket (SNC off)"),
    ("Per-domain RAM", "~503 GB (capacity floor)"),
    ("Memory BW (roofline ceiling)", "358.4 GB/s machine peak (8ch × DDR5-5600 × 8B); measured floor 214.4 GB/s = 60% of peak"),
    ("AMX bf16 (roofline ceiling)", "190.1 TFLOP/s machine peak (64c × 1024 FLOP/cyc × 2.9 GHz); measured floor 40.3 TFLOP/s = 21% of peak"),
    ("Ridge (AI*)", "~530 FLOP/byte (machine peak)"),
    ("Optimization basis", "Optimize every op toward MACHINE PEAK; when levers are exhausted and it plateaus below peak, surface the residual gap and stop that op"),
    ("Implication", "Decode weight-streaming GEMMs are BW-bound across M∈{1,8,16,32,64}; only attention/sparse-attend are compute-bound"),
]:
    row = tbl.add_row().cells
    row[0].paragraphs[0].add_run(k)
    row[1].paragraphs[0].add_run(v)

# ---- note on divergence ----
h1("Note — divergence from the published tree")
para("The codified tree ROI-skips low-share ops (\u201cdominant kernel at roofline\u201d). This pilot "
     "requires all ops at their ceiling (Phase A) before wall-time/Amdahl ordering (Phase B). This is "
     "reported to the tree owner as a methodology item; the pilot does not edit the tree.", GRAY)

d.save(OUT)
print("SAVED:", OUT)
