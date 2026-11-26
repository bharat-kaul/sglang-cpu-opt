"""Render the family overview from published data; optionally refresh caller rows."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

from plot_glm5_journey import DATA, text

# Family, display name, reviewed caller screen, historical modeled subgraph (if any).
ROWS = [
    ("native-linear", "Packed linear", "407926:fp8", None),
    ("clamped-activation-quantization", "Activation + FP8 quantization", "406473:adaptive", None),
    ("moe-router", "MoE router (retained donor)", "407952:donor", "router"),
    ("expert-group-gather-combine", "Expert gather + combine", "expert:moe", "combine"),
    ("kda-convolution-gates", "KDA convolution + gates", "406498:decode", None),
    ("kda-decode-recurrence", "KDA decode recurrence", "406498:decode", None),
    ("kda-prefill-chunk", "KDA prefill", "407968:extend_direct", None),
    ("sigmoid-gated-rmsnorm", "Sigmoid-gated RMSNorm", "406518:norm", None),
    ("mhc-pre-post", "mHC pre / post", "407897:pre", "mixed_norm"),
    ("mla-absorption-expansion", "MLA absorption / expansion", "407924:absorb", "absorb"),
    ("index-projection-norm-rotation", "Indexer projection + rotation", "407945:index", "rotate"),
    ("pooled-compression-cache-write", "Pooled compression + write", "407934:pool", "compress"),
    ("pooled-score-select-tail", "Pooled scoring + selection", "407955:mapped_selection", "tail"),
    ("latent-kv-pack-read", "Latent KV pack / read", "407602:read", "pack"),
    ("selected-latent-attention", "Selected latent attention", "407971:extend_reference", "attend"),
    ("norm-embedding-final-contraction", "Embedding / final head", "407644:selection_grid", "contract"),
    ("state-pool-metadata", "State pool + metadata", "407644:runtime_prepare", "allocate"),
]
CALLERS = DATA / "overview_callers.csv"
WAVE = DATA / "joined_screens_408917.csv"


def records(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def one(rows):
    row, = rows
    return row


def export(root):
    screens = records(root / "operative-wave/phase1-review/caller-screens.csv")
    router = json.loads((root / "dense-wave/router-full-407951-407952.json").read_text())
    output = []
    for family, _, screen, _ in ROWS:
        if family == "moe-router":
            r = one(r for r in router["407952"]["results"] if r["M"] == 8 and r["cold"])
            p = r["process_medians_ns"]
            values = ["M8:cold", "407952", p["source"], p["donor"]]
        else:
            r = one(r for r in screens if (r["family"], r["screen"]) == (family, screen))
            values = [r["coordinate"], r["source"], json.loads(r["reference_ns"]),
                      json.loads(r["measured_ns"])]
        output.append([family, screen, *values[:2], *map(json.dumps, values[2:])])
    with CALLERS.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family", "screen", "coordinate", "source", "baseline_process_ns", "native_process_ns"])
        w.writerows(output)


def render():
    callers = records(CALLERS)
    models = [r for r in records(DATA / "roofline_cold.csv") if r["M"] == "8"]
    if len(callers) != len(ROWS) or len(models) != sum(bool(op) for *_, op in ROWS):
        raise ValueError("unexpected overview coverage")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1100" font-family="DejaVu Sans">',
           '<rect width="100%" height="100%" fill="white"/>',
           text(20, 33, "GLM-5.3-Flash  |  Emerald Rapids CPU kernel performance", 25),
           text(20, 61, "17 families   •   Latency in microseconds (lower is better)   •   Evidence through job 408997", 17),
           text(20, 106, "Kernel family", 17), text(315, 106, "Measured caller", 18),
           text(315, 131, "Baseline → optimized path", 14),
           text(640, 106, "Roofline comparison: isolated subgraph", 18),
           text(640, 131, "Cold, M = 8   •   Earlier native implementation", 14)]
    for x, label in [(417, "Baseline"), (526, "Native"), (808, "Baseline"), (908, "Native"), (1090, "Model*")]:
        svg.append(text(x, 162, label, 15, "end"))
    svg.extend([text(640, 162, "Subgraph", 15), text(550, 162, "Speedup", 15)])

    def number(ns):
        us = float(ns) / 1000
        if not math.isfinite(us) or us <= 0:
            raise ValueError("invalid latency")
        return f"{us:,.0f}" if us >= 100 else f"{us:.3g}"

    for i, (family, name, screen, op) in enumerate(ROWS):
        y = 193 + 29 * i
        if i % 2 == 0:
            svg.append(f'<rect x="10" y="{y-20}" width="1100" height="28" fill="#eef4f7"/>')
        svg.append(text(20, y, name, 16))
        caller = one(r for r in callers if (r["family"], r["screen"]) == (family, screen))
        times = []
        for key, x in [("baseline_process_ns", 417), ("native_process_ns", 526)]:
            samples = json.loads(caller[key])
            if len(samples) != 3 or not all(math.isfinite(v) and v > 0 for v in samples):
                raise ValueError(f"{family}: expected three valid process medians")
            times.append(median(samples))
            svg.append(text(x, y, number(times[-1]), 16, "end"))
        ratio = times[0] / times[1]
        width = 24 * abs(math.log2(ratio))
        start = 552 if ratio >= 1 else 552 - width
        svg.extend([f'<rect x="{start}" y="{y-1}" width="{width}" height="5" fill="#087e83"/>',
                    text(552, y-4, f"{ratio:.2f}×", 12)])
        if op:
            model = one(r for r in models if r["op"] == op)
            svg.append(text(640, y, op.replace("mixed_norm", "mix/norm"), 14))
            for key, x in [("ref_ns", 808), ("native_ns", 908), ("model_ns", 1090)]:
                svg.append(text(x, y, number(model[key]), 16, "end"))
        else:
            svg.append(text(640, y, "—  no matched subgraph model", 14))
    svg += [text(20, 704, "* Provisional time for counted work only: compute + bytes for three subgraphs; bytes only for eight.", 16),
            text(20, 731, "Compare within a panel, not across panels. No measured whole-family ceiling is available for any row.", 16),
            text(20, 776, "New caller measurements vs frozen CPU   |   warm and cold", 20),
            text(550, 807, "Comparisons", 15), text(725, 807, "Speedup range", 15), text(974, 807, "Below parity", 15)]
    wave = records(WAVE) + records(DATA / "qkv_holdout_408994.csv")
    for i, (job, label, base, paths, count) in enumerate([
        ("408614", "MLA split normalization only", "source", {"native"}, 128),
        ("408615", "Mixed-request allocator preparation", "ref", {"native"}, 12),
        ("408617", "Layer 3 attention + cache lifecycle", "reference", {"candidate"}, 16),
        ("408621", "Layer 0 KDA: packed projections through output", "frozen", {"native"}, 26),
        ("408624", "mHC, QKV and FFN handoffs", "source", {"native", "fused"}, 160),
        ("408903", "MLA mapped decode + ring, M1–64 and controls", "reference", {"candidate"}, 142),
        ("408911", "KDA fresh extend, 60 new batch sizes", "frozen", {"native"}, 120),
        ("408912", "KDA dense decode, M1–64", "frozen", {"native"}, 128),
        ("408994", "Attention preparation + QKV, fixed path M1–64", "source", {"fixed"}, 128),
    ]):
        batch = [r for r in wave if r["job"] == job and r["numerator"] == base and r["denominator"] in paths]
        ratios = [float(r["median_paired_ratio"]) for r in batch]
        if (len(ratios) != count or len({(r["coordinate"], r["cache"], r["denominator"]) for r in batch}) != count
                or not all(math.isfinite(x) and x > 0 for x in ratios)):
            raise ValueError(f"{job}: incomplete or invalid comparison")
        y = 838 + 28 * i
        svg.extend([text(20, y, f"{job}   {label}", 16), text(600, y, count, 16, "middle"),
                    text(725, y, f"{min(ratios):.3f}–{max(ratios):.3f}×", 16),
                    text(1015, y, sum(x < 1 for x in ratios), 16, "middle")])
    svg.append("</svg>")
    subprocess.run(["rsvg-convert", "-o", str(DATA / "kernel_overview_v4.png")],
                   input="".join(svg).encode(), check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        export(Path(sys.argv[1]))
    render()
