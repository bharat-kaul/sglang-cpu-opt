"""Build DSv4-style GLM small multiples and a self-contained offline plot viewer."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

from plot_glm5_overview import DATA, ROWS, records
from plot_glm5_journey import text

BASE, NATIVE, MODEL = "#b74638", "#087e83", "#7946aa"
ANCHORS = {1, 2, 4, 8, 16, 32, 64}
PAIRS = DATA / "starting_baselines.csv"


def export_pairs(root):
    rows = []
    for family, folder, baseline, optimized in [
        ("native-linear", "native-linear-406483", "reference", "block64"),
        ("clamped-activation-quantization", "fused-activation-406473", "reference", "adaptive"),
        ("kda-convolution-gates", "kda-full-406498", "decode_strided", "decode_strided_selected"),
        ("sigmoid-gated-rmsnorm", "kda-norm-fusion-406518", "gated_norm", "gated_norm_fused_serial"),
    ]:
        raw = (root / folder / "summary.json").read_bytes()
        report = json.loads(raw)
        if str(report["job"]) != folder.split("-")[-1] or len({p["pid"] for p in report["processes"]}) != 3:
            raise ValueError(folder)
        if family == "clamped-activation-quantization":
            by_m = {}
            for p in report["processes"]:
                rep = (root / folder / p["file"]).read_bytes()
                if hashlib.sha256(rep).hexdigest() != p["sha256"]:
                    raise ValueError(p["file"])
                for c in json.loads(rep)["cases"]:
                    if c["variant"] == "emr_prebuild_reference" and c["I"] == 2048:
                        by_m.setdefault(c["M"], []).append(c["timing"]["cache_flushed"]["median_seconds"])
            for m, samples in by_m.items():
                if len(samples) != 3:
                    raise ValueError((folder, m))
                rows.append([family, report["job"], hashlib.sha256(raw).hexdigest(), m, "emr_prebuild_reference", "Base", *samples])
        for r in report["rows"]:
            variant = r.get("variant", r.get("op"))
            if variant not in (baseline, optimized):
                continue
            if family == "native-linear" and (r["op"] != "fp8_mm_bf16" or r["shape"][1:] != [4096, 12288]):
                continue
            if family == "clamped-activation-quantization":
                if r["I"] != 2048 or r["method"] != "cache_flushed":
                    continue
                samples = r["process_medians_seconds"]
            else:
                samples = r["timing"]["cache_flushed"]["process_medians_seconds"]
            if len(samples) != 3 or not all(math.isfinite(v) and v > 0 for v in samples):
                raise ValueError((folder, variant))
            m = r["shape"][0] if "shape" in r else r["M"]
            rows.append([family, report["job"], hashlib.sha256(raw).hexdigest(), m, variant,
                         "Base" if variant == baseline else "Native", *samples])
    with PAIRS.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family", "job", "summary_sha256", "M", "variant", "series", "rep0_s", "rep1_s", "rep2_s"])
        w.writerows(rows)


def panels():
    wave = records(DATA / "roofline_cold.csv")
    pairs = records(PAIRS)
    grid = records(DATA / "kda_grid_409001.csv")
    bw, = {float(r["triad_GBs"]) for r in wave}
    bf16, = {float(r["library_Gflops"]) for r in wave if r["dtype"] == "bf16"}
    for family, label, _, op in ROWS:
        curves = {}
        def add(key, m, v):
            points = curves.setdefault(key, {})
            if m in points or not 1 <= m <= 64 or not math.isfinite(v) or v <= 0:
                raise ValueError((family, key, m, v))
            points[m] = v
        if op:
            detail = f"{op} subgraph · original candidate"
            for r in wave:
                if r["op"] == op:
                    for key, col in [("Base", "ref_ns"), ("Native", "native_ns"), ("Model", "model_ns")]:
                        add(key, int(r["M"]), float(r[col]) / 1e6)
        elif family in {"kda-prefill-chunk", "kda-decode-recurrence"}:
            mode = "fresh" if family == "kda-prefill-chunk" else "decode"
            detail = f"packed layer-0 caller · {mode}"
            for r in grid:
                if r["heat"] == "cold" and r["mode"] == mode:
                    p = json.loads(r["process_medians_ns"])
                    add("Base", int(r["m"]), median(x["frozen"] for x in p) / 1e6)
                    add("Native", int(r["m"]), median(x["native"] for x in p) / 1e6)
        else:
            detail = {
                "native-linear": "packed FP8 multiply · 4096 × 12288",
                "clamped-activation-quantization": "frozen vs adaptive fusion · width 2048",
                "kda-convolution-gates": "strided decode + commit · selected dispatch",
                "sigmoid-gated-rmsnorm": "gated norm · frozen vs fused serial",
            }[family]
            for r in pairs:
                if r["family"] == family:
                    add(r["series"], int(r["M"]), 1000 * median(float(r[f"rep{i}_s"]) for i in range(3)))
        if not op:
            for m in curves["Native"]:
                flops = 0
                if family == "native-linear":
                    n, k = 4096, 12288
                    size = m*k + n*k + 4*(m*k//128 + n*k//128**2) + 2*m*n
                    flops = 2*m*n*k
                elif family == "clamped-activation-quantization":
                    size = m*(5*2048 + 4*(2048//128))
                elif family == "sigmoid-gated-rmsnorm":
                    size = 6*m*8192 + 2*128
                else:
                    # FP32 temporal pool read/write and BF16 three-tap convolution history.
                    state = 2*(4*64*128**2 + 2*3*24576)
                    if family == "kda-convolution-gates":
                        size = m*state
                    else:
                        weights = 4096*(24576+64+2*128) + 2*128*8192 + 8192*4096
                        size = 2*weights + (m*state if mode == "decode" else state//2)
                        flops = 2*m*weights
                add("Model", m, max(size/bw, flops/bf16)/1e6)
        if set(curves) != {"Base","Native","Model"} or not ANCHORS <= curves["Native"].keys() or any(p.keys() != curves["Native"].keys() for p in curves.values()):
            raise ValueError(f"Incomplete matched curves: {family}")
        yield label, detail, {k: curves[k] for k in ("Base", "Native", "Model") if k in curves}


def figure(sweep):
    title = "Kernel latency across batch sizes" if sweep else "Kernel optimization journeys at M = 32"
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1152" height="1390" viewBox="0 0 1152 1390" font-family="DejaVu Sans,sans-serif">',
           '<rect width="100%" height="100%" fill="white"/>', text(24, 35, f"GLM-5.3-Flash   |   {title}", 25),
           text(24, 65, "Cache-flushed EMR latency in milliseconds (log scale; lower is faster). Each panel compares the same work.", 17),
           text(24, 89, "Red: measured starting CPU path     Teal: measured optimized path     Purple dashed: modeled streaming floor (not a measured ceiling)", 15),
           text(24, 110, "Speedup = baseline / optimized. Model gap = optimized / modeled floor; it is not a guaranteed remaining speedup.", 14)]
    for n, (label, detail, curves) in enumerate(panels()):
        x, y = 62 + n % 3 * 380, 169 + n // 3 * 209
        w, h = 291, 134
        svg += [text(x-45, y-39, label, 17), text(x-45, y-19, detail, 13)]
        vals = [v for p in curves.values() for m, v in p.items() if sweep or m == 32]
        lo = math.floor(math.log10(min(vals)) * 2) / 2 - .1
        hi = math.ceil(math.log10(max(vals)) * 2) / 2 + .1
        def pos(v):
            return y + h * (hi - math.log10(v)) / (hi - lo)
        for power in range(math.ceil(lo), math.floor(hi)+1):
            yy = pos(10**power)
            svg += [f'<path d="M{x} {yy}h{w}" stroke="#dce2e7"/>', text(x-4, yy+4, f"{10**power:g}", 11, "end")]
        keys = [k for k in curves if k != "Model"]
        for i, (k, p) in enumerate(curves.items()):
            color = {"Base": BASE, "Model": MODEL, "Native": NATIVE}[k]
            dash = ' stroke-dasharray="5 3"' if k == "Model" else ""
            if sweep:
                coords = [(x+w*math.log2(m)/6, pos(v)) for m, v in sorted(p.items())]
                points = " ".join(f"{a:.2f},{b:.2f}" for a,b in coords)
                svg.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2.3" fill="none"{dash}/>')
                if k != "Base":
                    svg.extend(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="1.8" fill="{color}"/>' for a,b in coords)
            elif k == "Model":
                svg.append(f'<path d="M{x} {pos(p[32])}h{w}" stroke="{color}" stroke-width="2"{dash}/>')
                svg.append(text(x+w, pos(p[32])-4, f"{p[32]:.3g} ms · gap {curves['Native'][32]/p[32]:.1f}×", 11, "end"))
            else:
                xx = x + (keys.index(k)+.5)*w/len(keys)
                yy = pos(p[32])
                caption = "Starting baseline" if k == "Base" else "Optimized"
                value = f"{p[32]:.3g} ms"
                if k == "Native" and "Base" in curves:
                    value += f"  ({curves['Base'][32]/p[32]:.2f}×)"
                svg.extend([f'<rect x="{xx-21}" y="{yy}" width="42" height="{y+h-yy}" fill="{color}"/>',
                            text(xx, yy-3, value, 12, "middle"), text(xx, y+h+15, caption, 12, "middle")])
        if sweep:
            svg.extend(text(x+w*math.log2(m)/6, y+h+15, f"{m}", 12, "middle") for m in sorted(ANCHORS))
    svg.append("</svg>")
    return "".join(svg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        export_pairs(Path(sys.argv[1]))
    figures = []
    for sweep, tag in [(False, "journeys_m32"), (True, "journeys_msweep")]:
        svg = figure(sweep)
        subprocess.run(["rsvg-convert", "-o", str(DATA / f"{tag}_v4.png")], input=svg.encode(), check=True)
        figures.append(svg)
    (DATA / "journeys.html").write_text('<!doctype html><meta charset="utf-8"><title>GLM kernel journeys</title>'
        '<style>body{margin:auto;max-width:1152px}svg{width:100%;height:auto}</style>' + "\n".join(figures))
