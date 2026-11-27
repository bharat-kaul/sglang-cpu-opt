"""Build DSv4-style GLM small multiples and a self-contained offline plot viewer."""
import json
import math
import subprocess
from statistics import median

from plot_glm5_overview import DATA, ROWS, records
from plot_glm5_journey import text

BASE, PRIOR, NATIVE, MODEL = "#b74638", "#bd861c", "#087e83", "#7946aa"
ANCHORS = {1, 2, 4, 8, 16, 32, 64}


def panels():
    wave = records(DATA / "roofline_cold.csv")
    measures = records(DATA / "measurements.csv")
    grid = records(DATA / "kda_grid_409001.csv")
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
                    for key, col in [("Native", "native_ns"), ("Model", "model_ns")]:
                        add(key, int(r["M"]), float(r[col]) / float(r["ref_ns"]))
        elif family in {"kda-prefill-chunk", "kda-decode-recurrence"}:
            mode = "fresh" if family == "kda-prefill-chunk" else "decode"
            detail = f"packed layer-0 caller · {mode}"
            for r in grid:
                if r["heat"] == "cold" and r["mode"] == mode:
                    p = json.loads(r["process_medians_ns"])
                    add("Native", int(r["m"]), median(x["native"] / x["frozen"] for x in p))
        else:
            panel, variants, detail = {
                "native-linear": ("406483 | fp8_mm_bf16 [4096, 12288]", {"native_checks": "Checks", "vector_tiles": "Tiles", "block64": "Native"}, "packed FP8 · 4096 × 12288"),
                "clamped-activation-quantization": ("406473 | width 2048", {"adaptive": "Native"}, "adaptive activation + quant · width 2048"),
                "kda-convolution-gates": ("406498 | decode_strided", {"selected": "Native"}, "strided decode caller · includes recurrence"),
                "sigmoid-gated-rmsnorm": ("406518 | gated_norm", {"serial": "Serial", "fused_serial": "Native"}, "serial → fused gated norm"),
            }[family]
            for r in measures:
                if r["cache"] == "cold" and r["panel"] == panel and r["series"] in variants:
                    add(variants[r["series"]], int(r["M"]), 1 / float(r["speedup"]))
        if not ANCHORS <= curves["Native"].keys() or any(p.keys() != curves["Native"].keys() for p in curves.values()):
            raise ValueError(f"Incomplete matched curves: {family}")
        yield label, detail, {"Base": dict.fromkeys(curves["Native"], 1.), **curves}


def figure(sweep):
    title = "Kernel latency across batch sizes" if sweep else "Kernel optimization journeys at M = 32"
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1152" height="1390" viewBox="0 0 1152 1390" font-family="DejaVu Sans,sans-serif">',
           '<rect width="100%" height="100%" fill="white"/>', text(24, 35, f"GLM-5.3-Flash   |   {title}", 25),
           text(24, 65, "Emerald Rapids, cache-flushed. Latency / matching CPU baseline (1×); lower is faster. Log scale per panel.", 17),
           text(24, 89, "Red: baseline     Gold: measured variant     Teal: native candidate     Purple dashed: provisional subgraph model, not a ceiling", 15)]
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
            svg += [f'<path d="M{x} {yy}h{w}" stroke="#dce2e7"/>', text(x-4, yy+4, f"{10**power:g}×", 11, "end")]
        keys = [k for k in curves if k != "Model"]
        for i, (k, p) in enumerate(curves.items()):
            color = {"Base": BASE, "Model": MODEL, "Native": NATIVE}.get(k, PRIOR)
            dash = ' stroke-dasharray="5 3"' if k == "Model" else ""
            if sweep:
                coords = [(x+w*math.log2(m)/6, pos(v)) for m, v in sorted(p.items())]
                points = " ".join(f"{a:.2f},{b:.2f}" for a,b in coords)
                svg.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2.3" fill="none"{dash}/>')
                if k != "Base":
                    svg.extend(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="1.8" fill="{color}"/>' for a,b in coords)
            elif k == "Model":
                svg.append(f'<path d="M{x} {pos(p[32])}h{w}" stroke="{color}" stroke-width="2"{dash}/>')
            else:
                xx = x + (keys.index(k)+.5)*w/len(keys)
                yy = pos(p[32])
                svg.extend([f'<rect x="{xx-21}" y="{yy}" width="42" height="{y+h-yy}" fill="{color}"/>',
                            text(xx, yy-3, f"{1/p[32]:.2f}×", 12, "middle"), text(xx, y+h+15, k, 12, "middle")])
        if sweep:
            svg.extend(text(x+w*math.log2(m)/6, y+h+15, f"{m}", 12, "middle") for m in sorted(ANCHORS))
    svg.append("</svg>")
    return "".join(svg)


if __name__ == "__main__":
    figures = []
    for sweep, tag in [(False, "journeys_m32"), (True, "journeys_msweep")]:
        svg = figure(sweep)
        subprocess.run(["rsvg-convert", "-o", str(DATA / f"{tag}_v1.png")], input=svg.encode(), check=True)
        figures.append(svg)
    (DATA / "journeys.html").write_text('<!doctype html><meta charset="utf-8"><title>GLM kernel journeys</title>'
        '<style>body{margin:auto;max-width:1152px}svg{width:100%;height:auto}</style>' + "\n".join(figures))
