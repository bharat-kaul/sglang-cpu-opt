#!/usr/bin/env python3
"""Across-M optimization-journey chart (small multiples): for each authored kernel, the baseline (torch) vs
shipped latency across the FULL M=1..64 sweep, with the roofline floor overlaid where that is the right
ceiling. The M=32 journey bar chart (make_kernel_journeys.py) shows HOW the speed-up was built lever-by-lever
at one operating point; this shows HOW IT HOLDS across batch M -- several levers are M-dependent (adaptive
tile, serial-dispatch low-M, ISA crossover), so the single-M view hides their batch sensitivity.

Data: results/operating_point.json (committed measured record; per (kernel,M): torch_stream_us, cpp_stream_us,
floor_us, pct_roof_stream). Dependency-free SVG.

Usage: python make_kernel_journeys_msweep.py --out-dir ../../docs/plots
"""
import argparse
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# json kernel name -> (display label, show nominal roofline floor?). Floor shown only for the compute/BW-bound
# kernels where the byte/FLOP roof is the right ceiling; the dispatch/selection-bound kernels (floor~0) are
# labelled instead -- their operative ceiling is dispatch, not the nominal roof.
KERNELS = {
    "indexer_logits": ("indexer logits", True, "GEMM roofline"),
    "compressor_R128": ("compressor", True, "DRAM-BW roofline"),
    "sparse_bestof": ("sparse attend", True, "fp32-compute roofline"),
    "indexer_topk": ("indexer top-k", False, "selection-bound"),
    "sinkhorn": ("MHC sinkhorn", False, "dispatch-bound"),
    "combine": ("MHC combine", False, "operative-BW bound"),
}
MS = [1, 2, 4, 8, 16, 32, 64]
SUBTITLE = ("Baseline (torch) vs shipped kernel across the M=1..64 sweep, streaming (LLC-flushed), 3-replicate "
            "median (EMR). Gap between the lines = speed-up; it MOVES with M (adaptive tile, low-M serial "
            "dispatch, ISA crossover). Dashed = nominal roofline floor where that is the right ceiling. Log-log; "
            "lower is better.")


def esc(s):
    return html.escape(str(s))


def build_svg(data):
    cols, rows = 3, 2
    W = 1180
    panelW, panelH = 360, 232
    padL_panel, padR_panel, padT_panel, padB_panel = 52, 14, 48, 42
    gapX, gapY = 22, 56
    gridL, gridT = 26, 112
    H = gridT + rows * panelH + (rows - 1) * gapY + 24

    lmin, lmax = 8.0, 4400.0  # shared log-y latency range (us)
    xmin, xmax = math.log2(1), math.log2(64)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(f'<text x="{gridL}" y="34" font-size="21" font-weight="700" fill="#0b2447">Optimization journey across the M sweep \u2014 how the speed-up holds vs batch</text>')
    # wrap subtitle
    words = SUBTITLE.split()
    line, yy = "", 56
    for w in words:
        if len(line) + len(w) > 150:
            s.append(f'<text x="{gridL}" y="{yy}" font-size="13" fill="#5b6b80">{esc(line.strip())}</text>')
            line, yy = "", yy + 17
        line += w + " "
    s.append(f'<text x="{gridL}" y="{yy}" font-size="13" fill="#5b6b80">{esc(line.strip())}</text>')
    # legend
    lx = gridL
    s.append(f'<line x1="{lx}" y1="98" x2="{lx+22}" y2="98" stroke="#d1495b" stroke-width="2.5"/><text x="{lx+28}" y="102" font-size="12" fill="#444">torch baseline</text>')
    s.append(f'<line x1="{lx+150}" y1="98" x2="{lx+172}" y2="98" stroke="#1d7874" stroke-width="2.5"/><text x="{lx+178}" y="102" font-size="12" fill="#444">shipped kernel</text>')
    s.append(f'<line x1="{lx+300}" y1="98" x2="{lx+322}" y2="98" stroke="#6a4c93" stroke-width="2" stroke-dasharray="5 4"/><text x="{lx+328}" y="102" font-size="12" fill="#444">nominal roofline floor</text>')

    for i, (kname, (label, show_floor, kind)) in enumerate(KERNELS.items()):
        km = data["kernels"][kname]["M"]
        r, c = divmod(i, cols)
        px = gridL + c * (panelW + gapX)
        py = gridT + r * (panelH + gapY)
        ax0 = px + padL_panel
        ay0 = py + padT_panel
        aw = panelW - padL_panel - padR_panel
        ah = panelH - padT_panel - padB_panel
        ay1 = ay0 + ah

        def X(m):
            return ax0 + (math.log2(m) - xmin) / (xmax - xmin) * aw

        def Y(v):
            v = max(v, lmin)
            return ay1 - (math.log10(v) - math.log10(lmin)) / (math.log10(lmax) - math.log10(lmin)) * ah

        # panel frame + y gridlines
        s.append(f'<rect x="{ax0}" y="{ay0}" width="{aw}" height="{ah}" fill="#fbfcfe" stroke="#e6ecf3"/>')
        for gv in (10, 100, 1000):
            gy = Y(gv)
            if ay0 - 1 <= gy <= ay1 + 1:
                s.append(f'<line x1="{ax0}" y1="{gy:.1f}" x2="{ax0+aw}" y2="{gy:.1f}" stroke="#eef2f7"/>')
                s.append(f'<text x="{ax0-5}" y="{gy+4:.1f}" font-size="10" fill="#9db0c6" text-anchor="end">{gv}</text>')
        # x ticks
        for m in MS:
            xx = X(m)
            s.append(f'<text x="{xx:.1f}" y="{ay1+16:.1f}" font-size="10" fill="#9db0c6" text-anchor="middle">{m}</text>')
        s.append(f'<text x="{ax0+aw/2:.1f}" y="{ay1+33:.1f}" font-size="11" fill="#5b6b80" text-anchor="middle">batch M (tokens, log\u2082)</text>')

        torch = [km[str(m)]["torch_stream_us"] for m in MS]
        cpp = [km[str(m)]["cpp_stream_us"] for m in MS]
        floor = [km[str(m)]["floor_us"] for m in MS]
        sp = [t / c for t, c in zip(torch, cpp)]

        # floor line (nominal roofline) where it is the right ceiling
        if show_floor:
            pts = " ".join(f"{X(m):.1f},{Y(f):.1f}" for m, f in zip(MS, floor))
            s.append(f'<polyline points="{pts}" fill="none" stroke="#6a4c93" stroke-width="1.8" stroke-dasharray="5 4" opacity="0.9"/>')
        # baseline + shipped
        tp = " ".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(MS, torch))
        cp = " ".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(MS, cpp))
        s.append(f'<polyline points="{tp}" fill="none" stroke="#d1495b" stroke-width="2.4"/>')
        s.append(f'<polyline points="{cp}" fill="none" stroke="#1d7874" stroke-width="2.4"/>')
        for m, v in zip(MS, torch):
            s.append(f'<circle cx="{X(m):.1f}" cy="{Y(v):.1f}" r="2.4" fill="#d1495b"/>')
        for m, v in zip(MS, cpp):
            s.append(f'<circle cx="{X(m):.1f}" cy="{Y(v):.1f}" r="2.4" fill="#1d7874"/>')

        # title + speed-up range across M, stacked two lines to avoid overlap on narrow panels
        s.append(f'<text x="{ax0}" y="{ay0-27:.1f}" font-size="14" font-weight="700" fill="#0b2447">{esc(label)}</text>')
        s.append(f'<text x="{ax0}" y="{ay0-10:.1f}" font-size="11.5" fill="#1d7874">{min(sp):.1f}\u2013{max(sp):.1f}\u00d7 across M \u00b7 {esc(kind)}</text>')

    s.append('</svg>')
    return "\n".join(s), {kname: {"M": MS,
                                  "torch_us": [round(data["kernels"][kname]["M"][str(m)]["torch_stream_us"], 1) for m in MS],
                                  "shipped_us": [round(data["kernels"][kname]["M"][str(m)]["cpp_stream_us"], 1) for m in MS],
                                  "speedup": [round(data["kernels"][kname]["M"][str(m)]["torch_stream_us"] / data["kernels"][kname]["M"][str(m)]["cpp_stream_us"], 2) for m in MS]}
                          for kname in KERNELS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    ap.add_argument("--op-json", default=os.path.join(HERE, "results", "operating_point.json"))
    args = ap.parse_args()
    with open(args.op_json) as f:
        data = json.load(f)
    svg, summary = build_svg(data)
    os.makedirs(args.out_dir, exist_ok=True)
    out_svg = os.path.join(args.out_dir, "kernel_journeys_msweep.svg")
    with open(out_svg, "w") as f:
        f.write(svg)
    out_json = os.path.join(HERE, "results", "kernel_journeys_msweep.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_svg)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
