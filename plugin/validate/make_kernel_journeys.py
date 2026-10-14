#!/usr/bin/env python3
"""Per-kernel OPTIMIZATION-JOURNEY bar chart for the kernel-pilot page: for each authored kernel, the
sequence of named optimizations and the latency after each, so the reader sees HOW the final speedup was
built lever-by-lever. Latencies are M=32 streaming (us), from the committed measured record. Data lives in
results/kernel_journeys.json so the chart stays maintainable (edit the JSON, re-run). Dependency-free SVG.

Usage: python make_kernel_journeys.py --out-dir ../../docs/plots
"""
import argparse
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Each kernel: stages = ordered [(lever label, M=32 streaming us)] (first=torch, last=shipped); floor = the
# OPERATIVE roofline ceiling us (the honest achievable floor for the kernel's regime, not the nominal byte/FLOP
# roof where that is the wrong ceiling); kind = short label for the floor line.
JOURNEYS = {
    "indexer logits": {"stages": [("torch", 909), ("brgemm + fused epi", 343), ("512-bit flags", 293), ("fused q-prep + adaptive tile", 212)], "floor": 110, "kind": "operative (GEMM/dispatch)"},
    "compressor": {"stages": [("torch", 328), ("stream softmax + C1", 284), ("hand-exp Cephes (D1)", 64)], "floor": 48, "kind": "DRAM-BW"},
    "sparse attend": {"stages": [("torch", 740), ("fp32 bmm + fused softmax", 541), ("hand-exp Cephes", 465)], "floor": 276, "kind": "fp32 compute"},
    "indexer top-k": {"stages": [("torch", 511), ("nth_element + chunked", 40)], "floor": 40, "kind": "selection"},
    "MHC sinkhorn": {"stages": [("torch", 616), ("fused recurrence", 26)], "floor": 26, "kind": "dispatch"},
    "MHC combine": {"stages": [("torch", 62), ("accumulate-once + AVX-512", 27)], "floor": 26, "kind": "operative-BW"},
}
SUBTITLE = ("M=32 streaming (LLC-flushed) latency after each named lever, 3-replicate median (EMR). First bar = "
            "torch baseline; bars descend toward the dashed ROOFLINE / operative-floor ceiling. Green × = speed-up "
            "vs torch; blue % = of the roofline (or 'at ceiling'). Lower is better.")


def esc(s):
    return html.escape(str(s))


def build_svg():
    W, H = 1180, 640
    padL, padR, padT, padB = 58, 20, 86, 150
    plotW, plotH = W - padL - padR, H - padT - padB
    y0 = padT + plotH
    lmin, lmax = 16.0, 1100.0                                   # log-y latency range (us)

    def ly(v):
        return y0 - (math.log10(v) - math.log10(lmin)) / (math.log10(lmax) - math.log10(lmin)) * plotH

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(f'<text x="{padL}" y="34" font-size="21" font-weight="700" fill="#0b2447">Optimization journey per kernel \u2014 how each lever built the final speed-up</text>')
    s.append(f'<text x="{padL}" y="58" font-size="13" fill="#5b6b80">{esc(SUBTITLE)}</text>')
    # y gridlines (log)
    for gv in (16, 32, 64, 128, 256, 512, 1024):
        yy = ly(gv)
        s.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-padR}" y2="{yy:.1f}" stroke="#eef2f7"/>')
        s.append(f'<text x="{padL-6}" y="{yy+4:.1f}" font-size="11" fill="#9db0c6" text-anchor="end">{gv}</text>')
    s.append(f'<text x="16" y="{padT+plotH/2:.0f}" font-size="12" fill="#5b6b80" transform="rotate(-90 16 {padT+plotH/2:.0f})" text-anchor="middle">latency \u00b5s (log)</text>')

    n = len(JOURNEYS)
    gap = 26
    gw = (plotW - gap * (n - 1)) / n
    pal = ["#d62728", "#e8873a", "#6a8ec9", "#1f77b4"]          # baseline -> ... -> final
    gx = padL
    for name, d in JOURNEYS.items():
        stages, floor = d["stages"], d["floor"]
        m = len(stages)
        bw = min(46, (gw - 10 * (m - 1)) / m)
        inner = m * bw + 10 * (m - 1)
        bx = gx + (gw - inner) / 2
        final = stages[-1][1]
        base = stages[0][1]
        for i, (lab, v) in enumerate(stages):
            col = pal[-1] if i == m - 1 else (pal[0] if i == 0 else pal[min(i, len(pal) - 2)])
            yy = ly(v)
            s.append(f'<rect x="{bx:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{y0-yy:.1f}" fill="{col}" rx="2"/>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{yy-5:.1f}" font-size="11" font-weight="600" fill="#31465e" text-anchor="middle">{v}</text>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{y0+8:.1f}" font-size="10" fill="#5b6b80" text-anchor="end" transform="rotate(-38 {bx+bw/2:.1f} {y0+8:.1f})">{esc(lab)}</text>')
            bx += bw + 10
        # roofline / operative-floor ceiling: dashed line the bars descend toward
        fy = ly(floor)
        s.append(f'<line x1="{gx+6:.1f}" y1="{fy:.1f}" x2="{gx+gw-6:.1f}" y2="{fy:.1f}" stroke="#8a3ffb" stroke-width="2" stroke-dasharray="5 3"/>')
        s.append(f'<text x="{gx+gw-6:.1f}" y="{fy-4:.1f}" font-size="9.5" fill="#8a3ffb" text-anchor="end">roofline {floor} \u00b7 {esc(d["kind"])}</text>')
        pct = 100.0 * floor / final
        roof_lbl = "at ceiling" if pct >= 80 else f"{pct:.0f}% of roof"
        s.append(f'<text x="{gx+gw/2:.1f}" y="{padT-10:.1f}" font-size="13" font-weight="700" fill="#0b2447" text-anchor="middle">{esc(name)}</text>')
        s.append(f'<text x="{gx+gw/2:.1f}" y="{padT+8:.1f}" font-size="12" font-weight="700" fill="#1d9b62" text-anchor="middle">{base/final:.1f}\u00d7  <tspan fill="#8a3ffb">{roof_lbl}</tspan></text>')
        gx += gw + gap
    s.append('</svg>')
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(JOURNEYS, open(os.path.join(HERE, "results", "kernel_journeys.json"), "w"), indent=1)
    with open(os.path.join(a.out_dir, "kernel_journeys.svg"), "w") as f:
        f.write(build_svg() + "\n")
    print("wrote kernel_journeys.svg + results/kernel_journeys.json")


if __name__ == "__main__":
    main()
