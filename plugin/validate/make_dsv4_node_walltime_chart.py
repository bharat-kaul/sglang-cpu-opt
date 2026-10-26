#!/usr/bin/env python3
"""DeepSeek-V4-Flash CPU wall-time: baseline vs optimized vs roofline, EMR + GNR.

Two panels (prefill compute-bound, decode memory-bound). Each: bars for EMR baseline / EMR optimized /
GNR optimized (GNR a 'pending' ghost until the node frees) + a dashed roofline ceiling (compute-peak for
prefill, weight-streaming BW for decode). Data-driven from results/dsv4_node_walltime.json. SVG -> PNG
via rsvg-convert (matches the other chart generators).

Usage: python make_dsv4_node_walltime_chart.py --out-dir ../../docs/plots
"""
import argparse
import html
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"bad": "#c0504d", "good": "#2f6f4f", "pending": "#c9d3df"}


def esc(s):
    return html.escape(str(s))


def panel_svg(p, x0, y0, w, h):
    s = []
    bars = p["bars"]
    roof = p["roofline"]
    ymax = roof * 1.08
    plotH = h - 64  # leave room for phase title + x labels
    top = y0 + 30

    def Y(v):
        return top + plotH - (v / ymax) * plotH

    s.append(f'<text x="{x0}" y="{y0+16}" font-size="15" font-weight="700" fill="#0b2447">{esc(p["phase"])}</text>')
    # gridlines
    import math
    step = 200 if roof > 600 else 100
    gv = 0
    while gv <= ymax:
        yy = Y(gv)
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+w}" y2="{yy:.1f}" stroke="#eef2f7"/>')
        s.append(f'<text x="{x0-6}" y="{yy+4:.1f}" font-size="10" fill="#9db0c6" text-anchor="end">{gv}</text>')
        gv += step
    # roofline
    yr = Y(roof)
    s.append(f'<line x1="{x0}" y1="{yr:.1f}" x2="{x0+w}" y2="{yr:.1f}" stroke="#1b6ca8" stroke-width="2" stroke-dasharray="7 5"/>')
    s.append(f'<text x="{x0}" y="{yr-6:.1f}" font-size="10.5" font-weight="600" fill="#1b6ca8">{esc(p["roofline_label"])}</text>')
    # bars
    n = len(bars)
    gap = 42
    bw = (w - gap * (n - 1)) / n
    for i, b in enumerate(bars):
        x = x0 + i * (bw + gap)
        if b["value"] is None:
            # pending ghost: dashed outline to roofline-ish small height
            s.append(f'<rect x="{x:.1f}" y="{Y(roof*0.12):.1f}" width="{bw:.1f}" height="{top+plotH-Y(roof*0.12):.1f}" '
                     f'fill="none" stroke="#9db0c6" stroke-width="1.5" stroke-dasharray="5 4" rx="3"/>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{Y(roof*0.12)-8:.1f}" font-size="12" fill="#9db0c6" text-anchor="middle">pending</text>')
        else:
            yv = Y(b["value"])
            c = COL.get(b["kind"], "#5b6b80")
            s.append(f'<rect x="{x:.1f}" y="{yv:.1f}" width="{bw:.1f}" height="{top+plotH-yv:.1f}" fill="{c}" rx="3"/>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{yv-8:.1f}" font-size="15" font-weight="700" fill="{c}" text-anchor="middle">{b["value"]:.1f}</text>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{yv+16:.1f}" font-size="9.5" fill="#ffffff" text-anchor="middle">{100*b["value"]/roof:.0f}% roof</text>')
        # x labels
        s.append(f'<text x="{x+bw/2:.1f}" y="{top+plotH+16:.1f}" font-size="11.5" font-weight="600" fill="#0b2447" text-anchor="middle">{esc(b["label"])}</text>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{top+plotH+30:.1f}" font-size="9.5" fill="#5b6b80" text-anchor="middle">{esc(b["sub"])}</text>')
    s.append(f'<text x="{x0-40}" y="{top+plotH/2:.0f}" font-size="10.5" fill="#5b6b80" '
             f'transform="rotate(-90 {x0-40} {top+plotH/2:.0f})" text-anchor="middle">{esc(p["unit"])}</text>')
    return "".join(s)


def build_svg(d):
    W, H = 1240, 560
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(f'<text x="48" y="34" font-size="21" font-weight="700" fill="#0b2447">{esc(d["title"])}</text>')
    sub = d["subtitle"]
    words, line, yy = sub.split(" "), "", 56
    for wd in words:
        if len(line) + len(wd) > 150:
            s.append(f'<text x="48" y="{yy}" font-size="11.5" fill="#5b6b80">{esc(line)}</text>')
            line, yy = wd + " ", yy + 15
        else:
            line += wd + " "
    if line:
        s.append(f'<text x="48" y="{yy}" font-size="11.5" fill="#5b6b80">{esc(line)}</text>')
    panels = d["panels"]
    pw = (W - 120) / 2
    for i, p in enumerate(panels):
        s.append(panel_svg(p, 80 + i * (pw + 40), yy + 20, pw - 40, H - (yy + 20) - 24))
    s.append('</svg>')
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    ap.add_argument("--data", default=os.path.join(HERE, "results", "dsv4_node_walltime.json"))
    args = ap.parse_args()
    with open(args.data) as f:
        d = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    svg = os.path.join(args.out_dir, "dsv4_node_walltime.svg")
    with open(svg, "w") as f:
        f.write(build_svg(d))
    print("wrote", svg)
    png = os.path.join(args.out_dir, "dsv4_node_walltime_v2.png")  # bump _vN on any content change (camo cache)
    exe = shutil.which("rsvg-convert")
    if exe:
        subprocess.run([exe, "-z", "2", "-o", png, svg], check=True, capture_output=True)
        print("wrote", png)
    else:
        print("(no rsvg-convert; SVG only)")


if __name__ == "__main__":
    main()
