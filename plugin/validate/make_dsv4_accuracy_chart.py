#!/usr/bin/env python3
"""DeepSeek-V4-Flash CPU gsm8k accuracy progression vs the same-checkpoint GPU oracle.

Bars = CPU gsm8k at each fix stage (baseline red → prefill-fix amber → full-fix green); dashed line =
the GPU oracle on the SAME post-trained Flash checkpoint + SAME 8-shot protocol (the apples-to-apples
target). Data-driven from results/dsv4_accuracy_progression.json. Dependency-light SVG → PNG via
rsvg-convert (matches the other chart generators).

Usage: python make_dsv4_accuracy_chart.py --out-dir ../../docs/plots
"""
import argparse
import html
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

COLORS = {"bad": "#c0504d", "ok": "#d08a1d", "good": "#2f6f4f"}


def esc(s):
    return html.escape(str(s))


def build_svg(d):
    stages = d["stages"]
    ceil = d["ceiling"]
    W, H = 1120, 540
    padL, padR, padT, padB = 70, 24, 96, 104
    plotW, plotH = W - padL - padR, H - padT - padB
    y0 = padT + plotH
    ymax = 1.0

    def Y(v):
        return y0 - (v / ymax) * plotH

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(f'<text x="{padL}" y="36" font-size="22" font-weight="700" fill="#0b2447">{esc(d["title"])}</text>')
    # wrap subtitle
    sub = d["subtitle"]
    words, line, yy = sub.split(" "), "", 58
    for w in words:
        if len(line) + len(w) > 128:
            s.append(f'<text x="{padL}" y="{yy}" font-size="12.5" fill="#5b6b80">{esc(line)}</text>')
            line, yy = w + " ", yy + 17
        else:
            line += w + " "
    if line:
        s.append(f'<text x="{padL}" y="{yy}" font-size="12.5" fill="#5b6b80">{esc(line)}</text>')

    for gv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yg = Y(gv)
        s.append(f'<line x1="{padL}" y1="{yg:.1f}" x2="{W-padR}" y2="{yg:.1f}" stroke="#eef2f7"/>')
        s.append(f'<text x="{padL-8}" y="{yg+4:.1f}" font-size="11" fill="#9db0c6" text-anchor="end">{gv:.1f}</text>')
    s.append(f'<text x="18" y="{padT+plotH/2:.0f}" font-size="12" fill="#5b6b80" '
             f'transform="rotate(-90 18 {padT+plotH/2:.0f})" text-anchor="middle">gsm8k accuracy (8-shot CoT)</text>')

    yc = Y(ceil)
    s.append(f'<line x1="{padL}" y1="{yc:.1f}" x2="{W-padR}" y2="{yc:.1f}" stroke="#1b6ca8" stroke-width="2" stroke-dasharray="7 5"/>')
    s.append(f'<text x="{padL}" y="{yc-7:.1f}" font-size="12.5" font-weight="600" fill="#1b6ca8" '
             f'text-anchor="start">{esc(d["ceiling_label"])}</text>')

    n = len(stages)
    gap = 60
    bw = (plotW - gap * (n - 1)) / n
    for i, st in enumerate(stages):
        x = padL + i * (bw + gap)
        yv = Y(st["value"])
        col = COLORS.get(st["kind"], "#5b6b80")
        s.append(f'<rect x="{x:.1f}" y="{yv:.1f}" width="{bw:.1f}" height="{y0-yv:.1f}" fill="{col}" rx="3"/>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{yv-10:.1f}" font-size="19" font-weight="700" fill="{col}" '
                 f'text-anchor="middle">{st["value"]:.3f}</text>')
        # % of GPU ceiling
        pct = 100.0 * st["value"] / ceil
        s.append(f'<text x="{x+bw/2:.1f}" y="{yv+20:.1f}" font-size="11.5" fill="#ffffff" '
                 f'text-anchor="middle">{pct:.0f}% of GPU</text>')
        # x labels (support \n)
        for li, part in enumerate(st["label"].split("\n")):
            s.append(f'<text x="{x+bw/2:.1f}" y="{y0+22+li*16:.1f}" font-size="13" font-weight="600" '
                     f'fill="#0b2447" text-anchor="middle">{esc(part)}</text>')
        nlines = len(st["label"].split("\n"))
        s.append(f'<text x="{x+bw/2:.1f}" y="{y0+24+nlines*16:.1f}" font-size="11" fill="#5b6b80" '
                 f'text-anchor="middle">{esc(st["sub"])}</text>')

    s.append('</svg>')
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    ap.add_argument("--data", default=os.path.join(HERE, "results", "dsv4_accuracy_progression.json"))
    args = ap.parse_args()
    with open(args.data) as f:
        d = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    svg = os.path.join(args.out_dir, "dsv4_accuracy_progression.svg")
    with open(svg, "w") as f:
        f.write(build_svg(d))
    print("wrote", svg)
    png = os.path.join(args.out_dir, "dsv4_accuracy_progression_v1.png")  # bump to _v2 on any content change (camo cache)
    exe = shutil.which("rsvg-convert")
    if exe:
        subprocess.run([exe, "-z", "2", "-o", png, svg], check=True, capture_output=True)
        print("wrote", png)
    else:
        print("(no rsvg-convert; SVG only)")


if __name__ == "__main__":
    main()
