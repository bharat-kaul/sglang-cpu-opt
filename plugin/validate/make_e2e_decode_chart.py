#!/usr/bin/env python3
"""Decode (TPOT) wall-time progression vs roofline — the living E2E chart at the top of e2e-wall-time.md.

Bars = perf-proxy B=32 decode tok/s at each integration/optimization stage (measured solid, regression red,
pending ghost); dashed line = the decode BW roofline ceiling; each measured bar annotated with % of roofline.
Data-driven from results/e2e_decode_progression.json so it updates as stages land (edit the JSON, re-run).
Dependency-free SVG.

Usage: python make_e2e_decode_chart.py --out-dir ../../docs/plots
"""
import argparse
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return html.escape(str(s))


def build_svg(d):
    stages = d["stages"]
    ceil = d["ceiling_tok_s"]
    W, H = 1180, 520
    padL, padR, padT, padB = 70, 24, 92, 86
    plotW, plotH = W - padL - padR, H - padT - padB
    y0 = padT + plotH
    ymax = ceil * 1.12

    def Y(v):
        return y0 - (v / ymax) * plotH

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(f'<text x="{padL}" y="36" font-size="22" font-weight="700" fill="#0b2447">Decode (TPOT) wall-time progression vs roofline</text>')
    s.append(f'<text x="{padL}" y="60" font-size="13" fill="#5b6b80">Perf-proxy (4-layer dummy) B=32 decode tok/s as the authored kernels integrate (one at a time) then the E2E optimizations land. Higher is better.</text>')
    s.append(f'<text x="{padL}" y="78" font-size="12.5" fill="#8a4b00">Memory-bound: the ceiling is weight-streaming BW \u2014 decode\u2019s open work is overhead + integration hygiene, not new compute kernels.</text>')

    # y gridlines
    for gv in range(0, int(ymax) + 1, 20):
        yy = Y(gv)
        s.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-padR}" y2="{yy:.1f}" stroke="#eef2f7"/>')
        s.append(f'<text x="{padL-8}" y="{yy+4:.1f}" font-size="11" fill="#9db0c6" text-anchor="end">{gv}</text>')
    s.append(f'<text x="18" y="{padT+plotH/2:.0f}" font-size="12" fill="#5b6b80" transform="rotate(-90 18 {padT+plotH/2:.0f})" text-anchor="middle">decode tok/s (proxy B=32)</text>')

    # roofline ceiling line
    yc = Y(ceil)
    s.append(f'<line x1="{padL}" y1="{yc:.1f}" x2="{W-padR}" y2="{yc:.1f}" stroke="#6a4c93" stroke-width="2" stroke-dasharray="7 5"/>')
    s.append(f'<text x="{W-padR}" y="{yc-7:.1f}" font-size="12.5" font-weight="600" fill="#6a4c93" text-anchor="end">roofline ceiling {ceil:.0f} tok/s \u2014 {esc(d["ceiling_label"])}</text>')

    n = len(stages)
    gap = 34
    bw = (plotW - gap * (n - 1)) / n
    for i, st in enumerate(stages):
        x = padL + i * (bw + gap)
        cx = x + bw / 2
        status = st["status"]
        v = st["tok_s"]
        if status == "pending" or v is None:
            # ghost placeholder bar
            gh = Y(ceil * 0.5)
            s.append(f'<rect x="{x:.1f}" y="{gh:.1f}" width="{bw:.1f}" height="{y0-gh:.1f}" fill="#f1f3f7" stroke="#d6dee8" stroke-dasharray="4 3"/>')
            s.append(f'<text x="{cx:.1f}" y="{(gh+y0)/2:.1f}" font-size="12" fill="#aab6c6" text-anchor="middle" transform="rotate(-90 {cx:.1f} {(gh+y0)/2:.1f})">pending</text>')
        else:
            fill = "#d1495b" if status == "regressed" else "#1d7874"
            yv = Y(v)
            s.append(f'<rect x="{x:.1f}" y="{yv:.1f}" width="{bw:.1f}" height="{y0-yv:.1f}" fill="{fill}" rx="2"/>')
            s.append(f'<text x="{cx:.1f}" y="{yv-20:.1f}" font-size="15" font-weight="700" fill="{fill}" text-anchor="middle">{v:.0f}</text>')
            pct = 100 * v / ceil
            tag = f'{pct:.0f}% of roof' + ('  \u25bc REGRESSED' if status == "regressed" else '')
            s.append(f'<text x="{cx:.1f}" y="{yv-5:.1f}" font-size="11" fill="{fill}" text-anchor="middle">{tag}</text>')
        # x labels
        s.append(f'<text x="{cx:.1f}" y="{y0+18:.1f}" font-size="12.5" font-weight="700" fill="#0b2447" text-anchor="middle">{esc(st["id"])}  {esc(st["label"])}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y0+34:.1f}" font-size="10.5" fill="#8294a8" text-anchor="middle">{esc(st.get("sub",""))}</text>')

    s.append(f'<text x="{padL}" y="{H-10}" font-size="11" fill="#9db0c6">Solid = measured (teal) \u00b7 ghost = pending \u00b7 dashed = BW roofline. Deltas from a controlled held-fixed A/B (off vs on, same job/config). Data: results/e2e_decode_progression.json.</text>')
    s.append('</svg>')
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    ap.add_argument("--data", default=os.path.join(HERE, "results", "e2e_decode_progression.json"))
    args = ap.parse_args()
    with open(args.data) as f:
        d = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "e2e_decode_progression.svg")
    with open(out, "w") as f:
        f.write(build_svg(d))
    print("wrote", out)


if __name__ == "__main__":
    main()
