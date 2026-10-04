#!/usr/bin/env python3
"""Baseline -> optimized -> roofline bar chart (3 bars) for a memory-bound op.

Companion to roofline_vs_measured.py / time_attribution_pivot.py. Renders the optimization JOURNEY
as three vertical bars in the roofline's native metric (DRAM GB/s): the un-optimized operating point,
the optimized operating point, and the hardware ceiling — each labeled with its % of the roofline.
All three values are DIRECTLY MEASURED (no derivation). Hand-written SVG + rsvg-convert (no matplotlib).

Usage: python perf_journey_chart.py --out-prefix results/<name>
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

# DeepSeek-V4-Flash — routed-expert MoE (68% of decode), EMR 1 socket, MXFP4 W4A16.
# Source: plugin/validate/results/deepseek_v4_flash_roofline.json (bench_mxfp4_moe_roofline.py).
TITLE = "DeepSeek-V4-Flash — routed-expert MoE decode: DRAM bandwidth vs operating point"
SUBTITLE = ("EMR 1 socket · 226 GB/s stream_triad · MXFP4 W4A16 · routed-expert MoE (68% of decode) · "
            "achieved BW vs batch size")
ROOFLINE = 226.0
BARS = [
    ("M = 1", "unbatched decode", 59.0, "#c0504d"),
    ("M = 32", "batched decode", 170.0, "#2f6f4f"),
    ("Roofline", "stream_triad ceiling", 226.0, "#8a8a8a"),
]
FOOTNOTE = ("Operating-point chart, NOT an optimization delta: a larger batch amortizes & dedups expert-weight "
            "streaming over more tokens, so achieved BW rises with M (arithmetic intensity, not a kernel change). "
            "Implementation optimizations (spin-wait fix 63x prefill / 10x decode, thread-cap, native-MXFP4) are a "
            "separate fixed-workload before/after story. At M=32 the kernel reaches 75% of the stream ceiling.")


def build_svg() -> str:
    W, H = 940, 540
    padL, padR, padT, padB = 74, 34, 100, 136
    plotW, plotH = W - padL - padR, H - padT - padB
    ymax = 240.0  # headroom above the 226 roofline
    y0 = padT + plotH

    def yv(v: float) -> float:
        return padT + plotH * (1 - v / ymax)

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' font-family='sans-serif'>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         f"<text x='{padL}' y='34' font-size='18' font-weight='bold'>{html.escape(TITLE)}</text>",
         f"<text x='{padL}' y='56' font-size='12' fill='#444'>{html.escape(SUBTITLE)}</text>"]
    # y-axis gridlines + labels
    for gv in (0, 60, 120, 180, 240):
        gy = yv(gv)
        p.append(f"<line x1='{padL}' y1='{gy:.1f}' x2='{padL+plotW}' y2='{gy:.1f}' stroke='#eee'/>")
        p.append(f"<text x='{padL-8}' y='{gy+4:.1f}' font-size='11' fill='#666' text-anchor='end'>{gv}</text>")
    p.append(f"<text x='18' y='{padT+plotH/2:.0f}' font-size='12' fill='#444' "
             f"transform='rotate(-90 18 {padT+plotH/2:.0f})' text-anchor='middle'>DRAM bandwidth (GB/s)</text>")
    # roofline reference line (the roofline BAR itself carries the value/percent label)
    ry = yv(ROOFLINE)
    p.append(f"<line x1='{padL}' y1='{ry:.1f}' x2='{padL+plotW}' y2='{ry:.1f}' stroke='#8a8a8a' "
             f"stroke-width='1.5' stroke-dasharray='6 4'/>")
    # bars
    n = len(BARS)
    slot = plotW / n
    bw = slot * 0.5
    for i, (label, sub, val, color) in enumerate(BARS):
        cx = padL + slot * (i + 0.5)
        x = cx - bw / 2
        top = yv(val)
        p.append(f"<rect x='{x:.1f}' y='{top:.1f}' width='{bw:.1f}' height='{y0-top:.1f}' "
                 f"fill='{color}' rx='3'/>")
        pct = 100 * val / ROOFLINE
        p.append(f"<text x='{cx:.1f}' y='{top-24:.1f}' font-size='17' font-weight='bold' "
                 f"text-anchor='middle'>{val:.0f} GB/s</text>")
        p.append(f"<text x='{cx:.1f}' y='{top-7:.1f}' font-size='13' fill='#333' "
                 f"text-anchor='middle'>{pct:.0f}% of roofline</text>")
        p.append(f"<text x='{cx:.1f}' y='{y0+22:.1f}' font-size='14' font-weight='bold' "
                 f"text-anchor='middle'>{html.escape(label)}</text>")
        p.append(f"<text x='{cx:.1f}' y='{y0+40:.1f}' font-size='11' fill='#555' "
                 f"text-anchor='middle'>{html.escape(sub)}</text>")
    # improvement arrow baseline -> optimized
    x1 = padL + slot * 0.5
    x2 = padL + slot * 1.5
    ay = yv(170.0) - 48
    p.append(f"<text x='{(x1+x2)/2:.1f}' y='{ay:.1f}' font-size='13' font-weight='bold' fill='#2f6f4f' "
             f"text-anchor='middle'>2.9× — batch amortization (operating point)</text>")
    # footnote (wrapped)
    words, line, lines = FOOTNOTE.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > 118:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    for k, ln in enumerate(lines):
        p.append(f"<text x='{padL}' y='{H-54+k*15:.0f}' font-size='11' fill='#555'>{html.escape(ln)}</text>")
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    out = Path(args.out_prefix)
    svg = out.with_suffix(".svg")
    png = out.with_suffix(".png")
    svg.write_text(build_svg())
    import shutil
    import subprocess
    exe = shutil.which("rsvg-convert")
    if exe:
        subprocess.run([exe, "-z", "2", "-o", str(png), str(svg)], check=True, capture_output=True)
    else:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2.0)
    print(f"wrote {svg} and {png}")


if __name__ == "__main__":
    main()
