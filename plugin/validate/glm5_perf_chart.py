#!/usr/bin/env python3
"""GLM-5.3 Flash CPU prefill optimization chart (baseline -> optimized -> roofline).

Honest 3-bar + a realistic-floor marker. All MEASURED (8-layer proxy, 256-tok prefill, GNR
256 threads) except the fused-kernel floor (projected). Metric = prefill time (log y-axis so
the AMX compute ceiling is visible next to the seconds-scale bars). Hand-written SVG +
rsvg-convert (matches perf_journey_chart.py).

Key honesty (see roofline json): the AMX dense-GEMM ceiling (~0.09s) is NOT reachable for the
gated-delta KDA recurrence — it is dispatch/small-op bound, and even the native AMX kernel is
only parity with torch compute. The realistic floor is ~2.7s (a fused CPU KDA kernel). The
achieved win is the warm-compiled router (bit-exact) + chunked KDA matmuls.
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

TITLE = "GLM-5.3 Flash — CPU prefill: baseline \u2192 optimized \u2192 roofline"
SUBTITLE = ("GNR 1 node (256 threads) \u00b7 8-layer proxy, 256-tok prefill \u00b7 fp8 W8A16 \u00b7 "
            "measured layer.total.pf (log scale)")
# (label, sublabel, seconds, color)
BARS = [
    ("Baseline", "@torch.compile router + KDA scan", 55.5, "#c0504d"),
    ("Optimized", "warm-compiled router + chunked KDA", 7.69, "#2f6f4f"),
    ("Roofline", "AMX bf16 dense-GEMM ceiling", 0.092, "#8a8a8a"),
]
FLOOR = 4.0  # realistic achievable floor (fused CPU KDA kernel; non-KDA ops + overhead remain), projected
FOOTNOTE = (
    "7.2× achieved (55.5→7.7s), faithful (prefill+decode parity: logits cos 0.9999, identical "
    "tokens). The dense-GEMM roofline (0.09s) is the hardware compute ceiling but is NOT reachable: the "
    "gated-delta KDA recurrence is dispatch/small-op bound (even the native AMX kernel = parity with torch "
    "compute), so the realistic floor is ~4s via a fused CPU KDA kernel. Router 27→0.1s = warming "
    "the CPU @torch.compile (bit-exact), not an eager bypass; KDA 24.6→3.7s = chunked matmuls (WY form)."
)


def build_svg() -> str:
    W, H = 960, 560
    padL, padR, padT, padB = 78, 34, 104, 150
    plotW, plotH = W - padL - padR, H - padT - padB
    y0 = padT + plotH
    ymin, ymax = 0.05, 80.0  # log range (s)

    def yv(v: float) -> float:
        lv = (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))
        return padT + plotH * (1 - lv)

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' font-family='sans-serif'>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         f"<text x='{padL}' y='34' font-size='18' font-weight='bold'>{html.escape(TITLE)}</text>",
         f"<text x='{padL}' y='56' font-size='12' fill='#444'>{html.escape(SUBTITLE)}</text>"]
    for gv in (0.05, 0.1, 0.5, 1, 5, 10, 50):
        gy = yv(gv)
        p.append(f"<line x1='{padL}' y1='{gy:.1f}' x2='{padL+plotW}' y2='{gy:.1f}' stroke='#eee'/>")
        p.append(f"<text x='{padL-8}' y='{gy+4:.1f}' font-size='11' fill='#666' text-anchor='end'>{gv:g}s</text>")
    p.append(f"<text x='20' y='{padT+plotH/2:.0f}' font-size='12' fill='#444' "
             f"transform='rotate(-90 20 {padT+plotH/2:.0f})' text-anchor='middle'>prefill time (s, log)</text>")
    # realistic floor line
    fy = yv(FLOOR)
    p.append(f"<line x1='{padL}' y1='{fy:.1f}' x2='{padL+plotW}' y2='{fy:.1f}' stroke='#2f6f4f' "
             f"stroke-width='1.3' stroke-dasharray='6 4'/>")
    p.append(f"<text x='{padL+plotW:.1f}' y='{fy-6:.1f}' font-size='11' fill='#2f6f4f' "
             f"text-anchor='end'>realistic floor ~{FLOOR:g}s (fused CPU KDA kernel)</text>")
    n = len(BARS)
    slot = plotW / n
    bw = slot * 0.46
    for i, (label, sub, val, color) in enumerate(BARS):
        cx = padL + slot * (i + 0.5)
        x = cx - bw / 2
        top = yv(val)
        p.append(f"<rect x='{x:.1f}' y='{top:.1f}' width='{bw:.1f}' height='{y0-top:.1f}' fill='{color}' rx='3'/>")
        p.append(f"<text x='{cx:.1f}' y='{top-10:.1f}' font-size='17' font-weight='bold' "
                 f"text-anchor='middle'>{val:g}s</text>")
        p.append(f"<text x='{cx:.1f}' y='{y0+22:.1f}' font-size='14' font-weight='bold' "
                 f"text-anchor='middle'>{html.escape(label)}</text>")
        p.append(f"<text x='{cx:.1f}' y='{y0+40:.1f}' font-size='10.5' fill='#555' "
                 f"text-anchor='middle'>{html.escape(sub)}</text>")
    # 7.2x arrow baseline -> optimized
    x1 = padL + slot * 0.5
    x2 = padL + slot * 1.5
    ay = yv(55.5) - 16
    p.append(f"<text x='{(x1+x2)/2:.1f}' y='{ay:.1f}' font-size='14' font-weight='bold' fill='#2f6f4f' "
             f"text-anchor='middle'>7.2\u00d7</text>")
    words, line, lines = FOOTNOTE.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > 120:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    for k, ln in enumerate(lines):
        p.append(f"<text x='{padL}' y='{H-66+k*15:.0f}' font-size='11' fill='#555'>{html.escape(ln)}</text>")
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default=str(Path(__file__).parent / "results" / "glm5_flash_prefill_journey"))
    args = ap.parse_args()
    out = Path(args.out_prefix)
    out.parent.mkdir(exist_ok=True)
    svg = out.with_suffix(".svg")
    svg.write_text(build_svg())
    png = out.with_suffix(".png")
    import shutil
    import subprocess
    exe = shutil.which("rsvg-convert")
    if exe:
        subprocess.run([exe, "-z", "2", "-o", str(png), str(svg)], check=True, capture_output=True)
    else:
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2.0)
        except Exception:
            print("(no rsvg-convert/cairosvg; SVG only)")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
