#!/usr/bin/env python3
"""End-to-end TIME-ATTRIBUTION pivot — where the wall-clock goes, per phase, summing to 100%.

Companion to roofline_vs_measured.py: that shows "how efficient is each kernel vs its ceiling"
(shortfall); THIS shows "how much wall-time does each op consume" (share). Renders one
100%-stacked horizontal bar PER PHASE (prefill, decode) + a markdown table, coloured by bucket
(kernel / torch / framework / other). An explicit OTHER/unattributed slice (phase wall − sum of
wrapped ops) keeps it honest at 100% — never normalise only the measured ops.

Input JSON:
{
  "model": str, "node": str, "precision": str, "batch": int,
  "phases": {
    "prefill": {"wall_s": float|null, "ops": [{"op": str, "s": float, "bucket": "kernel|torch|framework"}]},
    "decode":  {"wall_s": float|null, "ops": [...] }
  }
}

Usage: python time_attribution_pivot.py --in <profile.json> --out-prefix results/<name>
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

BUCKET_COLOR = {"kernel": "#2f6f4f", "torch": "#9db4d0", "framework": "#d08c3c", "other": "#b9b9b9"}


def _rows(phase: dict):
    ops = sorted(phase.get("ops", []), key=lambda o: o["s"], reverse=True)
    summed = sum(o["s"] for o in ops)
    wall = phase.get("wall_s") or summed
    rows = [(o["op"], o["s"], o.get("bucket", "kernel")) for o in ops]
    other = wall - summed
    if other > 0.01 * wall:
        rows.append(("other / unattributed", other, "other"))
    return rows, max(wall, summed)


def build_markdown(d: dict) -> str:
    unit = "% of phase wall-time"
    out = [f"# Time attribution (where the wall-clock goes) — {d.get('model','?')}", "",
           f"**Node** `{d.get('node','?')}` · **{d.get('precision','?')}** · batch {d.get('batch','?')} · "
           f"unit `{unit}` — each phase sums to 100%.", ""]
    for phase in ("prefill", "decode"):
        if phase not in d.get("phases", {}):
            continue
        rows, tot = _rows(d["phases"][phase])
        out += [f"## {phase}", "",
                "| op | bucket | time (s) | % of phase |",
                "|----|--------|----------|------------|"]
        for op, s, b in rows:
            out.append(f"| {op} | {b} | {s:.3f} | {100*s/tot:.1f} |")
        out.append("")
    out += [f"![time attribution](./{d.get('_img_name','time_attribution.png')})", "",
            "> Each bar is one phase, segments = share of that phase's wall-time (sum = 100%). "
            "`other / unattributed` = phase wall − sum(wrapped ops); a big slice there means more ops "
            "need wrapping before trusting the split."]
    return "\n".join(out) + "\n"


def build_svg(d: dict) -> str:
    phases = [p for p in ("prefill", "decode") if p in d.get("phases", {})]
    W, padL, padT, rowH, barH = 1180, 90, 92, 150, 40
    H = padT + len(phases) * rowH + 30
    plotW = W - padL - 230
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' font-family='sans-serif'>",
        f"<rect width='{W}' height='{H}' fill='white'/>",
        f"<text x='{padL}' y='30' font-size='18' font-weight='bold'>"
        f"{html.escape(str(d.get('model','?')))} — time attribution (share of wall-time, per phase)</text>",
        f"<text x='{padL}' y='52' font-size='13' fill='#444'>"
        f"{html.escape(str(d.get('node','?')))} · {html.escape(str(d.get('precision','?')))} · "
        f"batch {d.get('batch','?')} · each bar = 100%</text>",
    ]
    lx = padL
    for b, c in BUCKET_COLOR.items():
        parts.append(f"<rect x='{lx}' y='62' width='13' height='13' fill='{c}'/>"
                     f"<text x='{lx+17}' y='73' font-size='11'>{b}</text>")
        lx += 95
    for i, phase in enumerate(phases):
        rows, tot = _rows(d["phases"][phase])
        y = padT + i * rowH
        parts.append(f"<text x='{padL}' y='{y-6}' font-size='14' font-weight='bold'>{phase}</text>")
        x = padL
        for j, (op, s, b) in enumerate(rows):
            w = (s / tot) * plotW
            parts.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='{barH}' "
                         f"fill='{BUCKET_COLOR.get(b,'#888')}' stroke='white'/>")
            pct = 100 * s / tot
            if w > 44:  # label inside wide segments
                parts.append(f"<text x='{x+w/2:.1f}' y='{y+barH/2+4:.1f}' font-size='11' "
                             f"fill='white' text-anchor='middle'>{pct:.0f}%</text>")
            # op legend to the right, stacked
            ly = y + 2 + j * 13
            if ly < y + barH + 60:
                parts.append(f"<rect x='{padL+plotW+14}' y='{ly}' width='10' height='10' fill='{BUCKET_COLOR.get(b,'#888')}'/>"
                             f"<text x='{padL+plotW+28}' y='{ly+9}' font-size='10'>{html.escape(op)} ({pct:.0f}%)</text>")
            x += w  # advance so segments STACK left→right (not overlap at padL)
        parts.append(f"<rect x='{padL}' y='{y}' width='{plotW}' height='{barH}' fill='none' stroke='#333'/>")
    parts.append("</svg>")
    return "\n".join(parts)


def _rasterize(svg: Path, png: Path) -> bool:
    import shutil
    import subprocess
    exe = shutil.which("rsvg-convert")
    if exe:
        try:
            subprocess.run([exe, "-z", "2", "-o", str(png), str(svg)], check=True, capture_output=True)
            return True
        except Exception:
            pass
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2.0)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    d = json.loads(Path(args.inp).read_text())
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    svg = out.parent / (out.name + ".svg")
    png = out.parent / (out.name + ".png")
    svg.write_text(build_svg(d))
    d["_img_name"] = out.name + (".png" if _rasterize(svg, png) else ".svg")
    (out.parent / (out.name + ".md")).write_text(build_markdown(d))
    print(f"wrote {out}.md, {out}.svg and {d['_img_name']}")


if __name__ == "__main__":
    main()
