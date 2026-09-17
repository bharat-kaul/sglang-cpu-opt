#!/usr/bin/env python3
"""Roofline-target-vs-measured report + chart for published enablement results.

Consumes the `model-profile-hotspots` output (per-op measured value + roofline
target + phase share) and emits, for the GitHub results page:
  - a Markdown table ranked by the biggest recoverable gap (shortfall x share),
  - a dependency-free SVG grouped bar chart (roofline achievable vs measured per op),
  - the model-level roofline tok/s target vs measured headline.

"Higher is better" convention: every metric is a throughput (tok/s or TF/s), so
efficiency = measured / roofline_target in (0, 1]. For a time-based measurement,
convert to a rate before feeding this tool. `measured: null` renders as PENDING
(target published now; the gap fills in once the kernel runs) — which is the point:
publish the target up front, then show how close the measured comes.

Input JSON schema:
{
  "model": str, "node": str, "precision": str, "phase": str, "batch": int,
  "unit": "tok/s" | "TF/s",
  "model_roofline_target": float, "model_measured": float|null,
  "ops": [ {"op": str, "roofline_target": float, "measured": float|null,
            "share_pct": float, "regime": "memory|compute", "note": str? } ]
}

Usage:
  python roofline_vs_measured.py --in <profile.json> --out-prefix <results/name>
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _eff(measured, target):
    if measured is None or not target:
        return None
    return max(0.0, min(1.0, measured / target))


def _fmt(v, unit=""):
    return "PENDING" if v is None else f"{v:.1f}{unit}"


def build_markdown(d: dict) -> str:
    unit = d.get("unit", "tok/s")
    ops = list(d.get("ops", []))
    for o in ops:
        o["_eff"] = _eff(o.get("measured"), o.get("roofline_target"))
        # recoverable end-to-end fraction: shortfall x share
        shortfall = 1.0 if o["_eff"] is None else (1.0 - o["_eff"])
        o["_recover"] = shortfall * (o.get("share_pct", 0.0) / 100.0)
    ops.sort(key=lambda o: o["_recover"], reverse=True)

    mt, mm = d.get("model_roofline_target"), d.get("model_measured")
    meff = _eff(mm, mt)
    lines = [
        f"# Roofline target vs measured — {d.get('model','?')}",
        "",
        f"**Node** `{d.get('node','?')}` · **{d.get('precision','?')}** · "
        f"**{d.get('phase','?')}** · batch {d.get('batch','?')} · unit `{unit}`"
        + (f" · **{d.get('params')}**" if d.get('params') else ""),
        "",
        f"**Model-level:** roofline target **{_fmt(mt, ' '+unit)}** · "
        f"measured **{_fmt(mm, ' '+unit)}**"
        + (f" · **{meff*100:.0f}% of achievable**" if meff is not None else " (target published; measured to follow)"),
        "",
        "Per-op, ranked by recoverable end-to-end fraction (shortfall × phase share):",
        "",
        f"| op | phase share % | regime | roofline ({unit}) | measured ({unit}) | efficiency | recoverable % |",
        "|----|---------------|--------|-------------------|-------------------|------------|---------------|",
    ]
    for o in ops:
        eff = o["_eff"]
        lines.append(
            f"| {o['op']} | {o.get('share_pct',0):.1f} | {o.get('regime','?')} | "
            f"{_fmt(o.get('roofline_target'))} | {_fmt(o.get('measured'))} | "
            f"{'PENDING' if eff is None else f'{eff*100:.0f}%'} | {o['_recover']*100:.1f} |"
        )
    lines += [
        "",
        "![roofline vs measured](./" + d.get("_img_name", "roofline_vs_measured.png") + ")",
        "",
        "> Bars: roofline-achievable (target) vs measured per op. A tall gap on a "
        "high-share op is the top optimization RoI. `PENDING` = target published; "
        "measured fills in when the kernel runs.",
    ]
    return "\n".join(lines) + "\n"


def build_svg(d: dict) -> str:
    unit = d.get("unit", "tok/s")
    ops = list(d.get("ops", []))
    ops.sort(key=lambda o: o.get("share_pct", 0.0), reverse=True)
    if not ops:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='40'></svg>"
    n = len(ops)
    row_h, pad_l, pad_t, width, bar_gap = 46, 210, 86, 1120, 6
    height = pad_t + n * row_h + 40
    plot_w = width - pad_l - 200
    vmax = max(
        max((o.get("roofline_target") or 0) for o in ops),
        max((o.get("measured") or 0) for o in ops),
        1e-9,
    )

    def x(v):
        return pad_l + (v / vmax) * plot_w

    subtitle = (
        f"Machine: {d.get('node', '?')}  \u00b7  {d.get('precision', '?')}  \u00b7  "
        f"batch {d.get('batch', '?')}  \u00b7  {d.get('phase', '?')}"
        + (f"  \u00b7  {d.get('params')}" if d.get('params') else "")
    )
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' font-family='sans-serif'>",
        f"<rect width='{width}' height='{height}' fill='white'/>",
        f"<text x='{pad_l}' y='26' font-size='18' font-weight='bold'>"
        f"{html.escape(str(d.get('model','?')))} — roofline vs measured ({html.escape(unit)})</text>",
        f"<text x='{pad_l}' y='48' font-size='13' fill='#444'>{html.escape(subtitle)}</text>",
        f"<rect x='{pad_l}' y='62' width='14' height='14' fill='#9db4d0'/>"
        f"<text x='{pad_l+20}' y='74' font-size='12'>roofline target</text>"
        f"<rect x='{pad_l+140}' y='62' width='14' height='14' fill='#2f6f4f'/>"
        f"<text x='{pad_l+160}' y='74' font-size='12'>measured</text>",
    ]
    bh = (row_h - 2 * bar_gap) / 2
    for i, o in enumerate(ops):
        y0 = pad_t + i * row_h
        tgt = o.get("roofline_target") or 0
        meas = o.get("measured")
        eff = _eff(meas, tgt)
        label = f"{o['op']} ({o.get('share_pct',0):.0f}%)"
        parts.append(
            f"<text x='{pad_l-8}' y='{y0+row_h/2}' font-size='12' text-anchor='end'>"
            f"{html.escape(label)}</text>"
        )
        parts.append(
            f"<rect x='{pad_l}' y='{y0+bar_gap}' width='{x(tgt)-pad_l:.1f}' height='{bh:.1f}' fill='#9db4d0'/>"
        )
        if meas is None:
            parts.append(
                f"<rect x='{pad_l}' y='{y0+bar_gap+bh}' width='{x(tgt)-pad_l:.1f}' height='{bh:.1f}' "
                f"fill='none' stroke='#2f6f4f' stroke-dasharray='4 3'/>"
                f"<text x='{x(tgt)+6:.1f}' y='{y0+bar_gap+bh*1.7:.1f}' font-size='11' fill='#2f6f4f'>PENDING</text>"
            )
        else:
            parts.append(
                f"<rect x='{pad_l}' y='{y0+bar_gap+bh}' width='{x(meas)-pad_l:.1f}' height='{bh:.1f}' fill='#2f6f4f'/>"
                f"<text x='{x(max(tgt,meas))+6:.1f}' y='{y0+row_h/2+4:.1f}' font-size='11'>"
                f"{eff*100:.0f}% of achievable</text>"
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _rasterize_png(svg_path: Path, png_path: Path) -> bool:
    """SVG -> PNG so GitHub renders the chart (it refuses to render SVG).

    Prefers rsvg-convert; falls back to cairosvg. Returns True on success.
    """
    import shutil
    import subprocess

    exe = shutil.which("rsvg-convert")
    if exe:
        try:
            subprocess.run(
                [exe, "-z", "2", "-o", str(png_path), str(svg_path)],
                check=True,
                capture_output=True,
            )
            return True
        except Exception:
            pass
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2.0)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-prefix", required=True, help="path prefix for .md/.svg/.png outputs")
    args = ap.parse_args()

    d = json.loads(Path(args.inp).read_text())
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out.parent / (out.name + ".svg")
    png_path = out.parent / (out.name + ".png")
    svg_path.write_text(build_svg(d))
    # GitHub renders PNG reliably but refuses SVG; embed the PNG when we can
    # rasterize, else fall back to the SVG reference.
    d["_img_name"] = out.name + (".png" if _rasterize_png(svg_path, png_path) else ".svg")
    (out.parent / (out.name + ".md")).write_text(build_markdown(d))
    print(f"wrote {out}.md, {out}.svg and {d['_img_name']}")


if __name__ == "__main__":
    main()
