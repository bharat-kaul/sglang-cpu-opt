#!/usr/bin/env python3
"""Dependency-free SVG chart of the DSv4-Flash CPU end-to-end wall-time PROGRESSION (baseline -> each
integrated optimization) for the GitHub page. Reads results/wall_time_progress.json; a stage with a numeric
full-depth decode tok/s is drawn as a solid bar, a PENDING/OOM stage as a dashed placeholder (so the pipeline
is visible and fills in as Phase-2 lands). North-Star = full-depth (43L) single-stream decode tok/s (B=1),
with prefill TTFT annotated. No matplotlib -> renders on github.com.

Usage: python make_e2e_walltime_plot.py --out-dir ../../docs/plots
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _txt(x, y, s, size=12, anchor="start", color="#222", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="monospace" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{_esc(s)}</text>')


def _num(v):
    return v if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    a = ap.parse_args()
    d = json.load(open(os.path.join(RES, "wall_time_progress.json")))
    op = d.get("operating_point", {})
    hp = d.get("headline_progression")

    # Prefer the focused headline progression (43L B=32 full-node); fall back to the per-op ledger.
    rows = []
    if hp:
        axis_label = hp.get("axis", "decode tok/s (43L)")
        northstar = ("North-Star: full-depth (43L) B=32 decode tok/s at FULL-NODE (all 3 SNC NUMA domains / 128c) "
                     "\u2014 the max-throughput config; higher = better; prefill TTFT annotated.")
        for st in hp["stages"]:
            rows.append((st["label"], _num(st.get("decode_tok_s")), _num(st.get("prefill_s"))))
    else:
        axis_label = "decode tok/s (B=1, 43L)"
        northstar = (f"North-Star: full-depth (43L) single-stream decode tok/s (B={op.get('batch_B',1)}, "
                     f"S={op.get('input_S',1024)}) \u2014 higher = better; prefill TTFT annotated.")
        steps = d.get("ledger", [])
        for st in steps:
            lbl = st.get("op") or "baseline"
            lbl = {"baseline": "baseline", "indexer_logits": "+indexer\nlogits", "indexer_topk": "+indexer\ntop-k",
                   "compressor": "+compressor", "sparse_attend": "+sparse\nattend", "mhc_sinkhorn": "+MHC\nsinkhorn",
                   "mhc_hash": "+MHC\nhash", "laneA_verify": "+LaneA\nverify", "system_overhead": "+system\noverhead"}.get(lbl, lbl)
            rows.append((lbl, _num(st.get("decode_tok_s_full")), _num(st.get("prefill_s_full"))))

    W, H = 860, 460
    L, R, T, B = 70, 30, 74, 70
    pw, ph = W - L - R, H - T - B
    done = [r[1] for r in rows if r[1] is not None]
    ymax = max(done + [1.0]) * 1.8 if done else 1.0        # headroom for future gains
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
         _txt(16, 24, "DSv4-Flash CPU end-to-end wall time \u2014 progressive improvement", 15, "start", "#111", "bold"),
         _txt(16, 42, northstar, 10, "start", "#666"),
         _txt(16, 57, "Solid = measured (fingerprint-gated). Dashed = pending (D2/D3: port torch decode ops, MoE "
                      "thread-cap). Baseline = torch-ref DSA, dummy-weight two-depth extrapolation.", 10, "start", "#999")]
    # y gridlines
    for g in range(0, 6):
        yv = ymax * g / 5
        y = T + ph - ph * (yv / ymax)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(_txt(L - 8, y + 4, f"{yv:.2f}", 10, "end", "#666"))
    s.append(_txt(18, T + ph / 2, axis_label, 11, "middle", "#444")
             .replace('text-anchor="middle"', f'text-anchor="middle" transform="rotate(-90 18 {T+ph/2:.0f})"'))
    n = len(rows)
    gw = pw / max(n, 1)
    bw = gw * 0.6
    for i, (lbl, tok, pre) in enumerate(rows):
        cx = L + gw * i + gw / 2
        bx = cx - bw / 2
        if tok is not None:
            by = T + ph - ph * (min(tok, ymax) / ymax)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{T+ph-by:.1f}" fill="#1f77b4"/>')
            s.append(_txt(cx, by - 14, f"{tok:.2f}", 11, "middle", "#111", "bold"))
            s.append(_txt(cx, by - 3, f"tok/s", 8, "middle", "#1f77b4"))
            if pre is not None:
                s.append(_txt(cx, T + ph - 6, f"TTFT {pre:.0f}s", 8, "middle", "#fff"))
        else:
            # pending placeholder: dashed outline at a modest height
            ph0 = ph * 0.20
            by = T + ph - ph0
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{ph0:.1f}" fill="#f5f5f5" '
                     f'stroke="#bbb" stroke-dasharray="4,3"/>')
            s.append(_txt(cx, by - 4, "pending", 9, "middle", "#bbb"))
        # stage label (wrap on \n)
        for li, part in enumerate(lbl.split("\n")):
            s.append(_txt(cx, T + ph + 16 + li * 11, part, 9, "middle", "#444", "bold" if i == 0 else "normal"))
    out = os.path.join(a.out_dir, "e2e_walltime.svg")
    with open(out, "w") as f:
        f.write("\n".join(s) + "\n</svg>\n")
    print("wrote", out, f"({sum(1 for r in rows if r[1] is not None)} measured / {n} stages)")


if __name__ == "__main__":
    main()
