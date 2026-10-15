#!/usr/bin/env python3
"""Generate dependency-free SVG performance plots for the C/C++ kernel-pilot GitHub page from the committed,
self-identifying artifacts: results/roofline_sweep.json (optimized kernels vs nominal-reference roofline) and
results/standalone_opt.json (baseline->optimized standalone deltas). No matplotlib; pure-string SVG so it
renders on github.com. All numbers are NOMINAL-reference (not an absolute ceiling); labels carry the caveat.

Usage: python make_kernel_pilot_plots.py --out-dir ../../docs/plots
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
MS = [1, 2, 4, 8, 16, 32, 64]

# useful-byte contracts (identical to aggregate_roofline.py) for the BW-efficiency view; indexer/topk at S=1024
BW = 358.4e9
_B4, _I8, _H, _S, _D = 4.0, 8.0, 64, 1024, 128
BYTES = {
    "indexer_logits/tiled": lambda M: M * _H * _D * _B4 + M * _S * _D * _B4 + M * _H * _B4 + M * _S * _B4,
    "indexer_topk": lambda M: M * _S * _B4 + M * 512 * _I8,
    "compressor/r128d512": lambda M: M * 128 * 512 * _B4 * 2 + M * 512 * _B4 + 128 * 512 * _B4,
    "compressor/r8d512": lambda M: M * 8 * 512 * _B4 * 2 + M * 512 * _B4 + 8 * 512 * _B4,
    "compressor/r8d128": lambda M: M * 8 * 128 * _B4 * 2 + M * 128 * _B4 + 8 * 128 * _B4,
    "sparse/bestof": lambda M: M * 512 * 512 * _B4 + 2 * M * 64 * 512 * _B4,
    "sinkhorn": lambda M: M * 24 * _B4 + 3 * _B4 + 24 * _B4 + M * 24 * _B4,
    "combine": lambda M: M * 4 * 4096 * _B4 + M * 4 * _B4 + M * 4096 * _B4,
}


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _txt(x, y, s, size=12, anchor="start", color="#222", weight="normal"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="monospace" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{_esc(s)}</text>'


def _frame(W, H, title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
         _txt(16, 24, title, 15, "start", "#111", "bold")]
    if subtitle:
        s.append(_txt(16, 42, subtitle, 11, "start", "#666"))
    return s


def line_chart(fname, title, subtitle, series, ylabel, ymax, pct=True):
    """series: list of (name, [y per M], color)."""
    W, H = 760, 420
    L, R, T, B = 70, 180, 60, 50
    pw, ph = W - L - R, H - T - B
    s = _frame(W, H, title, subtitle)
    # axes + gridlines
    for g in range(0, 6):
        yv = ymax * g / 5
        y = T + ph - ph * (yv / ymax)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(_txt(L - 8, y + 4, f"{yv:.0f}", 10, "end", "#666"))
    s.append(_txt(18, T + ph / 2, ylabel, 11, "middle", "#444") .replace('text-anchor="middle"', f'text-anchor="middle" transform="rotate(-90 18 {T+ph/2:.0f})"'))
    for i, M in enumerate(MS):
        x = L + pw * i / (len(MS) - 1)
        s.append(_txt(x, T + ph + 18, f"M={M}", 10, "middle", "#666"))
    for idx, (name, ys, color) in enumerate(series):
        pts = []
        for i, yv in enumerate(ys):
            if yv is None:
                continue
            x = L + pw * i / (len(MS) - 1)
            y = T + ph - ph * (min(yv, ymax) / ymax)
            pts.append((x, y))
        if pts:
            s.append('<polyline fill="none" stroke="' + color + '" stroke-width="2" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
            for x, y in pts:
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        ly = T + 6 + idx * 18
        s.append(f'<rect x="{L+pw+14}" y="{ly-9}" width="12" height="12" fill="{color}"/>')
        s.append(_txt(L + pw + 30, ly + 1, name, 10, "start", "#333"))
    with open(fname, "w") as f:
        f.write("\n".join(s) + "\n</svg>\n")


def grouped_bars(fname, title, subtitle, groups, series, ylabel, ymax, logy=False, fmt="{:.0f}"):
    """series: list of (name, [val per group], color)."""
    import math
    W, H = 760, 430
    L, R, T, B = 72, 150, 60, 55
    pw, ph = W - L - R, H - T - B
    s = _frame(W, H, title, subtitle)

    def ypix(v):
        if logy:
            lo, hi = math.log10(max(ymax[0], 1e-6)), math.log10(ymax[1])
            return T + ph - ph * (math.log10(max(v, ymax[0])) - lo) / (hi - lo)
        return T + ph - ph * (v / ymax[1])
    if logy:
        import math as _m
        decade = int(_m.floor(_m.log10(ymax[0])))
        while 10 ** decade <= ymax[1]:
            yv = 10 ** decade
            y = ypix(yv)
            s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="#eee"/>')
            s.append(_txt(L - 8, y + 4, f"{yv:g}", 10, "end", "#666"))
            decade += 1
    else:
        for g in range(0, 6):
            yv = ymax[1] * g / 5
            y = ypix(yv)
            s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="#eee"/>')
            s.append(_txt(L - 8, y + 4, fmt.format(yv), 10, "end", "#666"))
    s.append(_txt(18, T + ph / 2, ylabel, 11, "middle", "#444").replace('text-anchor="middle"', f'text-anchor="middle" transform="rotate(-90 18 {T+ph/2:.0f})"'))
    ng, ns = len(groups), len(series)
    gw = pw / ng
    bw = gw * 0.8 / ns
    for gi, g in enumerate(groups):
        gx = L + gw * gi
        for si, (name, vals, color) in enumerate(series):
            v = vals[gi]
            if v is None:
                continue
            bx = gx + gw * 0.1 + si * bw
            by = ypix(v)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw*0.9:.1f}" height="{T+ph-by:.1f}" fill="{color}"/>')
        s.append(_txt(gx + gw / 2, T + ph + 18, g, 10, "middle", "#666"))
    for si, (name, vals, color) in enumerate(series):
        ly = T + 6 + si * 18
        s.append(f'<rect x="{L+pw+14}" y="{ly-9}" width="12" height="12" fill="{color}"/>')
        s.append(_txt(L + pw + 30, ly + 1, name, 10, "start", "#333"))
    with open(fname, "w") as f:
        f.write("\n".join(s) + "\n</svg>\n")


def speedup_chart(fname, title, subtitle, series, ylabel, ticks):
    """Log-y speedup line chart; series: list of (name, [y per M], color, dash_bool)."""
    import math
    W, H = 840, 450
    L, R, T, B = 70, 250, 62, 50
    pw, ph = W - L - R, H - T - B
    lo, hi = math.log10(ticks[0]), math.log10(ticks[-1])
    s = _frame(W, H, title, subtitle)

    def ypix(v):
        return T + ph - ph * (math.log10(max(v, ticks[0] * 0.9)) - lo) / (hi - lo)
    for tk in ticks:
        y = ypix(tk)
        col = "#bbb" if tk == 1 else "#eee"
        da = ' stroke-dasharray="4 3"' if tk == 1 else ''
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{col}"{da}/>')
        s.append(_txt(L - 8, y + 4, f"{tk:g}x", 10, "end", "#666"))
    s.append(_txt(18, T + ph / 2, ylabel, 11, "middle", "#444").replace('text-anchor="middle"', f'text-anchor="middle" transform="rotate(-90 18 {T+ph/2:.0f})"'))
    for i, M in enumerate(MS):
        x = L + pw * i / (len(MS) - 1)
        s.append(_txt(x, T + ph + 18, f"M={M}", 10, "middle", "#666"))
    for idx, (name, ys, color, dash) in enumerate(series):
        pts = [(L + pw * i / (len(MS) - 1), ypix(v)) for i, v in enumerate(ys)]
        da = ' stroke-dasharray="6 4"' if dash else ''
        s.append('<polyline fill="none" stroke="' + color + f'" stroke-width="2"{da} points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
        for x, y in pts:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        ly = T + 6 + idx * 18
        s.append(f'<line x1="{L+pw+14}" y1="{ly-4}" x2="{L+pw+30}" y2="{ly-4}" stroke="{color}" stroke-width="2"{da}/>')
        s.append(_txt(L + pw + 34, ly - 1, name, 10, "start", "#333"))
    with open(fname, "w") as f:
        f.write("\n".join(s) + "\n</svg>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cvt = json.load(open(os.path.join(RES, "operating_point.json")))["kernels"]   # promoted variants, 512-bit, STREAMING

    # byte + FLOP contracts keyed by the cpp_vs_torch kernel names; floor = max(bytes/BW, FLOPs/peak).
    AMX, FP32 = 124.5184e12, 7.7824e12
    KC = {  # name: (flops(M), bytes(M), peak, pretty, oneDNN_donor)
        "indexer_logits": (lambda M: 2 * M * 64 * 1024 * 128 + 3 * M * 64 * 1024, lambda M: M * 64 * 128 * 4 + M * 1024 * 128 * 4 + M * 64 * 4 + M * 1024 * 4, AMX, "indexer logits", True),
        "indexer_topk": (lambda M: 0.0, lambda M: M * 1024 * 4 + M * 512 * 8, FP32, "indexer top-k", False),
        "compressor_R128": (lambda M: 3.0 * M * 128 * 512, lambda M: M * 128 * 512 * 4 * 2 + M * 512 * 4 + 128 * 512 * 4, FP32, "compressor", False),
        "sparse_bestof": (lambda M: 4.0 * M * 64 * 512 * 512, lambda M: M * 512 * 512 * 4 + 2 * M * 64 * 512 * 4, FP32, "sparse attend", True),
        "sinkhorn": (lambda M: float(M * 4 * 4 * 20 * 5), lambda M: M * 24 * 4 + 3 * 4 + 24 * 4 + M * 24 * 4, FP32, "MHC sinkhorn", False),
        "combine": (lambda M: float(M * 4096 * 7), lambda M: M * 4 * 4096 * 4 + M * 4 * 4 + M * 4096 * 4, FP32, "MHC combine", False),
    }
    order = ["indexer_logits", "indexer_topk", "compressor_R128", "sparse_bestof", "sinkhorn", "combine"]

    def floor_us(k, M):
        flf, byf, pk = KC[k][0], KC[k][1], KC[k][2]
        return max(byf(M) / BW, flf(M) / pk) * 1e6

    # Plot A (operating point M=32, S=1024): per-kernel roofline floor vs baseline(torch) vs optimized(C/C++)
    M0 = 32
    labels = [KC[k][3] + ("\u2020" if KC[k][4] else "") for k in order]   # dagger = oneDNN donor
    floors = [floor_us(k, M0) for k in order]
    bases = [cvt[k]["M"]["32"]["torch_stream_us"] for k in order]
    opts = [cvt[k]["M"]["32"]["cpp_stream_us"] for k in order]
    grouped_bars(os.path.join(a.out_dir, "operating_point.svg"),
                 "Per-kernel at the operating point (M=32, S=1024), STREAMING: roofline vs baseline vs C/C++",
                 "latency us, log, cold/streaming (LLC-flushed). baseline = torch op the kernel replaces; \u2020 = oneDNN/MKL GEMM donor. Promoted variants, 512-bit flags. EMR jobs 385457-59 (3-rep median).",
                 labels, [("roofline floor", floors, "#2ca02c"), ("baseline (torch)", bases, "#d62728"), ("optimized (C/C++)", opts, "#1f77b4")],
                 "latency us (log)", (min(floors) * 0.5, max(bases) * 1.6), logy=True, fmt="{:.2f}")

    # Plot B (M-sweep 1..64): % of nominal roofline achieved per kernel -> how AI growth improves efficiency
    series = []
    for i, k in enumerate(order):
        ys = [100.0 * floor_us(k, M) / cvt[k]["M"][str(M)]["cpp_stream_us"] for M in MS]
        series.append((KC[k][3] + ("\u2020" if KC[k][4] else ""), ys, PAL[i % len(PAL)]))
    line_chart(os.path.join(a.out_dir, "roofline_msweep.svg"),
               "% of nominal roofline vs M  (streaming; arithmetic intensity rises with M)",
               "%roof(M) = floor(M)/measured(M); floor(M)=max(bytes(M)/BW, FLOPs(M)/peak) RE-EVALUATED at each M against FIXED nominal peaks (BW, ISA/dtype TF) \u2014 peaks constant, the binding arm (BW vs compute) can flip with M. NOT an achievable ceiling; streaming. \u2020 = oneDNN donor. EMR jobs 385457-59 (3-rep median).",
               series, "% of nominal roofline", 100.0)

    # operating-point table for the page
    print(f"{'kernel':>16} {'impl':>8} {'base_us':>8} {'cpp_us':>8} {'speedup':>8} {'floor_us':>9} {'%roof':>6}")
    for k in order:
        c = cvt[k]["M"]["32"]
        fl = floor_us(k, M0)
        print(f"{k:>16} {('oneDNN' if KC[k][4] else 'authored'):>8} {c['torch_stream_us']:>8.1f} {c['cpp_stream_us']:>8.1f} "
              f"{c['speedup_stream']:>7.1f}x {fl:>9.3f} {100*fl/c['cpp_stream_us']:>5.0f}%")
    print("wrote operating_point.svg + roofline_msweep.svg ->", a.out_dir)


if __name__ == "__main__":
    main()
