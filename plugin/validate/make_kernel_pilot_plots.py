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
MS = [1, 8, 16, 32, 64]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "docs", "plots"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rs = json.load(open(os.path.join(RES, "roofline_sweep.json")))["kernels"]
    so = json.load(open(os.path.join(RES, "standalone_opt.json")))

    # Plot 1: achieved % of nominal-reference roofline, per kernel, vs M
    series = []
    for i, lb in enumerate(rs):
        ys = [rs[lb][str(M)]["achieved_pct"] for M in MS]
        series.append((lb, ys, PAL[i % len(PAL)]))
    line_chart(os.path.join(a.out_dir, "achieved_vs_roofline.svg"),
               "Optimized kernels: % of nominal-reference roofline (higher = closer)",
               "floor=max(bytes/BW,FLOPs/peak) vs NOMINAL peak (1.9GHz base; NOT an achievable ceiling). EMR job 384677.",
               series, "achieved % of nominal roofline", 100.0)

    # Plot 1b: achieved BW efficiency = (useful bytes / measured) / peak BW, per kernel vs M (tests BW-bound)
    bws = []
    for i, lb in enumerate(rs):
        ys = [100.0 * (BYTES[lb](M) / BW) / (rs[lb][str(M)]["median_meas_us"] / 1e6) for M in MS]
        bws.append((lb, ys, PAL[i % len(PAL)]))
    line_chart(os.path.join(a.out_dir, "bw_efficiency.svg"),
               "Achieved BW efficiency = (useful bytes / measured) / peak 358.4 GB/s  (S=1024)",
               "high+rising => BW-bound; low/flat => compute (sparse) or dispatch (sinkhorn/topk) bound. NOT all kernels are BW-bound.",
               bws, "% of peak DRAM BW (358.4 GB/s)", 100.0)

    # Plot 2: compressor R128/D512 -- roofline floor vs baseline vs optimized (us, log)
    cf = [rs["compressor/r128d512"][str(M)]["floor_us"] for M in MS]
    co = [rs["compressor/r128d512"][str(M)]["median_meas_us"] for M in MS]
    c1 = so["C1_compressor_one_exp"]["speedup_median_old_over_new"]["r128d512"]
    cb = [co[i] * c1[f"M{M}"] for i, M in enumerate(MS)]            # baseline = optimized x (old/new) speedup
    grouped_bars(os.path.join(a.out_dir, "compressor_3series.svg"),
                 "Compressor R128/D512: roofline floor vs baseline vs optimized (C1)",
                 "latency us, log. baseline = optimized x measured C1 speedup; floor = nominal ref. Bit-exact.",
                 [f"M={M}" for M in MS],
                 [("roofline floor", cf, "#2ca02c"), ("baseline", cb, "#d62728"), ("optimized", co, "#1f77b4")],
                 "latency us (log)", (min(cf) * 0.7, max(cb) * 1.3), logy=True, fmt="{:.1f}")

    # Plot 3: standalone optimization speedup (baseline->optimized), kept experiments
    def spd(d, key):
        return [d[key].get(f"M{M}") for M in MS]
    c1r = so["C1_compressor_one_exp"]["speedup_median_old_over_new"]
    series3 = [
        ("C1 compressor r128d512", spd(c1r, "r128d512"), PAL[0]),
        ("C1 compressor r8d512", spd(c1r, "r8d512"), PAL[1]),
        ("C1 compressor r8d128", spd(c1r, "r8d128"), PAL[2]),
        ("I1 indexer bf16-KV", [so["I1_indexer_bf16kv_drop_unused_alloc"]["speedup_median_old_over_new"][f"M{M}"] for M in MS], PAL[3]),
        ("I2 bf16-KV vs fp32-KV", [so["I2_indexer_bf16kv_storage_contract"]["speedup_median_bf16kv_over_fp32kv"][f"M{M}"] for M in MS], PAL[4]),
    ]
    grouped_bars(os.path.join(a.out_dir, "opt_speedup.svg"),
                 "Standalone optimization: baseline -> optimized speedup (kept wins)",
                 "speedup = old/new; 1.0 = no change. Paired 3-process medians, EMR OMP64. jobs 384621/384675.",
                 [f"M={M}" for M in MS],
                 series3, "speedup (x)", (0.0, 1.8), logy=False, fmt="{:.1f}")
    print("wrote 3 SVGs ->", a.out_dir)


if __name__ == "__main__":
    main()
