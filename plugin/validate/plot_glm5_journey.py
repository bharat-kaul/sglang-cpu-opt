"""Render the checked-in GLM timeline CSV to GitHub-friendly PNG small multiples."""
import csv
import html
import math
import subprocess
from collections import defaultdict
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
DATA = DOCS / "plots/glm5"
COLORS = ["#087e83", "#bb4e12", "#6644ae", "#3068af", "#995078"]


def text(x, y, value, size=13, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="#23364b">{html.escape(str(value))}</text>')


def render(group, panels, mode):
    height = 52 + 215 * math.ceil(len(panels) / 2)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" '
           'font-family="DejaVu Sans,sans-serif">', '<rect width="100%" height="100%" fill="white"/>',
           text(16, 23, f"{group.replace('_',' ').title()}   |   {mode} speedup vs matching CPU reference", 18),
           text(16, 43, "Horizontal axis: batch M (log2)    •    Vertical axis: speedup (log2); above 1× is faster. Points are measurements. ", 12)]
    for i, (panel, series) in enumerate(panels.items()):
        left, top = 54 + i % 2 * 500, 83 + i // 2 * 215
        values = [v for points in series.values() for _, v in points]
        low = min(-0.05, math.log2(min(values))); high = max(0.05, math.log2(max(values)))
        def y(v): return top + 96 * (high - math.log2(v)) / (high - low)
        def x(m): return left + 412 * math.log2(m) / 6
        svg.append(text(left, top-12, panel, 14))
        for v in sorted({1.,min(values),max(values)}):
            if v!=1 and abs(y(v)-y(1))<12: continue
            svg.extend([text(left-4,y(v)+4,f"{v:.2g}×",11,"end"),
                        f'<path d="M{left} {y(v)}h412" stroke="{"#707070" if v==1 else "#dce1e4"}"/>'])
        for m in (1,2,4,8,16,32,64): svg.append(text(x(m),top+113,m,11,"middle"))
        for j,(label,points) in enumerate(series.items()):
            color=COLORS[j%len(COLORS)]
            coords=" ".join(f"{x(m):.2f},{y(v):.2f}" for m,v in sorted(points))
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{coords}"/>')
            for m,v in points: svg.append(f'<circle cx="{x(m):.2f}" cy="{y(v):.2f}" r="2.1" fill="{color}"/>')
            tx,ty=left+(j%2)*203,top+132+(j//2)*15
            svg.extend([f'<path d="M{tx} {ty-4}h14" stroke="{color}" stroke-width="3"/>',text(tx+18,ty,label,11)])
    svg.append("</svg>")
    subprocess.run(["rsvg-convert","-o",str(DATA/f"{group}_{mode}_v1.png")],
                   input="".join(svg).encode(),check=True)


if __name__ == "__main__":
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    seen=set()
    with (DATA/"measurements.csv").open() as f:
        for row in csv.DictReader(f):
            group,mode,panel,series=(row[k] for k in ("group","cache","panel","series"))
            m=int(row["M"]); v=float(row["speedup"])
            key=(group,mode,panel,series,m)
            if key in seen or not 1<=m<=64 or not math.isfinite(v) or v<=0:
                raise ValueError(key)
            seen.add(key);groups[group,mode][panel][series].append((m,v))
    for (group,mode),panels in groups.items(): render(group,panels,mode)
