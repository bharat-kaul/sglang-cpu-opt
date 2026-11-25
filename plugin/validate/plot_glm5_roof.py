"""Plot cold subgraph latencies against explicitly provisional EMR references."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

from plot_glm5_journey import DATA, text

OPS = "router combine mixed_norm absorb rotate compress tail pack attend contract allocate".split()
MS = (1, 2, 4, 8, 16, 32, 64)
CSV = DATA / "roofline_cold.csv"


def work(op, m):
    # Unique useful input + final output bytes. No intermediate or cache traffic.
    byte_counts = {
        "router": 4*288*(4096+1) + m*(2*4096+8*8),
        "combine": 2*m*4096*(8+1+1),
        "mixed_norm": m*(4*4096*4+4*4+2*4096)+2*4096,
        "absorb": 2*64*(256*512+m*(256+512)),
        "rotate": m*32*(128*3+4),
        "compress": m*(4*128*6+128+4)+4*128*4,
        "tail": m*(8+2051*4)+4*sum(min(2048,(2046+i%8)//4*4) for i in range(m)),
        "pack": m*(512*2+528),
        "attend": 2*512*(257+2*m*64),
        "contract": 2*m*(16384+4096),
        "allocate": 8*(4*m+sum((62+i%4)%64 == 0 for i in range(m))),
    }
    gemms = {"router": (2*m*4096*288, "fp32"), "absorb": (2*m*64*256*512, "bf16"),
             "attend": (4*m*64*257*512, "fp32")}
    return byte_counts[op], *gemms.get(op, (0, "none"))


def pinned(root, path, digest):
    raw = (root / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Unexpected benchmark artifact: {path}")
    return json.loads(raw)


def export(root):
    wave = pinned(root, "native-wave/summary-407573.json",
                  "8556e7fc6ee414fc5a46c8579a9cc96c6576d6b46d41f0225fa48eb20c3d0fd7")
    rates = pinned(root, "primitive-baseline-406478/summary.json",
                   "c5b7be4f837bef774f5ef6dd34ee6a94d7ae8ea6b54142eb8cd64c021c0a24f2")["calibration"]
    rows = []
    for r in wave["rows"]:
        if not r["cold"]:
            continue
        op, m = r["op"], r["M"]
        size, flops, dtype = work(op, m)
        bw = rates["triad_algorithmic_gbs"]["median"]
        compute = rates[dtype+"_gflops"]["median"] if flops else 0
        floor = max(size/bw, flops/compute if flops else 0)
        ref, native = (median(r[k+"_medians_ns"]) for k in ("ref", "native"))
        rows.append([op, m, r["job"], size, flops, dtype, bw, compute, floor, ref, native])
    if len(rows) != 77 or {(r[0], r[1]) for r in rows} != {(op,m) for op in OPS for m in MS}:
        raise ValueError("Incomplete or duplicated cold timing coordinates")
    with CSV.open("w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow("op M job logical_bytes gemm_flops dtype triad_GBs library_Gflops model_ns ref_ns native_ns".split())
        out.writerows(rows)


def render():
    with CSV.open() as f:
        points = {(r["op"], int(r["M"])): r for r in csv.DictReader(f)}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1130" font-family="DejaVu Sans">',
           '<rect width="100%" height="100%" fill="white"/>',
           text(16, 25, "EMR cold latency vs provisional roofline model", 21),
           text(16, 48, "Blue: CPU reference     Orange: native candidate     Gray dashed: model time (not a qualified ceiling)", 15),
           text(16, 69, "Batch M on x; latency in microseconds on log y (lower is faster). Eight non-GEMM panels show the byte term only.", 13)]
    for i, op in enumerate(OPS):
        left, top = 64 + i%2*500, 112 + i//2*173
        svg.append(text(left, top-10, op, 15))
        keys = ("ref_ns", "native_ns", "model_ns")
        series = [[float(points[op,m][k])/1000 for m in MS] for k in keys]
        low = math.floor(math.log10(min(map(min,series))))
        high = math.ceil(math.log10(max(map(max,series))))
        def y(log_us):
            return top + 106*(high-log_us)/(high-low)
        for e in range(low, high+1):
            svg += [text(left-4, y(e)+4, f"{10**e:g}", 11, "end"),
                    f'<path d="M{left} {y(e)}h390" stroke="#e2e6ea"/>']
        for j,m in enumerate(MS):
            svg.append(text(left+j*65,top+124,m,11,"middle"))
        for j,(seq,color) in enumerate(zip(series, ("#3068af","#bb4e12","#575f68"))):
            xy = [(left+k*65,y(math.log10(v))) for k,v in enumerate(seq)]
            coords = " ".join(f"{x:.2f},{v:.2f}" for x,v in xy)
            dash = ' stroke-dasharray="5 4"' if j == 2 else ''
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
            if j != 2:
                svg.extend(f'<circle cx="{x}" cy="{v}" r="2.4" fill="{color}"/>' for x,v in xy)
    subprocess.run(["rsvg-convert","-o",str(DATA/"family_roofline_cold_v1.png")],
                   input=("".join(svg)+"</svg>").encode(),check=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        export(Path(sys.argv[1]))
    render()
