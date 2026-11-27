"""Export paired EMR caller measurements, then render the committed CSV."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

data = Path(__file__).resolve().parents[2] / "docs/plots/glm5"
table = data / "implementation_callers.csv"


def export(source):
    def load(lane):
        return json.loads((source / f"implemented-{lane}.json").read_text())

    rows = []
    def add(name, job, old, new, ratios):
        assert len(ratios) == 3 and all(0 < n < float("inf") for n in [old, new, *ratios])
        rows.append((name, job, old, new, *ratios))

    kda = next(r for r in load("kda")["records"] if r["job"] == "409937")
    cell = next(c for c in kda["cells"] if c["case"] == "64/five")["heat"]["cold"]
    add("KDA prefill | M32, five fresh, flushed", kda["job"], cell["selected_ms"],
        cell["candidate_ms"], cell["paired_speedups"])
    attention = load("attention")["timings"]
    for key, label in [
        ("L3 indexer dense2048", "Pooled indexer | M32, S2048, warm"),
        ("Radix fresh M32", "Fresh RadixAttention | M32, warm"),
        ("L3 indexer masked secondary local scope", "Index projection in caller | M32, all padded, warm"),
        ("L3 indexer all-close secondary local scope", "Ring update in caller | M32, all close, warm"),
    ]:
        t = attention[key]
        add(label, t["job"], t["median_ms"]["selected"], t["median_ms"]["integrated"],
            t["ratios"]["selected/integrated"]["process_ratios"])
    head = load("runtime")
    t = next(x for x in head["timings"] if x["m"] == x["k"] == 64 and x["cold"])
    add("Output head | M64, all selected, flushed", "/".join(x["job"] for x in head["evidence"]),
        t["previous_ms"], t["implemented_ms"], t["ratios"])
    expert = load("experts")["selected"]
    t = next(x for x in expert["caller_headline"] if x["distribution"] == "repeat" and x["cold"])
    add("mHC pre in FFN | M8, repeated routes, flushed", expert["job"], t["selected_ms"],
        t["candidate_ms"], [t["paired_range"][0], t["median_ratio"], t["paired_range"][1]])
    with table.open("w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["caller_condition","job","previous_ms","implemented_ms","ratio_low","ratio_median","ratio_high"])
        for name, job, old, new, *ratios in rows:
            w.writerow([name, job, old, new, *sorted(ratios)])


def render():
    with table.open() as f:
        rows = list(csv.DictReader(f))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="418">',
           '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#182936">']
    def text(x,y,value,size):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(value)}</text>')
    text(20,34,"Implemented caller gains on EMR",26)
    text(20,58,"Previous selected caller / implemented caller. Dots: paired medians. Lines: three-process ranges.",16)
    scale = lambda ratio: 480 + (ratio-1)*110
    for tick in range(1,6):
        x = scale(tick)
        svg.append(f'<path d="M{x} 85v290" stroke="#d8dde0"/>')
        text(x-8,399,f"{tick}×",15)
    for i,r in enumerate(rows):
        y = 111+i*39
        lo,mid,hi = (float(r[k]) for k in ("ratio_low","ratio_median","ratio_high"))
        text(20,y+5,r["caller_condition"],15)
        svg.append(f'<path d="M{scale(lo)} {y}H{scale(hi)}" stroke="#126871" stroke-width="4"/>')
        svg.append(f'<circle cx="{scale(mid)}" cy="{y}" r="6" fill="#126871"/>')
        text(scale(hi)+10,y+5,f"{mid:.2f}×",15)
    svg.append("</g></svg>")
    subprocess.run(["rsvg-convert","-o",str(data/"implementation_callers_v1.png")],
                   input="".join(svg).encode(),check=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        export(Path(sys.argv[1]))
    render()
