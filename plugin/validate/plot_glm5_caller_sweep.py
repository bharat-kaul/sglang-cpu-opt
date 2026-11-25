"""Render cold, full-width EMR caller sweeps from the checked-in timing table."""
import csv
import math
from collections import defaultdict
from statistics import median

from plot_glm5_journey import DATA, render


def main():
    panels = defaultdict(lambda: defaultdict(list))
    coords = set()
    with (DATA/"full_width_sweep.csv").open() as f:
        for r in csv.DictReader(f):
            op, m = r["caller"], int(r["M"])
            if (op,m) in coords:
                raise ValueError(f"duplicate {op} at {m}")
            coords.add((op,m))
            samples = [[float(r[f"{k}{i}_ns"]) for i in range(3)] for k in ("ref","native")]
            if not all(math.isfinite(x) and x>0 for a in samples for x in a):
                raise ValueError("invalid process median")
            panels[op]["native"].append((m, median(samples[0])/median(samples[1])))
    if coords != {(op,m) for op in ("pre","post","moe","ffn") for m in range(1,65)}:
        raise ValueError("incomplete cold sweep")
    render("full_width_callers", panels, "cold")


if __name__=="__main__":
    main()
