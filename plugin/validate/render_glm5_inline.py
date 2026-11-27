"""Regenerate image-independent charts in the GLM journey from its published CSVs."""
import json
import math
import re
from collections import defaultdict
from statistics import median

from plot_glm5_overview import CALLERS, DATA, ROWS, one, records


def table(header, rows):
    def line(cells):
        return "| " + " | ".join(str(c).replace("|", r"\|") for c in cells) + " |"
    return "\n".join([line(header), line(["---"] * len(header)), *map(line, rows)])


def overview():
    callers = records(CALLERS)
    models = {r["op"]: float(r["model_ns"]) / 1000 for r in records(DATA / "roofline_cold.csv") if r["M"] == "8"}
    rows = []
    for family, label, screen, op in ROWS:
        r = one(r for r in callers if (r["family"], r["screen"]) == (family, screen))
        a, b = (median(json.loads(r[k])) / 1000 for k in ("baseline_process_ns", "native_process_ns"))
        rows.append([label, f"{a:,.2f}", f"{b:,.2f}", f"{a/b:.2f}×", "▏" + "█" * round(4 * max(0, math.log2(a/b))),
                     f"{op}: {models[op]:.3g}" if op else "—"])
    return table(["Family", "Baseline µs", "Native µs", "Speedup", "Gain (4 blocks = 2×)", "Isolated M8 model µs¹"], rows) + (
        "\n\n¹ The modeled time is for the named **subgraph**, not the caller in the latency columns. "
        "Whole-family rooflines: **0/17 verified**.")


def curves(group, heat):
    data = defaultdict(dict)

    def point(label, m, value):
        m, value = int(m), float(value)
        if m in data[label] or not math.isfinite(value) or value <= 0:
            raise ValueError((group, label, m, value))
        data[label][m] = value

    for r in records(DATA / "measurements.csv"):
        if (r["group"], r["cache"]) == (group, heat):
            point(f"{r['panel']} · {r['series']}", r["M"], r["speedup"])
    if data:
        return data
    cols = {
        "dense_sweep": ("op", "speedup"), "mla_checks": ("op", "speedup"),
        "schedule_trials": ("tag", "paired_speedup"), "fresh_attention": ("cache_save", "paired_median_speedup"),
    }
    name = {"full_width_callers": "full_width_sweep", "family_roofline": "roofline_cold"}.get(group, group)
    for r in records(DATA / f"{name}.csv"):
        if group in cols:
            label, value = cols[group]
            point(f"{label}: {r[label]}", r["M"], r[value])
        elif group == "full_width_callers":
            a, b = (median(float(r[f"{k}{i}_ns"]) for i in range(3)) for k in ("ref", "native"))
            point(r["caller"], r["M"], a / b)
        elif group == "family_roofline":
            for k in ("ref_ns", "native_ns", "model_ns"):
                point(f"{r['op']} · {k[:-3]}", r["M"], float(r[k]) / 1000)
        elif group == "cached_latent":
            for k in ("reference", "native_read"):
                point(f"S={r['S']} vs {k}", r["M"], r[f"speedup_vs_{k}"])
        elif group == "fp8_shapes" and r["cold"] == "True":
            point(f"{r['tag']} · {r['boundary']}", r["m"], r["ref_over_wave"])
        elif group == "pooled_commit" and r["fixture"] == "mixed":
            point(f"round={r['scale_round']} vs {r['comparator']}", r["M"], r["paired_median"])
        elif group == "kda_extend_native" and r["comparison"] == "selected" and r["prefixes"] == "[0]" and r["padding"] == "0":
            m, = json.loads(r["lengths"])
            if 1 <= m <= 64:
                point(f"heads={r['source_heads']}", m, r["median_speedup"])
    if not data:
        raise ValueError(f"No measurements for {group}")
    return data


def chart(path):
    if path.startswith("kernel_overview_"):
        return overview()
    group, heat, _ = path.rsplit("_", 2)
    unit = "µs" if group == "family_roofline" else "×"
    ms = [1, 2, 4, 8, 16, 32, 64]
    rows = []
    for label, points in curves(group, heat).items():
        values = [math.log(points[m]) for m in sorted(points)]
        lo, hi = min(values), max(values)
        trend = "".join("▁▂▃▄▅▆▇█"[round(7 * (v-lo)/(hi-lo)) if hi > lo else 0] for v in values)
        rows.append([label, trend, *(f"{points[m]:.3g}" if m in points else "—" for m in ms)])
    return table([f"Series ({unit})", "Trend by M¹", *(f"M{m}" for m in ms)], rows) + (
        "\n\n¹ One tick per measured batch size, in ascending order; height is log-scaled within each row. "
        "Numeric columns show selected batch sizes.")


if __name__ == "__main__":
    page = DATA.parents[1] / "glm5-kernel-performance-journey.md"
    original = page.read_text()
    start, end = "<!-- glm5-inline -->", "<!-- /glm5-inline -->"
    raw = re.sub(rf"{start}\n.*?\n{end}", lambda m: one(re.findall(r"^!\[.*$", m[0], re.M)), original, flags=re.S)
    def replace(match):
        if match[1].startswith("journeys_"):
            return match[0]
        return (f"{start}\n\n{match[0]}\n\n<details><summary>Show numeric data</summary>\n\n"
                f"{chart(match[1])}\n\n</details>\n\n{end}")
    page.write_text(re.sub(r"^!\[[^\n]*\]\(plots/glm5/([\w]+)\.png\)$", replace, raw, flags=re.M))
