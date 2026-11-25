"""Extract paired speedups from completed GLM campaign evidence (argument: probe directory)."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import median

from plot_glm5_journey import DATA

root = Path(sys.argv[1])
sources = {}
points = []


def read(path, lines=False):
    p = root / path
    raw = p.read_bytes()
    sources[path] = hashlib.sha256(raw).hexdigest()
    return list(map(json.loads, raw.splitlines())) if lines else json.loads(raw)


def point(group, panel, series, m, mode, speedup, job):
    points.append((group, panel, series, m, "cold" if mode == "cache_flushed" else mode, speedup, job))


for job in (406471,406472,406473):
    for r in read(f"fused-activation-{job}/summary.json")["rows"]:
        point("activation", f'{job} | width {r["I"]}', r["variant"], r["M"], r["method"], r["speedup_vs_reference"], job)
for job in (406480,406483):
    for r in read(f"native-linear-{job}/summary.json")["rows"]:
        if r["variant"] != "reference":
            m,n,k = r["shape"]
            for mode,v in r["speedup_vs_reference"].items():
                point("linear",f"{job} | {r['op']} [{n}, {k}]",r["variant"],m,mode,v,job)
for experiment,job in (("native",406495),("norm",406496),("commit",406497),("full",406498),("norm-fusion",406518)):
    for r in read(f"kda-{experiment}-{job}/summary.json")["rows"]:
        for base in ("decode_contiguous","decode_strided","gated_norm"):
            if r["op"].startswith(base+"_"):
                for mode,t in r["timing"].items():
                    point("decode_norm",f"{job} | {base}",r["op"][len(base)+1:],r["M"],mode,t["speedup_vs_reference"],job)
for r in read("prefill-timing-407379/summary.json")["rows"]:
    if r["path"]!="reference":
        point("prefill",r["boundary"],r["path"],r["M"],r["mode"],r["speedup"],407379)
times = {}
for i in range(3):
    header,*rows,end=read(f"wave-407549/rep-{i}.jsonl",True)
    assert header["header"]["job"]=="407549" and end=={"complete":True} and len(rows)==42
    for r in rows:
        for mode,t in r["ns"].items():
            times.setdefault((r["boundary"],r["M"],r["path"],mode),[]).append(median(t))
for (boundary,m,path,mode),ts in times.items():
    if path=="selected":
        ref=times[boundary,m,"reference",mode]
        assert len(ref)==len(ts)==3
        point("holdout",boundary,path,m,mode,median(ref)/median(ts),407549)
for r in read("native-wave/summary-407573.json")["rows"]:
    point("family_candidates",r["op"],"native",r["M"],"cold" if r["cold"] else "warm",r["speedup"],r["job"])
for r in read("native-wave/summary-407602.json")["rows"]:
    point("caller_comparisons",r["op"],"candidate",r["M"],"cold" if r["cold"] else "warm",r["speedup"],r["job"])

DATA.mkdir(parents=True,exist_ok=True)
with (DATA/"measurements.csv").open("w",newline="") as f:
    writer=csv.writer(f,lineterminator="\n")
    writer.writerow(("group","panel","series","M","cache","speedup","job"))
    writer.writerows(points)
(DATA/"sources.json").write_text(json.dumps(sources,indent=2,sort_keys=True)+"\n")
