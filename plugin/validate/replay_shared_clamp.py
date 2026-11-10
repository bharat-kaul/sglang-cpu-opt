#!/usr/bin/env python3
"""Replay the SHARED-expert clamp on captured operands (reviewer step 1-2, CPU-native, fail-closed).

For every capture (case x layer), drive the PRIVATE shared-clamp donor on the EXACT captured operands
(packed fp8 weights preserved) two ways:
  OFF (limit<0): must reproduce the captured production `out_cpu_combined_noclamp` BIT-EXACT
                 (reviewer requirement: inactive clamp == production, including the routed/shared add).
  ON  (limit=10): clamp the shared SwiGLU; the ON-vs-OFF delta is attributable ONLY to the clamp.

Reports, per case/layer, whether the clamp changes the output at all and — critically — at the DECISION
ROW (the last prompt token, whose hidden drives the next-token logit). If the decision-row output is
unchanged for every failing case, the missing shared clamp CANNOT explain those failures (lead dead).

  python replay_shared_clamp.py --caps "/scratch/bkaul/dsv4_shared_cap/cap_397780.*.pt" \
      --frozen /scratch/bkaul/dsv4_shared_cap/frozen_targets.json --out /path/shared_clamp_replay.json
"""
import argparse
import glob
import json
from pathlib import Path
import runpy

import torch

LIMIT = 10.0


def _metrics(a, b):
    d = (a.float() - b.float())
    return {"max_abs": d.abs().max().item(),
            "l2": d.norm().item(),
            "rel_l2": (d.norm() / b.float().norm().clamp_min(1e-20)).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True, help="glob of capture .pt files")
    ap.add_argument("--frozen", required=True, help="frozen_targets.json (q <-> sha12 + failing/control tag)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import hashlib
    frozen = json.load(open(args.frozen))["items"]
    sha2q = {hashlib.sha256(json.dumps([int(x) for x in it["token_ids"]]).encode()).hexdigest()[:12]: it["q"]
             for it in frozen}
    FAILING = {5, 29, 81, 146, 197, 244, 247, 417, 435, 502, 769, 893, 924, 995, 1067, 1191, 1229}
    CONTROLS = {302, 562, 640, 693, 1175, 1271}

    donor = Path(__file__).parents[1] / "intel_cpu_models/_dsv4_shared_clamp.py"
    so = runpy.run_path(str(donor))["load"]()

    files = sorted(glob.glob(args.caps))
    if not files:
        raise SystemExit(f"no captures matched {args.caps}")
    rows = []
    bitexact_fail = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        hid = d["hidden"].to(torch.bfloat16).contiguous()
        fe = d["fused_experts_out"].to(torch.bfloat16).contiguous()
        kw = (d["gate_up_weight"], d["down_weight"], fe, float(d["routed_scaling_factor"]),
              False, bool(d["is_int8"]), bool(d["is_fp8"]), d["w1_scale"], d["w2_scale"],
              list(d["block_size"]), True)
        so.set_shared_clamp(-1.0)
        out_off = so.shared_experts(hid.clone(), *kw).float()
        so.set_shared_clamp(LIMIT)
        out_on = so.shared_experts(hid.clone(), *kw).float()

        ref = d["out_cpu_combined_noclamp"].float()
        off_bitexact = torch.equal(out_off, ref)
        if not off_bitexact:
            bitexact_fail.append((f, _metrics(out_off, ref)))

        dr = int(d["decision_row"])
        delta = out_on - out_off
        dr_changed = not torch.equal(out_on[dr], out_off[dr])
        rows_changed = int((delta.abs().amax(dim=1) > 0).sum().item())
        sha12 = Path(f).name.split(".")[1]
        q = sha2q.get(sha12)
        rows.append({
            "file": Path(f).name, "q": q, "layer": int(d["layer_id"]),
            "cohort": ("failing" if q in FAILING else "control" if q in CONTROLS else "other"),
            "off_reproduces_production_bitexact": bool(off_bitexact),
            "clamp_changes_output": bool(rows_changed > 0),
            "clamp_changes_decision_row": bool(dr_changed),
            "rows_changed": rows_changed, "n_rows": int(d["n_rows"]),
            "overall": _metrics(out_on, out_off),
            "decision_row": _metrics(out_on[dr:dr + 1], out_off[dr:dr + 1]),
        })
        print(f"[replay] q{q} L{d['layer_id']} off_bitexact={off_bitexact} "
              f"dr_changed={dr_changed} rows_changed={rows_changed}/{d['n_rows']} "
              f"dr_max|Δ|={rows[-1]['decision_row']['max_abs']:.4g}", flush=True)

    if bitexact_fail:
        for f, m in bitexact_fail:
            print(f"[replay] BITEXACT-FAIL {f}: {m}", flush=True)
        raise SystemExit(f"FATAL: {len(bitexact_fail)} clamp-OFF outputs are NOT bit-exact vs production "
                         "(donor not faithful) — fail-closed")

    # Aggregate the decisive question: does the clamp touch the decision row of the FAILING cases?
    def agg(cohort):
        r = [x for x in rows if x["cohort"] == cohort]
        return {"n": len(r),
                "decision_row_changed": sum(x["clamp_changes_decision_row"] for x in r),
                "any_output_changed": sum(x["clamp_changes_output"] for x in r),
                "max_decision_row_abs": max((x["decision_row"]["max_abs"] for x in r), default=0.0)}
    summary = {"limit": LIMIT, "n_captures": len(rows),
               "all_off_bitexact": True,
               "by_cohort": {c: agg(c) for c in ("failing", "control", "other")},
               "by_layer": {}}
    for lid in sorted({x["layer"] for x in rows}):
        r = [x for x in rows if x["layer"] == lid]
        summary["by_layer"][lid] = {"n": len(r),
                                    "decision_row_changed": sum(x["clamp_changes_decision_row"] for x in r),
                                    "max_decision_row_abs": max(x["decision_row"]["max_abs"] for x in r)}
    out = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("SHARED_CLAMP_REPLAY_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
