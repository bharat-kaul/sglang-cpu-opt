"""Bind GPU replay to completed, production-reproducing CPU capture runs."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(root):
    jobs = [(1175, 14, "baseline", 398194), (1175, 14, "shared", 398192),
            (1239, 15, "baseline", 398193), (1239, 15, "shared", 398195)]
    captures = []
    reproduction = []
    trigger = None
    for q, shard, arm, job in jobs:
        case = root / f"q{q}"
        plan = json.loads((case / "capture_plan.json").read_text())
        run = case / f"{arm}_{job}"
        generation = json.loads((run / "gens.json").read_text())
        expected_ids = json.loads((case / "question_ids.json").read_text())
        rows = generation["gens"]
        if [r["q"] for r in rows] != expected_ids:
            raise RuntimeError("production batch membership mismatch")
        if arm == "baseline":
            old_path = Path(f"/scratch/bkaul/dsv4_clamp_gens_397690_{shard}.json")
        else:
            paths = list(Path(f"/scratch/bkaul/dsv4_residual_fullacc_20261105/shard_{shard}").glob("shared_*/gens.json"))
            if len(paths) != 1:
                raise RuntimeError("ambiguous shared-ON reference")
            old_path = paths[0]
        expected_hash = plan["historical_files"][str(old_path)]
        if hashlib.sha256(old_path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError("historical comparator changed")
        old = {r["q"]: r for r in json.loads(old_path.read_text())["gens"]}
        if any(r["text"] != old[r["q"]]["text"] or r["pred"] != old[r["q"]]["pred"] for r in rows):
            raise RuntimeError(f"capture run does not reproduce original batch outputs: {job}")
        target = next(r for r in rows if r["q"] == q)
        prefix = plan["common_generated_ids"]
        if target["output_token_ids"][:len(prefix)] != prefix:
            raise RuntimeError("fresh native tokens do not reproduce the proposed common prefix")
        runtime = json.loads((run / "runtime.json").read_text())
        if runtime["chunked_prefill_size"] != 512:
            raise RuntimeError("production prefill setting changed")
        replay = json.loads((run / "shared_capture/replay.json").read_text())
        expected = {(d, layer) for d in plan["decisions"] for layer in range(43)}
        actual = [(r["decision"], r["layer"]) for r in replay["rows"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise RuntimeError("native replay coverage incomplete or duplicated")
        for decision, layer in sorted(expected):
            path = run / "shared_capture" / f"d{decision}.L{layer}.pt"
            if not path.with_suffix(".replay.pt").is_file():
                raise RuntimeError("missing native CPU paired outputs")
            captures.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        reproduction.append({"q": q, "job": job, "arm": arm, "matched_texts": len(rows),
                             "pred": target["pred"], "gold": target["gold"],
                             "fresh_decision_token": target["output_token_ids"][len(prefix)]})
        if trigger is None:
            trigger = plan["prompt_ids"]
    result = {"captures": captures, "trigger_ids": trigger, "reproduction": reproduction}
    (root / "gpu_manifest.json").write_text(json.dumps(result, indent=2))
    print("REGRESSION_CPU_REPRODUCTION", json.dumps(reproduction), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    prepare(parser.parse_args().root)
