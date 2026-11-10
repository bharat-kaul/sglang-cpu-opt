"""Native CPU shared off/on replay on saved production operands."""

import argparse
import json
from pathlib import Path
import runpy

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    expected = {(d, layer) for d in plan["decisions"] for layer in range(43)}
    files = list(args.directory.glob("d*.L*.pt"))
    actual = {(int(p.name.split(".")[0][1:]), int(p.name.split(".")[1][1:])) for p in files}
    if actual != expected:
        raise RuntimeError(f"capture coverage mismatch: missing={expected-actual}, extra={actual-expected}")
    loader = Path(__file__).parents[1] / "intel_cpu_models/_dsv4_shared_clamp.py"
    so = runpy.run_path(str(loader))["load"]()
    torch.set_num_threads(8)
    rows = []
    for path in sorted(files):
        data = torch.load(path, weights_only=False)
        weights = torch.load(args.directory / f"weights.L{data['layer']}.pt", weights_only=False)
        limit = weights["limit"]
        results = {}
        for arm, value in (("off", -1.0), ("on", limit)):
            so.set_shared_clamp(value)
            for boundary, routed, factor in (("combined", data["routed"], data["factor"]),
                                             ("direct", None, None)):
                output = so.shared_experts(
                    data["hidden"].clone(), weights["w1"], weights["w2"], routed,
                    factor, False, False, True, weights["s1"], weights["s2"], weights["block"], True,
                )
                if not torch.isfinite(output).all():
                    raise RuntimeError("nonfinite shared replay")
                results[f"{arm}_{boundary}"] = output.clone()
        native_arm = "on" if data["shared_clamp"] == "1" else "off"
        if not torch.equal(results[f"{native_arm}_combined"], data["combined"]):
            raise RuntimeError(f"native replay does not reproduce captured output: {path}")
        torch.save(results, path.with_suffix(".replay.pt"))
        rows.append({"file": path.name, "native_equal": True,
                     "target_row": data["target_row"], "layer": data["layer"], "decision": data["decision"]})
    (args.directory / "replay.json").write_text(json.dumps({"q": plan["q"], "rows": rows}, indent=2))
    print(f"SHARED_REGRESSION_REPLAY_COMPLETE q={plan['q']} captures={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
