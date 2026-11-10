"""Compare native shared outputs on exactly matching CPU-captured inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def metrics(a, b, row):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError("shared output shape/dtype mismatch")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("nonfinite shared comparison")
    delta = a.float() - b.float()
    selected = delta[row].norm().item()
    norm = b[row].float().norm().item()
    return {"max_abs": delta.abs().max().item(), "target_l2": selected,
            "target_reference_l2": norm, "target_rel_l2": selected / norm if norm else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("job")
    args = parser.parse_args()
    root, job = args.root, args.job
    manifest_path = root / "gpu_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    directory = root / f"gpu_{job}"
    manifests = [json.loads((directory / f"rank{r}/complete.json").read_text()) for r in range(2)]
    for r, data in enumerate(manifests):
        if data["rank"] != r or data["job"] != job or data["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise RuntimeError("GPU manifest identity mismatch")
    records = []
    for item in manifest["captures"]:
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("evaluated CPU artifact changed")
        source = torch.load(path, weights_only=False)
        cpu = torch.load(path.with_suffix(".replay.pt"), weights_only=False)
        native_arm = "on" if source["shared_clamp"] == "1" else "off"
        if not torch.equal(cpu[f"{native_arm}_combined"], source["combined"]):
            raise RuntimeError("CPU replay no longer matches evaluated native output")
        name = f"q{source['q']}.cpu{source['job']}.d{source['decision']}.L{source['layer']}.pt"
        ranks = [torch.load(directory / f"rank{r}" / name, weights_only=False) for r in range(2)]
        for r, gpu in enumerate(ranks):
            if gpu["source_sha256"] != item["sha256"] or gpu["rank"] != r or gpu["job"] != job:
                raise RuntimeError("GPU sample source identity mismatch")
            if not torch.equal(gpu["input"], source["hidden"]):
                raise RuntimeError("GPU did not consume captured CPU operands")
            for variant in ("on", "repeat", "inactive"):
                if not torch.equal(gpu["variants"][variant]["output"], ranks[0]["variants"][variant]["output"]):
                    raise RuntimeError("reduced GPU output differs between ranks")
            if not torch.equal(gpu["variants"]["on"]["output"], gpu["variants"]["repeat"]["output"]):
                raise RuntimeError("GPU repeat mismatch")
        gpu = ranks[0]["variants"]
        row = source["target_row"]
        records.append({
            "q": source["q"], "cpu_job": source["job"], "layer": source["layer"],
            "decision": source["decision"], "target_row": row,
            "cpu_on_gpu_on": metrics(cpu["on_direct"], gpu["on"]["output"], row),
            "cpu_off_gpu_on": metrics(cpu["off_direct"], gpu["on"]["output"], row),
            "cpu_off_gpu_inactive": metrics(cpu["off_direct"], gpu["inactive"]["output"], row),
        })
    result = {"gpu_job": job, "n": len(records), "records": records,
              "scope": "Native CPU/GPU shared output after native dtype storage and GPU TP reduction; not model accuracy"}
    (directory / "comparison.json").write_text(json.dumps(result, indent=2))
    print("SHARED_REGRESSION_COMPARISON_COMPLETE", len(records), flush=True)
    for q in (1175, 1239):
        candidates = [r for r in records if r["q"] == q and r["decision"] > 0]
        worst = sorted(candidates, key=lambda r: r["cpu_on_gpu_on"]["target_rel_l2"] or 0, reverse=True)[:5]
        print("LARGEST_SHARED_DIFFERENCES", q, json.dumps(worst), flush=True)


if __name__ == "__main__":
    main()
