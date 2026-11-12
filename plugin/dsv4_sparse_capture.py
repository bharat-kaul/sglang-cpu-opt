"""Bounded read-only capture at the actual FP32 serving sparse wrapper."""

import functools
import hashlib
import json
from pathlib import Path

import torch


LAYERS = (0, 2, 3)  # SWA, C4, C128 in the pinned Flash configuration.


class SparseCapture:
    def __init__(self, plan_path, output):
        self.plan = json.loads(Path(plan_path).read_text())
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.progress = {}
        self.seen = set()
        self.pending = []
        self.record = None

    def begin(self, record):
        if self.pending:
            raise RuntimeError("unflushed sparse captures")
        self.record = record
        case = record["case"]
        if record["phase"] == "prefill":
            self.progress[case] = self.progress.get(case, 0) + record["tokens"]

    def wrap(self, fn, get_layer):
        @functools.wraps(fn)
        def wrapped(q, kv, mask, sink, scale, torch_fn):
            record = self.record
            layer = get_layer()
            point = self.plan.get(record["case"]) if record is not None else None
            selected = point is not None and layer in LAYERS
            if selected:
                selected = (
                    self.progress[record["case"]] == point["batch"] * point["length"]
                    if record["phase"] == "prefill"
                    else record["tokens"] == point["batch"]
                    and record["seq_lens"] == [point["length"] + 1] * point["batch"]
                )
            key = None if record is None else (record["case"], record["phase"], layer)
            selected = selected and key not in self.seen
            if selected:
                inputs = {name: t.detach().clone() for name, t in
                          (("q", q), ("kv", kv), ("mask", mask), ("sink", sink))}
            result = fn(q, kv, mask, sink, scale, torch_fn)
            if selected:
                self.seen.add(key)
                self.pending.append({
                    "identity": {k: record[k] for k in
                                 ("job", "pid", "sequence", "case", "phase", "tokens",
                                  "seq_lens", "threads_before", "affinity")},
                    "layer": layer, "scale": float(scale), **inputs,
                    "output": result.detach().clone(),
                })
            return result
        return wrapped

    def flush(self):
        files = []
        for sample in self.pending:
            ident = sample["identity"]
            path = self.output / f'{ident["pid"]}-{ident["sequence"]}-L{sample["layer"]}.pt'
            with path.open("xb") as stream:
                torch.save(sample, stream)
            files.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "layer": sample["layer"]})
        self.pending.clear()
        return files
