import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsv4_sparse_capture import SparseCapture
import replay_serving_sparse as replay
from replay_serving_sparse import compact_pool, replay_sample, summarize_boundaries, validate_sample
from test_serving_screen import event as worker_event


def event(phase, sequence, tokens, length):
    return {"job": "test", "pid": 7, "sequence": sequence, "case": "point", "phase": phase,
            "tokens": tokens, "seq_lens": [length], "threads_before": 32, "affinity": list(range(32))}


def operands():
    q = torch.ones(2, 64, 512)
    kv = torch.arange(5 * 512, dtype=torch.float32).reshape(5, 512) / 1000
    mask = torch.tensor([[False, True, False, True, False],
                         [False, False, False, True, False]])
    sink = -torch.ones(64)
    return q, kv, mask, sink


def fake_native(q, kv, mask, sink, scale):
    return (mask.float() @ kv).unsqueeze(1).expand_as(q).clone()


class SparseReplayTests(unittest.TestCase):
    def test_capture_is_bounded_and_owns_input_output_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"point": {"batch": 1, "length": 4}}))
            capture = SparseCapture(path, Path(tmp) / "captures")
            q, kv, mask, sink = operands()
            wrapped = capture.wrap(lambda q, k, m, s, scale, oracle: fake_native(q, k, m, s, scale),
                                   lambda: 2)
            for i, length in enumerate((2, 4)):
                capture.begin(event("prefill", i, 2, length))
                result = wrapped(q, kv, mask, sink, 0.1, None)
                files = capture.flush()
                self.assertEqual(len(files), i)
            sample = torch.load(files[0]["file"], weights_only=True)
            validate_sample(sample, event("prefill", 1, 2, 4), 2)
            expected = result.clone()
            result.zero_()
            kv.zero_()
            self.assertTrue(torch.equal(sample["output"], expected))
            self.assertGreater(float(sample["kv"].abs().sum()), 0)
            capture.begin(event("prefill", 2, 2, 6))
            wrapped(q, kv, mask, sink, 0.1, None)
            self.assertEqual(capture.flush(), [])

    def test_replay_preserves_order_and_checks_outputs(self):
        q, kv, mask, sink = operands()
        k, m = compact_pool(kv, mask)
        self.assertTrue(torch.equal(k, kv[[1, 3]]))
        self.assertTrue(torch.equal(m, mask[:, [1, 3]]))
        sample = {"q": q, "kv": kv, "mask": mask, "sink": sink, "scale": 0.1,
                  "output": fake_native(q, kv, mask, sink, 0.1),
                  "identity": event("prefill", 1, 2, 4), "layer": 2}
        validate_sample(sample, sample["identity"], 2)
        report = replay_sample(sample, fake_native)
        self.assertEqual(report["union_visible_rows"], 2)
        self.assertTrue(all(len(v) == 3 for v in report["samples_ns"].values()))
        for field in ("q", "kv", "sink", "output"):
            bad = copy.deepcopy(sample)
            bad[field].flatten()[0] = float("nan")
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                validate_sample(bad, sample["identity"], 2)
        bad = copy.deepcopy(sample)
        bad["identity"]["tokens"] = 64
        with self.assertRaisesRegex(RuntimeError, "identity"):
            validate_sample(bad, sample["identity"], 2)
        bad = copy.deepcopy(sample)
        bad["output"].add_(1)
        with self.assertRaisesRegex(RuntimeError, "byte mismatch"):
            replay_sample(bad, fake_native)

    def test_capture_and_compilation_forwards_are_excluded(self):
        events = []
        for phase in ("prefill", "decode"):
            for i in range(3):
                e = event(phase, i, 2, 4)
                e.update(forward_ns=100, captures=[{"layer": 2}] if i == 1 else [],
                         rows=[{"op": "test", "calls": 1, "inclusive_ns": 50, "exclusive_ns": 50}])
                events.append(e)
        report = summarize_boundaries(events)["point"]
        self.assertEqual(report["prefill"]["sequences"], [2])
        self.assertEqual(report["decode"]["sequences"], [0, 2])

    def test_replay_entry_rejects_bad_evidence_without_complete_output(self):
        for corruption in (None, "nan", "coordinate", "missing", "hash", "output"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture_dir = root / "worker/captures"
                capture_dir.mkdir(parents=True)
                (root / "inputs.json").write_text("{}")
                identity = {"job": "test", "base_commit": replay.BASE_COMMIT, "sources": {},
                            "installed_sources_and_library": {},
                            "prompt_sha256": replay.digest(root / "inputs.json"),
                            "cases": [{"batch": 1, "input_len": 4}]}
                for name in ("identity.json", "complete.json"):
                    (root / name).write_text(json.dumps(identity))
                events = []
                for i, phase in enumerate(["prefill", "prefill"] + ["decode"] * 4):
                    e = worker_event("b1-s4-meaningful", True, phase, i)
                    e["tokens"] = 2 if phase == "prefill" else 1
                    e["seq_lens"] = [2 if i == 0 else i + 3]
                    e["affinity"] = list(range(32))
                    e["captures"] = []
                    if i in (1, 2):
                        for layer in (0, 2, 3):
                            q, kv, mask, sink = operands()
                            q, mask = q[:e["tokens"]], mask[:e["tokens"]]
                            sample = {
                                "identity": {k: e[k] for k in (
                                    "job", "pid", "sequence", "case", "phase", "tokens",
                                    "seq_lens", "threads_before", "affinity")},
                                "layer": layer, "scale": 0.1, "q": q, "kv": kv,
                                "mask": mask, "sink": sink,
                                "output": fake_native(q, kv, mask, sink, 0.1),
                            }
                            if corruption == "nan":
                                sample["output"][0, 0, 0] = float("nan")
                            elif corruption == "coordinate":
                                sample["identity"]["tokens"] = 64
                            elif corruption == "output":
                                sample["output"].add_(1)
                            path = capture_dir / f"{i}-{layer}.pt"
                            torch.save(sample, path)
                            e["captures"].append({"file": str(path), "layer": layer,
                                                  "sha256": replay.digest(path)})
                    events.append(e)
                if corruption == "missing":
                    events[1]["captures"].pop()
                if corruption == "hash":
                    events[1]["captures"][0]["sha256"] = "0" * 64
                (root / "worker/events.7.jsonl").write_text(
                    "".join(json.dumps(e) + "\n" for e in events))
                native = types.SimpleNamespace(sparse_attend_masked=fake_native, __file__=__file__)
                plugin = types.ModuleType("intel_cpu_models")
                plugin._dsa_ckernels = types.SimpleNamespace(_mod=lambda *a: native)
                with patch.dict(os.environ, {"SLURM_JOB_ID": "test"}), \
                        patch.dict(sys.modules, {"intel_cpu_models": plugin}), \
                        patch.object(replay, "MEANINGFUL_POINTS", [(1, 4)]), \
                        patch.object(torch, "set_num_threads"):
                    if corruption:
                        with self.assertRaises(RuntimeError):
                            replay.run(root)
                        self.assertFalse((root / "replay.json").exists())
                        self.assertFalse((root / "replay-partial.json").exists())
                    else:
                        replay.run(root)
                        report = json.loads((root / "replay.json").read_text())
                        self.assertEqual(len(report["sparse"]), 6)


if __name__ == "__main__":
    unittest.main()
