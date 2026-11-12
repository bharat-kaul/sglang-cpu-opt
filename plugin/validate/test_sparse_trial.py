import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import analyze_sparse_trial as analyzer
import bench_long_prefill_sparse as donor
import dsv4_serving_probe as probe
from dsv4_sparse_trial import STATE_MAPS, SparseTrial, shared_chunked
from screen_serving_opportunities import BASE_COMMIT, EXPECTED_ENV, digest, write_json
from test_serving_screen import event


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


class TrialTests(unittest.TestCase):
    def test_candidate_matches_measured_formulation(self):
        torch.manual_seed(7)
        for n in (1, 32, 33):
            q, kv, sink = torch.randn(n, 2, 4), torch.randn(7, 4), torch.randn(2)
            for mask in (torch.zeros(n, 7, dtype=torch.bool), torch.ones(n, 7, dtype=torch.bool),
                         torch.rand(n, 7) > .5):
                args = q, kv, mask, sink, .5
                self.assertTrue(torch.equal(shared_chunked(*args).view(torch.uint8),
                                            donor.shared_chunked(*args).view(torch.uint8)))

    def test_worker_trial_dispatch_reset_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(zip(analyzer.CASES, (False, True, False)))
            write_json(root / "plan.json", plan)
            infra = types.SimpleNamespace(**{key: {"old": 1} for key in STATE_MAPS},
                                          _SEL_HOLDER={"pos_kept": "old"})
            trial = SparseTrial(root / "plan.json", root / "tensors", infra, probe)
            batch = types.SimpleNamespace(positions=torch.tensor([0]), input_ids=torch.tensor([2]),
                                          req_pool_indices=torch.tensor([0]), out_cache_loc=torch.tensor([0]))
            q, kv, mask, sink = torch.randn(1, 2, 4), torch.randn(3, 4), torch.ones(1, 3, dtype=torch.bool), torch.randn(2)
            calls = []

            def original(*args):
                calls.append(1)
                return torch.zeros_like(q)

            wrapped = trial.wrap(original)
            with patch.object(probe, "_active", probe.Recorder()):
                for case in analyzer.CASES:
                    for phase in ("prefill", "decode"):
                        record = {"case": case, "phase": phase, "tokens": 1, "job": "test",
                                  "pid": 7, "sequence": len(calls)}
                        trial.begin(record, batch)
                        self.assertEqual(record["trial"]["state_reset"], phase == "prefill")
                        self.assertTrue(all(not getattr(infra, key) for key in STATE_MAPS))
                        for layer, ratio in ((0, 0), (2, 4), (3, 128)):
                            probe._active.layer, trial.ratio = layer, ratio
                            result = wrapped(q, kv, mask, sink, .5, None)
                            selected = case == analyzer.CASES[1] and phase == "prefill" and ratio == 4
                            self.assertEqual(trial.calls[-1]["candidate"], selected)
                            self.assertTrue(torch.equal(result, shared_chunked(q, kv, mask, sink, .5)
                                                        if selected else torch.zeros_like(q)))
                        with self.assertRaisesRegex(RuntimeError, "incomplete"):
                            trial.finish(types.SimpleNamespace(next_token_logits=torch.ones(1, 4)))
                self.assertEqual(len(calls), 17)
                with patch("dsv4_sparse_trial.shared_chunked", side_effect=ValueError("injected")):
                    trial.begin({"case": analyzer.CASES[1], "phase": "prefill", "tokens": 1}, batch)
                    trial.ratio = 4
                    with self.assertRaises(ValueError):
                        wrapped(q, kv, mask, sink, .5, None)
                    with self.assertRaisesRegex(RuntimeError, "caught fallback"):
                        trial.finish(None)
                trial.case = None
                batch.positions = torch.tensor([1])
                with self.assertRaisesRegex(RuntimeError, "fresh prefill"):
                    trial.begin({"case": analyzer.CASES[0], "phase": "prefill"}, batch)

    def make_evidence(self, root):
        source = root / "source"
        (source / "plugin").mkdir(parents=True)
        (source / "plugin/probe.py").write_text("# fixture\n")
        (root / "dependency.so").write_bytes(b"fixture")
        config = {"num_hidden_layers": 43, "hidden_size": 4, "hc_mult": 4,
                  "vocab_size": 4, "head_dim": 512, "num_attention_heads": 64,
                  "compress_ratios": [0 if i < 2 else 4 if i % 2 == 0 else 128 for i in range(43)]}
        write_json(root / "config.json", config)
        write_json(root / "inputs.json", {"3072": [[2] * 3072]})
        identity = {
            "job": "test", "node": os.uname().nodename, "base_commit": BASE_COMMIT,
            "environment": EXPECTED_ENV, "sources": {"plugin/probe.py": digest(source / "plugin/probe.py")},
            "installed_sources_and_library": {str(root / "dependency.so"): digest(root / "dependency.so")},
            "model_config_path": str(root / "config.json"), "model_config_sha256": digest(root / "config.json"),
            "prompt_sha256": digest(root / "inputs.json"), "cases": [],
        }
        write_json(root / "identity.json", identity)
        write_json(root / "complete.json", identity)
        (root / "worker/trial").mkdir(parents=True)
        events = []
        for arm, case in enumerate(analyzer.CASES):
            write_json(root / f"{case}-responses.json", [{"output_ids": [3, 3]}])
            for i in range(7):
                e = event(case, True, "prefill" if i < 6 else "decode", arm * 7 + i)
                e.update(tokens=512 if i < 6 else 1, seq_lens=[512 * (i + 1)] if i < 6 else [3073],
                         affinity=[0], forward_ns=10000 if arm != 1 else 9000)
                calls = []
                for layer, ratio in enumerate(config["compress_ratios"]):
                    selected = arm == 1 and i < 6 and ratio == 4
                    c = {"layer": layer, "ratio": ratio, "candidate": selected,
                         "q_shape": [e["tokens"], 64, 512], "kv_shape": [128, 512], "mask_hash": text_hash("mask")}
                    if layer == 0:
                        c.update(q_hash=text_hash("q"), kv_hash=text_hash("kv"))
                    calls.append(c)
                    e["rows"].append({"layer": layer, "op": "trial.sparse", "calls": 1,
                                      "metadata": {"ratio": ratio, "candidate": selected},
                                      "inclusive_ns": 10, "exclusive_ns": 10})
                path = root / f"worker/trial/7-{e['sequence']}.pt"
                payload = {k: e[k] for k in ("job", "pid", "sequence", "case", "phase")}
                payload.update(hidden_last_row={l: torch.ones(1, 4, 4) for l in range(43)},
                               logits=torch.arange(4).reshape(1, 4).float())
                torch.save(payload, path)
                e["trial"] = {
                    "candidate": arm == 1, "state_reset": i == 0, "calls": calls,
                    "positions_hash": text_hash(f"pos{i}"), "input_ids_hash": text_hash(f"in{i}"),
                    "req_pool_indices": [0], "out_cache_loc_hash": text_hash(f"loc{i}"),
                    "tensors": str(path), "tensors_sha256": digest(path),
                }
                events.append(e)
        return source, events

    def test_analyzer_full_entry_and_invalid_evidence(self):
        mutations = ("none", "identity", "missing-layer", "illegal-dispatch", "coordinates",
                     "nan-candidate", "nan-baseline", "tensor-identity", "response",
                     "baseline-drift", "shape", "affinity", "missing-logits", "source")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, events = self.make_evidence(root)
                if mutation == "identity":
                    events[7]["job"] = "other"
                elif mutation == "missing-layer":
                    events[7]["trial"]["calls"].pop()
                elif mutation == "illegal-dispatch":
                    events[13]["trial"]["calls"][2]["candidate"] = True
                elif mutation == "coordinates":
                    events[7]["trial"]["out_cache_loc_hash"] = text_hash("other")
                elif mutation == "shape":
                    events[7]["trial"]["calls"][2]["q_shape"][0] = 1
                elif mutation == "affinity":
                    events[0]["affinity"] = [1]
                elif mutation == "response":
                    write_json(root / f"{analyzer.CASES[1]}-responses.json", [{"output_ids": [2, 3]}])
                elif mutation == "source":
                    (source / "plugin/probe.py").write_text("changed")
                elif mutation in ("nan-candidate", "nan-baseline", "tensor-identity",
                                  "baseline-drift", "missing-logits"):
                    e = events[14 if mutation in ("nan-baseline", "baseline-drift") else 12]
                    path = Path(e["trial"]["tensors"])
                    payload = torch.load(path, weights_only=True)
                    if mutation.startswith("nan"):
                        payload["logits"][0, 0] = float("nan")
                    elif mutation == "baseline-drift":
                        payload["hidden_last_row"][2][0, 0, 0] = 2
                    elif mutation == "missing-logits":
                        payload["logits"] = None
                    else:
                        payload["job"] = "other"
                    torch.save(payload, path)
                    e["trial"]["tensors_sha256"] = digest(path)
                (root / "worker/events.7.jsonl").write_text(
                    "".join(json.dumps(e) + "\n" for e in events))
                with patch.object(analyzer, "ROOT", source), patch.dict(os.environ, SLURM_JOB_ID="test"):
                    if mutation == "none":
                        analyzer.analyze(root)
                        result = json.loads((root / "trial-analysis.json").read_text())
                        self.assertTrue(result["baseline_repeatability_bytes"])
                        self.assertAlmostEqual(result["observed_prefill_speedup_vs_baseline_a"], 10 / 9)
                        self.assertTrue(result["promotion"].startswith("BLOCKED"))
                    else:
                        with self.assertRaises(RuntimeError):
                            analyzer.analyze(root)
                        self.assertEqual((root / "trial-analysis.json").exists(), mutation == "baseline-drift")
                        if mutation == "baseline-drift":
                            result = json.loads((root / "trial-analysis.json").read_text())
                            self.assertIsNone(result["observed_prefill_speedup_vs_baseline_a"])


if __name__ == "__main__":
    unittest.main()
