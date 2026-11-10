import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

import analyze_prefix_trace
import run_prefix_trace


class PrefixAnalysisTests(unittest.TestCase):
    def test_driver_checkpoints_results_and_propagates_reproduction_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [
                {"q": q, "gold": 11, "prompt": str(q), "text": "The answer is 11.",
                 "output_token_ids": [3]}
                for q in (1175, 1168, 1239)
            ]
            (root / "plan.json").write_text(json.dumps({
                "model": "fixture", "targets": [1175, 1168, 1239], "last_decision": 0,
                "batches": [items[:2], items[2:]],
            }))
            (root / "source_hashes.json").write_text("{}")
            incorrect = False
            shutdowns = []
            test = self

            class Engine:
                def __init__(self, **kwargs):
                    test.assertEqual(Path(os.environ["DSV4_PREFIX_TRACE_PHASE"]).read_text(), "warmup")

                def get_server_info(self):
                    return {"device": "cpu"}

                def generate(self, prompts, sampling_params):
                    rank = Path(os.environ["DSV4_PREFIX_TRACE_OUT"]) / "rank0"
                    rank.mkdir(exist_ok=True)
                    for prompt in prompts:
                        (rank / f"natural.q{prompt}.d0.pt").write_bytes(b"fixture")
                    test.assertTrue(all(p["max_new_tokens"] == 256 for p in sampling_params))
                    return [
                        {"text": "The answer is 7." if incorrect else "The answer is 11.",
                         "output_ids": [3], "meta_info": {}} for _ in prompts
                    ]

                def shutdown(self):
                    shutdowns.append(True)

            env = {"SLURM_JOB_ID": "1", "INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC": "0"}
            with patch.dict(os.environ, env), patch.dict(sys.modules, {"sglang": SimpleNamespace(Engine=Engine)}):
                argv = ["run_prefix_trace.py", str(root / "plan.json"), str(root / "good"),
                        "--device", "cpu"]
                with patch.object(sys, "argv", argv):
                    run_prefix_trace.main()
                self.assertTrue((root / "good/complete.json").is_file())
                self.assertTrue((root / "good/natural.raw.batch0.json").is_file())
                incorrect = True
                argv[2] = str(root / "bad")
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                    RuntimeError, "CPU baseline did not reproduce"
                ):
                    run_prefix_trace.main()
                self.assertTrue((root / "bad/natural.raw.batch0.json").is_file())
                self.assertFalse((root / "bad/complete.json").exists())
            self.assertEqual(len(shutdowns), 2)

    def test_complete_comparison_and_wrong_coordinate_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").mkdir()
            (root / "source/fixture").write_text("fixture")
            (root / "source_hashes.json").write_text(json.dumps({
                "fixture": hashlib.sha256(b"fixture").hexdigest(),
            }))
            original = {
                "q": 1175, "gold": 11, "text": "The answer is 11.",
                "prompt_ids": [1, 2], "output_token_ids": [3],
            }
            plan = {"batches": [[original]], "targets": [1175], "last_decision": 0}
            (root / "plan.json").write_text(json.dumps(plan))
            plan_hash = analyze_prefix_trace.digest(root / "plan.json")
            source_hash = analyze_prefix_trace.digest(root / "source_hashes.json")
            boundaries = {}
            for layer in range(43):
                for boundary in ("attn.in", "attn.out", "mlp.in", "mlp.out"):
                    boundaries[f"L{layer}.{boundary}"] = torch.tensor([1.0, -2.0])
                boundaries[f"L{layer}.topk.out"] = [torch.tensor([2, 1]), torch.tensor([0.6, 0.4])]
            for boundary in ("hc_head.in", "hc_head.out", "final_norm.out", "lm_head.in"):
                boundaries[boundary] = torch.tensor([1.0, -2.0])
            for device, job, ranks, phases in (
                ("cpu", "1", 1, ("natural",)), ("cuda", "2", 2, ("natural", "forced"))
            ):
                directory = root / ("cpu_1" if device == "cpu" else "gpu_2")
                directory.mkdir()
                identity = {"job": job, "device": device, "plan_sha256": plan_hash,
                            "source_manifest_sha256": source_hash}
                for name in ("identity", "complete"):
                    (directory / f"{name}.json").write_text(json.dumps(identity))
                for phase in phases:
                    record = {**original, "pred": 11}
                    (directory / f"{phase}.batch0.json").write_text(json.dumps([record]))
                    for rank in range(ranks):
                        rank_dir = directory / f"rank{rank}"
                        rank_dir.mkdir(exist_ok=True)
                        payload = {
                            "q": 1175, "decision": 0, "phase": phase, "job": job,
                            "plan_sha256": plan_hash, "sequence_ids": [1, 2],
                            "target_row": 1, "logit_row": 0, "positions": [0, 1],
                            "input_ids": [1, 2], "slots": [7], "slot": 7,
                            "logits": torch.tensor([0.0, 1.0, 2.0, 3.0]),
                            "boundaries": boundaries,
                        }
                        torch.save(payload, rank_dir / f"{phase}.q1175.d0.pt")
            argv = ["analyze_prefix_trace.py", str(root), "1", "2"]
            with patch.object(sys, "argv", argv):
                analyze_prefix_trace.main()
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["cpu_reproduced_original_cases"], 1)
            self.assertEqual(summary["matched_decisions"], 1)
            summary_path.unlink()
            path = root / "gpu_2/rank0/forced.q1175.d0.pt"
            payload = torch.load(path, weights_only=True)
            payload["positions"][1] = 9
            torch.save(payload, path)
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                RuntimeError, "inconsistent request coordinates"
            ):
                analyze_prefix_trace.main()
            self.assertFalse(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
