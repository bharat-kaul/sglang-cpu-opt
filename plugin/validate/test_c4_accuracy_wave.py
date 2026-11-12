import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import c4_accuracy_wave as wave
from dsv4_accuracy_probe import AccuracyProbe
from dsv4_sparse_trial import STATE_MAPS


def dispatch(arm, phase):
    rows = []
    for layer in range(43):
        ratio = 0 if layer < 2 else 4 if layer % 2 == 0 else 128
        rows.append({"layer": layer, "ratio": ratio,
                     "candidate": arm == "candidate" and phase == "prefill" and ratio == 4})
    return rows


class AccuracyTests(unittest.TestCase):
    def test_worker_hook_gating_state_and_caught_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infra = types.SimpleNamespace(**{key: {"old": 1} for key in STATE_MAPS},
                                          _SEL_HOLDER={"pos_kept": "old"})
            plan = {"a": "baseline", "b": "candidate"}
            probe = AccuracyProbe(root / "control.json", plan, infra, root / "events.jsonl")
            batch = types.SimpleNamespace(input_ids=torch.tensor([1]), positions=torch.tensor([0]),
                                          out_cache_loc=torch.tensor([0]), seq_lens=torch.tensor([1]))
            q, kv, mask, sink = torch.ones(1, 2, 4), torch.ones(3, 4), torch.ones(1, 3, dtype=torch.bool), torch.zeros(2)
            called = []

            def original(*args):
                called.append(1)
                return torch.zeros_like(q)

            sparse = probe.wrap_sparse(original)
            with patch.object(torch, "get_num_threads", return_value=32), patch.dict(os.environ, SLURM_JOB_ID="test"):
                for case, arm in plan.items():
                    for phase in ("prefill", "decode"):
                        wave.write_json(root / "control.json", {"case": case, "arm": arm})
                        batch.forward_mode = types.SimpleNamespace(is_decode_or_idle=lambda: phase == "decode")
                        probe.begin(batch)
                        self.assertTrue(all(not getattr(infra, key) for key in STATE_MAPS))
                        for item in dispatch(arm, phase):
                            probe.layer = item["layer"], item["ratio"]
                            result = sparse(q, kv, mask, sink, .5, None)
                            self.assertEqual(bool(result.any()), item["candidate"])
                        probe.finish(types.SimpleNamespace(next_token_logits=torch.ones(1, 4)), None, 1000)
                self.assertEqual(len(called), 4 * 43 - 21)
                batch.forward_mode = types.SimpleNamespace(is_decode_or_idle=lambda: True)
                probe.begin(batch)
                probe.failed = True
                with self.assertRaisesRegex(RuntimeError, "caught native failure"):
                    probe.finish(None, None, 1000)
                self.assertIsNone(probe.active)
            probe.stream.close()
            rows = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual([r["reset"] for r in rows[:4]], [True, False, True, False])
            self.assertIn("error", rows[-1])

    def test_full_wave_and_adversarial_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source-base.txt").write_text(wave.BASE_COMMIT)
            (root / "source-sha256.txt").write_text("fixture")
            (root / "array-job.txt").write_text("123")
            with patch.object(wave, "prepare_inputs", return_value={
                n: [[1] * n for _ in range(64)] for n in (128, 3072)
            }):
                wave.prepare(root)
            library = root / "common.so"
            library.write_bytes(b"fixture")
            prompts, golds = wave.inputs()
            answers = dict(zip(prompts, golds))
            instances = []

            class Engine:
                def __init__(self, **kwargs):
                    self.sequence = 0
                    self.stopped = False
                    self.flushes = 0
                    self.out = Path(os.environ["DSV4_C4_ACCURACY_CONTROL"]).parent
                    self.kwargs = kwargs
                    instances.append(self)

                def get_server_info(self):
                    return {"enable_mixed_chunk": False}

                def flush_cache(self):
                    self.flushes += 1
                    return types.SimpleNamespace(success=True)

                def generate(self, inputs=None, params=None, *, input_ids=None, sampling_params=None):
                    ctl = json.loads((self.out / "control.json").read_text())
                    perf = input_ids is not None
                    if perf:
                        inputs = input_ids
                    length = len(inputs[0]) if perf else 100
                    e = {
                        **ctl, "phase": "prefill", "reset": True, "job": os.environ["SLURM_JOB_ID"],
                        "pid": 7, "sequence": self.sequence, "tokens": len(inputs) * length,
                        "seq_lens": [length] * len(inputs),
                        "threads": 32, "affinity": [0], "forward_ns": 1,
                        "sparse": dispatch(ctl["arm"], "prefill"),
                        "first_input_hash": wave.prompt_hash(str(inputs[0])),
                        "first_positions_hash": "position", "first_cache_hash": "cache",
                    }
                    self.sequence += 1
                    with (self.out / "events/events.7.jsonl").open("a") as stream:
                        stream.write(json.dumps(e) + "\n")
                        if perf:
                            for step in range(1, 5):
                                stream.write(json.dumps({
                                    **e, "phase": "decode", "reset": False, "tokens": len(inputs),
                                    "seq_lens": [length + step] * len(inputs), "sequence": self.sequence,
                                    "sparse": dispatch(ctl["arm"], "decode"),
                                }) + "\n")
                                self.sequence += 1
                    if perf:
                        return [{"text": "fixture", "output_ids": [3] * 5,
                                 "meta_info": {"prompt_tokens": length, "completion_tokens": 5}}
                                for _ in inputs]
                    return [{"text": f"The answer is {answers[p]}.", "output_ids": [3],
                             "meta_info": {"prompt_tokens": 100, "completion_tokens": 1,
                                           "finish_reason": {"type": "stop"}}} for p in inputs]

                def shutdown(self):
                    self.stopped = True

            modules = {
                "intel_cpu_models": types.ModuleType("intel_cpu_models"),
                "sglang": types.SimpleNamespace(Engine=Engine, __version__="fixture"),
                "sgl_kernel": types.SimpleNamespace(common_ops=types.SimpleNamespace(__file__=str(library))),
            }
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                for index in range(wave.WORKERS):
                    with patch.dict(os.environ, {**wave.ENV, "SLURM_JOB_ID": str(124 + index),
                                                 "SLURM_ARRAY_JOB_ID": "123"}):
                        wave.worker(root, index)
                    (root / f"workers/{index}/exit_code").write_text("0")
            result = wave.summarize(root, final=True)
            self.assertTrue(result["complete"])
            self.assertEqual(result["paired"], 1311)
            self.assertEqual(result["decision"], "NO_OBSERVED_LOSS")
            self.assertEqual(len(instances), 6)
            self.assertTrue(all(e.stopped for e in instances))
            self.assertEqual(sum(e.flushes for e in instances), 96 + 6 * 12)
            self.assertTrue(all(p["process_pairs"] == 6 for p in result["performance"]["points"].values()))
            perf_path = root / "workers/0/performance/perf-b1-s3072-measured-candidate.json"
            original_perf = json.loads(perf_path.read_text())
            perf_events = perf_path.with_name("perf-b1-s3072-measured-candidate-events.json")
            original_perf_events = perf_events.read_text()
            for mutation in ("wrong-batch", "missing-case", "nan-time", "short-output", "wrong-schedule",
                             "candidate-decode", "wrong-anchor"):
                with self.subTest(performance_mutation=mutation):
                    packet = copy.deepcopy(original_perf)
                    if mutation == "wrong-batch":
                        packet["batch"] = 64
                    elif mutation == "nan-time":
                        packet["elapsed_s"] = "NaN"
                    elif mutation == "short-output":
                        packet["responses"][0]["output_ids"] = [3]
                    elif mutation in ("wrong-schedule", "candidate-decode", "wrong-anchor"):
                        events = json.loads(original_perf_events)
                        if mutation == "wrong-schedule":
                            events[1]["seq_lens"] = [3074]
                        elif mutation == "candidate-decode":
                            events[1]["sparse"][2]["candidate"] = True
                        else:
                            events[0]["first_input_hash"] = "other"
                        wave.write_json(perf_events, events)
                        packet["events_sha256"] = wave.digest(perf_events)
                    wave.write_json(perf_path, packet)
                    if mutation == "missing-case":
                        perf_path.unlink()
                    with self.assertRaises(RuntimeError):
                        wave.summarize(root, final=True)
                    self.assertFalse((root / "summary.json").exists())
                    wave.write_json(perf_path, original_perf)
                    perf_events.write_text(original_perf_events)
            path = root / "workers/0/cases/s00-b00-candidate.json"
            original = json.loads(path.read_text())
            for mutation in ("wrong-q", "wrong-job", "wrong-score", "wrong-prompt", "missing-events",
                             "wrong-token", "missing-case", "illegal-dispatch"):
                with self.subTest(mutation=mutation):
                    packet = copy.deepcopy(original)
                    event_path = path.with_name("s00-b00-candidate-events.json")
                    original_events = event_path.read_text()
                    if mutation == "wrong-q":
                        packet["records"][0]["q"] = 1
                    elif mutation == "wrong-job":
                        packet["identity"]["job"] = "other"
                    elif mutation == "wrong-score":
                        packet["records"][0]["correct"] = False
                    elif mutation == "wrong-prompt":
                        packet["records"][0]["prompt_sha256"] = "other"
                    elif mutation == "missing-events":
                        packet["events_sha256"] = "bad"
                    elif mutation == "wrong-token":
                        packet["records"][0]["output_token_ids"] = [True]
                    elif mutation == "illegal-dispatch":
                        events = json.loads(original_events)
                        events[0]["sparse"][0]["candidate"] = True
                        wave.write_json(event_path, events)
                        packet["events_sha256"] = wave.digest(event_path)
                    wave.write_json(path, packet)
                    if mutation == "missing-case":
                        path.unlink()
                    with self.assertRaises(RuntimeError):
                        wave.summarize(root, final=True)
                    self.assertFalse((root / "summary.json").exists())
                    wave.write_json(path, original)
                    event_path.write_text(original_events)
            # A real wrong answer is valid evidence, but must produce a negative disposition.
            packet = copy.deepcopy(original)
            packet["records"][0].update(text="The answer is -999.", pred=-999, correct=False, invalid=False)
            wave.write_json(path, packet)
            result = wave.summarize(root, final=True)
            self.assertEqual(result["decision"], "DO_NOT_PROMOTE")
            self.assertEqual(result["regressions"], [0])

    def test_monitor_aborts_on_invalid_evidence(self):
        with patch.object(wave, "summarize", side_effect=RuntimeError("bad evidence")), \
                patch.object(wave.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "bad evidence"):
                wave.watch(Path("/fixture"), "123")
            run.assert_called_once_with(["scancel", "123"], check=True)


if __name__ == "__main__":
    unittest.main()
