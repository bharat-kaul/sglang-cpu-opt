import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "screen_test", Path(__file__).with_name("screen_serving_opportunities.py"))
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


def event(case, enabled, phase, sequence):
    rows = [{"layer": i, "op": "layer.total", "calls": 1, "metadata": {"threads": 32},
             "inclusive_ns": 10, "exclusive_ns": 1} for i in range(43)]
    rows.extend({"layer": 2, "op": op, "calls": 1, "inclusive_ns": 10, "exclusive_ns": 5}
                for op in ("_batched_mla_attend", "compressor.projection",
                           "native_wrapper.sparse_attend_masked", "indexer.shared_kv_expand"))
    return {"case": case, "enabled": enabled, "pid": 7, "sequence": sequence, "job": "test",
            "tokens": 128 if phase == "prefill" else 1,
            "seq_lens": [128 if phase == "prefill" else 129],
            "threads_before": 32, "threads_after": 32,
            "phase": phase, "forward_ns": 1000, "rows": rows if enabled else [],
            "dispatch_marks": {}, "failures": []}


class ScreenTests(unittest.TestCase):
    def test_gate_rejects_missing_coverage_and_invalid_timing(self):
        events = [event("x", True, p, i) for i, p in enumerate(("prefill", "decode"))]
        coordinates = dict(job="test", batch=1, length=128, output_len=2)
        screen.validate_events(events, "x", True, 43, **coordinates)
        mutations = []
        bad = copy.deepcopy(events)
        bad[0]["rows"].pop(0)
        mutations.append(bad)
        bad = copy.deepcopy(events)
        bad[0]["rows"][0]["inclusive_ns"] = float("nan")
        mutations.append(bad)
        bad = copy.deepcopy(events)
        bad[0]["dispatch_marks"]["2:sparse_attend:eager"] = 1
        mutations.append(bad)
        mutations.append(events + [events[0]])
        mutations.append(events[:1])
        for field, value in (("job", "another-job"), ("pid", True), ("tokens", 127),
                             ("seq_lens", [129]), ("forward_ns", float("inf")),
                             ("threads_before", 58), ("threads_after", 24),
                             ("forward_ns", 181_000_000_000)):
            bad = copy.deepcopy(events)
            bad[0][field] = value
            mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(RuntimeError):
                screen.validate_events(bad, "x", True, 43, **coordinates)
        summary = screen.opportunity_summary(events)
        self.assertAlmostEqual(summary["prefill"]["buckets"]["indexer_replication"]
                               ["share_of_instrumented_forward"], 0.01)
        self.assertIsNone(summary["prefill"]["buckets"]["compressor_casts"]
                          ["ideal_speedup_if_removed"])

    def exercise_run(self, corrupt, meaningful=False, trial=False):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            source = Path(tmp) / "source"
            source.mkdir()
            (Path(tmp) / "source-base.txt").write_text(screen.BASE_COMMIT)
            args = types.SimpleNamespace(output=out, model="/model", data="/data",
                                         batches=[1], lengths=[128], reps=3, output_len=2,
                                         meaningful=meaningful, c4_trial=trial)
            class Engine:
                seq = 0
                stopped = False
                flushes = 0

                def __init__(self, **kwargs):
                    pass

                def get_server_info(self):
                    return {"device": "cpu"}

                def flush_cache(self):
                    Engine.flushes += 1
                    return types.SimpleNamespace(success=corrupt != "flush")

                def generate(self, **kwargs):
                    ctl = json.loads((out / "control.json").read_text())
                    batch, length = len(kwargs["input_ids"]), len(kwargs["input_ids"][0])
                    count = kwargs["sampling_params"]["max_new_tokens"]
                    events = [event(ctl["case"], ctl["enabled"], p, self.seq + i)
                              for i, p in enumerate(["prefill"] + ["decode"] * (count - 1))]
                    self.seq += len(events)
                    for i, e in enumerate(events):
                        e["tokens"] = batch * length if i == 0 else batch
                        e["seq_lens"] = [length + i] * batch
                        if meaningful or trial:
                            e["captures"] = [{"layer": l} for l in (0, 2, 3)] if i <= 1 else []
                            if trial:
                                e.pop("captures")
                                e["rows"].append({"layer": 2, "op": "trial.sparse", "calls": 1,
                                                  "inclusive_ns": 10, "exclusive_ns": 10})
                            e["rows"].extend(
                                {"layer": l, "op": op, "calls": 1, "inclusive_ns": 10,
                                 "exclusive_ns": 1}
                                for l in range(43) for op in ("moe.total", "moe.routed", "moe.shared",
                                                              "mhc.pre", "mhc.post", "attention.total"))
                    if corrupt == "capture":
                        events[0]["captures"] = []
                    if corrupt == "boundary":
                        events[0]["rows"] = [r for r in events[0]["rows"] if r["op"] != "moe.shared"]
                    if corrupt and ctl["enabled"]:
                        if corrupt == "coverage":
                            events[0]["rows"] = []
                        elif corrupt == "coordinates":
                            events[0]["tokens"] = 127
                        elif corrupt == "slow":
                            events[0]["forward_ns"] = 181_000_000_000
                        elif corrupt == "threads":
                            events[0]["threads_before"] = 58
                        elif corrupt not in ("capture", "boundary"):
                            events[0]["job"] = "another-job"
                    worker = out / "worker"
                    worker.mkdir(exist_ok=True)
                    with (worker / "events.7.jsonl").open("a") as stream:
                        for e in events:
                            stream.write(json.dumps(e) + "\n")
                    return [{"output_ids": list(range(count))} for _ in range(batch)]

                def shutdown(self):
                    Engine.stopped = True

            sglang = types.ModuleType("sglang")
            sglang.Engine = Engine
            sglang.__file__ = str(Path(tmp) / "sglang/__init__.py")
            native = types.ModuleType("sgl_kernel")
            native.common_ops = types.SimpleNamespace(__file__=str(Path(tmp) / "common_ops.so"))
            plugin = types.ModuleType("intel_cpu_models")
            with patch.dict(os.environ, {**screen.EXPECTED_ENV, "SLURM_JOB_ID": "test"}), \
                    patch.dict(sys.modules, {"sglang": sglang, "sgl_kernel": native,
                                            "intel_cpu_models": plugin}), \
                    patch.object(screen, "digest", return_value=screen.ACCURACY_MANIFEST), \
                    patch.object(screen, "ROOT", source), \
                    patch.object(screen, "prepare_inputs", return_value={
                        s: [[3] * s for _ in range(64)] for s in (128, 3072)}):
                if corrupt:
                    with self.assertRaises(RuntimeError):
                        screen.run(args)
                    self.assertFalse((out / "complete.json").exists())
                else:
                    screen.run(args)
                    result = json.loads((out / "complete.json").read_text())
                    self.assertEqual(len(result["cases"][0]["measurements"]),
                                     1 if meaningful else 3 if trial else 8)
                    self.assertEqual(len(result["cases"]), 3 if meaningful else 1)
                    if trial:
                        self.assertEqual(Engine.flushes, 3)
                self.assertTrue(Engine.stopped)

    def test_full_driver_success(self):
        self.exercise_run(False)

    def test_trial_driver_and_flush_failure(self):
        self.exercise_run(False, trial=True)
        self.exercise_run("flush", trial=True)
        self.exercise_run("boundary", trial=True)

    def test_full_driver_rejects_invalid_worker_before_completion(self):
        for corruption in ("coverage", "coordinates", "identity", "slow", "threads"):
            with self.subTest(corruption=corruption):
                self.exercise_run(corruption)

    def test_meaningful_driver_is_one_request_per_point(self):
        self.exercise_run(False, meaningful=True)

    def test_meaningful_driver_rejects_missing_instruments(self):
        for corruption in ("capture", "boundary"):
            self.exercise_run(corruption, meaningful=True)


if __name__ == "__main__":
    unittest.main()
