import copy
import io
import json
import os
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

import compare_socket_walltime as bench
import summarize_socket_walltime as report
from test_c4_accuracy_wave import dispatch


def fixture(spec, platform="gnr"):
    packets, events = [], []
    for rank in spec["ranks"]:
        node, first = bench.LAYOUTS[platform][rank]
        identity = {"rank": rank, "platform": platform, "node": "host", "job": "123"}
        start = 1_000_000_000 + rank * 10
        rows = []
        for step in range(5):
            phase = "prefill" if step == 0 else "decode"
            row = {
                "case": spec["case"], "arm": spec["arm"], "job": "123", "pid": 100 + rank,
                "sequence": step, "phase": phase, "reset": step == 0,
                "threads": 32, "affinity": [first], "forward_ns": 10,
                "start_ns": start + 10 + step * 20,
                "tokens": spec["batch"] * (spec["length"] if step == 0 else 1),
                "seq_lens": [spec["length"] + step] * spec["batch"],
                "sparse": dispatch(spec["arm"], phase),
            }
            if step == 0:
                row.update(first_input_hash="input", first_positions_hash="pos", first_cache_hash="cache")
            rows.append(row)
        responses = [{"output_ids": [1] * 5, "meta_info": {
            "prompt_tokens": spec["length"], "completion_tokens": 5}} for _ in range(spec["batch"])]
        packets.append({
            **spec, "identity": identity, "started_ns": start, "ended_ns": start + 200,
            "elapsed_s": 200 / 1e9, "responses": responses,
            "memory": {"pid": 100 + rank, "VmSwap_kib": 0, "numa_kib": {str(node): 100000}},
        })
        events.append(rows)
    return packets, events


class SocketTests(unittest.TestCase):
    def test_publication_rejects_changed_plan_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text('{"baseline_only": false}')
            out = root / "summary.json"
            spec = {**report.RUNS["emr"], "root": str(root)}
            with patch.dict(report.RUNS, emr=spec):
                with self.assertRaisesRegex(RuntimeError, "Unexpected plan identity"):
                    report.publish(out)
            self.assertFalse(out.exists())

    def test_summary_rejects_invalid_samples(self):
        self.assertEqual(report.distribution([2, 4])["median"], 3)
        for values in ([], [0], [-1], [float("nan")], [float("inf")], [True]):
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, "Invalid"):
                report.distribution(values)

    def test_cross_platform_publication_rejects_unmatched_evidence(self):
        plan = {k: "matched" for k in (
            "base_commit", "compute_sources", "installed_sources", "environment",
            "inputs_sha256", "sampling", "checkpoint_files", "model_config_sha256",
            "checkpoint_index_sha256", "cases",
        )}
        functions = {k: "same function" for k in (
            "worker", "aggregate", "run_case", "cases", "memory", "validate_locality",
        )}
        runs = {}
        for platform in ("emr", "gnr"):
            points = {k: {"median": (4 if platform == "emr" else 2)}
                      for k in ("isolated_request_s", "socket_generated_tokens_s",
                                "socket_prefill_tokens_s", "socket_decode_tokens_s")}
            runs[platform] = [
                {"identities": {0: {"torch": "torch", "sglang": "sglang", "common_ops_sha256": "same"}},
                 "points": {"b1-s3072": points}},
                copy.deepcopy(plan), copy.deepcopy(functions),
                {"b1-s3072": {"anchors": ["same"], "schedule": ["same"], "output_ids": [[1, 2]]}},
            ]
        for mutation in (
            None,
            lambda x: x[1].update(compute_sources="other"),
            lambda x: x[1].update(inputs_sha256="other"),
            lambda x: x[2].update(aggregate="other"),
            lambda x: x[3]["b1-s3072"].update(output_ids=[[1, 3]]),
            lambda x: x[0]["identities"][0].update(common_ops_sha256="other"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                values = copy.deepcopy(runs)
                if mutation:
                    mutation(values["gnr"])
                out = Path(tmp) / "summary.json"
                with patch.object(report, "load_run", side_effect=lambda platform, spec: values[platform]):
                    if mutation:
                        with self.assertRaises(RuntimeError):
                            report.publish(out)
                        self.assertFalse(out.exists())
                    else:
                        result = report.publish(out)
                        self.assertEqual(result["gnr_over_emr"]["b1-s3072"]["isolated_request"], 2)
                        self.assertTrue(out.exists())
                        with self.assertRaisesRegex(RuntimeError, "overwrite"):
                            report.publish(out)

    def test_main_source_binding_rejects_changed_compute(self):
        name = "plugin/intel_cpu_models/model.py"
        content = b"main implementation"
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as source:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            source.addfile(member, io.BytesIO(content))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source" / name
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            with patch.object(bench.subprocess, "check_output", return_value=archive.getvalue()) as git:
                result = bench.main_compute_sources(root)
                self.assertEqual(result, {name: bench.digest(path)})
                self.assertIn(bench.BASE_COMMIT, git.call_args.args[0])
                path.write_bytes(b"experimental implementation")
                with self.assertRaisesRegex(RuntimeError, "differs from main"):
                    bench.main_compute_sources(root)

    def test_remote_cache_complete_and_failed_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "a.safetensors").write_bytes(b"checkpoint")
            bench.write_json(root / "plan.json", {"test": True})
            config = {"remote_cache": True, "checkpoint_files": {"a.safetensors": {"size": 10}}}
            original_read = Path.read_text

            def read(path, *args, **kwargs):
                if path.name == "meminfo":
                    return "Node memory evidence"
                return original_read(path, *args, **kwargs)

            with patch.object(bench, "MODEL", model), patch.object(bench, "plan", return_value=config), \
                    patch.object(Path, "read_text", read), \
                    patch.dict(os.environ, SLURM_JOB_ID="123"), \
                    patch.object(bench.subprocess, "check_output") as show:
                show.return_value = "policy: bind\nmembind: 0\n"
                with self.assertRaisesRegex(RuntimeError, "must interleave"):
                    bench.prewarm(root)
                self.assertFalse((root / "remote-cache.json").exists())
                show.return_value = ("policy: interleave\ninterleavemask: 3 4 5 \n"
                                     "cpubind: 3 4 5 \nmembind: 0 1 2 3 4 5 \n")
                for policy in (
                    show.return_value.replace("interleavemask: 3 4 5", "interleavemask: 0 1 2"),
                    show.return_value + "interleavemask: 3 4 5\n",
                    show.return_value.replace("cpubind: 3 4 5", "cpubind: 0 1 2"),
                    "policy: interleave\nmembind: 3 4 5\n",
                ):
                    with self.assertRaisesRegex(RuntimeError, "must interleave"):
                        bench.validate_cache_policy(policy)
                config["checkpoint_files"]["a.safetensors"]["size"] = 11
                with self.assertRaisesRegex(RuntimeError, "size changed"):
                    bench.prewarm(root)
                self.assertFalse((root / "remote-cache.json").exists())
                self.assertIn("after", json.loads((root / "remote-cache-progress.json").read_text()))
                config["checkpoint_files"]["a.safetensors"]["size"] = 10
                bench.prewarm(root)
                record = json.loads((root / "remote-cache.json").read_text())
                self.assertEqual(record["files"]["a.safetensors"]["sha256"],
                                 bench.digest(model / "a.safetensors"))
                self.assertEqual(record["job"], "123")
                with self.assertRaisesRegex(RuntimeError, "already completed"):
                    bench.prewarm(root)

    def test_memory_retains_rejected_values(self):
        for swap in (0, 2048):
            def read(path):
                if path.name == "status":
                    return f"VmRSS:\t100 kB\nVmHWM:\t200 kB\nVmSwap:\t{swap} kB\n"
                return "1000 bind:0 anon=25 N0=25 kernelpagesize_kB=4\n"

            with patch.object(Path, "read_text", read):
                record = bench.memory(123)
            self.assertEqual(record["VmSwap_kib"], swap)
            self.assertEqual(record["numa_kib"], {"0": 100})
            if swap:
                with self.assertRaisesRegex(RuntimeError, "swap_kib=2048"):
                    bench.validate_locality(record, 0)
            else:
                bench.validate_locality(record, 0)
        with self.assertRaisesRegex(RuntimeError, "resident_numa_kib={}"):
            bench.validate_locality({"pid": 123, "VmSwap_kib": 0, "numa_kib": {}}, 0)

    def test_coordinate_positive_pair(self):
        specs = list(bench.cases("gnr"))[:2]
        fixtures = {s["case"]: fixture(s) for s in specs}
        identity = fixtures[specs[0]["case"]][0][0]["identity"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.write_json(root / "plan.json", {"test": True})

            class Child:
                pid = 123456789

                def poll(self):
                    return None

                def wait(self, timeout):
                    return 0

            def wait(path, children=(), timeout=3600):
                if path.name == "ready.json":
                    path.parent.mkdir()
                    rank = int(path.parent.name.removeprefix("rank"))
                    bench.write_json(path.parent / "identity.json", {**identity, "rank": rank})
                elif path.suffix == ".json":
                    packets, events = copy.deepcopy(fixtures[path.stem])
                    ep = path.with_name(path.stem + ".events.json")
                    bench.write_json(ep, events[0])
                    packets[0]["events_sha256"] = bench.digest(ep)
                    bench.write_json(path, packets[0])

            with patch.object(bench, "plan", return_value={"baseline_only": False, "remote_cache": False}), \
                    patch.object(bench, "topology"), patch.object(bench, "cases", return_value=specs), \
                    patch.object(bench.subprocess, "Popen", return_value=Child()), \
                    patch.object(bench, "wait_file", side_effect=wait), patch.object(bench.os, "killpg"), \
                    patch.dict(os.environ, SLURM_JOB_ID="123"):
                bench.coordinate(root, "gnr")
            self.assertTrue((root / "gnr" / "complete.json").exists())
            self.assertEqual(len(list((root / "gnr").glob("*.aggregate.json"))), 2)

    def test_native_only_full_coordinate_and_repeat_guard(self):
        specs = list(bench.cases("gnr", baseline_only=True))
        self.assertEqual(len(specs), 30)
        self.assertEqual({s["arm"] for s in specs}, {"baseline"})
        for mutate in (False, True):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bench.write_json(root / "plan.json", {"test": True})
                fixtures = {s["case"]: fixture(s) for s in specs}

                class Child:
                    pid = 123456789

                    def poll(self):
                        return None

                    def wait(self, timeout):
                        return 0

                def wait(path, children=(), timeout=3600):
                    rank = int(path.parent.name.removeprefix("rank"))
                    if path.name == "ready.json":
                        path.parent.mkdir()
                        identity = {"rank": rank, "platform": "gnr", "node": "host", "job": "123"}
                        bench.write_json(path.parent / "identity.json", identity)
                    elif path.suffix == ".json":
                        packets, events = copy.deepcopy(fixtures[path.stem])
                        index = packets[0]["ranks"].index(rank)
                        packet, ev = packets[index], events[index]
                        if mutate and packet["stage"] == "measured":
                            packet["responses"][0]["output_ids"][0] = 2
                        ep = path.with_name(path.stem + ".events.json")
                        bench.write_json(ep, ev)
                        packet["events_sha256"] = bench.digest(ep)
                        bench.write_json(path, packet)

                with patch.object(bench, "plan", return_value={"baseline_only": True, "remote_cache": False}), \
                        patch.object(bench, "topology"), \
                        patch.object(bench.subprocess, "Popen", return_value=Child()), \
                        patch.object(bench, "wait_file", side_effect=wait), patch.object(bench.os, "killpg"), \
                        patch.dict(os.environ, SLURM_JOB_ID="123"):
                    if mutate:
                        with self.assertRaisesRegex(RuntimeError, "generated tokens changed"):
                            bench.coordinate(root, "gnr")
                        self.assertFalse((root / "gnr" / "complete.json").exists())
                    else:
                        bench.coordinate(root, "gnr")
                        self.assertTrue((root / "gnr" / "complete.json").exists())
                        self.assertEqual(len(list((root / "gnr").glob("*.aggregate.json"))), 30)

    def test_schedule_and_aggregate(self):
        specs = list(bench.cases("gnr"))
        self.assertEqual(len(specs), 60)
        self.assertEqual(len({s["case"] for s in specs}), 60)
        for arm in ("baseline", "candidate"):
            for batch, length in bench.PERF_POINTS:
                rows = [s for s in specs if s["stage"] == "measured" and s["arm"] == arm
                        and (s["batch"], s["length"]) == (batch, length)]
                self.assertEqual(sum(s["group"] == "socket" for s in rows), 3)
                self.assertEqual(sum(s["group"] != "socket" for s in rows), 3)
        for spec in specs:
            packets, events = fixture(spec)
            result = bench.aggregate(packets, events, spec)
            expected_wall = (200 + 10 * (max(spec["ranks"]) - min(spec["ranks"]))) / 1e9
            self.assertEqual(result["request_wall_s"], expected_wall)
            self.assertEqual(result["generated_tokens_s"],
                             len(spec["ranks"]) * spec["batch"] * 5 / expected_wall)

    def test_reject_invalid_aggregate(self):
        spec = next(s for s in bench.cases("gnr") if s["group"] == "socket")
        for mutation in (
            lambda p, e: p.pop(),
            lambda p, e: e.pop(),
            lambda p, e: p[1]["identity"].update(rank=0),
            lambda p, e: p[1]["identity"].update(job="different"),
            lambda p, e: p[0].update(elapsed_s=float("nan")),
            lambda p, e: p[0]["responses"][0].update(output_ids=[1]),
            lambda p, e: p[1]["memory"].update(numa_kib={"0": 100000}),
            lambda p, e: p[0]["memory"].update(VmSwap_kib=1),
            lambda p, e: p[0]["memory"].update(pid=999),
            lambda p, e: e[0][0].update(start_ns=0),
            lambda p, e: e[1][0].update(affinity=[0]),
            lambda p, e: e[1][0].update(first_input_hash="other"),
            lambda p, e: e[1][0].update(pid=100),
            lambda p, e: e[0][0]["sparse"].pop(),
        ):
            packets, events = fixture(spec)
            mutation(packets, events)
            with self.assertRaises(RuntimeError):
                bench.aggregate(packets, events, spec)

    def test_coordinate_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = next(bench.cases("gnr"))
            packets, events = fixture(spec)

            class Child:
                pid = 123456789
                returncode = None

                def poll(self):
                    return None

                def wait(self, timeout):
                    return 0

            def wait(path, children=(), timeout=3600):
                if path.name == "ready.json":
                    path.parent.mkdir()
                    rank = int(path.parent.name.removeprefix("rank"))
                    identity = {**packets[0]["identity"], "rank": rank}
                    bench.write_json(path.parent / "identity.json", identity)
                    bench.write_json(path, identity)
                elif path.name.endswith(".json"):
                    bad = copy.deepcopy(packets[0])
                    bad["identity"]["job"] = "wrong"
                    bench.write_json(path, bad)

            with patch.object(bench, "plan", return_value={"baseline_only": False, "remote_cache": False}), \
                    patch.object(bench, "topology"), patch.object(bench, "cases", return_value=[spec]), \
                    patch.object(bench.subprocess, "Popen", return_value=Child()) as popen, \
                    patch.object(bench, "wait_file", side_effect=wait), patch.object(bench.os, "killpg"):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    bench.coordinate(root, "gnr")
                self.assertEqual(popen.call_count, 1)
            self.assertFalse((root / "gnr" / "complete.json").exists())
            self.assertEqual(list((root / "gnr").glob("*.aggregate.json")), [])


if __name__ == "__main__":
    unittest.main()
