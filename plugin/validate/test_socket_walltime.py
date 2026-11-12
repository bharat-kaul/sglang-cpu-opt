import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_socket_walltime as bench
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

            with patch.object(bench, "plan", return_value={}), \
                    patch.object(bench, "topology"), patch.object(bench, "cases", return_value=specs), \
                    patch.object(bench.subprocess, "Popen", return_value=Child()), \
                    patch.object(bench, "wait_file", side_effect=wait), patch.object(bench.os, "killpg"), \
                    patch.dict(os.environ, SLURM_JOB_ID="123"):
                bench.coordinate(root, "gnr")
            self.assertTrue((root / "gnr" / "complete.json").exists())
            self.assertEqual(len(list((root / "gnr").glob("*.aggregate.json"))), 2)

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

            with patch.object(bench, "plan", return_value={}), \
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
