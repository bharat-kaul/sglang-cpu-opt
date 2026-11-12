import ast
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import bench_long_prefill_sparse as bench


def small_inputs():
    g = torch.Generator().manual_seed(41)
    q, kv, sink = (torch.randn(*shape, generator=g) for shape in ((5, 3, 8), (9, 8), (3,)))
    mask = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 0, 0],
    ], dtype=torch.bool)
    return q, kv, mask, sink, 0.125


def fake_donor(q, kv, sink, scale):
    scores = torch.bmm(q, kv.transpose(1, 2)) * scale
    probs = torch.cat([scores, sink.view(1, -1, 1).expand(q.shape[0], -1, 1)], -1).softmax(-1)
    return torch.bmm(probs[..., :-1], kv)


class LongPrefillSparseTests(unittest.TestCase):
    def test_shared_path_matches_authoritative_native_branch_oracle(self):
        source = bench.FROZEN / "plugin/intel_cpu_models/_dsv4_cpu_infra.py"
        tree = ast.parse(source.read_text())
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                     and node.name == "_oracle" and "q_in.reshape" in ast.unparse(node)]
        self.assertEqual(len(functions), 1)
        q, kv, mask, sink, scale = small_inputs()
        namespace = {"torch": torch, "q_in": q, "kv_sh": kv, "mask": mask,
                     "attn_sink": sink, "softmax_scale": scale,
                     "T": q.shape[0], "H": q.shape[1], "D": q.shape[2], "nk": kv.shape[0]}
        code = compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec")
        exec(code, namespace)
        actual = namespace["_oracle"]()
        self.assertTrue(bench.same_bytes(actual, bench.shared_gemm(q, kv, mask, sink, scale)))
        self.assertTrue(torch.equal(actual[0], torch.zeros_like(actual[0])))

    def test_grouping_preserves_sparse_selection_and_sinks(self):
        args = small_inputs()
        reference = bench.shared_gemm(*args)
        for result in (bench.shared_chunked(*args, chunk=2),
                       bench.gathered_donor(*args, fake_donor)):
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(result[0], torch.zeros_like(result[0])))
        q, kv, mask, sink, scale = args
        # A dropped mask or sink must be distinguishable from the correct formulation.
        self.assertFalse(torch.allclose(reference, bench.shared_gemm(
            q, kv, torch.ones_like(mask), sink, scale)))
        self.assertFalse(torch.allclose(reference, bench.shared_gemm(
            q, kv, mask, sink + 5, scale)))

    def test_timing_entry_and_invalid_evidence(self):
        q, kv, mask, sink, scale = small_inputs()
        sample = {"q": q, "kv": kv, "mask": mask, "sink": sink, "scale": scale,
                  "output": bench.shared_gemm(q, kv, mask, sink, scale),
                  "identity": {"test": True}, "layer": 2}
        native = types.SimpleNamespace(sparse_attend_masked=bench.shared_gemm,
                                       sparse_attend_fp32bmm=fake_donor,
                                       sparse_attend_fp32bmm_hx=fake_donor)
        result = bench.measure(sample, native)
        self.assertEqual(set(result["samples_ns"]), {
            "online_native", "shared_fp32_gemm", "shared_fp32_gemm_chunk32",
            "gathered_fp32_donor", "gathered_fp32_donor_hx"})
        self.assertTrue(all(len(x) == 3 for x in result["samples_ns"].values()))
        bad = copy.deepcopy(sample)
        bad["output"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            bench.measure(bad, native)
        bad = copy.deepcopy(sample)
        bad["output"].add_(1)
        with self.assertRaisesRegex(RuntimeError, "captured output bytes"):
            bench.measure(bad, native)

    def test_full_entry_wrong_capture_hash_has_no_output(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(torch, "get_num_threads", return_value=32), \
                patch.object(bench.os, "sched_getaffinity", return_value=set(range(32))):
            output = Path(tmp) / "results"
            with patch.object(bench, "CAPTURE_HASHES", {0: "0" * 64}):
                with self.assertRaisesRegex(RuntimeError, "capture hash mismatch"):
                    bench.run(output)
            self.assertFalse(output.exists())

    def test_full_entry_accepts_actual_capture_identity(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(torch, "get_num_threads", return_value=32), \
                patch.object(bench.os, "sched_getaffinity", return_value=set(range(32))), \
                patch.dict(bench.os.environ, {
                    "SLURM_JOB_ID": "test", "OMP_NUM_THREADS": "32",
                    "OMP_WAIT_POLICY": "active", "GOMP_SPINCOUNT": "1000000"}), \
                patch.object(bench, "measure", side_effect=lambda s, n: {
                    "layer": s["layer"], "median_ns": {}, "speedup_vs_online": {}}):
            output = Path(tmp) / "results"
            bench.run(output)
            self.assertTrue((output / "complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
