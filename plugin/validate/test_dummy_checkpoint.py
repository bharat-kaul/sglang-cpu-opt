import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
import torch

import make_dummy_checkpoint as builder
import _smoke


class DummyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text('{"model_type":"llama","hidden_size":8,"torch_dtype":"float32"}')
        self.spec_path = self.root / "spec.json"
        self.output = self.root / "checkpoint"
        shapes = {
            "q_proj.weight": [8, 8], "k_proj.weight": [2, 8], "v_proj.weight": [2, 8],
            "o_proj.weight": [8, 8], "gate_proj.weight": [16, 8], "up_proj.weight": [16, 8],
            "down_proj.weight": [8, 16], "norm.weight": [8],
        }
        self.spec = {
            "schema_version": 1, "config_sha256": builder.file_digest(self.config),
            "source_revision": "test-fixture-not-a-model",
            "tensors": [{"name": name, "shape": shape, "dtype": "float32",
                         "fill": "ones" if name == "norm.weight" else "uniform"}
                        for name, shape in shapes.items()],
        }
        self.seal()

    def seal(self):
        self.spec_path.write_text(json.dumps(self.spec))
        self.spec_hash = builder.file_digest(self.spec_path)

    def run_create(self, output=None, *extra):
        return subprocess.run([
            sys.executable, "-B", builder.__file__, "create",
            "--config", str(self.config), "--spec", str(self.spec_path),
            "--spec-sha256", self.spec_hash, "--output", str(output or self.output),
            "--max-bytes", "1048576", "--shard-bytes", "512", *extra,
        ], capture_output=True, text=True, timeout=60)

    def create(self):
        result = self.run_create()
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)["manifest_sha256"]

    def weights(self, output=None):
        result = {}
        for path in (output or self.output).glob("*.safetensors"):
            result.update(load_file(path))
        return result

    def test_shared_weights_are_independent_of_shard_packing_and_inventory_order(self):
        manifest_hash = self.create()
        self.spec["tensors"].reverse()
        self.seal()
        second = self.root / "second"
        result = self.run_create(second, "--shard-bytes", "4096")
        self.assertEqual(result.returncode, 0, result.stderr)
        first, other = self.weights(), self.weights(second)
        for name in first:
            self.assertTrue(torch.equal(first[name], other[name]), name)
        self.assertTrue(torch.equal(first["norm.weight"], torch.ones(8)))
        verify = subprocess.run([sys.executable, "-B", builder.__file__, "verify", str(self.output),
                                 "--manifest-sha256", manifest_hash], capture_output=True, text=True)
        self.assertEqual(verify.returncode, 0, verify.stderr)

    def test_budget_and_invalid_inventory_reject_without_output(self):
        for args in (("--max-bytes", "100"), ("--shard-bytes", "1"), ("--seed", "-1")):
            result = self.run_create(None, *args)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertFalse(self.output.exists())
        original = copy.deepcopy(self.spec)
        for mutation in ("duplicate", "shape", "packed", "config"):
            with self.subTest(mutation=mutation):
                self.spec = copy.deepcopy(original)
                if mutation == "duplicate":
                    self.spec["tensors"].append(self.spec["tensors"][0])
                elif mutation == "shape":
                    self.spec["tensors"][0]["shape"] = [True, 8]
                elif mutation == "packed":
                    self.spec["tensors"][0]["dtype"] = "uint8"
                else:
                    self.spec["config_sha256"] = "0" * 64
                self.seal()
                result = self.run_create()
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertFalse(self.output.exists())

    def test_quantized_config_and_untrusted_spec_are_blocked(self):
        self.config.write_text('{"quantization_config":{"quant_method":"fp8"}}')
        self.spec["config_sha256"] = builder.file_digest(self.config)
        self.seal()
        result = self.run_create()
        self.assertEqual(result.returncode, 2)
        self.assertIn("separately reviewed", result.stderr)
        self.config.write_text("{}")
        self.spec["config_sha256"] = builder.file_digest(self.config)
        self.seal()
        self.spec_hash = "0" * 64
        self.assertEqual(self.run_create().returncode, 2)
        self.assertFalse(self.output.exists())

    def test_no_overwrite_and_checkpoint_tampering_are_blocked(self):
        expected = self.create()
        before = (self.output / "oracle-manifest.json").read_bytes()
        self.assertEqual(self.run_create().returncode, 2)
        self.assertEqual((self.output / "oracle-manifest.json").read_bytes(), before)
        shard = next(self.output.glob("*.safetensors"))
        with shard.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\xff")
        result = subprocess.run([sys.executable, "-B", builder.__file__, "verify", str(self.output),
                                 "--manifest-sha256", expected], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("digest mismatch", result.stderr)

    def test_capture_runner_binds_actual_inputs_and_checkpoint_without_launching_model(self):
        expected = self.create()
        capture = self.root / "capture.pt"
        calls = []

        class FixtureEngine:
            def __init__(self, **kwargs):
                self.server_args = types.SimpleNamespace(**kwargs)

            def generate(self, **kwargs):
                calls.append(kwargs)
                torch.save({"pf.L0.r0": torch.ones(1, 8)}, capture)
                return [{"text": "fixture"}]

            def shutdown(self):
                calls.append("shutdown")

        env = {
            "MODEL_PATH": str(self.output), "LOAD_FORMAT": "safetensors", "DTYPE": "float32",
            "ORACLE_CHECKPOINT_SHA256": expected, "TOKEN_IDS": "1,2,3", "MAX_NEW": "1",
            "MAX_RUNNING": "1", "DISABLE_RADIX": "1", "FULLCAP_RANK": "0",
            "FULLCAP_FILE": str(capture), "CONTEXT_LEN": "128", "CHUNKED_PREFILL_SIZE": "128",
        }
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"sglang": types.SimpleNamespace(Engine=FixtureEngine)}):
            _smoke.main()
            with self.assertRaisesRegex(ValueError, "already exist"):
                _smoke.main()
        run = json.loads(Path(str(capture) + ".oracle.json").read_text())
        self.assertEqual(run["input_ids"], calls[0]["input_ids"][0])
        self.assertEqual(run["checkpoint_sha256"], expected)
        self.assertEqual(run["capture_sha256"], builder.file_digest(capture))
        self.assertEqual(calls[-1], "shutdown")

    def test_fullcap_rank_selection_prevents_multiple_writers(self):
        source = Path(builder.__file__).parents[1] / "_fullcap_io.py"
        spec = importlib.util.spec_from_file_location("fullcap_io_under_test", source)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        target = self.root / "rank.pt"
        distributed = types.ModuleType("sglang.srt.distributed")
        with patch.dict(sys.modules, {"sglang.srt.distributed": distributed}), patch.dict(os.environ, {"FULLCAP_RANK": "0"}):
            distributed.get_tensor_model_parallel_rank = lambda: 1
            helper.save_fullcap({"value": torch.zeros(1)}, target)
            self.assertFalse(target.exists())
            distributed.get_tensor_model_parallel_rank = lambda: 0
            helper.save_fullcap({"value": torch.ones(1)}, target)
            distributed.get_tensor_model_parallel_rank = lambda: 1
            helper.save_fullcap({"value": torch.zeros(1)}, target)
        value = torch.load(target, weights_only=True)["value"]
        self.assertTrue(torch.equal(value, torch.ones(1)), value)

    def test_supported_dtypes_scalar_and_zero_initialization(self):
        self.spec["tensors"] = [
            {"name": dtype, "shape": [3, 8], "dtype": dtype, "fill": "uniform"}
            for dtype in ("float16", "bfloat16", "float32")
        ] + [{"name": "scalar", "shape": [], "dtype": "float32", "fill": "zeros"}]
        self.seal()
        self.create()
        weights = self.weights()
        self.assertEqual(weights["scalar"].shape, torch.Size([]))
        self.assertEqual(weights["scalar"].item(), 0)
        for dtype in ("float16", "bfloat16", "float32"):
            self.assertEqual(weights[dtype].dtype, getattr(torch, dtype))
            self.assertTrue(torch.isfinite(weights[dtype]).all())

    @unittest.skipUnless(os.environ.get("TEST_SGLANG_LOADERS") == "1", "requires installed SGLang CPU runtime")
    def test_real_sglang_weight_loaders_at_tp1_tp2_tp4(self):
        from sglang.srt.layers.linear import (
            ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear,
        )

        self.create()
        w = self.weights()
        x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
        for size in (1, 2, 4):
            columns, rows, gates, ups, queries = [], [], [], [], []
            for rank in range(size):
                kwargs = dict(bias=False, params_dtype=torch.float32, tp_rank=rank, tp_size=size)
                column = ColumnParallelLinear(8, 8, **kwargs)
                column.weight_loader(column.weight, w["q_proj.weight"])
                self.assertTrue(torch.equal(column.weight, w["q_proj.weight"].chunk(size, 0)[rank]))
                columns.append(torch.nn.functional.linear(x, column.weight))
                row = RowParallelLinear(8, 8, reduce_results=False, **kwargs)
                row.weight_loader(row.weight, w["o_proj.weight"])
                self.assertTrue(torch.equal(row.weight, w["o_proj.weight"].chunk(size, 1)[rank]))
                rows.append(torch.nn.functional.linear(x.chunk(size, 1)[rank], row.weight))
                merged = MergedColumnParallelLinear(8, [16, 16], **kwargs)
                merged.weight_loader(merged.weight, w["gate_proj.weight"], 0)
                merged.weight_loader(merged.weight, w["up_proj.weight"], 1)
                gate, up = merged.weight.chunk(2, 0)
                self.assertTrue(torch.equal(gate, w["gate_proj.weight"].chunk(size, 0)[rank]))
                self.assertTrue(torch.equal(up, w["up_proj.weight"].chunk(size, 0)[rank]))
                gates.append(torch.nn.functional.linear(x, gate))
                ups.append(torch.nn.functional.linear(x, up))
                qkv = QKVParallelLinear(8, 2, 4, total_num_kv_heads=1, **kwargs)
                for label in ("q", "k", "v"):
                    qkv.weight_loader(qkv.weight, w[f"{label}_proj.weight"], label)
                q, k, v = qkv.weight.split([8 // size, 2, 2], 0)
                self.assertTrue(torch.equal(q, w["q_proj.weight"].chunk(size, 0)[rank]))
                self.assertTrue(torch.equal(k, w["k_proj.weight"]))
                self.assertTrue(torch.equal(v, w["v_proj.weight"]))
                queries.append(torch.nn.functional.linear(x, q))
            for parts, name in ((columns, "q_proj.weight"), (gates, "gate_proj.weight"),
                                (ups, "up_proj.weight"), (queries, "q_proj.weight")):
                torch.testing.assert_close(torch.cat(parts, -1), torch.nn.functional.linear(x, w[name]), atol=1e-9, rtol=1e-6)
            torch.testing.assert_close(sum(rows), torch.nn.functional.linear(x, w["o_proj.weight"]), atol=1e-9, rtol=1e-6)

    def test_legacy_initializer_is_tp1_only_and_preserves_tp1_values(self):
        module_spec = importlib.util.spec_from_file_location("dummy_under_test", Path(builder.__file__).parents[1] / "_dummy_determinism.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        names = ["sglang", "sglang.srt", "sglang.srt.model_loader",
                 "sglang.srt.model_loader.weight_utils", "sglang.srt.model_loader.loader",
                 "sglang.srt.distributed"]
        modules = {name: types.ModuleType(name) for name in names}
        wu = modules[names[3]]
        wu.initialize_dummy_weights = lambda model: None
        world = modules[names[-1]]
        world.get_tensor_model_parallel_world_size = lambda: 2
        model = torch.nn.Linear(8, 8, bias=False)
        untouched = model.weight.detach().clone()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"DETERMINISTIC_DUMMY": "1"}):
            module.install_deterministic_dummy()
            with self.assertRaisesRegex(RuntimeError, "TP=1 only"):
                wu.initialize_dummy_weights(model)
            self.assertTrue(torch.equal(model.weight, untouched))
        # Reinstall to bind the TP1 function, then check against the original algorithm.
        wu.initialize_dummy_weights = lambda model: None
        world.get_tensor_model_parallel_world_size = lambda: 1
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"DETERMINISTIC_DUMMY": "1"}):
            module.install_deterministic_dummy()
            wu.initialize_dummy_weights(model)
        import numpy as np
        rng = np.random.RandomState(int(hashlib.sha256(b"weight").hexdigest()[:8], 16))
        expected = torch.from_numpy(rng.uniform(-1e-3, 1e-3, size=(8, 8)).astype("float32"))
        self.assertTrue(torch.equal(model.weight, expected))


if __name__ == "__main__":
    unittest.main()
