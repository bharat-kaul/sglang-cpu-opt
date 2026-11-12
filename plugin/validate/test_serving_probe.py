import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path("/scratch/bkaul/dsv4_latest_fullacc_20261107/source")


def load_probe():
    spec = importlib.util.spec_from_file_location("probe_test", ROOT / "plugin/dsv4_serving_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StripProbe(ast.NodeTransformer):
    def visit_Import(self, node):
        if any(a.name == "dsv4_serving_probe" for a in node.names):
            return None
        return node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "_serving_probe"):
            return None
        return self.generic_visit(node)

    def visit_With(self, node):
        if (len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call)
                and isinstance(node.items[0].context_expr.func, ast.Attribute)
                and isinstance(node.items[0].context_expr.func.value, ast.Name)
                and node.items[0].context_expr.func.value.id == "_serving_probe"):
            return [self.visit(child) for child in node.body]
        return self.generic_visit(node)


class ProbeTests(unittest.TestCase):
    def test_infrastructure_math_unchanged(self):
        relative = "plugin/intel_cpu_models/_dsv4_cpu_infra.py"
        old = ast.parse((FROZEN / relative).read_text())
        new = StripProbe().visit(ast.parse((ROOT / relative).read_text()))
        self.assertEqual(ast.dump(old), ast.dump(new))
        # Discriminating control: a changed tensor operation must not pass the comparison.
        mutated = ast.parse((FROZEN / relative).read_text().replace(
            "q = torch.matmul(q.float()", "q = torch.matmul(q.double()", 1))
        self.assertNotEqual(ast.dump(old), ast.dump(mutated))

    def test_nested_exclusive_accounting(self):
        probe = load_probe()
        rec = probe.Recorder()
        with patch.object(probe.time, "perf_counter_ns", side_effect=[0, 10, 30, 50]):
            with rec.region("parent"):
                with rec.region("child"):
                    pass
        rows = {r["op"]: r for r in rec.rows.values()}
        self.assertEqual(rows["parent"]["inclusive_ns"], 50)
        self.assertEqual(rows["parent"]["exclusive_ns"], 30)
        self.assertEqual(rows["child"]["exclusive_ns"], 20)

    def test_control_rejects_malformed(self):
        probe = load_probe()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control.json"
            for value in ({"case": "x", "enabled": 1}, {"case": "", "enabled": True},
                          {"case": "x"}, {"case": "x", "enabled": True, "extra": 0}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    probe.read_control(path)
            path.write_text('{"case":"ok","enabled":false}')
            self.assertEqual(probe.read_control(path), {"case": "ok", "enabled": False})

    def exercise_worker(self, trial=False):
        probe = load_probe()
        packages = [
            "intel_cpu_models", "sglang", "sglang.srt", "sglang.srt.models",
            "sglang.srt.layers", "sglang.srt.layers.attention",
            "sglang.srt.layers.attention.dsv4",
        ]
        modules = {name: types.ModuleType(name) for name in packages}
        for module in modules.values():
            module.__path__ = []
        leaves = [
            "intel_cpu_models._dsv4_cpu_infra", "intel_cpu_models._dsa_ckernels",
            "sglang.srt.models.deepseek_v4", "sglang.srt.layers.attention.dsv4.compressor",
            "sglang.srt.layers.attention.deepseek_v4_backend",
        ]
        modules.update({name: types.ModuleType(name) for name in leaves})
        for name, module in modules.items():
            if "." in name:
                parent, leaf = name.rsplit(".", 1)
                setattr(modules[parent], leaf, module)
        infra, ck, model, comp, backend = (modules[n] for n in leaves)
        infra._TIMEIT_ON = False
        for name in ("_batched_mla_attend", "_stash_kv_gather",
                     "_build_coverage_mask_vectorized", "_compress_prefill_core"):
            setattr(infra, name, lambda x: x)
        for name in ("indexer_logits", "indexer_topk", "compress_softmax_pool", "sparse_attend_masked"):
            setattr(ck, name, lambda x: x)
        ck.mark = lambda *a: None

        class Compressor:
            ratio = 4
            is_in_indexer = False
            weight = torch.eye(4, dtype=torch.bfloat16)

            def compute_kv_score(self, x, forward_batch):
                return torch.mm(x.float(), self.weight.float().t())

        class Backend:
            def forward_core_compressor(self, x):
                return x

            def forward_c4_indexer(self, x):
                return x

        class Layer:
            layer_id = 0
            self_attn = types.SimpleNamespace(compress_ratio=0)

            def forward(self, x, batch):
                y = Compressor().compute_kv_score(x, batch)
                try:
                    return ck.indexer_logits(y)
                except ValueError:
                    return y

        class Model:
            def forward(self, input_ids, positions, forward_batch):
                return Layer().forward(input_ids, forward_batch)

        comp.Compressor = Compressor
        backend.DeepseekV4AttnBackend = Backend
        model.DeepseekV4DecoderLayer = Layer
        model.DeepseekV4ForCausalLM = Model
        x = torch.arange(8).reshape(2, 4).to(torch.bfloat16)
        batch = types.SimpleNamespace(
            input_ids=x, seq_lens=torch.tensor([4, 4]),
            forward_mode=types.SimpleNamespace(is_decode_or_idle=lambda: False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            ctl = Path(tmp) / "control.json"
            env = {"DSV4_SERVING_PROBE_CONTROL": str(ctl), "DSV4_SERVING_PROBE_OUT": tmp}
            if trial:
                def fail_finish(result):
                    raise RuntimeError("injected finalization failure")
                fake = types.SimpleNamespace(
                    failed=False, begin=lambda *a: None, wrap=lambda fn: fn,
                    layer_output=lambda *a: None, finish=fail_finish)
                modules["probe_test"] = probe
                modules["dsv4_sparse_trial"] = types.SimpleNamespace(SparseTrial=lambda *a: fake)
                env.update(DSV4_SPARSE_TRIAL_PLAN="fixture", DSV4_SERVING_PROBE_LIGHT="1")
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env), \
                    patch.object(probe, "install_boundaries"):
                probe.install()
                if trial:
                    ctl.write_text(json.dumps({"case": "trial", "enabled": False}))
                    with self.assertRaisesRegex(RuntimeError, "requires enabled"):
                        Model().forward(x, None, batch)
                    ctl.write_text(json.dumps({"case": "trial", "enabled": True}))
                    with self.assertRaisesRegex(RuntimeError, "injected finalization"):
                        Model().forward(x, None, batch)
                    last = json.loads(next(Path(tmp).glob("events.*")).read_text().splitlines()[-1])
                    self.assertEqual(last["error"], "capture-finalization:RuntimeError")
                    self.assertIsNone(probe._active)
                    return
                outputs = []
                for enabled in (False, True):
                    ctl.write_text(json.dumps({"case": str(enabled), "enabled": enabled}))
                    outputs.append(Model().forward(x, None, batch))
                self.assertTrue(torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8)))
                rows = [json.loads(line) for line in next(Path(tmp).glob("events.*")).read_text().splitlines()]
                self.assertEqual(rows[0]["rows"], [])
                ops = {row["op"] for row in rows[1]["rows"]}
                self.assertIn("compressor.aten.aten.mm.default", ops)
                self.assertIn("compressor.aten.aten._to_copy.default", ops)
                self.assertIn("layer.total", ops)
                self.assertTrue(all(r["layer"] == 0 for r in rows[1]["rows"]))
                self.assertIsNone(probe._active)

                with patch.object(probe, "MAX_FORWARD_SECONDS", 0):
                    with self.assertRaisesRegex(RuntimeError, "exceeded 0s budget"):
                        Model().forward(x, None, batch)
                self.assertIsNone(probe._active)
                last = json.loads(next(Path(tmp).glob("events.*")).read_text().splitlines()[-1])
                self.assertEqual(last["error"], "RuntimeError")

                def fail(x):
                    raise ValueError("native failure")

                ck.indexer_logits = probe.wrap_call(fail, "native_wrapper.indexer_logits")
                with self.assertRaisesRegex(RuntimeError, "inside fallback"):
                    Model().forward(x, None, batch)
                self.assertIsNone(probe._active)
                last = json.loads(next(Path(tmp).glob("events.*")).read_text().splitlines()[-1])
                self.assertEqual(last["error"], "caught-instrumented-failure")
                self.assertTrue(last["failures"])

    def test_worker_entry_output_identity_and_caught_failure(self):
        self.exercise_worker()

    def test_trial_finalization_failure_is_logged_and_propagated(self):
        self.exercise_worker(trial=True)


if __name__ == "__main__":
    unittest.main()
