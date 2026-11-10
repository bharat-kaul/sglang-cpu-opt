import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "shared_regression", Path(__file__).parents[1] / "dsv4_layer_capture/shared_regression.py"
)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class Mode:
    def __init__(self, decode=False):
        self.decode = decode
        self.name = "DECODE" if decode else "EXTEND"

    def is_extend(self):
        return not self.decode

    def is_decode(self):
        return self.decode


class CaptureTests(unittest.TestCase):
    def test_precision_diagnostic_requires_shared_clamp_and_valid_flag(self):
        import runpy
        path = Path(__file__).parents[1] / "intel_cpu_models/_dsv4_shared_clamp.py"
        loader = runpy.run_path(str(path))
        for clamp, diagnostic in (("0", "1"), ("1", "invalid")):
            with patch.dict("os.environ", {"INTEL_CPU_DSV4_SHARED_CLAMP": clamp,
                                          "INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC": diagnostic}):
                with self.assertRaises(ValueError):
                    loader["install_op_swap"]()

    def test_precision_operand_restore_and_gate_partition_order(self):
        path = Path(__file__).with_name("replay_shared_precision_factorial.py")
        with patch.dict(sys.modules, {"compare_shared_regression_gpu": SimpleNamespace(metrics=None)}):
            spec = importlib.util.spec_from_file_location("precision_factorial", path)
            precision = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(precision)
        q = torch.full((2, 128), -2.0).to(torch.float8_e4m3fn)
        data = {"quantized_input": q, "input_scale": torch.full((2, 1), 0.125)}
        restored, wide = precision.restored_activation(data)
        self.assertTrue(torch.equal(restored, torch.full((2, 128), -0.25, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(restored.float(), wide))
        joined = precision.join_gate([torch.arange(4).view(4, 1), torch.arange(4, 8).view(4, 1)])
        self.assertEqual(joined.flatten().tolist(), [0, 1, 4, 5, 2, 3, 6, 7])
        self.assertTrue(precision.same_fp8(q, q.clone()))
        with self.assertRaisesRegex(RuntimeError, "layout"):
            precision.restored_activation({**data, "input_scale": torch.ones(2, 2)})
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            precision.restored_activation({**data, "input_scale": torch.zeros(2, 1)})

    def test_quant_tap_preserves_native_call_and_restores_on_failure(self):
        path = Path(__file__).parents[1] / "shared_regression_gpu/native_quant.py"
        spec = importlib.util.spec_from_file_location("native_quant", path)
        quant = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quant)
        output = torch.ones(2, 128, dtype=torch.bfloat16)
        args = (torch.ones(2, 128).to(torch.float8_e4m3fn),
                torch.ones(128, 128).to(torch.float8_e4m3fn),
                torch.ones(2, 1), torch.ones(1, 1), [128, 128])

        def native(*a, **k):
            self.assertIs(a[0], args[0])
            return output

        utils = SimpleNamespace(w8a8_block_fp8_matmul_deepgemm=native,
                                w8a8_block_fp8_matmul_triton=native)
        modules = {name: SimpleNamespace() for name in
                   ("sglang", "sglang.srt", "sglang.srt.layers")}
        modules["sglang.srt.layers.quantization"] = SimpleNamespace(fp8_utils=utils)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            stage = {}
            with quant.capture_gemm(stage, "gate", root, 0, 0, "test", {}):
                self.assertIs(utils.w8a8_block_fp8_matmul_deepgemm(*args), output)
            self.assertIs(utils.w8a8_block_fp8_matmul_deepgemm, native)
            self.assertTrue(torch.equal(stage["gate_gemm"]["output"], output))
            saved = torch.load(root / "weights.L0.gate.pt", weights_only=False)
            self.assertTrue(torch.equal(saved["weight"].view(torch.uint8), args[1].view(torch.uint8)))
            with self.assertRaisesRegex(RuntimeError, "was not observed"):
                with quant.capture_gemm({}, "down", root, 0, 0, "test", {}):
                    pass
            self.assertIs(utils.w8a8_block_fp8_matmul_triton, native)
            invalid = list(args)
            invalid[2] = torch.full((2, 1), float("nan"))
            with self.assertRaisesRegex(RuntimeError, "invalid native"):
                with quant.capture_gemm({}, "down", root, 0, 0, "test", {}):
                    utils.w8a8_block_fp8_matmul_deepgemm(*invalid)
            self.assertIs(utils.w8a8_block_fp8_matmul_deepgemm, native)
            self.assertFalse((root / "weights.L0.down.pt").exists())

    def test_chunk_identity_native_output_and_pre_mutation_operands(self):
        def native(h, w1, w2, routed, *args):
            h.add_(routed)
            return h

        class MoE:
            def __init__(self, layer):
                self.layer_id = layer
                self.shared_experts = SimpleNamespace(swiglu_limit=10.0)

            def forward_cpu(self, h):
                w = torch.ones(2, 2)
                return torch.ops.sgl_kernel.shared_expert_cpu(
                    h, w, w, torch.full_like(h, 2), 1.5, True,
                    False, True, w, w, [128, 128], True,
                )

        class Model:
            def forward(self, ids, positions, batch):
                return [MoE(i).forward_cpu(torch.ones(len(ids), 2)) for i in range(43)]

        def batch(prefix, length, decode=False):
            return SimpleNamespace(
                forward_mode=Mode(decode), req_pool_indices=torch.tensor([7]),
                extend_prefix_lens_cpu=[prefix], extend_seq_lens_cpu=[length],
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"q": 1175, "prompt_ids": [10, 20, 30],
                                        "common_generated_ids": [40], "decisions": [0, 1]}))
            with patch.object(torch.ops.sgl_kernel, "shared_expert_cpu", native, create=True):
                capture.install(Model, MoE, str(plan), str(root / "captures"))
                model = Model()
                model.forward(torch.tensor([99, 20, 30]), torch.tensor([0, 1, 2]), batch(0, 3))
                self.assertEqual(list((root / "captures").iterdir()), [])
                model.forward(torch.tensor([10, 20]), torch.tensor([0, 1]), batch(0, 2))
                self.assertEqual(list((root / "captures").iterdir()), [])
                outputs = model.forward(torch.tensor([30]), torch.tensor([2]), batch(2, 1))
                self.assertTrue(all(torch.equal(x, torch.full_like(x, 3)) for x in outputs))
                model.forward(torch.tensor([40]), torch.tensor([3]), batch(0, 1, True))
                self.assertIs(torch.ops.sgl_kernel.shared_expert_cpu, native)
            files = list((root / "captures").glob("d*.pt"))
            self.assertEqual(len(files), 86)
            data = torch.load(root / "captures/d1.L42.pt", weights_only=False)
            self.assertEqual(data["sequence_ids"], [10, 20, 30, 40])
            self.assertEqual(data["target_row"], 0)
            self.assertTrue(torch.equal(data["hidden"], torch.ones(1, 2)))
            self.assertTrue(torch.equal(data["combined"], torch.full((1, 2), 3.0)))
            self.assertEqual(len(list((root / "captures").glob("weights.*"))), 43)

    def test_gpu_comparison_rejects_invalid_outputs(self):
        path = Path(__file__).with_name("compare_shared_regression_gpu.py")
        spec = importlib.util.spec_from_file_location("compare_shared", path)
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        a = torch.ones(2, 4, dtype=torch.bfloat16)
        self.assertEqual(compare.metrics(a, a.clone(), 1)["target_rel_l2"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            compare.metrics(a, torch.full_like(a, float("nan")), 1)
        with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
            compare.metrics(a, a.float(), 1)


if __name__ == "__main__":
    unittest.main()
