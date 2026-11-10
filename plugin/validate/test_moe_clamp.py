"""Exercise the gated packed path, including tiny-GEMM and BRGEMM regimes."""
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


MODULE = runpy.run_path(str(Path(__file__).parents[1] / "intel_cpu_models/_dsv4_moe_clamp.py"))


class MoEClampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kernel = MODULE["load"]()

    def test_gate_off_last_all_and_invalid(self):
        for flag, expected in (("0", (False, False)), ("last", (False, True)), ("1", (True, True))):
            with patch.dict(os.environ, {MODULE["FLAG"]: flag}):
                self.assertEqual(tuple(MODULE["enabled"](layer) for layer in (3, 42)), expected)
        with patch.dict(os.environ, {MODULE["FLAG"]: "typo"}):
            with self.assertRaises(ValueError):
                MODULE["enabled"](42)

    def test_signed_split_activation_not_gpt_oss(self):
        torch.manual_seed(7)
        x = (torch.randn(7, 128) * 20).bfloat16()
        g, u = x.float().chunk(2, dim=-1)
        expected = (torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10)).bfloat16()
        actual = self.kernel.activation(x, 10.)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
        self.assertFalse(torch.equal(actual, (torch.nn.functional.silu(g) * u).bfloat16()))
        for limit in (0., -10., float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                self.kernel.activation(x, limit)

    def test_packed_tiny_and_brgemm_keep_inactive_clamp_equal(self):
        torch.manual_seed(3)
        e, n, k = 4, 64, 64
        w1 = torch.randint(0, 256, (e, 2 * n, k // 2), dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8)
        s1 = torch.full((e, 2 * n, k // 32), 125, dtype=torch.uint8)
        s2 = torch.full((e, k, n // 32), 125, dtype=torch.uint8)
        layer = SimpleNamespace(
            w13_weight=torch.ops.sgl_kernel.convert_weight_packed(w1),
            w2_weight=torch.ops.sgl_kernel.convert_weight_packed(w2),
            w13_weight_scale=torch.ops.sgl_kernel.convert_scale_packed(s1),
            w2_weight_scale=torch.ops.sgl_kernel.convert_scale_packed(s2),
            moe_runner_config=SimpleNamespace(
                activation="silu", gemm1_alpha=None, gemm1_clamp_limit=None, swiglu_limit=10.))
        for m in (1, 3, 64):
            with self.subTest(m=m):
                x = (torch.randn(m, k) * 4).bfloat16()
                ids = torch.stack((torch.arange(m) % e, (torch.arange(m) + 1) % e), dim=-1).int()
                weights = torch.full((m, 2), 0.5)
                args = (x, layer.w13_weight, layer.w2_weight, weights, ids, False, 4,
                        layer.w13_weight_scale, layer.w2_weight_scale, None, None, None,
                        None, None, None)
                original = torch.ops.sgl_kernel.fused_experts_cpu(*args, None, True, "silu")
                inactive = self.kernel.fused_experts(*args, 1e6, True, "silu")
                self.assertTrue(torch.equal(original, inactive), f"native donor drift for M={m}")
                corrected = MODULE["fused_experts"](x, layer, weights, ids)
                self.assertTrue(torch.isfinite(corrected).all())
                self.assertFalse(torch.equal(original, corrected))


if __name__ == "__main__":
    unittest.main()
