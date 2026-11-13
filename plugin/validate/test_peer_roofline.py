import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PeerRooflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "plugin/validate"
        self.tools.mkdir(parents=True)
        for name in ("peer_roofline.py", "enablement_gate.py"):
            shutil.copyfile(Path(__file__).with_name(name), self.tools / name)
        self.profile = {"part": "synthetic", "achievable": {
            "streamed_gemm_tflops_per_socket": 10, "streamed_gemm_estimated": False,
        }}
        self.config = {"hidden_size": 32, "num_attention_heads": 2, "intermediate_size": 64}
        self.baselines = {
            "tolerances": {"eff_abs_min": 0.7, "eff_rel_min": 0.9},
            "per_op_efficiency": {"dense_gemm": [
                {"layer": role, "donor": "fixture", "M": 64, "K": k, "N": n, "measured": True, "tflops": 8}
                for role, k, n in (("qkv", 32, 96), ("o", 32, 32), ("gate_up", 32, 128), ("down", 64, 32))
            ]},
        }
        self.measured = {role: 7.2 for role in ("qkv", "o", "gate_up", "down")}

    def run_tool(self, raw=False):
        for name, value in (
            ("profile", self.profile), ("config", self.config),
            ("baselines", self.baselines), ("measured", self.measured),
        ):
            (self.root / f"{name}.json").write_text(json.dumps(value))
        cmd = [
            sys.executable, str(self.tools / "peer_roofline.py"),
            "--profile", str(self.root / "profile.json"),
            "--config", str(self.root / "config.json"),
            "--baselines", str(self.root / "baselines.json"), "--m", "64",
        ]
        cmd += ["--run"] if raw else ["--measured-json", str(self.root / "measured.json")]
        return subprocess.run(cmd, capture_output=True, text=True)

    def assert_failure(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(result.stdout or result.stderr)
        self.assertNotEqual(record["overall_verdict"], "PASS")

    def test_both_thresholds_pass_but_rounding_cannot_hide_failure(self):
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["overall_verdict"], "PASS")
        self.measured["qkv"] = 7.199
        result = self.run_tool()
        self.assert_failure(result)
        self.assertEqual(json.loads(result.stdout)["ops"][0]["verdict"], "FAIL_REL(wiring)")
        for donor in self.baselines["per_op_efficiency"]["dense_gemm"]:
            donor["tflops"] = 5
        self.measured = {role: 4.5 for role in self.measured}
        result = self.run_tool()
        self.assert_failure(result)
        self.assertEqual(json.loads(result.stdout)["ops"][0]["verdict"], "FAIL_ABS")

    def test_missing_measurements_donors_proxy_and_wrong_shape_fail(self):
        original = copy.deepcopy(self.baselines)
        for mutation in (
            lambda b: b["per_op_efficiency"].clear(),
            lambda b: b["per_op_efficiency"]["dense_gemm"][0].update(M=2048),
            lambda b: b["per_op_efficiency"]["dense_gemm"][0].update(K=64),
            lambda b: b["per_op_efficiency"]["dense_gemm"][0].update(fidelity="proxy"),
            lambda b: b["per_op_efficiency"]["dense_gemm"][0].update(measured="true"),
        ):
            self.baselines = copy.deepcopy(original)
            mutation(self.baselines)
            self.assert_failure(self.run_tool())
        self.baselines = original
        self.measured.pop("qkv")
        self.assert_failure(self.run_tool())

    def test_invalid_numbers_on_candidate_donor_target_and_threshold(self):
        original = (copy.deepcopy(self.measured), copy.deepcopy(self.baselines), copy.deepcopy(self.profile))
        for value in (float("nan"), float("inf"), 0, -1, True):
            for side in ("candidate", "donor", "target", "threshold"):
                with self.subTest(side=side, value=value):
                    self.measured, self.baselines, self.profile = copy.deepcopy(original)
                    if side == "candidate":
                        self.measured["qkv"] = value
                    elif side == "donor":
                        self.baselines["per_op_efficiency"]["dense_gemm"][0]["tflops"] = value
                    elif side == "target":
                        self.profile["achievable"]["streamed_gemm_tflops_per_socket"] = value
                    else:
                        self.baselines["tolerances"]["eff_abs_min"] = value
                    self.assert_failure(self.run_tool())

    def test_provisional_ceiling_blocks_and_child_failure_propagates(self):
        self.profile["achievable"]["streamed_gemm_estimated"] = True
        self.assert_failure(self.run_tool())
        self.profile["achievable"]["streamed_gemm_estimated"] = False
        fake = self.root / "tools/gemm_amx_bf16.py"
        fake.parent.mkdir()
        fake.write_text('print(\'{"tflops": 7.2}\')\nraise SystemExit(3)\n')
        result = self.run_tool(raw=True)
        self.assert_failure(result)
        self.assertIn("non-zero exit status 3", result.stderr)

    def test_duplicate_mapping_keys_are_not_silently_overwritten(self):
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.root / "profile.json").write_text(
            '{"part":"synthetic","achievable":{},'
            '"achievable":{"streamed_gemm_tflops_per_socket":10,"streamed_gemm_estimated":false}}'
        )
        result = subprocess.run(result.args, capture_output=True, text=True)
        self.assert_failure(result)
        self.assertIn("duplicate YAML key", result.stderr)


if __name__ == "__main__":
    unittest.main()
