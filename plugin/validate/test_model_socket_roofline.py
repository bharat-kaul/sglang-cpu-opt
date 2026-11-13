import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import model_socket_roofline as model


class AnalyticalTargetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = json.loads((model.RESULTS / "dsv4_socket_roofline_inputs.json").read_text())

    def test_routing_and_tile_work(self):
        self.assertEqual(model.distinct_experts(32, "historical59"), 59)
        self.assertGreater(model.distinct_experts(32, "uniform"), 59)
        self.assertGreater(model.distinct_experts(256, "historical59"), 200)
        self.assertEqual(model.padded_assignments(32, 59), 59 * 16)
        for m in (1, 32, 256):
            self.assertGreaterEqual(model.padded_assignments(m, model.distinct_experts(m, "uniform")), m * 6)
        for m in (0, -1, True, 32.5):
            with self.assertRaises(ValueError):
                model.distinct_experts(m, "uniform")

    def test_schedule_logits_and_resource_accounting(self):
        inputs = self.inputs
        prefill = model.phase_budget(inputs, "emr", "prefill", model.SCENARIOS["central"])
        decode = model.phase_budget(inputs, "gnr", "decode", model.SCENARIOS["central"])
        self.assertEqual(prefill["socket_tokens"], 2 * 32 * 128)
        self.assertEqual(decode["socket_tokens"], 3 * 32 * 4)
        self.assertEqual(prefill["groups"]["head_embedding"]["flops"],
                         2 * 2 * 4096 * 129280 * 2 * 16)
        self.assertEqual(prefill["groups"]["mhc_head_fp32"]["flops"],
                         2 * 256 * 4 * 16384 * 2 * 16)
        self.assertAlmostEqual(prefill["phase_seconds"],
                               sum(r["seconds"] for r in prefill["groups"].values()))
        slower = model.phase_budget(inputs, "emr", "decode",
                                    {**model.SCENARIOS["central"], "weight_passes": 2})
        central = model.phase_budget(inputs, "emr", "decode", model.SCENARIOS["central"])
        self.assertLess(slower["tokens_per_second"], central["tokens_per_second"])

    def test_measurement_does_not_define_target(self):
        original = model.evaluate(self.inputs)
        changed = copy.deepcopy(self.inputs)
        for platform in model.LAYOUTS:
            for phase in ("prefill", "decode"):
                changed["measured"][platform][phase]["median"] *= 100
        rerun = model.evaluate(changed)
        self.assertEqual(original["results"], rerun["results"])
        self.assertTrue(all(
            row["disposition"] == "CONTRADICTED_AS_CEILING_BY_EXISTING_MEASUREMENT"
            for point in rerun["comparison"].values() for row in point.values()))

    def test_cli_rejects_changed_schedule_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, output = root / "inputs.json", root / "output.json"
            for field, value in (("schedule", {}), ("compute_commit", "wrong")):
                with self.subTest(field=field):
                    changed = copy.deepcopy(self.inputs)
                    changed[field] = value
                    path.write_text(json.dumps(changed))
                    run = subprocess.run([sys.executable, model.__file__, "--inputs", str(path),
                                          "--output", str(output)], capture_output=True, text=True)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("Input source/schedule mismatch", run.stderr)
                    self.assertFalse(output.exists())
            path.write_text(json.dumps(self.inputs))
            subprocess.run([sys.executable, model.__file__, "--inputs", str(path),
                            "--output", str(output)], check=True, capture_output=True)
            self.assertEqual(set(json.loads(output.read_text())["results"]),
                             {"emr_prefill", "gnr_prefill", "emr_decode", "gnr_decode"})

    def test_invalid_scenario_parameters(self):
        for key in ("compute_fraction", "vector_fraction", "launch_us", "weight_passes", "dequant_lane_ops"):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    model.phase_budget(self.inputs, "emr", "decode",
                                       {**model.SCENARIOS["central"], key: value})


if __name__ == "__main__":
    unittest.main()
