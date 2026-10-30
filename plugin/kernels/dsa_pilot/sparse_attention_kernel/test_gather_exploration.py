import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from report_gather_exploration import HERE, summarize


class GatherExplorationTests(unittest.TestCase):
    def setUp(self):
        self.runs = [
            json.loads((HERE / "results/gather_exploration" / f"replicate-{i}.json").read_text())
            for i in range(1, 4)
        ]

    def test_actual_evidence(self):
        s = summarize(self.runs)
        self.assertEqual(s["indexed_cold"]["baseline"]["median_us"], 292.106)
        self.assertGreater(s["indexed_cold"]["hardware_gather"]["latency_change_pct"], 10)
        self.assertEqual(s["stage_cases_per_variant_per_process"], 36)

    def test_invalid_evidence(self):
        for mutate in (
            lambda r: r[1].update(pid=r[0]["pid"]),
            lambda r: r[1].update(node="different"),
            lambda r: r[1]["indexed_cold"]["timings"]["hardware_gather"].update(median_us=1),
            lambda r: r[1]["indexed_cold"]["timings"]["baseline"]["samples_us"].__setitem__(0, float("nan")),
            lambda r: r[1]["stage_bitwise_cases"].pop(),
            lambda r: r[1]["build"].update(experiment_sha256="wrong"),
        ):
            with self.subTest(mutation=mutate):
                runs = copy.deepcopy(self.runs)
                mutate(runs)
                with self.assertRaises(ValueError):
                    summarize(runs)

    def test_invalid_cli_has_no_outputs(self):
        with tempfile.TemporaryDirectory(prefix="gather-report-test-") as tmp:
            directory = Path(tmp)
            self.runs[1]["attention_bitwise_cases"].pop()
            for i, run in enumerate(self.runs, 1):
                (directory / f"replicate-{i}.json").write_text(json.dumps(run))
            result = subprocess.run(
                [sys.executable, str(HERE / "report_gather_exploration.py"), "--results", tmp],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Incomplete bitwise", result.stderr)
            self.assertFalse((directory / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
