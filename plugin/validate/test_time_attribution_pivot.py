import json
from pathlib import Path
import unittest

from time_attribution_pivot import build_markdown


class TimeAttributionFormattingTests(unittest.TestCase):
    def test_units_and_nonzero_durations(self):
        for seconds, unit, displayed in (
            (2.0, "s", "2.000"),
            (0.002, "ms", "2.000"),
            (0.000002, "us", "2.000"),
            (0.000000002, "ns", "2.000"),
            (1e-15, "ns", "1e-06"),
        ):
            with self.subTest(seconds=seconds):
                report = build_markdown({"phases": {"decode": {
                    "wall_s": seconds * 2,
                    "ops": [{"op": "work", "s": seconds}],
                }}})
                self.assertIn(f"time ({unit})", report)
                self.assertIn(f"| work | kernel | {displayed} | 50.0 |", report)
                self.assertIn(f"| other / unattributed | other | {displayed} | 50.0 |", report)

    def test_mixed_scales_and_zero(self):
        report = build_markdown({"phases": {"decode": {
            "wall_s": 1.000002,
            "ops": [{"op": "large", "s": 1.0}, {"op": "small", "s": 0.000002},
                    {"op": "zero", "s": 0.0}],
        }}})
        self.assertIn("time (us)", report)
        self.assertIn("| small | kernel | 2.000 |", report)
        self.assertIn("| zero | kernel | 0.000 |", report)

    def test_sparse_attention_report(self):
        source = Path(__file__).resolve().parents[1] / (
            "kernels/dsa_pilot/sparse_attention_kernel/results/pivot-input.json"
        )
        data = json.loads(source.read_text())
        report = build_markdown(data)
        self.assertIn("time (us)", report)
        self.assertIn("| gather / packing / initialization | kernel | 199.626 | 67.1 |", report)
        self.assertIn("| AMX QK | kernel | 25.622 | 8.6 |", report)
        self.assertIn("| other / unattributed | other | 27.119 | 9.1 |", report)
        self.assertNotIn("| 0.000 |", report)


if __name__ == "__main__":
    unittest.main()
