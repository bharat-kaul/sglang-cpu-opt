import copy
import unittest

from calibrate_socket_targets import ITERATIONS, aggregate


class CalibrationTests(unittest.TestCase):
    def rows(self):
        return [
            {"platform": "emr", "rank": rank, "numa": 0, "cpus": list(range(rank * 32, (rank + 1) * 32)),
             "threads": 32, "samples_ns": [100] * ITERATIONS, "start_ns": 10 + rank,
             "end_ns": 1000 + rank, "job": "123", "host": "emr-host", "round": 0,
             "script_sha256": "source", "torch": "torch", "work_per_iteration": 1024,
             "unit": "bytes", "pid": 100 + rank, "case": "triad"}
            for rank in range(2)
        ]

    def test_shared_domain_uses_common_span_not_sum_of_rates(self):
        result = aggregate(self.rows(), "emr", "triad")
        self.assertAlmostEqual(result["socket_rate"], 1024 * ITERATIONS * 2 / (991 / 1e9))

    def test_rejects_mismatched_or_invalid_replicas(self):
        for key, value in (("job", "other"), ("round", 1), ("script_sha256", "other"),
                           ("rank", 0), ("pid", 100), ("numa", 1), ("cpus", [0]),
                           ("threads", 64), ("samples_ns", [0] * ITERATIONS),
                           ("unit", "tokens"), ("work_per_iteration", -1),
                           ("start_ns", 1001)):
            with self.subTest(key=key):
                rows = copy.deepcopy(self.rows())
                rows[1][key] = value
                with self.assertRaises(ValueError):
                    aggregate(rows, "emr", "triad")


if __name__ == "__main__":
    unittest.main()
