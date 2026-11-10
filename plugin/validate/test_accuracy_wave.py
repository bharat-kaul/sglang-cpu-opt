import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class AccuracyWaveTests(unittest.TestCase):
    def run_wave(self, fail_shard=None, loader_limit=2):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cpu").mkdir()
            (root / "load_slots").mkdir()
            (root / "cpu_loader_limit").write_text(f"{loader_limit}\n")
            source = root / "source/plugin/validate"
            source.mkdir(parents=True)
            startup = root / "bash_env"
            startup.write_text("source() { :; }\nsleep() { command sleep 0.02; }\n")
            worker = root / "worker.py"
            worker.write_text(
                "import os,sys,time\n"
                "from pathlib import Path\n"
                "out=Path(sys.argv[1])/'cpu'/('shard_'+sys.argv[3])\n"
                "(out/'start').write_text(str(time.monotonic()))\n"
                "time.sleep(0.1)\n"
                "if sys.argv[3]==os.environ.get('TEST_FAIL_SHARD'): sys.exit(7)\n"
                "(out/'ready').write_text(str(time.monotonic()))\n"
                "(out/'runtime.json').write_text('{}')\n"
                "time.sleep(0.5)\n"
                "(out/'end').write_text(str(time.monotonic()))\n"
                "(out/'complete').touch()\n"
            )
            (source / "run_accuracy_shard.sh").write_text(
                '#!/bin/bash\nexec "$TEST_PYTHON" "$1/worker.py" "$@"\n'
            )
            launcher = Path(__file__).with_name("run_accuracy_emr_wave.sbatch")
            processes = []
            logs = []
            try:
                for shard in range(loader_limit + 2):
                    env = dict(os.environ, BASH_ENV=str(startup), TEST_PYTHON=sys.executable,
                               SLURM_ARRAY_TASK_ID=str(shard), SLURM_JOB_ID=str(100 + shard))
                    if fail_shard is not None:
                        env["TEST_FAIL_SHARD"] = str(fail_shard)
                    log = (root / f"log{shard}").open("w")
                    logs.append(log)
                    processes.append(subprocess.Popen(
                        ["bash", str(launcher), str(root)], env=env, stdout=log,
                        stderr=subprocess.STDOUT,
                    ))
                codes = [process.wait(timeout=20) for process in processes]
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                for log in logs:
                    log.close()
            self.assertEqual(list((root / "load_slots").iterdir()), [])
            for shard, code in enumerate(codes):
                expected = 7 if shard == fail_shard else 0
                self.assertEqual(code, expected, (root / f"log{shard}").read_text())
                self.assertEqual(int((root / f"cpu/shard_{shard}/exit_code").read_text()), expected)
            if fail_shard is None:
                loading, live = [], []
                for shard in range(loader_limit + 2):
                    directory = root / f"cpu/shard_{shard}"
                    start, ready, end = [
                        float((directory / name).read_text()) for name in ("start", "ready", "end")
                    ]
                    loading.extend(((start, 1), (ready, -1)))
                    live.extend(((start, 1), (end, -1)))
                def maximum(events):
                    count, peak = 0, 0
                    for _, change in sorted(events):
                        count += change
                        peak = max(peak, count)
                    return peak
                self.assertEqual(maximum(loading), loader_limit)
                self.assertGreaterEqual(maximum(live), loader_limit + 1)

    def test_two_loaders_but_parallel_generation(self):
        self.run_wave()

    def test_failed_loader_releases_slot_and_propagates_exit(self):
        self.run_wave(fail_shard=1)

    def test_four_loaders_but_parallel_generation(self):
        self.run_wave(loader_limit=4)


if __name__ == "__main__":
    unittest.main()
