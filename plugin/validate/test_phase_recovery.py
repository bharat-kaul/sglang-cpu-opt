import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

import enablement_gate as gate
import run_enablement_phase as runner
import test_enablement_gate as fixtures


class PhaseRecoveryTests(unittest.TestCase):
    """Real local processes and signals, synthetic raw evidence; no cluster/model jobs."""

    def setUp(self):
        self.fx = fixtures.EnablementGateTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.run_dir = self.fx.base / "run"
        self.control = self.fx.base / "control.json"
        self.hold = self.fx.base / "hold"
        self.started = self.fx.base / "started.json"
        self.trace = self.fx.base / "trace.jsonl"
        self.children = []
        self.addCleanup(self.cleanup_children)
        self.fx.rebind_adapter(
            "import json, os, signal, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "root = Path(__file__).parent.parent\n"
            "request = json.load(sys.stdin)\n"
            "with (root/'trace.jsonl').open('a') as stream:\n"
            "    stream.write(json.dumps([request['phase'],request['check'],os.getpid()])+'\\n')\n"
            "control = json.loads((root/'control.json').read_text())\n"
            "is_negative = 'missing' in next(iter(request['inputs'].values()))\n"
            "if [request['phase'],request['check']] == control['target'] and (not control['negative_only'] or is_negative):\n"
            "    mode = control['mode']\n"
            "    if mode == 'crash':\n"
            "        print(Path(next(iter(request['inputs'].values()))).read_text())\n"
            "        sys.exit(7)\n"
            "    if mode == 'descendant':\n"
            "        child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
            "        (root/'started.tmp').write_text(json.dumps({'pid':os.getpid(),'child':child.pid,'gate':os.getppid()}))\n"
            "        os.replace(root/'started.tmp',root/'started.json')\n"
            "        time.sleep(30)\n"
            "    if mode == 'hold' and (root/'hold').exists():\n"
            "        (root/'started.tmp').write_text(json.dumps({'pid':os.getpid(),'gate':os.getppid()}))\n"
            "        os.replace(root/'started.tmp',root/'started.json')\n"
            "        while (root/'hold').exists(): time.sleep(.01)\n"
            "print(Path(next(iter(request['inputs'].values()))).read_text())\n"
        )
        # Rebuild fixture artifact identities after committing the instrumented test adapter.
        for phase, record in self.fx.evidence["phases"].items():
            for name, entry in record["checks"].items():
                response = self.fx.responses[(phase, name)]
                for role, ref in list(entry["inputs"].items()):
                    entry["inputs"][role] = self.fx.artifact(ref["path"], response)
                for control in entry["negative_controls"]:
                    for role, ref in list(control["inputs"].items()):
                        control["inputs"][role] = self.fx.artifact(ref["path"], {**response, "observations": []})
        self.through = "discovery"
        self.fx.seal(self.through)
        self.mode("pass")

    def cleanup_children(self):
        self.hold.unlink(missing_ok=True)
        for process in self.children:
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
        if self.started.exists():
            pid = json.loads(self.started.read_text())["pid"]
            if self.live(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if (self.run_dir / "lease.lock").exists():
            self.wait_until(self.lease_free, "descendant lease release")

    def mode(self, value, target=("discovery", "scope"), negative_only=False):
        self.control.write_text(json.dumps({"mode": value, "target": list(target), "negative_only": negative_only}))

    def command(self, *extra, approvals=True):
        cmd = [
            sys.executable, "-B", runner.__file__, "--repo", str(self.fx.repo),
            "--plan", str(self.fx.plan_path), "--plan-sha256", self.fx.plan_hash,
            "--evidence", str(self.fx.evidence_path), "--through", self.through,
            "--run-dir", str(self.run_dir),
        ]
        if approvals:
            cmd += ["--approvals", str(self.fx.approvals_path), "--approvals-sha256", self.fx.approvals_hash]
        return cmd + list(extra)

    def run_phase(self, *extra, approvals=True):
        return subprocess.run(self.command(*extra, approvals=approvals), capture_output=True, text=True, timeout=30)

    def launch(self):
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(process)
        return process

    def state(self):
        return json.loads((self.run_dir / "state.json").read_text())

    def traces(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()] if self.trace.exists() else []

    def wait_until(self, predicate, description):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail(f"timed out waiting for {description}")

    @staticmethod
    def live(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
        except FileNotFoundError:
            return False

    def lease_free(self):
        with (self.run_dir / "lease.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        return True

    def test_success_is_immutable_and_cannot_be_relaunched(self):
        result = self.run_phase()
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        report = self.run_dir / state["accepted"]["path"]
        before = report.read_bytes()
        traces = self.traces()
        result = self.run_phase("--resume")
        self.assertEqual(result.returncode, 2)
        self.assertIn("already accepted", result.stderr)
        self.assertEqual(self.traces(), traces)
        self.assertEqual(report.read_bytes(), before)

    def test_signal_after_acceptance_commit_cannot_demote_or_relaunch_phase(self):
        harness = self.fx.base / "interrupt_after_commit.py"
        harness.write_text(
            "import os, signal, sys\n"
            f"sys.path.insert(0, {str(Path(runner.__file__).parent)!r})\n"
            "import run_enablement_phase as runner\n"
            "original = runner.atomic_json\n"
            "def interrupted_commit(path, value):\n"
            "    original(path, value)\n"
            "    if value['status'] == 'PASSED': os.kill(os.getpid(), signal.SIGTERM)\n"
            "runner.atomic_json = interrupted_commit\n"
            "sys.exit(runner.main())\n"
        )
        command = self.command()
        command[2] = str(harness)
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(self.state()["status"], "PASSED")
        traces = self.traces()
        self.assertIn("already accepted", self.run_phase("--resume", "--retry").stderr)
        self.assertEqual(self.traces(), traces)

    def test_crashes_do_not_pass_or_retry_automatically_and_budget_survives_restart(self):
        self.mode("crash")
        for attempt in range(1, 4):
            flags = () if attempt == 1 else ("--resume", "--retry")
            result = self.run_phase(*flags)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertEqual(len(self.state()["attempts"]), attempt)
            self.assertEqual(len(self.traces()), attempt)
            self.assertFalse((self.run_dir / f"attempt-{attempt:04d}" / "report.json").exists())
            self.assertIn("adapter exit 7", (self.run_dir / f"attempt-{attempt:04d}" / "stderr.log").read_text())
        result = self.run_phase("--resume", "--retry")
        self.assertIn("budget exhausted", result.stderr)
        self.assertEqual(len(self.traces()), 3)
        result = self.run_phase("--resume", "--retry", "--max-attempts", "4")
        self.assertIn("budget changed", result.stderr)

    def test_timeout_terminates_adapter_and_descendant(self):
        self.mode("descendant")
        self.fx.plan["checks"]["discovery"]["scope"]["timeout_seconds"] = 0.5
        self.fx.seal(self.through)
        result = self.run_phase()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("adapter timeout", (self.run_dir / "attempt-0001" / "stderr.log").read_text())
        started = json.loads(self.started.read_text())
        self.wait_until(lambda: not self.live(started["pid"]) and not self.live(started["child"]), "timeout cleanup")
        self.assertTrue(self.lease_free())

    def test_sigterm_then_resume_replays_prefix_and_rejects_stale_prior_artifact(self):
        self.through = "bringup"
        self.fx.seal(self.through)
        self.mode("hold", ("bringup", "correctness"), negative_only=True)
        self.hold.touch()
        process = self.launch()
        self.wait_until(self.started.exists, "held bringup check")
        process.terminate()
        process.communicate(timeout=8)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(self.state()["status"], "INTERRUPTED")
        self.wait_until(lambda: not self.live(json.loads(self.started.read_text())["pid"]), "interrupt cleanup")
        self.hold.unlink()
        self.mode("pass")
        self.assertIn("explicit", self.run_phase("--resume").stderr)
        artifact = self.fx.bundle / "discovery-scope-data.json"
        original = artifact.read_bytes()
        artifact.write_text("{}")
        result = self.run_phase("--resume", "--retry")
        self.assertEqual(result.returncode, 2)
        self.assertIn("digest mismatch", (self.run_dir / "attempt-0002" / "stderr.log").read_text())
        artifact.write_bytes(original)
        result = self.run_phase("--resume", "--retry")
        self.assertEqual(result.returncode, 0, result.stderr)
        scope_replays = [entry for entry in self.traces() if entry[:2] == ["discovery", "scope"]]
        self.assertEqual(len(scope_replays), 4)  # Both positive/control pairs; no cached-prefix shortcut.

    def test_sigkill_keeps_lease_until_child_finishes_and_orphan_report_is_not_adopted(self):
        self.mode("hold")
        self.hold.touch()
        process = self.launch()
        self.wait_until(self.started.exists, "active adapter")
        result = self.run_phase("--resume", "--retry")
        self.assertIn("live owner", result.stderr)
        process.kill()
        process.communicate(timeout=5)
        result = self.run_phase("--resume", "--retry")
        self.assertIn("live owner", result.stderr)
        self.assertEqual(len(self.traces()), 1)
        self.hold.unlink()
        self.wait_until(self.lease_free, "orphan gate completion")
        self.assertTrue((self.run_dir / "attempt-0001" / "report.json").exists())
        result = self.run_phase("--resume")
        self.assertIn("explicit", result.stderr)
        self.assertEqual(self.state()["status"], "INTERRUPTED")
        self.mode("pass")
        result = self.run_phase("--resume", "--retry")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.state()["accepted"]["path"], "attempt-0002/report.json")

    def test_missing_rejected_and_renewed_approval_pause_without_spending_attempts(self):
        self.through = "standalone"
        self.fx.seal(self.through)
        result = self.run_phase(approvals=False)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(self.state()["status"], "WAITING_APPROVAL")
        self.assertEqual(self.state()["attempts"], [])
        approved_bytes = self.fx.approvals_path.read_bytes()
        denied = json.loads(approved_bytes)
        denied["approvals"]["standalone"]["decision"] = "REJECTED"
        self.fx.approvals_path.write_text(json.dumps(denied))
        self.fx.approvals_hash = gate.digest(self.fx.approvals_path.read_bytes())
        result = self.run_phase("--resume")
        self.assertEqual(result.returncode, 3)
        self.assertEqual(self.traces(), [])
        self.fx.approvals_path.write_bytes(approved_bytes)
        self.fx.approvals_hash = gate.digest(approved_bytes)
        result = self.run_phase("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_killed_gate_cannot_release_an_active_adapters_lease(self):
        self.mode("hold")
        self.hold.touch()
        process = self.launch()
        self.wait_until(self.started.exists, "active adapter")
        started = json.loads(self.started.read_text())
        os.kill(started["gate"], signal.SIGKILL)
        process.communicate(timeout=5)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(self.state()["status"], "BLOCKED")
        self.assertIn("live owner", self.run_phase("--resume", "--retry").stderr)
        self.assertEqual(len(self.traces()), 1)
        self.hold.unlink()
        self.wait_until(self.lease_free, "orphan adapter completion")
        self.mode("pass")
        result = self.run_phase("--resume", "--retry")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_evidence_requires_new_identity_and_renewed_approval(self):
        self.through = "standalone"
        self.fx.seal(self.through)
        old_approval = self.fx.approvals_path.read_bytes()
        self.assertEqual(self.run_phase(approvals=False).returncode, 3)
        self.fx.evidence["phases"]["discovery"]["audit"]["reviewer"] = "new-evidence-reviewer"
        self.fx.seal(self.through)
        renewed = self.fx.approvals_path.read_bytes()
        result = self.run_phase("--resume")
        self.assertIn("identity changed", result.stderr)
        self.run_dir = self.fx.base / "revised-run"
        self.fx.approvals_path.write_bytes(old_approval)
        self.fx.approvals_hash = gate.digest(old_approval)
        result = self.run_phase()
        self.assertEqual(result.returncode, 3)
        self.assertIn("stale human approval", result.stderr)
        self.assertEqual(self.traces(), [])
        self.fx.approvals_path.write_bytes(renewed)
        self.fx.approvals_hash = gate.digest(renewed)
        self.assertEqual(self.run_phase("--resume").returncode, 0)

    def test_truncated_journal_and_partial_output_never_count_as_completion(self):
        self.mode("crash")
        self.assertEqual(self.run_phase().returncode, 2)
        journal = self.run_dir / "state.json"
        original = journal.read_bytes()
        journal.write_text('{"schema_version":')
        result = self.run_phase("--resume", "--retry")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(len(self.traces()), 1)
        journal.unlink()
        result = self.run_phase()
        self.assertEqual(result.returncode, 2)
        self.assertIn("unjournaled files", result.stderr)
        self.assertEqual(len(self.traces()), 1)
        journal.write_bytes(original)
        (self.run_dir / "attempt-0001" / "report.json").write_text('{"verdict":"GREEN"}')
        self.mode("pass")
        result = self.run_phase("--resume", "--retry")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.state()["accepted"]["path"], "attempt-0002/report.json")


if __name__ == "__main__":
    unittest.main()
