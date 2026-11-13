import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import enablement_gate as gate


class EnablementGateTests(unittest.TestCase):
    """Synthetic raw-observation adapters; no model/GPU/benchmark work."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "source"
        self.repo.mkdir()
        self.bundle = self.base / "bundle"
        self.bundle.mkdir()
        self.output = self.base / "report.json"
        self.output.write_text("previous accepted report\n")
        self.adapter = self.repo / "replay.py"
        self.adapter.write_text(
            "import json, sys\n"
            "request = json.load(sys.stdin)\n"
            "with open(next(iter(request['inputs'].values()))) as stream:\n"
            "    print(stream.read())\n"
        )
        self.git("init", "-q")
        self.git("add", "replay.py")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        self.identity = {
            "model": "synthetic", "source_commit": self.git("rev-parse", "HEAD").strip(),
            "config_sha256": "1" * 64, "hardware_sha256": "2" * 64,
            "runtime_sha256": "3" * 64, "checkpoint_revision": "fixture-weights",
            "reference_revision": "fixture-reference", "dtype": "bf16", "node": "fixture-xeon",
        }
        self.cases = [
            {"op": "linear", "batch": b, "shape": f"M{b}K32N32", "dispatch": "native", "distribution": "signed"}
            for b in (1, 64)
        ]
        self.plan = {
            "schema_version": 1, "gate_sha256": gate.digest(Path(gate.__file__).read_bytes()),
            "identity": self.identity, "batches": [1, 64],
            "ops": ["linear"], "kernel_cases": self.cases, "checks": {},
        }
        self.evidence = {"schema_version": 1, "identity": self.identity, "phases": {}}
        self.responses = {}
        for phase, names in gate.PHASES.items():
            self.plan["checks"][phase] = {}
            checks = {}
            for name in names:
                roles = ["candidate", "baseline", "calibration"] if name == "performance" else (
                    ["candidate", "baseline"] if name == "peer" else ["candidate", "reference"] if name == "correctness" else ["data"]
                )
                if name == "infrastructure":
                    roles = ["inventory", "execution_plan"]
                spec = {
                    "adapter": {"path": "replay.py", "sha256": gate.digest(self.adapter.read_bytes())},
                    "inputs": roles, "cases": copy.deepcopy(self.cases), "timeout_seconds": 5,
                    "limits": {metric: bound for metric, (_, bound) in gate.METRICS[name].items()},
                }
                if name in ("performance", "peer"):
                    spec["baseline_identity"] = self.identity
                if name == "performance":
                    spec["target_ms"] = [8, 8]
                self.plan["checks"][phase][name] = spec
                observations = []
                for case in self.cases:
                    if name in ("performance", "peer"):
                        observations.append({
                            "case": case,
                            "samples": [
                                {
                                    side: {"identity": self.identity, "process_id": f"{side}-{i}", "ms": 10 + i}
                                    for side in ("candidate", "baseline")
                                }
                                for i in range(3)
                            ],
                        })
                    else:
                        observations.append({
                            "case": case,
                            "metrics": {key: bound for key, (_, bound) in gate.METRICS[name].items()},
                        })
                response = {"identity": self.identity, "phase": phase, "check": name, "observations": observations}
                self.responses[(phase, name)] = response
                checks[name] = {
                    "inputs": {role: self.artifact(f"{phase}-{name}-{role}.json", response) for role in roles},
                    "negative_controls": [{
                        "name": "missing-joint-coverage",
                        "inputs": {
                            role: self.artifact(f"{phase}-{name}-{role}-missing.json", {**response, "observations": []})
                            for role in roles
                        },
                        "expected_error": "missing observation coordinates",
                    }],
                }
            self.evidence["phases"][phase] = {
                "checks": checks,
                "audit": {"checks_sha256": "", "reviewer": "fixture-auditor",
                          "goal_alignment": "Synthetic Xeon batch1-64 acceptance control, not model proof.", "findings": []},
            }
        self.plan_path = self.base / "plan.json"
        self.evidence_path = self.bundle / "evidence.json"
        self.approvals_path = self.base / "approvals.json"
        self.seal()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def artifact(self, name, value):
        data = json.dumps(value).encode()
        (self.bundle / name).write_bytes(data)
        return {"path": name, "sha256": gate.digest(data)}

    def seal(self, through="final"):
        self.plan_path.write_text(json.dumps(self.plan))
        self.plan_hash = gate.digest(self.plan_path.read_bytes())
        phases = list(gate.PHASES)[:list(gate.PHASES).index(through) + 1]
        evidence = copy.deepcopy(self.evidence)
        evidence["phases"] = {p: evidence["phases"][p] for p in phases}
        for phase in phases:
            evidence["phases"][phase]["audit"]["checks_sha256"] = gate.checks_digest(self.plan_hash, evidence, phase)
        self.evidence_path.write_text(json.dumps(evidence))
        approvals = {
            "schema_version": 1,
            "approvals": {
                p: {"decision": "APPROVED", "reviewer": "external-fixture-reviewer",
                    "evidence_sha256": gate.approval_digest(self.plan_hash, evidence, p)}
                for p in phases if p in gate.REVIEW_PHASES
            },
        }
        self.approvals_path.write_text(json.dumps(approvals))
        self.approvals_hash = gate.digest(self.approvals_path.read_bytes())

    def run_gate(self, through="final", approvals=True):
        cmd = [
            sys.executable, str(Path(gate.__file__)), "--repo", str(self.repo),
            "--plan", str(self.plan_path), "--plan-sha256", self.plan_hash,
            "--evidence", str(self.evidence_path), "--through", through, "--output", str(self.output),
        ]
        if approvals:
            cmd += ["--approvals", str(self.approvals_path), "--approvals-sha256", self.approvals_hash]
        return subprocess.run(cmd, capture_output=True, text=True)

    def assert_blocked(self, result, message):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertEqual(self.output.read_text(), "previous accepted report\n")

    def replace_response(self, phase, name, response):
        entry = self.evidence["phases"][phase]["checks"][name]
        entry["inputs"] = {
            key: self.artifact(f"{phase}-{name}-{key}.json", response) for key in entry["inputs"]
        }
        self.seal()

    def test_full_gate_and_phase_prefix_positive_controls(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["verdict"], "GREEN")
        observation = report["checks"]["standalone"]["performance"]["observations"][0]
        self.assertAlmostEqual(observation["metrics"]["eff_abs"], 8 / 11)
        self.assertEqual(len(observation["samples"]), 3)
        self.seal("discovery")
        result = self.run_gate("discovery", approvals=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.output.read_text())["verdict"], "PHASE_PASS")

    def test_required_phase_check_and_joint_domain_cannot_be_removed(self):
        original = copy.deepcopy(self.plan)
        mutations = [
            (lambda p: p["checks"].pop("bringup"), "plan phases"),
            (lambda p: p["checks"]["standalone"].pop("fusion"), "standalone checks"),
            (lambda p: p["checks"]["discovery"].pop("infrastructure"), "discovery checks"),
            (lambda p: p["checks"]["discovery"]["infrastructure"]["inputs"].pop(), "missing raw input roles"),
            (lambda p: p["checks"]["standalone"]["native"]["cases"].pop(), "joint coverage"),
            (lambda p: p["batches"].append(65), "1..64"),
            (lambda p: p["checks"]["standalone"]["performance"]["limits"].update(eff_abs=0.1), "weakened"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                self.plan = copy.deepcopy(original)
                mutate(self.plan)
                self.seal()
                self.assert_blocked(self.run_gate(), message)

    def test_numeric_failures_on_both_sides_and_duplicate_processes(self):
        original = self.responses[("standalone", "performance")]
        mutations = [
            (lambda r: r["observations"][0]["samples"][0]["baseline"].update(ms=float("nan")), "non-finite"),
            (lambda r: r["observations"][0]["samples"][0]["candidate"].update(ms=0), "positive"),
            (lambda r: r["observations"][0]["samples"][0]["candidate"].update(ms=True), "finite"),
            (lambda r: r["observations"][0]["samples"].pop(), ">=3"),
            (lambda r: r["observations"][0]["samples"][1]["baseline"].update(process_id="baseline-0"), "duplicate baseline"),
            (lambda r: [s["candidate"].update(ms=20) for s in r["observations"][0]["samples"]], "eff_abs"),
            (lambda r: [s["baseline"].update(ms=9) for s in r["observations"][0]["samples"]], "speedup"),
            (lambda r: [s["candidate"].update(ms=1) for s in r["observations"][0]["samples"]], "target/model review"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                response = copy.deepcopy(original)
                mutate(response)
                self.replace_response("standalone", "performance", response)
                self.assert_blocked(self.run_gate(), message)

    def test_infrastructure_gaps_staleness_and_overcommit_block_discovery(self):
        for metric in ("missing", "stale", "overcommitted"):
            with self.subTest(metric=metric):
                response = copy.deepcopy(self.responses[("discovery", "infrastructure")])
                response["observations"][0]["metrics"][metric] = 1
                self.replace_response("discovery", "infrastructure", response)
                self.assert_blocked(self.run_gate(), f"infrastructure.{metric}")

    def test_identity_coordinate_and_missing_evidence_rejections(self):
        original = self.responses[("standalone", "correctness")]
        for mutate, message in (
            (lambda r: r["identity"].update(config_sha256="9" * 64), "identity mismatch"),
            (lambda r: r["observations"][0]["case"].update(batch=2), "coordinate"),
            (lambda r: r["observations"].pop(), "missing observation"),
            (lambda r: r["observations"].append(r["observations"][0]), "duplicate observation"),
            (lambda r: r["observations"][0]["metrics"].update(violations=1), "violations"),
        ):
            response = copy.deepcopy(original)
            mutate(response)
            self.replace_response("standalone", "correctness", response)
            self.assert_blocked(self.run_gate(), message)
        self.replace_response("standalone", "correctness", copy.deepcopy(original))
        (self.bundle / "standalone-correctness-candidate.json").write_text("{}")
        self.assert_blocked(self.run_gate(), "digest mismatch")

    def test_audits_and_external_approvals_are_required_and_bound(self):
        self.assert_blocked(self.run_gate(approvals=False), "required human approvals")
        evidence = json.loads(self.evidence_path.read_text())
        evidence["phases"]["discovery"]["audit"]["checks_sha256"] = "0" * 64
        self.evidence_path.write_text(json.dumps(evidence))
        self.assert_blocked(self.run_gate(), "stale self-audit")
        self.seal()
        approvals = json.loads(self.approvals_path.read_text())
        approvals["approvals"]["standalone"]["evidence_sha256"] = "0" * 64
        self.approvals_path.write_text(json.dumps(approvals))
        self.approvals_hash = gate.digest(self.approvals_path.read_bytes())
        self.assert_blocked(self.run_gate(), "stale human approval")
        self.seal()
        ref = self.evidence["phases"]["discovery"]["checks"]["scope"]["inputs"]["data"]
        self.evidence["phases"]["discovery"]["audit"]["findings"] = [
            {"issue": "missing branch", "status": "OPEN", "resolution": "pending", "evidence": ref}
        ]
        self.seal()
        self.assert_blocked(self.run_gate(), "unresolved audit finding")

    def test_negative_controls_must_reject_for_the_expected_reason(self):
        entry = self.evidence["phases"]["discovery"]["checks"]["scope"]
        original = copy.deepcopy(entry)
        entry["negative_controls"] = []
        self.seal()
        self.assert_blocked(self.run_gate(), "missing adversarial controls")
        entry.update(copy.deepcopy(original))
        entry["negative_controls"][0]["expected_error"] = "different failure"
        self.seal()
        self.assert_blocked(self.run_gate(), "wrong reason")
        entry.update(copy.deepcopy(original))
        good = self.artifact("not-actually-negative.json", self.responses[("discovery", "scope")])
        entry["negative_controls"][0]["inputs"] = {"data": good}
        self.seal()
        self.assert_blocked(self.run_gate(), "negative control was accepted")

    def test_trusted_plan_source_and_adapter_failures(self):
        self.plan_path.write_text(self.plan_path.read_text() + " ")
        self.assert_blocked(self.run_gate(), "digest mismatch")
        self.seal()
        self.adapter.write_text("raise SystemExit(3)\n")
        self.assert_blocked(self.run_gate(), "worktree is dirty")
        self.git("add", "replay.py")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "failing adapter")
        new_identity = {**self.identity, "source_commit": self.git("rev-parse", "HEAD").strip()}
        self.plan["identity"] = new_identity
        self.evidence["identity"] = new_identity
        for phase in self.plan["checks"].values():
            for spec in phase.values():
                spec["adapter"]["sha256"] = gate.digest(self.adapter.read_bytes())
        self.seal()
        self.assert_blocked(self.run_gate(), "adapter exit 3")

    def test_duplicate_json_keys_and_legacy_certificate_cannot_pass(self):
        self.evidence_path.write_text('{"schema_version":1,"schema_version":1}')
        self.assert_blocked(self.run_gate(), "duplicate JSON key")
        self.evidence_path.write_text('{"verdict":"GREEN"}')
        self.assert_blocked(self.run_gate(), "unexpected/missing fields")

    def test_alias_escape_and_rejection_without_existing_output(self):
        entry = self.evidence["phases"]["bringup"]["checks"]["correctness"]
        entry["inputs"]["reference"] = entry["inputs"]["candidate"]
        self.seal()
        self.assert_blocked(self.run_gate(), "alias the same artifact")
        entry = self.evidence["phases"]["discovery"]["checks"]["scope"]
        entry["inputs"]["data"]["path"] = "../plan.json"
        self.seal()
        self.assert_blocked(self.run_gate(), "outside-root")
        self.output.unlink()
        result = self.run_gate()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.output.exists())

    def rebind_adapter(self, source):
        self.adapter.write_text(source)
        self.git("add", "replay.py")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "adapter variant")
        self.identity["source_commit"] = self.git("rev-parse", "HEAD").strip()
        for phase in self.plan["checks"].values():
            for spec in phase.values():
                spec["adapter"]["sha256"] = gate.digest(self.adapter.read_bytes())

    def test_timeout_is_failure_not_a_negative_control_pass(self):
        self.rebind_adapter("import time\ntime.sleep(5)\n")
        self.plan["checks"]["discovery"]["scope"]["timeout_seconds"] = 0.05
        self.seal("discovery")
        self.assert_blocked(self.run_gate("discovery", approvals=False), "adapter timeout")

    def test_artifact_mutation_during_replay_cannot_publish(self):
        self.rebind_adapter(
            "import json,sys\nfrom pathlib import Path\n"
            "request=json.load(sys.stdin)\n"
            "path=Path(next(iter(request['inputs'].values())))\n"
            "reply=json.loads(path.read_text())\n"
            "reply['identity']=request['identity']\n"
            "print(json.dumps(reply))\n"
            "path.write_text('changed after evaluation')\n"
        )
        self.seal("discovery")
        self.assert_blocked(self.run_gate("discovery", approvals=False), "digest mismatch")

    def test_baseline_identity_and_peer_threshold_are_not_producer_flags(self):
        response = copy.deepcopy(self.responses[("standalone", "performance")])
        response["observations"][0]["samples"][0]["baseline"]["identity"] = {**self.identity, "node": "other-node"}
        self.replace_response("standalone", "performance", response)
        self.assert_blocked(self.run_gate(), "sample identity mismatch")
        self.replace_response("standalone", "performance", self.responses[("standalone", "performance")])
        response = copy.deepcopy(self.responses[("final", "peer")])
        for sample in response["observations"][0]["samples"]:
            sample["baseline"]["ms"] = 9.899
        self.replace_response("final", "peer", response)
        self.assert_blocked(self.run_gate(), "eff_rel")


if __name__ == "__main__":
    unittest.main()
