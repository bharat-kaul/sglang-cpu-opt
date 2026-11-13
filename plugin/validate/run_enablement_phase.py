#!/usr/bin/env python3
"""Single-owner, bounded, restartable LOCAL phase-gate replay; never submits model jobs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import enablement_gate as gate


STATES = {"READY", "RUNNING", "WAITING_APPROVAL", "BLOCKED", "INTERRUPTED", "PASSED"}


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_state(path, binding, max_attempts):
    state = gate.load_json(path)
    gate.exact(state, {"schema_version", "binding", "max_attempts", "status", "attempts", "reason", "accepted"}, "runner state")
    gate.require(type(state["schema_version"]) is int and state["schema_version"] == 1, "invalid runner state version")
    gate.require(state["binding"] == binding, "run identity changed; use new reviewed evidence and a new run directory")
    gate.require(type(state["max_attempts"]) is int and state["max_attempts"] == max_attempts, "retry budget changed")
    gate.require(isinstance(state["status"], str) and state["status"] in STATES, "invalid runner status")
    gate.require(isinstance(state["attempts"], list) and len(state["attempts"]) <= max_attempts, "invalid attempt count")
    gate.require(isinstance(state["reason"], str), "invalid runner reason")
    for number, attempt in enumerate(state["attempts"], 1):
        gate.exact(attempt, {"number", "status", "returncode"}, "attempt")
        gate.require(type(attempt["number"]) is int and attempt["number"] == number, "nonsequential attempt journal")
        gate.require(attempt["status"] in ("RUNNING", "BLOCKED", "INTERRUPTED", "PASSED"), "invalid attempt status")
        gate.require(attempt["returncode"] is None or type(attempt["returncode"]) is int, "invalid child return code")
        if number < len(state["attempts"]):
            gate.require(attempt["status"] in ("BLOCKED", "INTERRUPTED"), "unfinished/accepted prior attempt")
    if state["status"] == "PASSED":
        gate.exact(state["accepted"], {"path", "sha256"}, "accepted report")
        gate.require(state["attempts"] and state["attempts"][-1]["status"] == "PASSED", "inconsistent accepted state")
        expected_path = f"attempt-{len(state['attempts']):04d}/report.json"
        gate.require(state["accepted"]["path"] == expected_path, "accepted report destination mismatch")
        report_path = path.parent / expected_path
        gate.require(gate.digest(report_path.read_bytes()) == gate.sha(state["accepted"]["sha256"], "accepted digest"), "accepted report changed")
    else:
        gate.require(state["accepted"] is None, "nonaccepted state contains an accepted report")
        if state["status"] == "RUNNING":
            gate.require(state["attempts"] and state["attempts"][-1]["status"] == "RUNNING", "inconsistent running state")
        elif state["attempts"]:
            gate.require(state["attempts"][-1]["status"] in ("BLOCKED", "INTERRUPTED"), "inconsistent stopped state")
    if state["status"] == "READY":
        gate.require(not state["attempts"], "ready state cannot have prior attempts")
    if state["status"] == "PASSED":
        gate.require(state["attempts"][-1]["returncode"] == 0, "accepted attempt has nonzero/missing exit")
    return state


def validate_report(path, plan, plan_hash, evidence, through, approval_hash):
    report = gate.load_json(path)
    gate.exact(report, {"schema_version", "verdict", "through", "plan_sha256", "gate_sha256",
                        "evidence_sha256", "identity", "checks", "audits", "limitation", "approvals_sha256"},
               "gate report")
    gate.require(type(report["schema_version"]) is int and report["schema_version"] == 1, "invalid report version")
    expected = {
        "verdict": "GREEN" if through == "final" else "PHASE_PASS", "through": through,
        "plan_sha256": plan_hash, "gate_sha256": plan["gate_sha256"],
        "evidence_sha256": gate.approval_digest(plan_hash, evidence, through),
        "identity": plan["identity"], "approvals_sha256": approval_hash,
    }
    gate.require(all(report[key] == value for key, value in expected.items()), "gate report identity mismatch")
    phases = list(gate.PHASES)[:list(gate.PHASES).index(through) + 1]
    gate.exact(report["checks"], phases, "report phase coverage")
    gate.exact(report["audits"], phases, "report audits")
    for phase in phases:
        gate.exact(report["checks"][phase], gate.PHASES[phase], "report check coverage")
        gate.require(report["audits"][phase] == evidence["phases"][phase]["audit"], "report audit mismatch")


def execute(args, directory, lease):
    repo = args.repo.resolve()
    evidence_root = args.evidence.resolve().parent
    gate.require(not directory.is_relative_to(repo) and not directory.is_relative_to(evidence_root),
                 "run directory must be outside source and evidence trees")
    gate.require(bool(args.approvals) == bool(args.approvals_sha256), "supply approval file and trusted hash together")
    plan = gate.pinned_json(args.plan, args.plan_sha256)
    evidence_bytes = args.evidence.read_bytes()
    evidence = gate.parse_json(evidence_bytes)
    binding = {
        "repo": str(repo), "plan_sha256": args.plan_sha256,
        "evidence_sha256": gate.digest(evidence_bytes), "through": args.through,
        "runner_sha256": gate.digest(Path(__file__).read_bytes()),
    }
    state_path = directory / "state.json"
    if state_path.exists():
        gate.require(args.resume, "existing run requires --resume")
        state = read_state(state_path, binding, args.max_attempts)
        gate.require(state["status"] != "PASSED", "phase already accepted; do not relaunch it (next phase replays its prefix)")
        if state["status"] == "RUNNING":
            state["status"] = "INTERRUPTED"
            state["attempts"][-1]["status"] = "INTERRUPTED"
            state["reason"] = "previous owner stopped before committing a result; no report is adopted"
            atomic_json(state_path, state)
        if state["status"] in ("BLOCKED", "INTERRUPTED"):
            gate.require(args.retry, "failed/interrupted run requires explicit --resume --retry")
    else:
        gate.require(not args.resume, "cannot resume without a journal")
        gate.require(set(p.name for p in directory.iterdir()) == {"lease.lock"}, "unjournaled files require operator investigation")
        state = {"schema_version": 1, "binding": binding, "max_attempts": args.max_attempts,
                 "status": "READY", "attempts": [], "reason": "", "accepted": None}
        atomic_json(state_path, state)
    gate.require(len(state["attempts"]) < args.max_attempts, "retry budget exhausted; no child launched")
    try:
        approvals = gate.load_approvals(args.approvals, args.approvals_sha256)
        gate.preflight(plan, args.plan_sha256, evidence, args.through, approvals)
        gate.verify_repo(repo, plan["identity"]["source_commit"])
    except gate.ApprovalRequired as error:
        state.update(status="WAITING_APPROVAL", reason=str(error))
        atomic_json(state_path, state)
        raise
    except (gate.GateError, OSError, subprocess.SubprocessError) as error:
        state.update(status="BLOCKED", reason=str(error))
        atomic_json(state_path, state)
        raise
    number = len(state["attempts"]) + 1
    attempt_dir = directory / f"attempt-{number:04d}"
    attempt_dir.mkdir()
    state["attempts"].append({"number": number, "status": "RUNNING", "returncode": None})
    state.update(status="RUNNING", reason="")
    atomic_json(state_path, state)
    report_path = attempt_dir / "report.json"
    command = [
        sys.executable, "-B", str(Path(gate.__file__)), "--repo", str(repo),
        "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256,
        "--evidence", str(args.evidence.resolve()), "--through", args.through,
        "--output", str(report_path), "--lease-fd", str(lease.fileno()),
    ]
    if args.approvals:
        command += ["--approvals", str(args.approvals.resolve()), "--approvals-sha256", args.approvals_sha256]
    try:
        with (attempt_dir / "stdout.log").open("xb") as stdout, (attempt_dir / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       pass_fds=(lease.fileno(),), start_new_session=True)
            try:
                rc = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        state["attempts"][-1]["returncode"] = rc
        gate.require(rc == 0, f"gate exit {rc}; inspect {attempt_dir / 'stderr.log'}; no automatic retry")
        gate.require(args.evidence.read_bytes() == evidence_bytes, "evidence changed during gate execution")
        gate.pinned_json(args.plan, args.plan_sha256)
        if args.approvals:
            gate.pinned_json(args.approvals, args.approvals_sha256)
        validate_report(report_path, plan, args.plan_sha256, evidence, args.through, args.approvals_sha256)
    except (gate.GateError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        status = "INTERRUPTED" if isinstance(error, gate.GateInterrupted) else "BLOCKED"
        state["attempts"][-1]["status"] = status
        state.update(status=status, reason=str(error), accepted=None)
        atomic_json(state_path, state)
        raise
    state["accepted"] = {"path": str(report_path.relative_to(directory)), "sha256": gate.digest(report_path.read_bytes())}
    state["attempts"][-1]["status"] = "PASSED"
    state.update(status="PASSED", reason="")
    # A signal after this commit must not rewrite accepted state as a failed attempt.
    atomic_json(state_path, state)
    return state["accepted"]


def main():
    signal.signal(signal.SIGTERM, gate.interrupt)
    signal.signal(signal.SIGINT, gate.interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--through", choices=gate.PHASES, default="final")
    parser.add_argument("--approvals", type=Path)
    parser.add_argument("--approvals-sha256")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry", action="store_true", help="explicitly authorize ONE more attempt with identical inputs")
    args = parser.parse_args()
    try:
        gate.require(1 <= args.max_attempts <= 10, "max-attempts must be in 1..10")
        gate.require(not args.retry or args.resume, "--retry requires --resume")
        directory = args.run_dir.resolve()
        gate.require(not directory.is_relative_to(args.repo.resolve())
                     and not directory.is_relative_to(args.evidence.resolve().parent),
                     "run directory must be outside source and evidence trees")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "lease.lock").open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise gate.GateError("run has a live owner/descendant; do not retry or delete its lease") from error
            # Never explicitly unlock: descendants retain this same open-file lease after SIGKILL.
            accepted = execute(args, directory, lease)
        print(json.dumps({"status": "PASSED", "run_dir": str(directory), "accepted": accepted}))
        return 0
    except (gate.GateError, OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError) as error:
        status = "WAITING_APPROVAL" if isinstance(error, gate.ApprovalRequired) else "BLOCKED"
        print(json.dumps({"status": status, "reason": str(error)}), file=sys.stderr)
        return 3 if status == "WAITING_APPROVAL" else 2


if __name__ == "__main__":
    sys.exit(main())
