#!/usr/bin/env python3
"""Replay pinned phase checks; issue a certificate only after fail-closed acceptance.

The operator supplies independently trusted plan/approval hashes. Model-specific,
reviewed adapters validate raw artifacts; this gate enforces their identities,
coverage, numeric criteria, negative controls, audits and phase prerequisites.
See the enablement-certificate skill for the versioned JSON contracts.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile


PHASES = {
    "discovery": ("scope", "source_contract", "infrastructure", "calibration", "capacity"),
    "bringup": ("reference", "correctness"),
    "standalone": (
        "native", "correctness", "performance", "optimization", "fusion",
        "precision", "determinism",
    ),
    "integration": ("dispatch", "serving", "correctness", "performance"),
    "final": ("accuracy", "performance", "peer", "determinism", "publication", "diff"),
}
METRICS = {
    "scope": {"missing": ("max", 0)},
    "infrastructure": {"missing": ("max", 0), "stale": ("max", 0), "overcommitted": ("max", 0)},
    "source_contract": {"mismatches": ("max", 0)},
    "calibration": {"invalid_targets": ("max", 0)},
    "capacity": {"headroom_bytes": ("min", 1)},
    "reference": {"mismatches": ("max", 0)},
    "correctness": {"violations": ("max", 0)},
    "native": {"non_native": ("max", 0)},
    "optimization": {"unresolved": ("max", 0), "untested": ("max", 0)},
    "fusion": {"unresolved": ("max", 0)},
    "precision": {"violations": ("max", 0)},
    "determinism": {"violations": ("max", 0), "repeats": ("min", 3)},
    "dispatch": {"missing": ("max", 0), "non_native": ("max", 0)},
    "serving": {"violations": ("max", 0)},
    "accuracy": {"score_drop_pp": ("max", 1)},
    "publication": {"broken": ("max", 0)},
    "diff": {"core_changes": ("max", 0), "lint_errors": ("max", 0)},
    "performance": {"eff_abs": ("min", 0.70), "speedup": ("min", 1)},
    "peer": {"eff_rel": ("min", 0.90)},
}
IDENTITY_KEYS = {
    "model", "source_commit", "config_sha256", "checkpoint_revision",
    "reference_revision", "hardware_sha256", "runtime_sha256", "dtype", "node",
}
CASE_KEYS = {"op", "batch", "shape", "dispatch", "distribution"}
REVIEW_PHASES = {"standalone", "final"}


class GateError(ValueError):
    """An explicit rejection, never a success-shaped fallback."""


class ApprovalRequired(GateError):
    """Stop without running checks until a valid external approval is available."""


class GateInterrupted(GateError):
    """An interrupted replay is never a successful negative control."""


def interrupt(signum, frame):
    raise GateInterrupted(f"interrupted by signal {signum}")


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def require(condition, message):
    if not condition:
        raise GateError(message)


def exact(value, keys, where):
    require(type(value) is dict and set(value) == set(keys), f"{where}: unexpected/missing fields")
    return value


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()), f"{where}: expected nonempty string")
    return value


def number(value, where, positive=False):
    require(
        type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max
        and math.isfinite(value),
        f"{where}: expected finite number",
    )
    require(not positive or value > 0, f"{where}: expected positive number")
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def sha(value, where):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"{where}: invalid SHA256")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise GateError(f"non-finite JSON constant: {value}")


def parse_json(data):
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)


def load_json(path):
    return parse_json(Path(path).read_bytes())


def pinned_json(path, expected):
    data = Path(path).read_bytes()
    require(digest(data) == sha(expected, "expected digest"), f"{path}: digest mismatch")
    return parse_json(data)


def identity(value):
    exact(value, IDENTITY_KEYS, "identity")
    for key, item in value.items():
        text(item, key)
        if key.endswith("_sha256"):
            sha(item, key)
    require(re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]), "invalid source_commit")


def case_key(case):
    exact(case, CASE_KEYS, "case")
    require(type(case["batch"]) is int and 1 <= case["batch"] <= 64, "batch must be in 1..64")
    for key in CASE_KEYS - {"batch"}:
        text(case[key], f"case.{key}")
    return json_digest(case)


def cases(value):
    require(isinstance(value, list) and value, "cases must be nonempty")
    keys = [case_key(case) for case in value]
    require(len(set(keys)) == len(keys), "duplicate joint coordinate")
    return set(keys)


def validate_plan(plan):
    exact(plan, {"schema_version", "gate_sha256", "identity", "batches", "ops", "kernel_cases", "checks"}, "plan")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1, "unsupported plan version")
    require(plan["gate_sha256"] == digest(Path(__file__).read_bytes()), "gate implementation digest mismatch")
    identity(plan["identity"])
    batches = plan["batches"]
    require(isinstance(batches, list) and batches, "missing batch domain")
    require(all(type(b) is int and 1 <= b <= 64 for b in batches), "batch domain must be in 1..64")
    require(len(set(batches)) == len(batches) and {1, 64} <= set(batches), "batch domain must include 1 and 64, without duplicates")
    ops = plan["ops"]
    require(isinstance(ops, list) and ops, "empty operator inventory")
    for op in ops:
        text(op, "op")
    require(len(set(ops)) == len(ops), "duplicate operator")
    kernel_keys = cases(plan["kernel_cases"])
    require(
        {(c["op"], c["batch"]) for c in plan["kernel_cases"]} == {(op, b) for op in ops for b in batches},
        "kernel domain must cover every op x batch",
    )
    exact(plan["checks"], PHASES, "plan phases")
    for phase, required in PHASES.items():
        exact(plan["checks"][phase], required, f"{phase} checks")
        for name, spec in plan["checks"][phase].items():
            fields = {"adapter", "inputs", "cases", "limits", "timeout_seconds"}
            if name in ("performance", "peer"):
                fields |= {"baseline_identity"}
            if name == "performance":
                fields |= {"target_ms"}
            exact(spec, fields, f"{phase}.{name}")
            exact(spec["adapter"], {"path", "sha256"}, "adapter")
            text(spec["adapter"]["path"], "adapter path")
            sha(spec["adapter"]["sha256"], "adapter digest")
            keys = cases(spec["cases"])
            require(all(c["batch"] in batches for c in spec["cases"]), "case outside approved batch domain")
            if phase == "standalone":
                require(keys == kernel_keys, f"{name}: incomplete standalone joint coverage")
            if phase in ("integration", "final") and name in ("performance", "peer", "dispatch", "serving"):
                require({c["batch"] for c in spec["cases"]} == set(batches), f"{name}: incomplete model batch coverage")
            inputs = spec["inputs"]
            require(isinstance(inputs, list) and inputs, "empty input contract")
            for item in inputs:
                text(item, "input role")
            require(len(set(inputs)) == len(inputs), "duplicate input role")
            needed = {"candidate", "baseline", "calibration"} if name == "performance" else (
                {"candidate", "baseline"} if name == "peer" else {"candidate", "reference"} if name == "correctness" else set()
            )
            if name == "infrastructure":
                needed = {"inventory", "execution_plan"}
            require(needed <= set(inputs), f"{name}: missing raw input roles {needed - set(inputs)}")
            timeout = number(spec["timeout_seconds"], "timeout", positive=True)
            require(timeout <= 600, "gate replay timeout must be <=600 seconds; do model work outside gate")
            exact(spec["limits"], METRICS[name], "metric limits")
            for metric, (direction, boundary) in METRICS[name].items():
                limit = number(spec["limits"][metric], metric)
                require(limit >= 0, "negative acceptance limit")
                require(limit >= boundary if direction == "min" else limit <= boundary, f"{metric}: weakened mandatory limit")
            if name in ("performance", "peer"):
                identity(spec["baseline_identity"])
                for key in ("node", "hardware_sha256", "runtime_sha256", "dtype"):
                    require(spec["baseline_identity"][key] == plan["identity"][key], f"baseline {key} mismatch")
            if name == "performance":
                require(isinstance(spec["target_ms"], list) and len(spec["target_ms"]) == len(spec["cases"]), "target/case length mismatch")
                for target in spec["target_ms"]:
                    number(target, "target_ms", positive=True)


def resolve_artifact(root, ref):
    exact(ref, {"path", "sha256"}, "artifact")
    rel = Path(text(ref["path"], "artifact path"))
    require(not rel.is_absolute(), "artifact path must be relative")
    path = (root / rel).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"missing/outside-root artifact: {rel}")
    require(digest(path.read_bytes()) == sha(ref["sha256"], "artifact digest"), f"artifact digest mismatch: {rel}")
    return path


def check_result(result, plan, phase, name, spec):
    exact(result, {"identity", "phase", "check", "observations"}, "adapter result")
    require(result["identity"] == plan["identity"], "adapter identity mismatch")
    require(result["phase"] == phase and result["check"] == name, "adapter phase/check mismatch")
    observations = result["observations"]
    require(isinstance(observations, list), "observations must be a list")
    expected = {case_key(c): i for i, c in enumerate(spec["cases"])}
    seen = set()
    computed = []
    for observation in observations:
        timing = name in ("performance", "peer")
        exact(observation, {"case", "samples" if timing else "metrics"}, "observation")
        key = case_key(observation["case"])
        require(key in expected and key not in seen, "unexpected/duplicate observation coordinate")
        seen.add(key)
        if timing:
            samples = observation["samples"]
            require(isinstance(samples, list) and len(samples) >= 3, "need >=3 process samples on both sides")
            times = {"candidate": [], "baseline": []}
            for side in times:
                ids = set()
                expected_identity = plan["identity"] if side == "candidate" else spec["baseline_identity"]
                for sample in samples:
                    exact(sample, {"candidate", "baseline"}, "sample")
                    item = exact(sample[side], {"identity", "process_id", "ms"}, side)
                    require(item["identity"] == expected_identity, f"{side} sample identity mismatch")
                    process_id = text(item["process_id"], "process_id")
                    require(process_id not in ids, f"duplicate {side} process identity")
                    ids.add(process_id)
                    times[side].append(number(item["ms"], f"{side}.ms", positive=True))
            candidate = statistics.median(times["candidate"])
            baseline = statistics.median(times["baseline"])
            if name == "performance":
                metrics = {"eff_abs": spec["target_ms"][expected[key]] / candidate, "speedup": baseline / candidate}
                require(metrics["eff_abs"] <= 1, "measured faster than calibrated floor: target/model review required")
            else:
                metrics = {"eff_rel": baseline / candidate}
            computed.append({"case": observation["case"], "metrics": metrics, "samples": samples})
        else:
            metrics = observation["metrics"]
            exact(metrics, METRICS[name], "observed metrics")
            computed.append(observation)
        for metric, (direction, _) in METRICS[name].items():
            value = number(metrics[metric], metric)
            require(value >= 0 or metric == "score_drop_pp", f"{metric}: negative count")
            if metric not in ("headroom_bytes", "score_drop_pp", "eff_abs", "speedup", "eff_rel"):
                require(type(value) is int, f"{metric}: count must be integer")
            limit = spec["limits"][metric]
            require(value >= limit if direction == "min" else value <= limit, f"{phase}.{name}.{metric}: {value} fails {direction} {limit}")
    require(seen == set(expected), f"{phase}.{name}: missing observation coordinates")
    return computed


def replay(repo, root, plan, phase, name, spec, refs, lease_fd=None):
    exact(refs, spec["inputs"], f"{phase}.{name} input roles")
    inputs = {role: str(resolve_artifact(root, ref)) for role, ref in refs.items()}
    require(len(set(inputs.values())) == len(inputs), "raw input roles alias the same artifact")
    adapter = resolve_artifact(repo, spec["adapter"])
    require(adapter.suffix == ".py", "replay adapter must be a reviewed Python file")
    request = {"identity": plan["identity"], "phase": phase, "check": name, "cases": spec["cases"], "inputs": inputs}
    with subprocess.Popen(
        [sys.executable, "-B", str(adapter)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd=repo, text=True, start_new_session=True,
        pass_fds=() if lease_fd is None else (lease_fd,),
    ) as process:
        try:
            stdout, stderr = process.communicate(json.dumps(request, allow_nan=False), timeout=spec["timeout_seconds"])
        except subprocess.TimeoutExpired:
            raise GateError(f"{phase}.{name}: adapter timeout") from None
        finally:
            stop_process_group(process)
        require(process.returncode == 0, f"{phase}.{name}: adapter exit {process.returncode}: {stderr[-2000:]}")
    return check_result(parse_json(stdout), plan, phase, name, spec)


def checks_digest(plan_hash, evidence, through):
    names = list(PHASES)[:list(PHASES).index(through) + 1]
    return json_digest({
        "plan_sha256": plan_hash, "identity": evidence["identity"],
        "checks": {phase: evidence["phases"][phase]["checks"] for phase in names},
    })


def approval_digest(plan_hash, evidence, through):
    names = list(PHASES)[:list(PHASES).index(through) + 1]
    return json_digest({
        "plan_sha256": plan_hash, "identity": evidence["identity"],
        "phases": {phase: evidence["phases"][phase] for phase in names},
    })


def verify_repo(repo, expected):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    require(result.stdout.strip() == expected, "source commit mismatch")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True)
    require(not dirty.stdout.strip(), "source worktree is dirty")


def preflight(plan, plan_hash, evidence, through, approvals):
    validate_plan(plan)
    exact(evidence, {"schema_version", "identity", "phases"}, "evidence")
    require(type(evidence["schema_version"]) is int and evidence["schema_version"] == 1, "unsupported evidence version")
    require(evidence["identity"] == plan["identity"], "evidence identity mismatch")
    phases = list(PHASES)[:list(PHASES).index(through) + 1]
    exact(evidence["phases"], phases, "evidence phase prefix")
    for phase in phases:
        record = exact(evidence["phases"][phase], {"checks", "audit"}, f"{phase} record")
        exact(record["checks"], PHASES[phase], f"{phase} evidence checks")
        audit = exact(record["audit"], {"checks_sha256", "reviewer", "goal_alignment", "findings"}, "audit")
        require(audit["checks_sha256"] == checks_digest(plan_hash, evidence, phase), f"{phase}: stale self-audit")
    try:
        exact(approvals, {"schema_version", "approvals"}, "approval envelope")
        require(type(approvals["schema_version"]) is int and approvals["schema_version"] == 1, "unsupported approval version")
        exact(approvals["approvals"], set(phases) & REVIEW_PHASES, "required human approvals")
        for phase in phases:
            if phase in REVIEW_PHASES:
                approval = exact(approvals["approvals"][phase], {"decision", "reviewer", "evidence_sha256"}, "human approval")
                require(approval["decision"] == "APPROVED", f"{phase}: human approval required")
                text(approval["reviewer"], "human reviewer")
                require(approval["evidence_sha256"] == approval_digest(plan_hash, evidence, phase), f"{phase}: stale human approval")
    except GateError as error:
        raise ApprovalRequired(str(error)) from error
    return phases


def load_approvals(path, expected):
    if path is None:
        return {"schema_version": 1, "approvals": {}}
    try:
        return pinned_json(path, expected)
    except (GateError, FileNotFoundError, json.JSONDecodeError) as error:
        raise ApprovalRequired(str(error)) from error


def evaluate(repo, root, plan, plan_hash, evidence, through, approvals, lease_fd=None):
    phases = preflight(plan, plan_hash, evidence, through, approvals)
    verify_repo(repo, plan["identity"]["source_commit"])
    reports = {}
    for phase in phases:
        record = evidence["phases"][phase]
        audit = record["audit"]
        text(audit["reviewer"], "audit reviewer")
        text(audit["goal_alignment"], "Xeon batch1-64 goal alignment assessment")
        require(isinstance(audit["findings"], list), "audit findings must be a list")
        for finding in audit["findings"]:
            exact(finding, {"issue", "status", "resolution", "evidence"}, "audit finding")
            require(finding["status"] == "RESOLVED", "unresolved audit finding")
            text(finding["issue"], "finding issue")
            text(finding["resolution"], "finding resolution")
            resolve_artifact(root, finding["evidence"])
        reports[phase] = {}
        for name, spec in plan["checks"][phase].items():
            entry = exact(record["checks"][name], {"inputs", "negative_controls"}, "check evidence")
            controls = entry["negative_controls"]
            require(isinstance(controls, list) and controls, f"{phase}.{name}: missing adversarial controls")
            reports[phase][name] = replay(repo, root, plan, phase, name, spec, entry["inputs"], lease_fd)
            control_names = set()
            for control in controls:
                exact(control, {"name", "inputs", "expected_error"}, "negative control")
                label = text(control["name"], "negative control name")
                require(label not in control_names, "duplicate negative control")
                control_names.add(label)
                expected_error = text(control["expected_error"], "expected rejection")
                # Validate the fixture itself first; a missing file/hash is not a valid test.
                exact(control["inputs"], spec["inputs"], "negative input roles")
                for ref in control["inputs"].values():
                    resolve_artifact(root, ref)
                require(control["inputs"] != entry["inputs"], "negative control reuses positive inputs")
                try:
                    replay(repo, root, plan, phase, name, spec, control["inputs"], lease_fd)
                except GateInterrupted:
                    raise
                except GateError as error:
                    require(
                        "adapter exit" not in str(error) and "adapter timeout" not in str(error),
                        "crashed/timed-out adapter is not a discriminating negative control",
                    )
                    require(expected_error in str(error), f"negative control rejected for wrong reason: {error}")
                else:
                    raise GateError(f"{phase}.{name}: negative control was accepted")
            # Recheck input identities after adapter execution and bind the audit to these bytes.
            for ref in entry["inputs"].values():
                resolve_artifact(root, ref)
            reports[phase][name] = {"observations": reports[phase][name], "negative_controls_rejected": sorted(control_names)}
    # Recheck the whole evaluated prefix, not just the final adapter's files.
    verify_repo(repo, plan["identity"]["source_commit"])
    for phase in phases:
        for finding in evidence["phases"][phase]["audit"]["findings"]:
            resolve_artifact(root, finding["evidence"])
        for name, entry in evidence["phases"][phase]["checks"].items():
            resolve_artifact(repo, plan["checks"][phase][name]["adapter"])
            for refs in [entry["inputs"]] + [c["inputs"] for c in entry["negative_controls"]]:
                for ref in refs.values():
                    resolve_artifact(root, ref)
    return {
        "schema_version": 1, "verdict": "GREEN" if through == "final" else "PHASE_PASS",
        "through": through, "plan_sha256": plan_hash, "gate_sha256": plan["gate_sha256"],
        "evidence_sha256": approval_digest(plan_hash, evidence, through),
        "identity": plan["identity"], "checks": reports,
        "audits": {phase: evidence["phases"][phase]["audit"] for phase in phases},
        "limitation": "Acceptance depends on the independently approved plan and source-faithful replay adapters; this is not proof of universal optimality.",
    }


def main():
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True, help="independently approved plan digest")
    parser.add_argument("--evidence", type=Path, required=True, help="JSON; artifact paths relative to this file")
    parser.add_argument("--through", choices=PHASES, default="final")
    parser.add_argument("--approvals", type=Path)
    parser.add_argument("--approvals-sha256", help="digest from the external human approval channel")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        require(bool(args.approvals) == bool(args.approvals_sha256), "approval file and trusted digest must be supplied together")
        repo = args.repo.resolve()
        root = args.evidence.resolve().parent
        output = args.output.resolve()
        require(not output.is_relative_to(repo) and not output.is_relative_to(root), "output must be outside source and evidence trees")
        require(output not in {args.plan.resolve(), args.approvals.resolve() if args.approvals else None}, "output aliases a contract")
        plan = pinned_json(args.plan, args.plan_sha256)
        evidence_bytes = args.evidence.read_bytes()
        evidence = parse_json(evidence_bytes)
        if args.lease_fd is not None:
            require(args.lease_fd >= 3, "invalid inherited lease descriptor")
            os.fstat(args.lease_fd)
        approvals = load_approvals(args.approvals, args.approvals_sha256)
        report = evaluate(repo, root, plan, args.plan_sha256, evidence, args.through, approvals, args.lease_fd)
        report["approvals_sha256"] = args.approvals_sha256
        require(args.evidence.read_bytes() == evidence_bytes, "evidence changed during evaluation")
        pinned_json(args.plan, args.plan_sha256)
        if args.approvals:
            pinned_json(args.approvals, args.approvals_sha256)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(json.dumps({"verdict": report["verdict"], "through": args.through, "output": str(output)}))
        return 0
    except (GateError, OSError, json.JSONDecodeError, UnicodeError, subprocess.SubprocessError) as error:
        kind = "approval" if isinstance(error, ApprovalRequired) else "interrupted" if isinstance(error, GateInterrupted) else "validation"
        print(json.dumps({"verdict": "BLOCKED", "kind": kind, "reason": str(error)}, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
