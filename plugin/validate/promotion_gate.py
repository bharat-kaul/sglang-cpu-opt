#!/usr/bin/env python3
"""Promotion gate (playbook review round-2, finding #2): the single EXECUTABLE promotion decision.

A kernel set is PROMOTABLE to the next phase (end-to-end integration / deployment) ONLY if ALL hold:
  - F4 acceptance returned a RATIFIED PASS (PARTIAL / FAIL / screening-only never promote);
  - the acceptance POLICY is ratified (not PROPOSED/UNRATIFIED);
  - there are NO unresolved obligations (e.g. PENDING downstream budgets) in the kernel queue;
  - a human APPROVAL record exists AND is bound to the CURRENT evidence identity (git sha), i.e. not stale.
Written mandates are not promotion checks; THIS is. It is fail-closed: anything missing/stale/partial BLOCKS,
and attempting promotion while blocked exits nonzero. Agents consume the decision; they do not reinterpret
success text. Preserves human approval + the DSv4 campaign requirements.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
POLICY = os.path.join(RESULTS, "acceptance_policy.json")
QUEUE = os.path.join(RESULTS, "kernel_opt_queue.json")
APPROVAL = os.path.join(RESULTS, "promotion_approval.json")   # human sign-off, bound to an evidence git sha


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _dirty():
    try:
        return bool(subprocess.check_output(["git", "-C", HERE, "status", "--porcelain"], text=True).strip())
    except Exception:
        return True


def _policy_ratified(policy):
    s = str(policy.get("status", "")).upper()
    return "RATIFIED" in s and "UNRATIFIED" not in s and "PROPOSED" not in s


def _pending_obligations(queue):
    s = json.dumps(queue).upper()
    hits = []
    if "PENDING" in s:
        hits.append("queue contains PENDING obligation(s) (e.g. downstream error budgets)")
    if "SCREENING" in s and "NOT RATIFIED" in s:
        hits.append("sparse continuous SCREENING/PROPOSED, not ratified")
    return hits


def decide(f4_status, approval=None):
    """Return the promotion record. f4_status in {PASS, PARTIAL, FAIL}; approval = dict or None."""
    sha, dirty = _git_sha(), _dirty()
    policy = json.load(open(POLICY)) if os.path.exists(POLICY) else {}
    queue = json.load(open(QUEUE)) if os.path.exists(QUEUE) else {}
    if approval is None and os.path.exists(APPROVAL):
        approval = json.load(open(APPROVAL))

    reasons = []
    if f4_status != "PASS":
        reasons.append(f"F4 status is {f4_status} (only a RATIFIED PASS promotes; PARTIAL/screening never does)")
    if not _policy_ratified(policy):
        reasons.append(f"acceptance policy is UNRATIFIED ({policy.get('status', 'absent')!r})")
    obligations = _pending_obligations(queue)
    reasons += [f"unresolved obligation: {o}" for o in obligations]
    if dirty:
        reasons.append("working tree is DIRTY — evidence identity not bound to a committed sha")
    if not approval:
        reasons.append("no human approval record (results/promotion_approval.json)")
    elif approval.get("approves_sha") != sha:
        reasons.append(f"approval is STALE: approves_sha {approval.get('approves_sha')!r} != current {sha!r}")

    promote = len(reasons) == 0
    return {"decision": "PROMOTE" if promote else "BLOCKED", "promote": promote,
            "evidence_identity": {"git_sha": sha, "dirty": dirty, "reference_revision": policy.get("reference_revision")},
            "f4_status": f4_status, "policy_ratified": _policy_ratified(policy),
            "unresolved_obligations": obligations, "approval_present": bool(approval), "reasons": reasons}


def _f4_status():
    """Run F4 and map its exit (0=PARTIAL, 2=FAIL) to a status. F4 never emits a ratified PASS by design."""
    import f4_acceptance
    rc = f4_acceptance.run()
    return "FAIL" if rc == 2 else "PARTIAL"


def selftest():
    ok = True
    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")
    # PARTIAL F4 must BLOCK regardless of approval
    chk(not decide("PARTIAL")["promote"], "BLOCKS on PARTIAL F4")
    chk(not decide("FAIL")["promote"], "BLOCKS on FAIL F4")
    # even a hypothetical PASS blocks while policy is UNRATIFIED / obligations pending / no approval
    d = decide("PASS")
    chk(not d["promote"], "BLOCKS a PASS while policy UNRATIFIED / obligations pending / no approval")
    chk(any("UNRATIFIED" in r for r in d["reasons"]), "cites the unratified policy as a reason")
    # a stale approval (wrong sha) must not promote
    chk(not decide("PASS", approval={"approves_sha": "deadbeef", "by": "x"})["promote"], "BLOCKS on a STALE approval")
    print(f"  PROMOTION-GATE SELFTEST {'OK' if ok else 'FAILED'}")
    return 0 if ok else 2


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    rec = decide(_f4_status())
    print(json.dumps(rec, indent=2))
    print(f"\nPROMOTION: {rec['decision']}")
    sys.exit(0 if rec["promote"] else 3)       # fail-closed: nonzero when BLOCKED


if __name__ == "__main__":
    main()
