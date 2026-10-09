#!/usr/bin/env python3
"""review_loop — automate a playbook gate as an EXECUTOR <-> REVIEWER loop.

One agent (executor, e.g. Claude Opus 4.8) advances a playbook gate and emits artifacts + a response.
A DIFFERENT agent (reviewer, e.g. GPT Astra 6) audits the artifacts and emits a machine-readable VERDICT.
The orchestrator feeds a FAIL review back to the executor and iterates until PASS, a SURFACE-to-user
signal, or max_iterations. Agents are AGENT-AGNOSTIC: each role is a pluggable adapter (a shell command
template or a mock replay), and the user picks which model runs each role in the config.

This formalizes the manual roofline review loop (review1..7 -> response1..6 -> PASS) into a gated,
auditable automation. Zero third-party deps (stdlib only).

Verdict contract (the reviewer MUST emit exactly one of these blocks in its output or review file):
    <<<REVIEW-VERDICT
    {"status": "PASS|FAIL|SURFACE", "findings": ["..."], "surface": false, "note": "..."}
    REVIEW-VERDICT>>>
  PASS    -> gate accepted, loop ends (success).
  FAIL    -> findings fed back to the executor; loop continues.
  SURFACE -> (or "surface": true) a user decision/closure is required; loop stops and escalates.

Usage:
  review_loop.py run    --config config.json         # drive real agents per the config adapters
  review_loop.py demo                                # replay the real 7-round history with mock agents
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

_VERDICT_RE = re.compile(r"<<<REVIEW-VERDICT\s*(\{.*?\})\s*REVIEW-VERDICT>>>", re.DOTALL)
_STATUSES = ("PASS", "FAIL", "SURFACE")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _subst(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def run_agent(role_cfg, role, rnd, prompt, state_dir):
    """Invoke one agent role. Returns (output_text). Adapters: 'shell' (command template) or 'mock' (replay)."""
    os.makedirs(state_dir, exist_ok=True)
    prompt_file = os.path.join(state_dir, f"{role}_round{rnd}.prompt.txt")
    out_file = os.path.join(state_dir, f"{role}_round{rnd}.out.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)
    adapter = role_cfg.get("adapter", "shell")
    model = role_cfg.get("model", "")
    if adapter == "mock":
        # replay: read a prepared output for this (role, round)
        replay = _subst(role_cfg["replay"], role=role, round=rnd, model=model)
        replay = replay if os.path.isabs(replay) else os.path.join(_HERE, replay)
        out = open(replay).read() if os.path.exists(replay) else '<<<REVIEW-VERDICT {"status":"SURFACE","note":"no replay"} REVIEW-VERDICT>>>'
    elif adapter == "shell":
        cmd = _subst(role_cfg["cmd"], model=model, role=role, round=rnd,
                     prompt_file=prompt_file, workdir=role_cfg.get("workdir", "."))
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=role_cfg.get("timeout_s", 3600))
        out = r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else "")
    else:
        raise ValueError(f"unknown adapter {adapter!r}")
    with open(out_file, "w") as f:
        f.write(out)
    return out


def parse_verdict(text, review_file=None):
    """Extract the machine-readable verdict from the reviewer output or a designated review file."""
    for blob in (text, (open(review_file).read() if review_file and os.path.exists(review_file) else "")):
        m = _VERDICT_RE.search(blob or "")
        if m:
            try:
                v = json.loads(m.group(1))
            except json.JSONDecodeError as e:
                return {"status": "SURFACE", "note": f"malformed verdict JSON: {e}", "surface": True}
            st = str(v.get("status", "")).upper()
            if st not in _STATUSES:
                return {"status": "SURFACE", "note": f"invalid status {st!r}", "surface": True}
            v["status"] = st
            return v
    # fail-closed: no parseable verdict => surface to the user (never silently continue/pass)
    return {"status": "SURFACE", "note": "no machine-readable REVIEW-VERDICT block found", "surface": True}


def _commit(workdir, msg):
    try:
        subprocess.run(["git", "-C", workdir, "add", "-A"], check=False)
        subprocess.run(["git", "-C", workdir, "commit", "-q", "-m", msg], check=False)
    except Exception:  # noqa: BLE001
        pass


def loop(cfg):
    task = cfg["task"]
    name = task["name"]
    workdir = task.get("workdir", ".")
    state_dir = _subst(cfg.get("state_dir", os.path.join(_HERE, "runs", "{name}")), name=name)
    loopc = cfg.get("loop", {})
    max_it = int(loopc.get("max_iterations", 8))
    rev_out_tmpl = cfg["reviewer"].get("review_output", "")
    ledger = []
    last_review = ""
    print(f"[review_loop] gate={name}  executor={cfg['executor'].get('model')}  reviewer={cfg['reviewer'].get('model')}  max={max_it}")
    for rnd in range(1, max_it + 1):
        # 1) EXECUTE (address prior review if any)
        ex_prompt = _subst(_read(task["executor_prompt"]), name=name, round=rnd, workdir=workdir,
                           review_feedback=last_review or "(none — first round)")
        run_agent(cfg["executor"], "executor", rnd, ex_prompt, state_dir)
        if loopc.get("commit_between_rounds"):
            _commit(workdir, f"review_loop[{name}] round {rnd}: executor pass")
        # 2) REVIEW
        rv_prompt = _subst(_read(task["reviewer_prompt"]), name=name, round=rnd, workdir=workdir)
        rv_out = run_agent(cfg["reviewer"], "reviewer", rnd, rv_prompt, state_dir)
        rev_file = _subst(rev_out_tmpl, name=name, round=rnd) if rev_out_tmpl else None
        if rev_file and not os.path.isabs(rev_file):
            rev_file = os.path.join(workdir, rev_file)
        # 3) PARSE VERDICT (fail-closed)
        v = parse_verdict(rv_out, rev_file)
        ledger.append({"round": rnd, "status": v["status"], "findings": v.get("findings", []), "note": v.get("note", "")})
        print(f"[review_loop] round {rnd}: {v['status']}  findings={len(v.get('findings', []))}  {v.get('note','')[:80]}")
        # 4) DECIDE
        if v["status"] == "PASS":
            return _finish(name, "PASS", ledger, state_dir)
        if v["status"] == "SURFACE" or v.get("surface"):
            return _finish(name, "SURFACE", ledger, state_dir, reason=v.get("note", "reviewer requested user decision"))
        last_review = (open(rev_file).read() if rev_file and os.path.exists(rev_file) else rv_out)  # feed back
    return _finish(name, "SURFACE", ledger, state_dir, reason=f"max_iterations ({max_it}) reached without PASS")


def _read(p):
    p = p if os.path.isabs(p) else os.path.join(_HERE, p)
    return open(p).read()


def _finish(name, outcome, ledger, state_dir, reason=""):
    os.makedirs(state_dir, exist_ok=True)
    summary = {"gate": name, "outcome": outcome, "rounds": len(ledger), "reason": reason,
               "ledger": ledger, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(state_dir, "ledger.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[review_loop] OUTCOME={outcome} after {len(ledger)} round(s). {reason}")
    if outcome == "SURFACE":
        print("[review_loop] -> SURFACED to user for decision/closure.")
    return summary


def _demo():
    """Replay the REAL roofline review history (6 FAIL rounds R/F/G/H -> PASS) with mock agents."""
    cfg = json.load(open(os.path.join(_HERE, "config.demo.json")))
    return loop(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="drive real agents per a config")
    r.add_argument("--config", required=True)
    sub.add_parser("demo", help="replay the real 7-round history with mock agents")
    a = ap.parse_args()
    s = _demo() if a.cmd == "demo" else loop(json.load(open(a.config)))
    sys.exit(0 if s["outcome"] == "PASS" else 2)


if __name__ == "__main__":
    main()
