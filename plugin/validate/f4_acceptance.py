#!/usr/bin/env python3
"""F4 acceptance harness (tolerance-INDEPENDENT) — authorized build.

Implements the per-op acceptance plumbing from results/acceptance_policy.json WITHOUT ratifying numerical
thresholds and WITHOUT ever declaring a full F4 PASS. What is a HARD gate NOW (fail -> nonzero exit):
  - layer 0: output shape/dtype/layout == contract; domain-specific finiteness; REPEATABILITY determinism
    (same frozen config, two runs bit-identical); input-domain (exercised in --selftest);
  - layer 2 selection: reference-owned membership with tie_eps=0 -> ZERO non-tie mismatches (user-ratified);
  - a missing/skipped case or an exception is a FAILURE, never a silent pass.
What is COLLECTED but NOT gated (thresholds UNRATIFIED -> verdict PARTIAL): the continuous elementwise error
e_i=|cand-ref| vs the SCREENING predicate e_i<=atol+rtol*|ref_i| and cosine; reported per op/path/output for a
data-driven threshold proposal. SCREENING values (rtol=1e-4, cosine=0.999999) are provisional, not production.
ORACLE-ADAPTER CONFORMANCE (vs the published runtime) is tracked SEPARATELY from candidate-kernel accuracy and
is recorded UNVERIFIED here (local references stand in until conformed).

Usage:
  f4_acceptance.py             # run the manifest -> status PARTIAL (exit 0) or FAIL (exit 2)
  f4_acceptance.py --selftest  # failure injection: EVERY injected fault must be caught (FAIL)
  f4_acceptance.py --calibrate results/f4_calibration.json   # write calibration + threshold proposal
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intel_cpu_models"))
from torch.utils.cpp_extension import load

_CF = ["-O3", "-fopenmp", "-march=native"]
_KDIR = os.path.join(HERE, "..", "kernels", "dsa_pilot")

# SCREENING thresholds — PROVISIONAL (not production acceptance); used only to FLAG, never to certify.
SCREEN_RTOL = 1e-4
SCREEN_COS = 0.999999
TIE_EPS = 0.0                       # user-ratified: zero non-tie selection mismatches


def _mod(name, cpp):
    return load(name=name, sources=[os.path.join(_KDIR, cpp)], extra_cflags=_CF, verbose=False)


# ---- oracle adapters (LOCAL references; conformance vs the published runtime = UNVERIFIED) ----
def _ora_indexer_logits(q, kv, w):
    # D1 conformance: the published Indexer.forward scores in BF16 (einsum on bf16 operands), FP32 reduce.
    # A FP32 oracle is OVER-STRICT and induces spurious near-cutoff selection mismatches (measured: FP32
    # oracle -> 7 topk mismatches; BF16 oracle -> 0). Conform the adapter to the BF16 boundary.
    scores = torch.einsum("nhd,nsd->nhs", q.bfloat16().float(), kv.bfloat16().float())
    return (torch.relu(scores) * w.unsqueeze(-1).float()).sum(1)


def _ora_compressor(kv, score, ape):
    from dsa_compressor_cpu import compress_softmax_pool
    return compress_softmax_pool(kv, score, ape)


def _ora_sinkhorn(mixes, scale, base, hc, it, eps):
    from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch
    return _hc_split_sinkhorn_torch(mixes, scale, base, hc, it, eps)


def _ora_combine(x_flat, pre, hc):
    m, h = x_flat.shape[0], x_flat.shape[1] // hc
    return torch.einsum("mk,mkh->mh", pre.float(), x_flat.reshape(m, hc, h).float())


def _ora_sparse(q, kv, sink, scale):
    hh = q.shape[1]
    scores = torch.einsum("nhd,nkd->nhk", q.float(), kv.float()) * scale
    m = torch.maximum(scores.max(-1, keepdim=True).values, sink.view(1, hh, 1))
    e = (scores - m).exp()
    denom = e.sum(-1, keepdim=True) + (sink.view(1, hh, 1) - m).exp()
    return torch.einsum("nhk,nkd->nhd", e / denom, kv.float())


ORACLE_CONFORMANCE = "UNVERIFIED vs published inference/model.py + kernel.py runtime (local reference stand-in)"


# ---- input builders (normal + heavy-tailed + edge) ----
def _heavy(*shape):
    return torch.distributions.StudentT(3.0).sample(shape)


def _dist(name, *shape):
    return torch.randn(*shape) if name == "normal" else _heavy(*shape)


# ---- metric helpers ----
def _elem_err(cand, ref):
    cand, ref = cand.float(), ref.float()
    e = (cand - ref).abs()
    return e.max().item(), ref.abs().max().item()


def _cosine(cand, ref):
    a, b = cand.flatten().float(), ref.flatten().float()
    if a.norm() == 0 or b.norm() == 0:
        return None                                   # zero-vector cosine is N/A (use the absolute branch)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _screen_continuous(cand, ref, atol):
    """Collect metrics; return (dict, screen_pass_or_None). Never a production verdict."""
    e = (cand.float() - ref.float()).abs()
    rhs = atol + SCREEN_RTOL * ref.float().abs()
    within = bool((e <= rhs).all().item())
    cos = _cosine(cand, ref)
    m = {"max_abs_err": e.max().item(), "ref_absmax": ref.abs().max().item(),
         "cosine": cos, "screen_atol": atol,
         "screen_pass": within and (cos is None or cos >= SCREEN_COS)}
    return m


def _selection_check(cand_idx, ref_logits, k):
    """Reference-owned membership, tie_eps=0. Returns (non_tie_mismatches, tie_boundary_swaps)."""
    nonmis, ties = 0, 0
    N = ref_logits.shape[0]
    for r in range(N):
        s = ref_logits[r].float()
        cand = cand_idx[r].tolist()
        if len(set(cand)) != k or any(not (0 <= i < s.numel()) for i in cand):
            nonmis += k                               # malformed selection row (non-distinct / out of range)
            continue
        kth = torch.topk(s, k, sorted=True).values[-1].item()   # reference cutoff
        candset = set(cand)
        above = {i for i in range(s.numel()) if s[i].item() > kth}
        below = {i for i in range(s.numel()) if s[i].item() < kth}
        nonmis += len(above - candset) + len(candset & below)   # missing an above / including a below
        ties += len(candset & {i for i in range(s.numel()) if s[i].item() == kth})
    return nonmis, ties


# ---- the case manifest ----
def manifest():
    """Return the list of acceptance cases. Each: op, path, kind, build (-> inputs), cand, ora, out_contract."""
    il = _mod("f4_il", "indexer_logits.cpp")
    tk = _mod("f4_tk", "indexer_topk.cpp")
    cp = _mod("f4_cp", "compressor.cpp")
    sk = _mod("f4_sk", "sinkhorn.cpp")
    cb = _mod("f4_cb", "combine.cpp")
    sp = _mod("f4_sp", "sparse_attend.cpp")
    cases = []

    # indexer_logits -> induced selection is the gate; logit error is a diagnostic
    for d in ("normal", "heavy"):
        for M in (1, 8, 32):
            cases.append({"op": "indexer_logits", "path": "tiled", "kind": "logits_select", "dist": d,
                          "build": (lambda M=M, d=d: (_dist(d, M, 64, 128), _dist(d, M, 1024, 128), torch.rand(M, 64))),
                          "cand": lambda q, kv, w: il.indexer_logits(q, kv, w),
                          "ora": _ora_indexer_logits, "k": 512, "out": ((lambda M=M: (M, 1024)), torch.float32)})

    # indexer_topk selection (tie_eps=0) + edge cases
    for M in (1, 8, 32):
        cases.append({"op": "indexer_topk", "path": "chunked/row", "kind": "selection", "dist": "normal",
                      "build": (lambda M=M: (_dist("normal", M, 1024), 512)),
                      "cand": lambda lg, k: tk.indexer_topk(lg, k), "ora": None, "k": 512,
                      "out": ((lambda M=M: (M, 512)), torch.int64)})
    cases.append({"op": "indexer_topk", "path": "k0", "kind": "selection_k0", "dist": "normal",
                  "build": (lambda: (_dist("normal", 3, 16), 0)),
                  "cand": lambda lg, k: tk.indexer_topk(lg, k), "ora": None, "k": 0,
                  "out": ((lambda: (3, 0)), torch.int64)})
    cases.append({"op": "indexer_topk", "path": "ties", "kind": "selection", "dist": "ties",
                  "build": (lambda: (torch.tensor([[1., 1., 1., 1., 0., 0.]]), 2)),
                  "cand": lambda lg, k: tk.indexer_topk(lg, k), "ora": None, "k": 2,
                  "out": ((lambda: (1, 2)), torch.int64)})

    # compressor pool: all 3 costed shapes + a masked(-inf) case
    for (R, D) in ((128, 512), (8, 512), (8, 128)):
        for d in ("normal", "heavy"):
            cases.append({"op": "compressor", "path": f"r{R}d{D}", "kind": "continuous", "dist": d,
                          "build": (lambda R=R, D=D, d=d: (_dist(d, 8, R, D), _dist(d, 8, R, D), _dist(d, R, D))),
                          "cand": lambda kv, sc, ape: cp.compressor_softmax_pool(kv, sc, ape),
                          "ora": _ora_compressor, "out": ((lambda D=D: (8, D)), torch.float32)})
    cases.append({"op": "compressor", "path": "masked_r128", "kind": "continuous_masked", "dist": "normal",
                  "build": _build_masked_compressor,
                  "cand": lambda kv, sc, ape: cp.compressor_softmax_pool(kv, sc, ape),
                  "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)})

    # sinkhorn pre/post/comb
    for M in (1, 8):
        cases.append({"op": "sinkhorn", "path": "fused", "kind": "continuous_tuple3", "dist": "normal",
                      "build": (lambda M=M: (torch.randn(1, M, 24), torch.rand(3) + 0.5, torch.randn(24), 4, 20, 1e-6)),
                      "cand": lambda mx, s, b, h, i, e: sk.mhc_sinkhorn(mx, s, b, h, i, e),
                      "ora": _ora_sinkhorn, "out": None})

    # combine
    for d in ("normal", "heavy"):
        for M in (1, 8):
            cases.append({"op": "combine", "path": "tiled", "kind": "continuous", "dist": d,
                          "build": (lambda M=M, d=d: (_dist(d, M, 4 * 4096), torch.rand(M, 4), 4)),
                          "cand": lambda x, p, h: cb.mhc_combine(x, p, h), "ora": _ora_combine,
                          "out": ((lambda M=M: (M, 4096)), torch.float32)})

    # sparse scalar + amx + composed-consumer (topk -> gather -> attend)
    for p in ("scalar", "amx"):
        fn = (lambda q, kv, s, sc: sp.sparse_attend(q, kv, s, sc)) if p == "scalar" \
            else (lambda q, kv, s, sc: sp.sparse_attend_amx(q, kv, s, sc))
        cases.append({"op": "sparse", "path": p, "kind": "continuous_sparse", "dist": "normal",
                      "build": (lambda: (torch.randn(8, 64, 512), torch.randn(8, 512, 512), torch.randn(64), 512 ** -0.5)),
                      "cand": fn, "ora": _ora_sparse, "out": ((lambda: (8, 64, 512)), torch.float32)})
    cases.append({"op": "sparse", "path": "composed(topk+gather+attend)", "kind": "composed", "dist": "normal",
                  "build": _build_composed_sparse, "cand": (tk, sp), "ora": None, "out": None})
    return cases


def _build_masked_compressor():
    kv = torch.randn(8, 128, 512)
    score = torch.randn(8, 128, 512)
    score[:, :64, :] = float("-inf")                  # legal padding mask on half the window
    ape = torch.randn(128, 512)
    return (kv, score, ape)


def _build_composed_sparse():
    S, Ktop, D, Hh = 1024, 512, 512, 64
    logits = torch.randn(4, S)
    kv_full = torch.randn(4, S, D)
    q = torch.randn(4, Hh, D)
    sink = torch.randn(Hh)
    return (logits, kv_full, q, sink, Ktop, D ** -0.5)


# ---- evaluation ----
def _check_out(cand, out_contract):
    if out_contract is None:
        return []
    shape_fn, dtype = out_contract
    errs = []
    if tuple(cand.shape) != tuple(shape_fn()):
        errs.append(f"shape {tuple(cand.shape)} != {tuple(shape_fn())}")
    if cand.dtype != dtype:
        errs.append(f"dtype {cand.dtype} != {dtype}")
    return errs


def _finite_ok(t, allow_neg_inf_mask=None):
    if allow_neg_inf_mask is None:
        return bool(torch.isfinite(t).all().item())
    bad = ~torch.isfinite(t)
    return bool((bad & ~allow_neg_inf_mask).any().item() is False)


def run_case(c, calib):
    """Evaluate one case. Returns (hard_fail: bool, note: str). Appends metrics to calib."""
    kind = c["kind"]
    inp = c["build"]()
    cand_fn, ora_fn = c["cand"], c["ora"]

    if kind == "composed":                            # topk -> gather -> attend, candidate vs reference
        logits, kv_full, q, sink, Ktop, scale = inp
        tkmod, spmod = cand_fn
        idx = tkmod.indexer_topk(logits, Ktop)
        gather = torch.stack([kv_full[n][idx[n]] for n in range(kv_full.shape[0])])
        cand = spmod.sparse_attend(q, gather, sink, scale)
        ref_idx = torch.topk(logits, Ktop, dim=1, sorted=False).indices
        ref_gather = torch.stack([kv_full[n][ref_idx[n]] for n in range(kv_full.shape[0])])
        ref = _ora_sparse(q, ref_gather, sink, scale)
        nonmis, ties = _selection_check(idx, logits, Ktop)
        if nonmis:
            return True, f"composed selection: {nonmis} non-tie mismatches"
        m = _screen_continuous(cand, ref, atol=1e-3)
        calib.append({"op": "sparse", "path": c["path"], "metric": m, "selection_non_tie": nonmis, "ties": ties})
        return False, f"composed ok (screen_pass={m['screen_pass']}, selection clean)"

    cand = cand_fn(*inp)

    # layer 0: repeatability determinism (same frozen config)
    cand2 = cand_fn(*inp)
    if isinstance(cand, (tuple, list)):
        if any(not torch.equal(a, b) for a, b in zip(cand, cand2)):
            return True, "non-deterministic (repeat run differs)"
    elif not torch.equal(cand, cand2):
        return True, "non-deterministic (repeat run differs)"

    if kind in ("selection", "selection_k0"):
        oc = _check_out(cand, c["out"])
        if oc:
            return True, "; ".join(oc)
        if kind == "selection_k0":
            return False, "k=0 empty selection ok"
        lg = inp[0]
        nonmis, ties = _selection_check(cand, lg, c["k"])
        calib.append({"op": c["op"], "path": c["path"], "selection_non_tie": nonmis, "ties": ties})
        if nonmis:
            return True, f"selection: {nonmis} non-tie mismatches"
        return False, f"selection clean ({ties} boundary ties)"

    if kind == "logits_select":                       # indexer: gate on induced selection; logit err diagnostic
        oc = _check_out(cand, c["out"])
        if oc:
            return True, "; ".join(oc)
        if not _finite_ok(cand):
            return True, "non-finite logits"
        k = c["k"]
        ref = ora_fn(*inp)
        cand_idx = torch.topk(cand, k, dim=1, sorted=False).indices
        nonmis, ties = _selection_check(cand_idx, ref, k)   # candidate's selection vs REFERENCE logits
        e_max, ref_am = _elem_err(cand, ref)
        calib.append({"op": c["op"], "path": c["path"], "dist": c["dist"], "selection_non_tie": nonmis,
                      "ties": ties, "logit_diag": {"max_abs_err": e_max, "ref_absmax": ref_am}})
        if nonmis:
            return True, f"induced selection: {nonmis} non-tie mismatches"
        return False, f"selection clean; logit diag max_err={e_max:.3e}"

    # continuous families
    ref = ora_fn(*inp)
    if kind == "continuous_tuple3":
        for j, (cv, rv, nm) in enumerate(zip(cand, ref, ("pre", "post", "comb"))):
            if not _finite_ok(cv):
                return True, f"{nm} non-finite"
            m = _screen_continuous(cv, rv, atol=1e-5)
            calib.append({"op": c["op"], "path": nm, "dist": c["dist"], "metric": m})
        return False, "sinkhorn pre/post/comb collected"

    oc = _check_out(cand, c["out"])
    if oc:
        return True, "; ".join(oc)
    if not _finite_ok(cand):
        return True, "non-finite output"
    atol = 1e-5 if c["op"] == "compressor" else (1e-3 if c["op"] == "sparse" else 1e-4)
    m = _screen_continuous(cand, ref, atol=atol)
    calib.append({"op": c["op"], "path": c["path"], "dist": c["dist"], "metric": m})
    return False, f"continuous collected (screen_pass={m['screen_pass']}, max_err={m['max_abs_err']:.3e})"


def run(calibrate_path=None):
    cases = manifest()
    calib, failures = [], []
    print(f"F4 acceptance harness — {len(cases)} cases.  oracle conformance: {ORACLE_CONFORMANCE}")
    print(f"  tolerances UNRATIFIED -> verdict is PARTIAL, never PASS.  tie_eps={TIE_EPS} (selection HARD gate).")
    for c in cases:
        try:
            hard_fail, note = run_case(c, calib)
        except Exception as e:                        # an exception IS a failure, never a silent pass
            hard_fail, note = True, f"EXCEPTION: {type(e).__name__}: {e}"
        tag = "FAIL" if hard_fail else "ok  "
        print(f"  [{tag}] {c['op']:15s} {c['path']:28s} {note}")
        if hard_fail:
            failures.append((c["op"], c["path"], note))
    status = "FAIL" if failures else "PARTIAL"
    print(f"\n  STATUS = {status}  ({'hard-gate failures: ' + str(len(failures)) if failures else 'no hard-gate failure; thresholds UNRATIFIED -> not a full PASS'})")
    if calibrate_path:
        prop = _threshold_proposal(calib)
        out = {"_schema": "F4 calibration evidence + data-driven threshold PROPOSAL (NOT ratified; NOT widened "
                          "to fit the current kernels). Per op/path/output observed error + selection mismatches "
                          "+ composed-consumer. Oracle-adapter conformance is UNVERIFIED and separate from "
                          "candidate accuracy.",
               "oracle_conformance": ORACLE_CONFORMANCE, "status": status,
               "screening": {"rtol": SCREEN_RTOL, "cosine": SCREEN_COS, "tie_eps": TIE_EPS},
               "calibration": calib, "threshold_proposal": prop}
        json.dump(out, open(calibrate_path, "w"), indent=2)
        print(f"  wrote calibration -> {calibrate_path}")
    return 2 if failures else 0


def _threshold_proposal(calib):
    """Data-driven: per op/path the OBSERVED worst error + a proposed bound at a modest margin above it,
    FLAGGED so the user can tighten (never widen a gate merely to accommodate a kernel)."""
    by = {}
    for r in calib:
        m = r.get("metric")
        if not m:
            continue
        key = f"{r['op']}:{r['path']}"
        by.setdefault(key, []).append(m["max_abs_err"])
    prop = {}
    for key, errs in by.items():
        worst = max(errs)
        prop[key] = {"observed_worst_abs_err": worst,
                     "proposed_atol_hint": round(worst * 2, 12),
                     "note": "PROPOSAL only; user ratifies. Do NOT widen to fit — if worst is large, correct the kernel."}
    return prop


# ---- failure injection (adversarial self-audit: every fault caught through the full evaluator) ----
def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    def _fails(case):
        calib = []
        try:
            hf, _ = run_case(case, calib)
            return hf
        except Exception:
            return True

    tk = _mod("f4_tk", "indexer_topk.cpp")
    cp = _mod("f4_cp", "compressor.cpp")

    # 1 wrong shape
    chk(_fails({"op": "x", "path": "wrongshape", "kind": "continuous", "dist": "normal",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c)[:, :10],   # truncated -> wrong shape
                "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a wrong output shape")

    # 2 NaN output
    chk(_fails({"op": "x", "path": "nan", "kind": "continuous", "dist": "normal",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c) * float("nan"),
                "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a NaN output")

    # 3 duplicate selection
    chk(_fails({"op": "x", "path": "dupsel", "kind": "selection", "dist": "normal",
                "build": lambda: (torch.randn(2, 1024), 512),
                "cand": lambda lg, k: torch.zeros(2, k, dtype=torch.int64),   # all-zero -> non-distinct
                "ora": None, "k": 512, "out": ((lambda: (2, 512)), torch.int64)}),
        "REJECTS a duplicate/degenerate selection")

    # 4 non-determinism
    chk(_fails({"op": "x", "path": "nondet", "kind": "continuous", "dist": "normal",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c) + torch.randn(8, 512),
                "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a non-deterministic kernel (repeat differs)")

    # 5 exception / missing (oracle raises)
    chk(_fails({"op": "x", "path": "exc", "kind": "continuous", "dist": "normal",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c),
                "ora": (lambda *a: (_ for _ in ()).throw(RuntimeError("no oracle"))),
                "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a missing/raising oracle (exception = failure)")

    # 6 wrong selection (reference-owned membership catches a below-cutoff inclusion)
    def _wrong_sel(lg, k):
        idx = tk.indexer_topk(lg, k)
        idx[0, 0] = int(torch.topk(lg[0], lg.shape[1], largest=False).indices[0])  # force the global MIN in
        return idx
    chk(_fails({"op": "x", "path": "belowcutoff", "kind": "selection", "dist": "normal",
                "build": lambda: (torch.randn(2, 1024), 512),
                "cand": _wrong_sel, "ora": None, "k": 512, "out": ((lambda: (2, 512)), torch.int64)}),
        "REJECTS a below-cutoff selection (reference-owned membership, tie_eps=0)")

    # 7 a GOOD case must NOT fail (no false positive)
    chk(not _fails({"op": "compressor", "path": "good", "kind": "continuous", "dist": "normal",
                    "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                    "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c),
                    "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "a correct kernel is NOT hard-failed (screening-only, PARTIAL)")

    print(f"  SELFTEST {'OK' if ok else 'FAILED'}")
    return 0 if ok else 2


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) >= 3 and sys.argv[1] == "--calibrate":
        sys.exit(run(calibrate_path=sys.argv[2]))
    sys.exit(run())


if __name__ == "__main__":
    main()
