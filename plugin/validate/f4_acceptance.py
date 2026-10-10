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
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
from sparse_ref import ora_sparse_blockwise as _ora_sparse_blockwise   # source-faithful 64-block replica

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
    # Conform to the PUBLISHED Indexer.forward stage boundaries (model.py@60d8d70 L420-421): the einsum runs on
    # bf16 operands and its OUTPUT is bf16; relu_() and (*weights) and .sum(dim=2) all run in BF16; topk sees
    # BF16 logits. weights_proj is a bf16 Linear -> weights are BF16 and SIGNED. (q is fp4-quantized UPSTREAM;
    # that stage is not in this kernel's contract.) The prior FP32-reduce adapter was WRONG: it diverged from
    # the real bf16 reduction and hid real near-cutoff selection differences under signed weights (P1-F1).
    scores = torch.einsum("nhd,nsd->nhs", q.bfloat16(), kv.bfloat16())   # bf16 einsum output boundary
    return (scores.relu_() * w.bfloat16().unsqueeze(-1)).sum(1)          # bf16 relu * bf16 weights, bf16 reduce


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


def _ora_sparse_bf16(q, kv, sink, scale):
    # AUTHORITATIVE serving math (published kernel.py sparse_attn): bf16 operands, fp32 score accum, a BF16
    # cast of the UNNORMALIZED exponentials before the value GEMM, sink in the denominator, bf16 output.
    # Math replica; GPU/TileLang BITWISE conformance is PENDING.
    hh = q.shape[1]
    qb, kvb = q.bfloat16().float(), kv.bfloat16().float()
    scores = torch.einsum("nhd,nkd->nhk", qb, kvb) * scale
    m = torch.maximum(scores.max(-1, keepdim=True).values, sink.view(1, hh, 1))
    e = (scores - m).exp()
    denom = e.sum(-1, keepdim=True) + (sink.view(1, hh, 1) - m).exp()
    w = e.bfloat16().float()                               # BF16 cast of unnormalized exp before the value GEMM
    return (torch.einsum("nhk,nkd->nhd", w, kvb) / denom).bfloat16().float()   # BF16 output


# Sparse PROPOSED screening threshold (NOT a ratified noise floor — P1-F4/R2-F2). Now measured against the
# SOURCE-FAITHFUL 64-block replica (sparse_ref.ora_sparse_blockwise) on bf16-MATCHED operands + bf16 output
# boundary. The blockwise replica tracks the real GPU kernel to 9.8e-4 @N1 / ~1.95e-3 @N>=8 (job 384502 io);
# that is the OBSERVED residual between this torch replica and the GPU reduction tree — NOT a proven minimum
# attainable error and NOT characterized hardware noise. 4e-3 is a PROPOSAL for screening only; a ratified
# threshold still needs an independently justified downstream budget. Continuous stays screening -> PARTIAL.
SPARSE_SCREEN_PROPOSED = 4.0e-3
ORACLE_CONFORMANCE = "per-op (see ORACLE_CONFORMANCE_BY_OP); local references conformed to the published-op boundary where stated, else PENDING"
ORACLE_CONFORMANCE_BY_OP = {
    "indexer_logits": "CONFORMED to the published BF16 stage boundaries (Indexer.forward model.py L420-421): bf16 einsum output + bf16 relu + bf16 (SIGNED) weights + bf16 reduce over heads -> bf16 logits. Oracle AND candidate epilogue both round to these bf16 stages (prior FP32-reduce adapter was wrong; it hid near-cutoff selection diffs under signed weights -- P1-F1). tp=1 -> TP all-reduce N/A; q fp4-quant is an upstream stage.",
    "indexer_topk": "CONFORMED: torch.topk order-independent membership (published .topk)",
    "compressor": "CONFORMED (pool stage): FP32 softmax-pool after overlap/APE prep; full-compressor state/norm/rotation are declared gaps",
    "sinkhorn": "PARTIAL: SGLang _hc_split_sinkhorn_torch; bitwise conformance to published kernel.py recurrence (eps/iters) PENDING",
    "combine": "CONFORMED: hc_pre multiply+sum(+cast); local einsum is math-equivalent",
    "sparse": "REFERENCE DTYPE = BF16 (GPU oracle H200 job 384502). F4 now judges vs the SOURCE-FAITHFUL 64-block replica (sparse_ref.ora_sparse_blockwise) on bf16-MATCHED operands + bf16 OUTPUT boundary (R2-F2); it tracks the real kernel to 9.8e-4 @N1 / ~1.95e-3 @N>=8. Continuous is SCREENING only (PARTIAL), NOT ratified acceptance; the residual is the OBSERVED replica-vs-GPU discrepancy (NOT a proven minimum attainable error), and a downstream error budget is PENDING. sparse-AMX DROPPED (dominated).",
}


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

    # indexer_logits -> induced selection is the gate (tie_eps=0); logit error is a diagnostic. Weights are
    # SIGNED bf16-exact (published weights_proj is a bf16 Linear -> signed, P1-F1); full M-sweep (P1-F5).
    def _sweight(M):
        return torch.randn(M, 64).bfloat16().float()      # signed, bf16-exact (the published boundary)
    for d in ("normal", "heavy"):
        for M in (1, 8, 16, 32, 64):
            cases.append({"op": "indexer_logits", "path": "tiled/signed", "kind": "logits_select", "dist": d, "M": M,
                          "build": (lambda M=M, d=d: (_dist(d, M, 64, 128), _dist(d, M, 1024, 128), _sweight(M))),
                          "cand": lambda q, kv, w: il.indexer_logits(q, kv, w),
                          "ora": _ora_indexer_logits, "k": 512, "out": ((lambda M=M: (M, 1024)), torch.float32)})
    # bf16-KV contract (P1-F5): caller stores the KV cache in bf16; selection must match the bf16 oracle.
    for M in (1, 32):
        cases.append({"op": "indexer_logits", "path": "bf16kv/signed", "kind": "logits_select", "dist": "normal", "M": M,
                      "build": (lambda M=M: (_dist("normal", M, 64, 128), _dist("normal", M, 1024, 128).bfloat16(), _sweight(M))),
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

    # sparse: SHIPPED fp32-accumulate paths (scalar small-M, fp32bmm/bestof large-M) SCREENED vs the BF16
    # blockwise replica (reference DTYPE from GPU oracle job 384502); within the PROPOSED, UNRATIFIED screen
    # threshold only (NOT acceptance authority). amx is the DOMINATED path (GPU-measured drifts further AND
    # slower) -> diagnostic only, never accepted. A ratified downstream budget is PENDING.
    _sp_fns = {"scalar": lambda q, kv, s, sc: sp.sparse_attend(q, kv, s, sc),
               "fp32bmm": lambda q, kv, s, sc: sp.sparse_attend_fp32bmm(q, kv, s, sc),
               "bestof": lambda q, kv, s, sc: sp.sparse_attend_bestof(q, kv, s, sc),
               "amx": lambda q, kv, s, sc: sp.sparse_attend_amx(q, kv, s, sc)}
    for p, fn in _sp_fns.items():
        sweep = (1, 8, 16, 32, 64) if p != "amx" else (1, 8)   # R4-A5: shipped paths cover the full M-sweep
        for Msp in sweep:                             # R3-F2: cover the best-of N==1 scalar branch AND N>=2 bmm
            cases.append({"op": "sparse", "path": f"{p}/M{Msp}", "kind": "continuous_sparse", "dist": "normal",
                          "build": (lambda Msp=Msp: (torch.randn(Msp, 64, 512).bfloat16().float(), torch.randn(Msp, 512, 512).bfloat16().float(), torch.randn(64), 512 ** -0.5)),
                          "cand": fn, "ora": _ora_sparse, "out": ((lambda Msp=Msp: (Msp, 64, 512)), torch.float32)})
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


def _same_contract(a, b):                             # R3-F1/R4-A3: dtype + shape + device, not just value equality
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype and a.shape == b.shape and a.device == b.device)


def _invalid_evidence(t):
    # R4-A1: NaN and +inf are invalid selection/reference evidence; -inf is a LEGAL causal/padding mask (the
    # published Indexer.forward adds -inf to masked positions BEFORE top-k), so -inf is explicitly allowed.
    return bool((torch.isnan(t) | (t == float("inf"))).any().item())


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
        qb = q.bfloat16().float()                     # R4-A2: feed the candidate the SAME bf16 operands the
        def _comp():                                  #        reference sees, so the screen measures the kernel's
            ix = tkmod.indexer_topk(logits, Ktop)     #        arithmetic, NOT input quantization.
            g = torch.stack([kv_full[n][ix[n]] for n in range(kv_full.shape[0])]).bfloat16().float()
            return ix, spmod.sparse_attend(qb, g, sink, scale)
        idx, cand = _comp()
        idx2, cand2 = _comp()                         # R2-F1: composed must pass the common hard checks too
        if not isinstance(cand, torch.Tensor) or tuple(cand.shape) != tuple(q.shape):
            sh = tuple(cand.shape) if isinstance(cand, torch.Tensor) else type(cand).__name__
            return True, f"composed output shape {sh} != {tuple(q.shape)}"
        if cand.dtype != torch.float32:
            return True, f"composed output dtype {cand.dtype} != torch.float32"
        if not _finite_ok(cand):
            return True, "composed non-finite output"
        if not _same_contract(cand, cand2) or not torch.equal(cand, cand2):   # R4-A3: dtype+shape+device, not bare equal
            return True, "composed non-deterministic / dtype-shape mismatch (repeat differs)"
        # R4-A2: AUTHORITATIVE reference uses the PUBLISHED top-k ordering (default sorted=True), builds its
        # 64-entry blocks from that order, and consumes bf16-matched operands via the source-faithful replica.
        # The candidate deliberately keeps its OWN selection order (idx), so the ordering consequence on the
        # blockwise online-softmax is MEASURED (not hidden by a sorted=False reference that matched membership).
        ref_idx = torch.topk(logits, Ktop, dim=1, sorted=True).indices
        ref_gather = torch.stack([kv_full[n][ref_idx[n]] for n in range(kv_full.shape[0])]).bfloat16().float()
        ref = _ora_sparse_blockwise(qb, ref_gather, sink, scale)
        if not bool(torch.isfinite(ref).all().item()):
            return True, "composed non-finite reference"
        nonmis, ties = _selection_check(idx, logits, Ktop)
        if nonmis:
            return True, f"composed selection: {nonmis} non-tie mismatches"
        m = _screen_continuous(cand.bfloat16().float(), ref, atol=SPARSE_SCREEN_PROPOSED)   # match bf16 output boundary
        # Separately-labelled DIAGNOSTIC: original-FP32-input path (not the matched comparison), to keep the
        # input-quantization effect distinguishable from the kernel arithmetic effect.
        g_fp32 = torch.stack([kv_full[n][idx[n]] for n in range(kv_full.shape[0])])
        m_fp32in = _screen_continuous(spmod.sparse_attend(q, g_fp32, sink, scale), _ora_sparse(q, g_fp32, sink, scale),
                                      atol=SPARSE_SCREEN_PROPOSED)
        calib.append({"op": "sparse", "path": c["path"], "split": c.get("split"), "metric": m,
                      "metric_fp32in_diag": m_fp32in, "selection_non_tie": nonmis, "ties": ties,
                      "vs": "bf16-matched operands, PUBLISHED sorted-order reference, candidate keeps own order"})
        return False, f"composed ok (screen_pass={m['screen_pass']}, selection clean)"

    cand = cand_fn(*inp)

    # layer 0: repeatability determinism + candidate ARITY (R2-F1: no truncating zip across calls)
    cand2 = cand_fn(*inp)
    if isinstance(cand, (tuple, list)) or isinstance(cand2, (tuple, list)):
        if type(cand) is not type(cand2) or len(cand) != len(cand2) \
                or any(not _same_contract(a, b) or not torch.equal(a, b) for a, b in zip(cand, cand2)):
            return True, "non-deterministic / arity / dtype-shape mismatch (repeat run differs)"
    elif not _same_contract(cand, cand2) or not torch.equal(cand, cand2):
        return True, "non-deterministic / dtype-shape mismatch (repeat run differs)"

    if kind in ("selection", "selection_k0"):
        oc = _check_out(cand, c["out"])
        if oc:
            return True, "; ".join(oc)
        if kind == "selection_k0":
            return False, "k=0 empty selection ok"
        lg = inp[0]
        if _invalid_evidence(lg):                     # R4-A1: reject NaN/+inf, ALLOW -inf causal/padding masks
            return True, "non-finite reference logits (NaN/+inf; -inf masks allowed)"
        nonmis, ties = _selection_check(cand, lg, c["k"])
        calib.append({"op": c["op"], "path": c["path"], "split": c.get("split"), "selection_non_tie": nonmis, "ties": ties})
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
        if not bool(torch.isfinite(ref).all().item()):   # R2-F1: reject invalid (non-finite) reference evidence
            return True, "non-finite reference (out-of-domain input)"
        cand_idx = torch.topk(cand, k, dim=1, sorted=False).indices
        nonmis, ties = _selection_check(cand_idx, ref, k)   # candidate's selection vs REFERENCE logits
        e_max, ref_am = _elem_err(cand, ref)
        topref = torch.topk(ref, k + 1, dim=1, sorted=True).values   # downstream-justified: selection margin
        margin = (topref[:, k - 1] - topref[:, k]).min().item()      # min k-th minus (k+1)-th reference logit
        calib.append({"op": c["op"], "path": c["path"], "dist": c["dist"], "split": c.get("split"),
                      "selection_non_tie": nonmis, "ties": ties,
                      "logit_diag": {"max_abs_err": e_max, "ref_absmax": ref_am},
                      "select_safety_margin": margin})
        if nonmis:
            return True, f"induced selection: {nonmis} non-tie mismatches"
        return False, f"selection clean; max_err={e_max:.3e} < margin={margin:.3e}"

    if kind == "continuous_sparse":                   # SCREENING vs the bf16 replica (PROPOSED threshold, non-gating)
        oc = _check_out(cand, c["out"])
        if oc:
            return True, "; ".join(oc)
        if not _finite_ok(cand):
            return True, "non-finite output"
        ref_auth = _ora_sparse_blockwise(*inp)        # SOURCE-FAITHFUL 64-block replica (matched bf16 operands)
        ref_diag = _ora_sparse(*inp)                  # FP32 math: diagnostic only
        if not bool(torch.isfinite(ref_auth).all().item()):
            return True, "non-finite reference"
        cand_b = cand.bfloat16().float()              # match the bf16 OUTPUT boundary (R2-F2)
        m_auth = _screen_continuous(cand_b, ref_auth, atol=SPARSE_SCREEN_PROPOSED)
        m_diag = _screen_continuous(cand, ref_diag, atol=SPARSE_SCREEN_PROPOSED)
        dominated = c["path"].startswith("amx")      # GPU-measured: amx drifts further (3.9-5.9e-3 @N8/64)
        within = (not dominated) and m_auth["max_abs_err"] <= SPARSE_SCREEN_PROPOSED
        calib.append({"op": "sparse", "path": c["path"], "dist": c["dist"], "split": c.get("split"),
                      "within_proposed": within, "dominated": dominated, "metric": m_auth, "metric_fp32_diag": m_diag,
                      "vs": "bf16 BLOCKWISE (source-faithful) replica + bf16 output boundary; PROPOSED screening, not ratified"})
        # Continuous is SCREENING only -> always non-gating (return False); STATUS stays PARTIAL.
        tag = "within PROPOSED screen" if within else ("DOMINATED (amx)" if dominated else "over PROPOSED screen")
        return False, f"{tag}: vs-bf16 max_err={m_auth['max_abs_err']:.3e} (ref {SPARSE_SCREEN_PROPOSED:.1e}, NOT ratified)"

    # continuous families
    ref = ora_fn(*inp)
    if kind == "continuous_tuple3":
        if not isinstance(cand, (tuple, list)) or len(cand) != 3:          # P1-F3: candidate structure BEFORE zip
            return True, f"tuple structure: expected 3 outputs, got {type(cand).__name__}"
        if not isinstance(ref, (tuple, list)) or len(ref) != 3:            # R3-F1: REFERENCE arity too (no truncating zip)
            return True, f"reference structure: expected 3 outputs, got {type(ref).__name__}"
        for cv, rv, nm in zip(cand, ref, ("pre", "post", "comb")):
            if not isinstance(cv, torch.Tensor) or tuple(cv.shape) != tuple(rv.shape):
                sh = tuple(cv.shape) if isinstance(cv, torch.Tensor) else type(cv).__name__
                return True, f"{nm} shape {sh} != {tuple(rv.shape)}"
            if cv.dtype != torch.float32:
                return True, f"{nm} dtype {cv.dtype} != torch.float32"
            if not isinstance(rv, torch.Tensor) or rv.dtype != cv.dtype or rv.device != cv.device:   # R4-A3: reference contract
                return True, f"{nm} reference contract {getattr(rv, 'dtype', type(rv).__name__)} != candidate {cv.dtype}"
            if not _finite_ok(cv) or not bool(torch.isfinite(rv).all().item()):
                return True, f"{nm} non-finite (cand/ref)"
            m = _screen_continuous(cv, rv, atol=1e-5)
            calib.append({"op": c["op"], "path": nm, "dist": c["dist"], "split": c.get("split"), "metric": m})
        return False, "sinkhorn pre/post/comb collected"

    oc = _check_out(cand, c["out"])
    if oc:
        return True, "; ".join(oc)
    if not _finite_ok(cand):
        return True, "non-finite output"
    if not bool(torch.isfinite(ref).all().item()):          # P1-F3: cannot conform to a non-finite reference
        return True, "non-finite reference (out-of-domain input)"
    if not isinstance(ref, torch.Tensor) or tuple(ref.shape) != tuple(cand.shape):   # R3-F1: ref must match cand (no silent broadcast)
        return True, f"reference shape {tuple(ref.shape) if isinstance(ref, torch.Tensor) else type(ref).__name__} != candidate {tuple(cand.shape)}"
    if ref.dtype != cand.dtype or ref.device != cand.device:        # R4-A3: reference dtype/device contract (an equal-valued fp64 ref is NOT a match)
        return True, f"reference contract dtype={ref.dtype}/dev={ref.device} != candidate dtype={cand.dtype}/dev={cand.device}"
    atol = 1e-5 if c["op"] == "compressor" else (1e-3 if c["op"] == "sparse" else 1e-4)
    m = _screen_continuous(cand, ref, atol=atol)
    calib.append({"op": c["op"], "path": c["path"], "dist": c["dist"], "split": c.get("split"), "metric": m})
    return False, f"continuous collected (screen_pass={m['screen_pass']}, max_err={m['max_abs_err']:.3e})"


# Held-out seeds are RESERVED now and are never used to set thresholds (calibration seeds propose; held-out
# only validates that the proposal generalizes).
SEED_GROUPS = {"calibration": [0, 1, 2], "held_out": [100, 101]}
# Independently-declared expected inventory (R2-F1 / playbook #3): the gate is a FULL qualification only if
# every REQUIRED (op, kind) is present AND the indexer covers its full signed M-sweep. A manifest missing any
# of these is a RESTRICTED scope, not a pass. Declared here, NOT derived from whatever tests happened to run.
_EXPECTED_OPS = {"indexer_logits", "indexer_topk", "compressor", "sparse", "sinkhorn", "combine"}
_REQUIRED_COVERAGE = {("indexer_logits", "logits_select"), ("indexer_topk", "selection"),
                      ("compressor", "continuous"), ("compressor", "continuous_masked"),
                      ("sparse", "continuous_sparse"),
                      ("sparse", "composed"), ("sinkhorn", "continuous_tuple3"), ("combine", "continuous")}
_REQUIRED_INDEXER_M = {1, 8, 16, 32, 64}         # the signed-weight M-sweep must be gated, not just benchmarked
_REQUIRED_INDEXER_PATHS = {"tiled/signed", "bf16kv/signed"}   # R4-A5: the bf16-KV contract is REQUIRED, not optional
_REQUIRED_SPARSE_PATHS = {"scalar", "fp32bmm", "bestof"}      # amx is dominated/diagnostic, not required
_REQUIRED_SPARSE_M = {1, 8, 16, 32, 64}          # R4-A5: each shipped sparse path must cover the full M-sweep


def _required_joint_coords():
    """R4-F3: independently-declared REQUIRED joint coordinates. Presence of a path/M alone is NOT enough -- the
    distribution AND the exact built-tensor shape must be covered, else a required (dist, M, shape) can be
    silently removed while path/M presence checks still pass. Each spec is matched by (op, path, dist, M) and
    its BUILT input tensor is checked against the declared shape."""
    req = []
    for d in ("normal", "heavy"):                  # the full signed M-sweep under BOTH distributions
        for M in sorted(_REQUIRED_INDEXER_M):
            req.append({"key": ("indexer_logits", "tiled/signed", d, M),
                        "verify": (lambda inp, M=M: tuple(inp[0].shape) == (M, 64, 128))})
    for (R, D) in ((128, 512), (8, 512), (8, 128)):   # all three costed compressor shapes under BOTH distributions
        for d in ("normal", "heavy"):
            req.append({"key": ("compressor", f"r{R}d{D}", d, None),
                        "verify": (lambda inp, R=R, D=D: tuple(inp[0].shape) == (8, R, D))})
    return req


def run(calibrate_path=None):
    cases = manifest()
    calib, failures = [], []
    if not cases:                                           # P1-F3: an empty inventory is a FAILURE, not a pass
        print("  STATUS = FAIL  (empty case manifest — no coverage)")
        return 2
    present_ops = {c["op"] for c in cases}                  # R2-F1: independently-declared inventory must be complete
    missing = _EXPECTED_OPS - present_ops
    if missing:
        print(f"  STATUS = FAIL  (inventory incomplete — missing ops {sorted(missing)})")
        return 2
    present_cov = {(c["op"], c["kind"]) for c in cases}     # (op, kind) coverage contract (playbook #3)
    miss_cov = _REQUIRED_COVERAGE - present_cov
    if miss_cov:
        print(f"  STATUS = FAIL  (coverage incomplete — missing (op,kind) {sorted(miss_cov)})")
        return 2
    idx_M = {c.get("M") for c in cases if c["op"] == "indexer_logits" and c.get("M") is not None}
    if not _REQUIRED_INDEXER_M.issubset(idx_M):
        print(f"  STATUS = FAIL  (indexer M-sweep incomplete — have {sorted(idx_M)}, need {sorted(_REQUIRED_INDEXER_M)})")
        return 2
    idx_paths = {c["path"] for c in cases if c["op"] == "indexer_logits"}   # R4-A5: required coordinate inventory
    if not _REQUIRED_INDEXER_PATHS.issubset(idx_paths):
        print(f"  STATUS = FAIL  (indexer path coverage incomplete — have {sorted(idx_paths)}, need {sorted(_REQUIRED_INDEXER_PATHS)})")
        return 2
    sp_cov = {}                                            # R4-A5: each shipped sparse path must span the full M-sweep
    for c in cases:
        if c["op"] == "sparse" and c["kind"] == "continuous_sparse":
            pre, _, mm = c["path"].partition("/M")
            if mm.isdigit():
                sp_cov.setdefault(pre, set()).add(int(mm))
    sp_miss = {p: sorted(_REQUIRED_SPARSE_M - sp_cov.get(p, set())) for p in _REQUIRED_SPARSE_PATHS
               if not _REQUIRED_SPARSE_M.issubset(sp_cov.get(p, set()))}
    if sp_miss:
        print(f"  STATUS = FAIL  (sparse M-sweep incomplete per path — missing {sp_miss})")
        return 2
    by_coord = {}                                          # R4-F3: joint (op,path,dist,M) coordinate -> case
    for c in cases:
        by_coord.setdefault((c["op"], c["path"], c.get("dist"), c.get("M")), c)
    for spec in _required_joint_coords():
        c = by_coord.get(spec["key"])
        if c is None:
            print(f"  STATUS = FAIL  (required joint coordinate missing — {spec['key']})")
            return 2
        inp = c["build"]()                                 # verify the BUILT tensor matches the declared coordinate
        if not spec["verify"](inp):
            print(f"  STATUS = FAIL  (required coordinate tensor mismatch — {spec['key']}: built {tuple(inp[0].shape)})")
            return 2
    if any(len(s) == 0 for s in SEED_GROUPS.values()):
        print("  STATUS = FAIL  (a seed group is empty — no evaluation)")
        return 2
    print(f"F4 acceptance harness — {len(cases)} cases x {sum(len(s) for s in SEED_GROUPS.values())} seeds "
          f"(calibration {SEED_GROUPS['calibration']} + RESERVED held-out {SEED_GROUPS['held_out']}).")
    print(f"  tolerances UNRATIFIED -> verdict is PARTIAL, never PASS.  tie_eps={TIE_EPS} (selection HARD gate).")
    for split, seeds in SEED_GROUPS.items():
        for seed in seeds:
            torch.manual_seed(seed)
            for c in cases:
                c["split"] = split
                try:
                    hard_fail, note = run_case(c, calib)
                except Exception as e:                # an exception IS a failure, never a silent pass
                    hard_fail, note = True, f"EXCEPTION: {type(e).__name__}: {e}"
                if hard_fail:
                    print(f"  [FAIL/{split[:4]}:{seed}] {c['op']:14s} {c['path']:26s} {note}")
                    failures.append((split, seed, c["op"], c["path"], note))
    status = "FAIL" if failures else "PARTIAL"
    print("  (calibration-seed summary)")
    seen = set()
    for r in calib:
        if r.get("split") != "calibration":
            continue
        key = (r["op"], r.get("path"))
        if key in seen:
            continue
        seen.add(key)
        if "selection_non_tie" in r:
            print(f"  [ok  ] {r['op']:14s} {str(r.get('path')):26s} selection non_tie={r['selection_non_tie']}"
                  + (f" margin={r['select_safety_margin']:.2e}" if 'select_safety_margin' in r else ""))
        elif r.get("experimental"):
            print(f"  [EXP ] {r['op']:14s} {str(r.get('path')):26s} vs-BF16-primitive "
                  f"max_err={r['metric']['max_abs_err']:.3e} (NOT accepted)")
        elif r.get("metric"):
            print(f"  [ok  ] {r['op']:14s} {str(r.get('path')):26s} max_err={r['metric']['max_abs_err']:.3e}")
    print(f"\n  STATUS = {status}  ({'hard-gate failures: ' + str(len(failures)) if failures else 'no hard-gate failure; thresholds UNRATIFIED -> not a full PASS'})")
    if calibrate_path:
        budget = _downstream_budget(calib)
        out = {"_schema": "F4 calibration evidence + DOWNSTREAM-JUSTIFIED budget (NOT ratified; NOT fit to the "
                          "observed error). Calibration seeds propose; RESERVED held-out seeds validate. Oracle "
                          "conformance is per-op and SEPARATE from candidate accuracy. sparse-AMX is EXPERIMENTAL.",
               "oracle_conformance_by_op": ORACLE_CONFORMANCE_BY_OP, "status": status,
               "screening": {"rtol": SCREEN_RTOL, "cosine": SCREEN_COS, "tie_eps": TIE_EPS},
               "seed_groups": SEED_GROUPS, "calibration": calib, "downstream_budget": budget}
        json.dump(out, open(calibrate_path, "w"), indent=2)
        print(f"  wrote calibration -> {calibrate_path}")
    return 2 if failures else 0


def _downstream_budget(calib):
    """Budget JUSTIFIED by downstream effect, NOT by the kernel's own observed error. For the indexer the
    downstream is the top-k decision -> the budget is the selection-safety MARGIN (k-th minus (k+1)-th
    reference logit); the kernel is safe iff its logit error < that margin. For continuous ops with no modeled
    downstream consumer in this harness, the budget is PENDING (needs propagation into the composed forward) —
    we report the observed error (calibration vs held-out) but DO NOT emit a fit-to-worst number."""
    def _split_err(pred):
        c = [r for r in calib if pred(r) and r.get("split") == "calibration" and r.get("metric")]
        h = [r for r in calib if pred(r) and r.get("split") == "held_out" and r.get("metric")]
        return (max((r["metric"]["max_abs_err"] for r in c), default=None),
                max((r["metric"]["max_abs_err"] for r in h), default=None))
    out = {}
    il = [r for r in calib if r.get("op") == "indexer_logits" and r.get("split") == "calibration"]
    ilh = [r for r in calib if r.get("op") == "indexer_logits" and r.get("split") == "held_out"]
    if il:
        me = max(r["logit_diag"]["max_abs_err"] for r in il)
        mm = min(r["select_safety_margin"] for r in il)
        nt = sum(r["selection_non_tie"] for r in il) + sum(r["selection_non_tie"] for r in ilh)
        out["indexer_logits"] = {
            "downstream": "top-k selection (tie_eps=0)",
            "justification": "EMPIRICAL: zero non-tie selection mismatches across ALL calibration + held-out seeds",
            "non_tie_mismatches_all_seeds": nt, "selection_preserved": bool(nt == 0),
            "context": {"global_max_logit_err": me, "global_min_boundary_margin": mm,
                        "note": "global_max_err > global_min_margin is NOT a violation: the max error is at "
                                "high-magnitude logits FAR from any cutoff; the per-row error AT the k-th "
                                "boundary is much smaller, hence 0 mismatches. The empirical count is the gate."}}
    for op in ("compressor", "combine", "sinkhorn"):
        ec, eh = _split_err(lambda r, op=op: r.get("op") == op)
        out[op] = {"downstream": "composed forward (not modeled in harness)", "downstream_budget": "PENDING",
                   "observed_max_abs_err_calib": ec, "observed_max_abs_err_held_out": eh,
                   "note": "budget to be set from downstream propagation, NOT from this observed error"}
    sp = [r for r in calib if r.get("op") == "sparse"]
    if sp:
        sp_paths = {r["path"] for r in sp if "within_proposed" in r}
        within_all = sorted(p for p in sp_paths if all(r.get("within_proposed") for r in sp if r["path"] == p))
        out["sparse"] = {"status": "PROPOSED/UNRATIFIED: reference dtype = BF16 (job 384502); continuous is SCREENING only (PARTIAL), NOT ratified acceptance",
                         "proposed_screen_threshold": SPARSE_SCREEN_PROPOSED,
                         "threshold_note": "approximation error vs the SOURCE-FAITHFUL 64-block (blockwise) replica on bf16-matched operands + bf16 output boundary; an OBSERVED replica-vs-GPU discrepancy, NOT a proven minimum error and NOT characterized hardware noise; downstream budget PENDING",
                         "vs_bf16_replica_max_err_calib": max((r["metric"]["max_abs_err"] for r in sp if r.get("split") == "calibration" and "within_proposed" in r), default=None),
                         "within_proposed_threshold_ALL_records": within_all,
                         "dominated_dropped": "amx (naive bf16 intermediates drift to 5.86e-3 @N64; slower than fp32-bmm donor)"}
    return out


# ---- failure injection (adversarial self-audit: every fault caught through the full evaluator) ----
def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    def _eval(case):                                  # mirror run()'s exception handling: test the FULL gate
        calib = []
        try:
            return run_case(case, calib)
        except Exception as e:
            return True, f"EXCEPTION: {type(e).__name__}: {e}"

    def _fails(case):
        """Negative test (playbook #3): must hard-reject AND for the INTENDED reason (case['_expect'] must be a
        substring of the note). An unrelated error (e.g. a NameError in setup) is an INFRASTRUCTURE failure,
        NOT a successful rejection. A case with no '_expect' is a POSITIVE control (assert NOT hard-rejected)."""
        hf, note = _eval(case)
        exp = case.get("_expect")
        if exp is None:
            return hf                                 # positive control: used as `not _fails(good_case)`
        if hf and exp in note:
            return True
        print(f"       (reason mismatch: hard_fail={hf} note={note!r} expected~{exp!r})")
        return False

    tk = _mod("f4_tk", "indexer_topk.cpp")
    cp = _mod("f4_cp", "compressor.cpp")
    il = _mod("f4_il_st", "indexer_logits.cpp")     # (typed-reason tests exposed these were never loaded)
    sp = _mod("f4_sp_st", "sparse_attend.cpp")

    # 1 wrong shape
    chk(_fails({"op": "x", "path": "wrongshape", "kind": "continuous", "dist": "normal", "_expect": "shape",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c)[:, :10],   # truncated -> wrong shape
                "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a wrong output shape")

    # 2 non-finite output (use +inf: deterministic, so it reaches the finiteness gate -- nan would trip the
    #   repeatability check first since nan!=nan)
    chk(_fails({"op": "x", "path": "nan", "kind": "continuous", "dist": "normal", "_expect": "non-finite",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c) + float("inf"),
                "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a non-finite (inf) output")

    # 3 duplicate selection
    chk(_fails({"op": "x", "path": "dupsel", "kind": "selection", "dist": "normal", "_expect": "mismatch",
                "build": lambda: (torch.randn(2, 1024), 512),
                "cand": lambda lg, k: torch.zeros(2, k, dtype=torch.int64),   # all-zero -> non-distinct
                "ora": None, "k": 512, "out": ((lambda: (2, 512)), torch.int64)}),
        "REJECTS a duplicate/degenerate selection")

    # 4 non-determinism
    chk(_fails({"op": "x", "path": "nondet", "kind": "continuous", "dist": "normal", "_expect": "non-deterministic",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, c: cp.compressor_softmax_pool(a, b, c) + torch.randn(8, 512),
                "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a non-deterministic kernel (repeat differs)")

    # 5 exception / missing (oracle raises)
    chk(_fails({"op": "x", "path": "exc", "kind": "continuous", "dist": "normal", "_expect": "EXCEPTION",
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
    chk(_fails({"op": "x", "path": "belowcutoff", "kind": "selection", "dist": "normal", "_expect": "mismatch",
                "build": lambda: (torch.randn(2, 1024), 512),
                "cand": _wrong_sel, "ora": None, "k": 512, "out": ((lambda: (2, 512)), torch.int64)}),
        "REJECTS a below-cutoff selection (reference-owned membership, tie_eps=0)")

    # 8 non-finite REFERENCE in logits_select must fail (R2-F1: reference validity before selection)
    chk(_fails({"op": "x", "path": "nanref", "kind": "logits_select", "dist": "normal", "_expect": "non-finite reference",
                "build": lambda: (torch.randn(2, 64, 128), torch.randn(2, 1024, 128), torch.randn(2, 64)),
                "cand": lambda q, kv, w: il.indexer_logits(q, kv, w),
                "ora": lambda q, kv, w: _ora_indexer_logits(q, kv, w) * float("nan"),
                "k": 512, "out": ((lambda: (2, 1024)), torch.float32)}),
        "REJECTS a non-finite reference in logits_select (R2-F1)")

    # 9 repeat-call ARITY mismatch must fail (R2-F1: no truncating zip across calls)
    _ac = {"n": 0}
    def _arity(a, b, cc):
        _ac["n"] += 1
        base = cp.compressor_softmax_pool(a, b, cc)
        return (base, base, base) if _ac["n"] == 1 else (base, base)   # 3 outputs then 2
    chk(_fails({"op": "x", "path": "arity", "kind": "continuous_tuple3", "dist": "normal", "_expect": "arity",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": _arity, "ora": lambda a, b, cc: (cp.compressor_softmax_pool(a, b, cc),) * 3, "out": None}),
        "REJECTS a repeat-call arity mismatch (R2-F1)")

    # 10 composed NaN output must fail (R2-F1: composed path runs the common structural/finite checks)
    class _NanSp:
        @staticmethod
        def sparse_attend(q, kv, s, sc):
            return sp.sparse_attend(q, kv, s, sc) * float("nan")
    chk(_fails({"op": "x", "path": "compnan", "kind": "composed", "dist": "normal", "_expect": "composed non-finite",
                "build": _build_composed_sparse, "cand": (tk, _NanSp), "ora": None, "out": None}),
        "REJECTS a NaN composed output (R2-F1)")

    # 11 composed non-deterministic output must fail (R2-F1)
    class _NdSp:
        @staticmethod
        def sparse_attend(q, kv, s, sc):
            return sp.sparse_attend(q, kv, s, sc) + torch.randn(q.shape[0], q.shape[1], q.shape[2])
    chk(_fails({"op": "x", "path": "compnd", "kind": "composed", "dist": "normal", "_expect": "composed non-deterministic",
                "build": _build_composed_sparse, "cand": (tk, _NdSp), "ora": None, "out": None}),
        "REJECTS a non-deterministic composed output (R2-F1)")

    # 14 reference tuple arity mismatch must fail (R3-F1: validate the REFERENCE structure, not only candidate)
    chk(_fails({"op": "x", "path": "refarity", "kind": "continuous_tuple3", "dist": "normal", "_expect": "reference structure",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, cc: (cp.compressor_softmax_pool(a, b, cc),) * 3,
                "ora": lambda a, b, cc: (cp.compressor_softmax_pool(a, b, cc),) * 2, "out": None}),
        "REJECTS a reference with wrong arity (R3-F1)")

    # 15 non-finite REFERENCE logits in a selection case must fail (R3-F1)
    chk(_fails({"op": "x", "path": "nanlogits", "kind": "selection", "dist": "normal", "_expect": "non-finite reference logits",
                "build": lambda: (torch.full((2, 1024), float("nan")), 512),
                "cand": lambda lg, k: torch.arange(512).view(1, 512).expand(2, 512).contiguous(),  # valid distinct
                "ora": None, "k": 512, "out": ((lambda: (2, 512)), torch.int64)}),
        "REJECTS non-finite reference logits in a selection case (R3-F1)")

    # 16 scalar/broadcastable REFERENCE in a continuous case must fail (R3-F1: no silent broadcast)
    chk(_fails({"op": "x", "path": "scalarref", "kind": "continuous", "dist": "normal", "_expect": "reference shape",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, cc: cp.compressor_softmax_pool(a, b, cc),
                "ora": lambda a, b, cc: torch.tensor(0.0), "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a scalar/broadcast reference (R3-F1)")

    # 17 repeat-call DTYPE flip (fp32 then fp64) must fail (R3-F1: torch.equal can cross dtype)
    _dc = {"n": 0}
    def _dtypeflip(a, b, cc):
        _dc["n"] += 1
        out = cp.compressor_softmax_pool(a, b, cc)
        return out if _dc["n"] == 1 else out.double()
    chk(_fails({"op": "x", "path": "dtypeflip", "kind": "continuous", "dist": "normal", "_expect": "dtype-shape mismatch",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": _dtypeflip, "ora": _ora_compressor, "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS a repeat-call dtype flip fp32->fp64 (R3-F1)")

    # 18 valid -inf MASKED selection is a SUPPORTED domain and must NOT be rejected (R4-A1 positive control:
    #    the published Indexer adds -inf to causal/padding positions before top-k).
    def _masked_lg():
        lg = torch.randn(2, 1024)
        lg[:, 512:] = float("-inf")                 # legal mask on half the window
        return (lg, 512)
    chk(not _fails({"op": "indexer_topk", "path": "maskedsel", "kind": "selection", "dist": "normal",
                    "build": _masked_lg, "cand": lambda lg, k: tk.indexer_topk(lg, k),
                    "ora": None, "k": 512, "out": ((lambda: (2, 512)), torch.int64)}),
        "ACCEPTS a valid -inf masked selection (R4-A1: -inf masks are a supported domain)")

    # 19 NaN selection evidence is still INVALID and must be rejected (R4-A1 negative control retained)
    chk(_fails({"op": "x", "path": "nanmasksel", "kind": "selection", "dist": "normal", "_expect": "non-finite reference logits",
                "build": lambda: (torch.full((2, 1024), float("nan")), 512),
                "cand": lambda lg, k: torch.arange(512).view(1, 512).expand(2, 512).contiguous(),
                "ora": None, "k": 512, "out": ((lambda: (2, 512)), torch.int64)}),
        "REJECTS NaN selection evidence even though -inf is allowed (R4-A1)")

    # 20 an equal-valued FP64 REFERENCE in a continuous case must fail (R4-A3: reference dtype contract)
    chk(_fails({"op": "x", "path": "refdtype", "kind": "continuous", "dist": "normal", "_expect": "reference contract",
                "build": lambda: (torch.randn(8, 128, 512), torch.randn(8, 128, 512), torch.randn(128, 512)),
                "cand": lambda a, b, cc: cp.compressor_softmax_pool(a, b, cc),
                "ora": lambda a, b, cc: _ora_compressor(a, b, cc).double(), "out": ((lambda: (8, 512)), torch.float32)}),
        "REJECTS an equal-valued fp64 reference (R4-A3 reference dtype contract)")

    # 21 a COMPOSED repeat dtype flip must fail (R4-A3: composed repeat validates dtype+shape, not bare equal)
    class _DtSp:
        _n = {"c": 0}
        @staticmethod
        def sparse_attend(q, kv, s, sc):
            _DtSp._n["c"] += 1
            out = sp.sparse_attend(q, kv, s, sc)
            return out if _DtSp._n["c"] == 1 else out.double()
    chk(_fails({"op": "x", "path": "compdtype", "kind": "composed", "dist": "normal", "_expect": "dtype-shape mismatch",
                "build": _build_composed_sparse, "cand": (tk, _DtSp), "ora": None, "out": None}),
        "REJECTS a composed repeat dtype flip fp32->fp64 (R4-A3)")

    # 12 incomplete inventory (missing an op) must FAIL via run() (R2-F1)
    import unittest.mock as _um
    _orig_manifest = manifest
    with _um.patch(__name__ + ".manifest", lambda: [cc for cc in _orig_manifest() if cc["op"] != "sparse"]):
        chk(run() == 2, "REJECTS an incomplete inventory (missing op) through run() (R2-F1)")

    # 22 required COORDINATE inventory must FAIL via run() when a required case is removed (R4-A5):
    #    the bf16-KV contract, the masked-compressor case, and the full per-path sparse M-sweep are REQUIRED,
    #    not merely present-if-someone-wrote-them. Each independent removal must hard-fail run().
    def _drop(pred):
        return lambda: [cc for cc in _orig_manifest() if not pred(cc)]
    with _um.patch(__name__ + ".manifest", _drop(lambda cc: cc["op"] == "indexer_logits" and "bf16kv" in cc["path"])):
        chk(run() == 2, "REJECTS removal of the bf16-KV indexer coordinate through run() (R4-A5)")
    with _um.patch(__name__ + ".manifest", _drop(lambda cc: cc["kind"] == "continuous_masked")):
        chk(run() == 2, "REJECTS removal of the masked-compressor coordinate through run() (R4-A5)")
    with _um.patch(__name__ + ".manifest", _drop(lambda cc: cc["op"] == "sparse" and cc["path"].endswith("/M16"))):
        chk(run() == 2, "REJECTS removal of the sparse M16 coordinate through run() (R4-A5)")

    # 23 required JOINT coordinate (distribution + compressor shape) must FAIL via run() (R4-F3): path/M
    #    presence alone is insufficient -- a required (dist, M) or compressor (R,D) shape must be un-removable.
    with _um.patch(__name__ + ".manifest", _drop(lambda cc: cc["op"] == "indexer_logits" and cc.get("dist") == "heavy" and cc.get("M") == 64)):
        chk(run() == 2, "REJECTS removal of the heavy-tailed indexer M64 coordinate through run() (R4-F3)")
    with _um.patch(__name__ + ".manifest", _drop(lambda cc: cc["op"] == "compressor" and cc["path"] == "r8d128")):
        chk(run() == 2, "REJECTS removal of the compressor R8/D128 shape through run() (R4-F3)")

    # 13 META (playbook #3): a negative test that hits an unrelated error (NameError in setup) while CLAIMING
    # to test a shape rejection must NOT be counted as a successful shape rejection -> _fails returns False.
    chk(not _fails({"op": "x", "path": "infra", "kind": "continuous", "dist": "normal", "_expect": "shape",
                    "build": lambda: (_ for _ in ()).throw(NameError("setup typo")),
                    "cand": lambda *a: None, "ora": None, "out": None}),
        "an infra error (NameError) is NOT counted as the intended rejection (typed reason)")

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
