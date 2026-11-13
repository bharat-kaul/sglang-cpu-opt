#!/usr/bin/env python3
"""Offline cosine+magnitude per-(pass,op,layer) diff of two FULLCAP .pt dumps.

Keys look like ``pf.L12.r3`` / ``pf.hcpost.L7.r3`` / ``logits.0.r3``. The rank suffix
(``.rN``) is stripped so a CPU rank-0 dump lines up with a TP-sharded GPU dump on the
UNSHARDED signals (decoder-layer hidden, hc_post, logits are all rank-replicated here).

Only replicated, already-global signals may be compared; sharded intermediate tensors
must first be reconstructed by a source-reviewed adapter. Duplicate normalized rank
keys are rejected, never overwritten. Acceptance needs an independently pinned expected
inventory and approved atol/rtol as well as cosine. Without these, output is diagnostic only.

    python diff_fullcap.py <gpu.pt> <cpu.pt> [--cos-thresh 0.999]
"""
import argparse
from pathlib import Path
import re
import sys

import torch

from enablement_gate import GateError, exact, load_json, number, pinned_json, require, sha
from make_dummy_checkpoint import file_digest


RUN_KEYS = {
    "checkpoint_sha256", "input_ids", "max_new_tokens", "dtype", "context_length",
    "chunked_prefill_size", "disable_radix_cache", "runner_sha256",
}


def _strip_rank(k: str) -> str:
    return re.sub(r"\.r-?\d+$", "", k)


def _load(path: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=True)
    require(type(d) is dict and d, "empty/non-dictionary capture")
    out = {}
    for k, v in d.items():
        require(isinstance(k, str) and k.strip(), "invalid capture key")
        key = _strip_rank(k)
        require(key not in out, f"duplicate rank-normalized key: {key}; select one replicated rank explicitly")
        require(isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.numel() > 0,
                f"{key}: expected nonempty floating tensor")
        require(torch.isfinite(v).all().item(), f"{key}: non-finite tensor")
        out[key] = v
    return out


def _sortkey(k: str):
    # Order by pass (pf before dc), then layer id, then op (hidden before hcpost), logits last.
    m = re.match(r"(pf|dc\d+)\.(?:(hcpost)\.)?L(-?\d+)", k)
    if m:
        ps = 0 if m.group(1) == "pf" else 1 + int(m.group(1)[2:])
        op = 1 if m.group(2) == "hcpost" else 0
        return (0, ps, int(m.group(3)), op)
    m = re.match(r"logits\.(\d+)", k)
    if m:
        return (1, 0, int(m.group(1)), 0)
    return (2, 0, 0, 0)


def compare(g: dict, c: dict, cos_thresh: float, atol: float | None,
            rtol: float | None, expected: dict | None) -> bool:
    require(set(g) == set(c), "capture coverage differs; missing/extra keys")
    if expected is not None:
        exact(expected, {"schema_version", "tensors"} | ({"run"} if "run" in expected else set()),
              "expected capture inventory")
        require(type(expected["schema_version"]) is int and expected["schema_version"] == 1, "unsupported capture inventory")
        exact(expected["tensors"], set(g), "source-derived expected coverage")
    passed = True
    for key in sorted(g, key=lambda k: (*_sortkey(k), k)):
        a, b = g[key], c[key]
        require(a.shape == b.shape and a.dtype == b.dtype, f"{key}: shape/dtype mismatch")
        if expected is not None:
            item = exact(expected["tensors"][key], {"shape", "dtype", "layout"}, "expected tensor")
            require(isinstance(item["shape"], list) and all(type(d) is int and d > 0 for d in item["shape"]),
                    f"{key}: invalid expected shape")
            require(item["layout"] == "replicated", "sharded captures require an explicit source-reviewed reconstruction")
            require(item["shape"] == list(a.shape) and item["dtype"] == str(a.dtype).removeprefix("torch."),
                    f"{key}: expected shape/dtype mismatch")
        af, bf = a.double().reshape(-1), b.double().reshape(-1)
        scale_a, scale_b = af.abs().max().item(), bf.abs().max().item()
        if scale_a == 0 or scale_b == 0:
            cos = 1.0 if scale_a == scale_b else 0.0
        else:
            cos = torch.nn.functional.cosine_similarity(af / scale_a, bf / scale_b, dim=0, eps=1e-12).item()
        error = (af - bf).abs()
        maximum = number(error.max().item(), f"{key}: max_abs")
        number(cos, f"{key}: cosine")
        elementwise = True if atol is None else bool(torch.all(error <= atol + rtol * af.abs()).item())
        ok = cos >= cos_thresh and elementwise
        passed &= ok
        print(f"{key}: cos={cos:.9g} max_abs={maximum:.9g} {'PASS' if ok else 'DIVERGE'}")
    return passed


def validate_run_contract(expected: dict) -> None:
    exact(expected, RUN_KEYS, "expected run contract")
    for key in ("checkpoint_sha256", "runner_sha256"):
        sha(expected[key], key)
    require(isinstance(expected["input_ids"], list) and expected["input_ids"]
            and all(type(i) is int and i >= 0 for i in expected["input_ids"]), "invalid expected token IDs")
    require(type(expected["max_new_tokens"]) is int and expected["max_new_tokens"] > 0, "invalid token limit")
    require(expected["dtype"] in ("float32", "float16", "bfloat16"), "invalid expected dtype")
    require(type(expected["context_length"]) is int and expected["context_length"] > 0, "invalid context length")
    require(type(expected["chunked_prefill_size"]) is int, "invalid chunked prefill size")
    require(expected["disable_radix_cache"] is True, "oracle requires disabled radix cache")


def check_run(path: str, expected: dict) -> dict:
    validate_run_contract(expected)
    run = load_json(path + ".oracle.json")
    exact(run, RUN_KEYS | {"schema_version", "device", "tp_size", "capture_rank", "capture_sha256"}, "capture run metadata")
    require(type(run["schema_version"]) is int and run["schema_version"] == 1, "invalid run metadata version")
    require(run["device"] in ("cpu", "cuda") and type(run["tp_size"]) is int and run["tp_size"] > 0, "invalid run device/TP")
    require(type(run["capture_rank"]) is int and run["capture_rank"] == 0, "oracle requires selected rank zero")
    actual = {key: run[key] for key in RUN_KEYS}
    validate_run_contract(actual)
    require(actual == expected, "run contract mismatch")
    require(file_digest(Path(path)) == sha(run["capture_sha256"], "capture_sha256"), "capture digest mismatch")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    require(type(raw) is dict and raw and all(isinstance(key, str) and key.endswith(".r0") for key in raw),
            "capture contains an unexpected/unidentified rank")
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("cpu")
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    ap.add_argument("--atol", type=float)
    ap.add_argument("--rtol", type=float)
    ap.add_argument("--expected", help="source-reviewed full tensor inventory JSON")
    ap.add_argument("--expected-sha256", help="independently approved inventory digest")
    args = ap.parse_args()
    try:
        require(0 < number(args.cos_thresh, "cos-thresh") <= 1, "cos-thresh must be in (0,1]")
        require((args.atol is None) == (args.rtol is None), "supply both atol and rtol")
        if args.atol is not None:
            require(number(args.atol, "atol") >= 0 and number(args.rtol, "rtol") >= 0, "negative tolerance")
        require(bool(args.expected) == bool(args.expected_sha256), "supply expected inventory and approved digest together")
        require(not Path(args.gpu).samefile(args.cpu), "reference and candidate alias the same capture")
        expected = pinned_json(args.expected, args.expected_sha256) if args.expected else None
        run_records = {}
        if expected is not None and "run" in expected:
            run_records = {path: check_run(path, expected["run"]) for path in (args.gpu, args.cpu)}
        passed = compare(_load(args.gpu), _load(args.cpu), args.cos_thresh, args.atol, args.rtol, expected)
        for path, run in run_records.items():
            require(file_digest(Path(path)) == run["capture_sha256"], "capture changed during comparison")
        if not passed:
            print("PARITY FAILED")
            return 1
        if expected is None or args.atol is None:
            print("DIAGNOSTIC ONLY: no acceptance without pinned inventory and approved atol/rtol")
        else:
            print("TENSOR PARITY PASSED for declared replicated signals"
                  + (" with matched checkpoint/input/run metadata" if run_records else "; run/weight provenance is a separate gate"))
        return 0
    except (GateError, OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
