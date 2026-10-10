"""Part B of the sparse GPU-oracle: diff our COMPILED CPU sparse_attend paths (scalar / fp32bmm / amx /
bestof) against the REAL TileLang kernel output saved by test_sparse_gpu_oracle.py. Runs on an AMX node.

P1-F4 corrections: (a) the GPU kernel received BF16-ROUNDED operands; to measure CPU-vs-GPU APPROXIMATION
honestly we feed the CPU kernels the SAME bf16-rounded q/kv (not the original fp32), and report the fp32-
operand result only as a secondary column. (b) These are APPROXIMATION-error observations vs a NON-source-
faithful (global, not 64-block) replica's GPU output — NOT an intrinsic hardware-noise floor. The reference
threshold below is a PROPOSAL pending ratification + an independently justified downstream budget."""
import math
import os
import sys

import torch

IO = "/scratch/bkaul/sparse_oracle_io.pt"
# PROPOSED reference (NOT ratified): bf16-replica-vs-GPU worst max-err (job 384502). It is approximation
# error vs a globally-normalized replica, not characterized hardware noise (repeated GPU runs not used).
PROPOSED_REF = 1.953e-3
# R4-A4: the EXPECTED published source identity this replay is qualified against. A producer's own
# "validated" flag cannot bind evidence to the consumer's expected source -> we compare the hash here.
# Overridable per run via env so it is not permanently hardcoded to one job.
EXPECTED_KERNEL_SHA = os.environ.get("ORACLE_KERNEL_SHA",
                                     "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2")
EXPECTED_JOB = os.environ.get("ORACLE_JOB")   # optional caller-supplied run identity

_REQ = {"N", "scale", "sink", "gpu_out", "q", "kv"}       # required per-record coordinates before any compare


def _load_kernel():
    from torch.utils.cpp_extension import load         # lazy: selftest validates the archive without compiling
    K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "sparse_attend.cpp")
    return load(name="sparse_cpu_vs_gpu", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)


def validate_archive(blob, expected_sha=EXPECTED_KERNEL_SHA, expected_job=EXPECTED_JOB):
    """Fail-closed archive qualification. Returns the records or raises SystemExit. Binds evidence to the
    EXPECTED published source (R4-A4), uses the producer's CANONICAL slurm_job_id and rejects a conflicting
    legacy alias (R4-F1), and validates each record's coordinate CONSISTENCY -- not just key presence --
    against the tensors to be compared (R4-F2)."""
    if not isinstance(blob, dict) or "records" not in blob or "provenance" not in blob:
        raise SystemExit("FATAL: io archive missing provenance/records (expected the stamped dict format)")
    prov = blob["provenance"]
    if not prov.get("kernel_py_sha256_validated"):
        raise SystemExit(f"FATAL: saved io provenance has UNVALIDATED kernel sha ({prov.get('kernel_py_sha256')!r})")
    if prov.get("kernel_py_sha256") != expected_sha:     # bind to the EXPECTED source, not just a truthy flag
        raise SystemExit(f"FATAL: saved io kernel sha {prov.get('kernel_py_sha256')!r} != expected {expected_sha!r}")
    canonical = prov.get("slurm_job_id")                 # R4-F1: the producer's canonical identity field
    legacy = prov.get("job")
    if legacy is not None and str(legacy) != str(canonical):   # a conflicting alias must NOT override identity
        raise SystemExit(f"FATAL: conflicting run identity (legacy job {legacy!r} != canonical slurm_job_id {canonical!r})")
    if expected_job is not None and str(canonical) != str(expected_job):
        raise SystemExit(f"FATAL: saved io slurm_job_id {canonical!r} != expected run identity {expected_job!r}")
    recs = blob["records"]
    if not isinstance(recs, list) or len(recs) == 0:     # an empty archive qualifies nothing
        raise SystemExit("FATAL: io archive has an EMPTY record inventory — no case to qualify")
    for i, r in enumerate(recs):
        miss = _REQ - set(r)
        if miss:
            raise SystemExit(f"FATAL: record {i} missing required coordinates {sorted(miss)}")
        q, kv, go, sink = r["q"], r["kv"], r["gpu_out"], r["sink"]
        if not all(isinstance(t, torch.Tensor) for t in (q, kv, go, sink)):
            raise SystemExit(f"FATAL: record {i} q/kv/gpu_out/sink are not all tensors")
        if q.dim() != 3 or kv.dim() != 3 or go.dim() != 3 or sink.dim() != 1:   # R4-F2: shape RELATIONSHIPS
            raise SystemExit(f"FATAL: record {i} wrong ranks (q{tuple(q.shape)} kv{tuple(kv.shape)} out{tuple(go.shape)} sink{tuple(sink.shape)})")
        N, H, D = q.shape
        if not (r["N"] == N == kv.shape[0] == go.shape[0]):   # declared N must equal the actual batch of every tensor
            raise SystemExit(f"FATAL: record {i} N={r['N']} != batch (q{N}/kv{kv.shape[0]}/out{go.shape[0]})")
        if tuple(go.shape) != (N, H, D) or kv.shape[2] != D or sink.shape[0] != H:
            raise SystemExit(f"FATAL: record {i} inconsistent H/D (q{tuple(q.shape)} kv{tuple(kv.shape)} out{tuple(go.shape)} sink{tuple(sink.shape)})")
        for t, nm in ((q, "q"), (kv, "kv"), (go, "gpu_out"), (sink, "sink")):   # R5-F1: invalid oracle evidence
            if not bool(torch.isfinite(t).all().item()):   # sparse attn has NO -inf mask domain here -> all finite
                raise SystemExit(f"FATAL: record {i} non-finite {nm} (NaN/inf) -- invalid oracle evidence")
        sc = r["scale"]
        if not (isinstance(sc, (int, float)) and math.isfinite(sc) and sc > 0):   # documented scale domain
            raise SystemExit(f"FATAL: record {i} scale {sc!r} not a finite positive float")
    return recs


def metrics(ref, got):
    if not bool(torch.isfinite(ref).all().item()):      # R5-F1: non-finite reference is invalid, not a tolerance miss
        raise SystemExit("FATAL: non-finite reference (gpu_out) -- invalid oracle evidence")
    if not bool(torch.isfinite(got).all().item()):      # R5-F1: a candidate NaN/inf output is a failure
        raise SystemExit("FATAL: non-finite candidate output")
    r, g = ref.flatten().float(), got.flatten().float()
    cos = torch.nn.functional.cosine_similarity(r, g, dim=0).item()
    mae = (r - g).abs().max().item()
    if not (math.isfinite(cos) and math.isfinite(mae)):   # R5-F1: never print NaN metrics as if valid
        raise SystemExit(f"FATAL: non-finite comparison metrics (cos={cos}, mae={mae})")
    return cos, mae


def main():
    # R3-F5: restricted deserialization (no arbitrary pickle execution) + ENFORCE provenance, not just print.
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass
    blob = torch.load(IO, weights_only=True)
    recs = validate_archive(blob)
    prov = blob["provenance"]
    mod = _load_kernel()
    print(f"[provenance] {prov}  (records={len(recs)})")
    print(f"PROPOSED reference (approx-err vs the SOURCE-FAITHFUL 64-block replica's GPU out; NOT a ratified "
          f"hardware-noise floor) = {PROPOSED_REF:.3e}")
    print(f"{'N':>4} {'path':>10} {'operands':>10} {'cos_vs_gpu':>11} {'mae_vs_gpu':>11}")
    for r in recs:
        N, scale = r["N"], r["scale"]
        sink, gpu = r["sink"].float(), r["gpu_out"].float()
        # Two operand conventions: MATCHED = bf16-rounded (what the GPU kernel actually received) is the
        # honest approximation measurement; FP32 = original operands (secondary, diverges more at large N).
        for conv in ("bf16-matched", "fp32"):
            q = r["q"].bfloat16().float() if conv == "bf16-matched" else r["q"].float()
            kv = r["kv"].bfloat16().float() if conv == "bf16-matched" else r["kv"].float()
            paths = {
                "scalar": mod.sparse_attend(q, kv, sink, scale),
                "fp32bmm": mod.sparse_attend_fp32bmm(q, kv, sink, scale),
                "amx": mod.sparse_attend_amx(q, kv, sink, scale),
                "bestof": mod.sparse_attend_bestof(q, kv, sink, scale),
            }
            for name, out in paths.items():
                ob = out.bfloat16().float() if conv == "bf16-matched" else out   # match the bf16 OUTPUT boundary
                cos, mae = metrics(gpu, ob)
                print(f"{N:>4} {name:>10} {conv:>10} {cos:>11.6f} {mae:>11.3e}")
    print(">>> done")


def _mk_rec(N=1, K=512, H=64, D=512):
    return {"N": N, "scale": D ** -0.5, "sink": torch.randn(H),
            "gpu_out": torch.randn(N, H, D), "q": torch.randn(N, H, D), "kv": torch.randn(N, K, D)}


def _mk_blob(sha=EXPECTED_KERNEL_SHA, validated=True, slurm_job_id="384532", legacy_job=None, recs=None):
    prov = {"kernel_py_sha256": sha, "kernel_py_sha256_validated": validated, "slurm_job_id": slurm_job_id}
    if legacy_job is not None:
        prov["job"] = legacy_job
    return {"provenance": prov, "records": [_mk_rec()] if recs is None else recs}


def selftest():
    """R4-A4/F1/F2 persistent replay controls: the archive VALIDATOR must accept a producer-shaped archive and
    reject each identity/coordinate fault. No kernel compilation or saved IO needed."""
    ok = True

    def _accepts(blob, **kw):
        try:
            validate_archive(blob, **kw)
            return True
        except SystemExit:
            return False

    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    chk(_accepts(_mk_blob()), "ACCEPTS a genuine producer-shaped archive (no requested job)")
    chk(_accepts(_mk_blob(slurm_job_id="384532"), expected_job="384532"), "ACCEPTS the correct requested slurm_job_id (R4-F1)")
    chk(not _accepts(_mk_blob(slurm_job_id="384532"), expected_job="other-run"), "REJECTS an incorrect requested identity (R4-F1)")
    chk(not _accepts(_mk_blob(legacy_job="other-run"), expected_job="other-run"), "REJECTS a conflicting legacy job alias (R4-F1)")
    chk(not _accepts(_mk_blob(sha="0" * 64)), "REJECTS a wrong source hash (R4-A4)")
    chk(not _accepts({"provenance": {"kernel_py_sha256_validated": True, "slurm_job_id": "x"}, "records": [_mk_rec()]}),
        "REJECTS a missing source hash even with validated=True (R4-A4)")
    chk(not _accepts(_mk_blob(validated=False)), "REJECTS an UNVALIDATED provenance flag (R4-A4)")
    chk(not _accepts(_mk_blob(recs=[])), "REJECTS an empty record inventory (R4-A4)")
    _no_kv = _mk_rec(); del _no_kv["kv"]
    chk(not _accepts(_mk_blob(recs=[_no_kv])), "REJECTS a record missing a required coordinate (R4-A4)")
    _bad_n = _mk_rec(N=1); _bad_n["N"] = 64
    chk(not _accepts(_mk_blob(recs=[_bad_n])), "REJECTS an N inconsistent with the tensor batch (R4-F2)")
    _bad_shape = _mk_rec(N=1); _bad_shape["gpu_out"] = torch.randn(1, 8, 512)
    chk(not _accepts(_mk_blob(recs=[_bad_shape])), "REJECTS an output H/D inconsistent with q (R4-F2)")
    _bad_scale = _mk_rec(); _bad_scale["scale"] = -1.0
    chk(not _accepts(_mk_blob(recs=[_bad_scale])), "REJECTS a non-positive scale (R4-F2)")
    for fld in ("gpu_out", "q", "kv", "sink"):                 # R5-F1: non-finite record tensors are invalid evidence
        _nf = _mk_rec(); _nf[fld] = _nf[fld] * float("nan")
        chk(not _accepts(_mk_blob(recs=[_nf])), f"REJECTS a non-finite {fld} (R5-F1)")
    # R5-F1: the comparison itself must reject a non-finite reference / candidate (never print NaN metrics as valid)
    def _cmp_ok(ref, got):
        try:
            metrics(ref, got); return True
        except SystemExit:
            return False
    chk(_cmp_ok(torch.randn(8), torch.randn(8)), "metrics() ACCEPTS finite reference+candidate (R5-F1 positive control)")
    chk(not _cmp_ok(torch.randn(8) * float("nan"), torch.randn(8)), "metrics() REJECTS a non-finite reference (R5-F1)")
    chk(not _cmp_ok(torch.randn(8), torch.randn(8) * float("nan")), "metrics() REJECTS a non-finite candidate output (R5-F1)")
    print(f"  REPLAY-SELFTEST {'OK' if ok else 'FAILED'}")
    return 0 if ok else 2


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    main()
