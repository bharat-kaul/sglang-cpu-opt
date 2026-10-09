#!/usr/bin/env python3
"""DSv4-Flash roofline-VS-observation join (authored ops).

Each row's IDEAL floor (max(bytes/BW, flops/peak)) is derived from the op's ACTUAL benchmark
INPUT/OUTPUT CONTRACT (F4) at M=32, with an EXPLICIT compute dtype that selects the compute
resource (G1): bf16 -> AMX, fp32 -> AVX-512 FP32 ceiling. Each row links an AUDITABLE RESULT
RECORD (results/op_passes/*.json) whose kept pass gives a MICROBENCH cosine/set-match + kernel
revision. That numerical match is tied to the tested shape/dtype/reference/tolerance; it is NOT a
current-target certificate -> E2E verification PENDING. Absolute latency and speedup are WITHHELD/
UNVERIFIED. No causal/overhead/donor-routing conclusions are drawn here.

PROVENANCE (two DISTINCT identities, kept separate — R2-P1): (a) TIMING = the measured M-sweep, sourced
from results/perf_sweep.json (its own SLURM job/node/build, printed at runtime); (b) CORRECTNESS = each
op's op_passes/*.json record from a SEPARATE historical microbench run (its own kernel revision). The two
are NOT the same run and are never conflated into one identity. Contract byte/FLOP expectations are
asserted below (fail-closed).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_roofline_p2 import REF_REVISION as REV, load_record, _rev_resolved, _ATTRIB  # centralized ref + record reader

_SPEC = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
P = json.load(open(_SPEC))
MP = P["machine_peak"]
BW = MP["mem_bw_gbps"] * 1e9                 # nominal target
PEAK = MP["amx_bf16_tflops"] * 1e12          # AMX bf16 ceiling
FP32_PEAK = MP["avx512_fp32_tflops"] * 1e12  # AVX-512 FP32 ceiling
CPEAK = {"bf16": PEAK, "fp32": FP32_PEAK}    # G1: compute dtype -> compute resource
B2, B4, I4, I8 = 2.0, 4.0, 4.0, 8.0          # bf16 / fp32 / int32 / int64 bytes

M = 32  # the single observed coordinate

# op: (name, flops(M), bytes(M), compute_dtype, record_filename, plateau_note)
# bytes/flops derived from each bench's call contract; evidence READ from the result record (H1).
OPS = [
    ("indexer logits (q.ck+reduce)",
     lambda M: 2*M*64*1024*128 + 3*M*64*1024,
     lambda M: M*64*128*B4 + M*1024*128*B4 + M*64*B4 + M*1024*B4,   # q + kv + w + logits (FP32 public boundary)
     "bf16", "indexer_logits.json",
     "MIXED precision: FP32 public storage/traffic, but the matmul runs in BF16 (AMX) + FP32 reduce -> compute resource is BF16. BW time (bytes/BW ~50us @M=32) EXCEEDS the BF16 compute floor -> the op is BW-bound. OPEN (F7): useful throughput << reference BW -> ranked ROI hypotheses (conversion/pack/reduction split, larger-M attribution), NOT a proven plateau. (The former M=1 ~10x-vs-torch regression was FIXED by the single tiled contract: M=1 is now 0.100ms.)"),
    ("indexer top-k (512 of 1024)",
     lambda M: 0,
     lambda M: M*1024*B4 + M*512*I8,                              # read logits (fp32) + write 512 idx (INT64, kernel returns kLong)
     "fp32", "indexer_topk.json",
     "latency/selection-bound (no AMX GEMM primitive). OPEN (F7): at S/k=2 the chunk cap gives <=2 first-stage tasks at small M (NOT 64) -> small-S simplification is a ranked ROI hypothesis; off-roof diagnostic only."),
    ("compressor softmax-pool",
     lambda M: 3*M*128*512,
     lambda M: M*128*512*B4 + M*128*512*B4 + M*512*B4 + 128*512*B4,  # kv + score + out + shared APE
     "fp32", "compressor.json",
     "BW-bound streaming online-softmax pool (no w[N,R,D] temporary). OPEN (F7): latency is near-flat M=8..32 while bytes grow ~4x, N-only parallelism + 2 exp/channel/window -> channel-tiling / exp-throughput are ranked ROI hypotheses, NOT a proven DRAM wall. Perf measured at R=128/D=512 only; R=8 pools uncharacterized."),
    ("sparse attend (MQA+sink)",
     lambda M: 4*M*64*512*512,
     lambda M: M*512*512*B4 + 2*M*64*512*B4,                       # latent KV + q + out
     "fp32", "sparse_attend.json",
     "SURFACED candidate (not certified production): scalar MQA+sink; the scalar RETAINS a ~3.55x M=1 advantage over torch (F3). Donor dispatch is NOT yet proven (needs a concrete donor entry + sink/mask/2-source capability + dispatch evidence). AMX variant differs in BF16 rounding."),
    ("MHC sinkhorn (hc=4,20it)",
     lambda M: M*4*4*20*5,
     lambda M: M*24*B4 + 3*B4 + 24*B4 + M*24*B4,                   # mixes + scale + base + pre/post/comb
     "fp32", "mhc_sinkhorn.json",
     "dispatch-bound (reference ~40 tiny torch ops/call); fused 20 iters -> ~20x vs torch. Plateau = tiny hc=4 op, latency-bound; off-roof diagnostic only."),
    ("MHC combine (reduce)",
     lambda M: M*4096*(4+3),                                       # 4 mul + 3 add per out elem
     lambda M: M*4*4096*B4 + M*4*B4 + M*4096*B4,                   # x + per-request pre + y
     "fp32", "mhc_combine.json",
     "BW/latency-bound; tiled accumulate-once (x read once, y written once). Near roof at M>=8. This bounds USEFUL-byte traffic, NOT executed traffic or ROI -> 'minimum traffic' is a HYPOTHESIS until a discriminating measurement (launch grain / adjacent fusion) supports it."),
]

# --- fail-closed contract assertions (byte/FLOP + compute-dtype from the bench contracts @ M=32) ---
_EXPECT = {
    "indexer logits (q.ck+reduce)": {"bytes": 17_965_056, "cdt": "bf16"},   # F6: FP32 public STORAGE, BF16 matmul COMPUTE
    "indexer top-k (512 of 1024)": {"bytes": 262_144, "cdt": "fp32"},       # F6: INT64 output (kernel returns kLong)
    "compressor softmax-pool": {"bytes": 17_104_896, "cdt": "fp32"},
    "sparse attend (MQA+sink)": {"cdt": "fp32"},                            # G1: FP32 ceiling, not AMX
    "MHC sinkhorn (hc=4,20it)": {"bytes": 6_252, "cdt": "fp32"},
    "MHC combine (reduce)": {"bytes": 2_621_952, "flops": 917_504, "cdt": "fp32"},
}
for _n, _flf, _byf, _cdt, _rec, _nt in OPS:
    _e = _EXPECT.get(_n)
    if _e:
        if "bytes" in _e:
            assert _byf(M) == _e["bytes"], f"{_n}: bytes {_byf(M)} != contract {_e['bytes']}"
        if "flops" in _e:
            assert _flf(M) == _e["flops"], f"{_n}: flops {_flf(M)} != contract {_e['flops']}"
        assert _cdt == _e["cdt"], f"{_n}: compute dtype {_cdt} != {_e['cdt']}"

# measured M-sweep (sourced from a raw record); None if an op has no measurement yet
_PS = json.load(open(os.path.join(os.path.dirname(__file__), "results", "perf_sweep.json")))
MS = _PS["ms"]
_REPORT = os.path.join(os.path.dirname(__file__), "reports", "dsv4_roofline_vs_measured_emr.txt")


def render():
    """Build the full report text (deterministic) so it can be printed AND verified against the saved file."""
    out = []

    def p(s=""):
        out.append(s)

    p(f"DSv4 roofline-VS-measured (authored ops)  rev {REV[:8]}  M-sweep {MS}  nominal BW={BW/1e9:.0f} GB/s "
      f"AMX={PEAK/1e12:.0f} TF FP32={FP32_PEAK/1e12:.1f} TF")
    p(f"  measured: {_PS['raw_record']}")
    p("  NOTE (R2-P1/R3-P1): the TIMING identity is the SLURM job/node above; the per-op CORRECTNESS record is "
      "a SEPARATE historical run. The exact kernel/benchmark BUILD/SOURCE DIGEST of the timing job is "
      "UNRESOLVED (the launcher records node/time/config, not a source hash) — it is NOT retroactively assigned "
      "to any commit.")
    p(f"  ideal_us = max(bytes/BW, FLOPs/peak) at the row's compute dtype (nominal peak); off = measured/ideal "
      f"(vs NOMINAL \u2014 the node reaches ~60-77% of nominal DRAM BW, so a BW-bound op is ~1.3-1.7x off from the wall alone).")
    for name, flf, byf, cdt, record, plateau in OPS:
        peak = CPEAK[cdt]
        rec = load_record(record)                                     # H1: evidence READ from record (fail-closed)
        rev = rec["kernel_rev"] if _rev_resolved(rec["kernel_rev"]) else f"UNRESOLVED({rec['kernel_rev'] or 'none'})"
        meas = _PS["ops"].get(name, {}).get("median_ms")
        p("-" * 104)
        p(f"{name}  [compute dtype={cdt}]")
        p(f"  correctness record (HISTORICAL, separate microbench run): {record} @ kernel-rev {rev}")
        p(f"    recorded verbatim: {rec['correctness']}   [{_ATTRIB}]")
        p(f"  timing (CURRENT, from the perf_sweep raw record above; NOT the correctness run):")
        p(f"  {'M':>4} {'bind':>5} {'ideal_us':>10} {'measured_us':>12} {'off_ceiling':>12}")
        for i, Mv in enumerate(MS):
            fl, by = flf(Mv), byf(Mv)
            t_cc = fl / peak if fl else 0.0
            ideal = max(by / BW, t_cc)
            bind = "C" if t_cc > by / BW else "B"
            if meas:
                m_us = meas[i] * 1e3
                off = m_us / (ideal * 1e6) if ideal else float("inf")
                p(f"  {Mv:>4} {bind:>5} {ideal*1e6:>10.2f} {m_us:>12.1f} {off:>11.1f}x")
            else:
                p(f"  {Mv:>4} {bind:>5} {ideal*1e6:>10.2f} {'n/a':>12} {'n/a':>12}")
        p(f"  plateau: {plateau}")
    p("-" * 104)
    p("off_ceiling is a DIAGNOSTIC vs the NOMINAL roof (not an achievability claim, and NOT a proof of a DRAM\n"
      "wall). The measured useful-byte throughput (e.g. ~39 GB/s indexer, ~64 GB/s compressor @M=64) is far\n"
      "below the reference BW range, so saturation is NOT established; competing causes (N-only parallelism,\n"
      "exp throughput, cache residency, conversion/pack/GEMM/epilogue split, M=1 regressions) remain OPEN,\n"
      "ranked ROI hypotheses to discriminate with same-work best-path A/Bs before any plateau claim (F7).\n"
      "Correctness is the recorded field VERBATIM (microbench; E2E verification PENDING). Measured latency is\n"
      "sourced from the raw record above (median of 3, threads bound, one NUMA domain). No donor-dispatch claim.")
    return "\n".join(out) + "\n"


def main():
    import sys
    text = render()
    if len(sys.argv) == 3 and sys.argv[1] == "--verify":
        saved = open(sys.argv[2]).read() if os.path.exists(sys.argv[2]) else ""
        if saved != text:
            sys.stderr.write(f"STALE REPORT: {sys.argv[2]} does not match the current generator output. "
                             f"Regenerate: dsv4_roofline_vs_measured.py > {sys.argv[2]}\n")
            sys.exit(2)
        print(f"report up to date: {sys.argv[2]}")
        return
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
