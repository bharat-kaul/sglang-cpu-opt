#!/usr/bin/env python3
"""DSv4-Flash IDEAL roofline — REFERENCE-CONFORMANT (pinned revision), contract + reference tested.

Model derived from the PINNED DeepSeek-V4-Flash checkpoint config + inference/model.py graph +
checkpoint-header dtypes (HF revision in REF_REVISION). IDEAL target: t = max(B/BW_peak, F/P_peak)
with a DECLARED workload + fusion. The self-test asserts REFERENCE CONFORMANCE (layer counts,
attention positions, MHC dims, tensor-family dtypes) AND reporting contracts (single-point
rendering, pool score-stream accounting, precision->compute-resource selection, tracker validity).

DECLARED WORKLOAD: independent decode requests, batch M, no shared prefix; weights read once/step;
KV/activations per request; flash/fused => no intermediate-score DRAM. Decode boundary = 4096 tokens
available, full 128-token window. Main model = 43 blocks (MTP excluded).

ROW KINDS (F1): 'gemm' = analytical ideal (all M); 'measured' = a SINGLE observed coordinate
(renders only at its M, N/A elsewhere, carries a benchmark source); 'unmodeled' = declared not-
modeled (renders n/m at every M, surfaced in the open-items tracker). Observations are never
rendered as ideal targets across the batch sweep.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import tempfile

# F5: single centralized reference identifier (pinned HF revision, 40 hex chars).
REF_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

_DEF = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
_TRACKER = os.path.join(os.path.dirname(__file__), "results", "roofline_open_items.json")
_OPDIR = os.path.join(os.path.dirname(__file__), "results", "op_passes")
_ap = argparse.ArgumentParser()
_ap.add_argument("--platform", default=_DEF)
_ap.add_argument("--selftest", action="store_true")
_a, _ = _ap.parse_known_args()
PLAT = json.load(open(_a.platform))
MP = PLAT.get("machine_peak") or {}
NAME = PLAT["name"]
BW = MP.get("mem_bw_gbps", PLAT["mem_bw_gbps"]) * 1e9
PEAK = MP.get("amx_bf16_tflops", PLAT["amx_bf16_tflops"]) * 1e12          # AMX bf16 ceiling
FP32_PEAK = MP.get("avx512_fp32_tflops", PLAT.get("amx_bf16_tflops")) * 1e12  # AVX-512 FP32 ceiling
RIDGE = PEAK / BW
DOMAIN_RAM = PLAT["domain_ram_gb"] * 1e9

# ---- dtype bytes/weight incl quant scales (checkpoint-header dtypes, R3/R4) ----
FP8 = 1.0 + 1.0 / 16384     # fp8 + E8M0 scale per 128x128 block (MLA proj, SHARED experts)
FP4 = 0.5 + 1.0 / 32        # MXFP4 + 1B scale per 32 (ROUTED experts only)
BF16 = 2.0                  # checkpoint BF16 storage (wo_a, compressor proj, router, lm_head)
FP32 = 4.0                  # MHC hc_* params + compressor state are F32
AB = 2.0                    # bf16 activation
KV_BPW = 2.0                # bf16 latent KV
BPWS = {"fp8": FP8, "fp4": FP4, "bf16": BF16, "fp32": FP32}
# F3: precision selects the COMPUTE resource (AMX bf16 vs AVX-512 FP32), not just storage bytes.
COMPUTE_PEAK = {"fp8": PEAK, "fp4": PEAK, "bf16": PEAK, "fp32": FP32_PEAK}
Ms = [1, 8, 16, 32, 64]
NA = None  # sentinel: "not rendered at this coordinate"

# ---- pinned config (REF_REVISION) ----
H, NH, HD = 4096, 64, 512          # hidden, attention heads, head_dim
QLORA, OLORA, OG = 1024, 1024, 8
VOCAB, E, TOPK, MOE_I = 129280, 256, 6, 2048
IDX_NH, IDX_HD, IDX_TOPK = 64, 128, 512
WINDOW, S = 128, 4096              # sliding window; decode boundary tokens available
HC = 4                             # hc_mult
COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0]   # 44 entries; [:43]=main, [43]=MTP
MAIN = COMPRESS_RATIOS[:43]
L = len(MAIN)                      # 43 main blocks
_cnt = collections.Counter(MAIN)
N_SWA, N_IDX, N_128 = _cnt[0], _cnt[4], _cnt[128]   # 2, 21, 20
N_COMP = N_IDX + N_128             # 41 layers carry a (main) compressor
N_MOE, N_HASH = L, 3               # all 43 MoE; first 3 hash-routed


def _attn_positions(ratio):
    if ratio == 0:
        return WINDOW
    if ratio == 4:
        return WINDOW + min(IDX_TOPK, S // ratio)   # 128 + min(512, 1024) = 640
    return WINDOW + S // ratio                       # 128 + 4096/128 = 160


SUM_POS = sum(_attn_positions(r) for r in MAIN)      # 16896


def fmt(t):
    if t is NA:
        return "N/A"
    for u, s in (("s", 1), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        if t * s >= 1 or u == "ns":
            return f"{t*s:6.1f}{u}"


def ridge_crossing(fpt, fixed_bytes, bpt, ridge):
    denom = fpt - ridge * bpt
    return None if denom <= 0 else ridge * fixed_bytes / denom


class Op:
    def __init__(self, name, lane, impl, prec, layers, flop, byts, note,
                 crossing=None, kind="gemm", lat_s=None, m_obs=None, src="",
                 record="", commit="", certifies="",
                 weight_bytes_per_layer=0.0, unmodeled=""):
        self.name, self.lane, self.impl, self.prec, self.layers = name, lane, impl, prec, layers
        self.flop, self.byts, self.note = flop, byts, note
        self.crossing, self.kind, self.lat_s = crossing, kind, lat_s
        self.m_obs, self.src = m_obs, src
        self.record, self.commit, self.certifies = record, commit, certifies
        self.peak = COMPUTE_PEAK.get(prec, PEAK)
        self.ridge = self.peak / BW
        self.weight_bytes = weight_bytes_per_layer * layers
        self.unmodeled = unmodeled

    def row(self, M):
        if self.kind == "unmodeled":
            return NA, "n/m", NA                       # never a number
        if self.kind == "observed":                     # G2: absolute latency withheld (unverified)
            return NA, "obs", NA                        # see AUDITABLE OBSERVATIONS block
        fl, by = self.flop(M), self.byts(M)
        ai = fl / by if by else 0.0
        return ai, ("C" if ai > self.ridge else "B"), max(by / BW, fl / self.peak)



def wgemm(name, K, N, prec, layers, lane, impl, groups=1, act="bf16", unmodeled=""):
    bpw, a_bpw = BPWS[prec], BPWS[act]
    w_el, fpt = groups * K * N, 2 * groups * K * N
    fixed, bpt = w_el * bpw, groups * (K + N) * a_bpw
    peak = COMPUTE_PEAK.get(prec, PEAK)
    return Op(name, lane, impl, prec, layers,
              flop=lambda M: fpt * M * layers, byts=lambda M: (fixed + bpt * M) * layers,
              note="weight-streaming" + (f"; groups={groups}" if groups > 1 else "")
                   + (f"; act={act}" if act != "bf16" else ""),
              crossing=ridge_crossing(fpt, fixed, bpt, peak / BW),
              weight_bytes_per_layer=fixed, unmodeled=unmodeled)


def _rev_resolved(r):
    return bool(re.fullmatch(r"[0-9a-f]{7,40}", (r or "").strip()))


def _rejects_load(record):
    try:
        load_record(record)
        return False
    except Exception:  # noqa: BLE001
        return True


def load_record(record):
    """H1: read an EXTERNAL op-pass result record (results/op_passes/<record>) and DERIVE its evidence
    from the KEPT pass. Fail-closed on missing/malformed/no-kept-pass. Kernel revision and correctness
    are READ from the record (never hand-typed); the kept-pass speedup ratio is returned as the record
    states it. The revision is separate from any narrative; unresolved ('pending') => not a certificate."""
    with open(os.path.join(_OPDIR, record)) as f:      # FileNotFoundError if absent
        rec = json.load(f)                             # JSONDecodeError if malformed
    kept = [p for p in rec.get("passes", []) if p.get("kept")]
    if not kept:
        raise ValueError(f"op_passes/{record}: no kept pass")
    k = kept[-1]
    return {"record": f"results/op_passes/{record}",
            "kernel_rev": str(k.get("commit", "")).strip(),
            "correctness": str(k.get("correctness", "")).strip() or "unrecorded",
            "kept_ratios": k.get("vs_ref", {})}


# 6th review: a recorded cosine/set-match is reported LITERALLY under this fixed attribution \u2014 never a
# keyword-derived positive-match / certification claim (which would "certify" an absent or FAIL record).
_ATTRIB = "Historical author-reported microbench result; test scope UNVERIFIED; current-target/E2E verification PENDING"


def _evidence_from_record(rec):
    """Build observation evidence from a loaded record dict. Reports the recorded correctness field
    VERBATIM (or 'unrecorded') under a fixed UNVERIFIED attribution; infers NO metric/outcome from text."""
    resolved = _rev_resolved(rec["kernel_rev"])
    rev = rec["kernel_rev"] if resolved else f"UNRESOLVED({rec['kernel_rev'] or 'none'})"
    return {"record": rec.get("record", ""), "kernel_rev": rev, "resolved": resolved,
            "correctness": rec["correctness"], "attribution": _ATTRIB,
            "speedup": "author-reported; UNVERIFIED (structured ratio at superseded context; see record verdict)"}


def observation_evidence(op):
    """H1 / 6th review: evidence read from the record's kept pass. The recorded correctness field is
    reported LITERALLY (absent => 'unrecorded'; a reported FAIL is retained as-is) under a fixed
    UNVERIFIED attribution \u2014 NO positive-match/certification is inferred from free text, and no outcome is
    asserted. Kernel revision from the record (unresolved='pending'); speedup UNVERIFIED; latency withheld."""
    return _evidence_from_record(load_record(op.record))


def observed(name, lane, impl, layers, m_obs, record):
    """G2/H1: an authored-op observation whose evidence is READ + validated from an external result
    record (results/op_passes/<record>), never hand-typed. Absolute latency withheld; correctness +
    kernel revision derived from the kept pass; speedup rendered UNVERIFIED unless the revision resolves."""
    return Op(name, lane, impl, "obs", layers, flop=lambda M: 0, byts=lambda M: 0,
              note=f"observed @M={m_obs}; record=results/op_passes/{record}",
              kind="observed", m_obs=m_obs, record=record)


def unmodeled_op(name, lane, impl, layers, why):
    """F1: declared NOT modeled (no identifiable observation). Renders n/m; surfaced in tracker."""
    return Op(name, lane, impl, "n/m", layers, flop=lambda M: 0, byts=lambda M: 0,
              note="EXPLICITLY-UNMODELED: " + why, kind="unmodeled", unmodeled=why)


def pool(name, win, D, calls, lane):
    """F2: softmax-pool reads BOTH FP32 state streams (KV + score) + shared FP32 APE, writes FP32.
    win positions, D channels; `calls` = amortized boundary calls/step. State dtype = FP32 (reference)."""
    fpt = 3 * win * D                                   # score+exp+weighted-sum
    per_req = (2 * win * D + D) * FP32                  # read kv + read score + write out (per request)
    shared = win * D * FP32                             # positional APE read once per call (shared over M)
    return Op(name, lane, "NEW-C++", "fp32", calls,
              flop=lambda M: fpt * M * calls, byts=lambda M: (per_req * M + shared) * calls,
              note=f"win={win} D={D} FP32 state (kv+score read, out write)+APE; amortized {calls:.3f} calls/step",
              crossing=None)


def attn_sparse():
    fpt_tot = 4 * NH * SUM_POS * HD
    kv_bytes = lambda M: M * SUM_POS * HD * KV_BPW
    qo_bytes = lambda M: L * 2 * M * NH * HD * AB
    return Op("MLA sparse attention (window+compressed, 43L)", "A", "donor MLA flash", "fp8", 1,
              flop=lambda M: fpt_tot * M, byts=lambda M: kv_bytes(M) + qo_bytes(M),
              note=f"single sparse op; positions/req summed={SUM_POS}; MQA flash; no score DRAM; AI M-invariant",
              crossing=None)


def indexer_fused():
    """Fused indexer over the COMPRESSED index context. Query is an ALREADY-PROJECTED bf16 activation
    (F4: FP4 in-place simulation does not establish packed query storage). Head-weights computed per
    request (M-scaled). fp32 logits out. The 1024->8192 projection matrix is a SEPARATE wgemm row."""
    idx_ctx = S // 4
    key_bpw, q_bpw = KV_BPW, BF16            # index-kv bf16; projected query bf16 activation
    matmul_fpt, vec_fpt = 2 * IDX_NH * idx_ctx * IDX_HD, 3 * IDX_NH * idx_ctx
    fpt = matmul_fpt + vec_fpt
    bpt = idx_ctx * IDX_HD * key_bpw + IDX_NH * IDX_HD * q_bpw + IDX_NH * 4 + idx_ctx * 4
    return Op("DSA indexer logits (GEMM+reduce FUSED)", "B", "NEW-C++", "bf16", N_IDX,
              flop=lambda M: fpt * M * N_IDX, byts=lambda M: bpt * M * N_IDX,
              note=f"fused; index ctx={idx_ctx}; bf16 projected query; per-request head-weights; no score DRAM",
              crossing=None)


def moe_experts():
    per, distinct = 3 * H * MOE_I, lambda M: E * (1 - (1 - TOPK / E) ** M)
    return Op("MoE routed experts (MXFP4, 3-matrix)", "A", "donor moe.cpp", "fp4", N_MOE,
              flop=lambda M: 2 * M * TOPK * per * N_MOE,
              byts=lambda M: (distinct(M) * per * FP4 + M * TOPK * (2 * H + MOE_I) * AB) * N_MOE,
              note="distinct experts=E*(1-(1-TOPK/E)^M); routed=MXFP4 0.53125",
              crossing=None, weight_bytes_per_layer=E * per * FP4)


def shared_expert():
    per = 3 * H * MOE_I
    fpt, fixed, bpt = 6 * H * MOE_I, per * FP8, 2 * H * AB
    return Op("shared-expert (3-matrix, FP8)", "A", "donor moe.cpp", "fp8", N_MOE,
              flop=lambda M: fpt * M * N_MOE, byts=lambda M: (fixed + bpt * M) * N_MOE,
              note="FP8 per checkpoint header (routed are FP4)",
              crossing=ridge_crossing(fpt, fixed, bpt, RIDGE), weight_bytes_per_layer=fixed)


def hc_fn():
    """MHC pre-mix projection: hc_mult*hidden -> (2+hc)*hc, FP32 params + FP32 operand (hc_pre casts
    the flattened input to FP32). Compute runs on the AVX-512 FP32 resource (F3). 2x/layer."""
    K, N = HC * H, (2 + HC) * HC             # 16384 -> 24
    fpt, fixed, bpt = 2 * K * N, K * N * FP32, (K + N) * FP32
    return Op("MHC hc_fn (16384->24, FP32)", "C", "NEW-C++", "fp32", 2 * L,
              flop=lambda M: fpt * M * 2 * L, byts=lambda M: (fixed + bpt * M) * 2 * L,
              note="K=hc_mult*H=16384; FP32 params + FP32 operand; FP32 compute resource",
              crossing=ridge_crossing(fpt, fixed, bpt, FP32_PEAK / BW), weight_bytes_per_layer=fixed)


def hc_post():
    fpt, bpt = 2 * HC * HC * H + HC * H, (2 * HC * H + HC * HC + H) * FP32
    return Op("MHC hc_post (post*x + comb@residual)", "C", "NEW-C++", "fp32", 2 * L,
              flop=lambda M: fpt * M * 2 * L, byts=lambda M: bpt * M * 2 * L,
              note="comb@residual bmm + elementwise; FP32 compute", crossing=None,
              unmodeled="hc_post norm/affine/sigmoid vector work excluded from this ideal target")


def hc_head():
    w_el, fpt = HC * HC * H, 2 * HC * HC * H + HC * H
    fixed, bpt = w_el * FP32, 2 * HC * H * FP32
    return Op("MHC hc_head (mixer, once, FP32)", "C", "NEW-C++", "fp32", 1,
              flop=lambda M: fpt * M, byts=lambda M: fixed + bpt * M,
              note="projection + weighted-sum; FP32 compute",
              crossing=ridge_crossing(fpt, fixed, bpt, FP32_PEAK / BW), weight_bytes_per_layer=fixed,
              unmodeled="hc_head RMSNorm + sigmoid gating + reduction vector work excluded from this ideal target")


# ================= ONE tensor inventory (OPS + capacity) =================
OPS = [
    wgemm("MLA wqkv_a (fused q_a+kv_a, 4096->1536)", H, QLORA + HD, "fp8", L, "A", "donor dsv2"),
    unmodeled_op("  q_norm RMSNorm(1024)", "A", "donor norm.cpp", L, "no identifiable measured coordinate/source"),
    wgemm("MLA wq_b (1024->32768)", QLORA, NH * HD, "fp8", L, "A", "donor dsv2"),
    unmodeled_op("  RoPE (yarn)", "A", "donor rope", L, "no identifiable measured coordinate/source"),
    unmodeled_op("  kv_norm RMSNorm(512)", "A", "donor norm.cpp", L, "no identifiable measured coordinate/source"),
    attn_sparse(),
    wgemm("MLA wo_a (grouped 8x 4096->1024)", NH * HD // OG, OLORA, "bf16", L, "A", "donor dsv2 BF16", groups=OG),
    wgemm("MLA wo_b (8192->4096)", OG * OLORA, H, "fp8", L, "A", "donor dsv2"),
    # --- DSA indexer (21 ratio-4 layers) ---
    wgemm("DSA indexer wq_b (1024->8192)", QLORA, IDX_NH * IDX_HD, "fp8", N_IDX, "A", "donor dsv2"),
    wgemm("DSA indexer weights_proj (4096->64)", H, IDX_NH, "bf16", N_IDX, "A", "donor gemm"),
    indexer_fused(),
    observed("DSA indexer topk-512 (over 1024)", "B", "NEW-C++", N_IDX, 32, "indexer_topk.json"),
    # --- compressors: projections (every token, FP32 state out) + pooling (boundary-amortized) (R2/F2) ---
    wgemm("main compressor wkv+wgate (r4, 4096->2048)", H, 2 * 2 * HD, "bf16", N_IDX, "A", "donor dsv2", act="fp32"),
    wgemm("main compressor wkv+wgate (r128, 4096->1024)", H, 2 * HD, "bf16", N_128, "A", "donor dsv2", act="fp32"),
    wgemm("indexer compressor wkv+wgate (r4, 4096->512)", H, 2 * 2 * IDX_HD, "bf16", N_IDX, "A", "donor dsv2", act="fp32"),
    pool("main pool (r4 overlap, win=8 D=512)", 8, HD, N_IDX / 4, "B"),
    pool("main pool (r128, win=128 D=512)", 128, HD, N_128 / 128, "B"),
    pool("indexer pool (r4 overlap, win=8 D=128)", 8, IDX_HD, N_IDX / 4, "B"),
    # --- MHC (every layer x2 pre + post; head once) ---
    hc_fn(),
    observed("MHC sinkhorn (hc=4, 20 iters)", "C", "NEW-C++", 2 * L, 32, "mhc_sinkhorn.json"),
    observed("MHC combine (hc_pre reduce)", "C", "NEW-C++", 2 * L, 32, "mhc_combine.json"),
    hc_post(),
    hc_head(),
    # --- MoE ---
    wgemm("MoE router gate (4096->256)", H, E, "bf16", N_MOE, "A", "donor gemm"),
    unmodeled_op("MoE hash route (3 layers, tid2eid gather)", "A", "donor", N_HASH,
                 "no identifiable measured coordinate/source"),
    moe_experts(),
    shared_expert(),
    # --- embedding / head ---
    unmodeled_op("embed (VocabParallel lookup)", "A", "donor embed", 1, "no identifiable measured coordinate/source"),
    wgemm("lm_head (4096->129280)", H, VOCAB, "bf16", 1, "A", "donor gemm"),
]


def capacity():
    return {op.name: op.weight_bytes / 1e9 for op in OPS if op.weight_bytes}


# ---- G3: STRICT tracker gate (fail-closed, schema + coverage, real validator) ----
_VALID_DISPOSITIONS = ("MODELED", "MEASURED", "EXPLICITLY-UNMODELED")
# Known coverage categories that MUST remain disposed (cannot silently disappear between revisions).
_REQUIRED_COVERAGE = (
    "KV / state / workspace",
    "masked shared-pool fallback",
    "hc_post/hc_head vector work",
    "indexer compressor",
    "compressor per-token state write",
)


def validate_tracker(trk):
    """The REAL validator (exercised by the self-test with invalid inputs). Raises ValueError on any
    schema/coverage violation. Disposition is a BOUNDED ENUM (exact match); qualifiers live in a
    separate field; identity + disposition-specific evidence are required; coverage must be preserved."""
    if not isinstance(trk, dict) or not isinstance(trk.get("items"), list):
        raise ValueError("tracker: missing 'items' list")
    if REF_REVISION not in trk.get("reference_revision", ""):
        raise ValueError(f"tracker: reference_revision must contain {REF_REVISION}")
    seen = set()
    for it in trk["items"]:
        if not isinstance(it, dict):
            raise ValueError("tracker: item is not an object")
        name = it.get("item")
        if not name or not isinstance(name, str):
            raise ValueError(f"tracker: item missing non-empty 'item' identity: {it!r}")
        if name in seen:
            raise ValueError(f"tracker: duplicate item identity {name!r}")
        seen.add(name)
        d = it.get("disposition")
        if d not in _VALID_DISPOSITIONS:                 # EXACT enum, not startswith
            raise ValueError(f"tracker: invalid disposition {d!r} for item {name!r}")
        if d in ("MODELED", "MEASURED") and not it.get("source"):
            raise ValueError(f"tracker: {d} item {name!r} needs a non-empty 'source'")
        if d == "EXPLICITLY-UNMODELED" and not (it.get("reason") and it.get("plan")):
            raise ValueError(f"tracker: EXPLICITLY-UNMODELED item {name!r} needs 'reason' and 'plan'")
    for cat in _REQUIRED_COVERAGE:                       # known exclusions cannot silently disappear
        if not any(cat in it.get("item", "") for it in trk["items"]):
            raise ValueError(f"tracker: required coverage category missing: {cat!r}")
    return trk


def load_tracker(path=_TRACKER):
    """Load + VALIDATE the open-items tracker. Raises on missing/malformed/invalid (fail-closed)."""
    with open(path) as f:                               # raises FileNotFoundError if absent
        trk = json.load(f)                              # raises JSONDecodeError if malformed
    return validate_tracker(trk)


def unmodeled_items(trk):
    live = [(op.name.strip(), op.unmodeled) for op in OPS if op.unmodeled]
    live += [(it["item"], it.get("reason", ""))
             for it in trk["items"] if it.get("disposition") == "EXPLICITLY-UNMODELED"]
    return live


def observations():
    return [op for op in OPS if op.kind == "observed"]


def selftest():
    """REFERENCE-CONFORMANCE + reporting-contract checks (R7/F1-F5). Gates report emission."""
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and c
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    # --- reference conformance ---
    chk(dict(collections.Counter(MAIN)) == {0: 2, 4: 21, 128: 20}, "main layer counts == {0:2,4:21,128:20}")
    chk(SUM_POS == 16896, f"attention positions/request summed == 16896 (got {SUM_POS})")
    chk(abs(attn_sparse().flop(1) - 2_214_592_512) < 1, "attention FLOPs@M=1 == 2,214,592,512")
    chk(abs(hc_fn().flop(1) - 67_633_152) < 1, "MHC hc_fn FLOPs@M=1 == 67,633,152 (K=16384)")
    chk(shared_expert().prec == "fp8", "shared expert dtype == FP8 (checkpoint header)")
    chk(moe_experts().prec == "fp4", "routed experts dtype == FP4 (MXFP4)")
    chk(hc_fn().prec == "fp32", "MHC hc_fn dtype == FP32 (checkpoint header)")
    a = attn_sparse()
    chk(abs(a.byts(8) - 8 * a.byts(1)) < 1, "attention bytes linear in M")
    g = wgemm("t", 4096, 64, "bf16", 1, "A", "x")
    chk(g.crossing is None, "narrow GEMM (4096->64) has NO ridge crossing")
    chk(abs(E * (1 - (1 - TOPK / E) ** 1) - TOPK) < 1e-9, "distinct experts@M=1 == TOPK")
    main_comp_gb = sum(capacity()[n] for n in capacity() if n.startswith("main compressor"))
    chk(abs(main_comp_gb - 0.520093696) < 1e-3, f"main compressor matrices == 0.520 GB (got {main_comp_gb:.4f})")
    idx_comp_gb = sum(capacity()[n] for n in capacity() if n.startswith("indexer compressor"))
    chk(abs(idx_comp_gb - 0.088080384) < 1e-3, f"indexer compressor matrices == 0.088 GB (got {idx_comp_gb:.4f})")
    shared_gb = capacity().get("shared-expert (3-matrix, FP8)", 0)
    chk(abs(shared_gb - 1.082196480) < 2e-3, f"shared-expert FP8 capacity == 1.082 GB (got {shared_gb:.4f})")

    # --- G2/H1: observed rows withhold absolutes; evidence is READ + validated from the record ---
    obs = [o for o in OPS if o.kind == "observed"]
    chk(all(o.row(M)[2] is NA for o in obs for M in Ms), "observed rows withhold absolute latency at EVERY M")
    # every observed record must actually load (fail-closed) and yield derived evidence
    ev = {}
    try:
        ev = {o.name: observation_evidence(o) for o in obs}
        chk(True, "every observed record LOADS + yields a kept pass (fail-closed)")
    except Exception as e:  # noqa: BLE001
        chk(False, f"observation record load FAILED: {e}")
    # H1: top-k revision is DERIVED from the record (not the discarded 06faec0); mismatch is impossible
    tk = next(o for o in obs if o.name.startswith("DSA indexer topk"))
    tk_ev = ev.get(tk.name, {})
    chk("06faec0" not in tk_ev.get("kernel_rev", ""),
        "top-k observation does NOT publish the discarded 06faec0 revision (H1)")
    # H1 / 6th review: evidence reports the recorded field LITERALLY under a fixed UNVERIFIED attribution,
    # and NEVER infers a positive-match/certification from free text (would "certify" an absent/FAIL record)
    chk(all(e["attribution"] == _ATTRIB and "certif" not in e["attribution"].lower() for e in ev.values())
        and bool(ev), "correctness carries the fixed UNVERIFIED/E2E-PENDING attribution (no certification claim)")
    chk(tk_ev.get("correctness") == "set-match 1.0 all M",
        "top-k correctness is the recorded field VERBATIM (not a derived label)")
    # regression: missing / explicit-failure / failed-cosine evidence must NEVER yield a certification claim
    for _corr in (None, "FAIL: numerical mismatch", "cos 0.0; FAIL"):
        _p = {"kept": True, "commit": "pending"}
        if _corr is not None:
            _p["correctness"] = _corr
        _tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"passes": [_p]}, _tf); _tf.close()
        _e = _evidence_from_record(load_record(os.path.abspath(_tf.name)))
        os.unlink(_tf.name)
        _blob = (_e["correctness"] + " " + _e["attribution"]).lower()
        chk("certif" not in _blob and _e["resolved"] is False
            and _e["correctness"] == (_corr if _corr else "unrecorded"),
            f"evidence makes NO certification claim for correctness={_corr!r} (literal field, UNVERIFIED)")
    # H1: speedup is never presented as a certificate (records' ratios are at a superseded context)
    chk(all("UNVERIFIED" in e["speedup"] for e in ev.values()) and bool(ev),
        "speedup is author-reported/UNVERIFIED (never certified)")
    # H1 fail-closed: a missing record raises (records are actually opened)
    chk(_rejects_load("does_not_exist.json"), "missing result record FAILS (record is opened, not assumed)")
    um = next(o for o in OPS if o.kind == "unmodeled")
    chk(all(um.row(M)[2] is NA for M in Ms), "unmodeled row renders a NUMBER at NO M")

    # --- F2: pool reads BOTH FP32 state streams + APE ---
    p = next(o for o in OPS if o.name.startswith("main pool (r128"))
    exp = ((2 * 128 * HD + HD) * FP32 * 1 + 128 * HD * FP32) * (N_128 / 128)   # per_req*M=1 + shared
    chk(abs(p.byts(1) - exp) < 1, "pool bytes include kv+score (2x FP32 win*D) + out + APE")
    chk(p.prec == "fp32", "pool state dtype == FP32 (reference compressor state)")

    # --- F3: precision selects the compute resource ---
    chk(COMPUTE_PEAK["fp32"] != COMPUTE_PEAK["bf16"], "FP32 and BF16 use DIFFERENT compute peaks")
    cb = wgemm("cb", 8192, 8192, "bf16", 1, "A", "x").row(4096)[2]
    cf = wgemm("cf", 8192, 8192, "fp32", 1, "A", "x", act="fp32").row(4096)[2]
    chk(cf > cb * 2, "compute-bound FP32 GEMM is slower than BF16 (FP32 peak < AMX peak)")

    # --- G3: strict tracker validation exercised on the REAL loader/validator ---
    try:
        trk = load_tracker()
        chk(True, "tracker loads + validates (enum, identity, per-disposition fields, coverage)")
    except Exception as e:  # noqa: BLE001
        trk = None
        chk(False, f"tracker validation FAILED: {e}")

    def _rejects(bad):
        try:
            validate_tracker(bad)
            return False
        except ValueError:
            return True

    base = [dict(it) for it in (trk["items"] if trk else [])]
    chk(_rejects({"reference_revision": REF_REVISION,
                  "items": base + [{"item": "probe", "disposition": "MODELED_BOGUS", "source": "s"}]}),
        "tracker REJECTS a non-enum disposition (MODELED_BOGUS)")
    chk(_rejects({"reference_revision": REF_REVISION, "items": [{"disposition": "MODELED"}]}),
        "tracker REJECTS an item with no identity/source")
    chk(_rejects({"reference_revision": REF_REVISION,
                  "items": [it for it in base if it.get("disposition") != "EXPLICITLY-UNMODELED"]}),
        "tracker REJECTS dropping all EXPLICITLY-UNMODELED coverage")
    chk(base != [] and _rejects({"reference_revision": REF_REVISION, "items": base + [dict(base[0])]}),
        "tracker REJECTS a duplicate item identity")

    print(f"  SELFTEST {'OK' if ok else 'FAILED'}")
    return ok


def p0(trk):
    cap = capacity()
    tot = sum(cap.values())
    print("=" * 96)
    print(f"P0 CAPACITY (one inventory)  platform={NAME}  nominal BW={BW/1e9:.1f} GB/s  "
          f"AMX-bf16={PEAK/1e12:.1f} TF  AVX512-fp32={FP32_PEAK/1e12:.2f} TF  ridge(bf16)={RIDGE:.0f}")
    print(f"  measured reference (node {PLAT.get('measurement_node','?')}): BW={PLAT['mem_bw_gbps']:.0f} GB/s  "
          f"AMX={PLAT['amx_bf16_tflops']:.0f} TF  (reference observation; nominal target != measured)")
    print("=" * 96)
    for n, g in sorted(cap.items(), key=lambda x: -x[1]):
        print(f"  {n[:48]:48s} {g:8.3f} GB")
    print(f"  {'RESIDENT (listed matrices only)':48s} {tot:8.3f} GB  -> excludes KV/state/scales/workspace/embed; "
          f"feasibility only, NOT runtime fit or TP-optimality")
    print("\n  EXPLICITLY-UNMODELED quantities (surfaced; excluded from every total/ranking):")
    for name, why in unmodeled_items(trk):
        print(f"    - {name}: {why[:92]}")


def phaseA():
    print("\n" + "=" * 128)
    print("PHASE A — IDEAL per-op roofline (reference-conformant). 'obs'=authored-op observation "
          "(absolute latency WITHHELD; see block below); 'n/m'=not modeled.")
    print("=" * 128)
    print(f"{'op':46s} {'lane/impl':22s} | " + " ".join(f"{'M='+str(m):>9s}" for m in Ms) + "  calls kind cross")
    print("-" * 128)
    for op in OPS:
        cells, kinds = [], []
        for M in Ms:
            _, k, t = op.row(M)
            kinds.append(k)
            cells.append(f"{fmt(t):>9s}")
        xs = {"observed": "obs", "unmodeled": "n/m"}.get(op.kind,
              "none" if op.crossing is None else f"M{op.crossing:.0f}")
        cs = f"{op.layers:.2f}" if isinstance(op.layers, float) else str(op.layers)
        print(f"{op.name[:46]:46s} {(op.lane+' '+op.impl)[:22]:22s} | " + " ".join(cells)
              + f"  {cs:>5s} {kinds[-1]:>4s} {xs}")
    print("\nGEMM/pool/attn times are IDEAL targets (all M). FP32 rows use the AVX-512 FP32 compute ceiling.\n"
          "'obs' rows withhold an absolute node latency (no auditable raw record); they are listed with their\n"
          "result record below. 'n/m' rows are declared not-modeled. Distance-from-roof is diagnostic only.")
    print("\n" + "-" * 128)
    print("AUDITABLE OBSERVATIONS (authored ops) — recorded field reported LITERALLY (absolute latency withheld):")
    for op in observations():
        e = observation_evidence(op)
        print(f"  {op.name[:44]:44s}  @M={op.m_obs}  record={e['record']}  kernel-rev={e['kernel_rev']}")
        print(f"      correctness (recorded, verbatim): {e['correctness']}")
        print(f"      {e['attribution']}")
        print(f"      speedup: {e['speedup']}")
    print("\n  kernel-rev + correctness are READ from the record's kept pass (not hand-typed). The correctness\n"
          "  field is reported VERBATIM under the attribution above \u2014 no positive match or certification is\n"
          "  inferred from it; an absent result is 'unrecorded', a reported failure is retained as-is.")


if __name__ == "__main__":
    print(f"REFERENCE-CONFORMANCE + CONTRACT SELFTEST (ref {REF_REVISION[:8]}):")
    passed = selftest()
    if _a.selftest:
        sys.exit(0 if passed else 1)
    if not passed:
        print("\n*** reference/contract checks FAILED \u2014 not emitting report ***")
        sys.exit(1)
    try:
        trk = load_tracker()
    except Exception as e:  # noqa: BLE001
        print(f"\n*** tracker gate FAILED ({e}) \u2014 not emitting report ***")
        sys.exit(1)
    print()
    p0(trk)
    phaseA()
