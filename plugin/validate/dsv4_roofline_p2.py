#!/usr/bin/env python3
"""DSv4-Flash Phase-A IDEAL roofline — one tensor inventory, invariant-tested.

Addresses the ten corrections in reports/dsv4_roofline_emr_verdict.md.
IDEAL target (optimization goal, NOT an achievability prediction):
    t_ideal = max(B / BW_peak, F / P_peak)
with the DECLARED workload + fusion assumptions below. p0() capacity and per-op
costs are derived from the SAME tensor inventory. Invariants are asserted before
the report is produced (run with --selftest to see them).

DECLARED WORKLOAD ASSUMPTIONS (change these, not the accounting):
  * Independent decode requests, batch M; NO shared prefix (each request owns its KV).
  * Flash/fused attention + fused indexer => intermediate scores are NOT materialized to DRAM.
  * Weights are streamed once per step and reused across the M tokens (weight bytes are M-independent).
  * KV / indexer-KV are per-request state (scale with M).
  * dtypes: fp8 proj (1 B + negligible 1/16384 block scale); MXFP4 experts (0.53125 B incl 1 B scale/32);
    bf16 activations & bf16 latent KV (2 B). Low-bit storage != native low-bit AMX (compute ref is bf16).
Per-layer invocation counts from config compress_ratios=[0,0,4,128,...,4,0] (43 layers):
  ratio-4 (compressor+indexer): 20 ; ratio-128 (compressor only): 20 ; SWA ratio-0: 3 ;
  MoE: all 43 (hash-routed first 3, score-routed 40) ; MHC hyper-connections: every layer + head.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_DEF = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
_ap = argparse.ArgumentParser()
_ap.add_argument("--platform", default=_DEF)
_ap.add_argument("--selftest", action="store_true")
_a, _ = _ap.parse_known_args()
with open(_a.platform) as f:
    PLAT = json.load(f)
MP = PLAT.get("machine_peak") or {}
NAME = PLAT["name"]
BW = MP.get("mem_bw_gbps", PLAT["mem_bw_gbps"]) * 1e9          # nominal target BW
PEAK = MP.get("amx_bf16_tflops", PLAT["amx_bf16_tflops"]) * 1e12  # nominal target compute
BW_MEAS = PLAT["mem_bw_gbps"] * 1e9                            # measured reference (same node as spec)
PEAK_MEAS = PLAT["amx_bf16_tflops"] * 1e12
RIDGE = PEAK / BW
DOMAIN_RAM = PLAT["domain_ram_gb"] * 1e9

# ---- dtype bytes/weight (correction 9: include quant scales) ----
FP8 = 1.0 + 1.0 / 16384      # fp8 + 1B ue8m0 scale per 128x128 block ~= 1.00006
FP4 = 0.5 + 1.0 / 32         # MXFP4 + 1B scale per 32 weights = 0.53125
BF16 = 2.0
AB = 2.0                     # bf16 activation
KV_BPW = 2.0                 # bf16 latent KV (published: "current implementation uses bf16")
Ms = [1, 8, 16, 32, 64]

# ---- DSv4-Flash config (grounded in published model.py + config.json) ----
H, L, NH, HD = 4096, 43, 64, 512          # hidden, layers, heads, head_dim
QLORA, OLORA, OG = 1024, 1024, 8          # q/o lora ranks, o_groups
VOCAB, E, TOPK, MOE_I = 129280, 256, 6, 2048
IDX_NH, IDX_HD, IDX_TOPK = 64, 128, 512
S = 4096                                   # raw decode context
RATIO4 = 4
IDX_CTX = S // RATIO4                       # ~1024 compressed index positions (correction 6)
# invocation counts from compress_ratios (correction 6/10)
N_IDX = 20       # ratio-4 layers carry the indexer
N_COMP = 40      # ratio-4 + ratio-128 layers carry a compressor
N_SWA = 3        # ratio-0 pure sliding-window layers
N_MOE = 43       # all layers are MoE
N_HASH = 3       # first 3 MoE layers are hash-routed
N_MHC = 43       # hyper-connections every layer (hc_pre/hc_post x2) + 1 head


def fmt(t):
    for u, s in (("s", 1), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        if t * s >= 1 or u == "ns":
            return f"{t*s:6.1f}{u}"


def ridge_crossing(fpt, fixed_bytes, bpt):
    """M where AI crosses RIDGE under the REAL traffic model (correction 4).
    fpt=FLOPs/token, fixed_bytes=M-independent bytes, bpt=bytes/token. None if it never crosses."""
    denom = fpt - RIDGE * bpt
    return None if denom <= 0 else RIDGE * fixed_bytes / denom


class Op:
    def __init__(self, name, lane, impl, prec, layers, flop, byts, note,
                 crossing=None, kind="gemm", lat_s=None, weight_bytes_per_layer=0.0):
        self.name, self.lane, self.impl, self.prec, self.layers = name, lane, impl, prec, layers
        self.flop, self.byts, self.note = flop, byts, note
        self.crossing, self.kind, self.lat_s = crossing, kind, lat_s
        self.weight_bytes = weight_bytes_per_layer * layers   # for one-inventory capacity

    def row(self, M):
        if self.kind == "latency":
            return 0.0, "lat", self.lat_s
        fl, by = self.flop(M), self.byts(M)
        ai = fl / by if by else 0.0
        t = max(by / BW, fl / PEAK)
        return ai, ("C" if ai > RIDGE else "B"), t


def wgemm(name, K, N, prec, layers, lane, impl, groups=1):
    """Weight-streaming GEMM. weights M-independent (shared across batch); activations per token.
    grouped (correction 2): weight=groups*K*N, activation=M*groups*(K+N), flops=2*M*groups*K*N."""
    bpw = {"fp8": FP8, "fp4": FP4, "bf16": BF16}[prec]
    w_el = groups * K * N
    fpt = 2 * groups * K * N
    fixed = w_el * bpw
    bpt = groups * (K + N) * AB
    return Op(name, lane, impl, prec, layers,
              flop=lambda M: fpt * M * layers,
              byts=lambda M: (fixed + bpt * M) * layers,
              note="weight-streaming; " + (f"groups={groups}; " if groups > 1 else ""),
              crossing=ridge_crossing(fpt, fixed, bpt),
              weight_bytes_per_layer=fixed)


def measured(name, lane, impl, lat_s, layers, note):
    """Latency-bound op modeled by a MEASURED pilot microbench time (correction 10) — NOT an
    invented floor and NOT a theoretical bound. lat_s = per-call seconds at the benched shape."""
    return Op(name, lane, impl, "meas", layers, flop=lambda M: 0, byts=lambda M: 0,
              note="MEASURED pilot microbench (not a theoretical floor): " + note,
              kind="latency", lat_s=lat_s * layers)


def attn(name, Sctx, layers, lane, impl):
    """MLA MQA attention (1 latent KV shared across heads), flash-fused (NO score DRAM),
    INDEPENDENT requests (KV read per request => scales with M) (correction 5).
    AI is M-invariant here (both FLOPs and bytes scale with M) => no ridge crossing."""
    fpt = 4 * NH * Sctx * HD                       # q.k + w.v
    bpt = Sctx * HD * KV_BPW + 2 * NH * HD * AB     # KV(per req) + q-in + out ; no score
    return Op(name, lane, impl, "fp8", layers,
              flop=lambda M: fpt * M * layers,
              byts=lambda M: bpt * M * layers,
              note="MQA flash-fused; KV per-request (xM); no score DRAM; AI M-invariant",
              crossing=None)


def indexer_fused():
    """Fused indexer: GEMM(q.ck) -> relu*weight*sum reduction, over COMPRESSED context (correction 6/7).
    Per request: keys IDX_CTX*IDX_HD, query IDX_NH*IDX_HD, fp32 head-weights IDX_NH, fp32 out IDX_CTX.
    No intermediate score DRAM (fused). Vector epilogue tracked in FLOPs but is AMX-light."""
    key_bpw, query_bpw = KV_BPW, FP4   # index-kv bf16; query fp4-sim (published)
    matmul_fpt = 2 * IDX_NH * IDX_CTX * IDX_HD
    vector_fpt = 3 * IDX_NH * IDX_CTX
    fpt = matmul_fpt + vector_fpt
    bpt = IDX_CTX * IDX_HD * key_bpw + IDX_NH * IDX_HD * query_bpw + IDX_NH * 4 + IDX_CTX * 4
    return Op("DSA indexer logits (GEMM+reduce FUSED)", "B", "NEW-C++", "bf16", N_IDX,
              flop=lambda M: fpt * M * N_IDX,
              byts=lambda M: bpt * M * N_IDX,
              note=f"fused, compressed ctx={IDX_CTX}; per-request keys/query/head-wts; no score DRAM; AI M-invariant",
              crossing=None)


def moe_experts():
    """Routed experts, 3-matrix SwiGLU, FUSED. distinct experts/step = E*(1-(1-TOPK/E)^M)
    (correction 9, distinct-top-k). MXFP4 weights 0.53125 B/weight."""
    per = 3 * H * MOE_I
    distinct = lambda M: E * (1 - (1 - TOPK / E) ** M)
    return Op("MoE experts (gate+up+SiLU+down, MXFP4)", "A", "donor moe.cpp", "fp4", N_MOE,
              flop=lambda M: 2 * M * TOPK * per * N_MOE,
              byts=lambda M: (distinct(M) * per * FP4 + M * TOPK * (2 * H + MOE_I) * AB) * N_MOE,
              note="BW-bound; distinct experts=E*(1-(1-TOPK/E)^M); weights grow with M via occupancy",
              crossing=None, weight_bytes_per_layer=E * per * FP4)   # capacity = ALL experts resident


def shared_expert():
    """Shared expert: THREE matrices (gate+up+down), fused (correction 8). 6*M*H*MOE_I FLOPs,
    3*H*MOE_I weights, 2*M*H activations (input once/output once)."""
    per = 3 * H * MOE_I
    fpt, fixed, bpt = 6 * H * MOE_I, per * FP4, 2 * H * AB
    return Op("shared-expert (3-matrix gate+up+down, MXFP4)", "A", "donor moe.cpp", "fp4", N_MOE,
              flop=lambda M: fpt * M * N_MOE,
              byts=lambda M: (fixed + bpt * M) * N_MOE,
              note="3 matrices (fusion keeps all 3 weights)",
              crossing=ridge_crossing(fpt, fixed, bpt), weight_bytes_per_layer=fixed)


# ================= tensor inventory (ONE source for OPS + capacity) =================
HC = 4


def hc_post():
    """MODELED (published model.py hc_post): y = post*x + sum_j comb[.,j,k]*residual[.,j,d].
    Per token: comb@residual bmm (2*HC*HC*H) + post*x elementwise (HC*H). Runs 2x/layer."""
    fpt = 2 * HC * HC * H + HC * H
    bpt = (2 * HC * H + HC * HC + H) * AB
    return Op("MHC hc_post (post*x + comb@residual)", "C", "NEW-C++", "bf16", 2 * L,
              flop=lambda M: fpt * M * 2 * L, byts=lambda M: bpt * M * 2 * L,
              note="MODELED: per-token comb@residual bmm + elementwise", crossing=None)


def hc_head():
    """MODELED (published model.py hc_head): RMSNorm + linear(hc_fn[HC, HC*H]) + sigmoid + weighted-sum.
    Once at the head. hc_fn weight is HC*HC*H."""
    w_el = HC * HC * H
    fpt = 2 * HC * HC * H + HC * H
    fixed, bpt = w_el * BF16, 2 * HC * H * AB
    return Op("MHC hc_head (LM-head mixer, once)", "C", "NEW-C++", "bf16", 1,
              flop=lambda M: fpt * M, byts=lambda M: fixed + bpt * M,
              note="MODELED: RMSNorm+linear(hc_fn)+sigmoid+weighted-sum, once",
              crossing=ridge_crossing(fpt, fixed, bpt), weight_bytes_per_layer=fixed)


OPS = [
    wgemm("MLA wqkv_a (fused q_a+kv_a, 4096->1536)", H, QLORA + HD, "fp8", L, "A", "donor dsv2"),
    measured("  q_norm RMSNorm(1024)", "A", "donor norm.cpp", 5e-6, L, "RMSNorm, L2-resident"),
    wgemm("MLA wq_b (1024->32768)", QLORA, NH * HD, "fp8", L, "A", "donor dsv2"),
    measured("  RoPE (yarn)", "A", "donor rope", 5e-6, L, "position-indexed rope"),
    measured("  kv_norm RMSNorm(512)", "A", "donor norm.cpp", 5e-6, L, "RMSNorm"),
    attn("MLA attn core (flash MQA, ctx=S)", S, L, "A", "donor intel_amx attn"),
    wgemm("MLA wo_a (grouped 8x 4096->1024)", NH * HD // OG, OLORA, "bf16", L, "A", "donor dsv2 (BF16 CPU)", groups=OG),
    wgemm("MLA wo_b (8192->4096)", OG * OLORA, H, "fp8", L, "A", "donor dsv2"),
    # --- DSA (ratio-4 indexer layers = 20) ---
    wgemm("DSA indexer wq_b (1024->8192)", QLORA, IDX_NH * IDX_HD, "fp8", N_IDX, "A", "donor dsv2"),
    wgemm("DSA indexer weights_proj (4096->64)", H, IDX_NH, "bf16", N_IDX, "A", "donor gemm"),
    indexer_fused(),
    measured("DSA indexer topk-512 (over ~1024)", "B", "NEW-C++", 0.04e-3, N_IDX, "nth_element over compressed ctx"),
    measured("DSA compressor (softmax-pool, D=512)", "B", "NEW-C++", 0.5e-3, N_COMP, "streaming online-softmax pool"),
    attn("DSA sparse attend (top-512 KV)", IDX_TOPK, N_IDX, "B", "donor MLA flash"),
    # --- MHC (hyper-connections every layer + head) ---
    wgemm("MHC hc_fn (16384->24, x2/layer)", NH * HD, (2 + 4) * 4, "bf16", 2 * L, "C", "NEW-C++"),
    measured("MHC sinkhorn (hc=4, 20 iters)", "C", "NEW-C++", 0.015e-3, 2 * L, "fused per-row iters"),
    measured("MHC combine (hc_pre reduce)", "C", "NEW-C++", 0.013e-3, 2 * L, "tiled accumulate-once"),
    hc_post(),
    hc_head(),
    # --- MoE ---
    wgemm("MoE router gate (4096->256)", H, E, "bf16", N_MOE, "A", "donor gemm"),
    measured("MoE hash route (3 layers, tid2eid gather)", "A", "donor", 0.005e-3, N_HASH, "lookup, not GEMM"),
    moe_experts(),
    shared_expert(),
    # --- embedding / head ---
    measured("embed (VocabParallel lookup)", "A", "donor embed", 10e-6, 1, "gather"),
    wgemm("lm_head (4096->129280)", H, VOCAB, "bf16", 1, "A", "donor gemm"),
]


def capacity():
    """Resident weights from the SAME inventory (correction 3/10). Returns dict GB."""
    gb = {}
    for op in OPS:
        if op.weight_bytes:
            gb[op.name] = op.weight_bytes / 1e9
    return gb


def selftest():
    """Invariant assertions BEFORE reporting (correction 10: test before regenerating)."""
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    # 1. independent-request ops: per-token bytes scale ~linearly with M (KV/activation per request)
    a = attn("t", S, 1, "A", "x")
    chk(abs(a.byts(8) - 8 * a.byts(1)) < 1, "attention bytes(M=8) == 8*bytes(M=1) (KV scales with M)")
    # 2. fused attention/indexer carry NO score term: byte(M)/M is constant (pure per-token)
    chk(abs(a.byts(16) / 16 - a.byts(1)) < 1, "attention has no M-independent (score/weight) byte term")
    ix = indexer_fused()
    chk(abs(ix.byts(16) / 16 - ix.byts(1) / 1 * 1) < 1e-3 * ix.byts(1), "fused indexer byte/token constant (no score DRAM)")
    # 3. weight-GEMM: weights M-independent -> byts(0-extrapolate) = fixed weight bytes == capacity term
    g = wgemm("t", 4096, 8192, "fp8", 1, "A", "x")
    chk(abs((g.byts(8) - g.byts(1)) / 7 - (g.flop(1) / 1 * 0 + (4096 + 8192) * AB)) < 1, "GEMM per-token bytes == (K+N)*AB")
    # 4. ridge_crossing returns None when AI saturates below ridge (narrow output)
    wp = wgemm("t", 4096, 64, "bf16", 1, "A", "x")
    chk(wp.crossing is None, "weights_proj (4096->64) has NO ridge crossing (AI saturates < ridge)")
    # 5. shared expert counts 3 matrices
    se = shared_expert()
    chk(abs(se.flop(1) / N_MOE - 6 * H * MOE_I) < 1, "shared expert FLOPs = 6*H*MOE_I (3 matrices)")
    # 6. occupancy: one top-6 request hits exactly ~6 distinct experts (not 5.91)
    d1 = E * (1 - (1 - TOPK / E) ** 1)
    chk(abs(d1 - TOPK) < 1e-9, f"distinct experts at M=1 == TOPK={TOPK} (got {d1:.4f})")
    # 7. capacity derived from inventory (no second hardcoded formula)
    chk(sum(capacity().values()) > 0, "capacity() derives from the OPS inventory")
    print(f"  SELFTEST {'OK' if ok else 'FAILED'}")
    return ok


def p0():
    cap = capacity()
    tot = sum(cap.values())
    print("=" * 96)
    print(f"P0 CAPACITY (from one inventory)  platform={NAME}  nominal BW={BW/1e9:.1f} GB/s  "
          f"AMX={PEAK/1e12:.1f} TF  ridge={RIDGE:.0f}")
    print(f"  measured reference (node {PLAT.get('measurement_node','?')}): "
          f"BW={BW_MEAS/1e9:.0f} GB/s  AMX={PEAK_MEAS/1e12:.0f} TF  (reference obs, not a target/floor)")
    print("=" * 96)
    for n, g in sorted(cap.items(), key=lambda x: -x[1]):
        print(f"  {n[:46]:46s} {g:8.2f} GB")
    print(f"  {'RESIDENT (listed weights only; excl. KV/scales/workspace)':46s} {tot:8.2f} GB  "
          f"-> {'fits' if tot < DOMAIN_RAM/1e9 else 'OOM'} one domain (feasibility, not TP-optimality)\n")


def phaseA():
    print("=" * 128)
    print(f"PHASE A — IDEAL per-op roofline (declared workload; max(B/BW,F/P)); invocation-counted; fused=no-score")
    print("=" * 128)
    print(f"{'op':44s} {'lane/impl':24s} | " + " ".join(f"{'M='+str(m):>9s}" for m in Ms) + "  calls bind  cross@ridge")
    print("-" * 128)
    for op in OPS:
        cells, regs = [], []
        for M in Ms:
            ai, reg, t = op.row(M)
            regs.append(reg)
            cells.append(f"{fmt(t):>9s}")
        xs = "lat" if op.kind == "latency" else ("none" if op.crossing is None else f"M≈{op.crossing:.0f}")
        print(f"{op.name:44s} {(op.lane+' '+op.impl)[:24]:24s} | " + " ".join(cells) + f"  {op.layers:>5d} {regs[-1]:>4s}  {xs}")
    print("\ntimes are PER-STEP (= per-call x calls/step); per-call = per-step / calls. "
          "AI: B=BW C=compute lat=measured; fused=no score DRAM; attn/indexer AI M-invariant.")
    print(f"{'op':44s} | " + " ".join(f"AI@{m:<3d}" for m in Ms))
    print("-" * 128)
    for op in OPS:
        ais = []
        for M in Ms:
            ai, reg, t = op.row(M)
            ais.append("  lat " if op.kind == "latency" else f"{ai:5.0f}{reg}")
        print(f"{op.name:44s} | " + " ".join(ais))


if __name__ == "__main__":
    print("INVARIANT SELFTEST:")
    passed = selftest()
    if _a.selftest:
        sys.exit(0 if passed else 1)
    if not passed:
        print("\n*** invariants FAILED — not emitting report ***")
        sys.exit(1)
    print()
    p0()
    phaseA()
