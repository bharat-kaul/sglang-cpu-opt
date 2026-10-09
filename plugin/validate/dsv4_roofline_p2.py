#!/usr/bin/env python3
"""DSv4-Flash IDEAL roofline — REFERENCE-CONFORMANT (pinned revision), invariant + reference tested.

Addresses gate review R1-R7 (reports/dsv4_flash_roofline_gate_review.md). The model is derived
from the PINNED DeepSeek-V4-Flash checkpoint config + inference/model.py graph + checkpoint-header
dtypes (HF revision 60d8d770770c6776ff598c94bb586a859a38244f1). The self-test now asserts
REFERENCE CONFORMANCE (layer counts, attention positions, MHC dims, tensor-family dtypes), not only
internal consistency. IDEAL target: t = max(B/BW_peak, F/P_peak) with declared workload + fusion.

DECLARED WORKLOAD: independent decode requests, batch M, no shared prefix; weights read once/step;
KV/activations per request; flash/fused => no intermediate-score DRAM. Decode boundary = 4096 tokens
available, full 128-token window. Main model = 43 blocks (MTP excluded).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_DEF = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
_ap = argparse.ArgumentParser()
_ap.add_argument("--platform", default=_DEF)
_ap.add_argument("--selftest", action="store_true")
_a, _ = _ap.parse_known_args()
PLAT = json.load(open(_a.platform))
MP = PLAT.get("machine_peak") or {}
NAME = PLAT["name"]
BW = MP.get("mem_bw_gbps", PLAT["mem_bw_gbps"]) * 1e9
PEAK = MP.get("amx_bf16_tflops", PLAT["amx_bf16_tflops"]) * 1e12
RIDGE = PEAK / BW
DOMAIN_RAM = PLAT["domain_ram_gb"] * 1e9

# ---- dtype bytes/weight incl quant scales (checkpoint-header dtypes, R3/R4) ----
FP8 = 1.0 + 1.0 / 16384     # fp8 + E8M0 scale per 128x128 block (MLA proj, SHARED experts)
FP4 = 0.5 + 1.0 / 32        # MXFP4 + 1B scale per 32 (ROUTED experts only)
BF16 = 2.0                  # checkpoint BF16 storage (wo_a, compressor proj, router, lm_head)
FP32 = 4.0                  # MHC hc_* params are F32 in the checkpoint
AB = 2.0                    # bf16 activation
KV_BPW = 2.0                # bf16 latent KV
BPWS = {"fp8": FP8, "fp4": FP4, "bf16": BF16, "fp32": FP32}
Ms = [1, 8, 16, 32, 64]

# ---- pinned config (HF rev 60d8d707...) ----
H, NH, HD = 4096, 64, 512          # hidden, attention heads, head_dim
QLORA, OLORA, OG = 1024, 1024, 8
VOCAB, E, TOPK, MOE_I = 129280, 256, 6, 2048
IDX_NH, IDX_HD, IDX_TOPK = 64, 128, 512
WINDOW, S = 128, 4096              # sliding window; decode boundary tokens available
# compress_ratios[:num_hidden_layers] from the pinned config (R1): 2x ratio-0, 21x ratio-4, 20x ratio-128
COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0]   # 44 entries; [:43]=main, [43]=MTP
MAIN = COMPRESS_RATIOS[:43]
L = len(MAIN)                      # 43 main blocks
_cnt = collections.Counter(MAIN)
N_SWA, N_IDX, N_128 = _cnt[0], _cnt[4], _cnt[128]   # 2, 21, 20
N_COMP = N_IDX + N_128             # 41 layers carry a (main) compressor
N_MOE, N_HASH = L, 3               # all 43 MoE; first 3 hash-routed


def _attn_positions(ratio):
    """Reference attention sources per layer: sliding window + compressed indices (R1)."""
    if ratio == 0:
        return WINDOW
    if ratio == 4:
        return WINDOW + min(IDX_TOPK, S // ratio)   # 128 + min(512, 1024) = 640
    return WINDOW + S // ratio                       # 128 + 4096/128 = 160


SUM_POS = sum(_attn_positions(r) for r in MAIN)      # 16896


def fmt(t):
    for u, s in (("s", 1), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        if t * s >= 1 or u == "ns":
            return f"{t*s:6.1f}{u}"


def ridge_crossing(fpt, fixed_bytes, bpt):
    denom = fpt - RIDGE * bpt
    return None if denom <= 0 else RIDGE * fixed_bytes / denom


class Op:
    def __init__(self, name, lane, impl, prec, layers, flop, byts, note,
                 crossing=None, kind="gemm", lat_s=None, weight_bytes_per_layer=0.0, unmodeled=""):
        self.name, self.lane, self.impl, self.prec, self.layers = name, lane, impl, prec, layers
        self.flop, self.byts, self.note = flop, byts, note
        self.crossing, self.kind, self.lat_s = crossing, kind, lat_s
        self.weight_bytes = weight_bytes_per_layer * layers
        self.unmodeled = unmodeled

    def row(self, M):
        if self.kind == "latency":
            return 0.0, "lat", self.lat_s
        fl, by = self.flop(M), self.byts(M)
        ai = fl / by if by else 0.0
        return ai, ("C" if ai > RIDGE else "B"), max(by / BW, fl / PEAK)


def wgemm(name, K, N, prec, layers, lane, impl, groups=1, unmodeled=""):
    bpw = BPWS[prec]
    w_el, fpt = groups * K * N, 2 * groups * K * N
    fixed, bpt = w_el * bpw, groups * (K + N) * AB
    return Op(name, lane, impl, prec, layers,
              flop=lambda M: fpt * M * layers, byts=lambda M: (fixed + bpt * M) * layers,
              note="weight-streaming" + (f"; groups={groups}" if groups > 1 else ""),
              crossing=ridge_crossing(fpt, fixed, bpt), weight_bytes_per_layer=fixed, unmodeled=unmodeled)


def measured(name, lane, impl, lat_s, layers, prov):
    """Latency row from a SINGLE-POINT measured observation (R5). prov = node/revision/dtype/batch/ctx.
    Valid only for that operating point; NOT a batch sweep and NOT a theoretical floor."""
    return Op(name, lane, impl, "meas", layers, flop=lambda M: 0, byts=lambda M: 0,
              note="single-point measured: " + prov, kind="latency", lat_s=lat_s * layers)


def pool(name, win, D, calls, lane, kv_bpw=KV_BPW):
    """Softmax-pool over `win` positions per channel D; `calls` = amortized boundary calls/step (R2)."""
    fpt = 3 * win * D                       # score+exp+weighted-sum
    bpt = (win * D + D) * kv_bpw             # read window, write 1 compressed token (per request)
    return Op(name, lane, "NEW-C++", "bf16", calls,
              flop=lambda M: fpt * M * calls, byts=lambda M: bpt * M * calls,
              note=f"win={win} D={D}; amortized {calls:.3f} calls/step", crossing=None)


def attn_sparse():
    """ONE MLA sparse attention per layer over window+compressed sources (R1), MQA flash-fused.
    Positions summed across the 43 main layers = SUM_POS; q/out counted once per layer."""
    fpt_tot = 4 * NH * SUM_POS * HD          # summed over layers
    kv_bytes = lambda M: M * SUM_POS * HD * KV_BPW
    qo_bytes = lambda M: L * 2 * M * NH * HD * AB
    return Op("MLA sparse attention (window+compressed, 43L)", "A", "donor MLA flash", "fp8", 1,
              flop=lambda M: fpt_tot * M, byts=lambda M: kv_bytes(M) + qo_bytes(M),
              note=f"single sparse op; positions/req summed={SUM_POS}; MQA flash; no score DRAM; AI M-invariant",
              crossing=None)


def indexer_fused():
    """Fused indexer over the COMPRESSED index context (seqlen/ratio4), ratio-4 layers only.
    Per request: keys IDX_CTX*IDX_HD, query IDX_NH*IDX_HD, fp32 head-weights IDX_NH, fp32 out IDX_CTX."""
    idx_ctx = S // 4
    key_bpw, q_bpw = KV_BPW, FP4             # index-kv bf16; query fp4-sim (reference)
    matmul_fpt, vec_fpt = 2 * IDX_NH * idx_ctx * IDX_HD, 3 * IDX_NH * idx_ctx
    fpt = matmul_fpt + vec_fpt
    bpt = idx_ctx * IDX_HD * key_bpw + IDX_NH * IDX_HD * q_bpw + IDX_NH * 4 + idx_ctx * 4
    return Op("DSA indexer logits (GEMM+reduce FUSED)", "B", "NEW-C++", "bf16", N_IDX,
              flop=lambda M: fpt * M * N_IDX, byts=lambda M: bpt * M * N_IDX,
              note=f"fused; index ctx={idx_ctx}; head-weights + fp32 out counted; no score DRAM",
              crossing=None)


def moe_experts():
    per, distinct = 3 * H * MOE_I, lambda M: E * (1 - (1 - TOPK / E) ** M)
    return Op("MoE routed experts (MXFP4, 3-matrix)", "A", "donor moe.cpp", "fp4", N_MOE,
              flop=lambda M: 2 * M * TOPK * per * N_MOE,
              byts=lambda M: (distinct(M) * per * FP4 + M * TOPK * (2 * H + MOE_I) * AB) * N_MOE,
              note="distinct experts=E*(1-(1-TOPK/E)^M); routed=MXFP4 0.53125",
              crossing=None, weight_bytes_per_layer=E * per * FP4)


def shared_expert():
    """Shared expert: 3 matrices, FP8 storage (checkpoint F8_E4M3, R3) \u2014 NOT MXFP4."""
    per = 3 * H * MOE_I
    fpt, fixed, bpt = 6 * H * MOE_I, per * FP8, 2 * H * AB
    return Op("shared-expert (3-matrix, FP8)", "A", "donor moe.cpp", "fp8", N_MOE,
              flop=lambda M: fpt * M * N_MOE, byts=lambda M: (fixed + bpt * M) * N_MOE,
              note="FP8 per checkpoint header (routed are FP4)",
              crossing=ridge_crossing(fpt, fixed, bpt), weight_bytes_per_layer=fixed)


def hc_fn():
    """MHC pre-mix projection: hc_mult*hidden -> (2+hc)*hc, FP32 (checkpoint F32 [24,16384]). 2x/layer."""
    K, N = 4 * H, (2 + 4) * 4                 # 16384 -> 24
    fpt, fixed, bpt = 2 * K * N, K * N * FP32, (K + N) * AB
    return Op("MHC hc_fn (16384->24, FP32)", "C", "NEW-C++", "fp32", 2 * L,
              flop=lambda M: fpt * M * 2 * L, byts=lambda M: (fixed + bpt * M) * 2 * L,
              note="K=hc_mult*H=16384; FP32 params", crossing=ridge_crossing(fpt, fixed, bpt),
              weight_bytes_per_layer=fixed)


def hc_post():
    HC = 4
    fpt, bpt = 2 * HC * HC * H + HC * H, (2 * HC * H + HC * HC + H) * AB
    return Op("MHC hc_post (post*x + comb@residual)", "C", "NEW-C++", "fp32", 2 * L,
              flop=lambda M: fpt * M * 2 * L, byts=lambda M: bpt * M * 2 * L,
              note="comb@residual bmm + elementwise", crossing=None,
              unmodeled="norm/affine/sigmoid vector work excluded from this ideal target")


def hc_head():
    HC = 4
    w_el, fpt = HC * HC * H, 2 * HC * HC * H + HC * H
    fixed, bpt = w_el * FP32, 2 * HC * H * AB
    return Op("MHC hc_head (mixer, once, FP32)", "C", "NEW-C++", "fp32", 1,
              flop=lambda M: fpt * M, byts=lambda M: fixed + bpt * M,
              note="projection + weighted-sum", crossing=ridge_crossing(fpt, fixed, bpt),
              weight_bytes_per_layer=fixed,
              unmodeled="RMSNorm + sigmoid gating + reduction vector work excluded from this ideal target")


# ================= ONE tensor inventory (OPS + capacity) =================
OPS = [
    wgemm("MLA wqkv_a (fused q_a+kv_a, 4096->1536)", H, QLORA + HD, "fp8", L, "A", "donor dsv2"),
    measured("  q_norm RMSNorm(1024)", "A", "donor norm.cpp", 5e-6, L, "donor norm; M=unmodeled (single-point)"),
    wgemm("MLA wq_b (1024->32768)", QLORA, NH * HD, "fp8", L, "A", "donor dsv2"),
    measured("  RoPE (yarn)", "A", "donor rope", 5e-6, L, "donor rope; single-point"),
    measured("  kv_norm RMSNorm(512)", "A", "donor norm.cpp", 5e-6, L, "donor norm; single-point"),
    attn_sparse(),
    wgemm("MLA wo_a (grouped 8x 4096->1024)", NH * HD // OG, OLORA, "bf16", L, "A", "donor dsv2 BF16", groups=OG),
    wgemm("MLA wo_b (8192->4096)", OG * OLORA, H, "fp8", L, "A", "donor dsv2"),
    # --- DSA indexer (21 ratio-4 layers) ---
    wgemm("DSA indexer wq_b (1024->8192)", QLORA, IDX_NH * IDX_HD, "fp8", N_IDX, "A", "donor dsv2"),
    wgemm("DSA indexer weights_proj (4096->64)", H, IDX_NH, "bf16", N_IDX, "A", "donor gemm"),
    indexer_fused(),
    measured("DSA indexer topk-512 (over 1024)", "B", "NEW-C++", 0.039e-3, N_IDX,
             "pcl-sprh02 DDR5600, M=32, S=1024, fp32 logits; single-point"),
    # --- compressors: projections (every token) + pooling (boundary-amortized) (R2) ---
    wgemm("main compressor wkv+wgate (r4, 4096->2048)", H, 2 * 2 * HD, "bf16", N_IDX, "A", "donor dsv2"),
    wgemm("main compressor wkv+wgate (r128, 4096->1024)", H, 2 * HD, "bf16", N_128, "A", "donor dsv2"),
    wgemm("indexer compressor wkv+wgate (r4, 4096->512)", H, 2 * 2 * IDX_HD, "bf16", N_IDX, "A", "donor dsv2"),
    pool("main pool (r4 overlap, win=8 D=512)", 8, HD, N_IDX / 4, "B"),
    pool("main pool (r128, win=128 D=512)", 128, HD, N_128 / 128, "B"),
    pool("indexer pool (r4 overlap, win=8 D=128)", 8, IDX_HD, N_IDX / 4, "B"),
    # --- MHC (every layer x2 pre + post; head once) ---
    hc_fn(),
    measured("MHC sinkhorn (hc=4, 20 iters)", "C", "NEW-C++", 0.015e-3, 2 * L,
             "pcl-sprh02, M=32, hc=4; single-point"),
    measured("MHC combine (hc_pre reduce)", "C", "NEW-C++", 0.013e-3, 2 * L,
             "pcl-sprh02, M=32, hc=4 H=4096; single-point"),
    hc_post(),
    hc_head(),
    # --- MoE ---
    wgemm("MoE router gate (4096->256)", H, E, "bf16", N_MOE, "A", "donor gemm"),
    measured("MoE hash route (3 layers, tid2eid gather)", "A", "donor", 0.005e-3, N_HASH, "lookup; single-point"),
    moe_experts(),
    shared_expert(),
    # --- embedding / head ---
    measured("embed (VocabParallel lookup)", "A", "donor embed", 10e-6, 1, "donor; single-point"),
    wgemm("lm_head (4096->129280)", H, VOCAB, "bf16", 1, "A", "donor gemm"),
]


def capacity():
    return {op.name: op.weight_bytes / 1e9 for op in OPS if op.weight_bytes}


def unmodeled_items():
    live = [(op.name, op.unmodeled) for op in OPS if op.unmodeled]
    try:
        trk = json.load(open(os.path.join(os.path.dirname(__file__), "results", "roofline_open_items.json")))
        live += [(it["item"], it.get("reason", it.get("disposition", "")))
                 for it in trk["items"] if "UNMODELED" in it.get("disposition", "")]
    except Exception:
        pass
    return live


def selftest():
    """REFERENCE-CONFORMANCE + invariant checks (R7). Gate report emission."""
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and c
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    chk(dict(collections.Counter(MAIN)) == {0: 2, 4: 21, 128: 20}, "main layer counts == {0:2,4:21,128:20} (pinned config)")
    chk(SUM_POS == 16896, f"attention positions/request summed == 16896 (got {SUM_POS})")
    chk(abs(attn_sparse().flop(1) - 2_214_592_512) < 1, "attention FLOPs@M=1 == 2,214,592,512 (reference graph)")
    chk(abs(hc_fn().flop(1) - 67_633_152) < 1, "MHC hc_fn FLOPs@M=1 == 67,633,152 (K=16384)")
    chk(shared_expert().prec == "fp8", "shared expert dtype == FP8 (checkpoint header, not MXFP4)")
    chk(moe_experts().prec == "fp4", "routed experts dtype == FP4 (MXFP4)")
    chk(hc_fn().prec == "fp32", "MHC hc_fn dtype == FP32 (checkpoint header)")
    a = attn_sparse()
    chk(abs(a.byts(8) - 8 * a.byts(1)) < 1, "attention bytes linear in M (independent requests)")
    g = wgemm("t", 4096, 64, "bf16", 1, "A", "x")
    chk(g.crossing is None, "narrow GEMM (4096->64) has NO ridge crossing")
    chk(abs(E * (1 - (1 - TOPK / E) ** 1) - TOPK) < 1e-9, "distinct experts@M=1 == TOPK")
    main_comp_gb = sum(capacity()[n] for n in capacity() if n.startswith("main compressor"))
    chk(abs(main_comp_gb - 0.520093696) < 1e-3, f"main compressor matrices == 0.520 GB (got {main_comp_gb:.4f})")
    idx_comp_gb = sum(capacity()[n] for n in capacity() if n.startswith("indexer compressor"))
    chk(abs(idx_comp_gb - 0.088080384) < 1e-3, f"indexer compressor matrices == 0.088 GB (got {idx_comp_gb:.4f})")
    shared_gb = capacity().get("shared-expert (3-matrix, FP8)", 0)
    chk(abs(shared_gb - 1.082196480) < 2e-3, f"shared-expert FP8 capacity == 1.082 GB (got {shared_gb:.4f})")
    print(f"  SELFTEST {'OK' if ok else 'FAILED'}")
    return ok


def p0():
    cap = capacity()
    tot = sum(cap.values())
    print("=" * 96)
    print(f"P0 CAPACITY (one inventory)  platform={NAME}  nominal BW={BW/1e9:.1f} GB/s  AMX={PEAK/1e12:.1f} TF  ridge={RIDGE:.0f}")
    print(f"  measured reference (node {PLAT.get('measurement_node','?')}): BW={PLAT['mem_bw_gbps']:.0f} GB/s  "
          f"AMX={PLAT['amx_bf16_tflops']:.0f} TF  (reference observation; nominal target != measured)")
    print("=" * 96)
    for n, g in sorted(cap.items(), key=lambda x: -x[1]):
        print(f"  {n[:48]:48s} {g:8.3f} GB")
    print(f"  {'RESIDENT (listed matrices only)':48s} {tot:8.3f} GB  -> excludes KV/state/scales/workspace/embed; "
          f"feasibility only, NOT runtime fit or TP-optimality")
    print("\n  UNMODELED quantities (surfaced, not in any total/ranking):")
    for name, why in unmodeled_items():
        print(f"    - {name}: {why[:88]}")


def phaseA():
    print("\n" + "=" * 128)
    print("PHASE A — IDEAL per-op roofline (reference-conformant; single fused sparse attention; "
          "boundary-amortized compression)")
    print("=" * 128)
    print(f"{'op':46s} {'lane/impl':22s} | " + " ".join(f"{'M='+str(m):>9s}" for m in Ms) + "  calls bind cross")
    print("-" * 128)
    for op in OPS:
        cells, regs = [], []
        for M in Ms:
            ai, reg, t = op.row(M)
            regs.append(reg)
            cells.append(f"{fmt(t):>9s}")
        xs = "lat" if op.kind == "latency" else ("none" if op.crossing is None else f"M{op.crossing:.0f}")
        cs = f"{op.layers:.2f}" if isinstance(op.layers, float) else str(op.layers)
        print(f"{op.name[:46]:46s} {(op.lane+' '+op.impl)[:22]:22s} | " + " ".join(cells) + f"  {cs:>5s} {regs[-1]:>4s} {xs}")
    print("\ntimes PER-STEP (= per-call x calls/step); fused=no score DRAM; attn/indexer AI M-invariant. "
          "Distance-from-roof is diagnostic only.")


if __name__ == "__main__":
    print("REFERENCE-CONFORMANCE + INVARIANT SELFTEST:")
    passed = selftest()
    if _a.selftest:
        sys.exit(0 if passed else 1)
    if not passed:
        print("\n*** reference/invariant checks FAILED \u2014 not emitting report ***")
        sys.exit(1)
    print()
    p0()
    phaseA()
