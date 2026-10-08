#!/usr/bin/env python3
"""DSv4-Flash Phase-A per-op roofline + fusion pass (skill-aligned, analytical).

Follows the glm5-perf skills model-op-decomposition -> fusion-analysis ->
model-roofline-analysis, and the pilot GOVERNING METHODOLOGY (two-phase):
  PHASE A (this script): discover EVERY op in a layer INCLUDING fused ops; for each,
    roofline + BINDING RESOURCE (BW/compute/latency) on the target platform; whether its
    arithmetic intensity is IMPROVABLE by a lever (increase M, fusion); the achievable
    CEILING; and the C/C++ implementation lane. NO op is ROI-skipped -- every op gets a
    ceiling target; ops whose gap cannot close by known means are flagged SURFACE-TO-USER.
  PHASE B (later): wall-time + Amdahl ordering, only after all ops are at ceiling.

Machine constants come ONLY from the platform spec (platform_scan.py / uarch-perf-probe);
nothing is hardcoded. Decode is the weight-streaming phase (batch sweep); prefill noted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_DEF = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
_ap = argparse.ArgumentParser()
_ap.add_argument("--platform", default=_DEF)
_a, _ = _ap.parse_known_args()
with open(_a.platform) as f:
    PLAT = json.load(f)
if PLAT.get("mem_bw_gbps") is None or PLAT.get("amx_bf16_tflops") is None:
    sys.exit(f"{_a.platform}: measure mem_bw_gbps / amx_bf16_tflops first (uarch-perf-probe).")

NAME = PLAT["name"]
# Reference FLOOR = measured-achievable (where we are today).
BW_MEAS = PLAT["mem_bw_gbps"] * 1e9
PEAK_MEAS = PLAT["amx_bf16_tflops"] * 1e12
# Roofline CEILING = machine (theoretical datasheet) peak; optimize against this.
_MP = PLAT.get("machine_peak") or {}
BW = _MP.get("mem_bw_gbps", PLAT["mem_bw_gbps"]) * 1e9
PEAK = _MP.get("amx_bf16_tflops", PLAT["amx_bf16_tflops"]) * 1e12
RIDGE = PEAK / BW
DOMAIN_RAM = PLAT["domain_ram_gb"] * 1e9
BPW = {"fp4": 0.5, "fp8": 1.0, "bf16": 2.0}
AB = 2.0  # bf16 activation bytes
Ms = [1, 8, 16, 32, 64]

# ---- DSv4-Flash config (real) ----
H, L, NH, HD = 4096, 43, 64, 512
QLORA, OLORA, OG = 1024, 1024, 8
VOCAB, E, TOPK, MOE_I = 129280, 256, 6, 2048
N_MOE, N_IDX = 41, 40
IDX_NH, IDX_HD, IDX_TOPK = 64, 128, 512
S = 4096  # representative decode context (attn/DSA anchor)


def fmt(t):
    for u, s in (("s", 1), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        if t * s >= 1 or u == "ns":
            return f"{t*s:6.1f}{u}"


def cross_M(bpw):
    """M at which a weight-streaming GEMM (AI=2M/bpw) crosses the ridge -> compute-bound."""
    return RIDGE * bpw / 2.0


class Op:
    def __init__(self, name, cls, lane, impl, prec, flop, byts, binding_note, improv):
        self.name, self.cls, self.lane, self.impl, self.prec = name, cls, lane, impl, prec
        self.flop, self.byts = flop, byts          # callables of M -> value
        self.binding_note, self.improv = binding_note, improv

    def row(self, M):
        fl, by = self.flop(M), self.byts(M)
        ai = fl / by if by else 0.0
        t = max(by / BW, fl / PEAK) if by else self._lat(M)
        if by == 0:
            reg = "latency"
        else:
            reg = "compute" if ai > RIDGE else "BW"
        return ai, reg, t

    def _lat(self, M):
        return self.lat_us * 1e-6 * self.layers


def wgemm(name, K, N, prec, layers, lane, impl, fused=False):
    bpw = BPW[prec]
    return Op(name, "GEMM", lane, impl, prec,
              flop=lambda M: 2 * M * K * N * layers,
              byts=lambda M: (K * N * bpw + (M * K + M * N) * AB) * layers,
              binding_note=f"BW<{cross_M(bpw):.0f}M, compute>=; fused_in={fused}",
              improv=f"↑M crosses ridge at M≈{cross_M(bpw):.0f}")


def latency(name, lane, impl, lat_us, layers, note):
    o = Op(name, "latency", lane, impl, "-",
           flop=lambda M: 0, byts=lambda M: 0, binding_note=note,
           improv="fuse/batch dispatch (no AI lever)")
    o.lat_us, o.layers = lat_us, layers
    return o


def attn(name, Sctx, prec, layers, lane, impl):
    bpw = BPW[prec]
    return Op(name, "attn", lane, impl, prec,
              flop=lambda M: 4 * M * NH * Sctx * HD * layers,
              byts=lambda M: (Sctx * HD * bpw + (M * NH * HD + M * NH * Sctx) * AB) * layers,
              binding_note="compute (flash-fused, no score materialize)",
              improv="AI set by context S + flash-fusion, not M")


def indexer_fused():
    """Fused DSA indexer: GEMM q·ck -> relu*weight*sum epilogue (Lane B, NEW C++)."""
    return Op("DSA indexer logits (GEMM+epilogue FUSED)", "fused", "B", "NEW-C++ (xpu-triton oracle)", "bf16",
              flop=lambda M: (2 * M * IDX_NH * S * IDX_HD + 3 * M * IDX_NH * S) * N_IDX,
              byts=lambda M: (S * IDX_HD * AB + M * IDX_NH * S * AB + M * S * 4) * N_IDX,
              binding_note="BW at small M (score materialize); anchor=context S",
              improv="FUSE epilogue -> drop score DRAM round-trip -> AI↑ (primary lever, not M)")


def moe_experts():
    """Routed experts, FUSED (gate+up + SiLU*mul + down). Anchor = tokens/expert; distinct
    experts per step = E*(1-(1-1/E)^(k*B)) (model-roofline-analysis batched-MoE formula)."""
    bpw = BPW["fp4"]
    per = 3 * H * MOE_I
    return Op("MoE experts (gate+up+SiLU+down FUSED, W4A16)", "fused", "A", "donor moe.cpp MXFP4", "fp4",
              flop=lambda M: 2 * M * TOPK * per * N_MOE,
              byts=lambda M: (E * (1 - (1 - 1.0 / E) ** (TOPK * M)) * per * bpw + M * TOPK * (2 * H + MOE_I) * AB) * N_MOE,
              binding_note="BW ALL M (routing fragments: tokens/expert=M*6/256<=1.5)",
              improv="↑M barely helps (needs avg_M>4 => M>170); lever=EP/grouping, NOT M in {1..64}")


OPS = [
    # --- MLA block (every op) ---
    wgemm("MLA wqkv_a (fused q_a+kv_a, 4096->1536)", H, QLORA + HD, "fp8", L, "A", "donor dsv2 qkv", fused=True),
    latency("  q_norm RMSNorm(1024)", "A", "donor norm.cpp", 5, L, "fuse into wq_b prologue"),
    wgemm("MLA wq_b (1024->32768)", QLORA, NH * HD, "fp8", L, "A", "donor dsv2"),
    latency("  RoPE (yarn)", "A", "donor rope", 5, L, "fuse into attn prologue"),
    latency("  kv_norm RMSNorm(512)", "A", "donor norm.cpp", 5, L, "fuse into attn prologue"),
    attn("MLA attn core (flash MQA /latent 512)", S, "fp8", L, "A", "donor intel_amx attn backend"),
    wgemm("MLA wo_a (grouped, 32768->1024)", NH * HD, OLORA, "fp8", L, "A", "donor dsv2"),
    wgemm("MLA wo_b (1024->4096)", OLORA, H, "fp8", L, "A", "donor dsv2"),
    # --- DSA (indexer layers) ---
    wgemm("DSA indexer wq (4096->8192)", H, IDX_NH * IDX_HD, "fp8", N_IDX, "A", "donor dsv2"),
    wgemm("DSA indexer wk (4096->128)", H, IDX_HD, "fp8", N_IDX, "A", "donor dsv2"),
    indexer_fused(),
    latency("DSA indexer topk-512", "B", "NEW-C++ (partial-sort)", 40, N_IDX, "sort; fuse with epilogue write"),
    latency("DSA compressor (softmax-pool)", "B", "NEW-C++", 30, N_IDX, "BW/latency; fuse pool into write"),
    attn("DSA sparse attend (top-512 KV)", IDX_TOPK, "fp8", N_IDX, "A", "donor dsv2 mla"),
    # --- MHC ---
    latency("MHC sinkhorn (20 iters)", "C", "NEW-C++", 60, L, "iterative; fuse iters, keep L2-resident"),
    latency("MHC hash (3 layers)", "C", "NEW-C++", 20, L, "fuse hash+assign"),
    # --- MoE ---
    wgemm("MoE router gate (4096->256)", H, E, "bf16", N_MOE, "A", "donor gemm"),
    moe_experts(),
    wgemm("shared-expert (gate+up+SiLU+down, 4096/2048)", H, MOE_I, "fp4", N_MOE, "A", "donor moe.cpp", fused=True),
    # --- embedding / head ---
    latency("embed (VocabParallel lookup)", "A", "donor embed", 10, 1, "gather; latency"),
    wgemm("lm_head (4096->129280)", H, VOCAB, "bf16", 1, "A", "donor gemm"),
]

# ---- fusion pass (fusion-analysis taxonomy + donor table) ----
FUSIONS = [
    ("vertical", "q_norm -> wq_b", "norm into GEMM prologue", "COVERED norm.cpp", "small; L2-resident"),
    ("vertical", "kv_norm+RoPE -> attn", "prologue into attn", "COVERED intel_amx attn", "position-indexed RoPE: fuse carefully"),
    ("horizontal", "q_a + kv_a -> wqkv_a", "fused QKV GEMM", "COVERED (already fused)", "1 packed GEMM"),
    ("epilogue", "gate+up + SiLU*mul -> down", "MoE W1 store step", "COVERED moe.cpp", "the big BW win; already donor-fused"),
    ("epilogue", "fp4/fp8 dequant+scale -> GEMM", "W4A16/W8A16 in-GEMM", "COVERED gemm_{fp8,mxfp4}", "removes a full dequant pass"),
    ("epilogue", "DSA indexer relu*weight*sum -> GEMM store", "indexer epilogue", "NEW-C++ (xpu-triton+adhoc oracle)", "drops [M,NH,S] score round-trip: primary indexer lever"),
    ("epilogue", "KV quant(fp8) + cache write", "write kernel", "COVERED kvcache.cpp", "on-write"),
    ("vertical", "compressor pool -> write", "softmax-pool epilogue", "NEW-C++", "fuse pool into compressed-KV write"),
]


def p0():
    experts = E * 3 * H * MOE_I * N_MOE * BPW["fp4"]
    shared = 3 * H * MOE_I * N_MOE * BPW["fp4"]
    mla = (H * (QLORA + HD) + QLORA * NH * HD + NH * HD * OLORA + OLORA * H) * L * BPW["fp8"]
    idx = (H * IDX_NH * IDX_HD + H * IDX_HD) * N_IDX * BPW["fp8"]
    head = VOCAB * H * BPW["bf16"] * 2
    tot = experts + shared + mla + idx + head
    print("=" * 96)
    print(f"P0 CAPACITY  platform={NAME}  domain={DOMAIN_RAM/1e9:.0f} GB  SNC={PLAT.get('snc')}  "
          f"ridge AI*={RIDGE:.0f} FLOP/byte")
    print("=" * 96)
    print(f"  ROOFLINE CEILING = MACHINE PEAK: BW={BW/1e9:.1f} GB/s  AMX bf16={PEAK/1e12:.1f} TFLOP/s  "
          f"(ridge {RIDGE:.0f} FLOP/byte)")
    print(f"  reference FLOOR  = measured:     BW={BW_MEAS/1e9:.1f} GB/s ({BW_MEAS/BW*100:.0f}% of peak)  "
          f"AMX bf16={PEAK_MEAS/1e12:.1f} TFLOP/s ({PEAK_MEAS/PEAK*100:.0f}% of peak)  -> gap = headroom")
    for n, b in [("MoE experts(fp4)", experts), ("shared(fp4)", shared), ("MLA proj(fp8)", mla),
                 ("indexer proj(fp8)", idx), ("embed+lm_head(bf16)", head)]:
        print(f"  {n:22s} {b/1e9:7.1f} GB")
    print(f"  {'RESIDENT':22s} {tot/1e9:7.1f} GB  -> {'FITS' if tot < DOMAIN_RAM else 'OOM'} one domain "
          f"({tot/1e9:.0f}<{DOMAIN_RAM/1e9:.0f}) => tp=1/EP=1\n")


def phaseA():
    print("=" * 128)
    print(f"PHASE A — PER-OP ROOFLINE + BINDING RESOURCE + AI-IMPROVABILITY + CEILING + C++ LANE  "
          f"(decode, context S={S}; EVERY op, none skipped)")
    print("=" * 128)
    print(f"ceiling = MACHINE PEAK (BW {BW/1e9:.0f} GB/s, AMX {PEAK/1e12:.0f} TF); times below are the "
          f"machine-peak floor each op is optimized toward")
    print(f"{'op':44s} {'impl-lane':30s} {'prec':4s} | " + " ".join(f"{'M='+str(m):>9s}" for m in Ms) + "  binding@M=1->64")
    print("-" * 128)
    for op in OPS:
        cells = []
        regs = []
        for M in Ms:
            ai, reg, t = op.row(M)
            regs.append(reg[0].upper())
            cells.append(f"{fmt(t):>9s}")
        trans = "->".join(regs)
        print(f"{op.name:44s} {op.impl:30s} {op.prec:4s} | " + " ".join(cells) + f"  {trans}")
    print()
    print("AI across M (regime B=BW C=compute L=latency) + improvability:")
    print(f"{'op':44s} | " + " ".join(f"AI@{m:<3d}" for m in Ms) + "  improvable?")
    print("-" * 128)
    for op in OPS:
        ais = []
        for M in Ms:
            ai, reg, t = op.row(M)
            ais.append("  -  " if op.cls == "latency" else f"{ai:4.0f}{reg[0].upper()}")
        print(f"{op.name:44s} | " + " ".join(ais) + f"  {op.improv}")


def fusion_pass():
    print("\n" + "=" * 110)
    print("FUSION PASS (fusion-analysis: taxonomy | fuse-into | mapping | note)  -- refines the roofline")
    print("=" * 110)
    print(f"{'kind':11s} {'candidate':34s} {'fuse into':26s} {'mapping':34s}")
    print("-" * 110)
    for kind, cand, into, mapping, note in FUSIONS:
        print(f"{kind:11s} {cand:34s} {into:26s} {mapping:34s}")
        print(f"{'':11s}   -> {note}")
    print("\nCeiling rule: the roofline ceiling is MACHINE PEAK (theoretical BW/compute from the datasheet), "
          "NOT the measured plateau.\nDrive EVERY op/fused-op toward its machine-peak binding ceiling "
          "(BW peak / compute peak / latency floor) by exercising ALL levers\n(fusion, tiling, packing, dtype, "
          "M/EP operating point, dispatch batching).\nWhen an op's levers are exhausted and performance PLATEAUS "
          "below machine peak, SURFACE the residual gap (measured vs machine peak)\nand STOP optimizing that op "
          "(Phase A exit for that op). Phase B (wall-time + Amdahl ordering) begins ONLY after every op has "
          "either reached machine peak or been surfaced as plateaued.")


if __name__ == "__main__":
    p0()
    phaseA()
    fusion_pass()
