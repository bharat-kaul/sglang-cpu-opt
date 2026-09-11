#!/usr/bin/env python3
"""Generic peer-relative roofline validator (model-agnostic).

Implements the `peer-relative-roofline` skill for ANY decoder model. It derives the
per-op linear-GEMM shapes from a model config, takes the model's measured per-op
TFLOP/s (either measured here via the raw AMX kernel, or ingested from an in-SGLang
profile), and reports for each hot op:

  * eff_abs  = achieved / streamed-GEMM achievable ceiling (from the hw profile)
  * eff_rel  = achieved / the DONOR model running the identical kernel at the
               nearest shape (from peer-baselines.yaml)

Verdict per op requires eff_rel >= tol (a correctly-wired model reusing the same
kernel must match its peers); eff_abs is reported for context. A high eff_abs with
low eff_rel is the signature of a wiring bug (missing prepack / AVX-512 fallback /
NUMA), not a kernel limit.

Shapes are derived generically:
  qkv     : K=hidden,               N=(n_heads + 2*n_kv_heads)*head_dim
  o       : K=n_heads*head_dim,     N=hidden
  gate_up : K=hidden,               N=2*intermediate
  down    : K=intermediate,         N=hidden
  moe w1  : K=hidden,               N=2*moe_intermediate   (per expert)
  moe w2  : K=moe_intermediate,     N=hidden               (per expert)

Usage (ingest measured numbers; runs anywhere):
  peer_roofline.py --profile <hw.yaml> --baselines <peer-baselines.yaml> \
      --config <hf_config.json>  --measured-json <model_ops.json>

Usage (measure the raw AMX kernel on-node; establishes/refreshes the baseline):
  srun ... peer_roofline.py --profile ... --baselines ... --config ... --run --m 2048
"""

import argparse
import json
import math
import os
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM = os.path.normpath(os.path.join(HERE, "..", "..", "tools", "gemm_amx_bf16.py"))


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    # tolerate nested text_config (multimodal) and HF field aliases
    cfg = cfg.get("text_config", cfg)
    g = lambda *ks, d=None: next((cfg[k] for k in ks if k in cfg), d)
    hidden = g("hidden_size", "n_embd")
    n_heads = g("num_attention_heads", "n_head")
    n_kv = g("num_key_value_heads", d=n_heads)
    head_dim = g("head_dim", d=hidden // n_heads if hidden and n_heads else None)
    inter = g("intermediate_size", "ffn_dim")
    n_exp = g("num_experts", "num_local_experts", "n_routed_experts", d=0)
    moe_inter = g("moe_intermediate_size", "expert_intermediate_size", d=None)
    if (n_exp or 0) and not moe_inter:
        moe_inter = inter  # OLMoE-style: expert size == intermediate_size
    return {
        "hidden": hidden, "n_heads": n_heads, "n_kv": n_kv, "head_dim": head_dim,
        "intermediate": inter, "num_experts": n_exp or 0, "moe_intermediate": moe_inter,
    }


def derive_shapes(c, m):
    hd = c["head_dim"]
    shapes = [
        ("qkv", m, c["hidden"], (c["n_heads"] + 2 * c["n_kv"]) * hd),
        ("o", m, c["n_heads"] * hd, c["hidden"]),
        ("gate_up", m, c["hidden"], 2 * c["intermediate"]),
        ("down", m, c["intermediate"], c["hidden"]),
    ]
    if c["num_experts"] and c["moe_intermediate"]:
        shapes += [
            ("w1", m, c["hidden"], 2 * c["moe_intermediate"]),
            ("w2", m, c["moe_intermediate"], c["hidden"]),
        ]
    return [{"op": op, "M": M, "K": K, "N": N} for op, M, K, N in shapes]


def measure_raw(shape, threads, iters):
    cmd = [sys.executable, GEMM, "--M", str(shape["M"]), "--K", str(shape["K"]),
           "--N", str(shape["N"]), "--iters", str(iters), "--threads", str(threads),
           "--tag", shape["op"]]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()
    for line in reversed(out):
        if line.strip().startswith("{"):
            return json.loads(line)["tflops"]
    raise RuntimeError(f"no gemm output for {shape}")


def donor_role_map(baselines):
    """Flatten baselines into {role: [ {K,N,tflops,donor} ]} across all ops."""
    roles = {}
    for op_kind, entries in (baselines.get("per_op_efficiency") or {}).items():
        for e in entries:
            if not e.get("measured") or e.get("tflops") is None:
                continue
            role = e.get("layer", op_kind)
            roles.setdefault(role, []).append(e)
    return roles


def nearest_donor(role_entries, K, N):
    """Same-role donor with the nearest (K,N) in relative log-distance."""
    best, best_d = None, math.inf
    for e in role_entries:
        dk = math.log((e["K"] or 1) / K) if e.get("K") else 0.0
        dn = math.log((e["N"] or 1) / N) if e.get("N") else 0.0
        d = dk * dk + dn * dn
        if d < best_d:
            best, best_d = e, d
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--baselines", required=True)
    p.add_argument("--config", required=True, help="HF config.json of the new model")
    p.add_argument("--m", type=int, default=2048, help="prefill token block")
    p.add_argument("--run", action="store_true", help="measure the raw AMX kernel on-node")
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--measured-json", default="", help="{op: tflops} from in-SGLang profiling")
    p.add_argument("--model", default="new_model")
    args = p.parse_args()

    with open(args.profile) as f:
        prof = yaml.safe_load(f)
    with open(args.baselines) as f:
        baselines = yaml.safe_load(f)
    ceiling = (prof.get("achievable", {}) or {}).get("streamed_gemm_tflops_per_socket")
    tol = (baselines.get("tolerances", {}) or {})
    eff_rel_min = tol.get("eff_rel_min", 0.90)
    eff_abs_min = tol.get("eff_abs_min", 0.70)

    cfg = load_config(args.config)
    shapes = derive_shapes(cfg, args.m)
    roles = donor_role_map(baselines)

    measured = {}
    if args.measured_json:
        with open(args.measured_json) as f:
            measured = {k: float(v) for k, v in json.load(f).items()
                        if not k.startswith("_")}

    rows, overall = [], "PASS"
    for s in shapes:
        if s["op"] in measured:
            tfl = measured[s["op"]]
        elif args.run:
            tfl = measure_raw(s, args.threads, args.iters)
        else:
            rows.append({**s, "tflops": None, "note": "no measurement (use --run or --measured-json)"})
            overall = "INCOMPLETE"
            continue
        donor = nearest_donor(roles.get(s["op"], []), s["K"], s["N"])
        eff_abs = round(tfl / ceiling, 3) if ceiling else None
        eff_rel = round(tfl / donor["tflops"], 3) if donor else None
        verdict = "PASS"
        if eff_rel is not None and eff_rel < eff_rel_min:
            verdict, overall = "FAIL_REL(wiring)", "FAIL"
        elif eff_rel is None:
            verdict = "NO_DONOR"
        rows.append({
            **s, "tflops": round(tfl, 2), "eff_abs": eff_abs, "eff_rel": eff_rel,
            "donor": f"{donor['donor']} K{donor['K']}xN{donor['N']}={donor['tflops']}TF" if donor else None,
            "verdict": verdict,
        })

    print(json.dumps({
        "model": args.model,
        "hardware_profile": prof.get("part"),
        "streamed_ceiling_tflops": ceiling,
        "eff_abs_min": eff_abs_min, "eff_rel_min": eff_rel_min,
        "m_block": args.m,
        "ops": rows,
        "overall_verdict": overall,
    }, indent=2))


if __name__ == "__main__":
    main()
