#!/usr/bin/env python3
"""Generic peer-relative roofline validator (model-agnostic).

Implements the `peer-relative-roofline` skill for ANY decoder model. It derives the
per-op linear-GEMM shapes from a model config, takes the model's measured per-op
TFLOP/s (either measured here via the raw AMX kernel, or ingested from an in-SGLang
profile), and reports for each hot op:

  * eff_abs  = achieved / streamed-GEMM achievable ceiling (from the hw profile)
  * eff_rel  = achieved / the DONOR model running the identical kernel at the
               identical M/K/N shape (from peer-baselines.yaml)

Verdict per op requires BOTH absolute and relative thresholds. Missing/invalid
evidence exits nonzero. This is a raw-GEMM report, NOT a full model certificate;
enablement_gate.py separately enforces reviewed identity, coverage and all phases.
A high eff_abs with
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
import os
import subprocess
import sys

from enablement_gate import GateError, load_json, number, parse_json, require

try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM = os.path.normpath(os.path.join(HERE, "..", "..", "tools", "gemm_amx_bf16.py"))


class StrictLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader, node):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        require(isinstance(key, str), "YAML keys must be strings")
        require(key not in mapping, f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node)
    return mapping


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def load_config(path):
    cfg = load_json(path)
    require(type(cfg) is dict, "model config must be an object")
    # tolerate nested text_config (multimodal) and HF field aliases
    cfg = cfg.get("text_config", cfg)
    require(type(cfg) is dict, "text_config must be an object")
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
    for name, value in (("hidden", hidden), ("n_heads", n_heads), ("n_kv", n_kv),
                        ("head_dim", head_dim), ("intermediate", inter)):
        require(type(value) is int and value > 0, f"config {name} must be a positive integer")
    require(type(n_exp) is int and n_exp >= 0, "num_experts must be a nonnegative integer")
    if n_exp:
        require(type(moe_inter) is int and moe_inter > 0, "moe_intermediate must be positive")
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
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    for line in reversed(out):
        if line.strip().startswith("{"):
            return number(parse_json(line)["tflops"], "benchmark tflops", positive=True)
    raise RuntimeError(f"no gemm output for {shape}")


def donor_role_map(baselines):
    """Flatten baselines into {role: [ {K,N,tflops,donor} ]} across all ops."""
    roles = {}
    per_op = baselines.get("per_op_efficiency", {})
    require(type(per_op) is dict, "per_op_efficiency must be an object")
    for op_kind, entries in per_op.items():
        require(isinstance(entries, list), f"{op_kind}: donor entries must be a list")
        for e in entries:
            require(type(e) is dict, "donor entry must be an object")
            if e.get("measured") is not True or e.get("tflops") is None:
                continue
            role = e.get("layer", op_kind)
            require(isinstance(role, str) and role.strip(), "invalid donor role")
            roles.setdefault(role, []).append(e)
    return roles


def matched_donor(role_entries, M, K, N):
    """Never certify against a nearest-but-different shape or a low-fidelity proxy."""
    matches = []
    for e in role_entries:
        if any(type(e.get(k)) is not int or e[k] != v for k, v in (("M", M), ("K", K), ("N", N))):
            continue
        if e.get("fidelity") == "proxy":
            continue
        number(e["tflops"], "donor tflops", positive=True)
        require(isinstance(e.get("donor"), str) and e["donor"].strip(), "missing donor identity")
        matches.append(e)
    return max(matches, key=lambda e: e["tflops"]) if matches else None


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
    require(args.m > 0 and args.threads > 0 and args.iters > 0, "m/threads/iters must be positive")

    with open(args.profile) as f:
        prof = yaml.load(f, Loader=StrictLoader)
    with open(args.baselines) as f:
        baselines = yaml.load(f, Loader=StrictLoader)
    require(type(prof) is dict and type(prof.get("achievable")) is dict, "missing achievable profile")
    require(type(baselines) is dict, "baselines must be an object")
    ceiling = (prof.get("achievable", {}) or {}).get("streamed_gemm_tflops_per_socket")
    tol = baselines.get("tolerances", {})
    require(type(tol) is dict, "tolerances must be an object")
    eff_rel_min = tol.get("eff_rel_min", 0.90)
    eff_abs_min = tol.get("eff_abs_min", 0.70)
    number(ceiling, "streamed ceiling", positive=True)
    require(0.90 <= number(eff_rel_min, "eff_rel_min") <= 1, "eff_rel_min must be in [0.90,1]")
    require(0.70 <= number(eff_abs_min, "eff_abs_min") <= 1, "eff_abs_min must be in [0.70,1]")

    cfg = load_config(args.config)
    shapes = derive_shapes(cfg, args.m)
    roles = donor_role_map(baselines)

    measured = {}
    if args.measured_json:
        raw = load_json(args.measured_json)
        require(type(raw) is dict, "measurements must be an object")
        measured = {k: number(v, f"{k} tflops", positive=True) for k, v in raw.items() if not k.startswith("_")}
        require(set(measured) <= {s["op"] for s in shapes}, "unexpected measured operation")

    rows, overall = [], "PASS"
    if prof.get("achievable", {}).get("streamed_gemm_estimated") is not False:
        overall = "INCOMPLETE"
    for s in shapes:
        if s["op"] in measured:
            tfl = measured[s["op"]]
        elif args.run:
            tfl = measure_raw(s, args.threads, args.iters)
        else:
            rows.append({**s, "tflops": None, "note": "no measurement (use --run or --measured-json)"})
            overall = "INCOMPLETE"
            continue
        donor = matched_donor(roles.get(s["op"], []), s["M"], s["K"], s["N"])
        eff_abs = number(tfl / ceiling, "eff_abs", positive=True)
        eff_rel = number(tfl / donor["tflops"], "eff_rel", positive=True) if donor else None
        verdict = "PASS"
        if eff_abs > 1:
            verdict, overall = "TARGET_REVIEW", "FAIL"
        elif eff_abs < eff_abs_min:
            verdict, overall = "FAIL_ABS", "FAIL"
        elif eff_rel is not None and eff_rel < eff_rel_min:
            verdict, overall = "FAIL_REL(wiring)", "FAIL"
        elif eff_rel is None:
            verdict, overall = "NO_DONOR", "FAIL"
        rows.append({
            **s, "tflops": tfl, "eff_abs": eff_abs, "eff_rel": eff_rel,
            "donor": f"{donor['donor']} K{donor['K']}xN{donor['N']}={donor['tflops']}TF" if donor else None,
            "verdict": verdict,
        })

    print(json.dumps({
        "model": args.model,
        "scope": "raw_gemm_thresholds_only_not_enablement_certificate",
        "hardware_profile": prof.get("part"),
        "streamed_ceiling_tflops": ceiling,
        "eff_abs_min": eff_abs_min, "eff_rel_min": eff_rel_min,
        "m_block": args.m,
        "ops": rows,
        "overall_verdict": overall,
    }, indent=2, allow_nan=False))
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GateError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, subprocess.SubprocessError) as error:
        print(json.dumps({"overall_verdict": "INVALID", "reason": str(error)}, allow_nan=False), file=sys.stderr)
        sys.exit(2)
