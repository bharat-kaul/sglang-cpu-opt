#!/usr/bin/env python3
"""Roofline verdict for a CPU kernel, driven entirely by a hardware profile.

Reads a YAML hardware profile, computes the AMX compute ceiling and the memory
bandwidth ceiling, the ridge point, the kernel's arithmetic intensity, and the
achieved efficiency. Swapping the profile retargets every number.
"""
import argparse
import json
import sys

try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: pip install pyyaml")


def load_profile(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--achieved-tflops", type=float, required=True)
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--freq", type=float, default=0.0,
                   help="measured AMX all-core GHz; overrides profile value")
    p.add_argument("--cores", type=int, default=0,
                   help="override active core count (e.g. single socket)")
    p.add_argument("--sockets", type=int, default=0,
                   help="override active socket count for memory BW")
    p.add_argument("--achievable-tflops-ceiling", type=float, default=0.0,
                   help="measured achievable AMX compute ceiling (microkernel); "
                        "if set, this is the PRIMARY gate")
    p.add_argument("--use-profile-achievable", action="store_true",
                   help="pull the achievable ceiling from the profile's "
                        "achievable block, scaled by sockets_used (parameterized gate)")
    args = p.parse_args()

    prof = load_profile(args.profile)
    cores = args.cores or prof["cores_total"]
    sockets = args.sockets or prof["sockets"]

    # Parameterized achievable ceiling straight from the profile (Day-0 path).
    if args.use_profile_achievable and not args.achievable_tflops_ceiling:
        ach = prof.get("achievable", {})
        per_socket = ach.get("bf16_compute_tflops_per_socket", 0.0)
        args.achievable_tflops_ceiling = per_socket * sockets
    freq = args.freq or prof["amx_all_core_freq_ghz"]
    fpc = prof["amx_bf16_flops_per_cycle_per_core"]

    peak_compute = cores * fpc * freq * 1e9 / 1e12  # TFLOP/s
    peak_bw = (sockets * prof["mem_channels_per_socket"]
               * prof["mem_speed_mts"] * 1e6
               * prof["mem_bytes_per_transfer"]) / 1e9  # GB/s
    ridge = (peak_compute * 1e12) / (peak_bw * 1e9)     # FLOP/byte

    # Ideal DRAM traffic for a BF16 GEMM (perfect cache blocking): read A + read
    # B + write C, 2 bytes/elem. This is a lower bound on bytes => upper bound AI.
    M, N, K = args.M, args.N, args.K
    bytes_moved = 2 * (M * K + K * N + M * N)
    flop = 2.0 * M * N * K
    ai = flop / bytes_moved

    compute_bound = ai >= ridge
    theo_ceiling = peak_compute if compute_bound else (ai * peak_bw / 1e3)

    # ACHIEVABLE ceiling (microkernel-measured) is the primary gate when provided.
    ceiling = args.achievable_tflops_ceiling or theo_ceiling
    ceiling_kind = "achievable" if args.achievable_tflops_ceiling else "theoretical"
    eff = args.achieved_tflops / ceiling

    if eff > 1.0:
        verdict = "INVALID_CEILING (>100%: re-measure ceiling or freq)"
    elif eff >= 0.70:
        verdict = "PASS"
    elif eff >= 0.50:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    print(json.dumps({
        "part": prof["part"],
        "cores_used": cores,
        "sockets_used": sockets,
        "amx_freq_ghz": freq,
        "peak_compute_tflops": round(peak_compute, 1),
        "achievable_ceiling_tflops": round(args.achievable_tflops_ceiling, 1) or None,
        "peak_mem_bw_gbs": round(peak_bw, 1),
        "ridge_flop_per_byte": round(ridge, 1),
        "arithmetic_intensity": round(ai, 1),
        "regime": "compute-bound" if compute_bound else "memory-bound",
        "gated_against": ceiling_kind,
        "applicable_ceiling_tflops": round(ceiling, 1),
        "achieved_tflops": round(args.achieved_tflops, 1),
        "efficiency_pct": round(eff * 100, 1),
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
