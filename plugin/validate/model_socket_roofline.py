#!/usr/bin/env python3
"""Analytical DSv4 socket targets. No model execution or tensor materialization."""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import subprocess

from calibrate_socket_targets import LAYOUTS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COMPUTE_COMMIT = "1ac61ecacdc46ae4118bb2faa1109348961d8c86"
CALIBRATION_HASHES = {
    "emr": "213568112928ae8f3ce6a9e99a4f757c41b22d515e33e0370b95d68d013e1b92",
    "gnr": "718af274ac660a1e3425c1404fac45ad10f8eab45fc39d99e75eeaa44951c18d",
}
SCENARIOS = {
    "central": {"routing": "historical59", "compute_fraction": 0.75,
                "vector_fraction": 0.5, "launch_us": 5, "weight_passes": 1,
                "dequant_lane_ops": 4},
    "optimistic": {"routing": "historical59", "compute_fraction": 1.0,
                   "vector_fraction": 0.75, "launch_us": 1, "weight_passes": 1,
                   "dequant_lane_ops": 4},
    "cautious": {"routing": "uniform", "compute_fraction": 0.5,
                 "vector_fraction": 0.25, "launch_us": 20, "weight_passes": 2,
                 "dequant_lane_ops": 4},
}
ASSUMPTIONS = [
    "Conditional analytical predictions, not measured ceilings, acceptance thresholds or proof of optimality.",
    "Same 16x256-token prefill forwards and four B32 decode forwards as all 15 archived socket samples.",
    "EMR 2x32 cores share NUMA0; GNR 3x32 cores use NUMA0/1/2. Work scales by replicas, calibrated aggregate rates do not.",
    "Historical59 is a coverage-curve scenario fitted to old B32 routing, not a current-input measurement.",
    "Uniform routing is a stress scenario, not an upper bound on all possible routing distributions.",
    "Balanced token-per-expert occupancy; routed AMX work padded to M16. No padding of useful token counts.",
    "Native MXFP4/FP8 weights plus scales; current CPU wo_a is retained BF16. No widened whole-expert temporaries.",
    "AMX matrix throughput transfers the warm M512/K4096/N4096 calibration with the declared fraction; dense decode uses M32.",
    "Nominal clocks EMR1.9/GNR2.0 GHz, 64 FP32 FLOP/core/cycle and 16 vector lanes/core/cycle, scaled by declared utilization.",
    "Four SIMD lane operations per converted weight is an instruction-budget assumption, not a measured conversion rate.",
    "Fused SIMD dequant, AMX contraction and weight streaming are pipelined: their maximum, not their sum, limits a contraction.",
    "Separate op classes are serialized. No overlap is credited between replicas' duplicated byte counts or consecutive layers.",
    "FP32 compressor/MHC and attention budgets preserve FP32 math; softmax/exp/div use explicit approximate vector-operation budgets.",
    "Fused norms/RoPE/quant/MHC/route/SiLU use logical traffic and operation budgets; exp is not treated as one FLOP.",
    "Attention shares KV across heads and avoids intermediate score DRAM; assumes BF16 working KV after packed-cache decode.",
    "Active experts stream once or twice per forward, no cross-layer/forward cache reuse. Replica-local weights are not shared.",
    "Native call overhead is an assumed 1/5/20us per modeled call, not the current Python scheduler overhead.",
    "Target is phase-forward work. Loading, tokenization, sampling/network and current framework gaps are outside the target.",
    "Calibration is cross-run (Nov13 versus Nov12 model timing), turbo enabled; no frequency-pinned or steady-state claim.",
    "Scenarios are not confidence intervals. A scenario slower than observed is contradicted for that workload; do not clamp its ratio.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_calibration(platform):
    path = RESULTS / f"dsv4_socket_calibration_{platform}_20261113.json"
    if sha(path) != CALIBRATION_HASHES[platform]:
        raise ValueError(f"Calibration content changed: {path}")
    data = json.loads(path.read_text())
    return data["summary"]["summary"]


def distinct_experts(tokens, routing):
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("Positive integer token count required")
    if routing == "historical59":
        probability = 1 - (1 - 59 / 256) ** (1 / 32)
    elif routing == "uniform":
        probability = 6 / 256
    else:
        raise ValueError("Unknown routing assumption")
    return max(6, min(256, math.ceil(256 * (1 - (1 - probability) ** tokens) - 1e-10)))


def padded_assignments(tokens, distinct):
    if type(distinct) is not int or not 6 <= distinct <= min(256, tokens * 6):
        raise ValueError("Invalid distinct expert count")
    # Balanced occupancy is an assumption, not a reconstruction of current routing.
    low, remainder = divmod(tokens * 6, distinct)
    return ((distinct - remainder) * math.ceil(low / 16)
            + remainder * math.ceil((low + 1) / 16)) * 16


def capture(checkpoint, archives, measured):
    if sha(measured) != "b7d8cda692699bfad130775d6df308c036fc783467d89a75aebe600befaaf90a":
        raise ValueError("Unexpected archived measurement report")
    report = json.loads(measured.read_text())
    config = json.loads((checkpoint / "config.json").read_text())
    required = {"hidden_size": 4096, "num_hidden_layers": 43, "n_routed_experts": 256,
                "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
                "hc_mult": 4, "hc_sinkhorn_iters": 20, "head_dim": 512,
                "num_attention_heads": 64, "index_head_dim": 128, "index_n_heads": 64,
                "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8,
                "vocab_size": 129280, "num_hash_layers": 3, "n_shared_experts": 1,
                "sliding_window": 128, "expert_dtype": "fp4"}
    if any(config[key] != value for key, value in required.items()):
        raise ValueError("This analytical model supports only the declared DSv4 Flash config")
    if config["compress_ratios"] != [0, 0] + [4, 128] * 20 + [4, 0]:
        raise ValueError("Unexpected compression schedule")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    headers, header_hashes, tensors = {}, {}, {}
    for name, shard in index.items():
        if name.startswith("mtp."):
            continue
        if shard not in headers:
            with (checkpoint / shard).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                raw = stream.read(size)
            headers[shard] = json.loads(raw)
            header_hashes[shard] = hashlib.sha256(raw).hexdigest()
        entry = headers[shard][name]
        if ".experts." in name and ".experts.0." not in name:
            continue
        tensors[name] = {"shape": entry["shape"], "dtype": entry["dtype"],
                         "bytes": entry["data_offsets"][1] - entry["data_offsets"][0]}
    events = []
    for platform, root in archives.items():
        for rank in range(len(LAYOUTS[platform])):
            for repeat in range(3):
                path = root / platform / f"rank{rank}" / f"socket-b32-s128-r{repeat}-baseline.events.json"
                rows = json.loads(path.read_text())
                expected = ([("prefill", 256, [128, 128])] * 16
                            + [("decode", 32, [s] * 32) for s in range(129, 133)])
                if [(r["phase"], r["tokens"], r["seq_lens"]) for r in rows] != expected:
                    raise ValueError(f"Forward schedule mismatch: {path}")
                events.append({"path": str(path), "sha256": sha(path)})
    return {"config": config, "tensors": tensors, "checkpoint": str(checkpoint),
            "config_sha256": sha(checkpoint / "config.json"),
            "index_sha256": sha(checkpoint / "model.safetensors.index.json"),
            "header_sha256": header_hashes, "compute_commit": COMPUTE_COMMIT,
            "schedule_evidence": events,
            "measurement_report_sha256": sha(measured),
            "measured": {platform: {
                "job": report["runs"][platform]["job"],
                "plan_sha256": report["runs"][platform]["plan_sha256"],
                **{phase: report["runs"][platform]["points"]["b32-s128"][f"socket_{phase}_tokens_s"]
                   for phase in ("prefill", "decode")}} for platform in LAYOUTS},
            "schedule": {"prefill": {"forwards": 16, "tokens": 256, "requests": 2},
                         "decode": {"forwards": 4, "tokens": 32, "contexts": [129, 130, 131, 132]}}}


def phase_budget(inputs, platform, phase, scenario):
    if phase not in ("prefill", "decode") or platform not in LAYOUTS:
        raise ValueError("Unknown phase/platform")
    for key in ("compute_fraction", "vector_fraction", "launch_us", "weight_passes", "dequant_lane_ops"):
        value = scenario[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid scenario parameter: {key}")
    cfg, tensors = inputs["config"], inputs["tensors"]
    rates = read_calibration(platform)
    replicas = len(LAYOUTS[platform])
    bw = rates["read"]["median"]
    dense = rates["bf16_m512" if phase == "prefill" else "bf16_m32"]["median"]
    matrix = rates["bf16_m512"]["median"] * scenario["compute_fraction"]
    dense *= scenario["compute_fraction"]
    # Nominal-clock AVX-512 issue model; not measured all-core clocks or SIMD ceilings.
    clock = {"emr": 1.9e9, "gnr": 2.0e9}[platform]
    fp32 = replicas * 32 * 64 * clock * scenario["vector_fraction"]
    lane_ops = fp32 / 4  # one 16-lane vector instruction/core/cycle before utilization
    groups = defaultdict(lambda: {"seconds": 0.0, "flops": 0.0, "logical_bytes": 0.0})
    m = 256 if phase == "prefill" else 32
    forwards = 16 if phase == "prefill" else 4
    distinct = distinct_experts(m, scenario["routing"])

    def add(name, flops=0, byte_count=0, rate=None, vector_ops=0, calls=1, pipeline=False):
        work_time = max(flops * replicas / (rate or fp32), byte_count * replicas / bw)
        vector_time = vector_ops * replicas / lane_ops
        # A fused AMX target can pipeline SIMD conversion with tile compute and streaming.
        seconds = (max(work_time, vector_time) if pipeline else work_time + vector_time)
        seconds += calls * scenario["launch_us"] / 1e6
        groups[name]["seconds"] += seconds * forwards
        groups[name]["flops"] += flops * replicas * forwards
        groups[name]["logical_bytes"] += byte_count * replicas * forwards

    for layer in range(43):
        prefix = f"layers.{layer}."
        for name, tensor in tensors.items():
            if not name.startswith(prefix) or ".experts." in name:
                continue
            shape = tensor["shape"]
            if len(shape) != 2 or not (name.endswith(".weight") or name.endswith("_fn")):
                continue
            n, k = shape
            weight = tensor["bytes"]
            scale = tensors.get(name.removesuffix(".weight") + ".scale")
            if scale:
                weight += scale["bytes"]
            compute_rate, category = dense, "dense_and_shared"
            if "compressor." in name:
                compute_rate, category = fp32, "compressor_projection_fp32"
            elif name.endswith("_fn"):
                compute_rate, category = fp32, "mhc_projection_fp32"
            if name.endswith("wo_a.weight"):
                weight = n * k * 2  # current CPU runtime retains logical BF16 wo_a
            input_width = k * 8 if name.endswith("wo_a.weight") else k
            activations = m * (input_width + n) * (4 if compute_rate == fp32 else 2)
            dequant_ops = (n * k * scenario["dequant_lane_ops"]
                           if tensor["dtype"] == "F8_E4M3" and "wo_a" not in name else 0)
            add(category, 2 * m * n * k, weight + activations, compute_rate, dequant_ops,
                pipeline=compute_rate != fp32)

        per_expert = sum(tensors[prefix + f"ffn.experts.0.{w}.{suffix}"]["bytes"]
                         for w in ("w1", "w2", "w3") for suffix in ("weight", "scale"))
        if per_expert != 13369344:
            raise ValueError("Native MXFP4 payload differs from the declared model")
        elems = 3 * 4096 * 2048
        add("routed_mxfp4", 2 * padded_assignments(m, distinct) * elems,
            distinct * per_expert * scenario["weight_passes"] + m * 6 * (2 * 4096 + 2048) * 2,
            matrix, distinct * elems * scenario["dequant_lane_ops"] * scenario["weight_passes"],
            calls=1, pipeline=True)
        ratio = cfg["compress_ratios"][layer]
        if phase == "prefill":
            positions = sum(min(s, 128) + (s // ratio if ratio else 0) for s in range(1, 129)) / 128
        else:
            positions = sum(128 + (s // ratio if ratio else 0) for s in range(129, 133)) / 4
        add("attention_fp32", 4 * m * 64 * 512 * positions,
            m * (positions * 512 * 2 + 2 * 64 * 512 * 2 + 512 * 2),
            vector_ops=m * 64 * positions * 16)
        if ratio == 4:
            context = 16.25 if phase == "prefill" else 32.25
            add("indexer", 2 * m * 64 * 128 * context,
                m * (context * 128 * 2 + 64 * 128 * 2 + context * 4),
                dense, vector_ops=m * (64 * context * 3 + 64 * 128 * 9))
        if ratio:
            # r128 pools once/request in prefill and never in decode contexts129..132.
            completed = (m / ratio if phase == "prefill" else m / 4 if ratio == 4 else 0)
            dims = 512 + (128 if ratio == 4 else 0)
            window = 8 if ratio == 4 else 128
            add("compressor_pool_state", byte_count=m * dims * (2 if ratio == 4 else 1) * 8
                + completed * window * dims * 8,
                vector_ops=completed * window * dims * 20)
        # Fused native estimates: norms, rope/quant, MHC mix/post/Sinkhorn, route and SiLU.
        add("vector_fusions", byte_count=m * (2 * (3 * 4 * 4096 + 4096) * 2
            + 2 * 4096 * 4 + 2 * (64 * 512 + 512) * 2),
            vector_ops=m * (2 * (4 * 4 * 4096 * 2 + 4 * 4 * 20 * 5 + 4 * 4096 * 4)
                           + 64 * 64 * 12 + 256 * 32 + 6 * 2048 * 24), calls=8)
    # Only the last token of each completed request needs vocab logits.
    requests = 2 if phase == "prefill" else 32
    add("head_embedding", 2 * requests * 4096 * 129280,
        tensors["head.weight"]["bytes"] + m * 4096 * 2 + requests * 129280 * 4,
        dense, calls=2)
    add("mhc_head_fp32", 2 * m * 4 * 16384,
        tensors["hc_head_fn"]["bytes"] + m * (4 * 4096 * 4 + 4096 * 2),
        fp32, vector_ops=m * 4 * 4096 * 8, calls=2)
    seconds = sum(row["seconds"] for row in groups.values())
    tokens = replicas * m * forwards
    return {"tokens_per_second": tokens / seconds, "phase_seconds": seconds,
            "socket_tokens": tokens, "distinct_experts_per_forward": distinct,
            "groups": dict(groups)}


def evaluate(inputs):
    expected_schedule = {"prefill": {"forwards": 16, "tokens": 256, "requests": 2},
                         "decode": {"forwards": 4, "tokens": 32, "contexts": [129, 130, 131, 132]}}
    if inputs["schedule"] != expected_schedule or inputs["compute_commit"] != COMPUTE_COMMIT:
        raise ValueError("Input source/schedule mismatch")
    result = {"status": "ANALYTICAL_ESTIMATE_NOT_MEASURED_CEILING",
              "compute_commit": inputs["compute_commit"],
              "calibration_sha256": CALIBRATION_HASHES, "scenarios": SCENARIOS,
              "assumptions": ASSUMPTIONS, "model_source_sha256": sha(__file__),
              "input_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
              "results": {}, "one_parameter_sensitivity": {}, "comparison": {}}
    for platform, phase in itertools.product(LAYOUTS, ("prefill", "decode")):
        point = f"{platform}_{phase}"
        result["results"][point] = {
            name: phase_budget(inputs, platform, phase, params) for name, params in SCENARIOS.items()}
        changes = {"routing": ["uniform"], "weight_passes": [2], "dequant_lane_ops": [2, 8],
                   "compute_fraction": [0.5, 1], "vector_fraction": [0.25, 0.75], "launch_us": [1, 20]}
        result["one_parameter_sensitivity"][point] = {
            f"{key}={value}": phase_budget(inputs, platform, phase, {**SCENARIOS["central"], key: value})[
                "tokens_per_second"] for key, values in changes.items() for value in values}
        measured = inputs["measured"][platform][phase]["median"]
        result["comparison"][point] = {
            name: {"measured_over_estimate": measured / row["tokens_per_second"],
                   "disposition": ("CONTRADICTED_AS_CEILING_BY_EXISTING_MEASUREMENT"
                                   if measured > row["tokens_per_second"]
                                   else "CONDITIONAL_PREDICTION_NOT_CERTIFIED")}
            for name, row in result["results"][point].items()}
    return result


def plot(inputs, result, stem):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760">',
           '<rect width="1200" height="760" fill="white"/>',
           '<g font-family="sans-serif" fill="#17324d">']

    def text(x, y, value, size=13, anchor="start", color="#17324d"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{value}</text>')

    text(35, 35, "DSv4: measured performance vs analytical roofline estimates", 23)
    text(35, 62, "B32 / S128 per replica; five output tokens; fixed32 threads. Higher is better.", 15)
    svg.append('<rect x="35" y="78" width="1130" height="49" fill="#fff3d6"/>')
    text(48, 98, "Cross-experiment baseline: dummy-weight proxy. EMR/GNR bars: matched real-weight measurements.", 14)
    text(48, 119, "Targets are conditional analytical predictions, not measured ceilings or certificates of optimality.", 14)
    text(35, 156, "Bars = measured / historical proxy", 14)
    text(430, 156, "Orange dash = central scenario", 14, color="#b05c00")
    text(810, 156, "Orange triangle = optimistic scenario", 14, color="#b05c00")
    colors = ["#8994a5", "#2473ad", "#38835b"]
    for panel, (phase, historical) in enumerate(zip(("prefill", "decode"), (79.8, 36.2))):
        left, bottom, height = 85 + panel * 590, 465, 230
        top_rate = 600 if phase == "prefill" else 240
        text(left + 220, 202, phase.capitalize() + " tokens/s", 19, "middle")
        for tick in range(5):
            value = tick * top_rate / 4
            y = bottom - height * tick / 4
            svg.append(f'<path d="M{left} {y}h460" stroke="#e5eaf0"/>')
            text(left - 12, y + 4, f"{value:.0f}", 12, "end")
        values = [historical] + [inputs["measured"][p][phase]["median"] for p in LAYOUTS]
        for i, value in enumerate(values):
            x = left + 35 + i * 150
            h = value / top_rate * height
            svg.append(f'<rect x="{x}" y="{bottom-h}" width="88" height="{h}" fill="{colors[i]}"/>')
            text(x + 44, bottom - h / 2 + 5, f"{value:.1f}", 14, "middle", "white")
            text(x + 44, 489, ["Baseline", "EMR", "GNR"][i], 14, "middle")
            text(x + 44, 510, ["Historical proxy", "2 replicas", "3 replicas"][i], 12, "middle")
            if i:
                point = result["results"][f"{('emr', 'gnr')[i-1]}_{phase}"]
                central, optimistic = [point[s]["tokens_per_second"] for s in ("central", "optimistic")]
                y, upper = [bottom - v / top_rate * height for v in (central, optimistic)]
                svg.append(f'<path d="M{x-10} {y}h108" stroke="#b05c00" stroke-width="2" stroke-dasharray="6,3"/>')
                svg.append(f'<path d="M{x+44} {upper-7}l-6 12h12z" fill="#b05c00"/>')
                text(x + 44, y - 9, f"{central:.0f}", 13, "middle", "#944c00")
                text(x + 44, upper - 14, f"{optimistic:.0f}", 13, "middle", "#944c00")
    notes = [
        "Bars: existing measurements, jobs EMR402060 / GNR402075; baseline: dummy-weight 4L/8L extrapolation.",
        "Not a controlled baseline speedup comparison. EMR uses 64 cores; GNR uses 96 of 128 cores.",
        "Targets are conditional predictions, not measured ceilings, confidence intervals or optimality certificates.",
        "Source-derived native bytes and observed forward shapes + hardware calibration402525/402526;",
        "assumed routing, SIMD conversion/issue rate, fusion and utilization. No new full-model runs.",
        "A cautious stress scenario is contradicted by existing throughput; retained in the detailed report.",
    ]
    for i, line in enumerate(notes):
        text(35, 557 + i * 30, line, 13)
    svg.append("</g></svg>")
    stem.with_suffix(".svg").write_text("\n".join(svg) + "\n")
    subprocess.run(["rsvg-convert", "-z", "2", "-o", str(stem.with_suffix(".png")),
                    str(stem.with_suffix(".svg"))], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, help="Checkpoint headers only; never loads weights")
    parser.add_argument("--emr-archive", type=Path)
    parser.add_argument("--gnr-archive", type=Path)
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--inputs", type=Path, default=RESULTS / "dsv4_socket_roofline_inputs.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "dsv4_socket_roofline_analytical.json")
    parser.add_argument("--plot", type=Path, help="Output chart stem, without extension")
    args = parser.parse_args()
    if args.capture:
        if not args.emr_archive or not args.gnr_archive or not args.measured:
            parser.error("Capture requires both archived run directories and the measurement report")
        inputs = capture(args.capture, {"emr": args.emr_archive, "gnr": args.gnr_archive}, args.measured)
    else:
        inputs = json.loads(args.inputs.read_text())
    output = evaluate(inputs)
    if args.capture:
        args.inputs.write_text(json.dumps(inputs, indent=2, allow_nan=False) + "\n")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(inputs, output, args.plot)
    for point, scenarios in output["results"].items():
        print(point, {name: round(row["tokens_per_second"], 1) for name, row in scenarios.items()})


if __name__ == "__main__":
    main()
