#!/usr/bin/env python3
"""Reconcile the pinned short EMR/GNR runs and publish a bounded comparison."""

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess

from screen_serving_opportunities import digest, write_json


RUNS = {
    "emr": {
        "root": "/scratch/bkaul/dsv4-socket-walltime/main-short-20261112-I8TNCU",
        "job": "402060",
        "plan": "e265eb8a15286134f8386c85dfcce9520439ab122d9ce6aa860d501111d727e9",
    },
    "gnr": {
        "root": "/scratch/bkaul/dsv4-socket-walltime/main-short-retry-20261112-Dg8eC6",
        "job": "402075",
        "plan": "fe015de5ccf911da8bbf4c43c983a06bd5a2123e9a9d5f4f41e84aa2ee3d62c6",
    },
}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def distribution(values):
    require(bool(values) and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                                for v in values), "Invalid timing/throughput samples")
    return {"samples": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def load_run(platform, spec):
    root = Path(spec["root"])
    require(digest(root / "plan.json") == spec["plan"], "Unexpected plan identity")
    p = read(root / "plan.json")
    require(digest(root / "source-sha256.txt") == p["source_manifest_sha256"],
            "Changed source manifest")
    subprocess.run(["sha256sum", "--quiet", "-c", str(root / "source-sha256.txt")],
                   cwd=root / "source", check=True)
    driver_path = root / "source/plugin/validate/compare_socket_walltime.py"
    require(digest(driver_path) == p["driver_sha256"], "Changed frozen driver")
    module_spec = importlib.util.spec_from_file_location(f"socket_{platform}", driver_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("Cannot load frozen driver")
    driver = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(driver)
    driver.plan(root)
    require(p["baseline_only"] is True, "Expected native-only run")
    require((root / f"{platform}-job.txt").read_text().strip() == spec["job"],
            "Unexpected submitted job")
    require((root / f"{platform}-exit-code").read_text().strip() == "0",
            "Launcher did not exit successfully")
    out = root / platform
    require(read(out / "complete.json") == {"job": spec["job"], "plan_sha256": spec["plan"]},
            "Missing or incorrect completion identity")
    identities, memory, libraries = {}, [], {}
    for rank, (node, first) in enumerate(driver.LAYOUTS[platform]):
        folder = out / f"rank{rank}"
        identity = read(folder / "identity.json")
        require(identity["job"] == spec["job"] and identity["plan_sha256"] == spec["plan"]
                and identity["rank"] == rank and identity["platform"] == platform
                and identity["numa"] == node and identity["cpus"] == list(range(first, first + 32))
                and identity["environment"] == {
                    **p["environment"], "SGLANG_CPU_OMP_THREADS_BIND": f"{first}-{first + 31}"},
                "Replica identity/configuration mismatch")
        require(read(folder / "complete.json") == identity, "Replica incomplete")
        native = read(folder / "native-builds.json")
        require(bool(native) and all(digest(name) == sha for name, sha in native.items()),
                "Missing or changed native builds")
        libraries[rank] = native
        m = read(folder / "load-memory.json")["scheduler"]
        driver.validate_locality(m, node)
        memory.append((rank, m))
        identities[rank] = identity
    require(len({v["pid"] for v in identities.values()}) == len(identities),
            "Duplicate replica process")
    require(len({v["node"] for v in identities.values()}) == 1, "Mixed replica hosts")

    results, signatures, outputs = [], {}, {}
    for case in driver.cases(platform, True):
        packets, events = [], []
        for rank in case["ranks"]:
            folder = out / f"rank{rank}"
            packet = read(folder / f"{case['case']}.json")
            require(packet["identity"] == identities[rank], "Case identity changed")
            ep = folder / f"{case['case']}.events.json"
            require(digest(ep) == packet["events_sha256"], "Changed case events")
            packets.append(packet)
            events.append(read(ep))
            memory.append((rank, packet["memory"]))
        result = driver.aggregate(packets, events, case)
        require(json.loads(json.dumps(result)) == read(out / f"{case['case']}.aggregate.json"),
                "Persisted aggregation differs from raw evidence")
        signature = [
            (s["anchors"], s["schedule"], s["inference_pid"],
             [r["output_ids"] for r in packet["responses"]])
            for s, packet in zip(result["samples"], packets)
        ]
        key = case["group"], case["batch"], case["length"]
        require(key not in signatures or signatures[key] == signature,
                "Native repetitions changed input/state/schedule/output")
        signatures[key] = signature
        point = f"b{case['batch']}-s{case['length']}"
        for packet, sample in zip(packets, result["samples"]):
            signature = {"anchors": sample["anchors"], "schedule": sample["schedule"],
                         "output_ids": [r["output_ids"] for r in packet["responses"]]}
            require(point not in outputs or outputs[point] == signature,
                    "Inputs or generated work differ across isolated/socket replicas")
            outputs[point] = signature
        results.append(result)
    require({f.stem.removesuffix(".aggregate") for f in out.glob("*.aggregate.json")}
            == {c["case"] for c in results}, "Unexpected aggregate coordinates")

    if platform == "gnr":
        cache = read(root / "remote-cache.json")
        driver.validate_cache_policy(cache["policy"])
        require(cache["job"] == spec["job"] and cache["plan_sha256"] == spec["plan"]
                and cache["node"] == identities[0]["node"]
                and set(cache["files"]) == set(p["checkpoint_files"])
                and all(cache["files"][n]["bytes"] == row["size"]
                        for n, row in p["checkpoint_files"].items()),
                "Remote-cache evidence mismatch")
    memory_rows = []
    for rank, row in memory:
        node = driver.LAYOUTS[platform][rank][0]
        driver.validate_locality(row, node)
        memory_rows.append({
            "rank": rank, "rss_gib": row["VmRSS_kib"] / 1048576,
            "hwm_gib": row["VmHWM_kib"] / 1048576, "swap_kib": row["VmSwap_kib"],
            "local_percent": 100 * row["numa_kib"][str(node)] / sum(row["numa_kib"].values()),
        })
    points = {}
    for batch, length in driver.PERF_POINTS:
        rows = [r for r in results if r["stage"] == "measured"
                and (r["batch"], r["length"]) == (batch, length)]
        isolated, socket = ([r for r in rows if (r["group"] == "socket") == flag]
                            for flag in (False, True))
        require(len(isolated) == len(identities) and len(socket) == driver.ROUNDS,
                "Incomplete measured sample counts")
        points[f"b{batch}-s{length}"] = {
            "isolated_request_s": distribution([r["request_wall_s"] for r in isolated]),
            "socket_request_s": distribution([r["request_wall_s"] for r in socket]),
            "socket_generated_tokens_s": distribution([r["generated_tokens_s"] for r in socket]),
            "socket_prefill_tokens_s": distribution([r["prefill_tokens_s"] for r in socket]),
            "socket_decode_tokens_s": distribution([r["decode_tokens_s"] for r in socket]),
            "isolated_prefill_s": distribution([r["samples"][0]["prefill_forward_s"] for r in isolated]),
            "isolated_decode_s": distribution([r["samples"][0]["decode_forward_s"] for r in isolated]),
        }
    functions = {n.name: ast.dump(n) for n in ast.parse(driver_path.read_text()).body
                 if isinstance(n, ast.FunctionDef)}
    return {
        "job": spec["job"], "root": str(root), "plan_sha256": spec["plan"],
        "source_manifest_sha256": p["source_manifest_sha256"],
        "groups": len(results), "replicas": len(identities), "identities": identities,
        "native_builds": libraries, "memory": memory_rows, "points": points,
    }, p, functions, outputs


def summarize(runs=RUNS):
    loaded = {platform: load_run(platform, runs[platform]) for platform in ("emr", "gnr")}
    emr, gnr = loaded["emr"], loaded["gnr"]
    for field in ("base_commit", "compute_sources", "installed_sources", "environment",
                  "inputs_sha256", "sampling", "checkpoint_files", "model_config_sha256",
                  "checkpoint_index_sha256", "cases"):
        require(emr[1][field] == gnr[1][field], f"Cross-platform mismatch: {field}")
    for name in ("worker", "aggregate", "run_case", "cases", "memory", "validate_locality"):
        require(emr[2][name] == gnr[2][name], f"Measurement function differs: {name}")
    runtime = {(row["torch"], row["sglang"], row["common_ops_sha256"])
               for run, _, _, _ in loaded.values() for row in run["identities"].values()}
    require(len(runtime) == 1, "Different cross-platform runtime/native dependency")
    require(emr[3] == gnr[3], "Cross-platform inputs, schedules or output tokens differ")
    speedup = {}
    for point, e in emr[0]["points"].items():
        g = gnr[0]["points"][point]
        speedup[point] = {
            "isolated_request": e["isolated_request_s"]["median"] / g["isolated_request_s"]["median"],
            "socket_request_throughput": g["socket_generated_tokens_s"]["median"]
                                         / e["socket_generated_tokens_s"]["median"],
            "socket_prefill_throughput": g["socket_prefill_tokens_s"]["median"]
                                         / e["socket_prefill_tokens_s"]["median"],
            "socket_decode_throughput": g["socket_decode_tokens_s"]["median"]
                                        / e["socket_decode_tokens_s"]["median"],
        }
    return {
        "main_commit": emr[1]["base_commit"], "aggregator_sha256": digest(__file__),
        "runs": {k: v[0] for k, v in loaded.items()}, "gnr_over_emr": speedup,
        "scope": "Short real-weight native-only fixed32 probe, five generated tokens. "
                 "Ratios of platform medians, not paired-trial ratios or confidence intervals. "
                 "EMR two isolated processes / two concurrent replicas; GNR three / three. "
                 "Three socket rounds reuse those resident processes. No tuned-platform-optimum, "
                 "steady-state decode or universal scaling claim. GNR socket samples are faster "
                 "than its earlier isolated samples; the mechanism is not measured.",
    }


def publish(output):
    require(not output.exists(), "Refusing to overwrite comparison")
    result = summarize()
    write_json(output, result)
    return result


def chart(result):
    colors = ("#92506b", "#2473ad", "#38835b", "#c08020")
    metrics = ("isolated_request", "socket_request_throughput",
               "socket_prefill_throughput", "socket_decode_throughput")
    labels = ("Single-replica request", "Socket request throughput",
              "Socket prefill throughput", "Socket decode throughput")
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="490" viewBox="0 0 1100 490">',
        '<rect width="1100" height="490" fill="white"/>',
        '<g font-family="sans-serif" fill="#17324d">',
        '<text x="45" y="32" font-size="20" font-weight="bold">DSv4: measured GNR / EMR performance</text>',
        '<text x="45" y="55" font-size="13">Above 1.0x favors GNR. Short real-weight probe; no C4 candidate.</text>',
    ]
    for i, (color, label) in enumerate(zip(colors, labels)):
        x = 45 + i * 265
        svg.extend([f'<rect x="{x}" y="75" width="12" height="12" fill="{color}"/>',
                    f'<text x="{x + 18}" y="86" font-size="12">{label}</text>'])
    for tick in (0, 0.5, 1, 1.5, 2):
        y = 385 - tick * 125
        svg.extend([
            f'<line x1="65" y1="{y}" x2="1060" y2="{y}" stroke="{("#526577" if tick == 1 else "#e5eaf0")}" stroke-dasharray="5 4"/>',
            f'<text x="55" y="{y + 4}" text-anchor="end" font-size="12">{tick:.1f}x</text>',
        ])
    for index, (point, ratios) in enumerate(result["gnr_over_emr"].items()):
        x0 = 120 + 330 * index
        for offset, (metric, color) in enumerate(zip(metrics, colors)):
            value = ratios[metric]
            x, height = x0 + offset * 53, value * 125
            svg.extend([
                f'<rect x="{x}" y="{385 - height:.2f}" width="40" height="{height:.2f}" fill="{color}"/>',
                f'<text x="{x + 20}" y="{377 - height:.2f}" text-anchor="middle" font-size="12">{value:.2f}x</text>',
            ])
        svg.append(f'<text x="{x0 + 100}" y="410" text-anchor="middle" font-size="14">{point.upper().replace("-", " / ")}</text>')
    svg.extend([
        '<text x="45" y="445" font-size="12">Fixed32 per replica. Socket: EMR 2 replicas / 64 cores; GNR 3 replicas / 96 cores.</text>',
        '<text x="45" y="465" font-size="12">Ratios of medians. Five output tokens (four decode steps), not steady state. Jobs 402060 / 402075.</text>',
        '</g></svg>',
    ])
    return "\n".join(svg) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    result = publish(args.output)
    if args.svg:
        args.svg.write_text(chart(result))
    print(json.dumps(result["gnr_over_emr"], indent=2))
