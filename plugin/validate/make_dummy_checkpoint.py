#!/usr/bin/env python3
"""Build a shared, unquantized oracle-only checkpoint from a reviewed global inventory."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from enablement_gate import GateError, digest, exact, load_json, parse_json, pinned_json, require, sha, text


DTYPES = {
    "float32": (torch.float32, 4, "F32"),
    "float16": (torch.float16, 2, "F16"),
    "bfloat16": (torch.bfloat16, 2, "BF16"),
}


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_spec(spec: dict, config_bytes: bytes) -> list[dict]:
    exact(spec, {"schema_version", "config_sha256", "source_revision", "tensors"}, "tensor spec")
    require(type(spec["schema_version"]) is int and spec["schema_version"] == 1, "unsupported spec version")
    require(sha(spec["config_sha256"], "config_sha256") == digest(config_bytes), "config digest mismatch")
    text(spec["source_revision"], "source_revision")
    tensors = spec["tensors"]
    require(isinstance(tensors, list) and tensors, "empty tensor inventory")
    names = set()
    for item in tensors:
        exact(item, {"name", "shape", "dtype", "fill"}, "tensor")
        name = text(item["name"], "tensor name")
        require(name not in names, f"duplicate tensor: {name}")
        names.add(name)
        require(isinstance(item["shape"], list) and all(type(d) is int and d > 0 for d in item["shape"]), "invalid global shape")
        require(isinstance(item["dtype"], str) and item["dtype"] in DTYPES, "only unquantized float32/float16/bfloat16 fixtures supported")
        require(item["fill"] in ("uniform", "ones", "zeros"), "invalid tensor fill")
    return sorted(tensors, key=lambda item: item["name"])


def make_tensor(item: dict, seed: int) -> torch.Tensor:
    tensor = torch.empty(item["shape"], dtype=DTYPES[item["dtype"]][0], device="cpu")
    if item["fill"] != "uniform":
        tensor.fill_(1 if item["fill"] == "ones" else 0)
        return tensor
    name_seed = int(digest(f"tp-oracle-v1:{seed}:{item['name']}".encode())[:8], 16)
    rng = np.random.RandomState(name_seed)
    flat = tensor.view(-1)
    # Keep generator temporaries bounded; changing shard packing never changes logical values.
    for start in range(0, flat.numel(), 262144):
        count = min(262144, flat.numel() - start)
        values = rng.uniform(-1e-3, 1e-3, size=count).astype("float32")
        flat[start:start + count].copy_(torch.from_numpy(values))
    return tensor


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def create(config_path: Path, spec_path: Path, spec_hash: str, output: Path,
           seed: int, max_bytes: int, shard_bytes: int) -> str:
    require(not output.exists(), "output already exists; never overwrite an oracle")
    require(max_bytes > 0 and shard_bytes > 0, "byte budgets must be positive")
    require(0 <= seed < 2**32, "seed must be uint32")
    config_bytes = config_path.read_bytes()
    config = parse_json(config_bytes)
    require(type(config) is dict, "config must be an object")
    for section in (config, config.get("text_config", {})):
        require(type(section) is dict, "text_config must be an object")
        require(not section.get("quantization_config") and not section.get("compression_config"),
                "quantized config requires a separately reviewed unquantized oracle config/inventory; never infer packed shapes")
    spec = pinned_json(spec_path, spec_hash)
    tensors = validate_spec(spec, config_bytes)
    groups: list[list[dict]] = []
    sizes: list[int] = []
    for item in tensors:
        size = math.prod(item["shape"]) * DTYPES[item["dtype"]][1]
        require(size <= shard_bytes, f"{item['name']}: tensor exceeds shard/memory budget")
        if not groups or sizes[-1] + size > shard_bytes:
            groups.append([])
            sizes.append(0)
        groups[-1].append(item)
        sizes[-1] += size
    # Reserve space for headers/manifests before allocating any tensor.
    estimate = sum(sizes) + len(config_bytes) + spec_path.stat().st_size + 1024 * len(tensors) + 65536
    require(estimate <= max_bytes, f"checkpoint exceeds output budget: estimated {estimate} > {max_bytes}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dummy-checkpoint-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "config.json").write_bytes(config_bytes)
        write_json(stage / "tensor-spec.json", spec)
        weight_map = {}
        for index, group in enumerate(groups, 1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            weights = {item["name"]: make_tensor(item, seed) for item in group}
            save_file(weights, stage / filename, metadata={"format": "pt"})
            del weights
            weight_map.update({item["name"]: filename for item in group})
        write_json(stage / "model.safetensors.index.json",
                   {"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map})
        manifest = {
            "schema_version": 1, "purpose": "tp-oracle-only", "seed": seed,
            "generator_sha256": file_digest(Path(__file__)),
            "spec_sha256": spec_hash, "payload_bytes": sum(sizes),
            "files": {p.name: file_digest(p) for p in sorted(stage.iterdir())},
        }
        # The original approved spec bytes are retained, not a reserialization with a different hash.
        (stage / "tensor-spec.json").write_bytes(spec_path.read_bytes())
        require(file_digest(stage / "tensor-spec.json") == spec_hash, "spec changed during generation")
        manifest["files"]["tensor-spec.json"] = spec_hash
        require(config_path.read_bytes() == config_bytes, "config changed during generation")
        write_json(stage / "oracle-manifest.json", manifest)
        require(sum(p.stat().st_size for p in stage.iterdir()) <= max_bytes, "actual checkpoint exceeds output budget")
        manifest_hash = file_digest(stage / "oracle-manifest.json")
        verify(stage, manifest_hash)
        require(not output.exists(), "output appeared during generation")
        os.rename(stage, output)
    return manifest_hash


def verify(root: Path, expected: str) -> dict:
    manifest = pinned_json(root / "oracle-manifest.json", expected)
    exact(manifest, {"schema_version", "purpose", "seed", "generator_sha256",
                     "spec_sha256", "payload_bytes", "files"}, "oracle manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["purpose"] == "tp-oracle-only", "not an oracle checkpoint")
    require(type(manifest["seed"]) is int and 0 <= manifest["seed"] < 2**32, "invalid seed")
    sha(manifest["generator_sha256"], "generator_sha256")
    sha(manifest["spec_sha256"], "spec_sha256")
    files = manifest["files"]
    required = {"config.json", "tensor-spec.json", "model.safetensors.index.json"}
    require(type(files) is dict and required <= set(files), "missing checkpoint files")
    require(set(p.name for p in root.iterdir()) == set(files) | {"oracle-manifest.json"}, "unexpected/missing checkpoint files")
    for name, expected_hash in files.items():
        require(name in required or re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name), "invalid checkpoint filename")
        path = root / name
        require(path.is_file() and not path.is_symlink(), "checkpoint files must be regular files")
        require(file_digest(path) == sha(expected_hash, name), f"{name}: digest mismatch")
    require(files["tensor-spec.json"] == manifest["spec_sha256"], "spec binding mismatch")
    spec = load_json(root / "tensor-spec.json")
    tensors = validate_spec(spec, (root / "config.json").read_bytes())
    index = load_json(root / "model.safetensors.index.json")
    exact(index, {"metadata", "weight_map"}, "checkpoint index")
    exact(index["metadata"], {"total_size"}, "index metadata")
    weight_map = index["weight_map"]
    exact(weight_map, {item["name"] for item in tensors}, "weight map")
    expected_items = {item["name"]: item for item in tensors}
    seen = set()
    for filename in files.keys() - required:
        with safe_open(root / filename, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                require(name in expected_items and name not in seen, "unknown/duplicate stored tensor")
                seen.add(name)
                require(weight_map[name] == filename, "weight map destination mismatch")
                stored = shard.get_slice(name)
                item = expected_items[name]
                require(stored.get_shape() == item["shape"] and stored.get_dtype() == DTYPES[item["dtype"]][2],
                        f"{name}: stored shape/dtype mismatch")
    require(seen == set(expected_items), "missing stored tensors")
    payload = sum(math.prod(item["shape"]) * DTYPES[item["dtype"]][1] for item in tensors)
    require(type(manifest["payload_bytes"]) is int and manifest["payload_bytes"] == payload
            and type(index["metadata"]["total_size"]) is int and index["metadata"]["total_size"] == payload,
            "payload size mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("create")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--spec-sha256", required=True, help="independently reviewed inventory hash")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=1234)
    build.add_argument("--max-bytes", type=int, required=True, help="hard total disk budget, including metadata")
    build.add_argument("--shard-bytes", type=int, default=256 * 1024**2, help="max tensor/shard payload; account for serialization copies too")
    check = commands.add_parser("verify")
    check.add_argument("checkpoint", type=Path)
    check.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            result = create(args.config, args.spec, args.spec_sha256, args.output,
                            args.seed, args.max_bytes, args.shard_bytes)
            print(json.dumps({"checkpoint": str(args.output), "manifest_sha256": result, "purpose": "tp-oracle-only"}))
        else:
            verify(args.checkpoint, args.manifest_sha256)
            print("Checkpoint hashes, inventory and stored shapes verified; not model parity.")
        return 0
    except (GateError, OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
