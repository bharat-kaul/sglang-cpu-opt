"""Native GPU shared MLP replay; no substituted model-accuracy claims."""

import hashlib
import json
import os
from pathlib import Path
from contextlib import nullcontext


def install():
    import torch
    import sglang.srt.models.deepseek_v4 as model
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.distributed import get_tensor_model_parallel_rank, tensor_model_parallel_all_reduce

    root = Path(os.environ["DSV4_SHARED_GPU_ROOT"])
    manifest = json.loads((root / "gpu_manifest.json").read_text())
    original = model.DeepseekV4ForCausalLM.forward
    done = False
    quant_capture = os.environ.get("DSV4_SHARED_GPU_QUANT_CAPTURE") == "1"

    def clone(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, (tuple, list)):
            return [clone(v) for v in value]
        if value is None:
            return None
        raise TypeError(f"unexpected GPU linear operand type {type(value)}")

    def forward(self, input_ids, positions, batch, *args, **kwargs):
        nonlocal done
        output = original(self, input_ids, positions, batch, *args, **kwargs)
        if done or input_ids.tolist() != manifest["trigger_ids"] or positions.tolist() != list(
            range(len(manifest["trigger_ids"]))
        ):
            return output
        done = True
        rank = get_tensor_model_parallel_rank()
        directory = root / f"gpu_{os.environ['SLURM_JOB_ID']}" / f"rank{rank}"
        directory.mkdir(parents=True)
        modules = {int(m.layer_id): m for m in self.modules() if isinstance(m, DeepseekV2MoE)}
        if set(modules) != set(range(43)):
            raise RuntimeError(f"unexpected GPU MoE inventory: {sorted(modules)}")
        records = []
        weight_cache = {}
        if quant_capture:
            from .native_quant import capture_gemm
            from sglang.srt.layers.quantization import fp8_utils
            from sglang.kernels.ops.quantization import fp8_kernel
            for module in (fp8_utils, fp8_kernel):
                (directory / Path(module.__file__).name).write_bytes(Path(module.__file__).read_bytes())
        for item in manifest["captures"]:
            path = Path(item["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError("CPU capture changed after GPU manifest creation")
            source = torch.load(path, map_location="cpu", weights_only=False)
            moe = modules[source["layer"]]
            mlp = moe.shared_experts
            limit = mlp.swiglu_limit
            if limit != 10.0 or moe.num_fused_shared_experts != 0:
                raise RuntimeError("unexpected native GPU shared clamp/fusion configuration")
            x = source["hidden"].to(device=input_ids.device)
            gate_forward = mlp.gate_up_proj.forward
            down_forward = mlp.down_proj.forward
            payload = {
                "source": str(path), "source_sha256": item["sha256"], "rank": rank,
                "job": os.environ["SLURM_JOB_ID"], "layer": source["layer"],
                "q": source["q"], "decision": source["decision"], "target_row": source["target_row"],
                "input": clone(x), "configured_limit": limit,
                "shared_tp1": bool(getattr(moe, "_shared_expert_tp1", False)),
                "down_reduce_results": bool(mlp.down_proj.reduce_results),
                "gate_quant_method": type(mlp.gate_up_proj.quant_method).__module__ + "." +
                                     type(mlp.gate_up_proj.quant_method).__name__,
                "down_quant_method": type(mlp.down_proj.quant_method).__module__ + "." +
                                     type(mlp.down_proj.quant_method).__name__,
                "gate_weight_dtype": str(mlp.gate_up_proj.weight.dtype),
                "down_weight_dtype": str(mlp.down_proj.weight.dtype),
                "variants": {},
            }
            for variant, clamp_limit in (("on", limit), ("repeat", limit), ("inactive", 1e30)):
                stage = {}
                capture_quant = quant_capture and source["decision"] > 0

                def tap(projection):
                    if not capture_quant:
                        return nullcontext()
                    return capture_gemm(stage, projection, directory, source["layer"],
                                        rank, os.environ["SLURM_JOB_ID"], weight_cache)

                def gate(*a, **k):
                    with tap("gate"):
                        result = gate_forward(*a, **k)
                    stage["gate_output"] = clone(result[0])
                    return result

                def down(*a, **k):
                    stage["down_input"] = clone(a[0])
                    with tap("down"):
                        result = down_forward(*a, **k)
                    stage["down_local_output"] = clone(result[0])
                    return result

                mlp.gate_up_proj.forward, mlp.down_proj.forward = gate, down
                mlp.swiglu_limit = clamp_limit
                try:
                    with torch.no_grad():
                        local = moe._forward_shared_experts(x.clone())
                        if local is None:
                            raise RuntimeError("native GPU did not execute shared expert")
                        value = local.clone()
                        if not mlp.down_proj.reduce_results and not getattr(moe, "_shared_expert_tp1", False):
                            value = tensor_model_parallel_all_reduce(value)
                        torch.cuda.synchronize()
                    if not torch.isfinite(value).all():
                        raise RuntimeError("nonfinite native GPU shared output")
                    stage["output"] = clone(value)
                    stage["limit"] = clamp_limit
                    if "gate_output" not in stage:
                        raise RuntimeError("native GPU gate projection was not observed")
                    if stage["gate_output"].float().abs().max().item() >= 1e30:
                        raise RuntimeError("large-limit control is not inactive")
                    payload["variants"][variant] = stage
                finally:
                    mlp.swiglu_limit = limit
                    mlp.gate_up_proj.forward, mlp.down_proj.forward = gate_forward, down_forward
            if not torch.equal(payload["variants"]["on"]["output"], payload["variants"]["repeat"]["output"]):
                raise RuntimeError("GPU unchanged-repeat control differs")
            filename = f"q{source['q']}.cpu{source['job']}.d{source['decision']}.L{source['layer']}.pt"
            torch.save(payload, directory / filename)
            records.append(filename)
            print(f"SHARED_GPU_REPLAY rank={rank} {filename}", flush=True)
        (directory / "complete.json").write_text(json.dumps({
            "job": os.environ["SLURM_JOB_ID"], "rank": rank, "records": records,
            "manifest_sha256": hashlib.sha256((root / "gpu_manifest.json").read_bytes()).hexdigest(),
            "scope": "native GPU shared MLP on CPU operands; not GPU model task accuracy",
            "quant_capture": quant_capture,
        }, indent=2))
        return output

    model.DeepseekV4ForCausalLM.forward = forward


if os.environ.get("DSV4_SHARED_GPU_ROOT"):
    install()
