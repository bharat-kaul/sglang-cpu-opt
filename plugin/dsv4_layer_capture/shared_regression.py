"""Capture native shared-expert operands at two request-bound decision points."""

import functools
import json
import os
from pathlib import Path


def install(model_class, moe_class, plan_path, directory):
    import torch

    plan = json.loads(Path(plan_path).read_text())
    root = Path(directory)
    root.mkdir()
    sequences = {}
    active = None
    seen = set()
    weights_saved = set()
    original_model = model_class.forward
    original_cpu = moe_class.forward_cpu

    def save(path, value):
        temporary = path.with_suffix(".tmp")
        torch.save(value, temporary)
        temporary.replace(path)

    @functools.wraps(original_model)
    def model_forward(self, input_ids, positions, batch, *args, **kwargs):
        nonlocal active
        ids, pos = input_ids.tolist(), positions.tolist()
        slots = batch.req_pool_indices.tolist()
        active = None
        if batch.forward_mode.is_extend():
            offset = 0
            for slot, prefix, length in zip(
                slots, batch.extend_prefix_lens_cpu, batch.extend_seq_lens_cpu
            ):
                chunk = ids[offset:offset + length]
                if prefix == 0:
                    sequences[slot] = []
                if slot not in sequences or len(sequences[slot]) != prefix:
                    raise RuntimeError("capture requires contiguous request history")
                if pos[offset:offset + length] != list(range(prefix, prefix + length)):
                    raise RuntimeError("capture position mismatch")
                sequences[slot].extend(chunk)
                offset += length
            rows = [sum(batch.extend_seq_lens_cpu[:j + 1]) - 1 for j in range(len(slots))]
        elif batch.forward_mode.is_decode():
            rows = list(range(len(slots)))
            for j, slot in enumerate(slots):
                if slot not in sequences or len(sequences[slot]) != pos[j]:
                    raise RuntimeError("decode capture history mismatch")
                sequences[slot].append(ids[j])
        else:
            rows = []
        for slot, row in zip(slots, rows):
            sequence = sequences[slot]
            for decision in plan["decisions"]:
                expected = plan["prompt_ids"] + plan["common_generated_ids"][:decision]
                if len(sequence) == len(expected) and sequence == expected:
                    active = {
                        "q": plan["q"], "decision": decision, "target_row": row,
                        "target_slot": slot, "sequence_ids": list(sequence),
                        "input_ids": ids, "positions": pos, "slots": slots,
                        "mode": batch.forward_mode.name,
                        "job": os.environ.get("SLURM_JOB_ID"),
                        "shared_clamp": os.environ.get("INTEL_CPU_DSV4_SHARED_CLAMP", "0"),
                    }
        try:
            output = original_model(self, input_ids, positions, batch, *args, **kwargs)
            if active is not None:
                missing = set(range(43)) - {layer for decision, layer in seen if decision == active["decision"]}
                if missing:
                    raise RuntimeError(f"missing captured shared layers: {missing}")
            return output
        finally:
            active = None

    @functools.wraps(original_cpu)
    def cpu_forward(self, hidden_states):
        if active is None:
            return original_cpu(self, hidden_states)
        layer = int(self.layer_id)
        key = (active["decision"], layer)
        if key in seen:
            raise RuntimeError(f"duplicate shared capture: {key}")
        original_op = torch.ops.sgl_kernel.shared_expert_cpu
        calls = 0

        def capture(*args):
            nonlocal calls
            calls += 1
            h, w1, w2, routed, factor, inplace, int8, fp8, s1, s2, block, vnni = args
            if not fp8 or int8 or not vnni:
                raise RuntimeError("expected packed FP8 shared-expert production path")
            payload = dict(active)
            payload.update(layer=layer, hidden=h.detach().clone(), routed=routed.detach().clone(),
                           factor=float(factor), inplace=bool(inplace))
            if layer not in weights_saved:
                save(root / f"weights.L{layer}.pt", {
                    "w1": w1.detach().clone(), "w2": w2.detach().clone(),
                    "s1": s1.detach().clone(), "s2": s2.detach().clone(),
                    "block": list(block), "limit": float(self.shared_experts.swiglu_limit),
                })
                weights_saved.add(layer)
            output = original_op(*args)
            payload["combined"] = output.detach().clone()
            save(root / f"d{active['decision']}.L{layer}.pt", payload)
            return output

        torch.ops.sgl_kernel.shared_expert_cpu = capture
        try:
            output = original_cpu(self, hidden_states)
        finally:
            torch.ops.sgl_kernel.shared_expert_cpu = original_op
        if calls != 1:
            raise RuntimeError(f"expected one shared call, got {calls}")
        seen.add(key)
        return output

    model_class.forward = model_forward
    moe_class.forward_cpu = cpu_forward
