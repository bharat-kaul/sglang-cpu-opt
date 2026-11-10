"""Read-only CPU forward trace for the residual-gap execution comparison."""

import functools
import json
import os


def install_execution_trace(model_class, prefix):
    if getattr(model_class, "_execution_trace_installed", False):
        return
    original = model_class.forward
    selected_ids = json.loads(os.environ.get("INTEL_CPU_DSV4_TRACE_TOKEN_IDS", "[]"))
    if (not isinstance(selected_ids, list) or
            any(type(token) is not int or token < 0 for token in selected_ids) or
            len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("trace token IDs must be a list of distinct nonnegative integers")
    stream = None
    step = 0

    @functools.wraps(original)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        nonlocal stream, step
        if stream is None:
            stream = open(f"{prefix}.{os.getpid()}.jsonl", "x", buffering=1)
        record = {
            "step": step,
            "pid": os.getpid(),
            "job": os.environ.get("SLURM_JOB_ID"),
            "mode": forward_batch.forward_mode.name,
            "input_ids": input_ids.tolist(),
            "positions": positions.tolist(),
            "req_pool_indices": forward_batch.req_pool_indices.tolist(),
            "seq_lens": forward_batch.seq_lens.tolist(),
            "extend_prefix_lens": forward_batch.extend_prefix_lens_cpu,
            "extend_seq_lens": forward_batch.extend_seq_lens_cpu,
        }
        output = original(self, input_ids, positions, forward_batch, *args, **kwargs)
        logits = output.next_token_logits
        if logits is not None:
            values, indices = logits.detach().topk(2, dim=-1)
            record["raw_next_logit_top2_ids"] = indices.tolist()
            record["raw_next_logit_top2_values"] = values.tolist()
            if selected_ids:
                if max(selected_ids) >= logits.shape[-1]:
                    raise ValueError("trace token ID exceeds the model vocabulary")
                record["selected_logit_ids"] = selected_ids
                record["selected_logit_values"] = logits.detach()[:, selected_ids].tolist()
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        if step % 32 == 0:
            print(f"DSV4_EXECUTION_TRACE step={step} mode={record['mode']} "
                  f"rows={len(record['input_ids'])}", flush=True)
        step += 1
        return output

    model_class.forward = forward
    model_class._execution_trace_installed = True
