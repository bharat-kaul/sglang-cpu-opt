"""Default-off, fail-closed dispatch for the matched C4 task-accuracy wave."""

import functools
import json
import os
from pathlib import Path
import time

import torch

from dsv4_sparse_trial import reset_inference_state, shared_chunked, tensor_hash


_installed = False


class AccuracyProbe:
    def __init__(self, control, plan, infra, output):
        self.control, self.plan, self.infra = Path(control), plan, infra
        self.stream = Path(output).open("x", buffering=1)
        self.previous_case = None
        self.active = None
        self.layer = None
        self.calls = []
        self.failed = False
        self.sequence = 0

    def begin(self, batch):
        if self.active is not None:
            raise RuntimeError("nested accuracy forward")
        ctl = json.loads(self.control.read_text())
        if (set(ctl) != {"case", "arm"} or ctl["case"] not in self.plan
                or ctl["arm"] != self.plan[ctl["case"]]):
            raise RuntimeError("unplanned accuracy control")
        phase = "decode" if batch.forward_mode.is_decode_or_idle() else "prefill"
        reset = ctl["case"] != self.previous_case
        if reset:
            if phase != "prefill" or int(batch.positions.min()) != 0:
                raise RuntimeError("accuracy batch must start at position zero")
            reset_inference_state(self.infra)
            self.previous_case = ctl["case"]
        if torch.get_num_threads() != 32:
            raise RuntimeError("accuracy worker must use fixed32 threads")
        self.calls, self.failed = [], False
        self.active = {
            **ctl, "phase": phase, "reset": reset, "job": os.environ["SLURM_JOB_ID"],
            "pid": os.getpid(), "sequence": self.sequence, "tokens": batch.input_ids.numel(),
            "threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
            "seq_lens": batch.seq_lens.tolist(),
        }
        if reset:
            self.active["first_input_hash"] = tensor_hash(batch.input_ids)
            self.active["first_positions_hash"] = tensor_hash(batch.positions)
            self.active["first_cache_hash"] = tensor_hash(batch.out_cache_loc)
        self.sequence += 1

    def wrap_sparse(self, original):
        @functools.wraps(original)
        def sparse(q, kv, mask, sink, scale, torch_fn):
            try:
                if self.active is None or self.layer is None:
                    raise RuntimeError("sparse call outside accuracy layer")
                layer, ratio = self.layer
                selected = (self.active["arm"] == "candidate"
                            and self.active["phase"] == "prefill" and ratio == 4)
                result = shared_chunked(q, kv, mask, sink, scale) if selected else original(
                    q, kv, mask, sink, scale, torch_fn)
                self.calls.append({"layer": layer, "ratio": ratio, "candidate": selected})
                return result
            except BaseException:
                self.failed = True
                raise
        return sparse

    def finish(self, result, error, elapsed):
        record = self.active
        try:
            if error is not None:
                record["error"] = error
                return
            if self.failed:
                raise RuntimeError("caught native failure in accuracy forward")
            if len(self.calls) != 43 or {c["layer"] for c in self.calls} != set(range(43)):
                raise RuntimeError("incomplete accuracy sparse dispatch")
            for c in self.calls:
                expected_ratio = 0 if c["layer"] < 2 else 4 if c["layer"] % 2 == 0 else 128
                selected = record["arm"] == "candidate" and record["phase"] == "prefill" and expected_ratio == 4
                if c["ratio"] != expected_ratio or c["candidate"] is not selected:
                    raise RuntimeError("illegal accuracy sparse dispatch")
            if torch.get_num_threads() != 32:
                raise RuntimeError("accuracy thread count changed")
            logits = result.next_token_logits
            if logits is not None and not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite accuracy logits")
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.update(sparse=self.calls, forward_ns=elapsed)
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
            self.active = None


def install():
    global _installed
    if _installed:
        return
    import intel_cpu_models._dsv4_cpu_infra as infra
    import intel_cpu_models._dsa_ckernels as ck
    import sglang.srt.models.deepseek_v4 as model

    if not ck.strict():
        raise RuntimeError("accuracy probe requires native strict mode")
    output = Path(os.environ["DSV4_C4_ACCURACY_OUT"])
    output.mkdir(exist_ok=True)
    probe = AccuracyProbe(
        os.environ["DSV4_C4_ACCURACY_CONTROL"],
        json.loads(Path(os.environ["DSV4_C4_ACCURACY_PLAN"]).read_text()),
        infra, output / f"events.{os.getpid()}.jsonl")
    ck.sparse_attend_masked = probe.wrap_sparse(ck.sparse_attend_masked)
    original_layer, original_forward, original_mark = (
        model.DeepseekV4DecoderLayer.forward, model.DeepseekV4ForCausalLM.forward, ck.mark)

    @functools.wraps(original_layer)
    def layer(self, *args, **kwargs):
        previous = probe.layer
        probe.layer = int(self.layer_id), int(self.self_attn.compress_ratio)
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            probe.layer = previous

    def mark(op, impl):
        if probe.active is not None and impl == "eager":
            probe.failed = True
            raise RuntimeError(f"unapproved eager dispatch: {op}")
        return original_mark(op, impl)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        batch = kwargs.get("forward_batch")
        if batch is None:
            batch = next((a for a in args if hasattr(a, "forward_mode")), None)
        if batch is None:
            raise RuntimeError("accuracy forward missing batch")
        probe.begin(batch)
        result, error = None, None
        start = time.perf_counter_ns()
        probe.active["start_ns"] = start
        try:
            result = original_forward(self, *args, **kwargs)
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            probe.finish(result, error, time.perf_counter_ns() - start)

    model.DeepseekV4DecoderLayer.forward = layer
    model.DeepseekV4ForCausalLM.forward = forward
    ck.mark = mark
    _installed = True
