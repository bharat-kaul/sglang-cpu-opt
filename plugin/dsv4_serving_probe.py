"""Read-only, worker-controlled attribution for the isolated serving screen."""

from contextlib import contextmanager, nullcontext
import functools
import json
import os
from pathlib import Path
import time


_active = None
_installed = False
MAX_FORWARD_SECONDS = 180


def tensor_meta(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def read_control(path):
    value = json.loads(Path(path).read_text())
    if (set(value) != {"case", "enabled"} or type(value["enabled"]) is not bool
            or not isinstance(value["case"], str) or not value["case"]
            or len(value["case"]) > 160):
        raise ValueError("invalid serving-probe control")
    return value


class Recorder:
    def __init__(self):
        self.layer = -1
        self.rows = {}
        self.stack = []
        self.failures = []
        self.dispatch = {}

    @contextmanager
    def region(self, name, metadata=None):
        key = (self.layer, name, json.dumps(metadata or {}, sort_keys=True))
        frame = [time.perf_counter_ns(), 0]
        self.stack.append(frame)
        try:
            yield
        except BaseException as exc:
            self.failures.append({"layer": self.layer, "op": name, "error": type(exc).__name__})
            raise
        finally:
            elapsed = time.perf_counter_ns() - frame[0]
            assert self.stack.pop() is frame
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.rows.setdefault(key, {
                "layer": key[0], "op": name, "metadata": metadata or {},
                "calls": 0, "inclusive_ns": 0, "exclusive_ns": 0,
            })
            row["calls"] += 1
            row["inclusive_ns"] += elapsed
            row["exclusive_ns"] += elapsed - frame[1]


def region(name, **metadata):
    return nullcontext() if _active is None else _active.region(name, metadata)


def tensor_region(name, *tensors, **metadata):
    if _active is None:
        return nullcontext()
    return _active.region(name, {**metadata, "tensors": [tensor_meta(t) for t in tensors]})


def wrap_call(fn, name, describe=None):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if _active is None:
            return fn(*args, **kwargs)
        metadata = describe(*args, **kwargs) if describe else {}
        with _active.region(name, metadata):
            return fn(*args, **kwargs)
    return wrapped


def install_boundaries(model, infra):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    import torch

    targets = [
        (model.DeepseekV4DecoderLayer, "hc_pre", "mhc.pre"),
        (model.DeepseekV4DecoderLayer, "hc_post", "mhc.post"),
        (model.MQALayer, "forward", "attention.total"),
        (DeepseekV2MoE, "forward_cpu", "moe.total"),
        (Fp8MoEMethod, "apply", "moe.routed"),
        (torch.ops.sgl_kernel, "shared_expert_cpu", "moe.shared"),
        (Fp8LinearMethod, "apply", "dense.fp8"),
        (UnquantizedLinearMethod, "apply", "dense.unquant"),
        (RMSNorm, "forward", "norm"),
        (LogitsProcessor, "forward", "logits"),
    ]
    for owner, method, label in targets:
        setattr(owner, method, wrap_call(getattr(owner, method), label))
    for name in ("_accumulate_request_blocks", "_publish_full_history_handoff",
                 "_overlap_transform", "_fp8_roundtrip_nope_bf16"):
        setattr(infra, name, wrap_call(getattr(infra, name), name))


def install():
    global _installed
    control = os.environ.get("DSV4_SERVING_PROBE_CONTROL")
    if os.environ.get("DSV4_C4_ACCURACY_CONTROL"):
        if control or os.environ.get("DSV4_SPARSE_TRIAL_PLAN") or os.environ.get("DSV4_SPARSE_CAPTURE_PLAN"):
            raise RuntimeError("accuracy and attribution probes are mutually exclusive")
        from dsv4_accuracy_probe import install as install_accuracy
        install_accuracy()
        return
    if not control or _installed:
        return
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    import intel_cpu_models._dsv4_cpu_infra as infra
    import intel_cpu_models._dsa_ckernels as ck
    import sglang.srt.models.deepseek_v4 as model
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    if infra._TIMEIT_ON:
        raise RuntimeError("serving probe requires TIMEIT=0 to avoid nested legacy instrumentation")
    output = Path(os.environ["DSV4_SERVING_PROBE_OUT"])
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / f"events.{os.getpid()}.jsonl").open("x", buffering=1)
    sequence = 0
    light = os.environ.get("DSV4_SERVING_PROBE_LIGHT") == "1"
    capture = None
    if os.environ.get("DSV4_SPARSE_CAPTURE_PLAN"):
        from dsv4_sparse_capture import SparseCapture
        capture = SparseCapture(os.environ["DSV4_SPARSE_CAPTURE_PLAN"], output / "captures")
    if light:
        install_boundaries(model, infra)
    trial = None
    if os.environ.get("DSV4_SPARSE_TRIAL_PLAN"):
        if not light or capture is not None:
            raise RuntimeError("C4 trial requires lightweight mode without sparse capture")
        import sys
        from dsv4_sparse_trial import SparseTrial
        trial = SparseTrial(os.environ["DSV4_SPARSE_TRIAL_PLAN"],
                            output / "trial", infra, sys.modules[__name__])

    class ProjectionOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            with tensor_region(f"compressor.aten.{func}", *tensors):
                return func(*args, **(kwargs or {}))

    original_projection = Compressor.compute_kv_score

    @functools.wraps(original_projection)
    def projection(self, x, forward_batch):
        if _active is None:
            return original_projection(self, x, forward_batch)
        with tensor_region("compressor.projection", x,
                           ratio=int(self.ratio), indexer=bool(self.is_in_indexer)):
            with nullcontext() if light else ProjectionOps():
                return original_projection(self, x, forward_batch)

    Compressor.compute_kv_score = projection
    original_layer = model.DeepseekV4DecoderLayer.forward

    @functools.wraps(original_layer)
    def layer(self, *args, **kwargs):
        if _active is None:
            return original_layer(self, *args, **kwargs)
        previous = _active.layer
        _active.layer = int(self.layer_id)
        if trial is not None:
            trial.ratio = int(self.self_attn.compress_ratio)
        try:
            with region("layer.total", threads=torch.get_num_threads()):
                result = original_layer(self, *args, **kwargs)
            if trial is not None:
                trial.layer_output(self, result)
            return result
        finally:
            _active.layer = previous

    model.DeepseekV4DecoderLayer.forward = layer
    for name in ("forward_core_compressor", "forward_c4_indexer"):
        setattr(DeepseekV4AttnBackend, name,
                wrap_call(getattr(DeepseekV4AttnBackend, name), "dsa." + name))
    for name in ("_batched_mla_attend", "_stash_kv_gather", "_build_coverage_mask_vectorized",
                 "_compress_prefill_core"):
        setattr(infra, name, wrap_call(getattr(infra, name), name))
    for name in ("indexer_logits", "indexer_topk", "compress_softmax_pool", "sparse_attend_masked"):
        setattr(ck, name, wrap_call(
            getattr(ck, name), "native_wrapper." + name,
            lambda *a, **k: {"tensors": [tensor_meta(t) for t in a if isinstance(t, torch.Tensor)]},
        ))
    if capture is not None:
        ck.sparse_attend_masked = capture.wrap(
            ck.sparse_attend_masked, lambda: -1 if _active is None else _active.layer)
    if trial is not None:
        ck.sparse_attend_masked = trial.wrap(ck.sparse_attend_masked)
    original_mark = ck.mark

    def mark(op, impl):
        if _active is not None:
            key = f"{_active.layer}:{op}:{impl}"
            _active.dispatch[key] = _active.dispatch.get(key, 0) + 1
        return original_mark(op, impl)

    ck.mark = mark
    original_forward = model.DeepseekV4ForCausalLM.forward

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        global _active
        nonlocal sequence
        if _active is not None:
            raise RuntimeError("nested model forward not supported by serving probe")
        ctl = read_control(control)
        if trial is not None and not ctl["enabled"]:
            raise RuntimeError("C4 trial requires enabled recording")
        batch = kwargs.get("forward_batch")
        if batch is None:
            batch = next((a for a in args if hasattr(a, "forward_mode")), None)
        if batch is None:
            raise RuntimeError("missing forward_batch")
        recorder = Recorder()
        record = {
            "job": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(),
            "sequence": sequence, **ctl,
            "phase": "decode" if batch.forward_mode.is_decode_or_idle() else "prefill",
            "tokens": int(batch.input_ids.numel()),
            "seq_lens": batch.seq_lens.detach().cpu().tolist(),
            "threads_before": torch.get_num_threads(),
            "affinity": sorted(os.sched_getaffinity(0)),
        }
        sequence += 1
        if capture is not None:
            capture.begin(record)
        if trial is not None:
            trial.begin(record, batch)
        _active = recorder if ctl["enabled"] else None
        start = time.perf_counter_ns()
        try:
            result = original_forward(self, *args, **kwargs)
            if time.perf_counter_ns() - start > MAX_FORWARD_SECONDS * 1_000_000_000:
                raise RuntimeError(f"serving screen forward exceeded {MAX_FORWARD_SECONDS}s budget")
        except BaseException as exc:
            record["error"] = type(exc).__name__
            raise
        else:
            if recorder.failures:
                record["error"] = "caught-instrumented-failure"
                raise RuntimeError(f"instrumented operation failed inside fallback: {recorder.failures}")
            if trial is not None and trial.failed:
                record["error"] = "caught-trial-failure"
                raise RuntimeError("sparse trial failed inside a caught fallback")
            return result
        finally:
            record["forward_ns"] = time.perf_counter_ns() - start
            _active = None
            record["threads_after"] = torch.get_num_threads()
            record["rows"] = list(recorder.rows.values())
            record["failures"] = recorder.failures
            record["dispatch_marks"] = recorder.dispatch
            try:
                if capture is not None:
                    record["captures"] = capture.flush()
                if trial is not None and "error" not in record:
                    trial.finish(result)
            except BaseException as exc:
                record["error"] = f"capture-finalization:{type(exc).__name__}"
                raise
            finally:
                # Marks can precede execution; timed wrapper failures are separate evidence.
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    model.DeepseekV4ForCausalLM.forward = forward
    _installed = True
