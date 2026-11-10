"""Request-bound native model tracing; optional GPU-only prefix forcing."""

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch


class Histories:
    def __init__(self):
        self.sequences = {}

    def update(self, ids, positions, slots, prefixes=None, lengths=None):
        if len(set(slots)) != len(slots) or len(ids) != len(positions):
            raise RuntimeError("invalid forward coordinates")
        if prefixes is None:
            prefixes = positions
            lengths = [1] * len(slots)
        if len(slots) != len(prefixes) or len(slots) != len(lengths):
            raise RuntimeError("request metadata length mismatch")
        if sum(lengths) != len(ids):
            raise RuntimeError("token rows do not match request lengths")
        offset, rows = 0, []
        for slot, prefix, length in zip(slots, prefixes, lengths):
            if length <= 0:
                raise RuntimeError("empty request segment")
            if prefix == 0:
                self.sequences[slot] = []
            if slot not in self.sequences or len(self.sequences[slot]) != prefix:
                raise RuntimeError("noncontiguous request history")
            if positions[offset:offset + length] != list(range(prefix, prefix + length)):
                raise RuntimeError("request positions do not match history")
            self.sequences[slot].extend(ids[offset:offset + length])
            rows.append(offset + length - 1)
            offset += length
        return rows


def select_row(value, row, nrows):
    if torch.is_tensor(value):
        if value.ndim == 0 or value.shape[0] != nrows:
            raise RuntimeError(f"unmapped boundary shape {tuple(value.shape)}, rows={nrows}")
        selected = value[row].detach().cpu().clone()
        if selected.is_floating_point() and not torch.isfinite(selected.float()).all():
            raise RuntimeError("nonfinite native boundary")
        return selected
    if isinstance(value, (tuple, list)):
        return [select_row(v, row, nrows) for v in value]
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"unsupported native boundary: {type(value)}")


def force_tokens(logits, requests, sequences):
    for logit_row, item, decision in requests:
        expected = item["prompt_ids"] + item["output_token_ids"][:decision]
        if sequences[logit_row] != expected or decision >= len(item["output_token_ids"]):
            raise RuntimeError("forced request does not match the recorded CPU prefix")
        token = item["output_token_ids"][decision]
        if token < 0 or token >= logits.shape[-1]:
            raise RuntimeError("forced token outside vocabulary")
        logits[logit_row].fill_(-torch.inf)
        logits[logit_row, token] = 0


def install(model_class, layer_class):
    if getattr(model_class, "_prefix_trace_installed", False):
        raise RuntimeError("prefix trace installed twice")
    plan_path = Path(os.environ["DSV4_PREFIX_TRACE_PLAN"])
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    items = [item for batch in plan["batches"] for item in batch]
    if len({tuple(item["prompt_ids"]) for item in items}) != len(items):
        raise RuntimeError("ambiguous request identities")
    history = Histories()
    active, nrows, counters = [], 0, {}
    seen = set()
    initialized = False
    directory = None
    original_model = model_class.forward

    def save(path, payload):
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def record(name, value):
        for capture in active:
            if name in capture["boundaries"]:
                raise RuntimeError(f"duplicate native boundary {name}")
            capture["boundaries"][name] = select_row(value, capture["target_row"], nrows)

    def wrap_module(module, name, input_name):
        def before(_module, args, kwargs):
            if active:
                value = args[0] if args else kwargs[input_name]
                record(name + ".in", value)

        def after(_module, args, kwargs, output):
            if active:
                if name.endswith(".topk"):
                    if not hasattr(output, "topk_ids") or not hasattr(output, "topk_weights"):
                        raise RuntimeError(f"unsupported native routing output {type(output)}")
                    output = (output.topk_ids, output.topk_weights)
                record(name + ".out", output)

        module.register_forward_pre_hook(before, with_kwargs=True)
        module.register_forward_hook(after, with_kwargs=True)

    def wrap_mhc(layer, method, name):
        original = getattr(layer, method)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not active:
                return original(*args, **kwargs)
            count = counters.get(name, 0)
            counters[name] = count + 1
            key = f"{name}.{count}"
            record(key + ".in", args[0] if method == "hc_pre" else args[:4])
            output = original(*args, **kwargs)
            record(key + ".out", output)
            return output

        setattr(layer, method, wrapped)

    def initialize(model):
        nonlocal directory
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        directory = Path(os.environ["DSV4_PREFIX_TRACE_OUT"]) / f"rank{rank}"
        directory.mkdir(parents=True, exist_ok=False)
        layers = {int(m.layer_id): m for m in model.modules() if isinstance(m, layer_class)}
        if set(layers) != set(range(43)):
            raise RuntimeError(f"unexpected native layer inventory: {sorted(layers)}")
        weights = {}
        for idx, layer in layers.items():
            stem = f"L{idx}"

            def before(_module, args, kwargs, stem=stem):
                if active:
                    hidden = args[1] if len(args) > 1 else kwargs["hidden_states"]
                    record(stem + ".in", hidden)

            def after(_module, args, kwargs, output, stem=stem):
                if active:
                    record(stem + ".out", output)

            layer.register_forward_pre_hook(before, with_kwargs=True)
            layer.register_forward_hook(after, with_kwargs=True)
            wrap_module(layer.self_attn, stem + ".attn", "x")
            wrap_module(layer.mlp, stem + ".mlp", "hidden_states")
            wrap_module(layer.mlp.gate, stem + ".gate", "hidden_states")
            wrap_module(layer.mlp.topk, stem + ".topk", "hidden_states")
            wrap_mhc(layer, "hc_pre", stem + ".mhc.pre")
            wrap_mhc(layer, "hc_post", stem + ".mhc.post")
            weights[idx] = {
                name: getattr(layer, name).detach().cpu().clone()
                for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                             "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base")
            }
            weights[idx]["input_norm"] = layer.input_layernorm.weight.detach().cpu().clone()
            weights[idx]["ffn_norm"] = layer.post_attention_layernorm.weight.detach().cpu().clone()
            weights[idx]["fused_mhc"] = layer.use_fused_mhc_post_pre
        original_head = model.model.hc_head

        def hc_head(*args, **kwargs):
            if active:
                record("hc_head.in", args[0])
            output = original_head(*args, **kwargs)
            if active:
                record("hc_head.out", output)
            return output

        model.model.hc_head = hc_head
        wrap_module(model.model.norm, "final_norm", "x")

        def head_input(_module, args, kwargs):
            if active:
                record("lm_head.in", args[1] if len(args) > 1 else kwargs["hidden_states"])

        model.logits_processor.register_forward_pre_hook(head_input, with_kwargs=True)
        weights["head"] = {
            name: getattr(model.model, name).detach().cpu().clone()
            for name in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        }
        save(directory / "mhc_weights.pt", weights)
        source = Path(inspect.getfile(layer_class))
        (directory / "native_decoder.py").write_bytes(source.read_bytes())

    @functools.wraps(original_model)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        nonlocal active, nrows, counters, initialized
        phase = Path(os.environ["DSV4_PREFIX_TRACE_PHASE"]).read_text().strip()
        if phase == "warmup" and not initialized:
            return original_model(self, input_ids, positions, forward_batch, *args, **kwargs)
        if phase not in ("natural", "forced"):
            raise RuntimeError(f"invalid trace phase {phase}")
        if not initialized:
            initialize(self)
            initialized = True
        if forward_batch.forward_mode.is_idle():
            return original_model(self, input_ids, positions, forward_batch, *args, **kwargs)
        ids, pos = input_ids.tolist(), positions.tolist()
        slots = forward_batch.req_pool_indices.tolist()
        if forward_batch.forward_mode.is_extend():
            prefixes = forward_batch.extend_prefix_lens_cpu
            lengths = forward_batch.extend_seq_lens_cpu
            rows = history.update(ids, pos, slots, prefixes, lengths)
        elif forward_batch.forward_mode.is_decode():
            prefixes, lengths = None, None
            rows = history.update(ids, pos, slots)
        else:
            raise RuntimeError(f"unsupported native forward mode {forward_batch.forward_mode}")
        if phase == "forced" and input_ids.device.type != "cuda":
            raise RuntimeError("prefix forcing is GPU-only; CPU must reproduce naturally")
        active, counters, requests = [], {}, []
        nrows = len(ids)
        sequences = [history.sequences[slot] for slot in slots]
        for logit_row, (slot, row) in enumerate(zip(slots, rows)):
            sequence = sequences[logit_row]
            for item in items:
                prompt = item["prompt_ids"]
                if sequence[:len(prompt)] != prompt:
                    continue
                decision = len(sequence) - len(prompt)
                requests.append((logit_row, item, decision))
                if item["q"] not in plan["targets"] or decision > plan["last_decision"]:
                    continue
                active.append({
                    "q": item["q"], "decision": decision, "phase": phase,
                    "sequence_ids": list(sequence), "target_row": row,
                    "logit_row": logit_row, "slot": slot, "input_ids": ids,
                    "positions": pos, "slots": slots, "mode": forward_batch.forward_mode.name,
                    "extend_prefix_lens": prefixes, "extend_seq_lens": lengths,
                    "job": os.environ.get("SLURM_JOB_ID"),
                    "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                    "boundaries": {},
                })
        try:
            output = original_model(self, input_ids, positions, forward_batch, *args, **kwargs)
            for capture in active:
                required = {
                    f"L{i}.{boundary}"
                    for i in range(43)
                    for boundary in ("in", "out", "attn.in", "attn.out",
                                     "mlp.in", "mlp.out", "topk.out")
                }
                required.update(("hc_head.in", "hc_head.out", "final_norm.in",
                                 "final_norm.out", "lm_head.in"))
                missing = required - capture["boundaries"].keys()
                if missing:
                    raise RuntimeError(f"missing native boundaries: {sorted(missing)}")
                logits = output.next_token_logits
                capture["logits"] = select_row(logits, capture["logit_row"], len(slots))
                key = (phase, capture["q"], capture["decision"])
                if key in seen:
                    raise RuntimeError(f"duplicate request coordinate {key}")
                save(directory / f"{phase}.q{capture['q']}.d{capture['decision']}.pt", capture)
                seen.add(key)
                print(f"PREFIX_TRACE phase={phase} q={capture['q']} d={capture['decision']}",
                      flush=True)
            if phase == "forced":
                force_tokens(output.next_token_logits, requests, sequences)
            return output
        finally:
            active = []

    model_class.forward = forward
    model_class._prefix_trace_installed = True


if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == "dsv4_prefix_trace":
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4ForCausalLM

    install(DeepseekV4ForCausalLM, DeepseekV4DecoderLayer)
    EntryClass = DeepseekV4ForCausalLM
