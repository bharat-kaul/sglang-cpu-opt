"""Isolated worker-only C4 trial; no package-default dispatch changes."""

import functools
import hashlib
import json
from pathlib import Path

import torch


STATE_MAPS = ("_KV_BUF", "_KV_VALID", "_COMPRESSED_STASH", "_COMPRESSED_HANDOFF",
              "_COMPRESSED_PERSIST", "_WIN_CARRY", "_TOPK_STASH")


def reset_inference_state(infra):
    for name in STATE_MAPS:
        getattr(infra, name).clear()
    infra._SEL_HOLDER["pos_kept"] = None


def tensor_hash(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def shared_chunked(q, kv, mask, sink, scale):
    pieces = []
    for i in range(0, q.shape[0], 32):
        x = q[i:i + 32]
        n, h, d = x.shape
        k = kv.shape[0]
        scores = (x.reshape(n * h, d) @ kv.t()).reshape(n, h, k) * scale
        scores = scores.masked_fill(~mask[i:i + 32].unsqueeze(1), float("-inf"))
        sinks = sink.reshape(1, h, 1).float().expand(n, h, 1)
        probs = torch.nan_to_num(
            torch.cat([scores, sinks], -1).softmax(-1)[..., :-1], nan=0.0)
        pieces.append((probs.reshape(n * h, k) @ kv).reshape(n, h, d))
    return torch.cat(pieces)


class SparseTrial:
    def __init__(self, path, output, infra, probe):
        self.plan = json.loads(Path(path).read_text())
        if self.plan != {
            "b1-s3072-trial-a": False, "b1-s3072-trial-candidate": True,
            "b1-s3072-trial-b": False,
        } or any(type(value) is not bool for value in self.plan.values()):
            raise ValueError("unexpected sparse trial plan")
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.infra, self.probe = infra, probe
        self.case = None
        self.record = None
        self.ratio = None
        self.hidden = {}
        self.calls = []
        self.failed = False

    def begin(self, record, batch):
        if record["case"] not in self.plan:
            raise RuntimeError("unplanned sparse trial forward")
        self.record = record
        self.hidden, self.calls, self.failed = {}, [], False
        self.ratio = None
        reset = record["case"] != self.case
        if reset:
            if record["phase"] != "prefill" or int(batch.positions.min()) != 0:
                raise RuntimeError("new trial arm must begin with a fresh prefill")
            reset_inference_state(self.infra)
            self.case = record["case"]
        record["trial"] = {
            "candidate": self.plan[self.case], "state_reset": reset,
            "positions_hash": tensor_hash(batch.positions),
            "input_ids_hash": tensor_hash(batch.input_ids),
            "req_pool_indices": batch.req_pool_indices.tolist(),
            "out_cache_loc_hash": tensor_hash(batch.out_cache_loc),
        }

    def layer_output(self, layer, result):
        value = result[0] if isinstance(result, tuple) else result
        if not isinstance(value, torch.Tensor) or value.shape[0] != self.record["tokens"]:
            raise RuntimeError("unexpected layer output shape")
        if int(layer.layer_id) in self.hidden:
            raise RuntimeError("duplicate trial layer output")
        self.hidden[int(layer.layer_id)] = value[-1:].detach().clone()

    def wrap(self, original):
        @functools.wraps(original)
        def sparse(q, kv, mask, sink, scale, torch_fn):
            try:
                layer = self.probe._active.layer
                selected = (self.plan[self.case] and self.record["phase"] == "prefill"
                            and self.ratio == 4)
                item = {"layer": layer, "ratio": self.ratio, "candidate": selected,
                        "q_shape": list(q.shape), "kv_shape": list(kv.shape),
                        "mask_hash": tensor_hash(mask)}
                if layer == 0:
                    item["q_hash"], item["kv_hash"] = tensor_hash(q), tensor_hash(kv)
                with self.probe.region("trial.sparse", ratio=self.ratio, candidate=selected):
                    if selected:
                        result = shared_chunked(q, kv, mask, sink, scale)
                    else:
                        result = original(q, kv, mask, sink, scale, torch_fn)
                self.calls.append(item)
                return result
            except BaseException:
                self.failed = True
                raise
        return sparse

    def finish(self, result):
        if self.failed:
            raise RuntimeError("sparse trial failed inside a caught fallback")
        if set(self.hidden) != set(range(43)) or {x["layer"] for x in self.calls} != set(range(43)):
            raise RuntimeError("incomplete trial layer/sparse coverage")
        if len(self.calls) != 43:
            raise RuntimeError("duplicate sparse trial layer call")
        logits = result.next_token_logits
        if logits is not None:
            logits = logits.detach().clone()
        tensors = list(self.hidden.values()) + ([] if logits is None else [logits])
        if any(not torch.isfinite(t).all() for t in tensors):
            raise RuntimeError("nonfinite downstream trial tensors")
        payload = {
            "job": self.record["job"], "pid": self.record["pid"],
            "sequence": self.record["sequence"], "case": self.case,
            "phase": self.record["phase"], "hidden_last_row": self.hidden, "logits": logits,
        }
        path = self.output / f'{self.record["pid"]}-{self.record["sequence"]}.pt'
        with path.open("xb") as stream:
            torch.save(payload, stream)
        self.record["trial"].update(
            calls=self.calls, tensors=str(path),
            tensors_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
