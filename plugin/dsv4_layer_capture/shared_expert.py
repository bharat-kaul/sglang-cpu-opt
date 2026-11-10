#!/usr/bin/env python3
"""DSv4 shared-expert matched-input CAPTURE (reviewer step 1; bounded, fail-closed).

The DSV4 CPU shared expert runs via `torch.ops.sgl_kernel.shared_expert_cpu(...)`, which has NO
swiglu_limit/clamp arg (plain silu*mul); GPU clamps (swiglu_limit=10.0). We capture that op's EXACT
operands + output on the current accumulator-on + routed-clamp-on config, PACKED weights preserved, bound to
a real request by TOKEN IDs (not row count), at the decision chunk/position, fail-closed.

Gated (inert otherwise):
  INTEL_CPU_DSV4_SHARED_CAPTURE = <dst prefix>        -> <prefix>.<sha12>.L<layer>.pt (one-shot per case+layer)
  INTEL_CPU_DSV4_SHARED_CAPTURE_LAYERS = "3,10,42"    (default "3")
  INTEL_CPU_DSV4_SHARED_CAPTURE_FROZEN = frozen json  (REQUIRED: items[].token_ids = full target seq;
                                                       capture fires on the chunk completing that seq)

Reviewer blockers fixed: (1) import use_intel_amx_backend from sglang.srt.utils; (2) bind to actual request
token IDs via the plugin-threaded self._cpu_hash_input_ids, accumulated across chunks -> fire on the chunk
that completes the target sequence (supports chunked prefill); reject when no frozen target is given;
(3) FAIL-CLOSED: save BEFORE marking done; any enabled-capture failure RAISES (no silent continue); missing
op/identity raises. Non-target layers/chunks delegate to the original forward_cpu (hash-routing retained).
"""
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False
_DONE = {}
_ACC = {}          # per-layer running token-id accumulator across chunks
_FROZEN = None


def _tok_sha(ids):
    return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()


def _frozen_seqs():
    global _FROZEN
    if _FROZEN is not None:
        return _FROZEN
    out = []
    f = os.environ.get("INTEL_CPU_DSV4_SHARED_CAPTURE_FROZEN")
    if f and os.path.exists(f):
        for it in json.load(open(f)).get("items", []):
            out.append(tuple(int(x) for x in it["token_ids"]))
    _FROZEN = out
    return out


def _match_final_chunk(lid, chunk_ids):
    """Accumulate this layer's per-chunk token IDs; return the FULL target seq (tuple) when the
    accumulation EXACTLY completes a frozen target (i.e. this is the decision/final chunk), else None.
    Resets when the accumulation can no longer be a prefix of any target. No frozen targets -> None."""
    frozen = _frozen_seqs()
    if not frozen or not chunk_ids:
        return None
    acc = _ACC.get(lid, []) + list(chunk_ids)
    t = tuple(acc)
    for f in frozen:
        if t == f:
            _ACC[lid] = []
            return f
    if any(len(t) < len(f) and f[:len(t)] == t for f in frozen):
        _ACC[lid] = acc
        return None
    ch = tuple(int(x) for x in chunk_ids)
    _ACC[lid] = list(ch) if any(len(ch) <= len(f) and f[:len(ch)] == ch for f in frozen) else []
    return None


def install_shared_capture():
    global _INSTALLED
    if _INSTALLED or not os.environ.get("INTEL_CPU_DSV4_SHARED_CAPTURE"):
        return
    try:
        import torch  # noqa: F401
        import sglang.srt.models.deepseek_v2 as _dv2
        from sglang.srt.utils import use_intel_amx_backend  # correct module (installed model uses this)
        from sglang.srt.models.deepseek_v2 import should_skip_post_experts_all_reduce
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
    except Exception as e:  # noqa: BLE001 — enabled-capture install failure must be loud, not silent
        raise RuntimeError(f"[SHARED_CAP] install import failed: {e!r}") from e
    import torch
    MoE = _dv2.DeepseekV2MoE
    if getattr(MoE, "_shared_cap_installed", False):
        return
    _INSTALLED = True
    MoE._shared_cap_installed = True
    _orig_forward_cpu = MoE.forward_cpu   # the CURRENT (plugin-patched, hash-routing-aware) forward_cpu
    dst = os.environ["INTEL_CPU_DSV4_SHARED_CAPTURE"]
    want = {int(x) for x in os.environ.get("INTEL_CPU_DSV4_SHARED_CAPTURE_LAYERS", "3").split(",")}
    if not _frozen_seqs():
        raise RuntimeError("[SHARED_CAP] INTEL_CPU_DSV4_SHARED_CAPTURE_FROZEN missing/empty — request "
                           "binding is required (row count is not identity)")

    def _cap_forward_cpu(self, hidden_states):
        lid = int(getattr(self, "layer_id", -1))
        # token IDs of THIS chunk, threaded by the plugin's forward wrapper for HashTopK
        chunk_ids = getattr(self, "_cpu_hash_input_ids", None)
        target = None
        if (lid in want) and chunk_ids is not None:
            cids = [int(x) for x in chunk_ids.reshape(-1).tolist()]
            target = _match_final_chunk(lid, cids)
        if target is None or _DONE.get((lid, _tok_sha(target))) or not hasattr(self, "shared_experts") \
                or not use_intel_amx_backend(self.shared_experts.gate_up_proj):
            return _orig_forward_cpu(self, hidden_states)   # delegate: native path + hash-routing retained

        # --- target layer, decision (final) chunk: reconstruct forward_cpu to capture shared_expert_cpu I/O.
        # Reconstruction mirrors UPSTREAM forward_cpu (gate->topk->experts->shared_expert_cpu). It is only
        # valid for NON-HASH layers (hash layers 0-2 need input_ids bound into topk); fail closed otherwise.
        if getattr(self, "is_hash", False):
            raise RuntimeError(f"[SHARED_CAP] L{lid} is a HASH layer; reconstruction would mis-route. "
                               "Capture only non-hash MoE layers (>=3).")
        if self.tp_size != 1:
            raise RuntimeError(f"[SHARED_CAP] tp_size={self.tp_size}; capture requires tp=1 so the shared "
                               "kernel output is the final (pre-allreduce==final) output.")
        hid = hidden_states.detach().clone()
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        fused_experts_out = self.experts(hidden_states=hidden_states, topk_output=topk_output)
        fe_snap = fused_experts_out.detach().float().cpu()   # BEFORE shared_expert_cpu (inplace=True may mutate)
        w1s = (self.shared_experts.gate_up_proj.weight_scale if self.shared_experts_is_int8
               else (self.shared_experts.gate_up_proj.weight_scale_inv if self.shared_experts_is_fp8 else None))
        w2s = (self.shared_experts.down_proj.weight_scale if self.shared_experts_is_int8
               else (self.shared_experts.down_proj.weight_scale_inv if self.shared_experts_is_fp8 else None))
        bsz = self.shared_experts_weight_block_size if self.shared_experts_is_fp8 else None
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states, self.shared_experts.gate_up_proj.weight, self.shared_experts.down_proj.weight,
            fused_experts_out, self.routed_scaling_factor, True,
            self.shared_experts_is_int8, self.shared_experts_is_fp8, w1s, w2s, bsz, True,
        )
        # NON-PERTURBATION PROOF (in-situ, bit-exact): run the genuine native forward_cpu on an identical
        # input; the capture's reconstruction MUST equal it bit-for-bit. We then RETURN the native output,
        # so capture-on generation == capture-off generation BY CONSTRUCTION (no separate off-run needed).
        native = _orig_forward_cpu(self, hid.clone())
        if not torch.equal(out, native):
            d = (out.float() - native.float()).abs().max().item()
            raise RuntimeError(f"[SHARED_CAP] L{lid} reconstruction != native (max|Δ|={d:g}); capture would "
                               "PERTURB generation — refusing (fail-closed).")
        # FAIL-CLOSED: persist FIRST; only mark done after a successful save; raise on any failure.
        tsha = _tok_sha(target)
        snap = {
            "layer_id": lid, "n_rows": int(hid.shape[0]),
            "target_token_sha256": tsha, "target_len": len(target),
            "target_token_ids": [int(x) for x in target],
            "decision_row": int(hid.shape[0]) - 1,   # last token of the completing chunk = next-token position
            "hidden": hid.float().cpu(),
            "gate_up_weight": self.shared_experts.gate_up_proj.weight.detach().cpu(),   # PACKED
            "down_weight": self.shared_experts.down_proj.weight.detach().cpu(),          # PACKED
            "gate_up_weight_dtype": str(self.shared_experts.gate_up_proj.weight.dtype),
            "down_weight_dtype": str(self.shared_experts.down_proj.weight.dtype),
            "fused_experts_out": fe_snap,   # routed output (added in-kernel), snapshotted pre-shared_expert_cpu
            "routed_scaling_factor": float(self.routed_scaling_factor),
            "is_int8": bool(self.shared_experts_is_int8), "is_fp8": bool(self.shared_experts_is_fp8),
            "w1_scale": (w1s.detach().cpu() if torch.is_tensor(w1s) else None),
            "w2_scale": (w2s.detach().cpu() if torch.is_tensor(w2s) else None),
            "block_size": (list(bsz) if bsz is not None else None), "is_vnni": True,
            "out_cpu_combined_noclamp": out.detach().float().cpu(),   # COMBINED shared+scaled-routed (not direct)
            "nonperturb_bitexact_vs_native": True,   # torch.equal(out, native) held at capture
            "swiglu_limit_config": (float(self.shared_experts.swiglu_limit)
                                    if getattr(self.shared_experts, "swiglu_limit", None) is not None else None),
            "note": "out is the COMBINED shared+scaled-routed output; direct shared is NOT recoverable by "
                    "subtraction (rounding). Reconstruct both sides with matched fused_experts_out+scaling.",
        }
        tag = tsha[:12]
        tmp = f"{dst}.{tag}.L{lid}.pt.tmp"
        torch.save(snap, tmp)
        os.replace(tmp, f"{dst}.{tag}.L{lid}.pt")
        _DONE[(lid, tsha)] = True   # only now (per case+layer)
        logger.warning("[SHARED_CAP] L%d captured decision chunk (rows=%d sha=%s, bitexact-vs-native) -> %s.%s.L%d.pt",
                       lid, hid.shape[0], tag, dst, tag, lid)
        return native   # genuine native output (proven == reconstruction) -> zero perturbation

    MoE.forward_cpu = _cap_forward_cpu
    logger.warning("[SHARED_CAP] installed (layers=%s, frozen targets=%d, dst=%s)",
                   sorted(want), len(_frozen_seqs()), dst)
