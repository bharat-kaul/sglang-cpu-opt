"""Experimental scheduling only; never imported by the serving model.

Batch ready windows through the unchanged production producer. Pooling remains
per request inside that producer; norm/RoPE/FP8 work is batched across requests.
"""

import torch


def accumulate_batched(infra, requests, common):
    """Same request tuples/common arguments as the scalar diagnostic adapter."""
    layer, ratio, d, rd, overlap, ape, weight, eps, freqs, fp8, dtype = common
    keys = [(layer, int(pool)) for pool, *_ in requests]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate request slots in compression batch")
    active = []
    for key, (_, kv, score, pos, reset) in zip(keys, requests):
        if reset:
            infra._WIN_CARRY.pop(key, None)
            infra._COMPRESSED_PERSIST.pop(key, None)
        if pos.numel() == 0:
            continue
        carry = infra._WIN_CARRY.get(key)
        if carry is None:
            carry = {"kv": kv.new_zeros(0, kv.shape[1]).float(),
                     "score": score.new_zeros(0, kv.shape[1]).float(),
                     "pos": pos.new_zeros(0).long(), "next_block": 0}
        carry["kv"] = torch.cat([carry["kv"], kv.detach().float()], 0)
        carry["score"] = torch.cat([carry["score"], score.detach().float()], 0)
        carry["pos"] = torch.cat([carry["pos"], pos.detach().reshape(-1).long()], 0)
        infra._WIN_CARRY[key] = carry
        active.append((key, carry))
    while active:
        ready = []
        for key, carry in active:
            nb = int(carry["next_block"])
            if carry["pos"].numel() == 0 or int(carry["pos"][-1]) < (nb + 1) * ratio - 1:
                continue
            base = (nb - 1 if nb >= 1 else nb) * ratio
            sel = (carry["pos"] >= base) & (carry["pos"] < (nb + 1) * ratio)
            kv, score, pos = (carry[name][sel] for name in ("kv", "score", "pos"))
            if int(pos[0]) > base:
                continue
            ready.append((key, carry, nb, kv, score, pos))
        if not ready:
            break
        lengths = [x[3].shape[0] for x in ready]
        comp, bp = infra._compress_prefill_core(
            torch.cat([x[3] for x in ready]), torch.cat([x[4] for x in ready]),
            ape, weight, eps, freqs, torch.cat([x[5] for x in ready]), lengths,
            ratio, d, rd, overlap, fp8, dtype,
        )
        if comp is None or bp is None:
            raise RuntimeError("ready compression windows produced no rows")
        offset = 0
        for (key, carry, nb, *_), length in zip(ready, lengths):
            offset += length // ratio
            row, pos = comp[offset - 1:offset].clone(), bp[offset - 1:offset].clone()
            previous = infra._COMPRESSED_PERSIST.get(key)
            infra._COMPRESSED_PERSIST[key] = (
                (row, pos) if previous is None else
                (torch.cat([previous[0], row]), torch.cat([previous[1], pos]))
            )
            carry["next_block"] = nb + 1
            keep = carry["pos"] >= nb * ratio
            for name in ("kv", "score", "pos"):
                carry[name] = carry[name][keep]
        if offset != comp.shape[0] or offset != bp.shape[0]:
            raise RuntimeError("batched producer row accounting mismatch")
        active = [(key, carry) for key, carry, *_ in ready]
    return [infra._COMPRESSED_PERSIST.get(key) for key in keys]


def accumulate_scalar(infra, requests, common):
    layer, ratio, d, rd, overlap, ape, weight, eps, freqs, fp8, dtype = common
    return [
        infra._accumulate_request_blocks(
            layer, pool, ratio, d, rd, overlap, ape, weight, eps, freqs,
            kv, score, pos, fp8, dtype, reset,
        )
        for pool, kv, score, pos, reset in requests
    ]
