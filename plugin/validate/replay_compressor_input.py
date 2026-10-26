#!/usr/bin/env python3
"""Offline compressor-input REPLAY (reviewer-recommended): load the production compressor-input
snapshot (INTEL_CPU_DSV4_COMP_INPUT_CAPTURE) and independently recompute the compressed KV from the
RAW CHECKPOINT weights (wkv/wgate/ape/norm) applied to the captured x, via the reference model.py
Compressor math. Compare to the production comp_kv -> names the wrong operand.

Isolates:
  A. fused-split: does production kv_score == cat([wkv(x), wgate(x)]) from raw weights?
  B. ape identity: does production ape (post undo-hotfix) == raw checkpoint ape?
  C. full compressed-KV: raw-weight reference vs production comp_kv (pool+norm+rope).

Usage: python plugin/validate/replay_compressor_input.py /scratch/bkaul/dsv4_compin_l2.pt
"""
import sys

import torch

CKPT = "/scratch/bkaul/models/DeepSeek-V4-Flash"
SHARD = "model-00004-of-00046.safetensors"  # layer-2 compressor weights


def load_raw(layer_id):
    from safetensors import safe_open
    out = {}
    with safe_open(f"{CKPT}/{SHARD}", "pt") as t:
        for name in ("wkv", "wgate", "ape", "norm"):
            key = f"layers.{layer_id}.attn.compressor.{name}" + ("" if name == "ape" else ".weight")
            out[name] = t.get_tensor(key).float()
    return out


def overlap_transform(tensor, ratio, d, value):
    b, s, _, _ = tensor.size()
    new = tensor.new_full((b, s, 2 * ratio, d), value)
    new[:, :, ratio:] = tensor[:, :, :, d:]
    new[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new


def apply_rope(x, freqs_real):  # x [S, rope_dim], freqs_real [S, rope_dim]
    rd = x.shape[-1]
    xp = x.reshape(-1, rd // 2, 2)
    f = freqs_real.reshape(-1, rd // 2, 2)
    xr, xi = xp[..., 0], xp[..., 1]
    cr, ci = f[..., 0], f[..., 1]
    return torch.stack([xr * cr - xi * ci, xr * ci + xi * cr], -1).reshape(-1, rd)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/bkaul/dsv4_compin_l2.pt"
    s = torch.load(path, weights_only=False)
    lid, ratio, d, rd = s["layer_id"], s["ratio"], s["head_dim"], s["rope_dim"]
    overlap = s["overlap"]
    coff = 2 if overlap else 1
    cw = coff * d
    x = s["x"]                       # [N, dim]
    kv_score_prod = s["kv_score"]    # [N, 2*cw] production fused projection
    ape_prod = s["ape_prod"]         # [ratio, cw] production (post undo-hotfix)
    comp_prod = s["comp_kv"]         # [S, d] production output
    bpos = s["bpos"].long()
    N = x.shape[0]
    print(f"layer {lid} ratio {ratio} head_dim {d} rope {rd} overlap {overlap} N={N} S={comp_prod.shape[0]}")

    raw = load_raw(lid)
    # --- A0. is the LIVE wkv_gate.weight == raw cat([wkv,wgate])? (weight-transform vs input-space) ---
    if s.get("wkv_gate_weight") is not None:
        wg = s["wkv_gate_weight"]                       # [2*cw, dim] live
        raw_cat = torch.cat([raw["wkv"], raw["wgate"]], dim=0)  # [2*cw, dim]
        if wg.shape == raw_cat.shape:
            cosW = torch.nn.functional.cosine_similarity(wg.reshape(1, -1), raw_cat.reshape(1, -1)).item()
            print(f"[A0] live wkv_gate.weight vs raw cat([wkv,wgate]): cos={cosW:.6f} "
                  f"maxabs={(wg-raw_cat).abs().max():.3e} shape={tuple(wg.shape)}")
            # try per-output-row correspondence (permutation detection)
            if cosW < 0.999:
                # does live row i match some raw row? check if live == raw with input-dim permutation
                d0 = (wg[0] - raw_cat[0]).abs().max().item()
                print(f"     row0 maxabs={d0:.3e}; live norm/row={wg.norm(dim=-1).mean():.3f} raw={raw_cat.norm(dim=-1).mean():.3f}")
        else:
            print(f"[A0] shape mismatch live={tuple(wg.shape)} raw_cat={tuple(raw_cat.shape)}")

    # --- A. fused-split check: production kv_score vs cat([wkv(x), wgate(x)]) ---
    kv_raw = x @ raw["wkv"].t()       # [N, cw]
    sc_raw = x @ raw["wgate"].t()     # [N, cw]
    fused_raw = torch.cat([kv_raw, sc_raw], -1)   # [N, 2*cw]
    cosA = torch.nn.functional.cosine_similarity(kv_score_prod.reshape(1, -1), fused_raw.reshape(1, -1)).item()
    print(f"\n[A] fused kv_score: prod vs raw cat([wkv,wgate]) cos={cosA:.6f} "
          f"maxabs={(kv_score_prod-fused_raw).abs().max():.3e}")
    # also check the split halves individually
    kprod, sprod = kv_score_prod[:, :cw], kv_score_prod[:, cw:2*cw]
    print(f"    kv half cos={torch.nn.functional.cosine_similarity(kprod.reshape(1,-1),kv_raw.reshape(1,-1)).item():.6f}"
          f"  score half cos={torch.nn.functional.cosine_similarity(sprod.reshape(1,-1),sc_raw.reshape(1,-1)).item():.6f}")

    # --- B. ape identity: production ape vs raw checkpoint ape ---
    cosB = torch.nn.functional.cosine_similarity(ape_prod.reshape(1, -1), raw["ape"].reshape(1, -1)).item()
    print(f"\n[B] ape: prod(post-undo-hotfix) vs raw ckpt ape cos={cosB:.6f} "
          f"maxabs={(ape_prod-raw['ape']).abs().max():.3e}")

    # --- C. full compressed KV from RAW weights (reference model.py pool+norm+rope) ---
    remainder = N % ratio
    cutoff = N - remainder
    kv = kv_raw[:cutoff].reshape(1, -1, ratio, cw)
    sc = sc_raw[:cutoff].reshape(1, -1, ratio, cw) + raw["ape"]
    if overlap:
        kv = overlap_transform(kv, ratio, d, 0.0)
        sc = overlap_transform(sc, ratio, d, float("-inf"))
    pooled = (kv * sc.softmax(2)).sum(2).squeeze(0)       # [S, d]
    nb = pooled.to(torch.bfloat16).float()
    nb = nb * torch.rsqrt(nb.square().mean(-1, keepdim=True) + s["norm_eps"]) * raw["norm"]
    comp_ref = nb.to(torch.bfloat16).float()
    if s.get("freqs_at_bpos") is not None:
        fr = s["freqs_at_bpos"].reshape(bpos.shape[0], -1)  # [S, rope_dim]
        comp_ref[:, d - rd:] = apply_rope(comp_ref[:, d - rd:], fr)
    # compare pre-fp8 (production comp_kv is post-fp8 on nope; compare directions)
    cosC = torch.nn.functional.cosine_similarity(comp_prod.reshape(1, -1), comp_ref.reshape(1, -1)).item()
    print(f"\n[C] compressed KV: production vs RAW-weight reference cos={cosC:.6f} "
          f"maxabs={(comp_prod-comp_ref).abs().max():.3e}")
    print(f"    norms: prod={comp_prod.norm(dim=-1).mean():.3f} ref={comp_ref.norm(dim=-1).mean():.3f}")
    print(f"\nFIRST failing boundary: "
          + ("A fused-split" if cosA < 0.999 else "B ape" if cosB < 0.999 else
             "C pool/norm/rope" if cosC < 0.999 else "NONE — compressed KV matches raw-weight reference"))


if __name__ == "__main__":
    main()
