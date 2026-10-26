#!/usr/bin/env python3
"""Offline layer-2 attention BOUNDARY analysis (reviewer-recommended). Loads the production snapshot
(INTEL_CPU_DSV4_BOUNDARY_CAPTURE) and replays an independent reference on IDENTICAL operands to find
the first failing boundary WITHOUT another model run.

Checks:
  1. Attention arithmetic self-consistency: independent softmax(q@K^T*scale, mask, sink)@V vs captured out.
  2. Membership: K_all[n_window:] == comp_kv handoff; compressed coverage count (last query == 291?).
  3. Softmax MASS per source (window / compressed / sink) + logit ranges  <-- reviewer asked for this.
  4. Key magnitudes: window vs compressed (dominance diagnosis, not count).
  5. Window mask structure (sliding-window size per query).

Usage: python plugin/validate/analyze_boundary.py /scratch/bkaul/dsv4_boundary_l2.pt
"""
import sys

import torch


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/bkaul/dsv4_boundary_l2.pt"
    s = torch.load(path, weights_only=False)
    q = s["q"]            # [T,1,H,D]
    K = s["K_all"]        # [nk, D]
    mask = s["mask"]      # [T, nk] bool
    scale = s["softmax_scale"]
    sink = s["attn_sink"]  # [H] or None
    out = s["out"]        # [T,H,hdv]
    n = s["n_window"]
    S = s["S_compressed"]
    comp = s["comp_kv"]   # [S, head_dim]
    bpos = s["bpos"]
    qpos = s["qpos"]
    ratio = s["ratio"]
    T, H, D = s["T"], s["H"], s["D"]
    hdv = s["hdv"]
    nk = K.shape[0]
    print(f"T={T} H={H} D={D} hdv={hdv} nk={nk} n_window={n} S_compressed={S} ratio={ratio}")

    # --- 1. attention self-consistency ---
    qb = q[:, 0, :, :D].reshape(T * H, D)
    sc = (qb.float() @ K[:, :D].float().t()).reshape(T, H, nk) * scale
    sc = sc.masked_fill(~mask.unsqueeze(1), float("-inf"))
    if sink is not None:
        snk = sink.reshape(1, H, 1).float().expand(T, H, 1)
        p = torch.nan_to_num(torch.cat([sc, snk], -1).softmax(-1)[..., :-1], nan=0.0)
    else:
        p = torch.nan_to_num(sc.softmax(-1), nan=0.0)
    ref_out = (p.reshape(T * H, nk).float() @ K[:, :hdv].float()).reshape(T, H, hdv)
    cos = torch.nn.functional.cosine_similarity(ref_out.reshape(1, -1), out.reshape(1, -1)).item()
    maxabs = (ref_out - out).abs().max().item()
    print(f"\n[1] attn self-consistency: cos={cos:.6f} maxabs={maxabs:.3e}  "
          f"({'OK -> arithmetic self-consistent; suspect OPERANDS' if cos > 0.999 else 'MISMATCH -> arithmetic bug'})")

    # --- 2. membership ---
    comp_in_K = K[n:n + S, :comp.shape[1]]
    me = (comp_in_K - comp).abs().max().item() if comp_in_K.shape == comp.shape else -1
    print(f"[2] K_all[n:n+S] == comp_kv: maxabs={me:.3e} (shape {tuple(comp_in_K.shape)} vs {tuple(comp.shape)})")
    last = T - 1
    comp_mask_last = mask[last, n:n + S]
    exp_last = int((qpos[last].item() >= (bpos.long() + ratio - 1)).sum())
    print(f"    last-query compressed blocks visible: got={int(comp_mask_last.sum())} expected(coverage)={exp_last}")
    win_last = int(mask[last, :n].sum())
    print(f"    last-query window keys visible: {win_last}")

    # --- 3. softmax mass per source ---
    win_mass = p[..., :n].sum(-1)              # [T,H]
    comp_mass = p[..., n:n + S].sum(-1)
    sink_mass = 1.0 - win_mass - comp_mass
    print(f"\n[3] softmax MASS (mean over T,H): window={win_mass.mean():.4f} "
          f"compressed={comp_mass.mean():.4f} sink={sink_mass.mean():.4f}")
    print(f"    last-query mass: window={win_mass[last].mean():.4f} compressed={comp_mass[last].mean():.4f} "
          f"sink={sink_mass[last].mean():.4f}")
    print(f"    logit range window=[{sc[...,:n][sc[...,:n]>-1e30].min():.2f},{sc[...,:n].max():.2f}] "
          f"compressed=[{sc[...,n:n+S][sc[...,n:n+S]>-1e30].min():.2f},{sc[...,n:n+S].max():.2f}]")

    # --- 4. key magnitudes ---
    win_norm = K[:n, :D].float().norm(dim=-1).mean().item()
    comp_norm = comp[:, :D].float().norm(dim=-1).mean().item()
    print(f"\n[4] key L2 norm: window={win_norm:.3f} compressed={comp_norm:.3f} "
          f"ratio={comp_norm / max(win_norm,1e-9):.3f}")

    # --- 5. window mask structure ---
    wsz = mask[:, :n].sum(-1)
    print(f"\n[5] window keys/query: min={int(wsz.min())} max={int(wsz.max())} "
          f"(expect ~sliding-window 128 for late queries)")


if __name__ == "__main__":
    main()
