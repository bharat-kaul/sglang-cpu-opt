"""Build a tiny, ARCH-FAITHFUL DeepSeek-V4-Flash config for the CPU kernel-wiring ladder.

Derive-from-real (not hand-written) so every loader-expected field is preserved: load the real
config.json, SHRINK only numeric dims + layer count, keep ALL arch SWITCHES (MLA head_dim/latent,
DSA per-layer compress_ratios incl. dense(0)/c4/c128, indexer, MHC num_hash_layers, MoE routing).
Drop quantization_config so dummy weights load bf16 — isolates the DSA/MHC/MLA/MoE WIRING from the
(separately tested) MXFP4/fp8 bridge. The point of this config: validate the fused C/C++ kernel
wiring (indexer / sparse_attend / compressor / MHC) against the eager-torch oracle in SECONDS.

NL=8 covers every op class: L0-1 dense attn (compress_ratio 0), L2-6 DSA sparse (c4/c128 mix),
L7 dense; MHC on first num_hash_layers; MoE sparse (first_k_dense_replace dense MLPs then sparse).
"""

import json
import os
import shutil
import sys

REAL = "/scratch/bkaul/models/DeepSeek-V4-Flash"
OUT = os.environ.get("TINY_OUT", "/scratch/bkaul/models/dsv4-tiny")
NL = 8  # layers


def shrink(c):
    c["num_hidden_layers"] = NL
    c["hidden_size"] = 512
    c["moe_intermediate_size"] = 128
    c["q_lora_rank"] = 128
    c["o_lora_rank"] = 128
    # MLA invariants — KEEP: single replicated latent KV head, 512-d absorbed latent (head_dim).
    c["num_key_value_heads"] = 1
    c["head_dim"] = 512          # MLA absorbed latent dim (kv_lora) — CPU bmm handles any; keep real
    c["num_attention_heads"] = 16  # shrink head COUNT (a size); >=16 keeps the AMX brgemm N-dim valid
    # Indexer switches: index_head_dim is hardcoded 128 in DSATokenToKVPool — KEEP. Head count >=32
    # so the fused indexer_logits brgemm (packs over index_n_heads) can create its AMX ukernel.
    c["index_head_dim"] = 128
    c["index_n_heads"] = 32
    c["index_topk"] = 512         # at tiny ctx top-k >= seq -> dense indexer (fine for wiring parity)
    # MoE routing: the CPU biased_grouped_topk kernel is hardcoded for the real (n_experts, top_k) —
    # keep REAL counts; only the expert WIDTH (moe_intermediate_size) shrinks.
    c["n_routed_experts"] = 256
    c["num_experts_per_tok"] = 6
    c["n_shared_experts"] = 1
    # MHC (hash-clustering) — keep the switch; num_hash_layers<=NL.
    c["num_hash_layers"] = min(int(c.get("num_hash_layers", 3)), 3)
    c["hc_mult"] = int(c.get("hc_mult", 4))
    c["hc_sinkhorn_iters"] = int(c.get("hc_sinkhorn_iters", 20))
    # DSA per-layer compress schedule: faithful mix of dense(0) + c4 + c128, length == NL.
    c["compress_ratios"] = [0, 0, 4, 128, 4, 128, 4, 0]
    c["sliding_window"] = int(c.get("sliding_window", 128))
    c["num_nextn_predict_layers"] = 0  # skip MTP draft for base bring-up
    c["o_groups"] = min(int(c.get("o_groups", 8)), 8)
    # Keep REAL vocab (we copy the real tokenizer; a shrunk vocab makes token ids overflow the table).
    # bf16 dummy: drop fp8/MXFP4 quant to isolate kernel wiring from the quant bridge.
    c.pop("quantization_config", None)
    return c


def main():
    c = json.load(open(os.path.join(REAL, "config.json")))
    c = shrink(c)
    os.makedirs(OUT, exist_ok=True)
    json.dump(c, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    for f in os.listdir(REAL):
        if f.endswith(".safetensors") or f == "config.json" or f.endswith(".index.json"):
            continue
        src = os.path.join(REAL, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT, f))
    print(f"[tiny] wrote {OUT} (NL={NL}, bf16 dummy)")
    print("[tiny] compress_ratios:", c["compress_ratios"])
    print("[tiny] num_hash_layers:", c["num_hash_layers"], " heads:", c["num_attention_heads"],
          " index_n_heads:", c["index_n_heads"])


if __name__ == "__main__":
    sys.exit(main())
