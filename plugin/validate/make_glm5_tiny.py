"""Build a tiny, ARCH-FAITHFUL Glm5Next config for the CPU bring-up ladder.

Derive-from-real (not hand-written) so every field the loader expects is preserved: load the
real config.json, SHRINK only the numeric dims + layer count, keep ALL arch switches (hybrid
layer_types, mhc, linear_attn_config, index_*, moe routing, first_k_dense_replace). Drop the
fp8 quantization_config so dummy weights load as bf16 — this isolates the KDA/MHC/MLA/MoE
WIRING from the (separately REUSE-tested) fp8 bridge. 8 layers cover every op class:
  L0-2 KDA + dense MLP (first_k_dense_replace=3), L3 DSA full-attn + sparse MoE,
  L4-6 KDA + sparse MoE, L7 DSA full-attn + sparse MoE; MHC on all.
"""

import json
import os
import shutil
import sys

REAL = "/scratch/bkaul/models/GLM-5.3-Flash"
OUT = os.environ.get("TINY_OUT", "/scratch/bkaul/models/glm5-tiny")
NL = 8  # layers


def shrink(c):
    tc = c["text_config"]
    tc["num_hidden_layers"] = NL
    tc["hidden_size"] = 256
    tc["intermediate_size"] = 128
    tc["moe_intermediate_size"] = 64
    # The CPU biased_grouped_topk kernel is hardcoded for the real models' (num_experts, top_k)
    # — "Unexpected num_experts: 16" / "Unexpected topk: 2". So keep REAL n_routed_experts=288 +
    # top-8; only the expert WIDTH (moe_intermediate_size) and the layer COUNT are shrinkable.
    tc["n_routed_experts"] = 288
    tc["num_experts_per_tok"] = 8
    tc["n_shared_experts"] = 1
    tc["n_group"] = 1
    tc["topk_group"] = 1
    tc["num_attention_heads"] = 4
    tc["num_key_value_heads"] = 4
    tc["q_lora_rank"] = 64
    tc["kv_lora_rank"] = 64
    tc["qk_nope_head_dim"] = 32
    tc["qk_rope_head_dim"] = 0
    tc["v_head_dim"] = 32
    tc["index_head_dim"] = 128  # DSATokenToKVPool hardcodes index_head_dim==128
    tc["index_n_heads"] = 2
    tc["index_topk"] = 8
    tc["index_kpool"] = 2
    # Keep the REAL vocab_size — we copy the real tokenizer, so a shrunk vocab makes
    # token ids exceed the embedding table (IndexError). Embedding/lm_head stay cheap
    # with dummy weights at hidden_size=256.
    tc["first_k_dense_replace"] = 3
    tc["num_nextn_predict_layers"] = 0  # skip MTP draft for base bring-up
    la = tc["linear_attn_config"]
    la["num_heads"] = 4
    la["head_dim"] = 32
    la["kda_layers"] = [0, 1, 2, 4, 5, 6]
    la["full_attn_layers"] = [3, 7]
    tc["layer_types"] = (
        ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
        + ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
    )
    tc["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * 5
    tc["indexer_types"] = ["full"] * NL
    # bf16 dummy: no fp8 quant (isolate wiring from the fp8 bridge).
    c.pop("quantization_config", None)
    # shrink the vision tower so __init__ is cheap; text bring-up never runs it.
    vc = c.get("vision_config")
    if isinstance(vc, dict):
        for k, v in list(vc.items()):
            if isinstance(v, int) and v > 64 and k not in ("image_size", "patch_size", "spatial_merge_size", "temporal_patch_size", "in_channels"):
                vc[k] = 64
        vc["depth"] = min(vc.get("depth", 2), 2)
    return c


def main():
    c = json.load(open(os.path.join(REAL, "config.json")))
    c = shrink(c)
    os.makedirs(OUT, exist_ok=True)
    json.dump(c, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    # tokenizer + any aux configs (not weights) from the real dir.
    for f in os.listdir(REAL):
        if f.endswith(".safetensors") or f == "config.json" or f.endswith(".index.json"):
            continue
        src = os.path.join(REAL, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT, f))
    print(f"[tiny] wrote {OUT} (NL={NL}, bf16 dummy, vision shrunk)")
    print("[tiny] layer_types:", c["text_config"]["layer_types"])
    print("[tiny] mlp_layer_types:", c["text_config"]["mlp_layer_types"])


if __name__ == "__main__":
    sys.exit(main())
