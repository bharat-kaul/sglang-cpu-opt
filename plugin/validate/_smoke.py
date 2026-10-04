"""Minimal CPU serve smoke test for the plugin override (run as a FILE, not stdin:
SGLang spawns scheduler subprocesses that re-import __main__).

  SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models \
  PYTHONPATH=<plugin> MODEL_PATH=<path> python _smoke.py
"""

import os


def main():
    import sglang as sgl

    # Pre-import the external plugin so its import-time patches (CPU arg-resolution
    # guards for intel_cpu_models; deterministic-dummy + capture hooks for either
    # package) install BEFORE the engine's resolve_once()/load_model runs.
    _pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    if _pkg:
        __import__(_pkg)

    model = os.environ["MODEL_PATH"]
    tp = int(os.environ.get("TP", "1"))
    device = os.environ.get("DEVICE", "cpu")
    load_format = os.environ.get("LOAD_FORMAT")  # e.g. "dummy" for bring-up
    print(f"[smoke] launching {device} engine: {model} tp={tp} load_format={load_format}", flush=True)
    kw = dict(
        model_path=model, device=device, tp_size=tp,
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=float(os.environ.get("MEM_FRAC", "0.5")),
        log_level=os.environ.get("LOG_LEVEL", "warning"),
    )
    if load_format:
        kw["load_format"] = load_format
    # Depth-reduced PERF PROXY: shrink num_hidden_layers (keep full WIDTH) + truncate the per-layer
    # type lists so the proxy keeps one of every op family (KDA/MLA/dense/MoE). Reads the model's own
    # config so it works for GLM's nested text_config + layer_types/mlp_layer_types lists.
    if os.environ.get("NUM_LAYERS"):
        import json as _json
        n = int(os.environ["NUM_LAYERS"])
        cfg = _json.load(open(os.path.join(model, "config.json")))
        tc = cfg.get("text_config", cfg)
        ov = {"num_hidden_layers": n}
        for lk in ("layer_types", "mlp_layer_types"):
            if isinstance(tc.get(lk), list):
                ov[lk] = tc[lk][:n]
        override = {"text_config": ov} if "text_config" in cfg else ov
        kw["json_model_override_args"] = _json.dumps(override)
        print(f"[smoke] PERF PROXY num_hidden_layers={n} override={override}", flush=True)
    if os.environ.get("MAX_RUNNING"):
        kw["max_running_requests"] = int(os.environ["MAX_RUNNING"])
    # Mamba/linear-attention models (GLM-5 KDA): the default extra_buffer radix-cache
    # strategy needs GPU-FLA; force the CPU-valid no_buffer (page_size=1) via env.
    if os.environ.get("MAMBA_RADIX_STRATEGY"):
        kw["mamba_radix_cache_strategy"] = os.environ["MAMBA_RADIX_STRATEGY"]
    if os.environ.get("PAGE_SIZE"):
        kw["page_size"] = int(os.environ["PAGE_SIZE"])
    # Cap context_length to shrink the DSA/MLA KV pool + mamba cache (sized by max_total_tokens);
    # a short-prompt per-layer capture needs almost none, and the model's native 1M context would
    # over-reserve and OOM a big real-weight CPU load.
    if os.environ.get("CONTEXT_LEN"):
        kw["context_length"] = int(os.environ["CONTEXT_LEN"])
    # Hybrid mamba+DSA page_size clash on CPU: mamba no_buffer radix cache needs
    # page_size=1 but DSA forces 64. Prefix caching of hybrid state is an optimization,
    # not correctness — disable_radix_cache routes to ChunkCache (no MambaComponent),
    # keeping DSA's page_size=64 for the MLA/DSA pool.
    if os.environ.get("DISABLE_RADIX") == "1":
        kw["disable_radix_cache"] = True
    # Prefill-only fingerprint capture doesn't need decode CUDA graphs; disabling them
    # sidesteps the whole class of decode-graph capture asserts (e.g. DSA index_kpool
    # group_topk constraints) that a shrunk tiny config otherwise trips on GPU.
    if os.environ.get("DISABLE_CUDA_GRAPH") == "1":
        kw["disable_cuda_graph"] = True
    e = sgl.Engine(**kw)
    # Prompts/max-new via env so GPU reference and CPU runs use IDENTICAL inputs (fingerprint align).
    # TOKEN_IDS (comma-separated) bypasses the tokenizer entirely so BOTH engines prefill byte-identical
    # token sequences — required for cross-engine per-layer parity when the two tokenizers differ (e.g.
    # a CPU fallback tokenizer that drops the GLM [gMASK]<sop> prefix the GPU container adds).
    max_new = int(os.environ.get("MAX_NEW", "8"))
    token_ids_env = os.environ.get("TOKEN_IDS", "")
    if token_ids_env:
        ids = [int(t) for t in token_ids_env.replace("|", ",").split(",") if t.strip() != ""]
        o = e.generate(input_ids=[ids], sampling_params={"temperature": 0.0, "max_new_tokens": max_new})
        rec = o[0] if isinstance(o, list) else o
        txt = rec["text"] if isinstance(rec, dict) else rec
        print("TOKEN_IDS:", ids, "-> OUT:", repr(txt), flush=True)
    else:
        prompts = os.environ.get("SMOKE_PROMPTS", "The capital of France is|2 + 2 =").split("|")
        for pr in prompts:
            o = e.generate(pr, {"temperature": 0.0, "max_new_tokens": max_new})
            txt = o["text"] if isinstance(o, dict) else o
            print("PROMPT:", repr(pr), "-> OUT:", repr(txt), flush=True)
    e.shutdown()
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    main()
