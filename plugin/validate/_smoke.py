"""Minimal CPU serve smoke test for the plugin override (run as a FILE, not stdin:
SGLang spawns scheduler subprocesses that re-import __main__).

  SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models \
  PYTHONPATH=<plugin> MODEL_PATH=<path> python _smoke.py
"""

import os


def main():
    import sglang as sgl

    model = os.environ["MODEL_PATH"]
    tp = int(os.environ.get("TP", "1"))
    print(f"[smoke] launching CPU engine: {model} tp={tp}", flush=True)
    e = sgl.Engine(
        model_path=model, device="cpu", tp_size=tp,
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=float(os.environ.get("MEM_FRAC", "0.5")),
        log_level="warning",
    )
    for pr in ["The capital of France is", "2 + 2 ="]:
        o = e.generate(pr, {"temperature": 0.0, "max_new_tokens": 8})
        txt = o["text"] if isinstance(o, dict) else o
        print("PROMPT:", repr(pr), "-> OUT:", repr(txt), flush=True)
    e.shutdown()
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    main()
