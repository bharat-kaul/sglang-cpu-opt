"""Private donor build for the SHARED-expert clamp diagnostic (reviewer step 1-2).

Mirrors the routed `_dsv4_moe_clamp` donor recipe (native packed GEMMs, private source copies, link
against the installed common_ops), but ALSO patches the SHARED fp8 kernel's SwiGLU activation and exposes
`shared_experts` (= the native `shared_expert_cpu`, renamed) plus a runtime `set_shared_clamp(limit)` toggle.

With the toggle OFF (limit < 0) the shared kernel runs the native `silu_and_mul_stub` and MUST reproduce
the production `shared_expert_cpu` output bit-exact. With it ON (limit > 0) the shared SwiGLU is clamped
(`dsv4_silu_clamp_mul`, GPU semantics). Everything else — the fp8 GEMMs, rounding, routed add, storage —
is the untouched native path, so any clamp-on/off delta is attributable ONLY to the clamp.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"unsupported SGLang CPU donor source: anchor count={text.count(old)} != 1")
    return text.replace(old, new, 1)


@functools.lru_cache(maxsize=1)
def load(precision_diagnostic=False):
    import sgl_kernel
    import sglang
    import torch
    from torch.utils.file_baton import FileBaton

    donor = Path(sglang.__file__).parent / "kernels/aot/csrc/cpu"
    library = Path(sgl_kernel.common_ops.__file__).resolve()
    epilogue = Path(__file__).parents[1] / "kernels/dsv4_moe_clamp.h"
    shared_h = Path(__file__).parents[1] / "kernels/dsv4_shared_clamp.h"
    quant_h = Path(__file__).parents[1] / "kernels/dsv4_shared_quant_diagnostic.h"
    files = sorted(donor.rglob("*.h")) + [donor / "moe.cpp", donor / "moe_fp8.cpp"]
    hashes = {str(path.relative_to(donor)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    identity = {
        "donor": hashes, "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "epilogue_sha256": hashlib.sha256(epilogue.read_bytes()).hexdigest(),
        "shared_h_sha256": hashlib.sha256(shared_h.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "variant": "shared_clamp",
    }
    if precision_diagnostic:
        identity["quant_h_sha256"] = hashlib.sha256(quant_h.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache/torch_extensions"))
    directory = directory / f"dsv4_shared_clamp_{key}"
    directory.mkdir(parents=True, exist_ok=True)
    baton = FileBaton(str(directory / "prepare.lock"))
    while not baton.try_acquire():
        baton.wait()
    try:
        return _build(donor, library, epilogue, shared_h, files, identity, key, directory,
                      quant_h if precision_diagnostic else None)
    finally:
        baton.release()


def _build(donor, library, epilogue, shared_h, files, identity, key, directory, quant_h=None):
    from torch.utils.cpp_extension import load as build

    for path in files:
        destination = directory / path.relative_to(donor)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(epilogue, directory / epilogue.name)
    shutil.copyfile(shared_h, directory / shared_h.name)
    if quant_h is not None:
        shutil.copyfile(quant_h, directory / quant_h.name)

    # moe_fp8.cpp: include the shared-clamp header and patch the SHARED fp8 activation (6-space anchor,
    # distinct from the routed 8-space one). Leave the routed path untouched (we do not call it here).
    fp = directory / "moe_fp8.cpp"
    text = _replace_once(fp.read_text(), '#include "moe.h"',
                         '#include "moe.h"\n#include "dsv4_shared_clamp.h"')
    shared_block = """  // stage 1.5: intermediate_cache1 = silu(intermediate_cache0)
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
    }
  });"""
    text = _replace_once(text, shared_block, """  // stage 1.5: intermediate_cache1 = silu(intermediate_cache0)  [DSv4 shared-clamp donor]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      if (dsv4_shared_clamp_limit_ref() > 0.0f) {
        dsv4_silu_clamp_mul(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N,
                            dsv4_shared_clamp_limit_ref());
      } else {
        silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
      }
    }
  });""")
    if quant_h is not None:
        text = _replace_once(text, '#include "dsv4_shared_clamp.h"',
                             '#include "dsv4_shared_quant_diagnostic.h"')
        start = text.index("void shared_expert_fp8_kernel_impl(")
        prefix, shared = text[:start], text[start:]
        shared = _replace_once(shared, "  // stage 1: intermediate_cache0 = hidden_states @ w1",
                              """  std::vector<scalar_t> quant_input;
  if (dsv4_shared_quantize_ref()) {
    quant_input.resize(M * K);
    dsv4_shared_quantize_activation(quant_input.data(), input, M, K);
    input = quant_input.data();
  }
  // stage 1: intermediate_cache0 = hidden_states @ w1""")
        shared = _replace_once(shared, "  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2",
                              """  if (dsv4_shared_quantize_ref()) {
    dsv4_shared_quantize_activation(ic1, ic1, M, N);
  }
  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2""")
        text = prefix + shared
    fp.write_text(text)

    # moe.cpp: rename shared_expert_cpu -> dsv4_shared_expert_cpu so pybind binds the LOCAL (patched)
    # definition; append a PYBIND module exposing it + the runtime clamp toggle.
    main = directory / "moe.cpp"
    text = _replace_once(main.read_text(), "at::Tensor shared_expert_cpu(",
                         "at::Tensor dsv4_shared_expert_cpu(")
    text = '#include "dsv4_shared_clamp.h"\n' + text + """
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_experts", &dsv4_shared_expert_cpu);
  m.def("set_shared_clamp", [](double limit) { dsv4_shared_clamp_limit_ref() = (float)limit; });
  m.def("get_shared_clamp", []() { return (double)dsv4_shared_clamp_limit_ref(); });
}
"""
    if quant_h is not None:
        text = _replace_once(text, '#include "dsv4_shared_clamp.h"',
                             '#include "dsv4_shared_quant_diagnostic.h"')
        text = _replace_once(text, '  m.def("shared_experts", &dsv4_shared_expert_cpu);',
                             """  m.def("shared_experts", &dsv4_shared_expert_cpu);
  m.def("set_shared_quantize", [](bool active) { dsv4_shared_quantize_ref() = active; });
  m.def("quantize_activation", &dsv4_shared_quantize_probe);""")
    main.write_text(text)
    (directory / "identity.json").write_text(json.dumps(identity, indent=2))
    print(f"DSV4_SHARED_CLAMP_BUILD {directory}", flush=True)
    return build(
        name=f"dsv4_shared_clamp_{key}", sources=[str(main), str(fp)],
        build_directory=str(directory), extra_include_paths=[str(directory)],
        extra_cflags=["-O3", "-fopenmp", "-mavx512f", "-mavx512bw", "-mavx512dq",
                      "-mavx512vl", "-mavx512bf16", "-mamx-tile", "-mamx-bf16", "-mamx-int8",
                      "-DCPU_CAPABILITY_AVX512",
                      "-Dfused_experts_fp_kernel_impl=dsv4_fused_experts_fp_kernel_impl",
                      "-Dshared_expert_fp8_kernel_impl=dsv4_shared_expert_fp8_kernel_impl"],
        extra_ldflags=[str(library), "-fopenmp", "-Wl,-Bsymbolic"],
        verbose=True,
    )


# --- Approach (A) wiring: swap the shared-expert op for the clamped donor (reviewer step 2) ----------
# Gated by INTEL_CPU_DSV4_SHARED_CLAMP (0=off/native, 1=on). When on, torch.ops.sgl_kernel.shared_expert_cpu
# is replaced by the donor's clamped kernel for ALL MoE layers (hash routing / forward_cpu untouched). An
# optional control file (INTEL_CPU_DSV4_SHARED_CLAMP_CTL) toggles clamp per-arm at runtime so a single
# model load can run the OFF (native, matched baseline) and ON arms. NO eager fallback: if the donor cannot
# build, load() raises and the enabled correction fails explicitly.
_SWAP_INSTALLED = False


def install_op_swap(limit: float = 10.0):
    global _SWAP_INSTALLED
    import os
    v = os.environ.get("INTEL_CPU_DSV4_SHARED_CLAMP", "0")
    quant = os.environ.get("INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC", "0")
    if quant not in ("0", "1") or (quant == "1" and v != "1"):
        raise ValueError("shared quant diagnostic must be 0/1 and requires shared clamp ON")
    if v not in ("0", "1"):
        raise ValueError("INTEL_CPU_DSV4_SHARED_CLAMP must be 0 or 1")
    if v != "1" or _SWAP_INSTALLED:
        return
    import logging
    import torch
    import sgl_kernel.common_ops  # noqa: F401 — force op registration
    logger = logging.getLogger(__name__)
    so = load(precision_diagnostic=quant == "1")
    if quant == "1":
        so.set_shared_quantize(True)
    ns = torch.ops.sgl_kernel
    orig = ns.shared_expert_cpu
    ctl = os.environ.get("INTEL_CPU_DSV4_SHARED_CLAMP_CTL", "")

    def wrap(*args):
        active = True
        if ctl:
            try:
                active = open(ctl).read().strip() == "1"
            except FileNotFoundError:
                active = True  # default to the correction when the control file is absent
        if not active:
            return orig(*args)  # explicit OFF arm (native) — NOT an error fallback
        so.set_shared_clamp(float(limit))
        return so.shared_experts(*args)

    ns.shared_expert_cpu = wrap
    _SWAP_INSTALLED = True
    logger.warning("[SHARED_CLAMP] op-swap installed (limit=%s, ctl=%s)", limit, ctl or "<always-on>")
    if quant == "1":
        logger.warning("[SHARED_QUANT_DIAGNOSTIC] CPU FP8 activation roundtrip enabled; not native GPU arithmetic")
