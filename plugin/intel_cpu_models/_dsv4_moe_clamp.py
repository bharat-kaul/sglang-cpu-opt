"""Private donor build: native packed GEMMs with the DSv4 activation epilogue."""
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


FLAG = "INTEL_CPU_DSV4_MXFP4_CLAMP"


def enabled(layer_id):
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1", "last"):
        raise ValueError(f"{FLAG} must be 0, 1, or last")
    return value == "1" or (value == "last" and layer_id == 42)


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("unsupported SGLang CPU donor source: epilogue anchor changed")
    return text.replace(old, new, 1)


@functools.lru_cache(maxsize=1)
def load():
    import sgl_kernel
    import sglang
    import torch
    from torch.utils.file_baton import FileBaton

    donor = Path(sglang.__file__).parent / "kernels/aot/csrc/cpu"
    library = Path(sgl_kernel.common_ops.__file__).resolve()
    kernel = Path(__file__).parents[1] / "kernels/dsv4_moe_clamp.h"
    files = sorted(donor.rglob("*.h")) + [donor / "moe.cpp", donor / "moe_fp8.cpp"]
    hashes = {str(path.relative_to(donor)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    identity = {
        "donor": hashes, "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "epilogue_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache/torch_extensions"))
    directory = directory / f"dsv4_moe_clamp_{key}"
    directory.mkdir(parents=True, exist_ok=True)
    baton = FileBaton(str(directory / "prepare.lock"))
    while not baton.try_acquire():
        baton.wait()
    try:
        return _build(donor, library, kernel, files, identity, key, directory)
    finally:
        baton.release()


def _build(donor, library, kernel, files, identity, key, directory):
    from torch.utils.cpp_extension import load as build

    # Private copies leave the installed library and checkout unchanged.
    for path in files:
        destination = directory / path.relative_to(donor)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(kernel, directory / kernel.name)
    fp = directory / "moe_fp8.cpp"
    text = _replace_once(fp.read_text(), '#include "moe.h"',
                         '#include "moe.h"\n#include "dsv4_moe_clamp.h"')
    old = "        silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);"
    text = _replace_once(text, old, """        if constexpr (is_mxfp4) {
          dsv4_silu_clamp_mul(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N, limit);
        } else {
""" + old + """
        }""")
    fp.write_text(text)
    main = directory / "moe.cpp"
    text = _replace_once(main.read_text(), "at::Tensor fused_experts_cpu(",
                         "at::Tensor dsv4_fused_experts_cpu(")
    dispatch = "  const CPUActMethod act_func = act_method_from_string(activation, alpha.has_value() && limit.has_value());"
    text = _replace_once(text, dispatch, """
  TORCH_CHECK(moe_comp_method == CPUQuantMethod::MXFP4 && is_vnni &&
              !alpha.has_value() && limit.has_value() && std::isfinite(limit.value()) &&
              limit.value() > 0 && activation.value_or("silu") == "silu",
              "DSv4 clamp requires packed MXFP4, plain SiLU and a positive limit");
""" + dispatch)
    text += """
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_experts", &dsv4_fused_experts_cpu);
  m.def("activation", [](at::Tensor x, double limit) {
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kBFloat16 &&
                x.dim() == 2 && x.is_contiguous() && x.size(1) % 64 == 0,
                "expected contiguous CPU BF16 [M, 2N], N divisible by 32");
    TORCH_CHECK(std::isfinite(limit) && limit > 0, "invalid clamp limit");
    auto out = at::empty({x.size(0), x.size(1) / 2}, x.options());
    int64_t n = out.size(1);
    for (int64_t row = 0; row < x.size(0); ++row) {
      dsv4_silu_clamp_mul(out.data_ptr<at::BFloat16>() + row * n,
          x.data_ptr<at::BFloat16>() + row * 2 * n,
          x.data_ptr<at::BFloat16>() + row * 2 * n + n, n, float(limit));
    }
    return out;
  });
}
"""
    text = '#include "dsv4_moe_clamp.h"\n' + text
    main.write_text(text)
    (directory / "identity.json").write_text(json.dumps(identity, indent=2))
    print(f"DSV4_MXFP4_CLAMP_BUILD {directory}", flush=True)
    return build(
        name=f"dsv4_moe_clamp_{key}", sources=[str(main), str(fp)],
        build_directory=str(directory), extra_include_paths=[str(directory)],
        extra_cflags=["-O3", "-fopenmp", "-mavx512f", "-mavx512bw", "-mavx512dq",
                      "-mavx512vl", "-mavx512bf16", "-mamx-tile", "-mamx-bf16",
                      "-mamx-int8", "-DCPU_CAPABILITY_AVX512",
                      "-Dfused_experts_fp_kernel_impl=dsv4_fused_experts_fp_kernel_impl"],
        extra_ldflags=[str(library), "-fopenmp", "-Wl,-Bsymbolic"],
        verbose=True,
    )


def fused_experts(x, layer, weights, ids):
    rc = layer.moe_runner_config
    limit = rc.swiglu_limit
    if (rc.activation != "silu" or rc.gemm1_alpha is not None
            or rc.gemm1_clamp_limit is not None or limit is None
            or not math.isfinite(float(limit)) or float(limit) <= 0):
        raise ValueError("DSv4 packed clamp requires plain SiLU and a positive swiglu_limit")
    if not getattr(layer, "_dsv4_clamp_logged", False):
        print(f"DSV4_MXFP4_CLAMP layer={getattr(layer, 'layer_id', 'test')} limit={limit}", flush=True)
        layer._dsv4_clamp_logged = True
    return load().fused_experts(
        x, layer.w13_weight, layer.w2_weight, weights, ids, False, 4,
        layer.w13_weight_scale, layer.w2_weight_scale, None, None, None,
        getattr(layer, "w13_weight_bias", None), getattr(layer, "w2_weight_bias", None),
        None, float(limit), True, "silu")
