"""JIT builder for the plugin's custom KDA CPU kernel (adapted from sgl-kernel fla.cpp).

Build-feasibility is PROVEN: this compiles the real AMX gated-delta kernel standalone via
torch.utils.cpp_extension in ~33s and it runs (matches the installed op within bf16/compile
variance). kda_fla.cpp is currently an UNMODIFIED copy (GDN per-head gate); the per-key KDA
port happens in-place here, validated against kda_recurrent.
"""
from __future__ import annotations

import os

from torch.utils.cpp_extension import load

_HERE = os.path.dirname(__file__)
_SGL_CPU_INC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))),
    "sglang", "python", "sglang", "kernels", "aot", "csrc", "cpu",
)
# fall back to the known absolute path if the relative layout differs
if not os.path.isdir(_SGL_CPU_INC):
    _SGL_CPU_INC = "/data/nfs_home/bkaul/sglang/python/sglang/kernels/aot/csrc/cpu"

_CFLAGS = [
    "-O3", "-march=sapphirerapids", "-mamx-tile", "-mamx-bf16",
    "-mavx512bf16", "-mavx512f", "-fopenmp", "-DCPU_CAPABILITY_AVX512",
]


def build(name: str = "kda_fla_jit"):
    return load(
        name=name,
        sources=[os.path.join(_HERE, "kda_fla.cpp")],
        extra_include_paths=[_SGL_CPU_INC],
        extra_cflags=_CFLAGS,
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )


if __name__ == "__main__":
    m = build()
    print(f"built: {m}")
