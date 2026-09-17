"""Measurement hygiene + environment/ISA capture for uArch Performance Probe.

Every reported constant is only as trustworthy as the measurement discipline behind
it, so timing here enforces warmup, best-of-N, and dead-code-elimination guards, and
the environment capture records the ISA + frequency-governor state so a reader can see
whether the run was clean (fixed frequency, expected ISA present).
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import time

_SINK = 0.0


def _keep(x: float) -> None:
    # Fold a scalar from each result into a global so the compiler/runtime cannot
    # elide the "unused" computation being timed.
    global _SINK
    try:
        _SINK += float(x)
    except Exception:
        _SINK += 1.0


def _consume(r) -> None:
    try:
        import torch

        if isinstance(r, torch.Tensor):
            _keep(r.reshape(-1)[0].item())
            return
    except Exception:
        pass
    if isinstance(r, (int, float)):
        _keep(r)
    else:
        _keep(1.0)


def timeit(fn, iters: int = 20, warmup: int = 3) -> dict:
    """Best-of-N wall time (seconds). Warmup excluded; DCE-guarded."""
    for _ in range(warmup):
        _consume(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn()
        dt = time.perf_counter() - t0
        _consume(r)  # consume OUTSIDE the timed region
        times.append(dt)
    times.sort()
    return {"best": times[0], "median": times[len(times) // 2], "iters": iters}


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _cpu_flags() -> set:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    return set(line.split(":", 1)[1].split())
    except Exception:
        pass
    return set()


def numa_nodes() -> int | None:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "NUMA node(s)" in line:
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return None


def _freq_hygiene() -> dict:
    h = {"governor": None, "turbo_disabled": None, "clean": None, "warnings": []}
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            h["governor"] = f.read().strip()
    except Exception:
        pass
    try:
        with open("/sys/devices/system/cpu/intel_pstate/no_turbo") as f:
            h["turbo_disabled"] = f.read().strip() == "1"
    except Exception:
        pass
    if h["governor"] not in (None, "performance"):
        h["warnings"].append(
            f"governor={h['governor']} (not 'performance'): frequencies may vary; "
            "peak/ridge constants may be under-reported"
        )
    if h["turbo_disabled"] is False:
        h["warnings"].append(
            "turbo enabled: throughput may be optimistic and non-reproducible; "
            "pin frequency for repeatable characterization"
        )
    h["clean"] = not h["warnings"]
    return h


def env_report() -> dict:
    flags = _cpu_flags()
    info = {
        "node": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": None,
        "numa_nodes": numa_nodes(),
        "isa": {
            k: (k in flags)
            for k in (
                "avx512f",
                "avx512_bf16",
                "amx_bf16",
                "amx_int8",
                "amx_tile",
                "avx512_vnni",
                "avx_vnni",
            )
        },
        "freq_hygiene": _freq_hygiene(),
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        info["torch"] = None
    return info


def avg(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
