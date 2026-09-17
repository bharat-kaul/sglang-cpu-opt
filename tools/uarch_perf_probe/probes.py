"""Probe library for uArch Performance Probe.

Each probe emits a specific machine CONSTANT tied to a kernel-authoring KNOB (see the
uarch-perf-probe SKILL). Probes are individually try/excepted by the runner so one
failure never aborts the suite; each returns a dict with a "status" field. All timing
goes through hygiene.timeit (warmup + best-of-N + DCE guard).

These are torch/oneDNN-level probes (they exercise the same AMX/BRGEMM stack our
kernels dispatch to, so the numbers are *achievable* peaks, not datasheet peaks).
Raw-intrinsic native probes (cycle-accurate dpbf16ps, tile-op latency) are a documented
C++ extension point — see SKILL "native probes".
"""
from __future__ import annotations

import os
import subprocess

from . import hygiene


def probe_compute_peak(n: int = 4096, dtypes=("bfloat16", "float32")) -> dict:
    """Achieved GEMM throughput per dtype -> compute-peak asymptote of the roofline."""
    import torch

    peak = {}
    for dt in dtypes:
        try:
            tdt = getattr(torch, dt)
            a = torch.randn(n, n).to(tdt)
            b = torch.randn(n, n).to(tdt)
            t = hygiene.timeit(lambda a=a, b=b: a @ b, iters=10, warmup=3)
            peak[dt] = {"gflops": 2 * n**3 / t["best"] / 1e9, "n": n}
        except Exception as e:  # noqa: BLE001
            peak[dt] = {"error": str(e)}
    # int8 matmul (2x compute ceiling if the ISA supports it)
    try:
        import torch

        if hasattr(torch, "_int_mm"):
            a = torch.randint(-8, 7, (n, n), dtype=torch.int8)
            b = torch.randint(-8, 7, (n, n), dtype=torch.int8).contiguous()
            t = hygiene.timeit(lambda a=a, b=b: torch._int_mm(a, b), iters=10, warmup=3)
            peak["int8"] = {"gops": 2 * n**3 / t["best"] / 1e9, "n": n}
    except Exception as e:  # noqa: BLE001
        peak["int8"] = {"error": str(e)}
    return {"status": "ok", "peak": peak}


def probe_memory_bandwidth(mb_list=(0.016, 0.128, 0.5, 2, 8, 32, 128, 512)) -> dict:
    """Cache-ladder read BW vs footprint (knees reveal L1/L2/L3) + DRAM triad BW."""
    import torch

    ladder = []
    for mb in mb_list:
        try:
            n = max(1024, int(mb * 1024 * 1024 / 4))
            x = torch.randn(n)
            t = hygiene.timeit(lambda x=x: x.sum(), iters=20, warmup=5)
            ladder.append({"mb": mb, "bw_gbps": n * 4 / t["best"] / 1e9})
        except Exception as e:  # noqa: BLE001
            ladder.append({"mb": mb, "error": str(e)})
    triad = None
    try:
        n = 64 * 1024 * 1024 // 4  # 64 MB f32 -> DRAM
        a = torch.randn(n)
        b = torch.randn(n)
        c = torch.randn(n)
        t = hygiene.timeit(lambda: torch.add(b, c, alpha=2.0, out=a), iters=10, warmup=3)
        triad = 3 * n * 4 / t["best"] / 1e9  # 2 read + 1 write
    except Exception as e:  # noqa: BLE001
        return {"status": "ok", "cache_ladder": ladder, "dram_triad_error": str(e)}
    return {"status": "ok", "cache_ladder": ladder, "dram_triad_bw_gbps": triad}


def probe_roofline_ridge(compute: dict, memory: dict) -> dict:
    """Ridge point (flops/byte) = measured compute-peak / measured DRAM BW, per dtype."""
    bw = memory.get("dram_triad_bw_gbps")
    ridge = {}
    if bw:
        for dt, v in compute.get("peak", {}).items():
            g = v.get("gflops") or v.get("gops")
            if g:
                ridge[dt] = round(g / bw, 2)
    return {
        "status": "ok" if ridge else "skipped",
        "ridge_flops_per_byte": ridge,
        "note": "ridge = compute_peak / dram_bw (both measured); below-ridge AI => memory-bound",
    }


def probe_gather_crossover(K=128, N=512, Ms=(1, 2, 4, 8, 16, 32, 64), pool=8192) -> dict:
    """Gather(index_select)+GEMM vs contiguous GEMM across M -> stage-then-BRGEMM crossover.

    Proxy for 'does staging gathered rows for AMX beat per-column tinygemm'. The real
    AMX-stage-vs-tinygemm decision is a C++ native probe; this captures the gather-cost
    trend at the torch level.
    """
    import torch

    try:
        b = torch.randn(K, N).to(torch.bfloat16)
        poolT = torch.randn(pool, K).to(torch.bfloat16)
        sweep = []
        for M in Ms:
            idx = torch.randint(0, pool, (M,))
            a = torch.randn(M, K).to(torch.bfloat16)
            t_mm = hygiene.timeit(lambda a=a, b=b: a @ b, iters=50, warmup=5)["best"]

            def gm(idx=idx, poolT=poolT, b=b):
                return poolT.index_select(0, idx) @ b

            t_gm = hygiene.timeit(gm, iters=50, warmup=5)["best"]
            sweep.append(
                {"M": M, "gemm_s": t_mm, "gather_gemm_s": t_gm, "overhead_frac": (t_gm - t_mm) / t_mm}
            )
        cross = next((r["M"] for r in sweep if r["overhead_frac"] < 0.25), None)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    return {
        "status": "ok",
        "sweep": sweep,
        "prefer_stage_M_ge": cross,
        "note": "smallest M where gather overhead < 25% of the GEMM (torch proxy)",
    }


def probe_threading(n: int = 4096, thread_list=None) -> dict:
    """Compute + BW scaling vs thread count -> cores-to-saturate-BW, grain guidance."""
    import torch

    maxt = os.cpu_count() or 8
    if thread_list is None:
        thread_list = sorted({1, 2, 4, 8, 16, 32, min(maxt, 64), maxt})
    orig = torch.get_num_threads()
    try:
        a = torch.randn(n, n).to(torch.bfloat16)
        b = torch.randn(n, n).to(torch.bfloat16)
        big = 64 * 1024 * 1024 // 4
        x = torch.randn(big)
        y = torch.randn(big)
        z = torch.randn(big)
        comp, mem = [], []
        for th in thread_list:
            torch.set_num_threads(th)
            tc = hygiene.timeit(lambda a=a, b=b: a @ b, iters=8, warmup=2)["best"]
            tm = hygiene.timeit(lambda: torch.add(y, z, alpha=2.0, out=x), iters=8, warmup=2)["best"]
            comp.append({"threads": th, "gflops": 2 * n**3 / tc / 1e9})
            mem.append({"threads": th, "bw_gbps": 3 * big * 4 / tm / 1e9})
        maxbw = max(m["bw_gbps"] for m in mem)
        sat = next((m["threads"] for m in mem if m["bw_gbps"] >= 0.9 * maxbw), None)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    finally:
        torch.set_num_threads(orig)
    return {
        "status": "ok",
        "compute_scaling": comp,
        "memory_scaling": mem,
        "cores_to_saturate_bw": sat,
    }


_NUMA_SNIPPET = (
    "import torch,time;"
    "n={mb}*1024*1024//4;"
    "a=torch.randn(n);b=torch.randn(n);c=torch.randn(n);"
    "[torch.add(b,c,alpha=2.0,out=a) for _ in range(3)];"
    "t0=time.perf_counter();"
    "[torch.add(b,c,alpha=2.0,out=a) for _ in range(10)];"
    "dt=(time.perf_counter()-t0)/10;"
    "print(3*n*4/dt/1e9)"
)


def probe_numa_snc(size_mb: int = 256, timeout: int = 60) -> dict:
    """Per-(cpu-domain, mem-domain) triad BW matrix -> SNC domain count, local/remote BW.

    This is what regenerates the 'one TP rank per domain + first-touch' rule and the
    per-domain bandwidth used to bake SNC into the roofline. Spawns numactl subprocesses.
    """
    n_nodes = hygiene.numa_nodes()
    if not n_nodes or not hygiene.have("numactl"):
        return {"status": "skipped", "reason": "numactl or NUMA topology unavailable"}
    snippet = _NUMA_SNIPPET.format(mb=size_mb)
    matrix = [[None] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            try:
                out = subprocess.run(
                    ["numactl", f"--cpunodebind={i}", f"--membind={j}", "python", "-c", snippet],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                matrix[i][j] = float(out.stdout.strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                matrix[i][j] = None
    local = [matrix[i][i] for i in range(n_nodes)]
    remote = [matrix[i][j] for i in range(n_nodes) for j in range(n_nodes) if i != j]
    return {
        "status": "ok",
        "n_domains": n_nodes,
        "bw_matrix_gbps": matrix,
        "local_bw_gbps": hygiene.avg(local),
        "remote_bw_gbps": hygiene.avg(remote),
    }
