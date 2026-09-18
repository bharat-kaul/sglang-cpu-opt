"""Isolation test: does mbind(MPOL_INTERLEAVE, MPOL_MF_MOVE) actually MIGRATE already-faulted
pages off node 0 across all NUMA nodes?

Reproduces the model-load pathology cheaply (no 8-min weight load): allocate a big tensor and
fault it while pinned to node 0 (so every page lands on node 0), then mbind-migrate it and check
per-node placement via /proc/self/numa_maps. Prints PASS/FAIL.
"""
import ctypes
import platform

import torch


def _mbind_interleave(t, nnodes):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = [
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_uint,
    ]
    SYS_mbind = 237
    MPOL_INTERLEAVE = 3
    MPOL_MF_MOVE = 1 << 1
    addr = t.data_ptr()
    nbytes = t.numel() * t.element_size()
    aligned = addr & ~4095
    size = nbytes + (addr - aligned)
    nodemask = ctypes.c_ulong((1 << nnodes) - 1)
    maxnode = ctypes.sizeof(nodemask) * 8
    ret = libc.syscall(
        ctypes.c_long(SYS_mbind),
        ctypes.c_void_p(aligned),
        ctypes.c_ulong(size),
        ctypes.c_int(MPOL_INTERLEAVE),
        ctypes.byref(nodemask),
        ctypes.c_ulong(maxnode),
        ctypes.c_uint(MPOL_MF_MOVE),
    )
    return ret, ctypes.get_errno()


def _pages_per_node():
    """Sum N<node>=<pages> across /proc/self/numa_maps -> {node: pages}."""
    per = {}
    with open("/proc/self/numa_maps") as f:
        for line in f:
            for tok in line.split():
                if tok.startswith("N") and "=" in tok:
                    n, _, v = tok[1:].partition("=")
                    if n.isdigit():
                        per[int(n)] = per.get(int(n), 0) + int(v)
    return per


def main():
    assert platform.machine() == "x86_64"
    libnuma = ctypes.CDLL("libnuma.so.1")
    libnuma.numa_num_configured_nodes.restype = ctypes.c_int
    nnodes = int(libnuma.numa_num_configured_nodes())
    print(f"nnodes={nnodes}")
    if nnodes <= 1:
        print("SKIP: single NUMA node, cannot test spread")
        return

    # Pin to node 0 so first-touch lands everything on node 0 (reproduce the pathology).
    libnuma.numa_run_on_node(0)
    n = 2 * 1024 * 1024 * 1024 // 4  # 2 GiB of float32
    t = torch.empty(n, dtype=torch.float32)
    t.fill_(1.0)  # fault every page (on node 0)

    before = _pages_per_node()
    print("BEFORE (pages/node):", {k: before[k] for k in sorted(before)})

    ret, err = _mbind_interleave(t, nnodes)
    print(f"mbind ret={ret} errno={err}")
    t[0] += 1  # touch after migration (no new faults expected; pages already resident)

    after = _pages_per_node()
    print("AFTER  (pages/node):", {k: after[k] for k in sorted(after)})

    node0 = after.get(0, 0)
    total = sum(after.values()) or 1
    frac0 = node0 / total
    spread = sum(1 for k in after if after[k] > total * 0.05)
    print(f"node0 fraction after = {frac0:.2f}; nodes with >5% = {spread}/{nnodes}")
    if frac0 < 0.5 and spread >= max(2, nnodes // 2):
        print("PASS: mbind migrated pages off node0 and spread across nodes")
    else:
        print("FAIL: pages still concentrated -> mbind did NOT migrate")


if __name__ == "__main__":
    main()
