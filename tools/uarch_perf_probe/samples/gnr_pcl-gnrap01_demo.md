# uArch Performance Probe report — pcl-gnrap01.sc.intel.com

- probe_version: 0.1  ·  torch: 2.12.0+cpu  ·  NUMA nodes: 6  ·  cores: 512
- ISA: avx512f, avx512_bf16, amx_bf16, amx_int8, amx_tile, avx512_vnni, avx_vnni
- freq hygiene: governor=performance turbo_disabled=False clean=False
  - ⚠ turbo enabled: throughput may be optimistic and non-reproducible; pin frequency for repeatable characterization

## Compute peak (achieved GEMM)
- bfloat16: 30921.2 GF/s
- float32: 8484.4 GF/s
- int8: 11661.3 GF/s

## Memory
- DRAM triad BW: 2151.6 GB/s
  -   0.016 MB: 12.3 GB/s
  -   0.128 MB: 1.5 GB/s
  -     0.5 MB: 5.4 GB/s
  -       2 MB: 17.6 GB/s
  -       8 MB: 65.2 GB/s
  -      32 MB: 273.9 GB/s
  -     128 MB: 2082.5 GB/s
  -     512 MB: 231.8 GB/s

## NUMA / SNC
- domains: 6  ·  local BW: 226.3 GB/s  ·  remote BW: 138.2 GB/s

## Derived kernel knobs
```json
{
  "one_tp_rank_per_domain": true,
  "n_tp_ranks_hint": 6,
  "remote_bw_penalty": 0.611,
  "per_domain_bw_gbps": 226.3,
  "prefer_amx_stage_M_ge": 1,
  "cores_to_saturate_bw": 64,
  "ridge_flops_per_byte": {
    "bfloat16": 14.37,
    "float32": 3.94,
    "int8": 5.42
  }
}
```
