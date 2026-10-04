// Layout-matched DRAM streaming microbench for the DeepSeek-V4-Flash routed-expert MoE.
//
// PURPOSE: split "access-pattern ceiling" from "kernel overhead" for the W4A16 MoE decode.
// The real kernel at M=32 streams the ACTIVATED experts' weights once each and hits 170 GB/s
// (75% of the 226 GB/s stream_triad ceiling). This probe streams the SAME bytes in the SAME
// expert-slab pattern with NO compute (pure read), so:
//   - if the probe reaches ~226           -> the pattern streams cleanly; the kernel's 25% gap is
//                                            COMPUTE/overlap (fix: prefetch/double-buffer, fuse dequant,
//                                            tiling) -- a fixed-point kernel win.
//   - if the probe also caps near ~170    -> the ACCESS PATTERN is the ceiling (scattered expert reads);
//                                            fix the DATA LAYOUT / token-grouping, not the compute loop.
//
// Patterns (same total bytes each):
//   triad-sweep       : one contiguous sweep of active*expert_bytes (absolute streaming ceiling)
//   contiguous-experts: expert slabs visited in index order (prefetcher-friendly)
//   scattered-experts : expert slabs visited in a random permutation (models top-k routing scatter)
//
// Build (on the target ISA, inside the sbatch):  g++ -O3 -fopenmp -march=native bw_probe.cpp -o bw_probe
// Args: <expert_bytes> <num_experts> <active> <pattern 0=contig 1=scatter 2=triad> <iters>
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Read every byte of [p, p+nbytes) via 512-bit loads, XOR-accumulate (prevents dead-code elim). No heavy compute.
static inline uint64_t read_slab_xor(const uint8_t *p, size_t nbytes) {
    __m512i acc = _mm512_setzero_si512();
    const __m512i *v = reinterpret_cast<const __m512i *>(p);
    size_t n = nbytes / 64;
    for (size_t i = 0; i < n; ++i)
        acc = _mm512_xor_si512(acc, _mm512_load_si512(v + i));
    uint64_t tmp[8];
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(tmp), acc);
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i) r ^= tmp[i];
    return r;
}

int main(int argc, char **argv) {
    size_t expert_bytes = argc > 1 ? strtoull(argv[1], nullptr, 10) : (12ull << 20);
    int num_experts = argc > 2 ? atoi(argv[2]) : 256;
    int active = argc > 3 ? atoi(argv[3]) : num_experts;
    int pattern = argc > 4 ? atoi(argv[4]) : 0;
    int iters = argc > 5 ? atoi(argv[5]) : 6;
    expert_bytes &= ~size_t(63);  // 64B align
    if (active > num_experts) active = num_experts;

    size_t total = expert_bytes * (size_t)num_experts;
    uint8_t *buf = static_cast<uint8_t *>(aligned_alloc(64, total));
    if (!buf) { fprintf(stderr, "alloc %zu bytes failed\n", total); return 1; }
    // Parallel first-touch (one store per 4 KB page) -> NUMA-local placement.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i += 4096) buf[i] = (uint8_t)(i >> 12);

    std::vector<int> order(num_experts);
    for (int i = 0; i < num_experts; ++i) order[i] = i;
    if (pattern == 1) {
        std::mt19937 rng(1234);
        std::shuffle(order.begin(), order.end(), rng);
    }
    order.resize(active);

    int nthreads = omp_get_max_threads();
    volatile uint64_t sink = 0;
    double best_gbs = 0.0;
    double read_gb = expert_bytes * (double)active / 1e9;
    for (int it = 0; it < iters; ++it) {
        uint64_t acc = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        if (pattern == 2) {  // one contiguous sweep
            size_t nb = expert_bytes * (size_t)active;
            #pragma omp parallel reduction(^ : acc)
            {
                int tid = omp_get_thread_num();
                size_t chunk = (nb / nthreads) & ~size_t(63);
                size_t s = chunk * tid;
                size_t e = (tid == nthreads - 1) ? nb : chunk * (tid + 1);
                acc ^= read_slab_xor(buf + s, e - s);
            }
        } else {  // expert slabs visited in `order`; threads split the slab list
            #pragma omp parallel reduction(^ : acc)
            {
                uint64_t a = 0;
                #pragma omp for schedule(static)
                for (int k = 0; k < active; ++k)
                    a ^= read_slab_xor(buf + (size_t)order[k] * expert_bytes, expert_bytes);
                acc ^= a;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        double gbs = read_gb / s;
        if (gbs > best_gbs) best_gbs = gbs;
        sink ^= acc;
    }
    const char *pn = pattern == 2 ? "triad-sweep" : (pattern == 1 ? "scattered-experts" : "contiguous-experts");
    printf("pattern=%-18s active=%3d expert_bytes=%zu read=%.2f GB threads=%d  BEST=%.1f GB/s (%.0f%% of 226 triad)\n",
           pn, active, expert_bytes, read_gb, nthreads, best_gbs, 100.0 * best_gbs / 226.0);
    return (int)(sink & 1);
}
