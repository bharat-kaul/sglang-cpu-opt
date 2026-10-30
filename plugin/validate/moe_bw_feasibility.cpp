// Feasibility microbench for the B=32 MoE-apply bandwidth task (gate §4.2, before any moe.cpp change).
// Tests WHETHER the ~100 GB/s B=32 plateau is the scattered-expert access-pattern ceiling, or memory-level-
// parallelism the kernel leaves on the table (recoverable by prefetch / multi-expert-in-flight).
//
// Allocates MXFP4-packed-shaped expert weights [E, per-expert bytes], streams `active` random experts at T
// threads under several strategies, reports achieved GB/s. Compile:
//   g++ -O3 -march=native -fopenmp moe_bw_feasibility.cpp -o moe_bw_feasibility
// Run (on an EMR node, single socket): numactl --physcpubind=0-57 --membind=0 ./moe_bw_feasibility <T> <active>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

static inline uint64_t stream_region(const uint8_t* p, size_t nbytes, int prefetch_ahead) {
    // AVX-512 streaming read + accumulate (prevents dead-code elimination). Optional SW prefetch.
    __m512i acc = _mm512_setzero_si512();
    size_t n = nbytes / 64;
    for (size_t i = 0; i < n; ++i) {
        if (prefetch_ahead) _mm_prefetch((const char*)(p + (i + prefetch_ahead) * 64), _MM_HINT_T0);
        acc = _mm512_add_epi64(acc, _mm512_loadu_si512((const void*)(p + i * 64)));
    }
    uint64_t s = 0; uint64_t tmp[8]; _mm512_storeu_si512(tmp, acc);
    for (int i = 0; i < 8; ++i) s += tmp[i];
    return s;
}

// Stream + a representative fp4->bf16 DEQUANT of the expert weight (the kernel's dominant per-expert compute).
// Each packed uint8 holds two fp4 nibbles -> two bf16 out. We emulate the unpack+scale cost: load 64B, split
// nibbles, convert to float via a 16-entry LUT, multiply by a block scale, store to a bf16 scratch. This is the
// compute that must hide under the ~235 GB/s stream; measuring stream+dequant tells us the compute-bound rate.
static inline uint64_t stream_dequant_region(const uint8_t* p, size_t nbytes, float* scratch, const float* lut) {
    uint64_t checksum = 0;
    size_t n = nbytes; // bytes
    for (size_t i = 0; i < n; ++i) {
        uint8_t b = p[i];
        int lo = b & 0xF, hi = (b >> 4) & 0xF;
        float a = lut[lo], c = lut[hi];
        scratch[(i & 1023) * 2] = a;              // write into a small rotating scratch (cache-resident)
        scratch[(i & 1023) * 2 + 1] = c;
        checksum += (uint64_t)(a + c);
    }
    return checksum;
}

int main(int argc, char** argv) {
    int T = argc > 1 ? atoi(argv[1]) : 32;
    int active = argc > 2 ? atoi(argv[2]) : 56;
    const int E = 256;
    // real MXFP4 packed shapes (uint8): w13 [E,4096,2048], w2 [E,4096,1024] -> per-expert bytes
    const size_t w13_e = (size_t)4096 * 2048;   // 8.4 MB
    const size_t w2_e  = (size_t)4096 * 1024;   // 4.2 MB
    const size_t per_e = w13_e + w2_e;          // ~12.6 MB
    const size_t total = (size_t)E * per_e;
    omp_set_num_threads(T);
    printf("alloc %.2f GB (E=%d, per-expert %.2f MB), T=%d active=%d\n",
           total / 1e9, E, per_e / 1e6, T, active);
    std::vector<uint8_t> w13((size_t)E * w13_e), w2((size_t)E * w2_e);
    #pragma omp parallel for
    for (size_t i = 0; i < w13.size(); ++i) w13[i] = (uint8_t)(i * 1315423911u);
    #pragma omp parallel for
    for (size_t i = 0; i < w2.size(); ++i) w2[i] = (uint8_t)(i * 2654435761u);

    std::mt19937 rng(0);
    std::vector<int> ids(E); for (int i = 0; i < E; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng); ids.resize(active);
    const double gb = (double)active * per_e / 1e9;

    auto bench = [&](const char* name, int prefetch, int experts_per_thread_interleave) {
        volatile uint64_t sink = 0;
        double best = 1e30;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (experts_per_thread_interleave <= 1) {
                // strategy: parallelize experts across threads (kernel-like: 1 expert = 1 unit of work)
                #pragma omp parallel for schedule(dynamic,1) reduction(+:sink)
                for (int j = 0; j < active; ++j) {
                    int e = ids[j];
                    sink += stream_region(&w13[(size_t)e * w13_e], w13_e, prefetch);
                    sink += stream_region(&w2[(size_t)e * w2_e], w2_e, prefetch);
                }
            } else {
                // strategy: each thread interleaves G experts in flight (more memory-level parallelism)
                int G = experts_per_thread_interleave;
                #pragma omp parallel reduction(+:sink)
                {
                    #pragma omp for schedule(dynamic, G)
                    for (int j = 0; j < active; ++j) {
                        int e = ids[j];
                        // touch w13 then w2; the dynamic chunk of G keeps G experts' streams near each other
                        sink += stream_region(&w13[(size_t)e * w13_e], w13_e, prefetch);
                        sink += stream_region(&w2[(size_t)e * w2_e], w2_e, prefetch);
                    }
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            best = std::min(best, s);
        }
        printf("  %-28s  %.2f ms   %.0f GB/s   (sink=%llu)\n",
               name, best * 1e3, gb / best, (unsigned long long)sink);
    };

    printf("streaming %.3f GB (%d experts) :\n", gb, active);
    bench("kernel-like (1 expert/unit)", 0, 1);
    bench("  + SW prefetch(8)", 8, 1);
    bench("  + interleave G=4/thread", 0, 4);
    bench("  + interleave G=4 + prefetch", 8, 4);

    // stream + scalar fp4->bf16 dequant (the per-expert compute that must hide under the stream)
    {
        float lut[16]; for (int i = 0; i < 16; ++i) lut[i] = (float)(i - 8) * 0.5f;
        volatile uint64_t sink = 0; double best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel reduction(+:sink)
            {
                std::vector<float> scratch(2048);
                #pragma omp for schedule(dynamic,1)
                for (int j = 0; j < active; ++j) {
                    int e = ids[j];
                    sink += stream_dequant_region(&w13[(size_t)e * w13_e], w13_e, scratch.data(), lut);
                    sink += stream_dequant_region(&w2[(size_t)e * w2_e], w2_e, scratch.data(), lut);
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        printf("  %-28s  %.2f ms   %.0f GB/s   (stream+scalar dequant = compute-bound rate; sink=%llu)\n",
               "stream + fp4 dequant", best * 1e3, gb / best, (unsigned long long)sink);
    }
    // contiguous upper bound: stream the first `active` experts as one flat sweep
    {
        volatile uint64_t sink = 0; double best = 1e30;
        size_t flat = (size_t)active * w13_e;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(static) reduction(+:sink)
            for (size_t c = 0; c < flat; c += 64) sink += stream_region(&w13[c], 64, 0);
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        printf("  %-28s  %.2f ms   %.0f GB/s   (contiguous upper bound, w13-only bytes)\n",
               "contiguous sweep", best * 1e3, (flat / 1e9) / best);
    }
    return 0;
}
