// Monolith THESIS counterfactual: how much of decode's ~60% barrier-spin is recoverable by collapsing
// per-op parallel regions into ONE cooperative region? Simulates a DSV4 layer as a sequence of N small ops
// (each a parallel_for over a modest array, like norms/rope/residual/MHC/gate + the MoE/attn blocks), at 32
// threads, two ways:
//   (A) per-op fork-join  : each op is its own `#pragma omp parallel for`  (current structure)
//   (B) one region        : one `#pragma omp parallel`, each op an in-region `#pragma omp for`  (monolith)
// Reports wall per "layer-step" and the A/B ratio = the fork-join overhead recoverable by region-collapse.
// Build: g++ -O3 -march=native -fopenmp monolith_thesis_poc.cpp -o poc ; run: numactl ... ./poc <threads> <ops>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

int main(int argc, char** argv) {
    int T = argc > 1 ? atoi(argv[1]) : 32;
    int NOPS = argc > 2 ? atoi(argv[2]) : 24;      // ~ops per DSV4 layer that each fork-join
    int NLAYERS = 43;
    size_t OPN = argc > 3 ? atoll(argv[3]) : 4096; // elements per op (small decode op, ~hidden dim)
    omp_set_num_threads(T);
    std::vector<float> buf(OPN * NOPS, 1.0f), out(OPN * NOPS, 0.0f);

    auto op_work = [&](int op, int i) {
        // a little real work per element (like a norm/rope/elementwise), memory + a few flops
        float x = buf[op * OPN + i];
        for (int r = 0; r < 8; ++r) x = x * 1.0001f + 0.5f;
        out[op * OPN + i] = x;
    };

    auto run_A = [&]() {  // per-op fork-join (current)
        for (int layer = 0; layer < NLAYERS; ++layer)
            for (int op = 0; op < NOPS; ++op) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < OPN; ++i) op_work(op, i);
            }
    };
    auto run_B = [&]() {  // one cooperative region per layer (monolith)
        for (int layer = 0; layer < NLAYERS; ++layer) {
            #pragma omp parallel
            {
                for (int op = 0; op < NOPS; ++op) {
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < OPN; ++i) op_work(op, i);
                }
            }
        }
    };

    auto bench = [&](const char* name, auto fn) {
        double best = 1e30;
        for (int rep = 0; rep < 10; ++rep) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        printf("  %-26s  %.3f ms / step  (sink=%.1f)\n", name, best * 1e3, out[0] + out[OPN]);
        return best;
    };

    // warm pool spin env should be set outside (OMP_WAIT_POLICY=active GOMP_SPINCOUNT=inf)
    printf("T=%d  ops/layer=%d  layers=%d  elems/op=%zu\n", T, NOPS, NLAYERS, OPN);
    double a = bench("(A) per-op fork-join", run_A);
    double b = bench("(B) one region/layer", run_B);
    printf("  => region-collapse speedup = %.2fx  (fork-join barrier overhead recovered)\n", a / b);
    return 0;
}
