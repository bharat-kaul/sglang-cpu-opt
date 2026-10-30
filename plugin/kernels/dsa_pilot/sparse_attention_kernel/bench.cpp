#include "sparse_attn.h"
#include <immintrin.h>
#include <omp.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (!std::getenv("SLURM_JOB_ID")) {
        std::fprintf(stderr, "Execute on an EMR Slurm allocation, not the login node.\n");
        return 1;
    }
    const int reps = argc > 1 ? std::atoi(argv[1]) : 100;
    if (reps < 1 || sparse_attn_emr_init()) {
        std::fprintf(stderr, "Invalid repetition count or AMX permission unavailable.\n");
        return 1;
    }
    FILE* control = nullptr;
    FILE* acknowledgement = nullptr;
    if (const char* path = std::getenv("SPARSE_PERF_CTL")) {
        const char* ack = std::getenv("SPARSE_PERF_ACK");
        if (!ack || !(control = std::fopen(path, "w")) ||
            !(acknowledgement = std::fopen(ack, "r"))) {
            std::perror("perf control FIFO");
            return 1;
        }
    }
    auto counters = [&](const char* command) {
        if (!control) return;
        std::fprintf(control, "%s\n", command);
        std::fflush(control);
        char reply[32];
        if (!std::fgets(reply, sizeof(reply), acknowledgement)) {
            std::fprintf(stderr, "Missing perf acknowledgement\n");
            std::exit(1);
        }
    };
    constexpr int B = 32, H = 64, D = 512, S = 1024, K = 512;
    std::vector<uint16_t> q(B * H * D), kv(B * S * D), out(B * H * D);
    std::vector<uint16_t> ring_q, ring_kv, ring_out;
    int ring_slots = 0, cursor = 0;
    std::vector<int32_t> idx(B * K);
    std::vector<float> sink(H);
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    auto bf16 = [](float x) {
        uint32_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    };
    for (auto& x : q) x = bf16(normal(rng));
    for (auto& x : kv) x = bf16(normal(rng));
    for (auto& x : sink) x = normal(rng);
    std::vector<int32_t> permutation(S);
    std::iota(permutation.begin(), permutation.end(), 0);
    for (int b = 0; b < B; ++b) {
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::copy_n(permutation.begin(), K, idx.begin() + b * K);
    }
    // Larger than the socket's 320 MiB LLC. Flush work is outside kernel timing.
    std::vector<float> flush(192 * 1024 * 1024);
    auto evict = [&] {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < flush.size(); ++i) flush[i] += 1.f;
    };
    auto call = [&](double* phases = nullptr) {
        const int slot = ring_slots ? cursor++ % ring_slots : 0;
        int status = sparse_attn_emr(
            ring_slots ? ring_q.data() + slot * q.size() : q.data(),
            ring_slots ? ring_kv.data() + slot * kv.size() : kv.data(), sink.data(), idx.data(),
            ring_slots ? ring_out.data() + slot * out.size() : out.data(),
            B, 1, H, S, K, 1.f / std::sqrt(float(D)), phases);
        if (status) { std::fprintf(stderr, "Kernel error %d\n", status); std::exit(1); }
    };
    std::vector<int> cpus(omp_get_max_threads());
    #pragma omp parallel
    cpus[omp_get_thread_num()] = sched_getcpu();
    std::printf("{\"threads\":%d,\"cpus\":[", omp_get_max_threads());
    for (size_t i = 0; i < cpus.size(); ++i) std::printf("%s%d", i ? "," : "", cpus[i]);
    std::printf("]");
    for (const char* mode : {"hot", "cold", "stream"}) {
        const bool cold = std::strcmp(mode, "cold") == 0;
        if (std::strcmp(mode, "stream") == 0) {
            ring_slots = 32;
            ring_q.resize(q.size() * ring_slots);
            ring_kv.resize(kv.size() * ring_slots);
            ring_out.resize(out.size() * ring_slots);
            #pragma omp parallel for schedule(static)
            for (int slot = 0; slot < ring_slots; ++slot) {
                std::copy(q.begin(), q.end(), ring_q.begin() + slot * q.size());
                std::copy(kv.begin(), kv.end(), ring_kv.begin() + slot * kv.size());
            }
        }
        for (int r = 0; r < (ring_slots ? 64 : 10); ++r) call();
        std::vector<double> elapsed;
        elapsed.reserve(reps);
        if (ring_slots) counters("enable");
        const double loop_start = omp_get_wtime();
        for (int r = 0; r < reps; ++r) {
            if (cold) evict();
            const double start = omp_get_wtime();
            call();
            elapsed.push_back((omp_get_wtime() - start) * 1e6);
        }
        const double loop_seconds = omp_get_wtime() - loop_start;
        if (ring_slots) counters("disable");
        std::sort(elapsed.begin(), elapsed.end());
        std::printf(",\"%s\":{\"median_us\":%.3f,\"min_us\":%.3f,\"max_us\":%.3f,\"loop_seconds\":%.9f,\"calls\":%d}",
                    mode, elapsed[reps / 2], elapsed[0], elapsed.back(), loop_seconds, reps);
        std::vector<double> trace[9];
        for (int r = 0; r < 15; ++r) {
            if (cold) evict();
            double phases[8] = {};
            const double start = omp_get_wtime();
            call(phases);
            trace[0].push_back((omp_get_wtime() - start) * 1e6);
            for (int p = 0; p < 4; ++p)
                trace[p + 1].push_back(phases[p] * 1e6 / omp_get_max_threads());
            for (int p = 4; p < 8; ++p) trace[p + 1].push_back(phases[p] * 1e6);
        }
        for (auto& v : trace) std::sort(v.begin(), v.end());
        std::printf(",\"profile_%s\":{\"wall_us\":%.3f,\"phases_us\":[%.3f,%.3f,%.3f,%.3f],"
                    "\"critical_phases_us\":[%.3f,%.3f,%.3f,%.3f]}",
                    mode, trace[0][7], trace[1][7], trace[2][7], trace[3][7], trace[4][7],
                    trace[5][7], trace[6][7], trace[7][7], trace[8][7]);
    }
    uint64_t checksum = 0;
    for (uint16_t x : out) checksum += x;
    std::printf(",\"checksum\":%llu}\n", static_cast<unsigned long long>(checksum));
    if (control) std::fclose(control);
    if (acknowledgement) std::fclose(acknowledgement);
}
