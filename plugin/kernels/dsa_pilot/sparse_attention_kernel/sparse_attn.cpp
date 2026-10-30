// Standalone EMR specialization of DeepSeek-V4's 64-key online sparse attention.
// Q/KV/P/output: BF16; scores, normalization, and output accumulation: FP32.
#include "sparse_attn.h"
#include <immintrin.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
constexpr int D = 512, Heads = 32, Block = 64;

struct alignas(64) TileConfig {
    uint8_t palette = 1, start = 0, reserved[14] = {};
    uint16_t columns[16] = {};
    uint8_t rows[16] = {};
    TileConfig() {
        for (int i = 0; i < 8; ++i) {
            columns[i] = 64;
            rows[i] = 16;
        }
    }
};
static_assert(sizeof(TileConfig) == 64);

inline void transpose16(__m512i* r) {
    __m512i t[16], u[16], v[16];
    for (int i = 0; i < 16; i += 4) {
        t[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
        t[i + 2] = _mm512_unpacklo_epi32(r[i + 2], r[i + 3]);
        t[i + 3] = _mm512_unpackhi_epi32(r[i + 2], r[i + 3]);
        u[i] = _mm512_unpacklo_epi64(t[i], t[i + 2]);
        u[i + 1] = _mm512_unpackhi_epi64(t[i], t[i + 2]);
        u[i + 2] = _mm512_unpacklo_epi64(t[i + 1], t[i + 3]);
        u[i + 3] = _mm512_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int i = 0; i < 16; i += 8)
        for (int j = 0; j < 4; ++j) {
            v[i + j] = _mm512_shuffle_i32x4(u[i + j], u[i + j + 4], 0x88);
            v[i + j + 4] = _mm512_shuffle_i32x4(u[i + j], u[i + j + 4], 0xdd);
        }
    for (int i = 0; i < 8; ++i) {
        r[i] = _mm512_shuffle_i32x4(v[i], v[i + 8], 0x88);
        r[i + 8] = _mm512_shuffle_i32x4(v[i], v[i + 8], 0xdd);
    }
}

// Cephes range reduction/polynomial, as in the pilot's avx512_exp.h.
// Here inputs are non-positive; explicitly zero masked and underflowed lanes.
inline __m512 exp_negative(__m512 x) {
    const __mmask16 live = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-87.33654475f), _CMP_GE_OQ);
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.33654475f));
    const __m512 n = _mm512_roundscale_ps(
        _mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1)));
    const __m512i power = _mm512_slli_epi32(
        _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    return _mm512_maskz_mov_ps(live, _mm512_mul_ps(y, _mm512_castsi512_ps(power)));
}

// C[32,32] += A[32,K] B[K/2,32,2]. Four independent accumulators.
inline void gemm32(const uint16_t* a, int lda, const uint16_t* b, int ldb,
                   float* c, int ldc, int k, bool add) {
    if (add) {
        _tile_loadd(0, c, ldc * 4);
        _tile_loadd(1, c + 16, ldc * 4);
        _tile_loadd(2, c + 16 * ldc, ldc * 4);
        _tile_loadd(3, c + 16 * ldc + 16, ldc * 4);
    } else {
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    }
    for (int p = 0; p < k; p += 32) {
        _tile_loadd(4, a + p, lda * 2);
        _tile_loadd(5, a + 16 * lda + p, lda * 2);
        _tile_loadd(6, b + (p / 2) * ldb, ldb * 2);
        _tile_loadd(7, b + (p / 2) * ldb + 32, ldb * 2);
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    }
    _tile_stored(0, c, ldc * 4);
    _tile_stored(1, c + 16, ldc * 4);
    _tile_stored(2, c + 16 * ldc, ldc * 4);
    _tile_stored(3, c + 16 * ldc + 16, ldc * 4);
}

struct alignas(64) Scratch {
    uint16_t gathered[Block * D], values[Block * D], query[D * Heads];
    uint16_t probability[Heads * Block];
    float scores[Block * Heads], accumulator[Heads * D];
    float maximum[Heads], denominator[Heads], correction[Heads];
};

void attend(const uint16_t* q, const uint16_t* kv, const int32_t* indices,
            const float* sink, uint16_t* out, int topk, float scale,
            Scratch& s, double* phases) {
    double stamp = phases ? omp_get_wtime() : 0;
    for (int d = 0; d < D; d += 32)
        for (int h = 0; h < Heads; h += 16) {
            __m512i rows[16];
            for (int i = 0; i < 16; ++i)
                rows[i] = _mm512_loadu_si512(q + (h + i) * D + d);
            transpose16(rows);
            for (int i = 0; i < 16; ++i)
                _mm512_store_si512(s.query + (d + 2 * i) * Heads + 2 * h, rows[i]);
        }
    std::fill_n(s.accumulator, Heads * D, 0.f);
    std::fill_n(s.maximum, Heads, -INFINITY);
    std::fill_n(s.denominator, Heads, 0.f);
    if (phases) { phases[0] += omp_get_wtime() - stamp; }

    alignas(64) static constexpr uint16_t even_odd_lo[32] = {
        0,32,1,33,2,34,3,35,4,36,5,37,6,38,7,39,
        8,40,9,41,10,42,11,43,12,44,13,45,14,46,15,47};
    alignas(64) static constexpr uint16_t even_odd_hi[32] = {
        16,48,17,49,18,50,19,51,20,52,21,53,22,54,23,55,
        24,56,25,57,26,58,27,59,28,60,29,61,30,62,31,63};
    const __m512i lo = _mm512_load_si512(even_odd_lo);
    const __m512i hi = _mm512_load_si512(even_odd_hi);
    const __m512 neg_inf = _mm512_set1_ps(-INFINITY);
    const __m512 zero = _mm512_setzero_ps();
    alignas(64) static constexpr uint16_t zeros[D] = {};

    for (int start = 0; start < topk; start += Block) {
        stamp = phases ? omp_get_wtime() : 0;
        bool valid[Block];
        const uint16_t* selected[Block];
        for (int j = 0; j < Block; ++j) {
            const int index = start + j < topk ? indices[start + j] : -1;
            valid[j] = index >= 0;
            selected[j] = valid[j] ? kv + int64_t(index) * D : zeros;
        }
        // Keep several rows in flight without flooding L2 with whole-block prefetches.
        for (int j = 0; j < 8; ++j)
            for (int d = 0; d < D; d += 32)
                _mm_prefetch(reinterpret_cast<const char*>(selected[j] + d), _MM_HINT_T0);
        for (int j = 0; j < Block; j += 2) {
            if (j + 8 < Block)
                for (int d = 0; d < D; d += 32) {
                    _mm_prefetch(reinterpret_cast<const char*>(selected[j + 8] + d), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(selected[j + 9] + d), _MM_HINT_T0);
                }
            for (int d = 0; d < D; d += 32) {
                const __m512i a = _mm512_loadu_si512(selected[j] + d);
                const __m512i b = _mm512_loadu_si512(selected[j + 1] + d);
                _mm512_store_si512(s.gathered + j * D + d, a);
                _mm512_store_si512(s.gathered + (j + 1) * D + d, b);
                _mm512_store_si512(s.values + j * D + d * 2, _mm512_permutex2var_epi16(a, lo, b));
                _mm512_store_si512(s.values + j * D + d * 2 + 32, _mm512_permutex2var_epi16(a, hi, b));
            }
        }
        if (phases) { phases[0] += omp_get_wtime() - stamp; stamp = omp_get_wtime(); }
        // Scores are key-major: pack the small stationary Q, not every KV transpose.
        gemm32(s.gathered, D, s.query, Heads * 2, s.scores, Heads, D, false);
        gemm32(s.gathered + 32 * D, D, s.query, Heads * 2, s.scores + 32 * Heads, Heads, D, false);
        if (phases) { phases[1] += omp_get_wtime() - stamp; stamp = omp_get_wtime(); }
        for (int h = 0; h < Heads; h += 16) {
            const __m512 old = _mm512_load_ps(s.maximum + h);
            __m512 maximum = old;
            for (int j = 0; j < Block; ++j) {
                __m512 v = valid[j] ? _mm512_mul_ps(
                    _mm512_load_ps(s.scores + j * Heads + h), _mm512_set1_ps(scale)) : neg_inf;
                _mm512_store_ps(s.scores + j * Heads + h, v);
                maximum = _mm512_max_ps(maximum, v);
            }
            const __mmask16 active = _mm512_cmp_ps_mask(maximum, neg_inf, _CMP_NEQ_OQ);
            const __m512 safe_max = _mm512_maskz_mov_ps(active, maximum);
            const __m512 correction = exp_negative(_mm512_sub_ps(old, safe_max));
            __m512 total = zero;
            for (int j = 0; j < Block; ++j) {
                __m512 p = exp_negative(_mm512_sub_ps(
                    _mm512_load_ps(s.scores + j * Heads + h), safe_max));
                _mm512_store_ps(s.scores + j * Heads + h, p);
                total = _mm512_add_ps(total, p);
            }
            _mm512_store_ps(s.denominator + h, _mm512_add_ps(total,
                _mm512_mul_ps(_mm512_load_ps(s.denominator + h), correction)));
            _mm512_store_ps(s.maximum + h, maximum);
            _mm512_store_ps(s.correction + h, correction);
        }
        for (int h = 0; h < Heads; h += 16)
            for (int j = 0; j < Block; j += 16) {
                __m512i rows[16];
                for (int i = 0; i < 16; ++i)
                    rows[i] = _mm512_load_si512(s.scores + (j + i) * Heads + h);
                transpose16(rows);
                for (int i = 0; i < 16; ++i)
                    _mm256_store_si256(reinterpret_cast<__m256i*>(s.probability + (h + i) * Block + j),
                                      (__m256i)_mm512_cvtneps_pbh(_mm512_castsi512_ps(rows[i])));
            }
        for (int h = 0; h < Heads; ++h) {
            const __m512 correction = _mm512_set1_ps(s.correction[h]);
            for (int d = 0; d < D; d += 16)
                _mm512_store_ps(s.accumulator + h * D + d,
                    _mm512_mul_ps(_mm512_load_ps(s.accumulator + h * D + d), correction));
        }
        if (phases) { phases[2] += omp_get_wtime() - stamp; stamp = omp_get_wtime(); }
        for (int d = 0; d < D; d += 32)
            gemm32(s.probability, Block, s.values + d * 2, D * 2,
                   s.accumulator + d, D, Block, true);
        if (phases) phases[3] += omp_get_wtime() - stamp;
    }
    stamp = phases ? omp_get_wtime() : 0;
    for (int h = 0; h < Heads; ++h) {
        // Rescale relative to the sink if it is larger, avoiding exp overflow.
        const float delta = sink[h] - s.maximum[h];
        const float correction = delta > 0 ? std::exp(-delta) : 1.f;
        const float denom = delta > 0 ? s.denominator[h] * correction + 1.f
                                     : s.denominator[h] + std::exp(delta);
        const __m512 inv = _mm512_set1_ps(correction / denom);
        for (int d = 0; d < D; d += 16)
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + h * D + d),
                (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(
                    _mm512_load_ps(s.accumulator + h * D + d), inv)));
    }
    if (phases) phases[2] += omp_get_wtime() - stamp;
}
}  // namespace

// Call once before execution; permission covers the process's OpenMP workers.
extern "C" int sparse_attn_emr_init() {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("amx-bf16") || !__builtin_cpu_supports("amx-tile") ||
        !__builtin_cpu_supports("avx512bf16") || !__builtin_cpu_supports("avx512bw"))
        return 1;
    return syscall(SYS_arch_prctl, 0x1023, 18) == 0 ? 0 : 2;
}

// Contiguous Q/O[B,M,H,512], KV[B,S,512], sink[H], indices[B,M,K].
// Finite inputs, H divisible by 32, scale > 0; -1 is the only masked index.
// Returns 3 for invalid arguments, 4 for invalid indices. Output must not alias
// inputs. On index errors output is unspecified. phases[8], when non-null,
// receives summed worker-seconds, then the slowest worker's four phase times.
extern "C" int sparse_attn_emr(
    const uint16_t* q, const uint16_t* kv, const float* sink, const int32_t* indices,
    uint16_t* out, int batch, int queries, int heads, int source, int topk,
    float scale, double* phases) {
    if (!q || !kv || !sink || (!indices && topk != 0) || !out || batch < 1 || queries < 1 ||
        heads < Heads || heads % Heads || source < 1 || topk < 0 ||
        !std::isfinite(scale) || scale <= 0)
        return 3;
    double p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    double critical[4] = {}, longest = 0;
    struct alignas(64) Phase { double seconds[4] = {}; };
    std::vector<Phase> per_thread(phases ? omp_get_max_threads() : 0);
    int invalid = 0;
    const int groups = heads / Heads;
    #pragma omp parallel
    {
        Scratch scratch;
        TileConfig config;
        _tile_loadconfig(&config);
        double times[4] = {};
        #pragma omp for schedule(static)
        for (int64_t task = 0; task < int64_t(batch) * queries * groups; ++task) {
            const int64_t row = task / groups, h = (task % groups) * Heads;
            bool bad = false;
            for (int j = 0; j < topk; ++j)
                bad |= indices[row * topk + j] < -1 || indices[row * topk + j] >= source;
            if (bad) {
                #pragma omp atomic write
                invalid = 1;
                continue;
            }
            attend(q + (row * heads + h) * D, kv + (row / queries) * source * D,
                   topk ? indices + row * topk : indices, sink + h, out + (row * heads + h) * D,
                   topk, scale, scratch, phases ? times : nullptr);
        }
        _tile_release();
        if (phases) std::copy_n(times, 4, per_thread[omp_get_thread_num()].seconds);
    }
    if (phases) {
        for (const auto& entry : per_thread) {
            const double* times = entry.seconds;
            p0 += times[0]; p1 += times[1]; p2 += times[2]; p3 += times[3];
            const double total = times[0] + times[1] + times[2] + times[3];
            if (total > longest) {
                longest = total;
                std::copy_n(times, 4, critical);
            }
        }
        phases[0] = p0; phases[1] = p1; phases[2] = p2; phases[3] = p3;
        std::copy_n(critical, 4, phases + 4);
    }
    return invalid ? 4 : 0;
}
