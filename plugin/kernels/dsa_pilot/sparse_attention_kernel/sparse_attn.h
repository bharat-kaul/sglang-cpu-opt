#ifndef SPARSE_ATTN_EMR_H
#define SPARSE_ATTN_EMR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Linux x86-64 EMR / AMX-BF16 + AVX512-BF16 only. Call once before using the kernel.
// Status: 0 success, 1 unsupported ISA, 2 Linux tile permission denied.
int sparse_attn_emr_init(void);

// Contiguous BF16 bit patterns: q/out[B,M,H,512], kv[B,S,512].
// FP32 sink[H]; int32 indices[B,M,K], with -1 masking a key.
// H must be a positive multiple of 32; B,M,S > 0; K >= 0; scale finite and > 0.
// Inputs must be finite with scores representable in FP32. Output cannot alias
// inputs. Caller owns the buffers, OpenMP thread count, and CPU/memory binding.
// Empty/all-masked rows produce zero. No heap allocation on the unprofiled path.
// Status: 0 success, 3 invalid arguments, 4 invalid index (output unspecified).
// Optional phases[8]: sum of worker-seconds for packing, QK, softmax/output, PV,
// followed by those same four times for the slowest worker. Pass NULL normally.
int sparse_attn_emr(const uint16_t* q, const uint16_t* kv, const float* sink,
                    const int32_t* indices, uint16_t* out, int batch, int queries,
                    int heads, int source, int topk, float scale, double* phases);

#ifdef __cplusplus
}
#endif
#endif
