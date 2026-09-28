"""Numeric parity gate for the native MXFP4 W4A16 CPU MoE kernel.

This is the correctness proof the coverage-gate's "COVERED" label does NOT provide: an
op whose stored weights are FP4 rides a bf16/fp8 donor kernel ONLY through a fp4->bf16
dequant+repack step, and that step must be verified numerically, not assumed.

It checks that `torch.ops.sgl_kernel.fused_experts_cpu(..., CPUQuantMethod.MXFP4, ...)`
(the exact call the plugin's MXFP4 MoE path makes) agrees with an INDEPENDENT torch
reference that dequantizes the SAME packed fp4 weights + e8m0 group-32 scales and runs a
dense top-k SwiGLU. The reference mirrors csrc/cpu/vec.h::cvt_mxfp4_e2m1_bf16_intrinsic_lut:
standard OCP e2m1 LUT, low-nibble-first packing, per-32 e8m0 scale = 2^(byte-127).

Run on an AMX node (needs the CPU AMX kernel):
  PYTHONPATH=<plugin> srun ... python plugin/validate/test_mxfp4_moe_cpu.py
"""
import torch

# Standard OCP e2m1: nibble -> value (bit3 = sign, bits2..0 = magnitude {0,.5,1,1.5,2,3,4,6}).
# vec.h encodes this via a reversed MXFP4_VALUES macro + _mm512_set_ps (which reverses).
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequant_mxfp4(packed_u8, e8m0_u8, group=32):
    """packed fp4 [.., K//2] uint8 + e8m0 [.., K//group] uint8 -> fp32 [.., K]."""
    low = (packed_u8 & 0xF).long()
    high = ((packed_u8 >> 4) & 0xF).long()
    *lead, khalf = packed_u8.shape
    nib = torch.empty(*lead, khalf * 2, dtype=torch.long)
    nib[..., 0::2] = low  # low nibble first (kernel: x0)
    nib[..., 1::2] = high  # high nibble second (kernel: x1 = x0 >> 4)
    vals = _E2M1[nib]  # [.., K]
    scale = torch.exp2(e8m0_u8.float() - 127.0)  # [.., K//group]
    scale = scale.repeat_interleave(group, dim=-1)  # [.., K]
    return vals * scale


def _rand_mxfp4(O, K, group=32):
    packed = torch.randint(0, 256, (O, K // 2), dtype=torch.uint8)
    # e8m0 near 127 (scale ~1) so magnitudes stay well-conditioned for the compare
    e8m0 = torch.randint(123, 131, (O, K // group), dtype=torch.uint8)
    return packed, e8m0


def _ref_moe(x, w13_u8, w13s, w2_u8, w2s, topk_w, topk_id):
    """Dense fp32 top-k SwiGLU from the SAME packed fp4 weights (independent oracle)."""
    import torch.nn.functional as F

    E = w13_u8.shape[0]
    T, hidden = x.shape
    out = torch.zeros(T, hidden, dtype=torch.float32)
    xf = x.float()
    w13 = torch.stack([_dequant_mxfp4(w13_u8[e], w13s[e]) for e in range(E)])  # [E,2I,hidden]
    w2 = torch.stack([_dequant_mxfp4(w2_u8[e], w2s[e]) for e in range(E)])  # [E,hidden,I]
    for e in range(E):
        mask = topk_id == e
        tok, slot = mask.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate_up = xf[tok] @ w13[e].t()
        g, u = gate_up.chunk(2, dim=-1)
        act = F.silu(g) * u
        oe = act @ w2[e].t()
        out.index_add_(0, tok, oe * topk_w[tok, slot].float().unsqueeze(-1))
    return out


def main() -> None:
    torch.manual_seed(0)
    from sglang.srt.layers.amx_utils import CPUQuantMethod

    try:
        import sgl_kernel  # noqa: F401  — registers torch.ops.sgl_kernel.* (needed without a model load)
    except Exception as e:  # noqa: BLE001
        print(f"MXFP4 MoE parity: SKIP (cannot import sgl_kernel: {e})")
        return
    if not hasattr(torch.ops.sgl_kernel, "fused_experts_cpu"):
        print("MXFP4 MoE parity: SKIP (sgl_kernel.fused_experts_cpu unavailable after import)")
        return

    E, hidden, inter, T, topk = 8, 128, 256, 6, 2
    N2 = 2 * inter
    x = torch.randn(T, hidden, dtype=torch.bfloat16)
    w13_u8, w13s = zip(*[_rand_mxfp4(N2, hidden) for _ in range(E)])
    w2_u8, w2s = zip(*[_rand_mxfp4(hidden, inter) for _ in range(E)])
    w13_u8 = torch.stack(w13_u8); w13s = torch.stack(w13s)
    w2_u8 = torch.stack(w2_u8); w2s = torch.stack(w2s)

    topk_id = torch.stack([torch.randperm(E)[:topk] for _ in range(T)]).to(torch.int32)
    topk_w = torch.rand(T, topk, dtype=torch.float32)

    ref = _ref_moe(x, w13_u8, w13s, w2_u8, w2s, topk_w, topk_id)

    # Kernel path — identical prepack to the plugin's MXFP4 MoE (convert_weight_packed on the
    # uint8 view => 4-bit packing; convert_scale_packed on the e8m0 bytes; fused_experts MXFP4).
    w13p = torch.ops.sgl_kernel.convert_weight_packed(w13_u8.contiguous())
    w2p = torch.ops.sgl_kernel.convert_weight_packed(w2_u8.contiguous())
    w13sp = torch.ops.sgl_kernel.convert_scale_packed(w13s.contiguous())
    w2sp = torch.ops.sgl_kernel.convert_scale_packed(w2s.contiguous())
    got = torch.ops.sgl_kernel.fused_experts_cpu(
        x, w13p, w2p, topk_w, topk_id, False, CPUQuantMethod.MXFP4,
        w13sp, w2sp, None, None, None, None, None, None, None, True, "silu",
    ).float()

    diff = (got - ref).abs()
    scale = ref.abs().max().clamp_min(1e-6)
    max_abs = diff.max().item()
    rel = (max_abs / scale).item()
    cos = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), ref.reshape(1, -1)
    ).item()
    print(f"MXFP4 MoE parity: max_abs_err={max_abs:.3e} rel={rel:.3e} cosine={cos:.6f}")
    # bf16 AMX accumulation vs fp32 reference -> tolerate small rel err; wiring bugs blow past this.
    assert cos >= 0.99 and rel <= 0.05, (
        f"MXFP4 MoE parity FAIL (cosine={cos:.6f}, rel={rel:.3e}) -> kernel/dequant/pack mismatch"
    )
    print("MXFP4 W4A16 CPU MoE kernel: numeric parity gate PASS")


if __name__ == "__main__":
    main()
