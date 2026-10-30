#include "sparse_attn.h"
#include <torch/extension.h>

#include <cmath>
#include <limits>

torch::Tensor sparse_attend_fp32bmm_hx(
    torch::Tensor q, torch::Tensor kv, torch::Tensor sink, double scale);

torch::Tensor sparse_attn_bf16_matched(
    torch::Tensor q, torch::Tensor kv, torch::Tensor sink,
    torch::Tensor indices, double scale) {
    TORCH_CHECK(q.device().is_cpu() && kv.device().is_cpu() &&
                sink.device().is_cpu() && indices.device().is_cpu(), "CPU tensors only");
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && sink.dim() == 1 &&
                indices.dim() == 3, "Expected Q[N,H,D], KV[N,K,D], sink[H], indices[N,1,K]");
    TORCH_CHECK(q.size(0) == kv.size(0) && q.size(0) == indices.size(0) &&
                indices.size(1) == 1 && indices.size(2) == kv.size(1) &&
                q.size(1) == sink.size(0) && q.size(2) == 512 && kv.size(2) == 512,
                "Incompatible shapes");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(1) % 32 == 0 && kv.size(1) > 0 &&
                q.size(0) <= std::numeric_limits<int>::max() &&
                q.size(1) <= std::numeric_limits<int>::max() &&
                kv.size(1) <= std::numeric_limits<int>::max(), "Unsupported dimensions");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32 && indices.is_contiguous(),
                "Contiguous INT32 indices required");
    TORCH_CHECK(std::isfinite(scale), "Scale must be finite");

    // Mirror the original FP32 function's input-normalization and output-ownership policy.
    auto qc = q.to(torch::kBFloat16).contiguous();
    auto kc = kv.to(torch::kBFloat16).contiguous();
    auto sc = sink.to(torch::kFloat32).contiguous();
    const float factor = scale > 0 ? static_cast<float>(scale) : 1.f / std::sqrt(512.f);
    auto out = torch::empty(qc.sizes(), qc.options());
    const int status = sparse_attn_emr(
        reinterpret_cast<const uint16_t*>(qc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(kc.data_ptr<at::BFloat16>()),
        sc.data_ptr<float>(), indices.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
        qc.size(0), 1, qc.size(1), kc.size(1), indices.size(2), factor, nullptr);
    TORCH_CHECK(status == 0, "sparse_attn_emr failed: status=", status);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize", []() {
        const int status = sparse_attn_emr_init();
        TORCH_CHECK(status == 0, "AMX initialization failed: status=", status);
    });
    m.def("fp32", &sparse_attend_fp32bmm_hx);
    m.def("bf16", &sparse_attn_bf16_matched);
}
