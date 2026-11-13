// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void append_kv_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor new_k,
    torch::Tensor new_v,
    int64_t start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("append_kv_", &append_kv_cuda,
               "Append K and V into fixed-capacity FP16 CUDA caches");
}
