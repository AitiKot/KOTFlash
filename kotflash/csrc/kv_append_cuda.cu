// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__global__ void append_kv_kernel(
    const half* __restrict__ src_k,
    const half* __restrict__ src_v,
    half* __restrict__ dst_k,
    half* __restrict__ dst_v,
    int64_t total,
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t dim,
    int64_t start,
    int64_t ksb, int64_t ksh, int64_t kss, int64_t ksd,
    int64_t vsb, int64_t vsh, int64_t vss, int64_t vsd,
    int64_t dsb, int64_t dsh, int64_t dss, int64_t dsd) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;

    const int64_t d = linear % dim;
    const int64_t token = (linear / dim) % sequence;
    const int64_t head = (linear / (dim * sequence)) % heads;
    const int64_t b = linear / (dim * sequence * heads);

    const int64_t src_ko = b * ksb + head * ksh + token * kss + d * ksd;
    const int64_t src_vo = b * vsb + head * vsh + token * vss + d * vsd;
    const int64_t dst_o = b * dsb + head * dsh + (start + token) * dss + d * dsd;

    dst_k[dst_o] = src_k[src_ko];
    dst_v[dst_o] = src_v[src_vo];
}

}  // namespace

void append_kv_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor new_k,
    torch::Tensor new_v,
    int64_t start) {
    TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() &&
                new_k.is_cuda() && new_v.is_cuda(),
                "all K/V cache tensors must be CUDA tensors");
    TORCH_CHECK(k_cache.device() == v_cache.device() &&
                k_cache.device() == new_k.device() &&
                k_cache.device() == new_v.device(),
                "all K/V cache tensors must be on the same CUDA device");
    TORCH_CHECK(k_cache.scalar_type() == at::kHalf &&
                v_cache.scalar_type() == at::kHalf &&
                new_k.scalar_type() == at::kHalf &&
                new_v.scalar_type() == at::kHalf,
                "CUDA append currently supports FP16 tensors only");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
                "K/V cache must have matching [B,H,capacity,D] shapes");
    TORCH_CHECK(new_k.dim() == 4 && new_v.sizes() == new_k.sizes(),
                "new K/V must have matching [B,H,S,D] shapes");
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),
                "destination K/V caches must be contiguous");
    TORCH_CHECK(new_k.size(0) == k_cache.size(0) &&
                new_k.size(1) == k_cache.size(1) &&
                new_k.size(3) == k_cache.size(3),
                "new K/V batch, heads and head_dim must match the cache");
    TORCH_CHECK(start >= 0 && start + new_k.size(2) <= k_cache.size(2),
                "append range exceeds cache capacity");
    TORCH_CHECK(new_k.size(2) > 0 && new_k.size(3) > 0,
                "new K/V dimensions must be non-empty");

    c10::cuda::CUDAGuard device_guard(k_cache.device());
    const int64_t batch = new_k.size(0);
    const int64_t heads = new_k.size(1);
    const int64_t sequence = new_k.size(2);
    const int64_t dim = new_k.size(3);
    const int64_t total = batch * heads * sequence * dim;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    append_kv_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(new_k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(new_v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
        total, batch, heads, sequence, dim, start,
        new_k.stride(0), new_k.stride(1), new_k.stride(2), new_k.stride(3),
        new_v.stride(0), new_v.stride(1), new_v.stride(2), new_v.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
