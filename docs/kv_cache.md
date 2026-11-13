# Caller-managed KV cache

KOTFlash exposes utilities for incremental one-token decoding. It does not
manage model projections, paged allocation, eviction, batching across requests
or cache quantization.

## Fixed-capacity cache

Use `PreallocatedKVCache` when the maximum context length is known:

```python
import torch
from kotflash.cache import PreallocatedKVCache

cache = PreallocatedKVCache(
    batch=2, kv_heads=2, max_len=8192, head_dim=64,
    device="cuda", dtype=torch.float16,
)
cache.append(k_prefill, v_prefill)

# After computing one new token's K/V, append and attend in one API call:
output = cache.append_and_attend(
    q_new, k_new, v_new,
    append_backend="auto",       # portable torch path by default
    attention_backend="auto",    # direct strided Triton path for measured T4 decode
)
```

`append` validates shape, device, dtype and capacity before writing. The
`length` property tracks the populated prefix. `view()` returns views backed
by the allocated storage and excludes unused slots. These views can be
non-contiguous because the allocated cache has a larger capacity than its
populated prefix. `contiguous_view()` explicitly copies the populated prefix.

For the measured SM75 decode layout (B=2, Hq=8, Hkv=2, Q=1, KV=2048, D=64,
FP16 on PyTorch 2.6.0+cu124 / Triton 3.2.x), `backend="auto"` can use an
inference-only Split-KV kernel that accepts the cache's padded strides. It
does not copy the populated K/V prefix into contiguous buffers. The generic
Triton training/autograd path still makes contiguous copies; other cache
shapes and software stacks use the conservative fallback until profiled.

## Convenience functions

- `append_kv_cache(k, v, new_k, new_v)` returns concatenated tensors without
  mutating the original cache. It copies the existing cache each time.
- `decode_step(q_new, k, v, k_new, v_new)` appends exactly one K/V position,
  then runs attention for exactly one new query position.

For one-token decoding, the updated cache contains only keys visible to the
new query, so `decode_step` uses `causal=False`. Do not pass future keys to
this helper. Multi-token chunk decoding requires a different causal alignment
and is not implemented by this utility.

## Scope and thread safety

The cache is caller-owned, append-only, and not thread-safe. It does not handle
resetting, paging, eviction, sliding windows, prefix sharing or concurrent
writes. Allocate one instance per independent request/sequence unless the
caller provides external synchronization.


## Optional one-launch CUDA C++ append

For FP16 CUDA caches, K and V can be written to their preallocated slots by a
single custom CUDA kernel instead of two PyTorch `copy_` calls:

```python
cache.append(k_new, v_new, backend="cuda")
```

The first call JIT-compiles the bundled first-party source with
`torch.utils.cpp_extension`, so the machine needs a compatible CUDA toolkit
and `nvcc`. To opt in automatically for all eligible cache appends, set
`KOTFLASH_ENABLE_CUDA_APPEND=1` before starting Python. The portable default
is `backend="torch"` (or `"auto"` with the environment variable unset).
This operator only appends K/V; the attention computation is still performed
by PyTorch SDPA or the separate Triton decode kernel. The CUDA extension is
experimental and must be benchmarked on the target GPU.


## Optional one-launch CUDA C++ append

For FP16 CUDA caches, K and V can be written to their preallocated slots by a
single custom CUDA kernel instead of two PyTorch `copy_` calls:

```python
cache.append(k_new, v_new, backend="cuda")
```

The first call JIT-compiles the bundled first-party source with
`torch.utils.cpp_extension`, so the machine needs a compatible CUDA toolkit
and `nvcc`. To opt in automatically for all eligible cache appends, set
`KOTFLASH_ENABLE_CUDA_APPEND=1` before starting Python. The portable default
is `backend="torch"` (or `"auto"` with the environment variable unset).
This operator only appends K/V; the attention computation is still performed
by PyTorch SDPA or the separate Triton decode kernel. The CUDA extension is
experimental and must be benchmarked on the target GPU.


## Fused T4 append + decode

For the exact measured T4 profile (B=2, Hq=8, Hkv=2, Q=1, past KV=2047,
D=64, FP16, PyTorch 2.6.0+cu124 / Triton 3.2.x), the default
`append_backend="auto"` in `append_and_attend()` uses an inference-only
fused Split-KV kernel. The launch writes the new K/V slot and computes its
partial attention result in the same kernel. A second kernel reduces the
partial outputs. This avoids a separate cache-append launch and avoids
copying the existing K/V prefix.

To force this experimental path on another eligible SM75 short-query / long
cache shape, set `append_backend="fused"` and `attention_backend="triton"`.
It is inference-only, FP16, head_dim 16–64 and requires Triton 3.2.x. It has no
backward implementation. The default `auto` route is restricted to the exact
T4 profile above; other shapes use the portable append + attention path.
