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

# After computing one new token's K/V:
cache.append(k_new, v_new)
output = cache.attend(q_new, backend="auto")
```

`append` validates shape, device, dtype and capacity before writing. The
`length` property tracks the populated prefix. `view()` returns views backed
by the allocated storage and excludes unused slots. These views can be
non-contiguous because the allocated cache has a larger capacity than its
populated prefix. `contiguous_view()` explicitly copies the populated prefix.

The current Triton attention wrapper requires contiguous input and will make
a contiguous copy when needed. The preallocated class avoids reallocating the
entire cache with `torch.cat`, but does **not** yet provide zero-copy Triton
attention. Removing that copy requires kernels that accept the cache's
strides; that optimization remains experimental and must be measured on T4.

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
