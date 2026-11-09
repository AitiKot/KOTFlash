# KOTFlash architecture

## Public tensor contract

The public API uses tensors in [B, H, S, D] layout:

- Q: [batch, query_heads, query_length, head_dim]
- K/V: [batch, kv_heads, key_length, head_dim]
- query_heads must be divisible by kv_heads.

Equal head counts implement MHA. Fewer KV heads implement GQA or MQA.

The public entry point is `kotflash.attention.attention`. It validates inputs and chooses one of three backends:

- `sdpa`: PyTorch scaled-dot-product attention.
- `triton`: explicitly requests the experimental SM75 implementation.
- `auto`: uses only narrowly measured T4 profiles and otherwise falls back to PyTorch SDPA.

## Triton forward

The standard Triton forward assigns a program to a query tile, batch item and query head. It streams over key/value tiles and maintains the online-softmax state:

- m: running row maximum of QK scores;
- l: running sum of exponentials relative to m;
- acc: running weighted-value accumulator.

The full [Sq, Sk] attention matrix is not materialized. Accumulators and log-sum-exp values use FP32; the experimental custom path accepts FP16 Q/K/V.

For causal attention, key positions greater than query positions are masked. Complete future K/V tiles are skipped. The mask follows PyTorch's upper-left alignment for unequal query and key lengths.

For head dimensions between 65 and 127, the forward implementation computes a 64-wide dot plus a padded tail dot. This avoids padding the entire operation to a 128-wide dot for dimensions such as 72 and 96. This path is implemented but still requires broader T4 profiling.

## Split-KV decode

For non-causal attention with short queries (Sq <= 16), long K/V context (Sk >= 1024) and D <= 64, the explicit Triton backend can split the K/V sequence across independent programs.

Each split writes:
- a normalized partial output;
- a row-wise partial log-sum-exp.

A reduction kernel combines partial outputs using the partial log-sum-exp values as weights. This is mathematically equivalent to a single softmax over the concatenated visible K/V context.

The caller owns the cache. KOTFlash does not append, allocate, page or evict KV entries. The caller must pass only keys visible to every query in the supplied Q chunk. For one-token decoding, when the cache includes the current token and contains no future keys, use `causal=False`.

The split-KV path is experimental. Its speed depends on query length, cache length, head dimension, GPU and software stack. A measured win for one exact shape must not be generalized to nearby shapes.

## Triton backward

The custom autograd function saves Q/K/V, the forward output and row log-sum-exp. The fused Triton backward computes dQ and dK/dV.

For shorter query sequences, dK/dV work is assigned by KV tile and accumulates contributions from the query tiles. For Sq >= 1024, the implementation splits query tiles into independent chunks, writes FP32 partial dK/dV buffers and reduces those buffers in a second kernel. This increases parallel work without floating-point atomics.

A batched PyTorch blockwise backward helper remains in the source as a reference/testing implementation; it is not the production Triton backward path.

## Runtime constraints

The experimental kernels currently target:
- CUDA NVIDIA Turing SM75;
- FP16 Q/K/V;
- head dimensions 16 through 128;
- upstream Triton 3.2.x, or a compatible Turing-enabled build.

The automatic performance profile is narrower than the explicit Triton backend. It is tied to the exact T4/PyTorch/CUDA/Triton stack and shapes measured so far. Unprofiled shapes use the conservative SDPA fallback.

## Deliberate non-goals

KOTFlash is an attention library, not a complete inference engine. It currently does not provide paged KV cache management, packed variable-length sequences, dropout, local-window attention, token sampling, model projections or a model-serving runtime.
