# KOTFlash design

## Layout

Public tensors use [B, H, S, D], as in PyTorch scaled_dot_product_attention.

## Forward

Each Triton program owns a block of query rows for one batch and query head. It streams over key/value blocks and maintains the online-softmax state (m, l, acc):

- m: running row maximum of QK scores
- l: running normalized exponential sum
- acc: running weighted value sum

Only the output and row-wise log-sum-exp are written. The complete quadratic attention matrix is never materialized.

## GQA/MQA

For Hq/Hkv = G, query head h reads KV head h // G. The backward loops over query heads and accumulates their dK/dV contributions into the corresponding KV head.

## Backward

v0.1 recomputes each score tile in PyTorch in FP32. This is a correctness-first memory-bounded implementation, not a fused performance kernel. A future CUDA/Triton backward should fuse the dQ and dK/dV paths where practical and avoid Python loops.

## Turing constraints

The initial path targets SM75 and FP16. Tile sizes must be benchmarked on a physical T4; do not transfer Ampere/Hopper tile assumptions blindly. head_dim is padded to a power of two in the Triton kernel, so non-power-of-two dimensions are supported functionally but may be inefficient.
