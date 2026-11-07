# KOTFlash design

## Layout

Public tensors use [B, H, S, D], as in PyTorch scaled_dot_product_attention. The query-head count must be divisible by the KV-head count.

## Forward

Each Triton program owns a block of query rows for one batch and query head. It streams over key/value blocks and maintains the online-softmax state (m, l, acc):

- m: running row maximum of QK scores
- l: running normalized exponential sum
- acc: running weighted value sum

Only the output and row-wise log-sum-exp are written. The complete quadratic attention matrix is never materialized. The implementation accumulates in FP32 and currently accepts FP16 inputs only.

## GQA/MQA

For Hq/Hkv = G, query head h reads KV head h // G. The backward loops over query heads and accumulates their dK/dV contributions into the corresponding KV head.

## Backward

The current backward recomputes each score tile in PyTorch using FP32 intermediates. It is a correctness-first, memory-bounded reference implementation, not a fused performance kernel. Python loops make it slow for long sequences. A future SM75 backward should use dedicated GPU kernels for dQ and dK/dV and avoid races while accumulating shared KV gradients.

## Turing constraints

The initial custom path targets SM75 and FP16. Upstream Triton 3.3 removed Turing support; use upstream Triton 3.2.x or a separately maintained Turing-enabled build. Tile sizes must be benchmarked on a physical T4; do not transfer Ampere/Hopper assumptions blindly.

Head dimensions are padded to the next power of two. For example, D=72 is padded to 128, so it is accepted but not yet efficient. Fixed block sizes are starting values, not measured optimal settings.
