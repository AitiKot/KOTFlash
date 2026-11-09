# Benchmarking

Performance measurements are hardware- and software-stack-specific. Always record the GPU, compute capability, driver, CUDA runtime, PyTorch version, Triton version, dtype, tensor shapes and benchmark parameters.

## Attention forward and backward

The main benchmark compares:
- native PyTorch GQA SDPA;
- PyTorch SDPA with KV heads expanded;
- explicit KOTFlash Triton;
- KOTFlash auto routing.

Example:

~~~bash
python benchmarks/benchmark_attention.py \
  --device cuda --batch 2 --q-heads 8 --kv-heads 2 \
  --seq 1024 --head-dim 64 --causal \
  --warmup 20 --iterations 50 --repeats 5
~~~

It reports forward latency, forward+backward latency, incremental peak allocation and maximum absolute output error. Expanded-KV timing includes `repeat_interleave`, which is an important T4 baseline when native GQA dispatches to a slower PyTorch kernel.

For a single-token decode-shaped attention call:

~~~bash
python benchmarks/benchmark_attention.py \
  --device cuda --batch 2 --q-heads 8 --kv-heads 2 \
  --seq 1 --kv-seq 2048 --head-dim 64 \
  --warmup 20 --iterations 50 --repeats 5
~~~

## KV-cache append plus decode

The dedicated decode benchmark separates attention-only timing from the cost of concatenating the new K/V token into the caller-managed cache:

~~~bash
python benchmarks/benchmark_decode.py \
  --batch 2 --q-heads 8 --kv-heads 2 --head-dim 64 \
  --kv-seq 2048 --warmup 20 --iterations 50 --repeats 5 \
  --output decode-results.json
~~~

The benchmark reports both CUDA-event time and synchronized wall-clock time. Batch tokens/second uses wall-clock latency for one decode position per batch. The example's torch.cat update copies the cache and is intentionally a simple reference; production models usually write into preallocated KV storage. Model projections, MLP and sampling are excluded, so this is not full-model tokens/second.

## Decoder-block training step

To understand whether an attention gain survives the surrounding projections, MLP, backward and optimizer update:

~~~bash
python benchmarks/benchmark_decoder_block.py \
  --batch 2 --seq 512 --q-heads 8 --kv-heads 2 --head-dim 64 \
  --warmup 3 --iterations 10
~~~

This is a synthetic single-block comparison, not a complete language-model training benchmark.

## Tile tuning

For a T4-specific forward tile sweep:

~~~bash
python benchmarks/tune_triton_tiles.py --seq 1024 --head-dim 72 --causal
~~~

Keep the full JSON output, including failed configurations and compiler/resource errors. A configuration that fails due to resource limits is useful information and must not be silently omitted.

## Measurement rules

1. Use the same tensors, dtype, masks and scaling for each backend.
2. Warm all implementations before timing.
3. Synchronize CUDA around measurements or use CUDA events.
4. Interleave or shuffle backend order to reduce systematic ordering bias.
5. Run several repeats and report medians and ranges.
6. Compare numerical outputs against a PyTorch reference.
7. Report failed or unavailable backends instead of hiding them.
8. Never extrapolate attention-only speedups to full-model throughput.
9. Do not compare results from different GPUs or software stacks as if they were controlled experiments.

For the reproducible T4 profile, use Tesla T4 / SM75, PyTorch 2.6.0+cu124 and Triton 3.2.x. Re-measure automatic routing when the stack or workload changes.
