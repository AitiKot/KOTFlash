# Benchmarking

The benchmark compares backends on the same inputs. It reports forward latency, forward+backward latency, incremental peak memory, and maximum absolute output error against PyTorch SDPA.

Example for a T4, GQA, a 1024-token sequence and a 72-wide head dimension:

```bash
python benchmarks/benchmark_attention.py \
  --device cuda --batch 2 --q-heads 16 --kv-heads 4 \
  --seq 1024 --head-dim 72 --causal \
  --warmup 20 --iterations 100
```

For results worth publishing, include:

- Exact GPU model and compute capability.
- Driver and CUDA runtime versions.
- PyTorch and Triton versions.
- Batch, query/KV head counts, query/KV sequence lengths, head dimension, dtype and causal mode.
- Warm-up and timed iteration counts.
- Full JSON output, including errors and unavailable backends.

Use stable GPU clocks/power state when possible, close competing GPU processes, run each configuration more than once, and compare against the same PyTorch build. Do not compare measurements gathered on different GPU models as if they were a controlled speedup test.

The current backward is a correctness-oriented Python loop. The benchmark reports forward-only and forward+backward separately so the slow reference backward is not hidden by a forward speed number.
