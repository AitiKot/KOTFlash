# KOT-V2 training optimisation plan for free Tesla T4 runs

This plan separates improvements that are already part of KOTFlash from changes
that must be tested in the KOT-V2 trainer. KOTFlash accelerates attention only;
it does not make the MLP, MoE routing, optimizer or data pipeline faster.

## 1. Measure the actual KOT-V2 attention geometry first

The current `auto` router is intentionally conservative. Its measured causal
T4 profile is limited to B=2, Hq=8, Hkv=2, D=64 and sequence lengths
128/256/512/1024/2048 on Tesla T4 + PyTorch 2.6.0 + CUDA 12.4 + Triton 3.2.x.
A different head count, head dimension, batch size or runtime falls back to
PyTorch SDPA. Do not assume that a benchmark at D=64 proves a win for D=128.

Run the shape sweep using the actual values from the KOT-V2 config:

```bash
KOTV2_Q_HEADS=16 KOTV2_KV_HEADS=4 KOTV2_HEAD_DIM=128 \
KOTV2_BATCH=1 bash benchmarks/run_kotv2_t4_profile.sh
```

The defaults (16 Q heads, 4 KV heads, D=128, batch 1) are a starting profile
based on the current project notes. Override them if the checkpoint config
differs. The script records attention forward/backward results at 512, 1024,
2048 and 4096 tokens, then attempts full synthetic decoder-block steps at
512 and 1024. The block test includes projections, MLP, backward and AdamW,
but remains a one-block proxy, not a full KOT-V2 run.

Use the results this way:

- Compare explicit `triton` against both native GQA SDPA and expanded-KV SDPA.
- Check numerical error and finite outputs, not latency alone.
- Keep `auto` on fallback until a shape has repeated, interleaved measurements
  on the exact T4 software stack.
- Only add a dispatch entry after the winner is stable over multiple runs.
  A single best-looking sample is not enough.

## 2. Training schedule: increase context progressively

For a from-scratch or continued-pretraining run, use a staged context schedule:

| Stage | Sequence length | Purpose |
|---|---:|---|
| 1 | 512 | Stabilise training and validate throughput |
| 2 | 1024 | Increase context without a sudden memory jump |
| 3 | 2048 | Train meaningful medium-range dependencies |
| 4 | 4096 | Adapt the model to the target context |

Resume the same checkpoint and optimizer/scheduler state at every transition.
Do not reset optimizer moments just because sequence length changes. Track
progress by **tokens seen**, not by step count. The 4096-token stage must contain
a meaningful share of the total training tokens; leaving almost all long-context
training to the final few steps risks undertraining that context length.

At each stage choose microbatch size from measured peak memory. If it must be
reduced, use gradient accumulation to preserve the intended global tokens per
optimizer update. On DDP, use `no_sync()` on non-final accumulation microsteps
so gradients are all-reduced once per optimizer update, not on every microstep.

## 3. T4 memory and distributed training

- Tesla T4 is an FP16-oriented target. Use FP16 autocast with GradScaler and
  finite-loss/gradient checks; do not assume BF16 has native T4 acceleration.
- Keep the existing Lion8bit path as the primary low-memory optimizer and
  benchmark alternatives only on equal tokens and comparable validation loss.
- DDP over two T4s replicates the model on each GPU. It improves data-parallel
  throughput, but it does **not** combine two 16-GB memories into one 32-GB
  allocation. Each rank must fit its model, gradients, optimizer state and
  activations independently.
- Measure memory after optimizer state has been initialised. Report allocated
  and reserved peaks, not only model weight size.
- Save model, optimizer, scaler, scheduler, global step, sampler state and RNG
  state in resumable checkpoints. A weights-only checkpoint is for inference,
  not an exact training resume.

## 4. MoE Top-4 is likely a larger optimisation target than attention

For the 8-expert Top-4 design, every token creates four expert assignments.
The trainer should dispatch only selected tokens to each expert rather than
running every expert on the full [batch, sequence] tensor and masking outputs
afterward. Track per-expert token counts, router entropy and auxiliary load
balancing loss. Start without dropping overflow tokens; first establish stable
routing and measure expert imbalance.

The interactive Mini lab now uses sparse token dispatch as an educational
implementation. It is not a production expert-parallel kernel: its Python
expert loop and dynamic `nonzero` operations are not a substitute for grouped
GEMM or a fused MoE dispatcher.

## 5. Activation and compute experiments, in priority order

1. **Sequence packing and pre-tokenisation.** Tokenise once, pack contiguous
   examples with EOS separators, and feed prebuilt token blocks. Keep data
   loading/tokenisation outside the timed GPU step.
2. **Gradient accumulation + DDP no_sync.** Preserve global tokens/update while
   fitting a smaller microbatch. Validate the exact reduction semantics.
3. **KOTFlash attention.** Replace only the attention call after RoPE, with
   Q/K/V in [B,H,S,D] and FP16. Benchmark it against both SDPA baselines at the
   real KOT-V2 head dimensions and sequence lengths.
4. **Activation checkpointing.** Apply to selected transformer blocks or large
   MLP/MoE subgraphs only when memory is limiting; measure the recompute cost.
5. **torch.compile.** Test after eager correctness. Dynamic expert routing can
   cause graph breaks or recompilation, so compare compile time, steady-state
   tokens/s and peak memory over a sufficiently long run.
6. **Kernel-level MoE optimisation.** Once router statistics are stable, test
   grouped expert GEMMs and fused dispatch/combine. Do not change Top-4 semantics
   in the name of speed without measuring validation quality.

## 6. A/B rules for every optimisation

Use identical initial weights, optimizer state, data batches, random seeds and
token counts. Interleave backend order where possible. Report:

- tokens/s including forward, backward and optimizer update;
- validation loss at equal tokens seen;
- peak allocated and reserved GPU memory;
- skipped/failed steps, non-finite gradients and expert-load statistics;
- exact GPU, driver, CUDA, PyTorch, Triton and optimizer versions.

KOTFlash's attention-only speedup is not the model's speedup. The decision to
enable it for pretraining must be based on an end-to-end KOT-V2 run.
