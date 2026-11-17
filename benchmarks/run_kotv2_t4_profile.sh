#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reproducible T4 profile for the attention geometry used by KOT-V2.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

Q_HEADS="${KOTV2_Q_HEADS:-16}"
KV_HEADS="${KOTV2_KV_HEADS:-4}"
HEAD_DIM="${KOTV2_HEAD_DIM:-128}"
BATCH="${KOTV2_BATCH:-1}"
WARMUP="${KOTV2_WARMUP:-10}"
ITERATIONS="${KOTV2_ITERATIONS:-30}"
REPEATS="${KOTV2_REPEATS:-5}"
SEQ_LENGTHS="${KOTV2_SEQ_LENGTHS:-512 1024 2048 4096}"
BLOCK_LENGTHS="${KOTV2_BLOCK_LENGTHS:-512 1024}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${KOTV2_OUT_DIR:-$ROOT_DIR/artifacts/kotv2-t4-$STAMP}"
mkdir -p "$OUT_DIR"

python - <<'PY'
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Select a Tesla T4 runtime.")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if "T4" not in name or capability != (7, 5):
    raise SystemExit(f"Expected Tesla T4 / SM75; got {name} / {capability}.")
if torch.__version__.split("+", 1)[0] != "2.6.0" or torch.version.cuda != "12.4":
    raise SystemExit(
        f"Expected PyTorch 2.6.0+cu124; got {torch.__version__} / CUDA {torch.version.cuda}."
    )
version = tuple(int(x) for x in triton.__version__.split("+", 1)[0].split(".")[:2])
if version != (3, 2):
    raise SystemExit(f"Expected Turing-compatible Triton 3.2.x; got {triton.__version__}.")
print(f"GPU: {name} / SM{capability[0]}{capability[1]}")
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; Triton: {triton.__version__}")
PY

cat > "$OUT_DIR/profile.txt" <<EOF
q_heads=$Q_HEADS
kv_heads=$KV_HEADS
head_dim=$HEAD_DIM
batch=$BATCH
warmup=$WARMUP
iterations=$ITERATIONS
repeats=$REPEATS
sequence_lengths=$SEQ_LENGTHS
block_lengths=$BLOCK_LENGTHS
EOF

for seq in $SEQ_LENGTHS; do
  echo "=== Attention A/B: B=$BATCH Hq=$Q_HEADS Hkv=$KV_HEADS D=$HEAD_DIM S=$seq ==="
  python benchmarks/benchmark_attention.py \
    --device cuda --batch "$BATCH" --q-heads "$Q_HEADS" \
    --kv-heads "$KV_HEADS" --seq "$seq" --head-dim "$HEAD_DIM" \
    --causal --warmup "$WARMUP" --iterations "$ITERATIONS" --repeats "$REPEATS" \
    > "$OUT_DIR/attention-seq-$seq.json"
done

for seq in $BLOCK_LENGTHS; do
  echo "=== Decoder block A/B: B=$BATCH Hq=$Q_HEADS Hkv=$KV_HEADS D=$HEAD_DIM S=$seq ==="
  if python benchmarks/benchmark_decoder_block.py \
      --batch "$BATCH" --seq "$seq" --q-heads "$Q_HEADS" \
      --kv-heads "$KV_HEADS" --head-dim "$HEAD_DIM" \
      --warmup 2 --iterations 5 \
      > "$OUT_DIR/decoder-block-seq-$seq.json" \
      2> "$OUT_DIR/decoder-block-seq-$seq.stderr"; then
    echo "Saved decoder-block-seq-$seq.json"
  else
    echo "Decoder-block profile failed at seq=$seq; see stderr report."
    printf '{"status":"failed","sequence":%s}\n' "$seq" \
      > "$OUT_DIR/decoder-block-seq-$seq.error.json"
  fi
done

echo "KOT-V2 T4 profile saved to: $OUT_DIR"
