#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$ROOT_DIR/artifacts/t4-validation-$STAMP"
mkdir -p "$OUT_DIR"

echo "KOTFlash T4 validation"
echo "Repository: $ROOT_DIR"
echo "Results:    $OUT_DIR"

python - <<'PY'
import torch
import triton

gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "CUDA unavailable"
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available.")
capability = torch.cuda.get_device_capability()
print(f"GPU: {gpu}")
print(f"Compute capability: {capability}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"Triton: {triton.__version__}")

if capability != (7, 5):
    raise SystemExit("ERROR: this validation script requires NVIDIA Turing SM75 (Tesla T4).")
if "T4" not in gpu:
    raise SystemExit("ERROR: expected a Tesla T4; refusing to label another GPU run as T4 validation.")
if torch.__version__.split("+", 1)[0] != "2.6.0" or torch.version.cuda != "12.4":
    raise SystemExit("ERROR: reproducible auto-profile validation requires PyTorch 2.6.0 + CUDA 12.4.")
parts = tuple(int(piece) for piece in triton.__version__.split("+", 1)[0].split(".")[:2])
if parts != (3, 2):
    raise SystemExit("ERROR: SM75 validation requires a Turing-compatible Triton 3.2.x build.")
PY

python -m compileall -q kotflash benchmarks tests examples
python -m pytest -q | tee "$OUT_DIR/pytest.txt"

for seq in 128 256 512 1024 2048; do
  echo "Attention training profile: seq=$seq"
  python benchmarks/benchmark_attention.py \
    --device cuda --batch 2 --q-heads 8 --kv-heads 2 \
    --seq "$seq" --head-dim 64 --causal \
    --warmup 20 --iterations 50 --repeats 5 \
    > "$OUT_DIR/attention-seq-$seq.json"
done

echo "Attention decode profile: Q=1, KV=2048"
python benchmarks/benchmark_attention.py \
  --device cuda --batch 2 --q-heads 8 --kv-heads 2 \
  --seq 1 --kv-seq 2048 --head-dim 64 \
  --warmup 20 --iterations 50 --repeats 5 \
  > "$OUT_DIR/attention-decode-q1-kv2048.json"

echo "Append + attention decode benchmark"
APPEND_BACKEND="${KOTFLASH_T4_APPEND_BACKEND:-auto}"
python benchmarks/benchmark_decode.py \
  --batch 2 --q-heads 8 --kv-heads 2 --head-dim 64 \
  --kv-seq 2048 --warmup 20 --iterations 50 --repeats 5 \
  --append-backend "$APPEND_BACKEND" \
  --output "$OUT_DIR/decode-append-attention.json"

echo "Synthetic decoder-block training benchmark"
python benchmarks/benchmark_decoder_block.py \
  --batch 2 --seq 512 --q-heads 8 --kv-heads 2 --head-dim 64 \
  --warmup 3 --iterations 10 \
  > "$OUT_DIR/decoder-block-seq512.json"

echo "Validation complete. Results saved to: $OUT_DIR"
