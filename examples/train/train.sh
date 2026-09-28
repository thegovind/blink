#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="lab${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL:-thegovind/blink-4b}
DEVICE=${DEVICE:-cuda}
TARGETS=${TARGETS:-attn,delta,mlp}
PORT=${PORT:-8000}
OUT_DIR=${OUT_DIR:-"${TMPDIR:-/tmp}/blink-train-example-$$"}

if [ -e "$OUT_DIR" ]; then
    echo "Choose a fresh OUT_DIR: $OUT_DIR already exists" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

if [ -n "${TRAIN_DATA:-}" ] || [ -n "${DEV_DATA:-}" ]; then
    : "${TRAIN_DATA:?Set TRAIN_DATA and DEV_DATA together}"
    : "${DEV_DATA:?Set TRAIN_DATA and DEV_DATA together}"
else
    head -n 16 examples/train/sample.jsonl > "$OUT_DIR/train.jsonl"
    tail -n 4 examples/train/sample.jsonl > "$OUT_DIR/dev.jsonl"
    TRAIN_DATA="$OUT_DIR/train.jsonl"
    DEV_DATA="$OUT_DIR/dev.jsonl"
fi

python -m torch.distributed.run --standalone --nproc_per_node=1 --module jevlab.train \
    --model "$MODEL" --data "$TRAIN_DATA" --dev "$DEV_DATA" --out "$OUT_DIR/run" \
    --device "$DEVICE" --template semif --targets "$TARGETS" --rank 16 --alpha 32 \
    --lr 5e-5 --epochs 1 --budget 1024 --accum 1 --max-len 512 --warmup 1

python -m jevlab.merge --base "$MODEL" --adapter "$OUT_DIR/run/final" --out "$OUT_DIR/merged"

cp serve.py space/blink.py space/graft_keys.py "$OUT_DIR/merged/"
python "$OUT_DIR/merged/serve.py" --model "$OUT_DIR/merged" --host 127.0.0.1 --port "$PORT" \
    > "$OUT_DIR/serve.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        cat "$OUT_DIR/serve.log" >&2
        exit 1
    fi
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/healthz" > "$OUT_DIR/health.json"; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    cat "$OUT_DIR/serve.log" >&2
    echo "Server did not become ready" >&2
    exit 1
fi

curl --silent --show-error --fail-with-body \
    --header 'Content-Type: application/json' \
    --data '{"state":{"goal":"Export the visible records","screen_text":["Records","Download CSV","Help"]},"questions":{"next":{"type":"choice","instructions":"Which control should be selected?","criteria":{"export":"Download CSV","help":"Help"}},"confirm":{"type":"noul","instructions":"Is confirmation required?"},"risk":{"type":"score","instructions":"Rate the risk.","criteria":["Low","Medium","High"]}}}' \
    "http://127.0.0.1:$PORT/v1/systemone"
printf '\n'
