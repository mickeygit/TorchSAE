#!/bin/bash

echo "[train.sh] Checking landmarks..."

A_DIR="/workspace/data/A"
B_DIR="/workspace/data/B"

need_landmarks=false

# A の landmarks チェック
if ls "$A_DIR"/*_landmarks.npy >/dev/null 2>&1; then
	echo "[train.sh] A: landmarks found."
else
	echo "[train.sh] A: landmarks missing."
	need_landmarks=true
fi

# B の landmarks チェック
if ls "$B_DIR"/*_landmarks.npy >/dev/null 2>&1; then
	echo "[train.sh] B: landmarks found."
else
	echo "[train.sh] B: landmarks missing."
	need_landmarks=true
fi

# 必要なら landmarks 生成
if [ "$need_landmarks" = true ]; then
	echo "[train.sh] Generating missing landmarks..."
	bash ./container-scripts/generate_all_landmarks.sh
else
	echo "[train.sh] All landmarks already exist. Skipping generation."
fi

# ---------------------------------------------------------
# Torch Training Launcher (DF / LIAE 両対応)
# ---------------------------------------------------------

set -e
export PYTHONPATH=/workspace

CONFIG_PATH=${1:-/workspace/app/liae_config.json}
PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/main.py

echo "[train.sh] Using config: $CONFIG_PATH"

LATEST_CKPT=$(ls -1 /workspace/models/step_*.pth 2>/dev/null | sort -V | tail -n 1)

if [ -n "$LATEST_CKPT" ]; then
	echo "[train.sh] Latest checkpoint detected: $LATEST_CKPT"
else
	echo "[train.sh] No checkpoint found. Starting from scratch."
fi

tmpfile=$(mktemp)

if [ -n "$LATEST_CKPT" ]; then
	echo "[train.sh] Updating resume_path in config"
	jq --arg p "$LATEST_CKPT" '.resume_path = $p' "$CONFIG_PATH" >"$tmpfile"
else
	echo "[train.sh] Clearing resume_path (null)"
	jq '.resume_path = null' "$CONFIG_PATH" >"$tmpfile"
fi

mv "$tmpfile" "$CONFIG_PATH"

echo "[train.sh] Starting training..."
$PYTHON $SCRIPT "$CONFIG_PATH"
