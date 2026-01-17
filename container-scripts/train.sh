#!/bin/bash

echo "[train.sh] Checking landmarks..."

bash ./container-scripts/generate_all_landmarks.sh

exit

# ---------------------------------------------------------
# Torch Training Launcher (DF / LIAE 両対応)
# ---------------------------------------------------------

set -e

export PYTHONPATH=/workspace

# ----------------------------------------
# Config path (引数 or デフォルト)
# ----------------------------------------
CONFIG_PATH=${1:-/workspace/app/liae_config.json}
PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/main.py

echo "[train.sh] Using config: $CONFIG_PATH"

# ----------------------------------------
# 最新 checkpoint を自動検出
# ----------------------------------------
LATEST_CKPT=$(ls -1 /workspace/models/resume_step_*.pth 2>/dev/null | sort -V | tail -n 1)

if [ -n "$LATEST_CKPT" ]; then
	echo "[train.sh] Latest checkpoint detected: $LATEST_CKPT"
else
	echo "[train.sh] No checkpoint found. Starting from scratch."
fi

# ----------------------------------------
# config.json の resume_path を自動書き換え
# ----------------------------------------
tmpfile=$(mktemp)

if [ -n "$LATEST_CKPT" ]; then
	echo "[train.sh] Updating resume_path in config"
	jq --arg p "$LATEST_CKPT" '.resume_path = $p' "$CONFIG_PATH" >"$tmpfile"
else
	echo "[train.sh] Clearing resume_path (null)"
	jq '.resume_path = null' "$CONFIG_PATH" >"$tmpfile"
fi

mv "$tmpfile" "$CONFIG_PATH"

# ----------------------------------------
# Python 実行
# ----------------------------------------
echo "[train.sh] Starting training..."
$PYTHON $SCRIPT "$CONFIG_PATH"
