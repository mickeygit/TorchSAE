#!/bin/bash

# ---------------------------------------------------------
# TorchSAE Training Launcher
# ---------------------------------------------------------
# python3.10 と python3.9 が共存しているため、
# torch が入っている python3.9 を絶対パスで指定する。
# ---------------------------------------------------------
apt update
apt install -y jq

export PYTHONPATH=/workspace

CONFIG_PATH=${1:-/workspace/app/df_config.json}
PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/main.py

# ---------------------------------------------------------
# 最新 checkpoint を自動検出
# ---------------------------------------------------------
LATEST_CKPT=$(ls -1 /workspace/models/resume_step_*.pth 2>/dev/null | sort -V | tail -n 1)

if [ -n "$LATEST_CKPT" ]; then
	echo "[train.sh] Latest checkpoint detected: $LATEST_CKPT"
else
	echo "[train.sh] No checkpoint found. Starting from scratch."
fi

# ---------------------------------------------------------
# config.json の resume_path を自動書き換え
# ---------------------------------------------------------
if [ -n "$LATEST_CKPT" ]; then
	echo "[train.sh] Updating resume_path in config: $CONFIG_PATH"

	# jq を使って JSON の resume_path を書き換える
	# ※ jq が無い場合は apt install jq が必要
	tmpfile=$(mktemp)
	jq --arg p "$LATEST_CKPT" '.resume_path = $p' "$CONFIG_PATH" >"$tmpfile"
	mv "$tmpfile" "$CONFIG_PATH"
else
	echo "[train.sh] Clearing resume_path (null)"
	tmpfile=$(mktemp)
	jq '.resume_path = null' "$CONFIG_PATH" >"$tmpfile"
	mv "$tmpfile" "$CONFIG_PATH"
fi

# ---------------------------------------------------------
# Python 実行
# ---------------------------------------------------------
echo "[train.sh] Starting training with config: $CONFIG_PATH"
$PYTHON $SCRIPT "$CONFIG_PATH"
