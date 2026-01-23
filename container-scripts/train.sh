#!/bin/bash
set -euo pipefail

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

export PYTHONPATH=/workspace

CONFIG_PATH=${1:-/workspace/app/liae_config.json}
PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/main.py

echo "[train.sh] Using config: $CONFIG_PATH"

# models ディレクトリが存在しない場合に備える
mkdir -p /workspace/models

# 最新 checkpoint の取得（安全版）
LATEST_CKPT=$(ls -1 /workspace/models/*step*.pth 2>/dev/null | sort -V | tail -n 1 || true)

if [ -n "${LATEST_CKPT}" ]; then
	echo "[train.sh] Latest checkpoint detected: $LATEST_CKPT"
else
	echo "[train.sh] No checkpoint found. Starting from scratch."
fi

# config のバックアップ
BACKUP_PATH="${CONFIG_PATH}.bak_$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG_PATH" "$BACKUP_PATH"
echo "[train.sh] Backup created: $BACKUP_PATH"

# jq の安全更新
tmpfile=$(mktemp)

if [ -n "${LATEST_CKPT}" ]; then
	echo "[train.sh] Updating resume_path in config"
	if jq --arg p "$LATEST_CKPT" '.resume_path = $p' "$CONFIG_PATH" >"$tmpfile"; then
		mv "$tmpfile" "$CONFIG_PATH"
	else
		echo "[ERROR] Failed to update config. Keeping original."
		rm "$tmpfile"
	fi
else
	echo "[train.sh] Clearing resume_path (null)"
	if jq '.resume_path = null' "$CONFIG_PATH" >"$tmpfile"; then
		mv "$tmpfile" "$CONFIG_PATH"
	else
		echo "[ERROR] Failed to clear resume_path. Keeping original."
		rm "$tmpfile"
	fi
fi

echo "[train.sh] Starting training..."
exec $PYTHON $SCRIPT "$CONFIG_PATH"
