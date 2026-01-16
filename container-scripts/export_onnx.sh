#!/bin/bash
set -e

# ============================================================
# TorchSAE ONNX Export Tool
# ============================================================
#  • DF / LIAE の両方に対応（両方あれば LIAE を優先）
#  • 最新 checkpoint を自動検出
#  • config / checkpoint / 出力パスを任意指定可能
#  • ONNX は /workspace/export/onnx/ に出力
# ============================================================

# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------
usage() {
	echo "------------------------------------------------------------"
	echo " TorchSAE ONNX Export Tool"
	echo "------------------------------------------------------------"
	echo "Usage: $0 [CONFIG_PATH] [CHECKPOINT_PATH] [OUTPUT_ONNX]"
	echo ""
	echo "Examples:"
	echo "  $0"
	echo "      → 自動選択された config + 最新 checkpoint → 自動命名"
	echo ""
	echo "  $0 liae_config.json"
	echo "      → 指定 config + 最新 checkpoint → liaemodel.onnx"
	echo ""
	echo "  $0 df_config.json ckpt.pth"
	echo "      → 指定 config + 指定 checkpoint → dfmodel.onnx"
	echo ""
	echo "  $0 df_config.json ckpt.pth out.onnx"
	echo "      → すべて指定"
	echo ""
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
	usage
	exit 0
fi

# ------------------------------------------------------------
# Config 自動選択（LIAE → DF の順で優先）
# ------------------------------------------------------------
CFG_LIAE="/workspace/app/liae_config.json"
CFG_DF="/workspace/app/df_config.json"

if [[ -f "$CFG_LIAE" && -f "$CFG_DF" ]]; then
	echo "[INFO] Found both liae_config.json and df_config.json"
	echo "[INFO] Prioritizing LIAE (liae_config.json)"
	DEFAULT_CFG="$CFG_LIAE"

elif [[ -f "$CFG_LIAE" ]]; then
	echo "[INFO] Using LIAE config: liae_config.json"
	DEFAULT_CFG="$CFG_LIAE"

elif [[ -f "$CFG_DF" ]]; then
	echo "[INFO] Using DF config: df_config.json"
	DEFAULT_CFG="$CFG_DF"

else
	echo "[ERROR] No config file found."
	echo "        Expected one of:"
	echo "          • $CFG_LIAE"
	echo "          • $CFG_DF"
	exit 1
fi

CFG_PATH=${1:-$DEFAULT_CFG}

# ------------------------------------------------------------
# Config の存在チェック
# ------------------------------------------------------------
if [[ ! -f "$CFG_PATH" ]]; then
	echo "[ERROR] Specified config file not found:"
	echo "        $CFG_PATH"
	exit 1
fi

# ------------------------------------------------------------
# 最新 checkpoint 自動検出
# ------------------------------------------------------------
LATEST_CKPT=$(ls -1 /workspace/models/resume_step_*.pth 2>/dev/null | sort -V | tail -n 1)

if [[ -z "$LATEST_CKPT" ]]; then
	echo "[ERROR] No checkpoint found in /workspace/models/"
	echo "        Expected files like:"
	echo "          resume_step_XXXXX.pth"
	exit 1
fi

CKPT_PATH=${2:-$LATEST_CKPT}

# ------------------------------------------------------------
# ONNX 出力フォルダ
# ------------------------------------------------------------
OUT_DIR="/workspace/export/onnx"
mkdir -p "$OUT_DIR"

OUT_PATH=${3:-}

# ------------------------------------------------------------
# model_type を config.json から取得（自動命名用）
# ------------------------------------------------------------
MODEL_TYPE=$(jq -r '.model_type' "$CFG_PATH" | tr '[:upper:]' '[:lower:]')

# ------------------------------------------------------------
# OUT_PATH が空なら自動命名
# ------------------------------------------------------------
if [[ -z "$OUT_PATH" ]]; then
	OUT_PATH="${OUT_DIR}/${MODEL_TYPE}model.onnx"
fi

# ------------------------------------------------------------
# 実行ログ
# ------------------------------------------------------------
echo "=== Exporting TorchSAE Model to ONNX ==="
echo "  config    : ${CFG_PATH}"
echo "  checkpoint: ${CKPT_PATH}"
echo "  output    : ${OUT_PATH}"
echo ""

# ------------------------------------------------------------
# 実行
# ------------------------------------------------------------
cd /workspace/app
PYTHON=/usr/bin/python3.9
SCRIPT=export_onnx.py

$PYTHON $SCRIPT "${CFG_PATH}" "${CKPT_PATH}" "${OUT_PATH}"

echo "=== ONNX Export Completed ==="
echo "Output file: ${OUT_PATH}"
