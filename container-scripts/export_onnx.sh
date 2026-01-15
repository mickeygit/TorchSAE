#!/bin/bash
set -e

# ============================================
# ツール説明
# ============================================
tool_info() {
	echo "------------------------------------------------------------"
	echo " DFModel ONNX Export Tool"
	echo "------------------------------------------------------------"
	echo "このツールは、TorchSAE/DFModel の学習済みチェックポイントから"
	echo "ONNX 形式のモデルを生成するためのユーティリティです。"
	echo ""
	echo "主な機能:"
	echo "  • 最新の resume_step_XXXXX.pth を自動検出して使用"
	echo "  • 任意の config.json / checkpoint / 出力パスを指定可能"
	echo "  • export_onnx.py を正しい引数順で呼び出し"
	echo ""
}

# ============================================
# Usage 表示
# ============================================
usage() {
	tool_info
	echo "Usage: $0 [CONFIG_PATH] [CHECKPOINT_PATH] [OUTPUT_ONNX]"
	echo ""
	echo "Examples:"
	echo "  $0"
	echo "      → config.json + 最新checkpoint → dfmodel.onnx"
	echo ""
	echo "  $0 df_config.json"
	echo "      → 指定config + 最新checkpoint → dfmodel.onnx"
	echo ""
	echo "  $0 df_config.json ckpt.pth"
	echo "      → 指定config + 指定checkpoint → dfmodel.onnx"
	echo ""
	echo "  $0 df_config.json ckpt.pth out.onnx"
	echo "      → すべて指定"
	echo ""
}

# -h / --help の場合は説明＋usage を表示して終了
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
	usage
	exit 0
fi

# ============================================
# 最新 checkpoint を自動検出
# ============================================
LATEST_CKPT=$(ls -1 /workspace/models/resume_step_*.pth 2>/dev/null | sort -V | tail -n 1)

if [ -n "$LATEST_CKPT" ]; then
	echo "[onnx_export.sh] Latest checkpoint detected: $LATEST_CKPT"
else
	echo "[onnx_export.sh] No checkpoint found."
	echo "                 ONNX export cannot proceed."
	exit 1
fi

# ============================================
# 引数処理（デフォルトは最新 checkpoint）
# ============================================
CFG_PATH=${1:-/workspace/app/df_config.json}
CKPT_PATH=${2:-$LATEST_CKPT}
OUT_PATH=${3:-/workspace/models/dfmodel.onnx}

PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/export_onnx.py

echo "=== Exporting DFModel to ONNX ==="
echo "  config    : ${CFG_PATH}"
echo "  checkpoint: ${CKPT_PATH}"
echo "  output    : ${OUT_PATH}"

cd /workspace/app

# export_onnx.py の引数順に合わせて実行
$PYTHON $SCRIPT "${CFG_PATH}" "${CKPT_PATH}" "${OUT_PATH}"

echo "=== ONNX Export Completed ==="
