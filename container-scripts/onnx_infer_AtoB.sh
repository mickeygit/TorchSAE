#!/bin/bash
set -e

# ============================================
# ツール説明
# ============================================
tool_info() {
	echo "------------------------------------------------------------"
	echo " TorchSAE (DF / LIAE) ONNX A→B Inference Tool"
	echo "------------------------------------------------------------"
	echo "このツールは、TorchSAE（DF / LIAE）でエクスポートした ONNX モデルと"
	echo "XSeg マスクを使用して、フォルダ A の画像を A→B 変換し、"
	echo "DFL 風マージを行うためのユーティリティです。"
	echo ""
	echo "主な機能:"
	echo "  • XSeg によるマスク生成"
	echo "  • TorchSAE（DF / LIAE）ONNX による A→B 推論"
	echo "  • DFL と同じ色合わせ（color transfer）"
	echo "  • マージモード選択（raw / alpha / erode / color）"
	echo ""
}

# ============================================
# Usage 表示
# ============================================
usage() {
	tool_info
	echo "Usage: $0 [MODEL_ONNX] [XSEG_ONNX] [FOLDER_A] [AtoB] [MODEL_SIZE] [MERGE_MODE]"
	echo ""
	echo "Arguments:"
	echo "  MODEL_ONNX     TorchSAE（DF / LIAE）ONNX ファイル"
	echo "  XSEG_ONNX      XSeg の ONNX ファイル"
	echo "  FOLDER_A       入力フォルダ（A）"
	echo "  AtoB           出力フォルダ"
	echo "  MODEL_SIZE     ONNX 入力サイズ（128 推奨）"
	echo "  MERGE_MODE     raw / alpha / erode / color"
	echo ""
	echo "Examples:"
	echo "  $0 liaemodel.onnx xseg.onnx ./A ./AtoB 128 color"
	echo "  $0 dfmodel.onnx   xseg.onnx ./A ./AtoB 128 color"
	echo ""
}

# -h / --help の場合は説明＋usage を表示して終了
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
	usage
	exit 0
fi

# ============================================
# デフォルト値
# ============================================
DEFAULT_MODEL="/workspace/export/onnx/liaemodel.onnx"
DEFAULT_XSEG="/workspace/xseg/XSeg_model_WF 5.0 model-20240130T133752Z-001.onnx"
DEFAULT_A="/workspace/data/A"
DEFAULT_ATOB="/workspace/data/AtoB"
DEFAULT_MODEL_SIZE=128
DEFAULT_MERGE_MODE="color"

MODEL_ONNX=${1:-$DEFAULT_MODEL}
XSEG_ONNX=${2:-$DEFAULT_XSEG}
FOLDER_A=${3:-$DEFAULT_A}
ATOB_FOLDER=${4:-$DEFAULT_ATOB}
MODEL_SIZE=${5:-$DEFAULT_MODEL_SIZE}
MERGE_MODE=${6:-$DEFAULT_MERGE_MODE}

PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/onnx_infer_AtoB.py

echo "=== Running TorchSAE (DF / LIAE) A→B Inference ==="
echo "  Model ONNX     : ${MODEL_ONNX}"
echo "  XSeg ONNX      : ${XSEG_ONNX}"
echo "  Input A        : ${FOLDER_A}"
echo "  Output AtoB    : ${ATOB_FOLDER}"
echo "  Model Size     : ${MODEL_SIZE}"
echo "  Merge Mode     : ${MERGE_MODE}"
echo "  Available Merge Modes : raw, alpha, erode, color"
echo ""

cd /workspace/app

# Python スクリプト実行
$PYTHON $SCRIPT "${MODEL_ONNX}" "${XSEG_ONNX}" "${FOLDER_A}" "${ATOB_FOLDER}" "${MODEL_SIZE}" "${MERGE_MODE}"

echo ""
echo "=== A→B Inference Completed (Shell) ==="
echo "  Merge Mode Used : ${MERGE_MODE}"
echo "========================================"
