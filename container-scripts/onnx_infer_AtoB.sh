#!/bin/bash
set -e

# ============================================
# ツール説明
# ============================================
tool_info() {
	echo "------------------------------------------------------------"
	echo " DFModel ONNX A→B Inference Tool (with XSeg + DFL Merge)"
	echo "------------------------------------------------------------"
	echo "このツールは、DFModel の ONNX モデルと XSeg モデルを使用して"
	echo "フォルダ A の画像を AtoB フォルダへ変換し、DFL 風マージを行います。"
	echo ""
	echo "主な機能:"
	echo "  • XSeg によるマスク生成"
	echo "  • DFModel による A→B 推論"
	echo "  • DFL と同じ色合わせ（color transfer）"
	echo "  • マージモード選択（raw / alpha / erode / color）"
	echo ""
}

# ============================================
# Usage 表示
# ============================================
usage() {
	tool_info
	echo "Usage: $0 [DFMODEL_ONNX] [XSEG_ONNX] [FOLDER_A] [AtoB] [MODEL_SIZE] [MERGE_MODE]"
	echo ""
	echo "Arguments:"
	echo "  DFMODEL_ONNX   DFModel の ONNX ファイル"
	echo "  XSEG_ONNX      XSeg の ONNX ファイル"
	echo "  FOLDER_A       入力フォルダ（A）"
	echo "  AtoB           出力フォルダ"
	echo "  MODEL_SIZE     ONNX 入力サイズ（128 推奨）"
	echo "  MERGE_MODE     raw / alpha / erode / color"
	echo ""
	echo "Examples:"
	echo "  $0 dfmodel.onnx xseg.onnx ./A ./AtoB 128 color"
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
DEFAULT_DFMODEL="/workspace/models/dfmodel.onnx"
DEFAULT_XSEG="/workspace/xseg/XSeg_model_WF 5.0 model-20240130T133752Z-001.onnx"
DEFAULT_A="/workspace/data/A"
DEFAULT_ATOB="/workspace/data/AtoB"
DEFAULT_MODEL_SIZE=128
DEFAULT_MERGE_MODE="color"

DFMODEL_ONNX=${1:-$DEFAULT_DFMODEL}
XSEG_ONNX=${2:-$DEFAULT_XSEG}
FOLDER_A=${3:-$DEFAULT_A}
ATOB_FOLDER=${4:-$DEFAULT_ATOB}
MODEL_SIZE=${5:-$DEFAULT_MODEL_SIZE}
MERGE_MODE=${6:-$DEFAULT_MERGE_MODE}

PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/onnx_infer_AtoB.py

echo "=== Running DFModel A→B Inference ==="
echo "  DFModel ONNX   : ${DFMODEL_ONNX}"
echo "  XSeg ONNX      : ${XSEG_ONNX}"
echo "  Input A        : ${FOLDER_A}"
echo "  Output AtoB    : ${ATOB_FOLDER}"
echo "  Model Size     : ${MODEL_SIZE}"
echo "  Merge Mode     : ${MERGE_MODE}"
echo "  Available Merge Modes : raw, alpha, erode, color"
echo ""

cd /workspace/app

# Python スクリプト実行
$PYTHON $SCRIPT "${DFMODEL_ONNX}" "${XSEG_ONNX}" "${FOLDER_A}" "${ATOB_FOLDER}" "${MODEL_SIZE}" "${MERGE_MODE}"

echo ""
echo "=== A→B Inference Completed (Shell) ==="
echo "  Merge Mode Used : ${MERGE_MODE}"
echo "========================================"
