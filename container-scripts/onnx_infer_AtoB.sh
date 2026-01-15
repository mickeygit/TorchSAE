#!/bin/bash
set -e

# ============================================
# ツール説明
# ============================================
tool_info() {
	echo "------------------------------------------------------------"
	echo " DFModel ONNX A→B Inference Tool"
	echo "------------------------------------------------------------"
	echo "このツールは、DFModel の ONNX モデルを使用して"
	echo "フォルダ A の画像を AtoB フォルダへ変換結果として保存します。"
	echo ""
	echo "主な機能:"
	echo "  • A フォルダの画像を一括処理"
	echo "  • A→B のみ推論（out_ab）"
	echo "  • AtoB フォルダは出力先として使用（入力不要）"
	echo "  • JPG 出力"
	echo "  • 入力パラメータ・処理件数・処理時間を表示"
	echo ""
}

# ============================================
# Usage 表示
# ============================================
usage() {
	tool_info
	echo "Usage: $0 [ONNX_MODEL] [FOLDER_A] [AtoB_OUTPUT_FOLDER] [MODEL_SIZE]"
	echo ""
	echo "Arguments:"
	echo "  ONNX_MODEL           ONNX モデルファイルのパス"
	echo "  FOLDER_A             入力画像フォルダ（A）"
	echo "  AtoB_OUTPUT_FOLDER   A→B の出力先フォルダ（入力不要）"
	echo "  MODEL_SIZE           ONNX 入力サイズ（DFModel の model_size）"
	echo ""
	echo "Examples:"
	echo "  $0"
	echo "      → /workspace/models/dfmodel.onnx を使用し、"
	echo "         /workspace/data/A を入力、/workspace/data/AtoB を出力先として使用"
	echo ""
	echo "  $0 model.onnx A_folder AtoB 128"
	echo "      → 任意の ONNX / A / 出力先 / model_size を指定"
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
DEFAULT_ONNX="/workspace/models/dfmodel.onnx"
DEFAULT_A="/workspace/data/A"
DEFAULT_ATOB="/workspace/data/AtoB"
DEFAULT_MODEL_SIZE=128 # DFModel の model_size（ONNX 入力サイズ）

ONNX_PATH=${1:-$DEFAULT_ONNX}
FOLDER_A=${2:-$DEFAULT_A}
ATOB_FOLDER=${3:-$DEFAULT_ATOB}
MODEL_SIZE=${4:-$DEFAULT_MODEL_SIZE}

PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/onnx_infer_AtoB.py

echo "=== Running DFModel A→B Inference ==="
echo "  ONNX Model        : ${ONNX_PATH}"
echo "  Input Folder (A)  : ${FOLDER_A}"
echo "  Output Folder     : ${ATOB_FOLDER}"
echo "  ONNX Input Size   : ${MODEL_SIZE}"
echo ""

cd /workspace/app

# onnx_infer_AtoB.py の引数順に合わせて実行
$PYTHON $SCRIPT "${ONNX_PATH}" "${FOLDER_A}" "${ATOB_FOLDER}" "${MODEL_SIZE}"

echo ""
echo "=== A→B Inference Completed ==="
