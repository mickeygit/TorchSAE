#!/bin/bash
PYTHON=/usr/bin/python3.9

SCRIPT=/workspace/app/generate_meta_from_FAN_and_XSeg.py

# FAN モデルは face-alignment が内部で自動ダウンロードするので不要
# XSeg モデルは Python スクリプト側で DEFAULT_XSEG として指定済み

# --no-deps　でないと、numpyはpytrchが勝手にインストールされる
$PYTHON -m pip install face-alignment==1.3.4 --no-deps

echo "[META] Generating landmarks + XSeg for A..."
$PYTHON $SCRIPT /workspace/data/A

echo "[META] Generating landmarks + XSeg for B..."
$PYTHON $SCRIPT /workspace/data/B

echo "[META] Done."
