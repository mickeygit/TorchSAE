#!/bin/bash

# ---------------------------------------------------------
# TorchSAE Training Launcher
# ---------------------------------------------------------
# このコンテナには python3.10 と python3.9 が共存しているため、
# PATH に依存すると python3.10 が選ばれて torch が見えない。
# そのため、torch がインストールされている python3.9 を絶対パスで指定する。
# ---------------------------------------------------------

# /workspace を Python のモジュール検索パスに追加
export PYTHONPATH=/workspace

PYTHON=/usr/bin/python3.9
SCRIPT=/workspace/app/main.py # あなたの main スクリプトのパスに合わせて変更

# 引数はそのまま Python に渡す
$PYTHON $SCRIPT \
	--data-dir-a /workspace/data/A \
	--data-dir-b /workspace/data/B \
	--batch-size 8 \
	--max-steps 200000
