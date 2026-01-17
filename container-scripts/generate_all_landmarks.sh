#!/bin/bash
PYTHON=/usr/bin/python3.9

SCRIPT=/workspace/app/generate_landmarks_from_2DFAN.py

FAN_MODEL="/workspace/app/models/fan/2DFAN-4.pth.tar"

$PYTHON -m pip install face-alignment==1.3.4 --no-deps

echo "[Landmarks] Generating for A..."
$PYTHON $SCRIPT /workspace/data/A $FAN_MODEL

echo "[Landmarks] Generating for B..."
$PYTHON $SCRIPT /workspace/data/B $FAN_MODEL

echo "[Landmarks] Done."
