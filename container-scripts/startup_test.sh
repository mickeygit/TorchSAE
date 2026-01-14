#!/usr/bin/env bash
set -euo pipefail
#!/bin/bash

PYTHON=/usr/bin/python3.9

echo "===== Python version ====="
$PYTHON --version

echo "===== PyTorch CUDA availability ====="
$PYTHON - <<'EOF'
import torch
print(torch.cuda.is_available())
EOF

echo "===== NVIDIA-SMI ====="
nvidia-smi || echo "nvidia-smi not available"

echo "===== FFmpeg NVENC check ====="
ffmpeg -encoders 2>/dev/null | grep nvenc || echo "NVENC not found"

echo "===== Disk check ====="
df -h /

echo "===== Python import test ====="
$PYTHON - <<'EOF'
import sys
print("Python import OK")
EOF

echo "===== All tests completed ====="
