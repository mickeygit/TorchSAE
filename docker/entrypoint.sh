#!/bin/bash
set -e

# コマンド引数をチェック：bash/sh など対話型シェルの場合は初期化をスキップ
SKIP_GPU_INIT=false
if [[ "$1" == "bash" || "$1" == "sh" || "$1" == "/bin/bash" || "$1" == "/bin/sh" ]]; then
	SKIP_GPU_INIT=true
fi

# GPU 初期化をスキップしない場合のみ出力
if [[ "$SKIP_GPU_INIT" == "false" ]]; then
	echo "========================================"
	echo " GPU Environment Debug Info (CUDA 11.8)"
	echo "========================================"

	# GPU デバイス確認
	echo "[GPU] nvidia-smi:"
	if command -v nvidia-smi >/dev/null 2>&1; then
		nvidia-smi || echo "nvidia-smi failed"
	else
		echo "nvidia-smi not found"
	fi
else
	echo "[ENTRYPOINT] Interactive shell detected — skipping GPU initialization"
fi

# CUDA ライブラリ存在チェック（対話シェルの場合はスキップ）
if [[ "$SKIP_GPU_INIT" == "false" ]]; then
	check_lib() {
		local name="$1"
		local found
		found=$(ldconfig -p | grep "$name" || true)

		if [ -n "$found" ]; then
			echo "[OK] $name found:"
			echo "$found"
		else
			echo "[NG] $name NOT FOUND"
		fi
	}

	echo
	echo "=== CUDA Library Check ==="
	check_lib "libcublas.so"
	check_lib "libcublasLt.so"
	check_lib "libcufft.so"
	check_lib "libcurand.so"
	check_lib "libcusolver.so"
	check_lib "libcudart.so"
fi

# NVENC ライブラリチェック & 自動修復（対話シェルの場合はスキップ）
if [[ "$SKIP_GPU_INIT" == "false" ]]; then
	echo
	echo "=== NVENC Library Check & Auto-Fix (WSL2) ==="

	NVENC_DIR="/usr/lib/wsl/lib"

	if [ -d "$NVENC_DIR" ]; then
		cd "$NVENC_DIR"

		# 実際の libnvidia-encode.so.* を取得
		REAL_LIB=$(ls libnvidia-encode.so.* 2>/dev/null | head -n 1)

		if [ -z "$REAL_LIB" ]; then
			echo "[NG] NVENC library not found in $NVENC_DIR"
		else
			echo "[OK] Found NVENC library: $REAL_LIB"

			# libnvidia-encode.so.1 が存在するか確認
			if [ ! -e libnvidia-encode.so.1 ]; then
				echo "[FIX] Creating symlink: libnvidia-encode.so.1 -> $REAL_LIB"
				sudo ln -s "$REAL_LIB" libnvidia-encode.so.1
			else
				echo "[OK] libnvidia-encode.so.1 already exists"
			fi
		fi
	else
		echo "[SKIP] $NVENC_DIR not found (not WSL2?)"
	fi
fi

# Python パッケージバージョンチェック（対話シェルの場合はスキップ）
if [[ "$SKIP_GPU_INIT" == "false" ]]; then
	echo
	echo "=== Python Package Versions ==="
	python - <<'EOF'
import torch, onnxruntime, cupy, numpy, sys
import pkg_resources

print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {numpy.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"CuPy: {cupy.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
print(f"ONNX Runtime Providers: {onnxruntime.get_available_providers()}")
EOF

	echo "========================================"
	echo " GPU Environment Debug Completed"
	echo "========================================"
fi

# エントリポイント振る舞い
# - 引数が与えられていればそれを実行（例: python -m app.main）
# - 引数がなければデフォルト bash を起動（VSCode テスト実行用）
if [ "$#" -gt 0 ]; then
	exec "$@"
else
	echo "[ENTRYPOINT] Starting bash (default mode)"
	exec bash
fi
