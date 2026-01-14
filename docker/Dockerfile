# FaceRefinePlayer Dockerfile (CUDA 11.8 + Python 3.9 + JST + WSLg + NVDEC + NVENC 対応)

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tokyo

# NVENC を Docker に渡すための必須設定
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# pkg-config が ffnvcodec.pc を確実に検出するための設定
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}

WORKDIR /workspace

# ============================
# プロジェクト配置（context: docker）
# ============================
COPY requirements.txt /workspace/requirements.txt
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ============================
# 基本パッケージ + Python3.9
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3.9-distutils python3.9-venv \
    build-essential git ffmpeg locales language-pack-ja \
    fonts-noto-cjk fonts-noto-color-emoji \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libxcb-xinerama0 libxkbcommon-x11-0 libdbus-1-3 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-cursor0 \
    libfontconfig1 libfreetype6 libxcb-xfixes0 \
    x11-utils xdg-utils ca-certificates wget curl \
    # ffmpeg NVENC ビルドに必要な依存
    autoconf automake cmake pkg-config libtool yasm nasm \
    libass-dev libfreetype6-dev libvorbis-dev libopus-dev \
    libx264-dev libx265-dev libnuma-dev libvpx-dev libfdk-aac-dev zlib1g-dev libssl-dev \
    && locale-gen ja_JP.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ============================
# pip インストール
# ============================
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python3.9 get-pip.py \
    && rm get-pip.py

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

ENV PYTHONPATH=/workspace
ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8

RUN python -m pip install --upgrade pip setuptools wheel

# numpy は <2.0 で固定
RUN python -m pip --no-cache-dir install "numpy>=1.24.0,<2.0"

# ============================
# PyTorch 2.1.0 + cu118
# ============================
RUN python -m pip --no-cache-dir install \
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ============================
# CuPy / ONNX Runtime / insightface
# ============================
RUN python -m pip --no-cache-dir install cupy-cuda11x || true
RUN python -m pip --no-cache-dir install onnxruntime-gpu==1.18.1
RUN python -m pip --no-cache-dir install insightface==0.7.3

# ============================
# insightface ソース（学習用）
# ============================
RUN git clone https://github.com/deepinsight/insightface.git /workspace/insightface \
    && cd /workspace/insightface \
    && git fetch --all --tags \
    && git checkout v0.7

# ============================
# 残りの Python 依存関係
# ============================
RUN python -m pip --no-cache-dir install -r /workspace/requirements.txt
RUN python -m pip --no-cache-dir install --force-reinstall "numpy==1.26.4"
RUN python -m pip install pytest
RUN python -m pip install decord

# # ============================
# # NVENC / NVDEC 用ヘッダ（最新安定版：SDK 12.2）
# # ============================
# RUN git clone https://github.com/FFmpeg/nv-codec-headers --branch sdk/12.2 /tmp/nv-codec-headers && \
#     cd /tmp/nv-codec-headers && \
#     make -j"$(nproc)" && \
#     make install && \
#     ldconfig

# # ffnvcodec が正しく入ったか確認（デバッグ用）
# RUN ls -l /usr/local/lib/pkgconfig && \
#     pkg-config --modversion ffnvcodec

# # ============================
# # ★ FFmpeg（NVENC 対応）をソースからビルド
# # ============================
# RUN git clone https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg && \
#     cd /tmp/ffmpeg && \
#     ./configure \
#     --prefix=/usr/local \
#     --pkg-config-flags="--static" \
#     --extra-cflags="-I/usr/local/cuda/include" \
#     --extra-ldflags="-L/usr/local/cuda/lib64" \
#     --enable-cuda \
#     --enable-cuvid \
#     --enable-nvenc \
#     --enable-libnpp \
#     --enable-nonfree \
#     --enable-gpl \
#     --enable-libx264 \
#     --enable-libx265 \
#     --enable-libfdk-aac \
#     --enable-libass \
#     --enable-libfreetype \
#     --enable-libvpx \
#     --enable-libopus \
#     --enable-libvorbis \
#     --enable-openssl \
#     --enable-static \
#     --disable-debug \
#     --disable-doc \
#     --disable-ffplay \
#     && make -j"$(nproc)" && make install && ldconfig

# ============================
# データディレクトリ
# ============================
RUN mkdir -p /workspace/data/input /workspace/data/processed /workspace/data/annotations /workspace/logs

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
