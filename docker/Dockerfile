FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tokyo \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH} \
    PYTHONPATH=/workspace \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8

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
    # ffmpeg NVENC ビルドに必要な依存（現状はコメントアウト部分用）
    autoconf automake cmake pkg-config libtool yasm nasm \
    libass-dev libfreetype6-dev libvorbis-dev libopus-dev \
    libx264-dev libx265-dev libnuma-dev libvpx-dev libfdk-aac-dev zlib1g-dev libssl-dev \
    && locale-gen ja_JP.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ============================
# pip インストール（Python 3.9 に完全固定）
# ============================
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python3.9 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/local/bin/pip /usr/bin/pip \
    && python3.9 -m pip install --upgrade pip setuptools wheel

# numpy は <2.0 で固定
RUN python3.9 -m pip --no-cache-dir install "numpy>=1.24.0,<2.0"

# ============================
# PyTorch 2.1.0 + cu118
# ============================
RUN python3.9 -m pip --no-cache-dir install \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ============================
# CuPy / ONNX Runtime / insightface
# ============================
RUN python3.9 -m pip --no-cache-dir install cupy-cuda11x || true
RUN python3.9 -m pip --no-cache-dir install onnxruntime-gpu==1.18.1
RUN python3.9 -m pip --no-cache-dir install insightface==0.7.3

# # ============================
# # insightface ソース（学習用）が必要になったら有効化
# # ============================
# RUN git clone https://github.com/deepinsight/insightface.git /workspace/insightface \
#     && cd /workspace/insightface \
#     && git fetch --all --tags \
#     && git checkout v0.7

# ============================
# 残りの Python 依存関係
# ============================
RUN python3.9 -m pip --no-cache-dir install -r /workspace/requirements.txt \
    && python3.9 -m pip --no-cache-dir install --force-reinstall "numpy==1.26.4" \
    && python3.9 -m pip --no-cache-dir install pytest decord

# # ============================
# # NVENC / NVDEC 用ヘッダ（必要なら有効化）
# # ============================
# RUN git clone https://github.com/FFmpeg/nv-codec-headers --branch sdk/12.2 /tmp/nv-codec-headers && \
#     cd /tmp/nv-codec-headers && \
#     make -j"$(nproc)" && \
#     make install && \
#     ldconfig
#
# RUN ls -l /usr/local/lib/pkgconfig && \
#     pkg-config --modversion ffnvcodec
#
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

# /opt/scripts を PATH に追加する。
# /workspace は docker-compose の bind mount によりホスト側で完全に上書きされるため、
# コンテナ専用スクリプトは /opt/scripts に配置する。
# docker-compose.yml で ../container-scripts を /opt/scripts にマウントすることで、
# コンテナ専用スクリプトをホスト側で永続化しつつ、コンテナ内ではコマンドとして直接実行できる。
ENV PATH="/opt/scripts:${PATH}"

# pytorchをインストールしたpython3.9を優先する。(3.10をつかうことがある）
ENV PATH="/usr/bin:${PATH}"

# 追加
RUN apt-get update && apt-get install -y  jq

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
