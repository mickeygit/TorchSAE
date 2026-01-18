# TorchSAE

PyTorch ベースの DeepFaceLab SAEHD 風顔交換モデル学習フレームワーク

---

## 📋 目次

1. [プロジェクト概要](#-プロジェクト概要)
2. [フォルダ構造](#-フォルダ構造)
3. [各ディレクトリ・主要ファイルの役割](#-各ディレクトリ主要ファイルの役割)
4. [モデル構成](#-モデル構成)
5. [データフロー](#-データフロー)
6. [学習方法](#-学習方法)
7. [必要な環境・依存ライブラリ](#-必要な環境依存ライブラリ)
8. [実行例](#-実行例)
9. [注意点](#-注意点)
10. [今後の TODO・改善ポイント](#-今後の-todo改善ポイント)

---

## 🎯 プロジェクト概要

**TorchSAE** は、DeepFaceLab の SAEHD モデルを PyTorch で再実装した顔交換（Face Swap）学習フレームワークです。

### 主な特徴

- **PyTorch ベースの SAEHD 再構築**
  - 共有エンコーダ + デコーダ A/B
  - 交差再構成（A→A, B→B, A→B, B→A）
  - AMP（自動混合精度）対応で高速学習

- **XSeg マスク互換**
  - DeepFaceLab と同じ JPEG 埋め込みマスクを読み込み可能
  - マスクを用いた masked loss に対応
  - augment 時も画像とマスクを同期処理

- **モデル解像度 128 / 256 / 384**
  - `model_size` パラメータで選択可能
  - 内部構造（チャネル数・層数）は自動調整

- **学習プレビュー**
  - 指定ステップごとに再構成結果を可視化
  - A_original, A_recon, B_original, B_recon, A→B などを 1 枚に統合

- **追加学習（resume）対応**
  - checkpoint から学習再開
  - optimizer / step / scaler を完全復元

- **ONNX エクスポート**
  - 推論専用モデルを ONNX 形式で出力
  - ONNX Runtime GPU / TensorRT で高速推論可能

- **Docker 完全対応**
  - CUDA 11.8 + PyTorch 2.1 + ONNX Runtime GPU
  - 環境の完全再現が可能

---

## 📁 フォルダ構造

```
TorchSAE/
│
├── app/                           # メインアプリケーション
│   ├── __init__.py
│   ├── config.py                  # TrainConfig クラス
│   ├── main.py                    # CLI エントリーポイント
│   ├── trainer.py                 # 旧 Trainer（非推奨）
│   ├── export_onnx.py             # ONNX エクスポート
│   ├── onnx_infer_AtoB.py         # ONNX 推論スクリプト
│   ├── generate_meta_from_FAN_and_XSeg.py  # landmarks/XSeg 生成
│   │
│   ├── df_config.json             # DF モデル設定
│   ├── liae_config.json           # LIAE モデル設定
│   ├── saehd_config.json          # SAEHD モデル設定
│   │
│   ├── data/                      # データセット
│   │   ├── __init__.py
│   │   └── dataset.py             # FaceDataset（DFLJPG 対応）
│   │
│   ├── models/                    # モデル定義
│   │   ├── __init__.py
│   │   ├── autoencoder_df.py      # DF モデル
│   │   ├── autoencoder_liae.py    # LIAE モデル
│   │   ├── encoder_df.py          # DF エンコーダ
│   │   ├── decoder_df.py          # DF デコーダ
│   │   └── fan/                   # FAN（landmark 抽出）
│   │
│   ├── trainers/                  # トレーナー
│   │   ├── base_trainer.py        # BaseTrainer 基底クラス
│   │   ├── trainer_df.py          # DF 用トレーナー
│   │   └── trainer_liae.py        # LIAE 用トレーナー
│   │
│   ├── losses/                    # Loss 関数
│   │   └── loss_saehd_light.py    # DSSIM + landmark weighted loss
│   │
│   ├── merge_utils/               # マージユーティリティ
│   │   ├── __init__.py
│   │   ├── color_transfer.py
│   │   └── mask_utils.py
│   │
│   ├── utils/                     # ユーティリティ
│   │   ├── __init__.py
│   │   ├── checkpoint.py          # チェックポイント保存/ロード
│   │   ├── DFLJPG.py              # JPEG メタデータ読み込み
│   │   └── preview.py             # プレビュー画像生成
│   │
│   └── export/                    # エクスポート先
│
├── data/                          # 学習データ
│   ├── A/                         # データセット A（512×512 JPEG + XSeg）
│   │   ├── 00001.jpg
│   │   ├── 00001_landmarks.npy
│   │   └── ...
│   ├── B/                         # データセット B
│   │   ├── 00001.jpg
│   │   ├── 00001_landmarks.npy
│   │   └── ...
│   └── AtoB/                      # 推論結果出力先
│
├── models/                        # チェックポイント保存先
│   ├── step_500.pth
│   ├── step_1000.pth
│   └── ...
│
├── logs/                          # ログ
│   ├── previews/                  # プレビュー画像
│   └── tensorboard/               # TensorBoard ログ
│
├── export/                        # ONNX 出力先
│   └── onnx/
│
├── docker/                        # Docker 環境
│   ├── Dockerfile                 # Docker イメージ定義
│   ├── docker-compose.yml         # Compose 設定
│   ├── entrypoint.sh              # コンテナ起動スクリプト
│   └── requirements.txt           # Python 依存関係
│
├── container-scripts/             # コンテナ内実行スクリプト
│   ├── train.sh                   # 学習スクリプト
│   ├── export_onnx.sh             # ONNX エクスポート
│   ├── onnx_infer_AtoB.sh         # ONNX 推論
│   ├── generate_all_landmarks_and_XSeg.sh  # landmarks 生成
│   └── startup_test.sh            # 起動テスト
│
├── xseg/                          # XSeg モデル
│   └── XSeg_model_WF 5.0 model-*.onnx
│
├── config_dir/                    # 設定ファイル
├── scripts/                       # スクリプト
├── tests/                         # テスト
├── TorchSAE/                      # （サブモジュール）
│
└── menu.sh                        # メニュースクリプト
```

---

## 📦 各ディレクトリ・主要ファイルの役割

### `app/`
メインアプリケーションディレクトリ。モデル定義、データセット、トレーナー、ユーティリティを含む。

| ファイル/ディレクトリ | 役割 |
|----------------------|------|
| `main.py` | CLI エントリーポイント。`model_type` に応じて DF / LIAE を切り替え |
| `config.py` | `TrainConfig` クラス。JSON から設定を読み込み |
| `trainer.py` | 旧 Trainer（非推奨、基本は `trainers/` を使用） |
| `export_onnx.py` | チェックポイントから ONNX モデルをエクスポート |
| `onnx_infer_AtoB.py` | ONNX モデルで A→B 推論 |
| `generate_meta_from_FAN_and_XSeg.py` | FAN で landmarks、XSeg でマスクを生成 |

### `app/data/`
データセット定義。

| ファイル | 役割 |
|---------|------|
| `dataset.py` | `FaceDataset`。DFLJPG から画像・landmarks・XSeg マスクを読み込み、augment を適用 |

### `app/models/`
モデル定義。

| ファイル | 役割 |
|---------|------|
| `autoencoder_df.py` | DF モデル（共有 Encoder + Decoder A/B） |
| `autoencoder_liae.py` | LIAE モデル（Encoder + Inter + Decoder + Mask Decoder） |
| `encoder_df.py` | DF Encoder（3ch → latent） |
| `decoder_df.py` | DF Decoder（latent → 3ch） |
| `fan/` | FAN（Face Alignment Network）landmark 抽出モデル |

### `app/trainers/`
トレーナー定義。

| ファイル | 役割 |
|---------|------|
| `base_trainer.py` | `BaseTrainer` 基底クラス。Dataset / DataLoader / resume / checkpoint 保存を実装 |
| `trainer_df.py` | DF 用トレーナー |
| `trainer_liae.py` | LIAE 用トレーナー（recon loss + mask loss + landmark loss） |

### `app/losses/`
Loss 関数定義。

| ファイル | 役割 |
|---------|------|
| `loss_saehd_light.py` | DSSIM（構造的類似性）+ landmark 重み付け loss |

### `app/utils/`
ユーティリティ。

| ファイル | 役割 |
|---------|------|
| `checkpoint.py` | チェックポイント保存/ロード |
| `DFLJPG.py` | JPEG APP15 チャンクから landmarks / XSeg マスクを読み込み |
| `preview.py` | プレビュー画像生成（A_orig / A_recon / B_orig / B_recon / A→B） |

### `docker/`
Docker 環境定義。

| ファイル | 役割 |
|---------|------|
| `Dockerfile` | CUDA 11.8 + PyTorch 2.1 + ONNX Runtime GPU |
| `docker-compose.yml` | Compose 設定 |
| `entrypoint.sh` | コンテナ起動スクリプト |
| `requirements.txt` | Python 依存関係 |

### `container-scripts/`
コンテナ内実行スクリプト。

| ファイル | 役割 |
|---------|------|
| `train.sh` | 学習スクリプト。最新チェックポイント自動検出 + resume |
| `export_onnx.sh` | ONNX エクスポート |
| `onnx_infer_AtoB.sh` | ONNX 推論 |
| `generate_all_landmarks_and_XSeg.sh` | landmarks 生成 |

---

## 🏗️ モデル構成

### LIAE + SAEHD 風モデル

TorchSAE は **LIAE（DeepFaceLab の LIAE アーキテクチャ）** と **DF（DeepFaceLab 標準）** の 2 種類をサポート。

#### LIAE モデル構成

```
入力: RGB 画像 (3ch) + landmarks heatmap (1ch) → 4ch

Encoder (LIAEEncoder)
  ↓ 4ch → 128ch → 256ch → 512ch → 1024ch
  ↓ Flatten → latent vector

Inter (LIAEInter)
  ↓ FC → reshape → UpscaleBlock
  ↓ latent → feature map

Decoder (LIAEDecoder)
  ↓ UpscaleBlock × 3 + toRGB
  ↓ feature map → RGB (3ch)

MaskDecoder (LIAEMaskDecoder)
  ↓ Decoder (1ch 出力)
  ↓ feature map → mask (1ch)

Landmark Head
  ↓ FC → landmarks (68, 2)
```

#### DF モデル構成

```
入力: RGB 画像 (3ch)

Encoder (DFEncoder)
  ↓ 3ch → 64ch → 128ch → 256ch → 512ch
  ↓ Conv → latent feature map (ae_dims, H/16, W/16)

Decoder A / Decoder B (DFDecoder)
  ↓ Conv + Upsample × 4
  ↓ latent → RGB (3ch)

出力:
  - A→A (Encoder(A) → DecoderA)
  - B→B (Encoder(B) → DecoderB)
  - A→B (Encoder(A) → DecoderB)
  - B→A (Encoder(B) → DecoderA)
```

### Loss 関数

#### LIAE の場合
```python
recon_loss = DSSIM + landmark_weighted_loss
mask_loss = BCEWithLogitsLoss(mask_pred, mask_gt)
landmark_loss = L1Loss(lm_pred, lm_gt)

total_loss = recon_loss + mask_loss_weight * mask_loss + landmark_loss_weight * landmark_loss
```

#### DF の場合
```python
loss = L1Loss(A→A, A) + L1Loss(B→B, B) + L1Loss(A→B, B) + L1Loss(B→A, A)
```

---

## 🔄 データフロー

```
┌───────────────────────────────────────────────────────────┐
│  Dataset (FaceDataset)                                    │
│    - DFLJPG.load() → 画像 + landmarks + XSeg マスク       │
│    - Augmentation (warp / HSV / noise)                    │
│    - Resize (BILINEAR for image, NEAREST for mask)       │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  DataLoader                                               │
│    - batch_size / num_workers                             │
│    - shuffle / pin_memory                                 │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Model (LIAEModel / DFModel)                              │
│    - Encoder → latent                                     │
│    - Decoder A / B → recon                                │
│    - MaskDecoder → mask pred                              │
│    - Landmark Head → landmark pred                        │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Loss (SAEHDLightLoss / L1Loss)                           │
│    - DSSIM + landmark weighted loss                       │
│    - Mask BCE loss                                        │
│    - Landmark L1 loss                                     │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Optimizer (Adam / AdamW)                                 │
│    - Gradient clipping                                    │
│    - AMP (mixed precision)                                │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Checkpoint (step_xxxx.pth)                               │
│    - model state_dict                                     │
│    - optimizer state_dict                                 │
│    - scaler state_dict                                    │
│    - global_step                                          │
└───────────────────────────────────────────────────────────┘
```

---

## 🏋️ 学習方法

### 1. データ準備

```bash
data/
 ├── A/
 │    ├── 00001.jpg  ← 512×512 JPEG（XSeg マスク埋め込み）
 │    ├── 00001_landmarks.npy
 │    └── ...
 └── B/
      ├── 00001.jpg
      ├── 00001_landmarks.npy
      └── ...
```

- **画像**: 512×512 JPEG（DeepFaceLab で生成）
- **XSeg マスク**: JPEG APP15 チャンクに埋め込み
- **landmarks**: `*_landmarks.npy` ファイル（68 点、shape: (68, 2)）

landmarks が無い場合、`train.sh` が自動で FAN を使って生成します。

### 2. 学習スクリプト（`train.sh`）

`train.sh` は以下を自動実行します：

1. **landmarks チェック**: `*_landmarks.npy` が無ければ FAN で生成
2. **最新チェックポイント検出**: `models/step_*.pth` から最新を自動検出
3. **config 更新**: `resume_path` を最新チェックポイントに更新
4. **学習開始**: `python app/main.py <config.json>`

```bash
# LIAE で学習
bash container-scripts/train.sh /workspace/app/liae_config.json

# DF で学習
bash container-scripts/train.sh /workspace/app/df_config.json
```

### 3. resume の仕組み

`train.sh` は最新チェックポイントを自動検出し、`config.json` の `resume_path` を更新します。

```bash
LATEST_CKPT=$(ls -1 /workspace/models/step_*.pth 2>/dev/null | sort -V | tail -n 1)

if [ -n "$LATEST_CKPT" ]; then
    jq --arg p "$LATEST_CKPT" '.resume_path = $p' "$CONFIG_PATH" >"$tmpfile"
else
    jq '.resume_path = null' "$CONFIG_PATH" >"$tmpfile"
fi
```

`BaseTrainer._load_resume()` が checkpoint をロードし、model / optimizer / scaler / global_step を復元します。

### 4. プレビュー

`preview_interval` ごとにプレビュー画像を生成し、`logs/previews/step_xxxxx.png` に保存します。

プレビューには以下が含まれます：
- A_original
- A_xseg_mask_overlay
- A_recon（A→A）
- B_original
- B_xseg_mask_overlay
- B_recon（B→B）
- A_to_B（swap）

---

## 🛠️ 必要な環境・依存ライブラリ

### ハードウェア
- **NVIDIA GPU**: 8GB VRAM 以上推奨（RTX 3060 / 4060 以上）
- **CUDA**: 11.8 対応ドライバ

### ソフトウェア
- **Docker**: 20.10+
- **NVIDIA Container Toolkit**: GPU パススルー用
- **Docker Compose**: 1.29+（オプション）

### Python 依存関係（`requirements.txt`）

```txt
# GUI
PyQt5>=5.15.9
PyQt5-Qt5>=5.15.2
PyQt5-sip>=12.11.0

# 画像・動画処理
opencv-python>=4.8.0
Pillow>=10.0.0

# 数値計算
scipy>=1.11.0

# ユーティリティ
tqdm>=4.65.0
loguru>=0.7.0

# 設定管理
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### Dockerfile で固定されるバージョン

```dockerfile
# Python 3.9
# PyTorch 2.1.0 + cu118
# torchvision 0.16.0 + cu118
# numpy <2.0
# cupy-cuda11x
# onnxruntime-gpu 1.18.1
# insightface 0.7.3
```

---

## 🚀 実行例

### 1. Docker ビルド

```bash
cd docker
docker build -t torchsae:latest .
```

### 2. Docker Compose で起動

```bash
docker-compose up -d
docker exec -it torchsae bash
```

または手動起動：

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/app:/workspace/app \
  torchsae:latest bash
```

### 3. 学習開始

```bash
# LIAE で学習
bash container-scripts/train.sh /workspace/app/liae_config.json

# DF で学習
bash container-scripts/train.sh /workspace/app/df_config.json
```

### 4. resume（自動）

`train.sh` は自動で最新チェックポイントを検出し、resume します。

### 5. ONNX エクスポート

```bash
python app/export_onnx.py
```

または：

```bash
bash container-scripts/export_onnx.sh
```

### 6. ONNX 推論

```bash
python app/onnx_infer_AtoB.py
```

または：

```bash
bash container-scripts/onnx_infer_AtoB.sh
```

---

## ⚠️ 注意点

### 1. XSeg マスク
- **JPEG 埋め込み必須**: DeepFaceLab で生成した JPEG に XSeg マスクが埋め込まれている必要があります
- **DFLJPG.py で読み込み**: `DFLJPG.load()` が APP15 チャンクから XSeg マスクを抽出
- **マスクが無い場合**: `FaceDataset` は該当画像をスキップします

### 2. landmarks
- **68 点必須**: FAN で生成した landmarks（shape: (68, 2)）が必要
- **自動生成**: `train.sh` が landmarks が無い場合に自動生成します
- **ファイル命名**: `*_landmarks.npy` として保存（例: `00001_landmarks.npy`）

### 3. NEAREST resize（マスク）
- **マスクは NEAREST resize**: `FaceDataset` でマスクをリサイズする際は `Image.NEAREST` を使用
- **補間を避ける**: BILINEAR / BICUBIC を使うと、0/1 の境界がボケて精度が落ちます

```python
# 画像は BILINEAR
img = img.resize((self.size, self.size), Image.BILINEAR)

# マスクは NEAREST
mask = mask.resize((self.size, self.size), Image.NEAREST)
```

### 4. AMP（自動混合精度）
- **デフォルトで有効**: `amp: true` で高速化
- **CUDA < 11.0**: AMP が不安定な場合は `amp: false` に設定

### 5. Gradient Clipping
- **デフォルト: 1.0**: `clip_grad: 1` で勾配爆発を防止
- **勾配爆発が起きる場合**: `clip_grad` を 0.5 に下げる

### 6. resume
- **手動での resume**: `resume_path` を config に直接指定
- **自動 resume**: `train.sh` が最新チェックポイントを自動検出

### 7. モデルサイズ
- **128**: 軽量・高速（デバッグ用）
- **256**: 標準（推奨）
- **384**: 高品質（VRAM 12GB 以上推奨）

---

## 🚧 今後の TODO・改善ポイント

### 1. アーキテクチャ対応方針（DF → LIAE の順で実装）

TorchSAE は **DF（DeepFaceLab 標準アーキテクチャ）** を最初に実装します。

**DF を優先する理由**:
- 本家 SAEHD の標準構造であり、挙動比較が容易
- skip connection を含む対称構造で PyTorch と相性が良い
- 128 モデルでのデバッグが容易
- モジュール式（Encoder/Decoder 分離）への拡張が DF のほうが容易

**LIAE について**:
- DF より複雑（encoder/decoder の非対称構造、特殊な skip connection）
- DF の安定動作を確認した後に段階的に対応

### 2. モジュール式 Encoder/Decoder（Modular SAE）

将来的には、A→A、B→B、C→C を **個別に学習** できるモジュール式 autoencoder モードを追加します。

latent 仕様を統一することで、
- **Encoder_X + Decoder_Y の自由な組み合わせ（X→Y 変換）**
- **ONNX で Encoder/Decoder を独立エクスポートし、推論時に組み替え可能**

といった柔軟な構成を実現します。

### 3. latent distillation（互換性向上のための蒸留）

A-only モデルの latent を "教師" として B-only モデルの latent を近づける蒸留方式を検討します。

### 4. Adapter 層による柔軟な latent マッピング

```
Encoder_X → Adapter_X → shared latent
Decoder_Y はそのまま利用
```

という構成により、既存モデルを壊さずに互換性を確保する方式も検討します。

### 5. 本家 SAEHD 方式とのハイブリッド

単独 A/B/C モデルを事前学習として利用し、本家方式（A↔B の cross reconstruction）を高速化するアプローチも検討します。

### 6. TensorBoard 統合

現在は簡易ログのみ。TensorBoard でリアルタイムに loss / preview / learning rate をモニタリングできるように改善。

### 7. Multi-GPU 対応

`torch.nn.DataParallel` / `DistributedDataParallel` でマルチ GPU 学習に対応。

### 8. テストカバレッジ向上

`tests/` にユニットテストを追加し、CI/CD で自動テスト。

### 9. GUI プレビューアー

PyQt5 でリアルタイムプレビュー GUI を実装（オプション）。

### 10. ドキュメント拡張

- API リファレンス（Sphinx）
- チュートリアル動画
- 各 loss 関数の詳細説明

---

## 📜 ライセンス

MIT License

---

## 🙏 謝辞

本プロジェクトは DeepFaceLab の SAEHD モデルを参考に実装されています。

- [DeepFaceLab](https://github.com/iperov/DeepFaceLab)
- [InsightFace](https://github.com/deepinsight/insightface)
- [PyTorch](https://pytorch.org/)

---

## 📮 お問い合わせ

バグ報告・機能リクエストは GitHub Issues へお願いします。

---

**Happy Face Swapping! 🎭**
