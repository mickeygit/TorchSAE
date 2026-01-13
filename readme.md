了解だよ、ＧＩＴ。  
あなたが指定した **フォルダ構造の見直し・入力画像 512×512・モデル解像度 384/256/128・学習プレビュー仕様** をすべて反映した、**完全版 README.md** を作り直したよ。

そのままプロジェクトに貼れるように、構成も文章も整えてある。

---

# 📘 **TorchSAE — README.md（修正版・全文）**

```
# TorchSAE
PyTorch-based reimplementation of DeepFaceLab's SAEHD model, designed for modern GPU environments, reproducible training, and ONNX deployment.

TorchSAE は、DeepFaceLab の SAEHD モデルを参考にしつつ、  
PyTorch をベースに再構築した顔交換モデル学習フレームワークです。

- 入力画像は **512×512**（JPEG + XSeg マスク埋め込み）
- モデル解像度は **128 / 256 / 384** を選択可能
- A/B の autoencoder + cross-reconstruction
- XSeg マスク（JPEG 埋め込み）に対応
- 追加学習（resume）に対応
- ONNX 形式で推論モデルをエクスポート可能
- FaceRefinePlayer などの ONNX 推論パイプラインと互換
- Docker（CUDA 11.8 + PyTorch 2.1 + ONNX Runtime GPU）で完全再現可能

---

## ✨ Features

### 🔧 PyTorch ベースの SAEHD 再構築
- 共有エンコーダ + デコーダ A/B  
- 交差再構成（A→B / B→A）  
- latent サイズ・チャンネル数をパラメータ化  
- AMP（自動混合精度）対応で高速学習

### 🎭 XSeg マスク互換
- DeepFaceLab と同じ **JPEG 埋め込みマスク**を読み込み可能  
- マスクを用いた masked loss に対応  
- augment 時も画像とマスクを同期処理

### 📏 モデル解像度 128 / 256 / 384
- `--model-size 128`
- `--model-size 256`
- `--model-size 384`

内部構造（チャネル数・層数）は自動調整されます。

### 🖼 学習プレビュー（指定バッチ間隔）
指定したステップごとに以下を 1 枚にまとめて保存：

- **A_original**
- **A_xseg_mask_overlay**
- **A_recon（A→A）**
- **B_original**
- **B_xseg_mask_overlay**
- **B_recon（B→B）**
- **A_to_B（swap）**

保存先：`logs/previews/step_xxxxx.png`

### 🔁 追加学習（resume）
- checkpoint から学習再開  
- optimizer / step / config を完全復元

### 📦 ONNX エクスポート
- 推論専用モデル（Encoder + DecoderB）を ONNX 形式で出力  
- ONNX Runtime GPU / TensorRT で高速推論可能  
- FaceRefinePlayer と互換

---

## 📁 Project Structure

```
TorchSAE/
├── app/
│   ├── core/             # モデル構造（Encoder / Decoder / Autoencoder）
│   ├── preprocess/       # JPEGマスク抽出 / augment / DataLoader
│   ├── inference/        # ONNX export / runtime
│   ├── trainer/          # 学習ループ / loss / callbacks
│   ├── utils/            # logger / config / checkpoint
│   ├── gui/              # preview（任意）
│   └── main.py           # CLI エントリーポイント
├── data/                 # 顔画像 A/B（512×512 JPEG + XSeg マスク）
│   ├── A/
│   └── B/
├── docker/               # Dockerfile / compose / entrypoint
├── logs/                 # 学習ログ / プレビュー画像
│   ├── previews/
│   └── tensorboard/
├── models/               # checkpoint (.pth) / ONNX 出力
├── scripts/              # 実行スクリプト
├── tests/                # pytest
└── README.md
```

---

## 🚀 Quick Start

### 1. Docker ビルド

```
docker build -t torchsae .
```

### 2. コンテナ起動

```
docker run --gpus all -it --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  torchsae bash
```

---

## 🏋️ Training

### 1. データ準備（512×512 JPEG + XSeg マスク）

```
data/
 ├── A/
 │    ├── 00001.jpg  ← JPEG 内に XSeg マスク埋め込み
 │    ├── 00002.jpg
 │    └── ...
 └── B/
      ├── 00001.jpg
      ├── 00002.jpg
      └── ...
```

### 2. 学習開始

```
python app/main.py train \
  --data-dir-a data/A \
  --data-dir-b data/B \
  --model-size 256 \
  --batch-size 16 \
  --max-steps 200000 \
  --preview-interval 500 \
  --amp
```

### 3. 追加学習（resume）

```
python app/main.py train \
  --resume models/last.pth
```

---

## 👀 Preview

学習中、指定ステップごとに以下を 1 枚にまとめて保存：

- A_original  
- A_xseg_mask_overlay  
- A_recon（A→A）  
- B_original  
- B_xseg_mask_overlay  
- B_recon（B→B）  
- A_to_B（swap）  

保存先：`logs/previews/step_xxxxx.png`

---

## 📦 Export ONNX

```
python app/main.py export \
  --checkpoint models/last.pth \
  --output models/sae_256.onnx \
  --model-size 256 \
  --opset 17
```

---

## 🧪 Testing

```
pytest tests/
```

---

## 🛠 Requirements

- NVIDIA GPU（8GB〜推奨）  
- Docker + NVIDIA Container Toolkit  
- CUDA 11.8 対応ドライバ  
- 512×512 の顔パッチ（XSeg マスク埋め込み JPEG）

---

## 📜 License

MIT License

---
了解だよ、ＧＩＴ。  
TorchSAE の README に自然に溶け込むように、  
**「DF を最初に実装する方針」** を **Future Work セクションに統合した完成版**をまとめたよ。

そのまま README.md に貼れるように、  
文体・構成・粒度をプロジェクト仕様書レベルに整えてある。

---

# 📘 **README.md — Future Work（統合版）**

## 🚧 Future Work（今後の予定）

TorchSAE はまず **本家 SAEHD と同等の A→B 方式の安定実装**を最優先とします。  
PyTorch 化による挙動の違い（勾配安定性、AMP、マスク処理、skip connection など）を  
128 モデルから段階的に検証し、基盤を固めていきます。

そのうえで、以下の拡張を段階的に進めます。

---

## 🧱 1. アーキテクチャ対応方針（DF → LIAE の順で実装）

TorchSAE は **DF（DeepFaceLab 標準アーキテクチャ）** を最初に実装します。  
DF は本家 SAEHD で最も利用されている構造であり、  
PyTorch 化の初期検証において最も安定した挙動が期待できます。

### DF を優先する理由
- 本家 SAEHD の標準構造であり、挙動比較が容易  
- skip connection を含む対称構造で PyTorch と相性が良い  
- 128 モデルでのデバッグが容易  
- モジュール式（Encoder/Decoder 分離）への拡張が DF のほうが容易  
- PyTorch 化の基盤を固めるための最小構成として最適

### LIAE について
LIAE は DF より複雑で、  
- encoder/decoder の非対称構造  
- 特殊な skip connection  
- latent の扱いの違い  
など追加の実装コストが大きいため、  
**DF の安定動作を確認した後に段階的に対応する**方針とします。

---

## 🧩 2. モジュール式 Encoder/Decoder（Modular SAE）

将来的には、A→A、B→B、C→C を **個別に学習**できる  
モジュール式 autoencoder モードを追加します。

latent 仕様を統一することで、

- **Encoder_X + Decoder_Y の自由な組み合わせ（X→Y 変換）**  
- **ONNX で Encoder/Decoder を独立エクスポートし、推論時に組み替え可能**

といった柔軟な構成を実現します。

---

## 🧪 3. latent distillation（互換性向上のための蒸留）

A-only モデルの latent を “教師” として  
B-only モデルの latent を近づける蒸留方式を検討します。

これは本家 SAEHD のような強制的な共有 latent ではなく、  
**互換性を高めるための軽量な補助学習**として導入する可能性があります。

---

## 🔧 4. Adapter 層による柔軟な latent マッピング

Encoder_X → Adapter_X → shared latent  
Decoder_Y はそのまま利用

という構成により、  
**既存モデルを壊さずに互換性を確保する方式**も検討します。

---

## 🔄 5. 本家 SAEHD 方式とのハイブリッド

単独 A/B/C モデルを事前学習として利用し、  
本家方式（A↔B の cross reconstruction）を高速化するアプローチも検討します。

事前学習済み Encoder/Decoder を初期値として使うことで、

- 収束速度の改善  
- 安定性向上  
- 学習コスト削減  

が期待できます。

---

## 📌 設計上の注意（現時点の方針）

モジュール式に向けた latent の統一は、  
**本家方式の PyTorch 化が安定してから検討する**方針です。

理由：

- latent の仕様を固定すると、本家 SAEHD の柔軟性（archi/ae_dims/e_dims など）が制限される  
- PyTorch 化の挙動をまず本家方式で確認する必要がある  
- モジュール式は後からでも安全に導入できる

そのため、現段階では **DF を用いた本家方式の完全再現を最優先**とします。

---

必要なら、この Future Work に合わせて  
**main.py / trainer.py / model/ の設計図（DF 版）**もまとめられるよ。