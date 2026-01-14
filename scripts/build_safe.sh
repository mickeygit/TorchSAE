#!/usr/bin/env bash
#
# build_safe.sh（compose.yml の場所を明示指定 + latest チェック + 安全ビルド）
#
# 目的:
#   - 指定した docker-compose.yml を基準に安全ビルドを行う。
#   - compose.yml の image: が必ず :latest を含むことを保証する。
#   - ビルド成功時のみ latest を更新し、失敗時は安定版を保持する。
#   - 日付タグを付けて完全再現性とロールバック性を確保する。
#

set -e # エラーが出たら即終了（安全のため）

#
# 1. compose.yml の場所を明示指定
#    - ★あなたの環境に合わせてここを書き換える
#
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/../docker/docker-compose.yml"

echo "🔍 使用する compose ファイル: $COMPOSE_FILE"

#
# 2. compose.yml が存在するか確認
#
if [[ ! -f "$COMPOSE_FILE" ]]; then
	echo "❌ エラー: 指定された compose.yml が存在しません"
	echo "   パス: $COMPOSE_FILE"
	exit 1
fi

#
# 3. compose.yml の image: を抽出
#
RAW_IMAGE=$(grep -E "image:" "$COMPOSE_FILE" | awk '{print $2}')

if [[ -z "$RAW_IMAGE" ]]; then
	echo "❌ エラー: compose.yml に image: が見つかりません"
	exit 1
fi

echo "🔍 compose.yml の image: 行 → $RAW_IMAGE"

#
# 4. image: に latest が付いているかチェック
#
if [[ "$RAW_IMAGE" != *":latest" ]]; then
	echo "❌ エラー: compose.yml の image: に :latest がありません"
	echo "   image: は必ず :latest を含む必要があります"
	echo "   現在の設定: $RAW_IMAGE"
	exit 1
fi

#
# 5. latest を除いた純粋なイメージ名を抽出
#
IMAGE_NAME=$(echo "$RAW_IMAGE" | sed 's/:latest//')

echo "🔍 抽出したイメージ名: $IMAGE_NAME"

#
# 6. 日付タグ生成
#
DATE_TAG=$(date +"%Y%m%d_%H%M")

echo "=============================================="
echo "  安全ビルド開始: ${IMAGE_NAME}:${DATE_TAG}"
echo "=============================================="

#
# 7. compose.yml のあるディレクトリでビルド
#
COMPOSE_DIR=$(dirname "$COMPOSE_FILE")

echo "$COMPOSE_DIR"

cd "$COMPOSE_DIR"

if ! docker compose build; then
	echo "----------------------------------------------"
	echo "  ❌ ビルド失敗: latest は更新しません"
	echo "----------------------------------------------"
	exit 1
fi

echo "----------------------------------------------"
echo "  ✅ ビルド成功"
echo "----------------------------------------------"

#
# 8. 日付タグ付与
#
docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:${DATE_TAG}
echo "  付与: ${IMAGE_NAME}:${DATE_TAG}"

#
# 9. latest 更新（成功時のみ）
#
docker tag ${IMAGE_NAME}:${DATE_TAG} ${IMAGE_NAME}:latest
echo "  更新: ${IMAGE_NAME}:latest → ${DATE_TAG}"

echo "=============================================="
echo "  完了: 安定版 latest と日付タグが更新されました"
echo "----------------------------------------------"
echo "  利用可能なタグ:"
echo "    - ${IMAGE_NAME}:${DATE_TAG}"
echo "    - ${IMAGE_NAME}:latest"
echo "=============================================="
