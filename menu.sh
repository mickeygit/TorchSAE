#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.yml"
SERVICE_NAME="torchsae-service"

# 色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_header() {
	clear
	echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
	echo -e "${BLUE}║        Training Environment - Menu           ║${NC}"
	echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
	echo ""
}

get_status() {
	docker compose -f "$COMPOSE_FILE" ps --format json 2>/dev/null |
		jq -r '.[0].State // "not running"' 2>/dev/null ||
		echo "not running"
}

show_status() {
	local status=$(get_status)
	if [[ "$status" == "running" ]]; then
		echo -e "${GREEN}● コンテナ状態: 実行中${NC}"
	else
		echo -e "${RED}● コンテナ状態: 停止中${NC}"
	fi
	echo ""
}

show_menu() {
	show_header
	show_status
	echo -e "${BLUE}【Build】${NC}"
	echo "  1) build_safe.sh を実行（安全ビルド）"
	echo ""
	echo -e "${BLUE}【Training】${NC}"
	echo "  2) train を前景で開始"
	echo "  3) train をバックグラウンドで開始"
	echo ""
	echo -e "${BLUE}【コンテナ操作】${NC}"
	echo "  4) コンテナ bash に入る"
	echo "  5) コンテナ停止"
	echo ""
	echo "  0) 終了"
	echo ""
	echo -n "選択してください [0-5]: "
}

# ==========================
# 処理
# ==========================

run_build_safe() {
	echo -e "${GREEN}build_safe.sh を実行します...${NC}"
	bash "$ROOT_DIR/scripts/build_safe.sh"
	echo -e "${GREEN}build_safe.sh 完了${NC}"
	sleep 1
}

train_foreground() {
	echo -e "${GREEN}train を前景で開始します...${NC}"
	echo -e "${YELLOW}終了するには Ctrl+C${NC}"
	docker compose -f "$COMPOSE_FILE" run --rm "$SERVICE_NAME" python train.py
}

train_background() {
	echo -e "${GREEN}train をバックグラウンドで開始します...${NC}"
	docker compose -f "$COMPOSE_FILE" up -d
	docker compose -f "$COMPOSE_FILE" exec -d "$SERVICE_NAME" python train.py
	echo -e "${GREEN}バックグラウンドで train 開始${NC}"
	sleep 1
}

enter_bash() {
	local status=$(get_status)
	if [[ "$status" != "running" ]]; then
		echo -e "${YELLOW}コンテナが起動していません。bash で起動しますか？ [y/N]: ${NC}"
		read -r ans
		if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
			docker compose -f "$COMPOSE_FILE" run --rm -it "$SERVICE_NAME" bash
			return
		else
			echo -e "${RED}キャンセルしました${NC}"
			return
		fi
	fi

	echo -e "${GREEN}bash に入ります${NC}"
	docker compose -f "$COMPOSE_FILE" exec -it "$SERVICE_NAME" bash
}

stop_container() {
	echo -e "${YELLOW}コンテナを停止します...${NC}"
	docker compose -f "$COMPOSE_FILE" stop
	echo -e "${GREEN}停止完了${NC}"
	sleep 1
}

# ==========================
# メインループ
# ==========================

main() {
	while true; do
		show_menu
		read -r choice

		case "$choice" in
		1) run_build_safe ;;
		2) train_foreground ;;
		3) train_background ;;
		4) enter_bash ;;
		5) stop_container ;;
		0)
			echo -e "${GREEN}終了します${NC}"
			exit 0
			;;
		*)
			echo -e "${RED}無効な選択です${NC}"
			sleep 1
			;;
		esac
	done
}

main
