#!/usr/bin/env bash
# 将「GPU 等待 + 触发训练」类监控脚本放到后台运行（nohup），避免占用当前终端。
#
# 用法：
#   bash scripts/run_monitor_background.sh run_libero_MASK.sh
#   bash scripts/run_monitor_background.sh run_libero_MASK1.sh
#
# 可选环境变量：
#   WATCHER_LOG  监控脚本自身输出日志（默认：shared_workspace 下按脚本名命名）
#
# 示例：
#   WATCHER_LOG=/tmp/wait_gpu4.log bash scripts/run_monitor_background.sh run_libero_MASK.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_NAME="${1:-run_libero_MASK.sh}"
MONITOR_SCRIPT="${SCRIPT_DIR}/${MONITOR_NAME}"

if [[ ! -f "$MONITOR_SCRIPT" ]]; then
  echo "ERROR: not found: $MONITOR_SCRIPT" >&2
  exit 1
fi

DEFAULT_LOG="/data/shared_workspace/zhangshiqi/uwm_motion_data/libero_10/watcher_${MONITOR_NAME%.sh}.log"
WATCHER_LOG="${WATCHER_LOG:-$DEFAULT_LOG}"
mkdir -p "$(dirname "$WATCHER_LOG")"

nohup bash "$MONITOR_SCRIPT" >>"$WATCHER_LOG" 2>&1 &
echo "监控已后台启动，pid=$!"
echo "监控日志: $WATCHER_LOG"
echo "（训练任务日志仍由监控脚本内的 LOG_FILE 决定）"
