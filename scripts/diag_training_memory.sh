#!/usr/bin/env bash
# 训练时区分「整机 buff/cache」与「Python 进程 RSS」的快速诊断。
# 用法：在另一个终端训练运行时执行
#   bash scripts/diag_training_memory.sh
#
# 解读简要：
# - MemAvailable 低、Cached 很高：多为读 Zarr/磁盘导致的页缓存，不一定是进程泄漏。
# - 单个 python 的 RSS/RES 持续线性上涨：更像进程内累积，可配合降低 dataloader_num_workers、disable_rollout 做对照。

set -euo pipefail

echo "========== $(date -Iseconds) =========="
echo "---- free -w -h ----"
free -w -h || true

echo ""
echo "---- /proc/meminfo (关键行) ----"
grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree):' /proc/meminfo 2>/dev/null || true

echo ""
echo "---- Python 进程 RSS 排行 (RSS 单位 KB, 列: PID RSS CMD) ----"
ps -eo pid,rss,cmd --sort=-rss 2>/dev/null | awk 'NR==1 || /[Pp]ython/' | head -25

echo ""
echo "---- 提示 ----"
echo "若 Cached 很大但训练停掉后内存很快回落，多为内核页缓存。"
echo "对照实验: DL_NUM_WORKERS=4 bash scripts/libero_train.sh  或  DISABLE_ROLLOUT=True ..."
