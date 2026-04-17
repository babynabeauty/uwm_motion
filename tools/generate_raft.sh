#!/usr/bin/env bash
set -euo pipefail

# 批量为 task-list 中 enabled 的 RoboTwin 任务生成 flow（默认只保存 raw flow，不保存 latent）。
# 支持按 GPU 列表并行：任务均分到 N 张卡，每张卡 1 个进程。
#
# 用法：
#   bash tools/generate_raft.sh
#   GPUS_CSV="0,1,2,3" bash tools/generate_raft.sh
#   TASKS_CSV="adjust_bottle,turn_switch" GPUS_CSV="4,5" bash tools/generate_raft.sh
#   OVERWRITE=1 bash tools/generate_raft.sh

TASK_LIST_FILE="${TASK_LIST_FILE:-/data/workspace/zhangshiqi/uwm_motion/configs/task_lists/robotwin_files.json}"
TASKS_CSV="${TASKS_CSV:-}"
GPUS_CSV="${GPUS_CSV:-0}"
IMG_KEY="${IMG_KEY:-obs.head_camera}"
IMG_SIZE="${IMG_SIZE:-128}"
FRAME_SKIP="${FRAME_SKIP:-6}"
OVERWRITE="${OVERWRITE:-0}"
LOG_ROOT="${LOG_ROOT:-/data/shared_workspace/zhangshiqi/uwm_motion_data/log/robotwin}"
mkdir -p "${LOG_ROOT}"

IFS=',' read -ra GPUS <<< "${GPUS_CSV// /}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS_CSV 为空，请设置如 GPUS_CSV=0,1,2,3" >&2
  exit 1
fi

readarray -t TASKS < <(
  python3 - "${TASK_LIST_FILE}" "${TASKS_CSV}" <<'PY'
import json
import sys

task_list_file = sys.argv[1]
tasks_csv = sys.argv[2].strip()
selected = {x.strip() for x in tasks_csv.split(",") if x.strip()}

with open(task_list_file, "r") as f:
    data = json.load(f)

for item in data.get("items", []):
    if not item.get("enabled", True):
        continue
    task = str(item.get("task", "")).strip()
    dataset = str(item.get("dataset", "")).strip()
    if selected and task not in selected and dataset not in selected:
        continue
    if task:
        print(task)
PY
)

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "没有可运行任务（检查 TASK_LIST_FILE / enabled / TASKS_CSV）" >&2
  exit 1
fi

echo "GPUs (${#GPUS[@]}): ${GPUS[*]}"
echo "Tasks (${#TASKS[@]}): ${TASKS[*]}"

pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  shard=()
  for ((j=i; j<${#TASKS[@]}; j+=${#GPUS[@]})); do
    shard+=("${TASKS[$j]}")
  done
  if [[ ${#shard[@]} -eq 0 ]]; then
    continue
  fi

  tasks_csv_for_gpu="$(IFS=','; echo "${shard[*]}")"
  logfile="${LOG_ROOT}/raft_gpu${gpu}.log"

  (
    cmd=(
      python tools/generate_raft_flow.py
      --mode zarr
      --task_list_file "${TASK_LIST_FILE}"
      --tasks_csv "${tasks_csv_for_gpu}"
      --frame_skip "${FRAME_SKIP}"
      --no_save_optical_flow_raft_latent
      --image_key "${IMG_KEY}"
      --img_size "${IMG_SIZE}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
      cmd+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >> "${logfile}" 2>&1
  ) &
  pids+=("$!")
  echo "[LAUNCH] gpu=${gpu} tasks=${#shard[@]} log=${logfile}"
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
echo "All RAFT jobs finished."

# # 只保存 latent，不保存 raw flow
# python tools/generate_raft_flow.py --mode zarr --zarr_path /path/to/buffer.zarr --no_save_optical_flow

# # 两者都保存（默认）
# python tools/generate_raft_flow.py --mode zarr --zarr_path /path/to/buffer.zarr
