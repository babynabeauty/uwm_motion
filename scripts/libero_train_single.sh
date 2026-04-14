#!/usr/bin/env bash
# 多 RoboCasa 原子任务并行训练：24 个 dataset 映射到 8 张 GPU，每张卡上同时跑 JOBS_PER_GPU 个进程（默认 3），
# 便于前期并行写 .zarr（CPU/IO 为主）。进入正式训练后同卡多进程会争显存，若 OOM 请把 JOBS_PER_GPU 改为 1 或减小 batch。
# 用法：
#   bash scripts/libero_train_single.sh
#   GPUS_CSV="0,1,2,3,4,5,6,7" JOBS_PER_GPU=3 bash scripts/libero_train_single.sh
# 后台示例：
#   nohup bash scripts/libero_train_single.sh > robocasa_zarr.log 2>&1 &


set -euo pipefail

#_MV_no_MASK_no_mixture
#_MV_Mask_no_mixture
#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline
#libero_10_vqvae_stride8_voc256

# ---------- 任务列表配置 ----------
# 集中配置文件（可在此文件里改 enabled / 路径，不必反复改脚本）：
TASK_LIST_FILE="${TASK_LIST_FILE:-/data/workspace/zhangshiqi/uwm_motion/configs/task_lists/robocasa_atomic_files.json}"
# 可选：只跑指定任务名（task 字段），逗号分隔，例如：TASKS_CSV="OpenDrawer,TurnOffStove"
TASKS_CSV="${TASKS_CSV:-}"

# bash scripts/libero_train_single.sh
# # 只跑两个任务
# TASKS_CSV="OpenDrawer,TurnOffStove" bash scripts/libero_train_single.sh
# # 指定另一份清单
# TASK_LIST_FILE=/path/to/my_tasks.json bash scripts/libero_train_single.sh


if [[ ! -f "${TASK_LIST_FILE}" ]]; then
  echo "任务清单不存在: ${TASK_LIST_FILE}" >&2
  exit 1
fi

readarray -t DATASETS < <(
  python3 - "${TASK_LIST_FILE}" "${TASKS_CSV}" <<'PY'
import json
import sys

task_list_file = sys.argv[1]
tasks_csv = sys.argv[2].strip()

with open(task_list_file, "r") as f:
    data = json.load(f)

only = set()
if tasks_csv:
    only = {x.strip() for x in tasks_csv.split(",") if x.strip()}

items = data.get("items", [])
for it in items:
    if not it.get("enabled", True):
        continue
    task = str(it.get("task", ""))
    dataset = str(it.get("dataset", ""))
    if not dataset:
        continue
    if only and task not in only and dataset not in only:
        continue
    print(dataset)
PY
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "没有可运行的 dataset（检查 TASK_LIST_FILE / enabled / TASKS_CSV）" >&2
  exit 1
fi

# 每卡并行进程数（生成 zarr 阶段可开大；训练阶段易 OOM 时改为 1）
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"

# 可用 GPU（默认 8 张；可用 GPUS_CSV 覆盖）
if [[ -n "${GPUS_CSV:-}" ]]; then
  IFS=',' read -ra GPUS <<< "${GPUS_CSV// /}"
else
  GPUS=(0 1 2 3 4 5 6 7)
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS 为空：请设置 GPUS 数组或 GPUS_CSV，例如 GPUS_CSV=0,1,2,3,4,5,6,7" >&2
  exit 1
fi

# ---------- Conda / 环境 ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uwm

export PYTHONPATH="/data/workspace/zhangshiqi/uwm_motion:${PYTHONPATH:-}"
export WANDB_API_KEY=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
export WANDB_MODE=online
export WANDB_DIR=/data/workspace/zhangshiqi/uwm_motion/wandb
export UWM_RUN_DIR_BASE=/data/shared_workspace/zhangshiqi/uwm_motion_runs
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
mkdir -p "$WANDB_DIR" "$UWM_RUN_DIR_BASE" "$TMPDIR"

# Remove incompatible legacy cuDNN path and prioritize PyTorch bundled CUDA libs.
if command -v rg >/dev/null 2>&1; then
  LD_LIBRARY_PATH_CLEAN="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | rg -v 'cudnn-8\.2\.1-cuda11\.3_0/lib' | paste -sd ':' -)"
else
  LD_LIBRARY_PATH_CLEAN="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'cudnn-8\.2\.1-cuda11\.3_0/lib' | paste -sd ':' -)"
fi
TORCH_CUDNN_LIB="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib"
TORCH_CUBLAS_LIB="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="${TORCH_CUDNN_LIB}:${TORCH_CUBLAS_LIB}:${LD_LIBRARY_PATH_CLEAN}"

# ---------- 训练超参（与原脚本一致，可按需改）----------
action_len=8
codebook_size=64
DF=3
epoch=200
NUM_TOKEN=256
USE_VQVAE=False
PREFIX="/data/shared_workspace/zhangshiqi/uwm_motion_data/laq/laq/output"
VQVAE_CKPT="${PREFIX}/libero_robocasa/flow_vq_results_stride${action_len}_size${codebook_size}_df${DF}/flow_vqvae_epoch_${epoch}.pt"
BS=72
LR=2e-4
ROLLOUT=10
RESUME=False
LOG_ROOT="${LOG_ROOT:-/data/shared_workspace/zhangshiqi/uwm_motion_data/log/robocasa_atomic_parallel}"
mkdir -p "$LOG_ROOT"

run_one_dataset() {
  local DATASET="$1"
  local gpu="$2"
  local EXP_ID="${DATASET}_stride${action_len}_size${codebook_size}_df${DF}_e${epoch}_vqvae"
  local logfile="${LOG_ROOT}/${DATASET}.log"
  echo "[$(date -Iseconds)] GPU ${gpu} start ${DATASET} -> ${logfile}"
  CUDA_VISIBLE_DEVICES="${gpu}" python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    "exp_id=${EXP_ID}" \
    model.noise_pred_net.use_motion_token=False \
    model.noise_pred_net.motion_mask=False \
    model.mixture=0 \
    model.lambda_motion=0.05 \
    model.noise_pred_net.use_quantized_of="${USE_VQVAE}" \
    model.noise_pred_net.optical_flow_mask=True \
    model.noise_pred_net.quantized_of_vqvae_ckpt_path="${VQVAE_CKPT}" \
    model.noise_pred_net.quantized_of_vqvae_repo_path=/data/shared_workspace/zhangshiqi/uwm_motion_data/laq/laq \
    model.noise_pred_net.num_flow_tokens="${NUM_TOKEN}" \
    num_frames=9 \
    dataloader.num_workers=4 \
    dataloader.prefetch_factor=2 \
    model.action_len=8 \
    eval_every=1000 \
    save_every=1000 \
    rollout_every=1000 \
    num_rollouts="${ROLLOUT}" \
    num_steps=10000 \
    batch_size="${BS}" \
    optimizer.lr="${LR}" \
    "dataset=${DATASET}" \
    model.obs_encoder.use_language=False \
    model.obs_encoder.imagenet_norm=True \
    "resume=${RESUME}" \
    >>"${logfile}" 2>&1
  echo "[$(date -Iseconds)] GPU ${gpu} done  ${DATASET}"
}

# 映射：dataset 下标 d 放到 GPU d%G；每张卡上顺序取第 1、2、… 个属于自己的任务，且同卡最多同时跑 JOBS_PER_GPU 个。
# 例：24 任务、8 卡、JOBS_PER_GPU=3 → 卡 0 跑下标 0,8,16 且三者同时启动；共 24 个 python 同时跑。
G=${#GPUS[@]}
ND=${#DATASETS[@]}
cap=$((G * JOBS_PER_GPU))
if ((ND > cap)); then
  echo "警告: 任务数 ${ND} > ${G}×${JOBS_PER_GPU}=${cap}，多出的任务本脚本第一轮不会启动，请加卡、加大 JOBS_PER_GPU 或拆两次跑。" >&2
fi

echo "GPUs (${G}): ${GPUS[*]}  |  JOBS_PER_GPU=${JOBS_PER_GPU}  |  datasets=${ND}"
echo "Datasets: ${DATASETS[*]}"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  (
    pids=()
    n=0
    for ((j = i; j < ND; j += G)); do
      run_one_dataset "${DATASETS[j]}" "${gpu}" &
      pids+=($!)
      ((++n))
      if ((n >= JOBS_PER_GPU)); then
        break
      fi
    done
    if ((${#pids[@]} > 0)); then
      wait "${pids[@]}"
    fi
  ) &
done
wait
echo "[$(date -Iseconds)] 全部任务结束。"

# nohup 示例（每行独立注释，勿用 Python 三引号）：
# setsid nohup bash scripts/libero_train_single.sh > /data/shared_workspace/zhangshiqi/uwm_motion_data/log/robocasa_atomic_parallel/master.log 2>&1 &
