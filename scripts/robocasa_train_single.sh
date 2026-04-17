#!/usr/bin/env bash
set -euo pipefail

# Train enabled RoboCasa atomic tasks one by one.
#
# Usage:
#   bash scripts/robocasa_train_single.sh
#
# Common overrides:
#   GPU=7 bash scripts/robocasa_train_single.sh
#   RUN_TAG=baseline_0417 bash scripts/robocasa_train_single.sh
#   TASKS_CSV=CloseDoubleDoor,OpenDrawer bash scripts/robocasa_train_single.sh
#   TASK_LIST_FILE=/path/to/robocasa_atomic_files.json bash scripts/robocasa_train_single.sh
#
# Background example:
#   setsid nohup bash scripts/robocasa_train_single.sh \
#     > /data/shared_workspace/zhangshiqi/uwm_motion_data/log/robocasa_atomic_single/master.log 2>&1 &

REPO_ROOT="${REPO_ROOT:-/data/workspace/zhangshiqi/uwm_motion}"
TASK_LIST_FILE="${TASK_LIST_FILE:-${REPO_ROOT}/configs/task_lists/robocasa_atomic_files.json}"
TASKS_CSV="${TASKS_CSV:-}"

if [[ ! -f "${TASK_LIST_FILE}" ]]; then
  echo "Task list not found: ${TASK_LIST_FILE}" >&2
  exit 1
fi

readarray -t TASK_ROWS < <(
  python3 - "${TASK_LIST_FILE}" "${TASKS_CSV}" <<'PY'
import json
import sys

task_list_file = sys.argv[1]
tasks_csv = sys.argv[2].strip()

with open(task_list_file, "r") as f:
    data = json.load(f)

only = {x.strip() for x in tasks_csv.split(",") if x.strip()} if tasks_csv else set()
for item in data.get("items", []):
    if not item.get("enabled", True):
        continue
    task = str(item.get("task", "")).strip()
    dataset = str(item.get("dataset", "")).strip()
    if not task or not dataset:
        continue
    if only and task not in only and dataset not in only:
        continue
    print(f"{task}\t{dataset}")
PY
)

if [[ ${#TASK_ROWS[@]} -eq 0 ]]; then
  echo "No enabled RoboCasa atomic tasks found. Check TASK_LIST_FILE / enabled / TASKS_CSV." >&2
  exit 1
fi

# Activate conda only when requested. Leave empty if the caller already activated the env.
CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"

if command -v rg >/dev/null 2>&1; then
  LD_LIBRARY_PATH_CLEAN="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | rg -v 'cudnn-8\.2\.1-cuda11\.3_0/lib' | paste -sd ':' -)"
else
  LD_LIBRARY_PATH_CLEAN="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'cudnn-8\.2\.1-cuda11\.3_0/lib' | paste -sd ':' -)"
fi
TORCH_CUDNN_LIB="${TORCH_CUDNN_LIB:-/data/workspace/zhangshiqi/.conda/envs/uwm/lib/python3.10/site-packages/nvidia/cudnn/lib}"
TORCH_CUBLAS_LIB="${TORCH_CUBLAS_LIB:-/data/workspace/zhangshiqi/.conda/envs/uwm/lib/python3.10/site-packages/nvidia/cublas/lib}"
export LD_LIBRARY_PATH="${TORCH_CUDNN_LIB}:${TORCH_CUBLAS_LIB}:${LD_LIBRARY_PATH_CLEAN}"

GPU="${GPU:-7}"
LOG_ROOT="${LOG_ROOT:-/data/shared_workspace/zhangshiqi/uwm_motion_data/log/robocasa_atomic_single}"
RUN_TAG="${RUN_TAG:-baseline_$(date +%m%d)}"
EXP_PREFIX="${EXP_PREFIX:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
mkdir -p "${LOG_ROOT}"

action_len="${action_len:-8}"
codebook_size="${codebook_size:-64}"
DF="${DF:-3}"
epoch="${epoch:-200}"
NUM_TOKEN="${NUM_TOKEN:-256}"
PREFIX="${PREFIX:-/data/shared_workspace/zhangshiqi/uwm_motion_data/laq/laq/output/libero}"
VQVAE_CKPT="${VQVAE_CKPT:-/data/shared_workspace/zhangshiqi/uwm_motion_data/laq/laq/output/libero_robocasa/stride_8_size128_df3_eval/flow_vqvae_epoch_120.pt}"
USE_VQVAE="${USE_VQVAE:-False}"
BS="${BS:-72}"
LR="${LR:-1e-4}"
ROLLOUT="${ROLLOUT:-10}"
RESUME="${RESUME:-False}"
OPTICAL_FLOW_MASK="${OPTICAL_FLOW_MASK:-True}"
NUM_STEPS="${NUM_STEPS:-15000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
USE_LANGUAGE="${USE_LANGUAGE:-False}"
IMAGENET_NORM="${IMAGENET_NORM:-True}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-None}"

run_one_task() {
  local task="$1"
  local dataset="$2"
  local exp_base
  local exp_id
  local log_file

  exp_base="${dataset}_${RUN_TAG}"
  if [[ -n "${EXP_PREFIX}" ]]; then
    exp_id="${EXP_PREFIX}_${exp_base}"
  else
    exp_id="${exp_base}"
  fi
  log_file="${LOG_ROOT}/${exp_id}.log"

  echo "[$(date -Iseconds)] start task=${task} dataset=${dataset} exp_id=${exp_id} log=${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU}" python "${REPO_ROOT}/experiments/dp/train_robomimic.py" \
    --config-name train_dp_robomimic.yaml \
    "exp_id=${exp_id}" \
    model.noise_pred_net.use_motion_token=False \
    model.noise_pred_net.motion_mask=False \
    model.mixture=0 \
    model.lambda_motion=0.05 \
    "model.noise_pred_net.use_quantized_of=${USE_VQVAE}" \
    "model.noise_pred_net.optical_flow_mask=${OPTICAL_FLOW_MASK}" \
    "model.noise_pred_net.quantized_of_vqvae_ckpt_path=${VQVAE_CKPT}" \
    model.noise_pred_net.quantized_of_vqvae_repo_path=/data/shared_workspace/zhangshiqi/uwm_motion_data/laq/laq \
    "model.noise_pred_net.num_flow_tokens=${NUM_TOKEN}" \
    num_frames=9 \
    "dataloader.num_workers=${NUM_WORKERS}" \
    "dataloader.prefetch_factor=${PREFETCH_FACTOR}" \
    "model.action_len=${action_len}" \
    "eval_every=${EVAL_EVERY}" \
    "save_every=${SAVE_EVERY}" \
    "rollout_every=${ROLLOUT_EVERY}" \
    "num_rollouts=${ROLLOUT}" \
    "num_steps=${NUM_STEPS}" \
    "batch_size=${BS}" \
    "optimizer.lr=${LR}" \
    "dataset=${dataset}" \
    "model.obs_encoder.use_language=${USE_LANGUAGE}" \
    "model.obs_encoder.imagenet_norm=${IMAGENET_NORM}" \
    "resume=${RESUME}" \
    "pretrain_checkpoint_path=${PRETRAIN_CKPT}" \
    2>&1 | tee "${log_file}"

  echo "[$(date -Iseconds)] done  task=${task} dataset=${dataset} exp_id=${exp_id}"
}

echo "Task list: ${TASK_LIST_FILE}"
echo "Enabled tasks: ${#TASK_ROWS[@]}"
echo "GPU: ${GPU}"
echo "RUN_TAG: ${RUN_TAG}"
echo "LOG_ROOT: ${LOG_ROOT}"

failed=()
for row in "${TASK_ROWS[@]}"; do
  IFS=$'\t' read -r task dataset <<< "${row}"
  if ! run_one_task "${task}" "${dataset}"; then
    failed+=("${dataset}")
    echo "[$(date -Iseconds)] failed task=${task} dataset=${dataset}" >&2
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      echo "STOP_ON_ERROR=1, stop remaining tasks." >&2
      exit 1
    fi
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed datasets: ${failed[*]}" >&2
  exit 1
fi

echo "[$(date -Iseconds)] all enabled RoboCasa atomic tasks finished."
