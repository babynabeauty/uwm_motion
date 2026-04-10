#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline
#libero_10_vqvae_stride8_voc256
export PYTHONPATH=/data/workspace/zhangshiqi/uwm_motion:$PYTHONPATH
# Repo root that contains the `robocasa/` Python package (for imports + nohup bash -lc).
# Robocasa training: 默认 TRAIN_ALL_ROBOCASA=1，仅后台启动下方 ROB0CASA_DATASETS 列表（不跑 robocasa_18）。
# 单任务前台：TRAIN_ALL_ROBOCASA=0 DATASET=robocasa_OpenDrawer bash scripts/robocasa_train.sh
ROBOCASA_ROOT="${ROBOCASA_ROOT:-/data/workspace/zhangshiqi/robocasa}"
export ROBOCASA_ROOT
# Use the same interpreter as this shell; `bash -lc` may otherwise pick system `python`
# without conda packages (e.g. robocasa).
TRAIN_PYTHON="$(command -v python)"
export PYTHONPATH="/data/workspace/zhangshiqi/uwm_motion:${ROBOCASA_ROOT}:${PYTHONPATH:-}"
export WANDB_API_KEY=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
export WANDB_MODE=online
DF=3
epoch=200
NUM_TOKEN=256
# setsid nohup bash scripts/libero_train.sh > /data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_10/libero_10_stride8_size256_best_EMA.log 2>&1 &
PREFIX="/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/laq/laq/output"
EXP_ID="debug"
# EXP_ID="libero_10_stride8_baseline"
# EXP_ID="libero_10_stride8_of_EMA"
# EXP_ID="libero_10_stride${action_len}_size${codebook_size}_best_EMA"
# EXP_ID="libero_10_stride${action_len}_size${codebook_size}_df${DF}_${epoch}"
# VQVAE_CKPT="${PREFIX}/flow_vq_results_stride${action_len}_size${codebook_size}/flow_vqvae_best.pt"
# EXP_ID="robocasa_stride${action_len}_size${codebook_size}_df${DF}_${epoch}"
VQVAE_CKPT="${PREFIX}/flow_vq_results_stride${action_len}_size${codebook_size}_df${DF}/flow_vqvae_epoch_${epoch}.pt"
# VQVAE_CKPT="None"
USE_VQVAE=False
ROLLOUT=10
RESUME=False
TRAIN_ALL_ROBOCASA="${TRAIN_ALL_ROBOCASA:-1}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-8}}"
LOG_DIR="${LOG_DIR:-/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/robocasa_tasks}"
mkdir -p "$LOG_DIR"
BS="${BS:-72}"
LR="${LR:-2e-4}"
readarray -td, GPUS <<<"$GPU_LIST,"; unset 'GPUS[-1]'
# 后台多任务只跑此列表（需对应 configs/dataset/<name>.yaml）
ROB0CASA_DATASETS=(
  robocasa_CloseBlenderLid
  robocasa_CloseFridge
  robocasa_CoffeeSetupMug
  robocasa_OpenCabinet
  robocasa_OpenDrawer
  robocasa_PickPlaceCounterToCabinet
  robocasa_PickPlaceCounterToStove
  robocasa_PickPlaceDrawerToCounter
  robocasa_PickPlaceSinkToCounter
  robocasa_PickPlaceToasterToCounter
  robocasa_TurnOffStove
  robocasa_TurnOnSinkFaucet
)
# 仅当 TRAIN_ALL_ROBOCASA=0 时的单任务前台默认 dataset（可环境变量覆盖）
DATASET="${DATASET:-${ROB0CASA_DATASETS[0]}}"
launch_one () {
  local gpu="$1"
  local dataset="$2"
  local run_exp_id="${EXP_ID}_${dataset}"
  local log_path="${LOG_DIR}/${run_exp_id}.log"
  echo "Launching ${dataset} on GPU ${gpu} -> ${log_path}"
  setsid nohup bash -lc "
    export ROBOCASA_ROOT='${ROBOCASA_ROOT}'
    export PYTHONPATH=/data/workspace/zhangshiqi/uwm_motion:\${ROBOCASA_ROOT}:\${PYTHONPATH:-}
    export WANDB_API_KEY='${WANDB_API_KEY}'
    export WANDB_MODE='${WANDB_MODE}'
    export WANDB_DIR='${WANDB_DIR}'
    export UWM_RUN_DIR_BASE='${UWM_RUN_DIR_BASE}'
    export TMPDIR='${TMPDIR}'; export TMP='${TMP}'; export TEMP='${TEMP}'
    export LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'
    CUDA_VISIBLE_DEVICES=${gpu} '${TRAIN_PYTHON}' experiments/dp/train_robomimic.py \
      --config-name train_dp_robomimic.yaml \
      exp_id=${run_exp_id} \
      model.noise_pred_net.use_motion_token=False \
      model.noise_pred_net.motion_mask=False \
      model.mixture=0 \
      model.lambda_motion=0.05 \
      model.noise_pred_net.use_quantized_of=${USE_VQVAE} \
      model.noise_pred_net.optical_flow_mask=True \
      model.noise_pred_net.quantized_of_vqvae_ckpt_path='${VQVAE_CKPT}' \
      model.noise_pred_net.quantized_of_vqvae_repo_path=/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/laq/laq \
      model.noise_pred_net.num_flow_tokens=${NUM_TOKEN} \
      num_frames=9 \
      model.action_len=8 \
      eval_every=5000 \
      save_every=5000 \
      rollout_every=5000 \
      num_rollouts=${ROLLOUT} \
      num_steps=15000 \
      batch_size=${BS} \
      optimizer.lr=${LR} \
      dataset=${dataset} \
      model.obs_encoder.use_language=True \
      model.obs_encoder.imagenet_norm=False \
      resume=${RESUME} \
      model.obs_encoder.pretrained_weights=clip
  " >"$log_path" 2>&1 &
}
if [[ "$TRAIN_ALL_ROBOCASA" == "1" ]]; then
  if [[ "${#GPUS[@]}" -lt 1 ]]; then
    echo "ERROR: GPU_LIST is empty."
    exit 1
  fi
  for i in "${!ROB0CASA_DATASETS[@]}"; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    launch_one "$gpu" "${ROB0CASA_DATASETS[$i]}"
  done
  echo "All robocasa datasets launched in background."
  exit 0
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-8} python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id=${EXP_ID} \
    model.noise_pred_net.use_motion_token=False \
    model.noise_pred_net.motion_mask=False \
    model.mixture=0 \
    model.lambda_motion=0.05 \
    model.noise_pred_net.use_quantized_of=$USE_VQVAE \
    model.noise_pred_net.optical_flow_mask=True \
    model.noise_pred_net.quantized_of_vqvae_ckpt_path=$VQVAE_CKPT \
    model.noise_pred_net.quantized_of_vqvae_repo_path=/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/laq/laq \
    model.noise_pred_net.num_flow_tokens=$NUM_TOKEN \
    num_frames=9 \
    model.action_len=8 \
    eval_every=5000 \
    save_every=5000 \
    rollout_every=5000 \
    num_rollouts=$ROLLOUT \
    num_steps=150000 \
    batch_size=$BS \
    optimizer.lr=$LR \
    dataset=$DATASET \
    model.obs_encoder.use_language=True \
    model.obs_encoder.imagenet_norm=False \
    resume=$RESUME \
    model.obs_encoder.pretrained_weights=clip
