
#_MV_no_MASK_no_mixture
#_MV_Mask_no_mixture
#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline
#libero_10_vqvae_stride8_voc256

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uwm  


export PYTHONPATH=/data/workspace/zhangshiqi/uwm_motion:$PYTHONPATH
export WANDB_API_KEY=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
export WANDB_MODE=online
export WANDB_DIR=/data/workspace/zhangshiqi/uwm_motion/wandb
# export UWM_RUN_DIR_BASE=/data/shared_workspace/zhangshiqi/uwm_motion_runs
# export TMPDIR=/tmp
# export TMP=/tmp
# export TEMP=/tmp
# mkdir -p "$WANDB_DIR" "$UWM_RUN_DIR_BASE" "$TMPDIR"
LD_LIBRARY_PATH_CLEAN="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | rg -v 'cudnn-8\.2\.1-cuda11\.3_0/lib' | paste -sd ':' -)"
TORCH_CUDNN_LIB=/data/workspace/zhangshiqi/.conda/envs/uwm/lib/python3.10/site-packages/nvidia/cudnn/lib
TORCH_CUBLAS_LIB=/data/workspace/zhangshiqi/.conda/envs/uwm/lib/python3.10/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH="${TORCH_CUDNN_LIB}:${TORCH_CUBLAS_LIB}:${LD_LIBRARY_PATH_CLEAN}"

"""
setsid nohup bash scripts/libero_train.sh \
> /data/shared_workspace/zhangshiqi/uwm_motion_data/log/libero/libero_10_stride8_size256_df3_200_flow_matching_0411.log 2>&1 &
"""

"""
setsid nohup bash scripts/libero_train.sh \
> /data/shared_workspace/zhangshiqi/uwm_motion_data/log/libero/libero_10_stride8_baseline_0411.log 2>&1 &
"""


action_len=8
codebook_size=64
DF=3
epoch=200
NUM_TOKEN=256
PREFIX="/data/shared_workspace/zhangshiqi/uwm_motion_data/laq/laq/output/libero"
# EXP_ID="libero_10_stride8_baseline_0411"
# EXP_ID="debug"
EXP_ID="libero_10_stride${action_len}_size${codebook_size}_df${DF}_${epoch}_flow_matching_0411"
# VQVAE_CKPT="${PREFIX}/flow_vq_results_stride${action_len}_size${codebook_size}/flow_vqvae_best.pt"
# VQVAE_CKPT="${PREFIX}/flow_vq_results_stride${action_len}_size${codebook_size}/flow_vqvae_epoch_${epoch}.pt"
VQVAE_CKPT="${PREFIX}/flow_vq_results_stride${action_len}_size${codebook_size}_df${DF}/flow_vqvae_epoch_${epoch}.pt"
# VQVAE_CKPT="None"
USE_VQVAE=True
BS=72
LR=1e-4
ROLLOUT=10
RESUME=False
OPTICAL_FLOW_MASK=True
DATASET=libero_10
PRETRAIN_CKPT=None

CUDA_VISIBLE_DEVICES=6,7 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id=$EXP_ID \
    model.noise_pred_net.use_motion_token=False \
    model.noise_pred_net.motion_mask=False \
    model.mixture=0 \
    model.lambda_motion=0.05 \
    model.noise_pred_net.use_quantized_of=$USE_VQVAE \
    model.noise_pred_net.optical_flow_mask=$OPTICAL_FLOW_MASK \
    model.noise_pred_net.quantized_of_vqvae_ckpt_path=$VQVAE_CKPT \
    model.noise_pred_net.quantized_of_vqvae_repo_path=/data/shared_workspace/zhangshiqi/uwm_motion_data/laq/laq \
    model.noise_pred_net.num_flow_tokens=$NUM_TOKEN \
    num_frames=9 \
    dataloader.num_workers=4 \
    dataloader.prefetch_factor=2 \
    model.action_len=8 \
    eval_every=10000 \
    save_every=10000 \
    rollout_every=10000 \
    num_rollouts=$ROLLOUT \
    num_steps=150000 \
    batch_size=$BS \
    optimizer.lr=$LR \
    dataset=$DATASET \
    model.obs_encoder.use_language=True \
    model.obs_encoder.imagenet_norm=False \
    resume=$RESUME \
    model.obs_encoder.pretrained_weights=clip \
    # pretrain_checkpoint_path=$PRETRAIN_CKPT 



