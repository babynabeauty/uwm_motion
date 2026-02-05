# setsid nohup bash scripts/libero_train.sh > /data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_10/libero_10_no_MV_no_MASK_no_mixture.log 2>&1 &

#_MV_no_MASK_no_mixture
#_MV_Mask_no_mixture
#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline

export WANDB_API_KEY=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
export WANDB_MODE=online
CUDA_VISIBLE_DEVICES=6,7,8,9 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="libero_10_no_MV_no_MASK_no_mixture" \
    model.noise_pred_net.use_motion_token=False \
    model.noise_pred_net.motion_mask=False \
    eval_every=10000 \
    save_every=10000 \
    rollout_every=10000 \
    num_rollouts=2 \
    num_steps=500000 \
    batch_size=72 \
    optimizer.lr=2e-4 \
    dataset=libero_10 \
    model.mixture=0 \
    model.obs_encoder.use_language=True \
    model.obs_encoder.pretrained_weights=clip \
    model.obs_encoder.imagenet_norm=False
