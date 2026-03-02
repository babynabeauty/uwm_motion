# setsid nohup bash scripts/libero_train.sh > /data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_mug_microwave/libero_mug_microwave_RAFT_Mask_5_mixture_V2.log 2>&1 &

#_MV_no_MASK_no_mixture
#_MV_Mask_no_mixture
#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline

export WANDB_API_KEY=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
export WANDB_MODE=online
CUDA_VISIBLE_DEVICES=3,4 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="libero_mug_microwave_RAFT_Mask_5_mixture_V2" \
    model.noise_pred_net.use_motion_token=True \
    model.noise_pred_net.motion_mask=True \
    model.mixture=0.5 \
    eval_every=1000 \
    save_every=2000 \
    rollout_every=1000 \
    num_rollouts=50 \
    num_steps=50000 \
    batch_size=72 \
    optimizer.lr=2e-4 \
    dataset=libero_mug_microwave \
    model.obs_encoder.use_language=False \
    # model.obs_encoder.pretrained_weights=clip \
    model.obs_encoder.imagenet_norm=True \
    resume=False
