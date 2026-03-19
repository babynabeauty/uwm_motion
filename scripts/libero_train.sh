# setsid nohup bash scripts/libero_train.sh > /data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_bowl_drawer/libero_bowl_drawer_RAFT_no_MASK_5_mixture_V2.log 2>&1 &
# setsid nohup bash scripts/libero_train.sh > real_scoop_dp_baseline.log 2>&1 &

#_MV_no_MASK_no_mixture
#_MV_Mask_no_mixture
#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline
export PYTHONPATH=/data/workspace/zhangshiqi/uwm_motion:$PYTHONPATH

export WANDB_API_KEY=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
export WANDB_MODE=online


# ---- LIBERO (commented out) ----
CUDA_VISIBLE_DEVICES=4 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="debug" \
    model.noise_pred_net.use_motion_token=False \
    model.noise_pred_net.motion_mask=False \
    model.mixture=0.5 \
    eval_every=1000 \
    save_every=2000 \
    rollout_every=1000 \
    num_rollouts=50 \
    num_steps=50000 \
    batch_size=1 \
    optimizer.lr=2e-4 \
    dataset=libero_90 \
    model.obs_encoder.use_language=False \
    model.obs_encoder.imagenet_norm=True \
    resume=False
