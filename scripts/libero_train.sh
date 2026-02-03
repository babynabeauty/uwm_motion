# setsid nohup bash scripts/libero_train.sh > /data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/moka_moka/book_caddy_MV_no_MASK_no_mixture.log 2>&1 &

#_MV_no_MASK_no_mixture
#_MV_Mask_no_mixture
#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline

export WANDB_API_KEY=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
export WANDB_MODE=online
CUDA_VISIBLE_DEVICES=0,1 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="book_caddy_MV_no_MASK_no_mixture" \
    model.noise_pred_net.use_motion_token=True \
    model.noise_pred_net.motion_mask=False \
    eval_every=1000 \
    save_every=1000 \
    rollout_every=1000 \
    num_rollouts=20 \
    num_steps=50000 \
    batch_size=128 \
    optimizer.lr=3.5e-4 \
    dataset=libero_moka_moka \
    model.mixture=0 \
    
