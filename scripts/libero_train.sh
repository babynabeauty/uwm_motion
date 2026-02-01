# setsid nohup bash scripts/libero_train.sh > MV_no_MASK_mixture_libero_soup_cheese.log 2>&1 &
# export WANDB_API_KEY=你的wandb_api_key_粘贴在这里
# export WANDB_MODE=online
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=3 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="MV_no_MASK_mixture_libero_soup_cheese" \
    model.noise_pred_net.use_motion_token=True \
    model.noise_pred_net.motion_mask=False \
    batch_size=128 \
    optimizer.lr=3.5e-4 \
    dataset=libero_soup_cheese \
    model.mixture=0.3 \
    