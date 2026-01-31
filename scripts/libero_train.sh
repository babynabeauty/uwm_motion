# setsid nohup bash scripts/libero_train.sh > NO_MV_libero_bowl_drawer.log 2>&1 &
# export WANDB_API_KEY=你的wandb_api_key_粘贴在这里
# export WANDB_MODE=online

CUDA_VISIBLE_DEVICES=7,8 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="NO_MV_libero_bowl_drawer" \
    model.noise_pred_net.use_motion_token=False \
    batch_size=128 \
    optimizer.lr=3.5e-4 \
    dataset=libero_bowl_drawer 
    