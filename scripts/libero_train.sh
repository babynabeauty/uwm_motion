# setsid nohup bash scripts/libero_train.sh > libero_train_motion.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="debug" \
    model.noise_pred_net.use_motion_token=False 