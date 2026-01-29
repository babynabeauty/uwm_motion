# setsid nohup bash scripts/libero_train.sh > libero_train_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="my_dp_test"