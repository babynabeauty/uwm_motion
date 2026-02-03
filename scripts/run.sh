#!/bin/bash

# 1. 配置基础路径
LOG_BASE_DIR="/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/"
mkdir -p "$LOG_BASE_DIR"

# 2. 数据集列表
datasets=("libero_book_caddy")

# 3. 显卡配置
gpus=(3 4 5 6 7)
num_gpus=${#gpus[@]}
gpu_idx=0

export WANDB_API_KEY=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
export WANDB_MODE=online

# 4. 循环启动
for ds in "${datasets[@]}"; do
    settings=(
        "MV_no_MASK_no_mixture;True;False;0"
        "MV_Mask_no_mixture;True;True;0"
        "MV_no_Mask_3_mixture;True;False;0.3"
        "MV_Mask_3_mixture;True;True;0.3"
        "baseline;False;False;0"
    )

    for set_str in "${settings[@]}"; do
        IFS=";" read -r s_name u_mt m_mask m_mix <<< "$set_str"
        full_exp_id="${ds}_${s_name}"
        current_gpu=${gpus[$gpu_idx]}
        
        # 定义日志路径
        log_file="${LOG_BASE_DIR}/${ds}/${full_exp_id}.log"

        echo "[GPU $current_gpu] Launching $full_exp_id -> Log: $log_file"

        # --- 核心修改：使用 nohup 风格的重定向并后台化 ---
        CUDA_VISIBLE_DEVICES=$current_gpu python experiments/dp/train_robomimic.py \
            --config-name train_dp_robomimic.yaml \
            exp_id="$full_exp_id" \
            dataset="$ds" \
            model.noise_pred_net.use_motion_token="$u_mt" \
            model.noise_pred_net.motion_mask="$m_mask" \
            model.mixture="$m_mix" \
            eval_every=1000 \
            save_every=1000 \
            rollout_every=1000 \
            num_rollouts=50 \
            num_steps=50000 \
            batch_size=128 \
            optimizer.lr=3.5e-4 > "$log_file" 2>&1 &
        
        # 记录刚刚启动的任务 PID，防止终端关闭后进程被杀死
        disown %1 

        # GPU 调度逻辑
        gpu_idx=$(( gpu_idx + 1 ))
        if [ $gpu_idx -ge $num_gpus ]; then
            echo "All GPUs occupied. Waiting for any task to finish..."
            wait -n
            gpu_idx=$(( gpu_idx - 1 ))
        fi
    done
done

echo "All tasks have been submitted to background. You can check logs in $LOG_BASE_DIR"