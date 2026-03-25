# setsid nohup bash scripts/libero_train.sh > /data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_10/libero_10_vqvae_stride8_voc64.log 2>&1 &

#_MV_no_MASK_no_mixture
#_MV_Mask_no_mixture
#_MV_no_Mask_3_mixture
#_MV_Mask_3_mixture
#_baseline
#libero_10_vqvae_stride8_voc256
export PYTHONPATH=/data/workspace/zhangshiqi/uwm_motion_vqvae:$PYTHONPATH
export WANDB_API_KEY=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
export WANDB_MODE=online
CUDA_VISIBLE_DEVICES=2,3 python experiments/dp/train_robomimic.py \
    --config-name train_dp_robomimic.yaml \
    exp_id="libero_10_vqvae_stride8_voc64" \
    model.noise_pred_net.use_motion_token=False \
    model.noise_pred_net.motion_mask=False \
    model.mixture=0 \
    model.noise_pred_net.use_quantized_of=True \
    model.noise_pred_net.optical_flow_mask=False \
    model.noise_pred_net.quantized_of_vqvae_ckpt_path=/data/workspace/zhangshiqi/laq_flow/laq/flow_vq_results_stride8_size64/flow_vqvae_best.pt \
    model.noise_pred_net.quantized_of_vqvae_repo_path=/data/workspace/zhangshiqi/laq_flow/laq \
    model.noise_pred_net.num_flow_tokens=16 \
    num_frames=9 \
    model.action_len=8 \
    eval_every=10000 \
    save_every=10000 \
    rollout_every=10000 \
    num_rollouts=20 \
    num_steps=100000 \
    batch_size=72 \
    optimizer.lr=2e-4 \
    dataset=libero_10 \
    model.obs_encoder.use_language=False \
    model.obs_encoder.imagenet_norm=True \
    resume=False \
    # model.obs_encoder.pretrained_weights=clip \


