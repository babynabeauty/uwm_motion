# 只保存 raw flow（skip=8），不保存 latent（不加载 VAE）
python tools/generate_raft_flow.py \
    --mode zarr --zarr_path /data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10/libero_10.zarr \
    --frame_skip 16 \
    --no_save_optical_flow_raft_latent \
    --image_key obs.agentview_rgb \
    --img_size 128 
    # --overwrite \

# # 只保存 latent，不保存 raw flow
# python tools/generate_raft_flow.py --mode zarr --zarr_path /path/to/buffer.zarr --no_save_optical_flow

# # 两者都保存（默认）
# python tools/generate_raft_flow.py --mode zarr --zarr_path /path/to/buffer.zarr