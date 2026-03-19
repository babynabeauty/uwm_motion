import os

import torch.distributed as dist
from .dataset import LeRobotDataset


def make_lerobot_dataset(
    name: str,
    dataset_path: str,
    buffer_path: str,
    shape_meta: dict,
    seq_len: int,
    image_key_map: dict[str, str],
    val_ratio: float = 0.0,
    subsample_ratio: float = 1.0,
    compute_flow: bool = False,
    flow_camera_key: str = "head_rgb",
    flow_img_size: int = 224,
    sdxl_vae_path: str = "/data/shared_workspace/zhangshiqi/hf/models--stabilityai--sdxl-vae",
    normalize_action: bool = True,
):
    if not os.path.exists(buffer_path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            LeRobotDataset(
                name=name,
                dataset_path=dataset_path,
                buffer_path=buffer_path,
                shape_meta=shape_meta,
                seq_len=seq_len,
                image_key_map=image_key_map,
                compute_flow=compute_flow,
                flow_camera_key=flow_camera_key,
                flow_img_size=flow_img_size,
                sdxl_vae_path=sdxl_vae_path,
                normalize_action=False,
            )
    if dist.is_initialized():
        dist.barrier()

    train_set = LeRobotDataset(
        name=name,
        dataset_path=dataset_path,
        buffer_path=buffer_path,
        shape_meta=shape_meta,
        seq_len=seq_len,
        image_key_map=image_key_map,
        val_ratio=val_ratio,
        subsample_ratio=subsample_ratio,
        compute_flow=compute_flow,
        flow_camera_key=flow_camera_key,
        flow_img_size=flow_img_size,
        sdxl_vae_path=sdxl_vae_path,
        normalize_action=normalize_action,
    )
    val_set = train_set.get_validation_dataset()
    return train_set, val_set
