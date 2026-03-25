import os 
import h5py
import re
import numpy as np
import zarr
from numcodecs import VLenUTF8, VLenArray
from datasets.utils.file_utils import glob_all
from datasets.utils.buffer import CompressedTrajectoryBuffer
from models.common.language import CLIPTextEncoder
import ipdb

def get_task_name_from_path(hdf5_path: str) -> str: 
    hdf5_path = hdf5_path.split("/")[-1]
    task = re.sub(r'^[A-Z_0-9]+_', '', hdf5_path)
    task = re.sub(r'_demo.*$', '', task)
    task = task.replace('_', ' ')
    return task

def create_buffer(buffer_path: str): 
    _image_shapes = {
        "agentview_rgb": (128, 128, 3), 
        "eye_in_hand_rgb": (128, 128, 3)
    }
    _lowdim_shapes = {}
    _action_shape = (7,)
    metadata = {}
    for key, shape in _image_shapes.items():
        metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
    for key, shape in _lowdim_shapes.items():
        metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
    metadata["action"] = {"shape": _action_shape, "dtype": np.float32}
    buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path,
            metadata=metadata,
    )
    return buffer

if __name__=="__main__": 
    # 1. 路径设置
    buffer_path = "/data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10/libero_10.zarr"
    hdf5_path_globs = "/data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10/*.hdf5"
    hdf5_paths = glob_all(hdf5_path_globs)

    # ... 任务名称提取部分保持不变 ...
    # ipdb.set_trace()
    task_len2task_name = {}
    for hdf5_path in hdf5_paths: 
        task_name = get_task_name_from_path(hdf5_path)
        task_len = 0
        with h5py.File(hdf5_path) as f:
            demos = f["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                task_len += demo["actions"].shape[0]
        task_len2task_name[task_len] = task_name

    # 2. 以读写模式打开原始 Zarr
    # mode='a' 表示 read/write，如果不存在则创建（这里显然已存在）
    root = zarr.open(buffer_path, mode='a')
    meta = root.require_group("meta")
    
    episode_ends = meta["episode_ends"][:]
    episode_lens = np.zeros((len(episode_ends)))
    episode_lens[1:] = episode_ends[1:] - episode_ends[:-1]
    episode_lens[0] = episode_ends[0]
    task_lens = episode_lens.reshape(-1, 50).sum(axis=1).tolist()
    task_lens = [int(d) for d in task_lens]

    # 3. 在原始 meta 中创建 Dataset
    # 使用 require_dataset 可以防止重复创建报错，同时也支持覆盖
    if "input_ids" in meta:
        print("input_ids already exists, will overwrite.")
    
    input_ids_ds = meta.require_dataset(
        name="input_ids",
        shape=(len(episode_ends),),
        dtype=object,
        object_codec=VLenArray(np.int64),
    )
    
    attention_mask_ds = meta.require_dataset(
        name="attention_mask",
        shape=(len(episode_ends),),
        dtype=object,
        object_codec=VLenArray(np.int64),
    )

    # 4. 编码并直接写入
    text_encoder = CLIPTextEncoder(embed_dim=768)
    max_shape = 0
    # ipdb.set_trace()
    for i in range(len(task_lens)):
        task_name = task_len2task_name[task_lens[i]]
        print(f"Processing Task {i}: {task_name}") 
        
        # 提取编码（在循环外做一次编码，避免内部重复 50 次同样的计算）
        input_ids, attention_mask = text_encoder.encode(task_name)
        ids_1d = input_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
        mask_1d = attention_mask.detach().cpu().numpy().reshape(-1).astype(np.int64)
        
        for j in range(50): 
            idx = i * 50 + j
            input_ids_ds[idx] = ids_1d
            attention_mask_ds[idx] = mask_1d
            
            if ids_1d.shape[0] > max_shape:
                max_shape = ids_1d.shape[0]
        
        print("Current max_shape:", max_shape)

    print("Finished updating original zarr.")