import os
from typing import Optional, Union

import torch.distributed as dist
from .dataset import RobomimicDataset


def make_robomimic_dataset(
    name: str,
    hdf5_path_globs: Union[str, list[str]],
    buffer_path: str,
    shape_meta: dict,
    seq_len: int,
    val_ratio: float = 0.0,
    subsample_ratio: float = 1.0,
    flip_rgb: bool = False,
    action_slice: Optional[Union[list[int], tuple[int, int]]] = None,
):
    if action_slice is not None:
        action_slice_t = tuple(int(x) for x in action_slice)
        if len(action_slice_t) != 2:
            raise ValueError("action_slice must be length-2 [start, end) for numpy slicing")
        action_slice = action_slice_t
    else:
        action_slice = None

    # Cache compressed dataset in the main process
    if not os.path.exists(buffer_path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            RobomimicDataset(
                name=name,
                hdf5_path_globs=hdf5_path_globs,
                buffer_path=buffer_path,
                shape_meta=shape_meta,
                seq_len=seq_len,
                flip_rgb=flip_rgb,
                action_slice=action_slice,
            )
    if dist.is_initialized():
        dist.barrier()

    # Training dataset
    train_set = RobomimicDataset(
        name=name,
        hdf5_path_globs=hdf5_path_globs,
        buffer_path=buffer_path,
        shape_meta=shape_meta,
        seq_len=seq_len,
        val_ratio=val_ratio,
        subsample_ratio=subsample_ratio,
        flip_rgb=flip_rgb,
        action_slice=action_slice,
    )
    val_set = train_set.get_validation_dataset()
    return train_set, val_set
