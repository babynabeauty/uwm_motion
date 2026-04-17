import os
import json
from typing import Optional, Union

import torch.distributed as dist
from torch.utils.data import ConcatDataset

from .dataset import RobomimicDataset, RobomimicZarrDataset


def make_robomimic_dataset(
    name: str,
    hdf5_path_globs: Union[str, list[str]],
    buffer_path: str,
    shape_meta: dict,
    seq_len: int,
    obs_seq_len: Optional[int] = None,
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
                obs_seq_len=obs_seq_len,
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
        obs_seq_len=obs_seq_len,
        val_ratio=val_ratio,
        subsample_ratio=subsample_ratio,
        flip_rgb=flip_rgb,
        action_slice=action_slice,
    )
    val_set = train_set.get_validation_dataset()
    return train_set, val_set


def _load_task_items(task_list_file: str, enabled_only: bool = True):
    with open(task_list_file, "r") as f:
        data = json.load(f)

    items = []
    for item in data.get("items", []):
        if enabled_only and not item.get("enabled", True):
            continue
        items.append(item)
    return items


def _parse_tasks_csv(tasks_csv: str) -> set[str]:
    names = set()
    for raw in str(tasks_csv or "").split(","):
        name = raw.strip()
        if not name:
            continue
        names.add(name)
        names.add(name.replace(" ", "_"))
        names.add(name.replace("_", " "))
    return names


def make_robomimic_multizarr_dataset(
    name: str,
    task_list_file: str,
    shape_meta: dict,
    seq_len: int,
    obs_seq_len: Optional[int] = None,
    val_ratio: float = 0.0,
    subsample_ratio: float = 1.0,
    flip_rgb: bool = False,
    action_slice: Optional[Union[list[int], tuple[int, int]]] = None,
    enabled_only: bool = False,
    tasks_csv: str = "",
):
    del flip_rgb  # Prebuilt zarrs are loaded as-is.

    if action_slice is not None:
        action_slice_t = tuple(int(x) for x in action_slice)
        if len(action_slice_t) != 2:
            raise ValueError("action_slice must be length-2 [start, end) for numpy slicing")
        action_slice = action_slice_t
    else:
        action_slice = None

    task_items = _load_task_items(task_list_file, enabled_only=enabled_only)
    selected_tasks = _parse_tasks_csv(tasks_csv)
    if selected_tasks:
        task_items = [
            item
            for item in task_items
            if str(item.get("task", "")).strip() in selected_tasks
            or str(item.get("dataset", "")).strip() in selected_tasks
        ]
    train_sets = []
    val_sets = []
    skipped = []

    for item in task_items:
        zarr_path = str(item.get("zarr", "")).strip()
        if not zarr_path:
            skipped.append(f"{item.get('dataset', '<unknown>')}: missing zarr path")
            continue
        if not os.path.exists(zarr_path):
            skipped.append(f"{item.get('dataset', '<unknown>')}: zarr not found at {zarr_path}")
            continue

        task_name = str(item.get("task", "")).strip().replace("_", " ")
        dataset_name = str(item.get("dataset", "")).strip() or os.path.basename(zarr_path)
        dataset = RobomimicZarrDataset(
            name=dataset_name,
            zarr_path=zarr_path,
            shape_meta=shape_meta,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            task_name=task_name,
            val_ratio=val_ratio,
            subsample_ratio=subsample_ratio,
            action_slice=action_slice,
        )
        train_sets.append(dataset)
        val_sets.append(dataset.get_validation_dataset())

    if skipped:
        print("Skipped multizarr entries:")
        for msg in skipped:
            print(f"  - {msg}")

    if not train_sets:
        raise RuntimeError(
            f"No usable zarr datasets found from task list: {task_list_file}"
        )

    return ConcatDataset(train_sets), ConcatDataset(val_sets)
