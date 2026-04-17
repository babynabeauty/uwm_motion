import copy
import json
import os
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.obs_utils import unflatten_obs
from datasets.utils.sampler import TrajectorySampler


def _load_task_items(task_list_file: str, enabled_only: bool = True) -> list[dict]:
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


class RobotwinZarrDataset(Dataset):
    """UWM-motion view over RoboTwin zarr files.

    RoboTwin DP baseline zarrs store low-dimensional robot state as `state` and
    language tokens as per-step arrays. This adapter maps them into the batch
    structure expected by UWM-motion:
      state -> obs.<state_obs_key>
      input_ids/attention_mask [T, L] -> [L]
    """

    def __init__(
        self,
        name: str,
        zarr_path: str,
        shape_meta: dict,
        seq_len: int,
        obs_seq_len: Optional[int] = None,
        val_ratio: float = 0.0,
        subsample_ratio: float = 1.0,
        action_slice: Optional[tuple[int, int]] = None,
        state_key: str = "state",
        state_obs_key: str = "agent_pos",
        image_keys: Optional[list[str]] = None,
        include_language: bool = True,
    ):
        self.name = name
        self.zarr_path = zarr_path
        self.seq_len = seq_len
        self.obs_seq_len = obs_seq_len
        self._action_slice = action_slice
        self.state_key = state_key
        self.state_obs_key = state_obs_key
        self.include_language = include_language

        obs_shape_meta = shape_meta["obs"]
        self._image_shapes = {}
        self._lowdim_shapes = {}
        for key, attr in obs_shape_meta.items():
            obs_type = attr["type"]
            obs_shape = tuple(attr["shape"])
            if obs_type == "rgb":
                self._image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self._lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        self._action_shape = tuple(shape_meta["action"]["shape"])
        if self._action_slice is not None:
            lo, hi = self._action_slice
            if hi - lo != self._action_shape[0]:
                raise ValueError(
                    f"action_slice [{lo}:{hi}] has width {hi - lo} but shape_meta action "
                    f"expects {self._action_shape[0]}"
                )

        self.image_keys = list(image_keys) if image_keys is not None else list(self._image_shapes.keys())
        self.buffer = self._init_buffer(zarr_path)

        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0 and num_episodes >= 2:
            num_val_episodes = round(val_ratio * num_episodes)
            num_val_episodes = min(max(num_val_episodes, 1), num_episodes - 1)
            rng = np.random.default_rng(seed=0)
            val_inds = rng.choice(num_episodes, num_val_episodes, replace=False)
            val_mask[val_inds] = True
        self.val_mask = val_mask
        self.train_mask = ~val_mask

        if subsample_ratio < 1.0:
            train_indices = np.where(self.train_mask)[0]
            num_train_episodes = len(train_indices)
            if num_train_episodes > 1:
                num_subsampled = round(num_train_episodes * subsample_ratio)
                num_subsampled = max(1, min(num_subsampled, num_train_episodes))
                subsampled_train_mask = np.zeros(num_episodes, dtype=bool)
                rng = np.random.default_rng(seed=1)
                sampled_indices = rng.choice(train_indices, num_subsampled, replace=False)
                subsampled_train_mask[sampled_indices] = True
                self.train_mask = subsampled_train_mask

        self.sampler = self._make_sampler(self.train_mask)

    def _init_buffer(self, zarr_path: str):
        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        if self.state_obs_key in self._lowdim_shapes:
            metadata[self.state_key] = {
                "shape": self._lowdim_shapes[self.state_obs_key],
                "dtype": np.float32,
            }
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        buffer = CompressedTrajectoryBuffer(storage_path=zarr_path, metadata=metadata)
        if not buffer.restored:
            raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")
        return buffer

    def _make_sampler(self, episode_mask: np.ndarray):
        keep_keys = {"action", self.state_key}
        keep_keys.update(f"obs.{k}" for k in self.image_keys)
        if self.include_language:
            keep_keys.update({"input_ids", "attention_mask"})
        for n in (6, 8, 16):
            flow_key = f"optical_flow_{n}"
            if flow_key in self.buffer:
                keep_keys.add(flow_key)
            flow_latent_key = f"optical_flow_{n}_raft_latent"
            if flow_latent_key in self.buffer:
                keep_keys.add(flow_latent_key)

        exclude_keys = {k for k in self.buffer.keys() if k not in keep_keys}
        return TrajectorySampler(
            self.buffer,
            self.seq_len,
            episode_mask,
            exclude_keys=exclude_keys,
            obs_seq_len=self.obs_seq_len,
        )

    def _slice_actions(self, actions: np.ndarray) -> np.ndarray:
        out = np.asarray(actions, dtype=np.float32)
        if self._action_slice is None:
            return out
        lo, hi = self._action_slice
        return out[:, lo:hi]

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return (
            "<RobotwinZarrDataset>\n"
            f"name: {self.name}\n"
            f"path: {self.zarr_path}\n"
            f"seq_len: {self.seq_len}, obs_seq_len: {self.obs_seq_len}\n"
            f"image_keys: {self.image_keys}, state_obs_key: {self.state_obs_key}\n"
            f"num_samples: {len(self)}\n"
            f"{self.buffer}"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        if self.state_key in data:
            state = data.pop(self.state_key)
            if self.obs_seq_len is not None:
                state = state[: self.obs_seq_len]
            data[f"obs.{self.state_obs_key}"] = state

        if "action" in data:
            data["action"] = self._slice_actions(data["action"])

        if "input_ids" in data:
            data["input_ids"] = data["input_ids"][0]
        if "attention_mask" in data:
            data["attention_mask"] = data["attention_mask"][0]

        data = {k: torch.from_numpy(v) for k, v in data.items()}
        data = unflatten_obs(data)
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = self._make_sampler(self.val_mask)
        return val_set


def make_robotwin_zarr_dataset(
    name: str,
    task_list_file: str,
    shape_meta: dict,
    seq_len: int,
    obs_seq_len: Optional[int] = None,
    val_ratio: float = 0.0,
    subsample_ratio: float = 1.0,
    action_slice: Optional[Union[list[int], tuple[int, int]]] = None,
    enabled_only: bool = True,
    tasks_csv: str = "",
    state_key: str = "state",
    state_obs_key: str = "agent_pos",
    image_keys: Optional[list[str]] = None,
    include_language: bool = True,
):
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

        dataset_name = str(item.get("dataset", "")).strip() or os.path.basename(zarr_path)
        dataset = RobotwinZarrDataset(
            name=dataset_name,
            zarr_path=zarr_path,
            shape_meta=shape_meta,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            val_ratio=val_ratio,
            subsample_ratio=subsample_ratio,
            action_slice=action_slice,
            state_key=state_key,
            state_obs_key=state_obs_key,
            image_keys=image_keys,
            include_language=include_language,
        )
        train_sets.append(dataset)
        val_sets.append(dataset.get_validation_dataset())

    if skipped:
        print("Skipped robotwin entries:")
        for msg in skipped:
            print(f"  - {msg}")
    if not train_sets:
        raise RuntimeError(f"No usable zarr datasets found from task list: {task_list_file}")

    return ConcatDataset(train_sets), ConcatDataset(val_sets)
