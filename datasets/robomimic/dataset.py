import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.file_utils import glob_all
from datasets.utils.sampler import TrajectorySampler
from datasets.utils.obs_utils import unflatten_obs


class RobomimicDataset(Dataset):
    def __init__(
        self,
        name: str,
        hdf5_path_globs: str,
        buffer_path: str,
        shape_meta: dict,
        seq_len: int,
        val_ratio: float = 0.0,
        subsample_ratio: float = 1.0,
        flip_rgb: bool = False,
    ):
        self.name = name
        self.seq_len = seq_len
        self.flip_rgb = flip_rgb

        # Parse observation and action shapes
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

        # Compressed buffer to store episode data
        self.buffer = self._init_buffer(hdf5_path_globs, buffer_path)

        # Create training-validation split
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

        # Apply subsample_ratio to training episodes
        if subsample_ratio < 1.0:
            train_indices = np.where(self.train_mask)[0]
            num_train_episodes = len(train_indices)
            if num_train_episodes <= 1:
                pass
            else:
                num_subsampled = round(num_train_episodes * subsample_ratio)
                num_subsampled = max(1, min(num_subsampled, num_train_episodes))

                subsampled_train_mask = np.zeros(num_episodes, dtype=bool)
                rng = np.random.default_rng(seed=1)
                sampled_indices = rng.choice(train_indices, num_subsampled, replace=False)
                subsampled_train_mask[sampled_indices] = True
                self.train_mask = subsampled_train_mask

        # Sampler to draw sequences from buffer
        self.sampler = TrajectorySampler(self.buffer, self.seq_len, self.train_mask)

    def _init_buffer(self, hdf5_path_globs, buffer_path):
        hdf5_paths = glob_all(hdf5_path_globs)
        # Create metadata
        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}
        # Must match keys written in add_episode (zeros if absent in HDF5)
        metadata["optical_flow_raft_latent"] = {
            "shape": (4, 16, 16),
            "dtype": np.float32,
        }


        # Compute buffer capacity and total demos (order must match load loop)
        capacity = 0
        num_episodes = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as f:
                demos = f["data"]
                for i in range(len(demos)):
                    demo = demos[f"demo_{i}"]
                    capacity += demo["actions"].shape[0]
                num_episodes += len(demos)

        # Initialize buffer
        buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path,
            metadata=metadata,
            capacity=capacity,
        )

        if buffer.restored:
            if buffer.num_episodes > num_episodes:
                raise ValueError(
                    f"Buffer at {buffer_path} has {buffer.num_episodes} episodes but "
                    f"current HDF5 listing has {num_episodes}. Remove the buffer directory "
                    "or restore matching HDF5 files."
                )
            if buffer.capacity != capacity:
                raise ValueError(
                    f"Buffer capacity mismatch at {buffer_path}: zarr has {buffer.capacity} "
                    f"steps but current HDF5 globs sum to {capacity}. Delete the buffer to rebuild."
                )
            # Resume index: max(syncs successful writes vs cursor when some demos failed to read)
            demos_consumed = buffer.meta.attrs.get("demos_consumed")
            if demos_consumed is not None:
                start_idx = max(buffer.num_episodes, int(demos_consumed))
            else:
                start_idx = buffer.num_episodes
            if start_idx >= num_episodes:
                buffer.meta.attrs["demos_consumed"] = num_episodes
                return buffer
            print(
                f"Resuming buffer ({start_idx}/{num_episodes} source demos already processed, "
                f"{buffer.num_episodes} episodes in zarr)..."
            )
        else:
            start_idx = 0

        pbar = tqdm(
            total=num_episodes,
            initial=start_idx,
            desc="Loading episodes to buffer",
        )
        failed_episodes = 0
        global_idx = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as f:
                demos = f["data"]
                for i in range(len(demos)):
                    if global_idx < start_idx:
                        global_idx += 1
                        continue
                    try:
                        demo = demos[f"demo_{i}"]
                        episode = {}
                        for key in self._image_shapes.keys():
                            if self.flip_rgb:
                                episode[f"obs.{key}"] = demo["obs"][key][:][:, ::-1]
                            else:
                                episode[f"obs.{key}"] = demo["obs"][key][:]
                        for key in self._lowdim_shapes.keys():
                            episode[f"obs.{key}"] = demo["obs"][key][:]
                        episode["action"] = demo["actions"][:]
                        # episode["motion_vector"] = demo["motion_vectors"][:]
                        tlen = demo["actions"].shape[0]
                        if "optical_flow_raft_latent" in demo:
                            episode["optical_flow_raft_latent"] = demo[
                                "optical_flow_raft_latent"
                            ][:]
                        else:
                            episode["optical_flow_raft_latent"] = np.zeros(
                                (tlen, 4, 16, 16), dtype=np.float32
                            )

                        buffer.add_episode(episode)
                    except OSError as e:
                        failed_episodes += 1
                        print(f"\nWarning: Failed to read demo_{i} from {hdf5_path}: {e}")
                        print(f"Skipping this episode (total failed: {failed_episodes})...")
                    global_idx += 1
                    buffer.meta.attrs["demos_consumed"] = global_idx
                    pbar.update(1)
        pbar.close()
        buffer.meta.attrs["demos_consumed"] = num_episodes
        if failed_episodes > 0:
            print(f"\nTotal episodes failed to load: {failed_episodes}/{num_episodes}")
        return buffer

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return f"<RobomimicDataset>\nname: {self.name}\nnum_samples: {len(self)}\n{self.buffer}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Sample a sequence of observations and actions from the dataset.
        data = self.sampler.sample_sequence(idx)

        # Convert data to torch tensors
        data = {k: torch.from_numpy(v) for k, v in data.items()}

        # Unflatten observations
        data = unflatten_obs(data)
        
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = TrajectorySampler(self.buffer, self.seq_len, self.val_mask)
        return val_set
