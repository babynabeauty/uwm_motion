import copy
import glob
import json
import os

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import dask.array as da

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.normalizer import LinearNormalizer
from datasets.utils.sampler import TrajectorySampler
from datasets.utils.obs_utils import unflatten_obs


class LeRobotDataset(Dataset):
    def __init__(
        self,
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
        self.name = name
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.image_key_map = image_key_map
        self.compute_flow = compute_flow
        self.flow_camera_key = flow_camera_key
        self.flow_img_size = flow_img_size
        self.sdxl_vae_path = sdxl_vae_path

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

        self._reverse_image_key_map = {v: k for k, v in image_key_map.items()}

        self.buffer = self._init_buffer(buffer_path)

        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0:
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
            num_subsampled = max(1, round(num_train_episodes * subsample_ratio))
            subsampled_train_mask = np.zeros(num_episodes, dtype=bool)
            rng = np.random.default_rng(seed=1)
            sampled_indices = rng.choice(train_indices, num_subsampled, replace=False)
            subsampled_train_mask[sampled_indices] = True
            self.train_mask = subsampled_train_mask

        self.action_normalizer = self._init_action_normalizer() if normalize_action else None

        self._exclude_keys = {"optical_flow"} if self.compute_flow else set()
        self.sampler = TrajectorySampler(
            self.buffer, self.seq_len, self.train_mask, exclude_keys=self._exclude_keys,
        )

    #新增数据归一化
    def _init_action_normalizer(self) -> LinearNormalizer:
        actions = da.from_zarr(self.buffer["action"])
        min_action = actions.min(axis=0).compute()
        max_action = actions.max(axis=0).compute()
        scale = (max_action - min_action) / 2.0
        offset = (max_action + min_action) / 2.0
        print(f"Action normalizer: scale={scale}, offset={offset}")
        return LinearNormalizer(scale, offset)

    def _decode_video_frames(self, video_path: str, start_frame: int, num_frames: int) -> np.ndarray:
        """Decode a range of frames from an MP4 video file."""
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i < start_frame:
                continue
            if i >= start_frame + num_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        if len(frames) != num_frames:
            raise RuntimeError(
                f"Expected {num_frames} frames from {video_path} starting at {start_frame}, "
                f"got {len(frames)}"
            )
        return np.stack(frames)

    def _resize_images(self, images: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """Resize a batch of images (N, H, W, C) to target (th, tw)."""
        th, tw = target_hw
        if images.shape[1] == th and images.shape[2] == tw:
            return images
        resized = np.empty((images.shape[0], th, tw, images.shape[3]), dtype=np.uint8)
        for i in range(images.shape[0]):
            resized[i] = np.array(Image.fromarray(images[i]).resize((tw, th), Image.BILINEAR))
        return resized

    def _load_flow_models(self, device: torch.device):
        from tools.generate_raft_flow import build_raft, build_sdxl_vae
        print("Loading RAFT model for optical flow computation...")
        raft_model, raft_transforms = build_raft(device)
        print("Loading SDXL-VAE for flow latent encoding...")
        vae = build_sdxl_vae(device, self.sdxl_vae_path)
        return raft_model, raft_transforms, vae

    def _compute_episode_flow(
        self, frames: np.ndarray, raft_model, raft_transforms, vae, device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (optical_flow [T,H,W,2], flow_latent [T,4,h,w])."""
        from tools.generate_raft_flow import compute_raft_optical_flow, encode_flow_latents
        flows = compute_raft_optical_flow(
            frames, raft_model, raft_transforms, device,
            target_size=(self.flow_img_size, self.flow_img_size),
            batch_size=8,
        )
        latents = encode_flow_latents(flows, vae, device, latent_batch_size=16)
        return flows, latents

    def _init_buffer(self, buffer_path: str) -> CompressedTrajectoryBuffer:
        with open(os.path.join(self.dataset_path, "meta", "info.json")) as f:
            dataset_info = json.load(f)
        self._fps = dataset_info["fps"]

        ep_parquet_paths = sorted(
            glob.glob(os.path.join(self.dataset_path, "meta", "episodes", "chunk-*", "*.parquet"))
        )
        ep_meta = pd.concat([pd.read_parquet(p) for p in ep_parquet_paths], ignore_index=True)
        ep_meta = ep_meta.sort_values("episode_index").reset_index(drop=True)
        num_episodes = len(ep_meta)

        data_parquet_paths = sorted(
            glob.glob(os.path.join(self.dataset_path, "data", "chunk-*", "*.parquet"))
        )
        data_df = pd.concat([pd.read_parquet(p) for p in data_parquet_paths], ignore_index=True)
        data_df = data_df.sort_values("index").reset_index(drop=True)

        capacity = len(data_df)

        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        if self.compute_flow:
            flow_spatial = self.flow_img_size
            metadata["optical_flow"] = {
                "shape": (flow_spatial, flow_spatial, 2), "dtype": np.float32,
            }
            flow_latent_h = flow_spatial // 8
            flow_latent_w = flow_spatial // 8
            metadata["optical_flow_raft_latent"] = {
                "shape": (4, flow_latent_h, flow_latent_w), "dtype": np.float32,
            }

        buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path,
            metadata=metadata,
            capacity=capacity,
        )

        if buffer.restored:
            return buffer

        # Load flow models if needed (only during buffer creation)
        raft_model, raft_transforms, vae = None, None, None
        if self.compute_flow:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            raft_model, raft_transforms, vae = self._load_flow_models(device)

        video_frame_cache: dict[str, dict[tuple[int, int], np.ndarray]] = {}

        pbar = tqdm(total=num_episodes, desc="Loading episodes to buffer")
        for _, row in ep_meta.iterrows():
            ep_idx = int(row["episode_index"])
            ep_len = int(row["length"])
            from_idx = int(row["dataset_from_index"])
            to_idx = int(row["dataset_to_index"])

            ep_data = data_df.iloc[from_idx:to_idx]
            assert len(ep_data) == ep_len, (
                f"Episode {ep_idx}: expected {ep_len} rows, got {len(ep_data)}"
            )

            episode = {}

            actions = np.stack(ep_data["action"].values).astype(np.float32)
            episode["action"] = actions

            for lowdim_key in self._lowdim_shapes.keys():
                col_name = f"observation.{lowdim_key}"
                if col_name in ep_data.columns:
                    episode[f"obs.{lowdim_key}"] = np.stack(ep_data[col_name].values).astype(np.float32)

            flow_source_frames = None

            for image_key, target_shape in self._image_shapes.items():
                lerobot_key = self._reverse_image_key_map[image_key]
                chunk_col = f"videos/{lerobot_key}/chunk_index"
                file_col = f"videos/{lerobot_key}/file_index"
                from_ts_col = f"videos/{lerobot_key}/from_timestamp"
                to_ts_col = f"videos/{lerobot_key}/to_timestamp"

                chunk_idx = int(row[chunk_col])
                file_idx = int(row[file_col])
                from_ts = float(row[from_ts_col])

                cache_key = (chunk_idx, file_idx)
                if lerobot_key not in video_frame_cache:
                    video_frame_cache[lerobot_key] = {}

                if cache_key not in video_frame_cache[lerobot_key]:
                    video_path = os.path.join(
                        self.dataset_path, "videos", lerobot_key,
                        f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4",
                    )
                    container = av.open(video_path)
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    all_frames = []
                    for frame in container.decode(video=0):
                        all_frames.append(frame.to_ndarray(format="rgb24"))
                    container.close()
                    video_frame_cache[lerobot_key][cache_key] = np.stack(all_frames)

                all_video_frames = video_frame_cache[lerobot_key][cache_key]

                fps = self._fps
                start_frame = round(from_ts * fps)
                end_frame = start_frame + ep_len
                if end_frame > len(all_video_frames):
                    end_frame = len(all_video_frames)
                    start_frame = max(0, end_frame - ep_len)

                ep_frames = all_video_frames[start_frame:end_frame]

                if len(ep_frames) != ep_len:
                    pad_count = ep_len - len(ep_frames)
                    if pad_count > 0:
                        padding = np.repeat(ep_frames[-1:], pad_count, axis=0)
                        ep_frames = np.concatenate([ep_frames, padding], axis=0)
                    else:
                        ep_frames = ep_frames[:ep_len]

                # Keep original-resolution frames for flow before resizing
                if self.compute_flow and image_key == self.flow_camera_key:
                    flow_source_frames = ep_frames.copy()

                ep_frames = self._resize_images(ep_frames, (target_shape[0], target_shape[1]))
                episode[f"obs.{image_key}"] = ep_frames

            if self.compute_flow and flow_source_frames is not None:
                flows, latents = self._compute_episode_flow(
                    flow_source_frames, raft_model, raft_transforms, vae, device,
                )
                episode["optical_flow"] = flows
                episode["optical_flow_raft_latent"] = latents

            buffer.add_episode(episode)
            pbar.update(1)

            for lk in list(video_frame_cache.keys()):
                if len(video_frame_cache[lk]) > 2:
                    oldest_key = next(iter(video_frame_cache[lk]))
                    del video_frame_cache[lk][oldest_key]

        pbar.close()
        video_frame_cache.clear()

        # Free flow models
        if raft_model is not None:
            del raft_model, raft_transforms, vae
            torch.cuda.empty_cache()

        return buffer

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return f"<LeRobotDataset>\nname: {self.name}\nnum_samples: {len(self)}\n{self.buffer}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        #取数据的时候做归一化
        if self.action_normalizer is not None and "action" in data:
            # print("original action:", data["action"][0])
            data["action"] = self.action_normalizer(data["action"])
            # print("normalized action:", data["action"][0])
        data = {k: torch.from_numpy(v).float() for k, v in data.items()}
        data = unflatten_obs(data)
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = TrajectorySampler(
            self.buffer, self.seq_len, self.val_mask, exclude_keys=self._exclude_keys,
        )
        return val_set
