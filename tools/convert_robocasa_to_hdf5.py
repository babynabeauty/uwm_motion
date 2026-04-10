"""
Convert RoboCasa LeRobot-format dataset to robomimic-style HDF5 for uwm_motion training.

Two modes:
  --from_video (default): decode frames from existing MP4 videos — fast, no simulation.
  --from_render: replay states in simulation to render observations — slower, flexible.

The HDF5 format matches what RobomimicDataset expects:
  data/
    attrs: env_args (json), total, num_demos
    demo_0/
      actions  [T, action_dim]
      states   [T, state_dim]
      obs/
        robot0_agentview_left_image  [T, H, W, 3]
        robot0_agentview_right_image [T, H, W, 3]
        robot0_eye_in_hand_image     [T, H, W, 3]
      attrs: num_samples, model_file, ep_meta

Usage (video extraction, recommended):
    python tools/convert_robocasa_to_hdf5.py \
        --lerobot_dir /path/to/lerobot_dataset \
        --output /path/to/output.hdf5

Usage (simulation rendering):
    python tools/convert_robocasa_to_hdf5.py \
        --lerobot_dir /path/to/lerobot_dataset \
        --output /path/to/output.hdf5 \
        --from_render --gpu_id 0
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# HDF5 (robosuite native) action ordering for PandaOmron
_HDF5_ACTION_LAYOUT = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def _load_modality(lerobot_dir: Path) -> dict:
    path = lerobot_dir / "meta" / "modality.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _reorder_lerobot_to_hdf5(actions: np.ndarray, modality: dict) -> np.ndarray:
    """Reorder actions from LeRobot format to HDF5/robosuite native format."""
    action_info = modality["action"]
    reordered = np.zeros_like(actions)
    for key, (hdf5_start, hdf5_end) in _HDF5_ACTION_LAYOUT.items():
        if key not in action_info:
            continue
        lr_start = action_info[key]["start"]
        lr_end = action_info[key]["end"]
        reordered[:, hdf5_start:hdf5_end] = actions[:, lr_start:lr_end]
    return reordered


def _load_episode_states(episode_dir: Path) -> np.ndarray:
    return np.load(episode_dir / "states.npz")["states"]


def _load_episode_model_xml(episode_dir: Path) -> str:
    with gzip.open(episode_dir / "model.xml.gz", "rb") as f:
        return f.read().decode("utf-8")


def _load_episode_meta(episode_dir: Path) -> dict:
    with open(episode_dir / "ep_meta.json") as f:
        return json.load(f)


def _load_episode_actions(lerobot_dir: Path, ep_num: int, modality: dict) -> np.ndarray:
    data_files = sorted(lerobot_dir.glob(f"data/*/episode_{ep_num:06d}.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet found for episode {ep_num}")
    df = pd.read_parquet(data_files[0])
    actions = np.stack(df["action"].to_list()).astype(np.float32)
    if modality is not None:
        actions = _reorder_lerobot_to_hdf5(actions, modality)
    return actions


def _load_info(lerobot_dir: Path) -> dict:
    with open(lerobot_dir / "meta" / "info.json") as f:
        return json.load(f)


def _load_env_metadata(lerobot_dir: Path) -> dict:
    with open(lerobot_dir / "extras" / "dataset_meta.json") as f:
        meta = json.load(f)
    return meta["env_args"]


def _load_episodes_jsonl(lerobot_dir: Path) -> list:
    episodes = []
    with open(lerobot_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


# ── Video extraction ──────────────────────────────────────────────────────────

def _video_key_to_hdf5_key(video_key: str) -> str:
    """observation.images.robot0_agentview_left → robot0_agentview_left_image"""
    name = video_key.replace("observation.images.", "")
    return name + "_image"


def _extract_video_frames(
    video_path: Path, num_frames: int, target_h: int, target_w: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)


def _get_video_path(lerobot_dir: Path, info: dict, ep_num: int, video_key: str) -> Path:
    tmpl = info["video_path"]
    chunk_size = info.get("chunks_size", 1000)
    chunk_idx = ep_num // chunk_size
    rel = tmpl.format(episode_chunk=chunk_idx, video_key=video_key, episode_index=ep_num)
    return lerobot_dir / rel


# ── Simulation rendering ─────────────────────────────────────────────────────

def _create_env(env_meta, camera_names, camera_height, camera_width, gpu_id):
    import robosuite
    from copy import deepcopy

    kwargs = deepcopy(env_meta.get("env_kwargs", {}))
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    if gpu_id is not None:
        kwargs["render_gpu_device_id"] = gpu_id
    return robosuite.make(env_meta["env_name"], **kwargs)


def _render_episode(env, states, model_xml, ep_meta, camera_names):
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)

    env.reset()
    xml = env.edit_model_xml(model_xml) if hasattr(env, "edit_model_xml") else model_xml
    env.reset_from_xml_string(xml)
    env.sim.reset()

    obs_dict = {cam + "_image": [] for cam in camera_names}
    h = env.camera_heights[0] if isinstance(env.camera_heights, list) else env.camera_heights
    w = env.camera_widths[0] if isinstance(env.camera_widths, list) else env.camera_widths

    for t in range(len(states)):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        if hasattr(env, "update_state"):
            env.update_state()
        elif hasattr(env, "update_sites"):
            env.update_sites()
        for cam in camera_names:
            img = env.sim.render(height=h, width=w, camera_name=cam)[::-1]
            obs_dict[cam + "_image"].append(img)

    return {k: np.stack(v) for k, v in obs_dict.items()}


# ── Main conversion ──────────────────────────────────────────────────────────

def convert(args):
    lerobot_dir = Path(args.lerobot_dir)
    output_path = Path(args.output)
    action_dim = args.action_dim
    from_video = not args.from_render

    info = _load_info(lerobot_dir)
    modality = _load_modality(lerobot_dir)
    env_meta = _load_env_metadata(lerobot_dir)
    episodes = _load_episodes_jsonl(lerobot_dir)

    extras_dir = lerobot_dir / "extras"
    episode_dirs = sorted(extras_dir.glob("episode_*"))
    episode_dirs = [d for d in episode_dirs if d.name != "dataset_meta.json"]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* dirs in {extras_dir}")

    num_episodes = len(episode_dirs)
    print(f"Found {num_episodes} episodes in {extras_dir}")

    camera_names = args.camera_names
    video_keys = [f"observation.images.{c}" for c in camera_names]

    env = None
    if not from_video:
        env = _create_env(env_meta, camera_names, args.camera_height, args.camera_width, args.gpu_id)
        print(f"Environment created: {env_meta['env_name']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(output_path), "w") as f_out:
        data_grp = f_out.create_group("data")
        data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)

        total_samples = 0
        written = 0
        for ep_dir in tqdm(episode_dirs, desc="Converting"):
            ep_num = int(ep_dir.name.split("_")[-1])
            try:
                states = _load_episode_states(ep_dir)
                model_xml = _load_episode_model_xml(ep_dir)
                ep_meta_dict = _load_episode_meta(ep_dir)
                actions = _load_episode_actions(lerobot_dir, ep_num, modality)
            except Exception as e:
                print(f"\nSkip episode {ep_num}: {e}")
                continue

            ep_len = min(len(states), len(actions))

            if from_video:
                obs_dict = {}
                ok = True
                for vk in video_keys:
                    vp = _get_video_path(lerobot_dir, info, ep_num, vk)
                    if not vp.exists():
                        print(f"\nSkip episode {ep_num}: missing {vp}")
                        ok = False
                        break
                    frames = _extract_video_frames(vp, ep_len, args.camera_height, args.camera_width)
                    hdf5_key = _video_key_to_hdf5_key(vk)
                    if len(frames) < ep_len:
                        ep_len = len(frames)
                    obs_dict[hdf5_key] = frames[:ep_len]
                if not ok:
                    continue
            else:
                obs_dict = _render_episode(env, states[:ep_len], model_xml, ep_meta_dict, camera_names)
                for k in list(obs_dict.keys()):
                    obs_dict[k] = obs_dict[k][:ep_len]

            states = states[:ep_len]
            actions = actions[:ep_len]
            if action_dim is not None and actions.shape[1] > action_dim:
                actions = actions[:, :action_dim]

            demo_grp = data_grp.create_group(f"demo_{written}")
            demo_grp.create_dataset("actions", data=actions.astype(np.float32))
            demo_grp.create_dataset("states", data=states)
            demo_grp.attrs["num_samples"] = ep_len
            demo_grp.attrs["model_file"] = model_xml
            demo_grp.attrs["ep_meta"] = json.dumps(ep_meta_dict, indent=4)

            obs_grp = demo_grp.create_group("obs")
            for cam_key, imgs in obs_dict.items():
                obs_grp.create_dataset(cam_key, data=imgs.astype(np.uint8), compression="gzip")

            total_samples += ep_len
            written += 1

        data_grp.attrs["total"] = total_samples
        data_grp.attrs["num_demos"] = written

    if env is not None:
        env.close()
    print(f"Done. Wrote {output_path} ({total_samples} timesteps, {written} episodes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboCasa LeRobot → HDF5")
    parser.add_argument("--lerobot_dir", type=str, required=True,
                        help="Path to RoboCasa LeRobot dataset (contains data/, videos/, extras/, meta/)")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--camera_names", type=str, nargs="+",
                        default=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"])
    parser.add_argument("--camera_height", type=int, default=128)
    parser.add_argument("--camera_width", type=int, default=128)
    parser.add_argument("--action_dim", type=int, default=7,
                        help="Truncate to N dims after reorder (7 = arm EE + gripper)")
    parser.add_argument("--from_render", action="store_true",
                        help="Render in simulation instead of extracting from video")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU for off-screen render")
    args = parser.parse_args()
    convert(args)
