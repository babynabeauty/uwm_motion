# nohup uvicorn dp_server:app --host 0.0.0.0 --port 8003 > dp_server.log 2>&1 &
import functools
import os
import sys

import msgpack
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hydra.utils import instantiate

app = FastAPI()

# --- Configuration ---
CHECKPOINT_DIR = "/data/workspace/zhangshiqi/uwm_motion/bc_finetune/dp/real_scoop/real_scoop_baseline/0"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "models.pt")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, ".hydra", "config.yaml")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Client image key -> model obs_dict key
IMAGE_KEY_MAP = {
    "head": "head_rgb",
    "right_wrist": "right_wrist_rgb",
}

# --- Load config & model ---
print(f"Loading DP model from {CHECKPOINT_PATH} ...")
config = OmegaConf.load(CONFIG_PATH)

# Patch hydra-specific resolvers that cannot be resolved outside hydra
OmegaConf.update(config, "algo", "dp", force_add=True)
OmegaConf.update(config, "logdir", CHECKPOINT_DIR, force_add=True)
OmegaConf.resolve(config)

OBS_NUM_FRAMES = config.obs_num_frames  # 2
ACTION_LEN = config.model.action_len    # 16
ACTION_DIM = config.model.action_dim    # 7

# Image size the model asserts in obs_encoder (from shape_meta)
IMAGE_H = config.dataset.shape_meta.obs.head_rgb.shape[0]  # 128
IMAGE_W = config.dataset.shape_meta.obs.head_rgb.shape[1]  # 128

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

model = instantiate(config.model)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.noise_scheduler = DDIMScheduler.from_config(
    model.noise_scheduler.config, clip_sample=False
)
model.to(DEVICE)
model.eval()

action_normalizer = ckpt.get("action_normalizer")
if action_normalizer is not None:
    print(f"Action normalizer loaded: scale={action_normalizer.scale}, offset={action_normalizer.offset}")
else:
    print("WARNING: No action_normalizer in checkpoint -- raw action output")

print(
    f"Model loaded. obs_num_frames={OBS_NUM_FRAMES}, action_len={ACTION_LEN}, "
    f"action_dim={ACTION_DIM}, image_size=({IMAGE_H},{IMAGE_W})"
)

# --- Serialization helpers (identical to ACT server) ---

def pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.frombuffer(
            obj[b"data"], dtype=np.dtype(obj[b"dtype"])
        ).reshape(obj[b"shape"])
    return obj


unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
packb = functools.partial(msgpack.packb, default=pack_array)


def _resize_image(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize an HWC uint8 image to (h, w) if needed."""
    if image.shape[0] == h and image.shape[1] == w:
        return image
    return np.array(Image.fromarray(image).resize((w, h), Image.BILINEAR))


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_bytes(packb({"status": "ready", "model": "DP"}))

    # Per-connection observation history buffer: {model_key: [np.ndarray(H,W,C), ...]}
    obs_history: dict[str, list[np.ndarray]] = {v: [] for v in IMAGE_KEY_MAP.values()}

    while True:
        try:
            data = await websocket.receive_bytes()
            obs_dict = unpackb(data)
            image_dict = obs_dict.get("images", obs_dict.get(b"images", {}))

            # Update observation history with new frame
            for client_key, model_key in IMAGE_KEY_MAP.items():
                raw_img = image_dict.get(client_key)
                if raw_img is None:
                    raw_img = image_dict.get(client_key.encode())
                if raw_img is None:
                    raise KeyError(
                        f"Missing image '{client_key}' in payload. "
                        f"Expected keys: {list(IMAGE_KEY_MAP.keys())}"
                    )
                resized = _resize_image(raw_img, IMAGE_H, IMAGE_W)
                obs_history[model_key].append(resized)
                # Keep only the most recent frames
                if len(obs_history[model_key]) > OBS_NUM_FRAMES:
                    obs_history[model_key] = obs_history[model_key][-OBS_NUM_FRAMES:]

            # Build model input: (B=1, T=obs_num_frames, H, W, C) uint8
            model_input = {}
            for model_key in IMAGE_KEY_MAP.values():
                frames = list(obs_history[model_key])
                # Pad with first frame if not enough history yet
                while len(frames) < OBS_NUM_FRAMES:
                    frames.insert(0, frames[0])
                stacked = np.stack(frames, axis=0)  # (T, H, W, C)
                model_input[model_key] = (
                    torch.from_numpy(stacked).unsqueeze(0).to(DEVICE)  # (1, T, H, W, C)
                )

            # Inference -- model internally handles ToTensor, Resize, CenterCrop, ImageNet norm
            with torch.no_grad():
                actions = model.sample(model_input)  # (1, action_len, action_dim)

            actions_np = actions.squeeze(0).cpu().numpy().astype(np.float32)  # (16, 7)
            if action_normalizer is not None:
                actions_np = action_normalizer.reconstruct(actions_np)
            await websocket.send_bytes(packb({"action": actions_np}))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during inference: {e}")
            break
