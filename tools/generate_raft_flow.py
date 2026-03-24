import argparse
import glob
import math
import os

import cv2
import h5py
import numcodecs
import numpy as np
import torch
import zarr
from tqdm import tqdm

from diffusers import AutoencoderKL
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def build_raft(device: torch.device):
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    transforms = weights.transforms()
    return model, transforms


def build_sdxl_vae(device: torch.device, vae_path: str):
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        local_files_only=True,
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def _resize_frames(images: np.ndarray, target_size):
    t, h, w, _ = images.shape
    if (h, w) == target_size:
        return images

    out = []
    for i in range(t):
        out.append(
            cv2.resize(
                images[i],
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        )
    return np.stack(out, axis=0)


def compute_raft_optical_flow(
    images: np.ndarray,
    model,
    transforms,
    device: torch.device,
    target_size=(224, 224),
    batch_size=8,
):
    """
    Compute dense optical flow using RAFT.

    Args:
        images: [T, H, W, 3] uint8 RGB frames.
        model: RAFT model.
        transforms: torchvision RAFT preprocessing transform.
        device: torch device.
        target_size: (H, W), resized spatial size before inference.
        batch_size: pair batch size for inference.

    Returns:
        flows: [T, H, W, 2] float32.
    """
    t = images.shape[0]
    resized = _resize_frames(images, target_size)

    if t == 1:
        h, w = target_size
        return np.zeros((1, h, w, 2), dtype=np.float32)

    flow_chunks = []
    with torch.no_grad():
        for start in range(0, t - 1, batch_size):
            end = min(start + batch_size, t - 1)
            n = end - start

            # Adjacent pairs: (frame_i, frame_{i+1})
            img1 = resized[start:end]
            img2 = resized[start + 1 : end + 1]

            img1 = torch.from_numpy(img1).permute(0, 3, 1, 2).float() / 255.0
            img2 = torch.from_numpy(img2).permute(0, 3, 1, 2).float() / 255.0
            img1, img2 = transforms(img1, img2)

            flow_predictions = model(img1.to(device), img2.to(device))
            flow = flow_predictions[-1]  # [N, 2, H, W]
            flow = flow.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)

            if flow.shape[0] != n:
                raise RuntimeError("RAFT output batch size mismatch.")
            flow_chunks.append(flow)

    flows = np.concatenate(flow_chunks, axis=0)  # [T-1, H, W, 2]
    flows = np.concatenate([flows, np.zeros_like(flows[:1])], axis=0)  # [T, H, W, 2]
    return flows


def compute_raft_optical_flow_t_to_tplus8(
    images: np.ndarray,
    model,
    transforms,
    device: torch.device,
    frame_skip: int = 8,
    target_size=(224, 224),
    batch_size=8,
):
    """
    Compute dense optical flow between frame t and frame t+frame_skip.

    Args:
        images: [T, H, W, 3] uint8 RGB frames.
        model: RAFT model.
        transforms: torchvision RAFT preprocessing transform.
        device: torch device.
        frame_skip: temporal stride (default 8 for t and t+8).
        target_size: (H, W), resized spatial size before inference.
        batch_size: pair batch size for inference.

    Returns:
        flows: [T, H, W, 2] float32.
    """
    t = images.shape[0]
    resized = _resize_frames(images, target_size)

    if t == 1:
        h, w = target_size
        return np.zeros((1, h, w, 2), dtype=np.float32)

    flow_chunks = []
    with torch.no_grad():
        for start in range(0, t, batch_size):
            end = min(start + batch_size, t)
            n = end - start

            pairs = []
            for i in range(start, end):
                j = min(i + frame_skip, t - 1)
                if i == j:
                    pairs.append((i, i))
                else:
                    pairs.append((i, j))

            img1 = np.stack([resized[p[0]] for p in pairs], axis=0)
            img2 = np.stack([resized[p[1]] for p in pairs], axis=0)

            img1 = torch.from_numpy(img1).permute(0, 3, 1, 2).float() / 255.0
            img2 = torch.from_numpy(img2).permute(0, 3, 1, 2).float() / 255.0
            img1, img2 = transforms(img1, img2)

            flow_predictions = model(img1.to(device), img2.to(device))
            flow = flow_predictions[-1]
            flow = flow.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)

            for idx, (i, j) in enumerate(pairs):
                if i == j:
                    flow[idx] = 0.0

            flow_chunks.append(flow)

    flows = np.concatenate(flow_chunks, axis=0)
    return flows


def flow_to_rgb_uint8(flow: np.ndarray, clip_flow: float = 32.0) -> np.ndarray:
    """
    Convert flow [H, W, 2] to RGB uint8 using HSV color wheel.
    """
    fx = np.clip(flow[..., 0], -clip_flow, clip_flow)
    fy = np.clip(flow[..., 1], -clip_flow, clip_flow)
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # [0, 180) for OpenCV HSV
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip((mag / clip_flow) * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def encode_flow_latents(
    flows: np.ndarray,
    vae,
    device: torch.device,
    latent_batch_size: int = 16,
) -> np.ndarray:
    """
    Encode flow maps into SDXL-VAE latents.

    Args:
        flows: [T, H, W, 2] float32.
        vae: SDXL VAE.
        device: torch device.
        latent_batch_size: frame batch size for VAE encoding.

    Returns:
        latents: [T, 4, H/8, W/8] float32.
    """
    rgb_frames = np.stack([flow_to_rgb_uint8(flow) for flow in flows], axis=0)  # [T, H, W, 3]
    imgs = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float() / 255.0
    imgs = imgs * 2.0 - 1.0  # [0,1] -> [-1,1]

    latents = []
    scaling_factor = float(vae.config.scaling_factor)
    with torch.no_grad():
        for start in range(0, imgs.shape[0], latent_batch_size):
            end = min(start + latent_batch_size, imgs.shape[0])
            chunk = imgs[start:end].to(device)
            # Use latent mean to make supervision deterministic.
            posterior = vae.encode(chunk).latent_dist
            latent = posterior.mean * scaling_factor
            latents.append(latent.detach().cpu())
    latents = torch.cat(latents, dim=0).numpy().astype(np.float32)
    return latents


def process_hdf5_file(file_path, args, model, transforms, vae, device):
    try:
        with h5py.File(file_path, "r+") as f:
            demos = list(f["data"].keys())

            for demo_key in tqdm(demos, desc=f"Processing {os.path.basename(file_path)}", leave=False):
                demo_grp = f["data"][demo_key]
                if args.flow_key in demo_grp and not args.overwrite:
                    continue

                if "obs" not in demo_grp or "agentview_rgb" not in demo_grp["obs"]:
                    print(f"Warning: No obs/agentview_rgb found in {file_path}:{demo_key}")
                    continue

                rgb_data = demo_grp["obs"]["agentview_rgb"][:]  # [T, H, W, 3]
                flows = compute_raft_optical_flow(
                    rgb_data,
                    model=model,
                    transforms=transforms,
                    device=device,
                    target_size=(args.img_size, args.img_size),
                    batch_size=args.batch_size,
                )
                flow_latents = encode_flow_latents(
                    flows,
                    vae=vae,
                    device=device,
                    latent_batch_size=args.latent_batch_size,
                )

                if args.flow_key in demo_grp:
                    del demo_grp[args.flow_key]

                demo_grp.create_dataset(
                    args.flow_key,
                    data=flow_latents,
                    compression="gzip",
                    compression_opts=4,
                )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6):
    itemsize = np.dtype(dtype).itemsize
    rshape = list(shape[::-1])
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk = itemsize * np.prod(rshape[:i])
        next_chunk = itemsize * np.prod(rshape[: i + 1])
        if this_chunk <= target_chunk_bytes < next_chunk:
            split_idx = i
            break
    rchunks = rshape[:split_idx]
    item_chunk = itemsize * np.prod(rchunks)
    next_len = min(rshape[split_idx], math.ceil(target_chunk_bytes / item_chunk))
    rchunks.append(next_len)
    rchunks.extend([1] * (len(shape) - len(rchunks)))
    return tuple(rchunks[::-1])


def append_optical_flow_n_to_zarr(args, model, transforms, vae, device):
    """Append optical_flow_N and/or optical_flow_N_raft_latent to an existing .zarr buffer.
    N = args.frame_skip. Saves only when save_optical_flow / save_optical_flow_raft_latent is True.
    """
    save_flow = args.save_optical_flow
    save_latent = args.save_optical_flow_raft_latent
    
    if not args.save_optical_flow and not args.save_optical_flow_raft_latent:
        raise ValueError("At least one of --save_optical_flow or --save_optical_flow_raft_latent must be True")

    n = args.frame_skip
    flow_key = f"optical_flow_{n}"
    flow_latent_key = f"optical_flow_{n}_raft_latent"

    store = zarr.DirectoryStore(args.zarr_path)
    root = zarr.group(store=store)
    data = root["data"]
    meta = root["meta"]

    if "episode_ends" not in meta:
        raise ValueError(f"No episode_ends in {args.zarr_path}")
    episode_ends = meta["episode_ends"][:]
    capacity = episode_ends[-1] if len(episode_ends) > 0 else 0

    image_key = args.image_key
    if image_key not in data:
        raise ValueError(f"Image key {image_key} not in zarr. Available: {list(data.keys())}")

    img_arr = data[image_key]
    flow_spatial = args.img_size
    flow_shape = (2, flow_spatial, flow_spatial)  # (C, H, W) -> stored as (N, 2, H, W)
    flow_latent_h, flow_latent_w = flow_spatial // 8, flow_spatial // 8
    latent_shape = (4, flow_latent_h, flow_latent_w)

    if flow_key in data and not args.overwrite:
        print(f"{flow_key} already exists in {args.zarr_path}, skipping (use --overwrite to replace)")
        return
    if save_latent and flow_latent_key in data and not args.overwrite:
        print(f"{flow_latent_key} already exists in {args.zarr_path}, skipping (use --overwrite to replace)")
        return

    chunks_flow = get_optimal_chunks((capacity,) + flow_shape, np.float32)
    compressor = numcodecs.Blosc(cname="lz4", clevel=0, shuffle=numcodecs.Blosc.NOSHUFFLE)
    flow_arr = None
    flow_latent_arr = None

    if save_flow:
        if flow_key in data:
            del data[flow_key]
        chunks_flow = get_optimal_chunks((capacity,) + flow_shape, np.float32)
        flow_arr = data.zeros(
            flow_key,
            shape=(capacity,) + flow_shape,
            chunks=chunks_flow,
            dtype=np.float32,
            compressor=compressor,
        )
    if save_latent:
        if flow_latent_key in data:
            del data[flow_latent_key]
        chunks_latent = get_optimal_chunks((capacity,) + latent_shape, np.float32)
        flow_latent_arr = data.zeros(
            flow_latent_key,
            shape=(capacity,) + latent_shape,
            chunks=chunks_latent,
            dtype=np.float32,
            compressor=compressor,
        )

    prev_end = 0
    for ep_idx, end in enumerate(tqdm(episode_ends, desc="Episodes")):
        ep_len = end - prev_end
        frames = img_arr[prev_end:end]
        frames = np.asarray(frames)

        flows = compute_raft_optical_flow_t_to_tplus8(
            frames,
            model=model,
            transforms=transforms,
            device=device,
            frame_skip=n,
            target_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
        )

        if save_flow:
            flows_nchw = np.transpose(flows, (0, 3, 1, 2))  # (T, H, W, 2) -> (T, 2, H, W)
            flow_arr[prev_end:end] = flows_nchw
        if save_latent:
            latents = encode_flow_latents(
                flows,
                vae=vae,
                device=device,
                latent_batch_size=args.latent_batch_size,
            )
            flow_latent_arr[prev_end:end] = latents

        prev_end = end

    saved = [k for k, v in [(flow_key, save_flow), (flow_latent_key, save_latent)] if v]
    print(f"Appended {', '.join(saved)} to {args.zarr_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute RAFT flow: HDF5 mode or append optical_flow_8 to existing zarr"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hdf5", "zarr"],
        default="hdf5",
        help="hdf5: process HDF5 files; zarr: append optical_flow_N to existing .zarr",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Path to existing .zarr (required when --mode=zarr)",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="obs.head_rgb",
        help="Zarr key for RGB frames (e.g. obs.head_rgb)",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=8,
        help="Compute flow from frame t to t+N; output keys optical_flow_N and optical_flow_N_raft_latent",
    )
    parser.add_argument(
        "--save_optical_flow",
        action="store_true",
        default=True,
        help="Save raw optical flow to optical_flow_N (default: True)",
    )
    parser.add_argument(
        "--no_save_optical_flow",
        action="store_false",
        dest="save_optical_flow",
        help="Do not save raw optical flow",
    )
    parser.add_argument(
        "--save_optical_flow_raft_latent",
        action="store_true",
        default=True,
        help="Save VAE-encoded flow latent to optical_flow_N_raft_latent (default: True)",
    )
    parser.add_argument(
        "--no_save_optical_flow_raft_latent",
        action="store_false",
        dest="save_optical_flow_raft_latent",
        help="Do not save optical flow raft latent",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10",
        help="Path to dataset directory (HDF5 mode)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Square image size for RAFT inference (must be divisible by 8)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of adjacent frame pairs per RAFT forward",
    )
    parser.add_argument(
        "--latent_batch_size",
        type=int,
        default=16,
        help="Number of frames per SDXL-VAE encoding batch",
    )
    parser.add_argument(
        "--flow_key",
        type=str,
        default="optical_flow_raft_latent",
        help="Output latent dataset key under each demo group",
    )
    parser.add_argument(
        "--sdxl_vae_path",
        type=str,
        default="/data/shared_workspace/zhangshiqi/hf/models--stabilityai--sdxl-vae",
        help="Local path to SDXL-VAE weights (offline only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flow dataset",
    )
    args = parser.parse_args()

    if args.img_size % 8 != 0:
        raise ValueError("--img_size must be divisible by 8 for RAFT.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading RAFT model...")
    model, transforms = build_raft(device)

    need_vae = args.mode == "hdf5" or (
        args.mode == "zarr" and args.save_optical_flow_raft_latent
    )
    if need_vae:
        if not os.path.isdir(args.sdxl_vae_path):
            raise FileNotFoundError(f"Local SDXL-VAE path not found: {args.sdxl_vae_path}")
        print("Loading SDXL-VAE...")
        vae = build_sdxl_vae(device, args.sdxl_vae_path)
    else:
        vae = None

    if args.mode == "zarr":
        if not args.zarr_path or not os.path.isdir(args.zarr_path):
            raise ValueError("--zarr_path must point to an existing .zarr directory")
        if args.frame_skip < 1:
            raise ValueError("--frame_skip must be >= 1")
        append_optical_flow_n_to_zarr(args, model, transforms, vae, device)
        return

    files = glob.glob(os.path.join(args.dataset_dir, "**/*.hdf5"), recursive=True)
    files = [f for f in files if "motion" not in f]
    print(f"Found {len(files)} dataset files.")

    for file_path in files:
        print(f"Start processing: {file_path}")
        process_hdf5_file(file_path, args, model, transforms, vae, device)
        print(f"Finished: {file_path}")


if __name__ == "__main__":
    main()
