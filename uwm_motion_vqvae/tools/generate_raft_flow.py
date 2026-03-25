#!/usr/bin/env python3
"""
Compute RAFT optical flow and store both raw flow and SDXL-VAE latents into HDF5.
Output keys: optical_flow [T,H,W,2] float32, optical_flow_raft_latent [T,4,H/8,W/8] float32.
"""
"""
# 处理目录下所有 HDF5
python tools/generate_raft_flow.py /data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_0
# 处理单个文件
python tools/generate_raft_flow.py /data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5
# 覆盖已有键、并保存可视化
python tools/generate_raft_flow.py /path/to/dir --overwrite --save_vis
"""

import argparse
import glob
import os

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from diffusers import AutoencoderKL
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

OPTICAL_FLOW_KEY = "optical_flow"
OPTICAL_FLOW_RAFT_LATENT_KEY = "optical_flow_raft_latent"


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
    Returns flows: [T, H, W, 2] float32.
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


def flow_to_rgb_uint8(flow: np.ndarray, clip_flow: float = 32.0) -> np.ndarray:
    """Convert flow [H, W, 2] to RGB uint8 using HSV color wheel."""
    fx = np.clip(flow[..., 0], -clip_flow, clip_flow)
    fy = np.clip(flow[..., 1], -clip_flow, clip_flow)
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
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
    """Encode flow maps [T,H,W,2] into SDXL-VAE latents [T,4,H/8,W/8] float32."""
    rgb_frames = np.stack([flow_to_rgb_uint8(flow) for flow in flows], axis=0)
    imgs = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float() / 255.0
    imgs = imgs * 2.0 - 1.0  # [0,1] -> [-1,1]

    latents = []
    scaling_factor = float(vae.config.scaling_factor)
    with torch.no_grad():
        for start in range(0, imgs.shape[0], latent_batch_size):
            end = min(start + latent_batch_size, imgs.shape[0])
            chunk = imgs[start:end].to(device)
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

                need_flow = OPTICAL_FLOW_KEY not in demo_grp or args.overwrite
                need_latent = OPTICAL_FLOW_RAFT_LATENT_KEY not in demo_grp or args.overwrite
                if not need_flow and not need_latent:
                    continue

                if "obs" not in demo_grp or "agentview_rgb" not in demo_grp["obs"]:
                    print(f"Warning: No obs/agentview_rgb in {file_path}:{demo_key}")
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

                # Save optical_flow [T, H, W, 2] float32
                if need_flow:
                    if OPTICAL_FLOW_KEY in demo_grp:
                        del demo_grp[OPTICAL_FLOW_KEY]
                    demo_grp.create_dataset(
                        OPTICAL_FLOW_KEY,
                        data=flows,
                        compression="gzip",
                        compression_opts=4,
                    )

                # Save optical_flow_raft_latent [T, 4, H/8, W/8] float32
                if need_latent:
                    flow_latents = encode_flow_latents(
                        flows,
                        vae=vae,
                        device=device,
                        latent_batch_size=args.latent_batch_size,
                    )
                    if OPTICAL_FLOW_RAFT_LATENT_KEY in demo_grp:
                        del demo_grp[OPTICAL_FLOW_RAFT_LATENT_KEY]
                    demo_grp.create_dataset(
                        OPTICAL_FLOW_RAFT_LATENT_KEY,
                        data=flow_latents,
                        compression="gzip",
                        compression_opts=4,
                    )

                # Optional: save visualization
                if args.save_vis:
                    vis_key = f"{OPTICAL_FLOW_KEY}_vis"
                    if vis_key in demo_grp:
                        del demo_grp[vis_key]
                    flow_vis = np.stack(
                        [flow_to_rgb_uint8(flow, clip_flow=args.clip_flow) for flow in flows],
                        axis=0,
                    )
                    demo_grp.create_dataset(
                        vis_key,
                        data=flow_vis,
                        compression="gzip",
                        compression_opts=4,
                    )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RAFT flow, store optical_flow and optical_flow_raft_latent into HDF5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="/data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10",
        help="Path to a single .hdf5 file or a directory containing .hdf5 files",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Square image size for RAFT. Must be >= 128 and divisible by 8.",
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
        "--sdxl_vae_path",
        type=str,
        default="/data/shared_workspace/zhangshiqi/hf/models--stabilityai--sdxl-vae",
        help="Local path to SDXL-VAE weights (offline only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing optical_flow and optical_flow_raft_latent",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Also save optical_flow_vis (RGB visualization)",
    )
    parser.add_argument(
        "--clip_flow",
        type=float,
        default=32.0,
        help="Clip flow magnitude for visualization (only used when --save_vis)",
    )
    args = parser.parse_args()

    if args.img_size < 128:
        raise ValueError(
            f"--img_size must be >= 128 for RAFT (got {args.img_size}). "
            "RAFT requires feature maps of at least 16x16."
        )
    if args.img_size % 8 != 0:
        raise ValueError(f"--img_size must be divisible by 8 for VAE (got {args.img_size}).")
    if not os.path.isdir(args.sdxl_vae_path):
        raise FileNotFoundError(f"SDXL-VAE path not found: {args.sdxl_vae_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading RAFT model...")
    model, transforms = build_raft(device)
    print("Loading SDXL-VAE...")
    vae = build_sdxl_vae(device, args.sdxl_vae_path)
    print(f"Will compute flow at {args.img_size}x{args.img_size}")

    path = os.path.abspath(args.path)
    if os.path.isfile(path):
        if not path.lower().endswith(".hdf5"):
            raise ValueError(f"Input must be .hdf5 or directory (got file: {path})")
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**/*.hdf5"), recursive=True)
        files = [f for f in files if "motion" not in f]
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")
    print(f"Found {len(files)} dataset file(s).")

    for file_path in files:
        print(f"Start: {file_path}")
        process_hdf5_file(file_path, args, model, transforms, vae, device)
        print(f"Done: {file_path}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"  {OPTICAL_FLOW_KEY}: [T, {args.img_size}, {args.img_size}, 2] float32")
    print(f"  {OPTICAL_FLOW_RAFT_LATENT_KEY}: [T, 4, {args.img_size//8}, {args.img_size//8}] float32")
    if args.save_vis:
        print(f"  optical_flow_vis: [T, {args.img_size}, {args.img_size}, 3] uint8")
    print("=" * 60)


if __name__ == "__main__":
    main()
