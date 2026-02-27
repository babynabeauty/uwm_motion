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


def main():
    parser = argparse.ArgumentParser(
        description="Precompute RAFT flow and store SDXL-VAE flow latents into HDF5"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10",
        help="Path to dataset directory",
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
    if not os.path.isdir(args.sdxl_vae_path):
        raise FileNotFoundError(f"Local SDXL-VAE path not found: {args.sdxl_vae_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading RAFT model...")
    model, transforms = build_raft(device)
    print("Loading SDXL-VAE...")
    vae = build_sdxl_vae(device, args.sdxl_vae_path)

    files = glob.glob(os.path.join(args.dataset_dir, "**/*.hdf5"), recursive=True)
    files = [f for f in files if "motion" not in f]
    print(f"Found {len(files)} dataset files.")

    for file_path in files:
        print(f"Start processing: {file_path}")
        process_hdf5_file(file_path, args, model, transforms, vae, device)
        print(f"Finished: {file_path}")


if __name__ == "__main__":
    main()
