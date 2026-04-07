import copy
import os

from experiments.runtime_env import bootstrap_non_root_runtime

bootstrap_non_root_runtime()

import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from typing import Optional
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import sys
from datasets.utils.loader import make_distributed_data_loader
from experiments.utils import set_seed, init_wandb, init_distributed, is_main_process
import ipdb
"""构建EMA模型"""
def unwrap_model(model):
    return getattr(model, "module", model)


def init_ema_model(model):
    ema_model = copy.deepcopy(unwrap_model(model)).eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay: float):
    src_model = unwrap_model(model)
    for ema_param, src_param in zip(ema_model.parameters(), src_model.parameters()):
        ema_param.data.mul_(decay).add_(src_param.data, alpha=1.0 - decay)
    for ema_buffer, src_buffer in zip(ema_model.buffers(), src_model.buffers()):
        ema_buffer.data.copy_(src_buffer.data)


def build_frozen_flow_vqvae(config, device):
    noise_cfg = config.model.noise_pred_net
    if not bool(getattr(noise_cfg, "use_quantized_of", False)):
        return None, None, None

    ckpt_path = getattr(noise_cfg, "quantized_of_vqvae_ckpt_path", None)
    if not ckpt_path:
        raise ValueError(
            "model.noise_pred_net.use_quantized_of=True but quantized_of_vqvae_ckpt_path is not set."
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {ckpt_path}")

    repo_path = getattr(noise_cfg, "quantized_of_vqvae_repo_path", None)
    if not repo_path:
        raise ValueError(
            "quantized_of_vqvae_repo_path is not set; cannot import FlowVQVAE."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # Ensure we import FlowVQVAE from the specified repo_path, not a cached module.
    import importlib

    for k in list(sys.modules.keys()):
        if k == "flow_vq" or k.startswith("flow_vq."):
            del sys.modules[k]

    FlowVQVAE = importlib.import_module("flow_vq.vqvae_flow").FlowVQVAE

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_args = ckpt["args"] if isinstance(ckpt, dict) and "args" in ckpt else {}

    def _infer_int_arg(name: str, default: int) -> int:
        v = ckpt_args.get(name, None)
        if v is None:
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    # Infer hyperparams from checkpoint if not provided in ckpt_args.
    # Works for both dict ckpt and pure state_dict ckpt.
    if "vq.embeddings.weight" in state_dict:
        inferred_codebook_size = int(state_dict["vq.embeddings.weight"].shape[0])
        inferred_embedding_dim = int(state_dict["vq.embeddings.weight"].shape[1])
    else:
        inferred_codebook_size = int(noise_cfg.quantized_of_vocab_size)
        inferred_embedding_dim = 64

    # hidden_dim from first conv out channels if possible.
    inferred_hidden_dim = 128
    for key in ("encoder.0.0.weight", "encoder.0.weight"):
        if key in state_dict:
            inferred_hidden_dim = int(state_dict[key].shape[0])
            break

    # downsample_factor for laq_flow version: encoder has blocks [0..d-1] then final conv at index d.
    inferred_downsample_factor = 4
    enc_idx = []
    for k in state_dict.keys():
        if k.startswith("encoder."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                enc_idx.append(int(parts[1]))
    if enc_idx:
        inferred_downsample_factor = max(enc_idx)  # last conv index == downsample_factor

    hidden_dim = _infer_int_arg("hidden_dim", inferred_hidden_dim)
    embedding_dim = _infer_int_arg("embedding_dim", inferred_embedding_dim)
    codebook_size = _infer_int_arg("codebook_size", inferred_codebook_size)
    downsample_factor = _infer_int_arg("downsample_factor", inferred_downsample_factor)
    
    if int(noise_cfg.quantized_of_vocab_size) != codebook_size:
        if is_main_process():
            print(
                f"Override quantized_of_vocab_size from {noise_cfg.quantized_of_vocab_size} to "
                f"{codebook_size} (read from VQ-VAE checkpoint)."
            )
        config.model.noise_pred_net.quantized_of_vocab_size = codebook_size

    vqvae = FlowVQVAE(
        in_channels=2,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        codebook_size=codebook_size,
        downsample_factor=downsample_factor,
    ).to(device)
    vqvae.load_state_dict(state_dict, strict=True)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy = torch.zeros(1, 2, 128, 128, device=device)
        if hasattr(vqvae, "get_codebook_indices"):
            dummy_indices = vqvae.get_codebook_indices(dummy)  # [1, h', w']
        else:
            dummy_indices = vqvae.encode(dummy)  # [1, h', w']
        num_spatial_tokens = dummy_indices.shape[1] * dummy_indices.shape[2]
        background_id = int(dummy_indices.flatten()[0].item())
    if is_main_process():
        print(f"FlowVQVAE spatial resolution: {dummy_indices.shape[1]}x{dummy_indices.shape[2]}, "
              f"num_spatial_tokens (M) = {num_spatial_tokens}")
        print(f"FlowVQVAE background codebook ID (zero-flow): {background_id}")

    return vqvae, num_spatial_tokens, background_id


@torch.no_grad()
def extract_vq_indices_from_flow(
    flow_btchw: torch.Tensor,
    flow_vqvae,
) -> torch.Tensor:
    """
    Convert optical flow [B, T, 2, H, W] to spatial token ids [B, T, M]
    using FlowVQVAE.get_codebook_indices, where M = h' * w'.
    """
    b, t, c, h, w = flow_btchw.shape
    flat_flow = flow_btchw.reshape(b * t, c, h, w).contiguous()
    if hasattr(flow_vqvae, "get_codebook_indices"):
        indices = flow_vqvae.get_codebook_indices(flat_flow)  # [B*T, h', w']
    else:
        indices = flow_vqvae.encode(flat_flow)  # [B*T, h', w']
    m = indices.shape[1] * indices.shape[2]
    return indices.reshape(b, t, m).long()


def process_batch(batch, obs_horizon, action_horizon, use_quantized_of, device,
                   flow_vqvae: Optional[torch.nn.Module] = None,
                   background_id: Optional[int] = None):
    # Take the first `obs_horizon` observations
    obs = {k: v[:, :obs_horizon].to(device) for k, v in batch["obs"].items()}

    # Take the last `action_horizon` actions
    action = batch["action"][:, -action_horizon:].to(device)
    
    gt_motion = None
    if use_quantized_of:
        if action_horizon == 8:
            flow = batch["optical_flow_8"][:,0:1].to(device)
        elif action_horizon == 16:
            flow = batch["optical_flow_16"][:,0:1].to(device)
        else:
            raise ValueError(f"Unsupported action length: {action_horizon}")
        gt_motion = extract_vq_indices_from_flow(flow, flow_vqvae)
        #FIXME:过滤背景方法2
        if background_id is not None:
            gt_motion[gt_motion == background_id] = -1
    elif "optical_flow_raft_latent" in batch:
        gt_motion = batch["optical_flow_raft_latent"][:, -action_horizon:].to(device)
    # Add language tokens to observations
    if "input_ids" in batch and "attention_mask" in batch:
        obs["input_ids"] = batch["input_ids"].to(device)
        obs["attention_mask"] = batch["attention_mask"].to(device)
    return obs, action, gt_motion


def eval_one_epoch(config, data_loader, device, model, action_normalizer=None,
                   flow_vqvae: Optional[torch.nn.Module] = None,
                   background_id: Optional[int] = None):
    model.eval()
    model = unwrap_model(model)

    # Unnormalize actions
    if action_normalizer is not None:
        action_scale = torch.tensor(action_normalizer.scale[None], device=device)
        action_offset = torch.tensor(action_normalizer.offset[None], device=device)
        unnormalize = lambda a: a * action_scale + action_offset
    else:
        unnormalize = lambda a: a

    stats = {"loss": 0, "action_mse": 0}
    for batch in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
        # ------------ Preprocess data ------------ #
        obs, action, gt_motion = process_batch(
            batch, config.model.obs_encoder.num_frames, config.model.action_len,
            config.model.noise_pred_net.use_quantized_of, device, flow_vqvae,
            background_id=background_id,
        )
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            # ------------ Validation loss ------------ #
            loss = model(obs, action,gt_motion)
            dist.all_reduce(loss["loss"], op=dist.ReduceOp.AVG)
            #FIXME:这里的loss已经是一个dict  记得修改
            stats["loss"] += loss["loss"].item()

            # ------------ BC Inference ------------ #
            # Sample actions
            action_hat = model.sample(obs)
            # Unnormalize action and action_hat
            action_hat = unnormalize(action_hat)
            action = unnormalize(action)
            
            # print( "action_hat[0] **", action_hat[0])
            # print( "action[0] **", action[0])

            # Compute MSE loss
            mse = F.mse_loss(action_hat, action)

            # Collect results across all processes
            dist.all_reduce(mse, op=dist.ReduceOp.AVG)
            stats["action_mse"] += mse

    # Average over all batches
    stats = {k: v / len(data_loader) for k, v in stats.items()}
    return stats


def train_one_step(config, model, optimizer, scheduler, scaler, batch, device,
                   flow_vqvae: Optional[torch.nn.Module] = None,
                   background_id: Optional[int] = None):
    model.train()

    # --- Preprocess data ---
    obs, action, gt_motion = process_batch(
        batch, config.model.obs_encoder.num_frames, config.model.action_len,
        config.model.noise_pred_net.use_quantized_of, device, flow_vqvae,
        background_id=background_id,
    )

    # --- DP Training ---
    # Action prediction loss
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=config.use_amp
    ):
        loss = model(obs, action,gt_motion)
        info = {"loss": loss["loss"], "action_loss": loss["action_loss"],"motion_loss": loss["motion_loss"]}

    # Step optimizer
    optimizer.zero_grad()
    scaler.scale(loss["loss"]).backward()
    if config.clip_grad_norm:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    # Step scheduler
    scheduler.step()
    info["lr"] = scheduler.get_last_lr()[0]
    return loss, info


def maybe_resume_checkpoint(
    config, model, optimizer, scheduler, scaler, ckpt_name="models.pt"
):
    """Resume from a checkpoint if config.resume is True."""
    step = 0
    if config.resume:
        ckpt_path = os.path.join(config.logdir, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"] + 1
        print(f"Resumed training from step {step}")
    return step


def maybe_evaluate(
    config,
    step,
    model,
    loader,
    device,
    action_normalizer=None,
    flow_vqvae: Optional[torch.nn.Module] = None,
    eval_model=None,
    background_id: Optional[int] = None,
):
    """Evaluate if it's the correct step."""
    if step > 1:
        if step % config.eval_every == 0 or step == (config.num_steps - 1):
            target_model = eval_model if eval_model is not None else model
            stats = eval_one_epoch(config, loader, device, target_model, action_normalizer,
                                   flow_vqvae, background_id=background_id)
            if is_main_process():
                wandb.log({"global_step": step, **{f"eval/{k}": v for k, v in stats.items()}})
                print(f"Step {step} action mse: {stats['action_mse']:.4f}")


def maybe_save_checkpoint(
    config,
    step,
    model,
    optimizer,
    scheduler,
    scaler,
    action_normalizer=None,
    lowdim_normalizer=None,
    save_model=None,
    rollout_avg_sr=None,
    ckpt_name="models.pt",
):
    """Save latest checkpoint and optionally update rollout-best checkpoint."""
    if is_main_process() and (
        step % config.save_every == 0 or step == (config.num_steps - 1)
    ):
        target_model = unwrap_model(save_model if save_model is not None else model)
        ckpt = {
            "model": target_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "action_normalizer": action_normalizer,
            "lowdim_normalizer": lowdim_normalizer,
            "step": step,
        }
        # Always keep latest checkpoint for resume.
        latest_path = os.path.join(config.logdir, ckpt_name)
        torch.save(ckpt, latest_path)
        print(f"Saved latest checkpoint at step {step} to {latest_path}")

        # Keep only one best checkpoint based on rollout average success rate.
        if rollout_avg_sr is not None:
            best_meta_path = os.path.join(config.logdir, "best_ckpt_meta.pt")
            prev_best_sr = float("-inf")
            if os.path.exists(best_meta_path):
                try:
                    prev_best_sr = float(torch.load(best_meta_path, map_location="cpu").get("best_rollout_avg_sr", float("-inf")))
                except Exception:
                    prev_best_sr = float("-inf")

            if float(rollout_avg_sr) >= prev_best_sr:
                best_path = os.path.join(config.logdir, "models_best.pt")
                torch.save(ckpt, best_path)
                torch.save(
                    {"best_rollout_avg_sr": float(rollout_avg_sr), "best_step": int(step)},
                    best_meta_path,
                )
                print(
                    f"Updated best checkpoint: avg_success_rate={float(rollout_avg_sr):.4f}, "
                    f"step={step}, path={best_path}"
                )


def train(rank, world_size, config):
    # Set global seed
    set_seed(config.seed * world_size + rank)

    # Initialize distributed training
    init_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize WANDB
    if is_main_process():
        init_wandb(config, job_type="train")

    # Create dataset
    train_set, val_set = instantiate(config.dataset)
    train_loader, val_loader = make_distributed_data_loader(
        train_set, val_set, config.batch_size, rank, world_size
    )

    # Create model
    flow_vqvae, num_spatial_tokens, background_id = build_frozen_flow_vqvae(config, device)

    model = instantiate(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Load pretrained model
    if config.pretrain_checkpoint_path:
        ckpt = torch.load(config.pretrain_checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(
            f"Loaded pretraining checkpoint {config.pretrain_checkpoint_path}, step: {ckpt['step']}"
        )

        # Replace dataset normalizers to make sure data is normalized correctly
        if ckpt["action_normalizer"] is not None:
            train_set.action_normalizer = ckpt["action_normalizer"]
            val_set.action_normalizer = ckpt["action_normalizer"]
        if ckpt["lowdim_normalizer"] is not None:
            train_set.lowdim_normalizer = ckpt["lowdim_normalizer"]
            val_set.lowdim_normalizer = ckpt["lowdim_normalizer"]

    # Resume from checkpoint
    step = maybe_resume_checkpoint(config, model, optimizer, scheduler, scaler)
    ema_model = init_ema_model(model)
    epoch = step // len(train_loader)

    # Wrap model with DDP
    model = DistributedDataParallel(model, device_ids=[rank], static_graph=True)

    # Training loop
    pbar = tqdm(
        total=config.num_steps,
        initial=step,
        desc="Training",
        disable=not is_main_process(),
    )
    while step < config.num_steps:
        # Set epoch for distributed sampler to shuffle indices
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            # --- Training step ---
            loss, info = train_one_step(
                config, model, optimizer, scheduler, scaler, batch, device,
                flow_vqvae, background_id=background_id,
            )
            update_ema_model(ema_model, model, config.ema_decay)

            # --- Logging ---
            if is_main_process():
                pbar.set_description(f"step: {step}, loss: {loss['loss']:.4f},action_loss: {loss['action_loss']:.4f},motion_loss: {loss['motion_loss']:.4f}")
                wandb.log({"global_step": step, **{f"train/{k}": v for k, v in info.items()}})

            # --- Evaluate if needed ---
            maybe_evaluate(
                config, step, model, val_loader, device, train_set.action_normalizer,
                flow_vqvae, eval_model=ema_model, background_id=background_id,
            )

            # --- Save checkpoint if needed ---
            maybe_save_checkpoint(
                config,
                step,
                model,
                optimizer,
                scheduler,
                scaler,
                train_set.action_normalizer,
                train_set.lowdim_normalizer,
                save_model=ema_model,
            )

            step += 1
            pbar.update(1)
            if step >= config.num_steps:
                break

        epoch += 1


@hydra.main(version_base=None, config_path="../../configs", config_name="train_dp.yaml")
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    # Spawn processes
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
    # train(0, 1, config)



if __name__ == "__main__":
    main()
