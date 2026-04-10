import os

from experiments.runtime_env import bootstrap_non_root_runtime

bootstrap_non_root_runtime()

import hydra
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from transformers import CLIPTokenizer
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange, tqdm
import ipdb
from experiments.utils import find_free_port
from datasets.utils.loader import make_distributed_data_loader
from datasets.utils.file_utils import glob_all
from environments.robomimic import make_robomimic_env
from experiments.dp.train import (
    train_one_step,
    maybe_resume_checkpoint,
    maybe_evaluate,
    maybe_save_checkpoint,
    build_frozen_flow_vqvae,
    init_ema_model,
    update_ema_model,
)
from experiments.utils import (
    set_seed,
    init_wandb,
    init_distributed,
    is_main_process,
    get_rollout_instruction,
)

_CLIP_TOKENIZER = None

def _get_tokenizer():
    global _CLIP_TOKENIZER
    if _CLIP_TOKENIZER is None:
        _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(
            '/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32'
        )
    return _CLIP_TOKENIZER



def collect_rollout(config, model, device, rank=0, world_size=1):
    # import ipdb;ipdb.set_trace()
    model.eval()
    model = getattr(model, "module", model)

    all_hdf5_paths = glob_all(config.dataset.hdf5_path_globs)
    my_paths = all_hdf5_paths[rank::world_size]

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    render_gpu_id = int(cuda_devices[rank % len(cuda_devices)])

    tokenizer = _get_tokenizer()
    MAX_TEXT_LEN = 25

    all_results = {}
    last_video = None

    print(f"[Rank {rank}] Starting rollout: {len(my_paths)} tasks assigned", flush=True)

    for path in my_paths:
        task_name = os.path.basename(path).replace(".hdf5", "")
        instruction = get_rollout_instruction(config.dataset.name, path)

        tokens = tokenizer(
            instruction, padding='max_length', max_length=MAX_TEXT_LEN,
            truncation=True, return_tensors='pt'
        ).to(device)

        print(f"[Rank {rank}] Collecting rollouts for task: {task_name}", flush=True)
        record_video = (rank == 0)
        env = make_robomimic_env(
            dataset_name=config.dataset.name,
            dataset_path=path,
            shape_meta=config.dataset.shape_meta,
            obs_horizon=model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=record_video,
            render_gpu_id=render_gpu_id,
        )

        successes = []
        for e in trange(config.num_rollouts, desc=f"[Rank {rank}] Testing {task_name}"):
            env.seed(e)
            obs = env.reset()
            done = False
            while not done:
                obs_tensor = {k: torch.tensor(v, device=device)[None] for k, v in obs.items()}
                obs_tensor["input_ids"] = tokens["input_ids"]
                obs_tensor["attention_mask"] = tokens["attention_mask"]

                with torch.no_grad():
                    action = model.sample(obs_tensor)[0].cpu().numpy()

                obs, reward, done, info = env.step(action)
            successes.append(info["success"])

        task_sr = sum(successes) / len(successes)
        all_results[f"rollout/success_rate_{task_name}"] = task_sr
        print(f"[Rank {rank}] Task success: {task_sr:.4f}", flush=True)
        if record_video:
            last_video = env.get_video()
        env.close()

    print(f"[Rank {rank}] Finished all rollout tasks", flush=True)
    return all_results, last_video



def maybe_collect_rollout(config, step, model, device, rank, world_size, cpu_group=None, ema_model=None):
    if getattr(config, "disable_rollout", False):
        return None
    if step > 1 and (step % config.rollout_every == 0 or step == (config.num_steps - 1)):
        dist.barrier()

        rollout_model = ema_model if ema_model is not None else model
        local_results, local_video = collect_rollout(
            config, rollout_model, device, rank, world_size
        )

        print(f"[Rank {rank}] Gathering rollout results...", flush=True)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results, group=cpu_group)
        print(f"[Rank {rank}] Gather complete", flush=True)

        if is_main_process():
            merged = {}
            for r in gathered:
                merged.update(r)
            sr_values = [v for k, v in merged.items() if k.startswith("rollout/success_rate_")]
            avg_sr = sum(sr_values) / max(len(sr_values), 1)
            merged["rollout/avg_success_rate"] = avg_sr

            print(f"Step: {step} | Avg Success Rate: {avg_sr:.4f}", flush=True)
            for k, v in sorted(merged.items()):
                if k.startswith("rollout/success_rate_"):
                    print(f"  {k}: {v:.4f}", flush=True)

            if local_video is not None:
                video = local_video.transpose(0, 3, 1, 2)[None]
                merged["rollout/video"] = wandb.Video(video, fps=10)
            merged["global_step"] = step
            wandb.log(merged)

        dist.barrier()
        if is_main_process():
            return avg_sr
    return None


def train(rank, world_size, config):
    # Set global seed
    set_seed(config.seed * world_size + rank)

    # Initialize distributed training
    init_distributed(rank, world_size)
    cpu_group = dist.new_group(backend="gloo")
    device = torch.device(f"cuda:{rank}")

    # Initialize WANDB
    if is_main_process():
        init_wandb(config, job_type="train")

    # Create dataset
    # import ipdb;ipdb.set_trace()
    train_set, val_set = instantiate(config.dataset)
    train_loader, val_loader = make_distributed_data_loader(
        train_set, val_set, config.batch_size, rank, world_size
    )
    action_normalizer = getattr(train_set, "action_normalizer", None)
    if is_main_process():
        print(
            "Motion supervision key: optical_flow_raft_latent "
            "(fallback to motion_vector in train.py)."
        )
        if action_normalizer is not None:
            print(f"Action normalizer enabled: scale={action_normalizer.scale}, offset={action_normalizer.offset}")

    # Create model

    flow_vqvae, num_spatial_tokens, background_id = build_frozen_flow_vqvae(config, device)

    model = instantiate(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Load pretrained model (strict=False to allow new VQ-VAE params to stay randomly initialized)
    if config.pretrain_checkpoint_path:
        ckpt = torch.load(config.pretrain_checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if is_main_process():
            print(f"Loaded pretrain ckpt: {config.pretrain_checkpoint_path}, "
                  f"orig step: {ckpt.get('step', '?')}")
            if missing:
                print(f"  Missing keys (newly initialized): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected}")

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

        # Train for one epoch
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
                wandb.log({f"train/{k}": v for k, v in info.items()}, step=step)

            # --- Evaluate if needed ---
            maybe_evaluate(
                config, step, model, val_loader, device, action_normalizer,
                flow_vqvae, eval_model=ema_model, background_id=background_id,
            )

            # ---Collect environment rollouts if needed ---
            rollout_avg_sr = maybe_collect_rollout(
                config, step, model, device, rank, world_size, cpu_group, ema_model=ema_model
            )

            # --- Save checkpoint if needed ---
            maybe_save_checkpoint(
                config,
                step,
                model,
                optimizer,
                scheduler,
                scaler,
                action_normalizer,
                save_model=ema_model,
                rollout_avg_sr=rollout_avg_sr,
            )

            step += 1
            pbar.update(1)
            if step >= config.num_steps:
                break

        epoch += 1


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_dp_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    # Spawn processes

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = find_free_port()
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
    # train(0, 1, config)


if __name__ == "__main__":
    main()
