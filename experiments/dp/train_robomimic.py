import os
import json
import time

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
    spawn_distributed_training,
)

_CLIP_TOKENIZER = None


def _profile_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _profile_add(stats: dict[str, float], values: dict[str, float]):
    for key, value in values.items():
        stats[key] = stats.get(key, 0.0) + float(value)


def _profile_report(rank: int, step: int, stats: dict[str, float], count: int):
    if count <= 0:
        return
    ordered_keys = [
        "dataloader_wait",
        "to_device_and_vq",
        "forward",
        "backward",
        "optimizer",
        "scheduler",
        "ema",
        "wandb_log",
        "eval",
        "rollout",
        "checkpoint",
        "step_total",
    ]
    parts = []
    for key in ordered_keys:
        if key in stats:
            parts.append(f"{key}={stats[key] / count * 1000:.1f}ms")
    print(f"[Rank {rank}] profile step={step} avg over {count} steps | " + " | ".join(parts), flush=True)

def _get_tokenizer():
    global _CLIP_TOKENIZER
    if _CLIP_TOKENIZER is None:
        _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(
            '/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32'
        )
    return _CLIP_TOKENIZER


def _get_rollout_paths(config):
    if hasattr(config.dataset, "hdf5_path_globs"):
        return glob_all(config.dataset.hdf5_path_globs)

    if hasattr(config.dataset, "task_list_file"):
        with open(config.dataset.task_list_file, "r") as f:
            data = json.load(f)
        enabled_only = bool(getattr(config.dataset, "enabled_only", False))
        paths = []
        for item in data.get("items", []):
            if enabled_only and not item.get("enabled", True):
                continue
            hdf5_path = str(item.get("hdf5", "")).strip()
            if hdf5_path:
                paths.append(hdf5_path)
        return paths

    raise ValueError("Dataset config must define hdf5_path_globs or task_list_file for rollout.")


def _get_rollout_entries(config):
    if hasattr(config.dataset, "hdf5_path_globs"):
        return [
            {
                "task_name": os.path.basename(path).replace(".hdf5", ""),
                "metric_name": os.path.basename(path).replace(".hdf5", ""),
                "dataset_path": path,
            }
            for path in glob_all(config.dataset.hdf5_path_globs)
        ]

    if hasattr(config.dataset, "task_list_file"):
        with open(config.dataset.task_list_file, "r") as f:
            data = json.load(f)
        enabled_only = bool(getattr(config.dataset, "enabled_only", False))
        entries = []
        for item in data.get("items", []):
            if enabled_only and not item.get("enabled", True):
                continue
            hdf5_path = str(item.get("hdf5", "")).strip()
            if not hdf5_path:
                continue
            task_name = str(item.get("task", "")).strip() or os.path.basename(hdf5_path).replace(".hdf5", "")
            metric_name = str(item.get("dataset", "")).strip() or task_name
            entries.append(
                {
                    "task_name": task_name,
                    "metric_name": metric_name,
                    "dataset_path": hdf5_path,
                }
            )
        return entries

    raise ValueError("Dataset config must define hdf5_path_globs or task_list_file for rollout.")


def _log_dataset_summary(train_set, val_set, rank):
    if rank != 0:
        return

    print(f"Train dataset size: {len(train_set)}", flush=True)
    print(f"Val dataset size: {len(val_set)}", flush=True)

    if hasattr(train_set, "datasets"):
        print(f"Train dataset is a mixture of {len(train_set.datasets)} sub-datasets:", flush=True)
        for i, ds in enumerate(train_set.datasets):
            ds_name = getattr(ds, "name", f"dataset_{i}")
            ds_task = getattr(ds, "task_name", ds_name)
            print(
                f"  [{i}] task={ds_task} | dataset={ds_name} | samples={len(ds)}",
                flush=True,
            )



def collect_rollout(config, model, device, rank=0, world_size=1):
    # import ipdb;ipdb.set_trace()
    model.eval()
    model = getattr(model, "module", model)

    all_rollout_entries = _get_rollout_entries(config)
    my_entries = all_rollout_entries[rank::world_size]

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    render_gpu_id = int(cuda_devices[rank % len(cuda_devices)])

    tokenizer = _get_tokenizer()
    MAX_TEXT_LEN = 25

    all_results = {}
    last_video = None

    print(f"[Rank {rank}] Starting rollout: {len(my_entries)} tasks assigned", flush=True)

    for entry in my_entries:
        path = entry["dataset_path"]
        task_name = entry["task_name"]
        metric_name = entry["metric_name"]
        fallback_instruction = get_rollout_instruction(config.dataset.name, path)

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
            instruction = (
                env.get_current_language()
                if hasattr(env, "get_current_language")
                else None
            ) or fallback_instruction
            tokens = tokenizer(
                instruction,
                padding='max_length',
                max_length=MAX_TEXT_LEN,
                truncation=True,
                return_tensors='pt',
            ).to(device)
            if e == 0:
                print(
                    f"[Rank {rank}] Rollout instruction for {task_name}: {instruction}",
                    flush=True,
                )
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
        all_results[f"rollout/success_rate_{metric_name}"] = task_sr
        print(f"[Rank {rank}] Task success: {task_sr:.4f}", flush=True)
        if record_video:
            last_video = env.get_video()
        env.close()

    print(f"[Rank {rank}] Finished all rollout tasks", flush=True)
    return all_results, last_video



def maybe_collect_rollout(config, step, model, device, rank, world_size, cpu_group=None, ema_model=None):
    if getattr(config, "disable_rollout", False):
        return None
    if step > 1999 and (step % config.rollout_every == 0 or step == (config.num_steps - 1)):
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
    dl_cfg = OmegaConf.to_container(config.dataloader, resolve=True) if hasattr(config, "dataloader") else {}
    train_loader, val_loader = make_distributed_data_loader(
        train_set, val_set, config.batch_size, rank, world_size,
        **dl_cfg,
    )
    _log_dataset_summary(train_set, val_set, rank)
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
    profile_steps = _profile_env_int("UWM_PROFILE_STEPS", 0)
    profile_every = max(1, _profile_env_int("UWM_PROFILE_EVERY", 10))
    profile_rank = _profile_env_int("UWM_PROFILE_RANK", 0)
    profile_enabled = profile_steps > 0 and (profile_rank < 0 or rank == profile_rank)
    profile_stats = {}
    profile_count = 0
    if profile_enabled:
        print(
            f"[Rank {rank}] Profiling first {profile_steps} train steps "
            f"(report every {profile_every}; timings include CUDA synchronization).",
            flush=True,
        )

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
        data_iter = iter(train_loader)
        while True:
            do_profile = profile_enabled and profile_count < profile_steps
            fetch_t0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            if do_profile:
                step_stats = {"dataloader_wait": time.perf_counter() - fetch_t0}
                step_t0 = time.perf_counter()
                train_timings = {}
            else:
                step_stats = None
                train_timings = None

            # --- Training step ---
            loss, info = train_one_step(
                config, model, optimizer, scheduler, scaler, batch, device,
                flow_vqvae, background_id=background_id, timings=train_timings,
            )
            if do_profile:
                _profile_add(step_stats, train_timings)

            phase_t0 = time.perf_counter()
            update_ema_model(ema_model, model, config.ema_decay)
            if do_profile:
                step_stats["ema"] = time.perf_counter() - phase_t0

            # --- Logging ---
            phase_t0 = time.perf_counter()
            if is_main_process():
                pbar.set_description(f"step: {step}, loss: {loss['loss']:.4f},action_loss: {loss['action_loss']:.4f},motion_loss: {loss['motion_loss']:.4f}")
                wandb.log({f"train/{k}": v for k, v in info.items()}, step=step)
            if do_profile:
                step_stats["wandb_log"] = time.perf_counter() - phase_t0

            # --- Evaluate if needed ---
            phase_t0 = time.perf_counter()
            maybe_evaluate(
                config, step, model, val_loader, device, action_normalizer,
                flow_vqvae, eval_model=ema_model, background_id=background_id,
            )
            if do_profile:
                step_stats["eval"] = time.perf_counter() - phase_t0

            # ---Collect environment rollouts if needed ---
            phase_t0 = time.perf_counter()
            rollout_avg_sr = maybe_collect_rollout(
                config, step, model, device, rank, world_size, cpu_group, ema_model=ema_model
            )
            if do_profile:
                step_stats["rollout"] = time.perf_counter() - phase_t0

            # --- Save checkpoint if needed ---
            phase_t0 = time.perf_counter()
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
            if do_profile:
                step_stats["checkpoint"] = time.perf_counter() - phase_t0
                step_stats["step_total"] = time.perf_counter() - step_t0
                _profile_add(profile_stats, step_stats)
                profile_count += 1
                if profile_count % profile_every == 0 or profile_count == profile_steps:
                    _profile_report(rank, step, profile_stats, profile_count)

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
    OmegaConf.resolve(config)
    spawn_distributed_training(train, config)
    # train(0, 1, config)


if __name__ == "__main__":
    main()
