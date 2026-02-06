import hydra
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
import os
from experiments.utils import find_free_port
from datasets.utils.loader import make_distributed_data_loader
from datasets.utils.file_utils import glob_all
from environments.robomimic import make_robomimic_env
from experiments.dp.train import (
    train_one_step,
    maybe_resume_checkpoint,
    maybe_evaluate,
    maybe_save_checkpoint,
)
from experiments.utils import set_seed, init_wandb, init_distributed, is_main_process,get_libero_instruction


def collect_rollout(config, model, device):
    model.eval()
    model = getattr(model, "module", model)
    
    hdf5_paths = glob_all(config.dataset.hdf5_path_globs)
    tokenizer = CLIPTokenizer.from_pretrained('/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32')
    MAX_TEXT_LEN = 25 
    
    all_results = {}
    total_success_rate = 0.0
    last_video = None

    for path in hdf5_paths:
        task_name = os.path.basename(path).replace(".hdf5", "")
        instruction = get_libero_instruction(path) 
        
        tokens = tokenizer(
            instruction, padding='max_length', max_length=MAX_TEXT_LEN, 
            truncation=True, return_tensors='pt'
        ).to(device)

        # 修正：使用当前循环的具体 path
        print(f"Collecting rollouts for task: {task_name} from {path}")
        env = make_robomimic_env(
            dataset_name=config.dataset.name,
            dataset_path=path, 
            shape_meta=config.dataset.shape_meta,
            obs_horizon=model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=True,
        )

        successes = []
        for e in trange(config.num_rollouts, desc=f"Testing {task_name}", disable=not is_main_process()):
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
        total_success_rate += task_sr
        last_video = env.get_video()
        env.close() # 建议加上，释放环境资源

    avg_sr = total_success_rate / len(hdf5_paths)
    all_results["rollout/avg_success_rate"] = avg_sr # 修正点 2：返回整个字典
    return all_results, last_video

def maybe_collect_rollout(config, step, model, device):
    if step > -1:
        if is_main_process() and (
            step % config.rollout_every == 0 or step == (config.num_steps - 1)
        ):
            # 修正点 3：接收整个字典
            results_dict, video = collect_rollout(config, model, device)
            avg_sr = results_dict["rollout/avg_success_rate"]
            
            print(f"Step: {step} | Avg Success Rate: {avg_sr:.4f}")
            
            video = video.transpose(0, 3, 1, 2)[None]
            
            # 将所有指标（总分+各子任务分）和视频一起记录
            log_data = {**results_dict}
            log_data["rollout/video"] = wandb.Video(video, fps=10)
            wandb.log(log_data, step=step)
            
        dist.barrier()

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

    # Resume from checkpoint
    step = maybe_resume_checkpoint(config, model, optimizer, scheduler, scaler)
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
                config, model, optimizer, scheduler, scaler, batch, device
            )

            # --- Logging ---
            if is_main_process():
                pbar.set_description(f"step: {step}, loss: {loss['loss']:.4f},action_loss: {loss['action_loss']:.4f},motion_loss: {loss['motion_loss']:.4f}")
                wandb.log({f"train/{k}": v for k, v in info.items()})

            # --- Evaluate if needed ---
            maybe_evaluate(config, step, model, val_loader, device)

            # ---Collect environment rollouts if needed ---
            maybe_collect_rollout(config, step, model, device)

            # --- Save checkpoint if needed ---
            maybe_save_checkpoint(config, step, model, optimizer, scheduler, scaler)

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
