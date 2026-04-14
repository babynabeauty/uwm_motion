import json
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from omegaconf import OmegaConf

DATA_TYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_wandb(config, job_type):
    """Initialize WANDB logging. Only use in main process.

    Args:
        config: config dictionary
        job_type: "train" or "eval"
    """
    run_id_path = os.path.join(config.logdir, f"run_id_{job_type}.json")
    if config.resume and os.path.exists(run_id_path):
        # Load WANDB run ID from log directory
        with open(run_id_path, "r") as f:
            run_id = json.load(f)["run_id"]
    else:
        # Generate new WANDB run ID
        run_id = wandb.util.generate_id()
        with open(run_id_path, "w") as f:
            json.dump({"run_id": run_id}, f)

    wandb.init(
        project="video-action-learning",
        job_type=job_type,
        group=config.algo,
        name="_".join([config.exp_id, str(config.seed)]),
        config=OmegaConf.to_container(config, resolve=True),
        resume=config.resume,
        id=run_id,
    )


import socket
def find_free_port():
    """找到一个当前可用的空闲端口。
   一个临时socket绑定到端口0，系统会自动分配一个空闲端口。 通过创建
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return str(port)
    

def spawn_distributed_training(train_fn, config):
    """Set up MASTER_PORT/MASTER_ADDR, validate GPU visibility, then mp.spawn."""
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = find_free_port()
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError(
            f"No CUDA devices visible (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}). "
            f"Cannot start distributed training."
        )

    mp.spawn(train_fn, args=(world_size, config), nprocs=world_size, join=True)


def init_distributed(rank, world_size):
    """Initialize distributed training and set visible device.

    Args:
        rank: unique identifier of each process
        world_size: total number of processes
    """
    
    if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "23000"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=36000),
    )


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def soft_update(target, source, tau):
    """Soft update target model with source model."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class FreezeParameters:
    def __init__(self, params):
        self.params = params
        self.param_states = [p.requires_grad for p in self.params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]


LIBERO10_INFO = {
    'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5': "turn on the stove and put the moka pot on it",
    'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5': "put the black bowl in the bottom drawer of the cabinet and close it",
    'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5': "put the yellow and white mug in the microwave and close it",
    'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5': "put both moka pots on the stove",
    'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5': "put both the alphabet soup and the cream cheese box in the basket",
    'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5': "put both the alphabet soup and the tomato sauce in the basket",
    'LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5': "put both the cream cheese box and the butter in the basket",
    'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5': "put the white mug on the left plate and put the yellow and white mug on the right plate",
    'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5': "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5': "pick up the book and place it in the back compartment of the caddy"
}

def get_libero_instruction(hdf5_path):
    """
    鲁棒性获取指令：
    1. 尝试匹配文件名（针对 LIBERO-10）
    2. 如果匹配不到，则返回一个通用的默认指令
    """
    import os
    filename = os.path.basename(hdf5_path)
    # 优先匹配全名，匹配不到则返回默认
    return LIBERO10_INFO.get(filename, "robot complete the task")


def _instruction_from_robocasa_hdf5(hdf5_path):
    import h5py
    import json

    def _loads_attr(raw):
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            s = raw.decode("utf-8")
        else:
            s = str(raw)
        return json.loads(s)

    default = "robot complete the task"
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                return default
            data = f["data"]
            env_raw = data.attrs.get("env_args")
            if env_raw is not None:
                try:
                    env_args = _loads_attr(env_raw)
                    if isinstance(env_args, dict):
                        for k in ("lang", "language", "instruction", "task_language"):
                            v = env_args.get(k)
                            if v:
                                return str(v)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    pass
            for k in sorted(data.keys()):
                if not k.startswith("demo_"):
                    continue
                demo = data[k]
                em = demo.attrs.get("ep_meta")
                if em is None:
                    continue
                try:
                    ep = _loads_attr(em)
                    if isinstance(ep, dict):
                        for key in ("lang", "language", "instruction"):
                            v = ep.get(key)
                            if v:
                                return str(v)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    pass
                break
    except (OSError, KeyError):
        pass
    return default


def get_rollout_instruction(dataset_name, hdf5_path):
    """Language string for rollout CLIP tokenization (LIBERO / Robocasa / fallback)."""
    if dataset_name and "robocasa" in dataset_name:
        return _instruction_from_robocasa_hdf5(hdf5_path)
    return get_libero_instruction(hdf5_path)