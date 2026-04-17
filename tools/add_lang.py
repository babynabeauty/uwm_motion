import json
import os
import sys

import numpy as np
import zarr
from numcodecs import VLenArray

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.common.language import CLIPTextEncoder


def load_task_items(task_list_file: str):
    with open(task_list_file, "r") as f:
        data = json.load(f)
    items = []
    for item in data.get("items", []):
        if not item.get("enabled", True):
            continue
        task_name = str(item.get("task", "")).strip()
        zarr_path = str(item.get("zarr", "")).strip()
        if not task_name or not zarr_path:
            continue
        items.append(
            {
                "task": task_name.replace("_", " "),
                "dataset": str(item.get("dataset", "")).strip(),
                "zarr": zarr_path,
            }
        )
    return items


def write_lang_to_single_zarr(zarr_path: str, task_name: str, text_encoder: CLIPTextEncoder):
    root = zarr.open(zarr_path, mode="a")
    meta = root.require_group("meta")
    if "episode_ends" not in meta:
        raise KeyError(f"Missing meta/episode_ends in {zarr_path}")

    num_episodes = int(len(meta["episode_ends"]))
    if num_episodes == 0:
        print(f"[SKIP] {zarr_path}: no episodes.")
        return 0

    input_ids_ds = meta.require_dataset(
        name="input_ids",
        shape=(num_episodes,),
        dtype=object,
        object_codec=VLenArray(np.int64),
    )
    attention_mask_ds = meta.require_dataset(
        name="attention_mask",
        shape=(num_episodes,),
        dtype=object,
        object_codec=VLenArray(np.int64),
    )

    input_ids, attention_mask = text_encoder.encode(task_name)
    ids_1d = input_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
    mask_1d = attention_mask.detach().cpu().numpy().reshape(-1).astype(np.int64)

    for i in range(num_episodes):
        input_ids_ds[i] = ids_1d
        attention_mask_ds[i] = mask_1d
    return num_episodes


if __name__ == "__main__":
    task_list_file = os.environ.get(
        "TASK_LIST_FILE",
        "/data/workspace/zhangshiqi/uwm_motion/configs/task_lists/robotwin_files.json",
    )
    task_items = load_task_items(task_list_file)
    if not task_items:
        raise RuntimeError(f"No enabled task items found in {task_list_file}")

    text_encoder = CLIPTextEncoder(embed_dim=768)
    total_eps = 0
    for idx, item in enumerate(task_items):
        task_name = item["task"]
        dataset_name = item["dataset"]
        zarr_path = item["zarr"]
        if not os.path.exists(zarr_path):
            print(f"[SKIP] {dataset_name}: zarr not found: {zarr_path}")
            continue
        try:
            n = write_lang_to_single_zarr(zarr_path, task_name, text_encoder)
            total_eps += n
            print(f"[OK] {idx + 1}/{len(task_items)} {dataset_name}: wrote {n} episodes")
        except Exception as e:
            print(f"[ERR] {dataset_name}: {e}")

    print(f"Finished. Total episodes updated: {total_eps}")
