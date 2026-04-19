import json
import os
import sys
from typing import Union

import h5py
import numpy as np
import zarr
from numcodecs import VLenArray

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.common.language import CLIPTextEncoder


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


InstructionValue = Union[str, list[str]]


def _loads_json_attr(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _sorted_demo_keys(demos_group: h5py.Group) -> list[str]:
    keys = [k for k in demos_group.keys() if isinstance(k, str) and k.startswith("demo_")]

    def _key_order(key: str) -> tuple[int, str]:
        suffix = key[len("demo_") :]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(keys, key=_key_order)


def _lang_from_ep_meta(ep_meta) -> str:
    ep_meta = _loads_json_attr(ep_meta)
    if isinstance(ep_meta, dict):
        return str(ep_meta.get("lang", "")).strip()
    return ""


def read_hdf5_instruction(hdf5_path: str) -> str:
    if not hdf5_path:
        return ""
    with h5py.File(hdf5_path, "r") as f:
        if "data" in f:
            demos = f["data"]
            for demo_key in _sorted_demo_keys(demos):
                instruction = _lang_from_ep_meta(demos[demo_key].attrs.get("ep_meta"))
                if instruction:
                    return instruction

        env_args = _loads_json_attr(f.attrs.get("env_args")) or {}
        env_kwargs = env_args.get("env_kwargs") if isinstance(env_args, dict) else {}
        if isinstance(env_kwargs, dict):
            instruction = str(env_kwargs.get("lang", "")).strip()
            if instruction:
                return instruction

        for attr_name in ("lang", "language", "instruction"):
            instruction = str(f.attrs.get(attr_name, "")).strip()
            if instruction:
                return instruction
    return ""


def read_hdf5_episode_instructions(hdf5_path: str) -> list[str]:
    if not hdf5_path:
        return []
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return []
        instructions = []
        for demo_key in _sorted_demo_keys(f["data"]):
            instruction = _lang_from_ep_meta(f["data"][demo_key].attrs.get("ep_meta"))
            if not instruction:
                return []
            instructions.append(instruction)
        return instructions


def resolve_instruction(item: dict, instruction_source: str) -> InstructionValue:
    task_text = str(item.get("task", "")).strip().replace("_", " ")
    source = instruction_source.strip().lower()
    if source == "task":
        return task_text
    if source == "hdf5":
        hdf5_path = str(item.get("hdf5", "")).strip()
        instructions = read_hdf5_episode_instructions(hdf5_path)
        if instructions:
            return instructions
        instruction = read_hdf5_instruction(hdf5_path)
        return instruction or task_text
    if source in {"field", "json"}:
        for key in ("instruction", "language", "lang"):
            instruction = str(item.get(key, "")).strip()
            if instruction:
                return instruction
        return task_text
    raise ValueError(
        f"Unknown INSTRUCTION_SOURCE={instruction_source!r}. "
        "Use task, hdf5, or field."
    )


def load_task_items(
    task_list_file: str,
    enabled_only: bool = True,
    instruction_source: str = "task",
):
    with open(task_list_file, "r") as f:
        data = json.load(f)
    items = []
    for item in data.get("items", []):
        if enabled_only and not item.get("enabled", True):
            continue
        task_name = str(item.get("task", "")).strip()
        zarr_path = str(item.get("zarr", "")).strip()
        if not task_name or not zarr_path:
            continue
        items.append(
            {
                "task": task_name,
                "instruction": resolve_instruction(item, instruction_source),
                "dataset": str(item.get("dataset", "")).strip(),
                "zarr": zarr_path,
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return items


def _encode_instruction(instruction: str, text_encoder: CLIPTextEncoder):
    input_ids, attention_mask = text_encoder.encode(instruction)
    ids_1d = input_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
    mask_1d = attention_mask.detach().cpu().numpy().reshape(-1).astype(np.int64)
    return ids_1d, mask_1d


def write_lang_to_single_zarr(
    zarr_path: str,
    instruction: InstructionValue,
    text_encoder: CLIPTextEncoder,
):
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

    if isinstance(instruction, list):
        if len(instruction) != num_episodes:
            raise ValueError(
                f"Instruction count {len(instruction)} does not match zarr episodes {num_episodes}"
            )
        encoded_cache = {}
        for i, text in enumerate(instruction):
            if text not in encoded_cache:
                encoded_cache[text] = _encode_instruction(text, text_encoder)
            ids_1d, mask_1d = encoded_cache[text]
            input_ids_ds[i] = ids_1d
            attention_mask_ds[i] = mask_1d
    else:
        ids_1d, mask_1d = _encode_instruction(instruction, text_encoder)
        for i in range(num_episodes):
            input_ids_ds[i] = ids_1d
            attention_mask_ds[i] = mask_1d
    return num_episodes


def describe_instruction(instruction: InstructionValue) -> str:
    if not isinstance(instruction, list):
        return repr(instruction)
    unique = []
    seen = set()
    for text in instruction:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    if len(unique) == 1:
        return f"{unique[0]!r} x{len(instruction)}"
    preview = ", ".join(repr(text) for text in unique[:3])
    if len(unique) > 3:
        preview += ", ..."
    return f"{len(unique)} unique over {len(instruction)} episodes: {preview}"


if __name__ == "__main__":
    task_list_file = os.environ.get(
        "TASK_LIST_FILE",
        "/data/workspace/zhangshiqi/uwm_motion/configs/task_lists/robotwin_files.json",
    )
    enabled_only = not _env_flag("INCLUDE_DISABLED", default=False)
    instruction_source = os.environ.get("INSTRUCTION_SOURCE", "task")
    task_items = load_task_items(
        task_list_file,
        enabled_only=enabled_only,
        instruction_source=instruction_source,
    )
    if not task_items:
        raise RuntimeError(f"No task items found in {task_list_file}")

    print(
        f"Using task list: {task_list_file}\n"
        f"Instruction source: {instruction_source}\n"
        f"Enabled only: {enabled_only}"
    )

    text_encoder = CLIPTextEncoder(embed_dim=768)
    total_eps = 0
    for idx, item in enumerate(task_items):
        task_name = item["task"]
        instruction = item["instruction"]
        dataset_name = item["dataset"]
        zarr_path = item["zarr"]
        if not os.path.exists(zarr_path):
            print(f"[SKIP] {dataset_name}: zarr not found: {zarr_path}")
            continue
        try:
            n = write_lang_to_single_zarr(zarr_path, instruction, text_encoder)
            total_eps += n
            print(
                f"[OK] {idx + 1}/{len(task_items)} {dataset_name}: wrote {n} episodes "
                f"| task={task_name!r} | instruction={describe_instruction(instruction)}"
            )
        except Exception as e:
            print(f"[ERR] {dataset_name}: {e}")

    print(f"Finished. Total episodes updated: {total_eps}")
