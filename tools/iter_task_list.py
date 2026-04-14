#!/usr/bin/env python3
#读取task-list中的全部动作

## 输出 dataset
# python tools/iter_task_list.py --field dataset

# # 输出 hdf5 列
# python tools/iter_task_list.py --field hdf5

# # 输出 zarr 列（可直接喂给删除/检查脚本）
# python tools/iter_task_list.py --field zarr

# # 只过滤指定任务
# python tools/iter_task_list.py --field zarr --tasks-csv "OpenDrawer,TurnOffStove"

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterate centralized RoboCasa task list JSON."
    )
    parser.add_argument(
        "--list-file",
        default="/data/workspace/zhangshiqi/uwm_motion/configs/task_lists/robocasa_atomic_files.json",
        help="Path to task list JSON.",
    )
    parser.add_argument(
        "--field",
        choices=["task", "dataset", "hdf5", "zarr"],
        default="dataset",
        help="Which field to print, one per line.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include items with enabled=false.",
    )
    parser.add_argument(
        "--tasks-csv",
        default="",
        help="Optional filter by task or dataset names, comma-separated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    list_path = Path(args.list_file)
    if not list_path.exists():
        raise FileNotFoundError(f"Task list not found: {list_path}")

    with list_path.open("r") as f:
        data = json.load(f)

    only = {x.strip() for x in args.tasks_csv.split(",") if x.strip()}
    for item in data.get("items", []):
        if not args.include_disabled and not item.get("enabled", True):
            continue
        task = str(item.get("task", ""))
        dataset = str(item.get("dataset", ""))
        if only and task not in only and dataset not in only:
            continue
        value = item.get(args.field, "")
        if value:
            print(value)


if __name__ == "__main__":
    main()
