#!/usr/bin/env bash
# Batch-convert all RoboCasa atomic tasks from LeRobot format to HDF5.
#
# Usage:
#   bash scripts/convert_all_robocasa.sh
#   # Or run in background:
#   nohup bash scripts/convert_all_robocasa.sh > convert_robocasa.log 2>&1 &

set -euo pipefail

# export PYTHONPATH=/data/workspace/zhangshiqi/uwm_motion:$PYTHONPATH

SRC_ROOT="/data/shared_workspace/zhangshiqi/dataset/robocasa/v1.0/target/atomic"
DST_ROOT="/data/shared_workspace/zhangshiqi/dataset/robocasa/hdf5"
mkdir -p "$DST_ROOT"

CAMERA_H=128
CAMERA_W=128
ACTION_DIM=7

for task_dir in "$SRC_ROOT"/*/; do
    task_name=$(basename "$task_dir")

    # Each task may have one or more date sub-dirs; pick the latest one
    date_dir=$(ls -1d "$task_dir"*/ 2>/dev/null | sort | tail -1)
    if [ -z "$date_dir" ]; then
        echo "[WARN] No date sub-dir for $task_name, skipping"
        continue
    fi

    lerobot_dir="${date_dir}lerobot"
    if [ ! -d "$lerobot_dir" ]; then
        echo "[WARN] No lerobot/ dir in $date_dir, skipping"
        continue
    fi

    output_hdf5="${DST_ROOT}/${task_name}.hdf5"
    if [ -f "$output_hdf5" ]; then
        echo "[SKIP] $output_hdf5 already exists"
        continue
    fi

    echo "========================================"
    echo "Converting: $task_name"
    echo "  src: $lerobot_dir"
    echo "  dst: $output_hdf5"
    echo "========================================"

    python tools/convert_robocasa_to_hdf5.py \
        --lerobot_dir "$lerobot_dir" \
        --output "$output_hdf5" \
        --camera_height $CAMERA_H \
        --camera_width $CAMERA_W \
        --action_dim $ACTION_DIM

    echo ""
done

echo "All done. HDF5 files are in $DST_ROOT/"
ls -lh "$DST_ROOT/"
