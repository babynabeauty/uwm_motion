#!/usr/bin/env python3
"""检测 HDF5 中 optical_flow_raft_latent 和 optical_flow 键是否存在且可读。"""

import argparse
import glob
import sys

import h5py


def check_file(path: str) -> tuple[bool, list[str]]:
    """检查单个 HDF5 文件，即使文件严重损坏也会返回结果而不崩溃。"""
    issues = []
    ok = True
    
    # 1. 尝试打开文件头，如果文件头坏了，直接报错返回
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        issues.append(f"文件无法打开 (可能损坏严重): {e}")
        return False, issues

    try:
        if "data" not in f:
            issues.append("缺少 'data' 组")
            return False, issues
        
        demos = f["data"]
        # 遍历每个 demo
        for i in range(len(demos)):
            demo_key = f"demo_{i}"
            if demo_key not in demos:
                issues.append(f"缺少 {demo_key}")
                ok = False
                continue
            
            demo = demos[demo_key]
            
            # 2. 检查并读取 actions 长度，作为对比基准
            try:
                t = demo["actions"].shape[0]
            except Exception as e:
                issues.append(f"{demo_key}: 无法读取 'actions' - {e}")
                ok = False
                continue # 如果连 actions 都读不了，跳过后续 key 的对比

            # 3. 检查光流相关的键
            for key in ("optical_flow_raft_latent", "optical_flow"):
                if key not in demo:
                    # 某些 demo 可能确实不含这些 key，标记为 issue 但继续执行
                    issues.append(f"{demo_key}: 缺少 '{key}'")
                    ok = False
                    continue
                try:
                    # 强制读取数据 [:]，检测磁盘 I/O 是否真的能通
                    arr = demo[key][:] 
                    if arr.shape[0] != t:
                        issues.append(
                            f"{demo_key}: '{key}' 长度 {arr.shape[0]} 与 actions {t} 不一致"
                        )
                        ok = False
                except Exception as e:
                    issues.append(f"{demo_key}: 读取 '{key}' 失败 - {e}")
                    ok = False
    except Exception as e:
        issues.append(f"遍历过程中发生未知错误: {e}")
        ok = False
    finally:
        f.close() # 确保无论如何都会关闭文件句柄
        
    return ok, issues

def main():
    parser = argparse.ArgumentParser(description="检测 HDF5 中 optical_flow 相关键")
    parser.add_argument(
        "dir",
        nargs="?",
        default="/data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_90",
        help="HDF5 所在目录",
    )
    args = parser.parse_args()

    pattern = f"{args.dir.rstrip('/')}/*.hdf5"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"未找到匹配的 .hdf5 文件: {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"共 {len(paths)} 个文件\n")
    total_ok = 0
    total_fail = 0
    for path in paths:
        name = path.split("/")[-1]
        ok, issues = check_file(path)
        if ok:
            total_ok += 1
            print(f"[OK]  {name}")
        else:
            total_fail += 1
            print(f"[FAIL] {name}")
            for issue in issues:
                print(f"       {issue}")
            print()

    print("-" * 60)
    print(f"通过: {total_ok}, 存在问题: {total_fail}")
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()

#坏块
# KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_demo.hdf5
# STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy_demo.hdf5
# STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5