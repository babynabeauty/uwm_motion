import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ipdb
import os
import ipdb

# === 替换为你的 LIBERO 数据路径 ===
# 如果你还没有下载，可以先随便找一个 hdf5 文件，或者去 LIBERO github 下一个 sample
# HDF5_PATH = "/data1/dataset/libero/libero_90/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5" 
HDF5_PATH = "/data/shared_workspace/zhangshiqi/dataset/robocasa/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams.hdf5"
def inspect_libero(file_path):
    print(f"🚀 Loading LIBERO dataset: {file_path}")
    
    try:
        f = h5py.File(file_path, 'r')
    except Exception as e:
        print(f"❌ 无法打开文件: {e}")
        return

    ipdb.set_trace()
    
    demos = list(f['data'].keys())
    print(f"✅ Total Demos: {len(demos)}")
    
    # 2. 深入查看第一条轨迹 (demo_0)
    demo_key = demos[3]
    demo = f['data'][demo_key]
    
    print("\n📦 Demo Structure (demo_0):")
    # 打印 obs 下的所有 keys
    # demo的keys ['actions', 'dones', 'obs', 'rewards', 'robot_states', 'states']
    ipdb.set_trace()
    obs = demo['obs']
    print(f"   Observation Keys: {list(obs.keys())}")
    
    # 3. 提取关键数据
    # 图像 (注意：LIBERO 图像通常是 RGB，且上下颠倒的概率较低，但也需要检查)
    # 形状通常是 (N, H, W, 3)
    rgb_images = obs['agentview_rgb'][()]
    # TCP Pose (N, 3)
    ee_pos = obs['robot0_eef_pos'][()]
    # Gripper (N, 1)
    gripper = obs['robot0_gripper_qpos'][()] 

    N = rgb_images.shape[0]
    print(f"\n📏 Trajectory Length: {N}")
    print(f"   Image Shape: {rgb_images.shape}")
    print(f"   EE Pos Shape: {ee_pos.shape}")

    # 4. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图 1: 起始帧
    axes[0].imshow(rgb_images[0])
    axes[0].set_title("Start Frame (t=0)")
    
    # 图 2: 中间帧 (抓取发生大概率在这里)
    mid_idx = N // 2
    axes[1].imshow(rgb_images[mid_idx])
    axes[1].set_title(f"Middle Frame (t={mid_idx})")
    
    # 图 3: 结束帧
    axes[2].imshow(rgb_images[-1])
    axes[2].set_title(f"End Frame (t={N-1})")
    
    plt.tight_layout()
    plt.show()
    
    # 5. 检查物体 Pose 真值 (Object Ground Truth)
    # LIBERO 的物体信息有时藏在 states 里，或者 obs 的特定 key 里
    print("\n🔍 Checking for Object Poses in 'obs':")
    for key in obs.keys():
        if 'pos' in key or 'quat' in key:
            print(f"   - {key}: {obs[key].shape}")

if __name__ == "__main__":
    if os.path.exists(HDF5_PATH):
        inspect_libero(HDF5_PATH)
    else:
        print(f"⚠️ 文件 {HDF5_PATH} 不存在。请先下载 LIBERO 数据集。")
        print("下载命令示例: python download_libero_datasets.py (from official repo)")