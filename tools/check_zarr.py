import os
import numpy as np
import zarr
import matplotlib.pyplot as plt

# 你的 zarr 路径（按需改）
zarr_path = "/data/shared_workspace/zhangshiqi/dataset/robocasa/zarr/OpenStandMixerHead.zarr"

# 打开 zarr
z3 = zarr.open(zarr_path, mode="r")

# 读取 arr3
arr3 = z3["data"]["obs.robot0_agentview_left_image"]
print("arr3 shape:", arr3.shape, "dtype:", arr3.dtype)
import ipdb; ipdb.set_trace()  # 进入调试器查看 arr3 的内容和结构
# 可视化第 0 帧（可改 index）
idx = 100
img = np.asarray(arr3[idx])

# 若通道顺序是 CHW，转成 HWC
if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (3, 4):
    img = np.transpose(img, (1, 2, 0))

# 保存图片
save_path = "/data/workspace/zhangshiqi/uwm_motion/tools/arr3_robot0_agentview_left_image_idx0.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.figure(figsize=(6, 6))
if img.ndim == 2:
    plt.imshow(img, cmap="gray")
else:
    plt.imshow(img)
plt.title(f"obs.robot0_agentview_left_image[{idx}]")
plt.axis("off")
plt.tight_layout()
plt.savefig(save_path, dpi=150)
plt.close()

print("saved to:", save_path)