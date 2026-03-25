import zarr
import ipdb

root = "/data/shared_workspace/wangshaoxuan/libero/datasets/libero_90"
z1 = zarr.open(f"{root}/buffer.zarr", mode="r")
z2 = zarr.open(f"{root}/buffer_224.zarr", mode="r")
ipdb.set_trace()
z3 = zarr.open("/data/shared_workspace/zhangshiqi/dataset/libero/datasets/libero_10/libero_10.zarr", mode="r")
# 需要访问 data 下的 obs.agentview_rgb 数组
arr1 = z1["data"]["obs.agentview_rgb"]
arr2 = z2["data"]["obs.agentview_rgb"]
arr3 = z3['data']["obs.agentview_rgb"]

print(arr1.shape)  # 如 (N, H, W, 3)
print(arr2.shape)  # 如 (N, 224, 224, 3)
print(arr3.shape)