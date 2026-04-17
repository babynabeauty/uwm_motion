python - <<'PY'
import zarr
p = "//data/shared_workspace/zhangshiqi/dataset/RoboTwin/zarr/data/adjust_bottle-aloha-agilex_clean_50-50.zarr"
root = zarr.open(p, mode="r")
print(list(root["data"].keys()))
for k in root["data"].keys():
    a = root["data"][k]
    print(k, a.shape, a.chunks, a.dtype)
print(list(root["meta"].keys()))
print(root["meta"]["episode_ends"][:5])
PY
