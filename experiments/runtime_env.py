"""
Redirect temp and common caches off the root filesystem (/).

Call bootstrap_non_root_runtime() at the very top of training entry points,
before imports that may call tempfile.gettempdir() (e.g. wandb, hydra).
"""

from __future__ import annotations

import os


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bootstrap_non_root_runtime() -> None:
    """
    If UWM_ALLOW_ROOT_TMP is not set, force TMPDIR and default caches under
    UWM_DATA_ROOT (default: /data/shared_workspace/zhangshiqi/uwm_runtime).
    Existing WANDB_DIR / explicit user paths are preserved when already set.
    """
    if os.environ.get("UWM_ALLOW_ROOT_TMP", "").lower() in ("1", "true", "yes"):
        return

    default_root = "/data/shared_workspace/zhangshiqi/uwm_runtime"
    root = os.environ.get("UWM_DATA_ROOT", default_root)
    try:
        _ensure_dir(root)
    except OSError:
        root = "/data/workspace/zhangshiqi/.uwm_runtime"
        _ensure_dir(root)

    tmp = os.path.join(root, "tmp")
    _ensure_dir(tmp)
    os.environ["TMPDIR"] = tmp
    os.environ["TMP"] = tmp
    os.environ["TEMP"] = tmp

    cache = os.path.join(root, "cache")
    _ensure_dir(cache)

    if not os.environ.get("WANDB_DIR"):
        wb = os.path.join(root, "wandb")
        _ensure_dir(wb)
        os.environ["WANDB_DIR"] = wb

    os.environ.setdefault("XDG_CACHE_HOME", cache)
    torch_home = os.path.join(cache, "torch")
    hf_home = os.path.join(cache, "hf")
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache, "matplotlib"))
    for p in (
        os.environ["TORCH_HOME"],
        os.environ["HF_HOME"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["HF_DATASETS_CACHE"],
        os.environ["MPLCONFIGDIR"],
    ):
        _ensure_dir(p)
