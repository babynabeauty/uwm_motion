import os

from .wrappers import RoboMimicEnvWrapper, LIBEROEnvWrapper, RoboCasaEnvWrapper
import sys

libero_path = "/data/workspace/zhangshiqi/LIBERO"
if libero_path not in sys.path:
    sys.path.append(libero_path)

_robocasa_root = os.environ.get("ROBOCASA_ROOT", "/data/workspace/zhangshiqi/robocasa")
if _robocasa_root not in sys.path and os.path.isdir(
    os.path.join(_robocasa_root, "robocasa")
):
    sys.path.insert(0, _robocasa_root)

def make_robomimic_env(
    dataset_name,
    dataset_path,
    shape_meta,
    obs_horizon,
    max_episode_length,
    record=False,
    render_gpu_id=None,
):
    if "robomimic" in dataset_name:
        from robomimic.utils.env_utils import get_env_type, get_env_class
        from robomimic.utils.file_utils import (
            get_env_metadata_from_dataset,
            get_shape_metadata_from_dataset,
        )
        from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs

        # Initialize observation modalities
        rgb_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]
        low_dim_keys = [
            k for k, v in shape_meta["obs"].items() if v["type"] == "low_dim"
        ]
        all_obs_keys = rgb_keys + low_dim_keys
        initialize_obs_utils_with_obs_specs(
            {"obs": {"rgb": rgb_keys, "low_dim": low_dim_keys}}
        )

        # Create environment
        env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
        env_type = get_env_type(env_meta=env_meta)
        env_class = get_env_class(env_type=env_type)
        shape_meta = get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            all_obs_keys=all_obs_keys,
            verbose=True,
        )
        if render_gpu_id is not None:
            env_meta["env_kwargs"]["render_gpu_device_id"] = render_gpu_id
        elif os.environ.get("CUDA_VISIBLE_DEVICES", None):
            env_meta["env_kwargs"]["render_gpu_device_id"] = int(
                os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
            )
        env = env_class(
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=True,
            use_image_obs=shape_meta["use_images"],
            use_depth_obs=shape_meta["use_depths"],
            postprocess_visual_obs=False,  # use raw images
            **env_meta["env_kwargs"],
        )
        env = RoboMimicEnvWrapper(
            env, all_obs_keys, obs_horizon, max_episode_length, record=record
        )
    elif "libero" in dataset_name:
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import get_libero_path

        # Construct environment kwargs
        bddl_file_name = os.path.join(
            get_libero_path("bddl_files"),
            "libero_10",
            dataset_path.split("/")[-1].replace("_demo.hdf5", ".bddl"),
        )
        env_kwargs = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": 128,
            "camera_widths": 128,
        }

        if render_gpu_id is not None:
            env_kwargs["render_gpu_device_id"] = render_gpu_id
        elif os.environ.get("CUDA_VISIBLE_DEVICES", None):
            env_kwargs["render_gpu_device_id"] = int(
                os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
            )

        # Create environment
        env = OffScreenRenderEnv(**env_kwargs)

        # #增大视觉偏移
        # #FIXME
        # for initializer in env.problem_config.initializers:
        #         if hasattr(initializer, 'pos_sampler') and initializer.pos_sampler is not None:
        #             # 假设原本是 [-0.02, 0.02]，现在变为 [-0.05, 0.05]
        #             initializer.pos_sampler.x_range = [r + (0.03 if i==1 else -0.03) 
        #                                               for i, r in enumerate(initializer.pos_sampler.x_range)]
        #             initializer.pos_sampler.y_range = [r + (0.03 if i==1 else -0.03) 
        #                                               for i, r in enumerate(initializer.pos_sampler.y_range)]
                    
        obs_keys = list(shape_meta["obs"].keys())
        env = LIBEROEnvWrapper(
            env, obs_keys, obs_horizon, max_episode_length, record=record
        )
    elif "robocasa" in dataset_name:
        from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa
        from robocasa.utils.robomimic.robomimic_dataset_utils import (
            get_env_metadata_from_dataset as _rc_get_env_meta,
        )
        import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

        rgb_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]
        low_dim_keys = [
            k for k, v in shape_meta["obs"].items() if v["type"] == "low_dim"
        ]
        ObsUtils.initialize_obs_utils_with_obs_specs(
            {"obs": {"rgb": rgb_keys, "low_dim": low_dim_keys}}
        )

        cam_names = [k.replace("_image", "") for k in rgb_keys]
        cam_h, cam_w = shape_meta["obs"][rgb_keys[0]]["shape"][:2]

        raw_meta = _rc_get_env_meta(dataset_path=dataset_path)
        if isinstance(raw_meta, dict) and "env_kwargs" in raw_meta:
            env_name = raw_meta["env_name"]
            env_kwargs = dict(raw_meta["env_kwargs"])
        else:
            env_kwargs = dict(raw_meta)
            env_name = env_kwargs.pop("env_name", None)
            if env_name is None:
                raise ValueError(
                    "Robocasa HDF5 data.attrs['env_args'] must contain env_name "
                    f"(keys present: {list(env_kwargs.keys())[:30]})"
                )

        env_kwargs.pop("env_name", None)

        if render_gpu_id is not None:
            env_kwargs["render_gpu_device_id"] = render_gpu_id
        elif os.environ.get("CUDA_VISIBLE_DEVICES", None):
            env_kwargs.setdefault(
                "render_gpu_device_id",
                int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]),
            )

        env_kwargs.setdefault("camera_names", cam_names)
        env_kwargs.setdefault("camera_heights", cam_h)
        env_kwargs.setdefault("camera_widths", cam_w)

        env = EnvRobocasa(
            env_name=env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            postprocess_visual_obs=False,
            **env_kwargs,
        )
        obs_keys = list(shape_meta["obs"].keys())
        env = RoboCasaEnvWrapper(
            env,
            obs_keys,
            obs_horizon,
            max_episode_length,
            policy_action_dim=shape_meta["action"]["shape"][0],
            record=record,
        )
    else:
        raise NotImplementedError(f"Unsupported environment: {dataset_name}")
    return env
