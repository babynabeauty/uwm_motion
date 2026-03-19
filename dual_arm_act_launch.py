import threading
import time
import numpy as np
import pickle

import tyro
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.common.teleop_data_recorder import TeleopDataRecorder
from xrobotoolkit_teleop.hardware.dual_arm_ur_controller import DualArmURController
from xrobotoolkit_teleop.hardware.dual_gripper_controller import (
    DualGripperController,
)
from xrobotoolkit_teleop.hardware.ros_camera_controller import ROSCameraController
from xrobotoolkit_teleop.hardware.interface.universal_robots import (
    CONTROLLER_DEADZONE,
    LEFT_INITIAL_JOINT_DEG,
    LEFT_ROBOT_IP,
    LOOKAHEAD_TIME,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    RIGHT_INITIAL_JOINT_DEG,
    RIGHT_ROBOT_IP,
    SERVO_GAIN,
    SERVO_TIME,
    URController,
)
from xrobotoolkit_teleop.policy_controller.dual_arm_ur_gripper_controller import DualArmURGripperController
from xrobotoolkit_teleop.policy_controller.policy import (
    BasePolicy, 
    TrajectoryReplayPolicy, 
    Pi0WebDualArmPolicy,
    ActWebDualArmPolicy
)

def main(
    host: str, 
    port: int, 
    instruction: str, 
    reset: bool = False,
    visualize_placo: bool = False,
    scale_factor_xyz: float = 0.5,
    scale_factor_rot: float = 0.5,
    enable_gripper: bool = True,
    gripper_config_path: str = "config/gripper_config.yaml",
    enable_left_gripper: bool = True,
    enable_right_gripper: bool = True,
    enable_logging: bool = True,
    log_dir: str = "./logs",
    control_freq: float = 25.0,
    enable_camera: bool = True,
    camera_display: bool = True,
    control_mode: str = "right", 
    replan_step: int = 20, 
): 
    # --- 1-5 摄像头初始化部分保持不变 ---
    camera_controller = None
    if enable_camera:
        print("\n[5/5] Initializing ROS camera controller...")
        # import pdb;pdb.set_trace()
        try:
            camera_controller = ROSCameraController(
                camera_topics={
                    "left_wrist": "/camera_left/color/image_raw",
                    "right_wrist": "/camera_right/color/image_raw",
                    "head": "/camera_head/color/image_raw",
                },
                camera_fps=5, 
                enable_display=camera_display,
            )
            if camera_controller.connect():
                print("✅ Camera controller initialized")
            else:
                print("⚠️  Camera connection timeout")
                camera_controller = None
        except Exception as e:
            print(f"⚠️  Failed to initialize camera controller: {e}")
            camera_controller = None

    # --- 核心修改部分：替换 Policy ---
    
    # 使用 ACT 远程调用策略
    # 注意：ACT 模型通常需要传入更多的状态信息，确保你的服务端 API 支持 ACT 的输入格式
    policy = ActWebDualArmPolicy(
        host=host, 
        port=port,
        # 可以根据 ACT 的特性增加参数，如 temporal_agg (时间集成)
        # temporal_agg=True 
    )
     
    # import pdb;pdb.set_trace()
    # 包装到 Controller 中
    # ACT 的 replan_step 非常重要，决定了多少步重新请求一次 API
    arm_gripper_controller = DualArmURGripperController(
        policy,
        instruction=instruction, 
        control_freq=control_freq,
        camera_controller=camera_controller, 
        visualize_placo=visualize_placo,
        scale_factor_xyz=scale_factor_xyz,
        scale_factor_rot=scale_factor_rot,
        control_mode=control_mode, 
        replan_step=replan_step, # 对于 ACT，建议根据推理耗时调整此值
    )

    # --- 后续线程启动和关闭逻辑与原代码保持一致 ---
    if arm_gripper_controller:
        try:
            print("\nConnecting grippers...")
            arm_gripper_controller.gripper_connect()
            print("✅ Grippers connected")
        except Exception as e:
            print(f"⚠️  Failed to connect grippers: {e}")

    stop_signal = threading.Event()
    threads = []

    # 关节控制线程
    threads.append(threading.Thread(target=arm_gripper_controller.run_left_controller_thread, args=(stop_signal,), name="LeftArmThread"))
    threads.append(threading.Thread(target=arm_gripper_controller.run_right_controller_thread, args=(stop_signal,), name="RightArmThread"))
    threads.append(threading.Thread(target=arm_gripper_controller.run_gripper_controller_thread, args=(stop_signal,), name="GripperThread"))
    
    # 策略推理线程：这个线程会调用 ActWebDualArmPolicy 的远程 API
    threads.append(threading.Thread(target=arm_gripper_controller.run_policy_thread, args=(stop_signal,), name="PolicyThread"))

    print("\n" + "=" * 70)
    print("STARTING ACT CONTROL THREADS...")
    for thread in threads:
        thread.start()
        print(f"✅ {thread.name} started")

    try:
        while not stop_signal.is_set():
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_signal.set()

    for thread in threads:
        thread.join(timeout=2.0)
    
    arm_gripper_controller.close()
    print("Launch finish!")

if __name__ == "__main__":
    tyro.cli(main)