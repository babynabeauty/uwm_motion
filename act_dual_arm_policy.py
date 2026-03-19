import logging
import time
import functools
import numpy as np
import msgpack
import websockets.sync.client
from typing import Dict, Optional, Tuple
from typing_extensions import override
from xrobotoolkit_teleop.policy_controller.policy.base_policy import BasePolicy

# 序列化辅助函数 (保持不变)
def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)

class ActWebDualArmPolicy(BasePolicy): 
    def __init__(
        self, 
        host: str = "0.0.0.0", 
        port: Optional[int] = None, 
        api_key: Optional[str] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
            
        self._packer = Packer()
        self._api_key = api_key
        # 建立连接
        self._ws, self._server_metadata = self._wait_for_server()
        logging.info(f"✅ ACT Inference Server connected. Metadata: {self._server_metadata}")

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for ACT server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except Exception as e:
                logging.info(f"Still waiting for server... ({e})")
                time.sleep(3)

    @override
    def infer(self, obs: Dict) -> Dict:
        # 发送序列化后的数据
        data = self._packer.pack(obs)
        self._ws.send(data)
        
        # 接收响应
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in ACT inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self) -> None:
        """ACT 模型有时需要在新回合开始时重置隐变量 (z)，取决于实现"""
        # 如果远程 API 支持 reset 指令，可以在这里发送
        pass
    
    def step(
        self, 
        obs: Dict = None, 
        instruction: str = None, 
        **kwargs
    ):
        """
        处理 Observation 并调用远程推理
        obs 预期包含: head, left_wrist, right_wrist, state (qpos)
        """
        start_time = time.time()
        
        # 构建 ACT 预期的输入格式
        # 确保这些 key 与你服务端推理脚本中接收的 key 一致
        curr_obs = {
            "images": {
                "head": obs["head"],
                # "left_wrist": obs["left_wrist"], 
                "right_wrist": obs["right_wrist"] 
            },
            "qpos": obs["state"], # ACT 的关键输入：当前关节角度 + 夹爪位置
            "instruction": instruction,
        }
        
        # 远程推理
        try:
            # import pdb;pdb.set_trace()
            result = self.infer(curr_obs)
            # ACT 通常返回 [chunk_size, action_dim]
            # import pdb;pdb.set_trace()
            actions = result["action"]            
            inference_duration = time.time() - start_time
            print(f"[ACT INFO] Inference time: {inference_duration:.3f}s, Action chunk size: {actions.shape[0]}")
            
            return actions
        except Exception as e:
            print(f"❌ ACT Inference failed: {e}")
            # 返回一个零动作序列或维持现状，防止机器人失控
            return np.zeros((1, 14)) 
