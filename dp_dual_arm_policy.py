import logging
import time
import functools
import numpy as np
import msgpack
import websockets.sync.client
from typing import Dict, Optional, Tuple
from typing_extensions import override
from xrobotoolkit_teleop.policy_controller.policy.base_policy import BasePolicy


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


class DpWebDualArmPolicy(BasePolicy):
    """
    Client policy that calls the remote DP (Diffusion Policy) inference server.

    Key differences from ActWebDualArmPolicy:
      - No qpos / state input  (DP model was trained with use_low_dim=False)
      - No language instruction (DP model was trained with use_language=False)
      - Server maintains observation history (obs_num_frames=2) internally
      - Action output is (action_len=16, action_dim=7) for right arm only
    """

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
        self._ws, self._server_metadata = self._wait_for_server()
        logging.info(f"DP Inference Server connected. Metadata: {self._server_metadata}")

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for DP server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except Exception as e:
                logging.info(f"Still waiting for DP server... ({e})")
                time.sleep(3)

    @override
    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from DP inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self) -> None:
        """Reconnect to clear the server-side observation history buffer."""
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws, self._server_metadata = self._wait_for_server()

    def step(
        self,
        obs: Dict = None,
        instruction: str = None,
        **kwargs,
    ):
        """
        Package observation and call remote DP inference.

        obs is expected to contain:
          - "head":        np.ndarray (H, W, 3) uint8
          - "right_wrist": np.ndarray (H, W, 3) uint8

        Returns:
          np.ndarray of shape (action_len, action_dim)  e.g. (16, 7)
        """
        start_time = time.time()

        curr_obs = {
            "images": {
                "head": obs["head"],
                "right_wrist": obs["right_wrist"],
            },
        }

        try:
            result = self.infer(curr_obs)
            actions = result["action"]  # (16, 7)
            inference_duration = time.time() - start_time
            print(
                f"[DP INFO] Inference time: {inference_duration:.3f}s, "
                f"Action chunk: {actions.shape}"
            )
            return actions
        except Exception as e:
            print(f"DP Inference failed: {e}")
            return np.zeros((1, 7))
