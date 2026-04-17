import numpy as np
from .buffer import CompressedTrajectoryBuffer
from transformers import CLIPTokenizer

_PAD_TOKEN_ID = None


def _get_pad_token_id() -> int:
    global _PAD_TOKEN_ID
    if _PAD_TOKEN_ID is None:
        tokenizer = CLIPTokenizer.from_pretrained('/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _PAD_TOKEN_ID = int(tokenizer.pad_token_id)
    return _PAD_TOKEN_ID


class TrajectorySampler:
    """
    A class that samples sequences of observations and actions from a trajectory buffer.
    """

    def __init__(
        self,
        buffer: CompressedTrajectoryBuffer,
        seq_len: int,
        episode_mask: np.ndarray = None,
        exclude_keys: set[str] | None = None,
        obs_seq_len: int | None = None,
    ):
        """
        Initialize the trajectory sampler.

        Args:
            buffer: The trajectory buffer containing the data.
            seq_len: The length of the sequences to sample.
            episode_mask: A binary mask indicating valid episodes. If None, all episodes are valid.
            exclude_keys: Keys to skip when reading from the buffer (e.g. large unused arrays).
            obs_seq_len: Optional shorter prefix length for obs.* keys when the model only consumes
                the first few observation frames.
        """
        self.buffer = buffer
        self.seq_len = seq_len
        self.obs_seq_len = obs_seq_len
        self.keys = [k for k in self.buffer.keys() if k not in (exclude_keys or set())]
        self.episode_ends_np = np.asarray(self.buffer.episode_ends, dtype=np.int64)

        # Compute all possible sample indices
        indices = []
        episode_start = 0
        for i, episode_end in enumerate(self.buffer.episode_ends):
            if episode_mask is None or episode_mask[i]:
                for j in range(episode_start, episode_end + 1 - seq_len):
                    indices.append([j, j + seq_len])
            episode_start = episode_end
        self.indices = np.array(indices, dtype=np.int64)
        # import ipdb;ipdb.set_trace()
        print(f"Total number of valid sequences: {len(self.indices)}")

        self.pad_token_id = _get_pad_token_id()
        print("pad_token_id:", self.pad_token_id)


    def __len__(self) -> int:
        return len(self.indices)

    def sample_sequence(self, index: int) -> dict[str, np.ndarray]:
        start, end = self.indices[index]
        data = {}
        for key in self.keys:
            arr = self.buffer[key]
            key_end = end
            if self.obs_seq_len is not None and key.startswith("obs."):
                key_end = min(start + self.obs_seq_len, end)
            value = arr[start:key_end]
            data[key] = value
        
        if "input_ids" in self.buffer.meta and "attention_mask" in self.buffer.meta:
            episode_idx = int(np.searchsorted(self.episode_ends_np, start, side="right"))
            ids = self.buffer.meta["input_ids"][episode_idx]
            mask = self.buffer.meta["attention_mask"][episode_idx]
            
            L = 25
            # ids
            if ids.shape[0] >= L:
                ids_fixed = ids[:L]
            else:
                pad_amount = L - ids.shape[0]
                ids_fixed = np.concatenate([ids, np.full((pad_amount,), self.pad_token_id, dtype=np.int64)], axis=0)
            # mask
            if mask.shape[0] >= L:
                mask_fixed = mask[:L]
            else:
                pad_amount = L - mask.shape[0]
                mask_fixed = np.concatenate([mask, np.zeros((pad_amount,), dtype=np.int64)], axis=0)

            # reshape to (1, L) to match your previous convention
            data["input_ids"] = ids_fixed
            data["attention_mask"] = mask_fixed

        return data
