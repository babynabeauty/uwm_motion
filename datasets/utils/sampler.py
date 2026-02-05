import numpy as np
from .buffer import CompressedTrajectoryBuffer
from transformers import CLIPTokenizer

class TrajectorySampler:
    """
    A class that samples sequences of observations and actions from a trajectory buffer.
    """

    def __init__(
        self,
        buffer: CompressedTrajectoryBuffer,
        seq_len: int,
        episode_mask: np.ndarray = None,
    ):
        """
        Initialize the trajectory sampler.

        Args:
            buffer: The trajectory buffer containing the data.
            seq_len: The length of the sequences to sample.
            episode_mask: A binary mask indicating valid episodes. If None, all episodes are valid.
        """
        self.buffer = buffer
        self.seq_len = seq_len
        self.keys = list(self.buffer.keys())

        # Compute all possible sample indices
        indices = []
        episode_start = 0
        for i, episode_end in enumerate(self.buffer.episode_ends):
            if episode_mask is None or episode_mask[i]:
                for j in range(episode_start, episode_end + 1 - seq_len):
                    indices.append([j, j + seq_len])
            episode_start = episode_end
        self.indices = np.array(indices, dtype=np.int64)
        print(f"Total number of valid sequences: {len(self.indices)}")

        tokenizer = CLIPTokenizer.from_pretrained('/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32')

        # 手动设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token_id = tokenizer.pad_token_id
        print("pad_token_id:", self.pad_token_id)


    def __len__(self) -> int:
        return len(self.indices)

    def sample_sequence(self, index: int) -> dict[str, np.ndarray]:
        start, end = self.indices[index]
        data = {}
        for key in self.keys:
            arr = self.buffer[key]
            value = arr[start:end]
            data[key] = value
        
        if "input_ids" in self.buffer.meta and "attention_mask" in self.buffer.meta:
            episode_idx = 0  # Default to first episode
            for i, episode_end in enumerate(self.buffer.episode_ends):
                if start < episode_end:
                    episode_idx = i
                    break
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
