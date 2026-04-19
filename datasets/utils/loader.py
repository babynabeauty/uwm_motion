import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class DistributedBalancedConcatSampler(Sampler[int]):
    """Sample each child dataset in a ConcatDataset with equal probability under DDP."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not hasattr(dataset, "datasets") or not hasattr(dataset, "cumulative_sizes"):
            raise TypeError("DistributedBalancedConcatSampler requires a ConcatDataset")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        self.lengths = [len(ds) for ds in dataset.datasets]
        if not self.lengths or any(length <= 0 for length in self.lengths):
            raise ValueError(f"Invalid child dataset lengths: {self.lengths}")
        self.offsets = [0] + list(dataset.cumulative_sizes[:-1])
        self.samples_per_dataset = math.ceil(len(dataset) / len(self.lengths))
        raw_total = self.samples_per_dataset * len(self.lengths)
        self.num_samples = math.ceil(raw_total / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = []
        for offset, length in zip(self.offsets, self.lengths):
            if self.shuffle:
                local = torch.randint(
                    high=length,
                    size=(self.samples_per_dataset,),
                    generator=generator,
                    dtype=torch.int64,
                ).tolist()
            else:
                local = (torch.arange(self.samples_per_dataset) % length).tolist()
            indices.extend(offset + int(idx) for idx in local)

        if self.shuffle:
            perm = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in perm]

        if len(indices) < self.total_size:
            indices.extend(indices[: self.total_size - len(indices)])
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def make_distributed_data_loader(
    train_set: Dataset,
    val_set: Dataset,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    balance_datasets: bool = False,
):
    # Training sampler and loader
    if balance_datasets and hasattr(train_set, "datasets"):
        train_sampler = DistributedBalancedConcatSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Validation sampler and loader
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader
