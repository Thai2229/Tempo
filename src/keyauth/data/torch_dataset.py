"""PyTorch Dataset and P x K batch sampler for metric learning.

Triplet-style losses need several attempts from the same user in every batch
(to form positive pairs) and several users (to form negatives). PKBatchSampler
builds each batch from P random users x K random attempts per user.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from keyauth.data.container import KeystrokeData
from keyauth.data.preprocess import Normalizer


class KeystrokeDataset(Dataset):
    """Items are (x, length, label):
    x      float32 (L, 3) normalized timings, zero padded
    length int64 number of keystrokes
    label  int64 user index (from label_map)
    """

    def __init__(self, data: KeystrokeData, normalizer: Normalizer, label_map: dict[str, int] | None = None):
        if label_map is None:
            label_map = {u: i for i, u in enumerate(data.users)}
        unknown = set(data.users) - set(label_map)
        if unknown:
            raise ValueError(f"users missing from label_map: {sorted(unknown)[:5]}")
        self.label_map = label_map
        self.x = torch.from_numpy(normalizer(data))
        self.lengths = torch.from_numpy(data.lengths.astype(np.int64))
        self.labels = torch.tensor([label_map[u] for u in data.user_ids], dtype=torch.int64)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.lengths[i], self.labels[i]


class PKBatchSampler(Sampler[list[int]]):
    """Yields batches of P users x K attempts (indices into the dataset).

    Users with fewer than K attempts are sampled with replacement. Each epoch
    reshuffles, deterministically from `seed`.
    """

    def __init__(
        self,
        labels: Sequence[int] | torch.Tensor,
        p: int,
        k: int,
        batches_per_epoch: int | None = None,
        seed: int = 0,
    ):
        labels = np.asarray(labels)
        self.by_label = {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}
        if p < 2 or k < 2 or len(self.by_label) < p:
            raise ValueError(f"need p>=2, k>=2 and at least p users; got p={p}, k={k}, users={len(self.by_label)}")
        self.p = p
        self.k = k
        if batches_per_epoch is None:
            batches_per_epoch = max(1, len(labels) // (p * k))
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self._epoch))
        self._epoch += 1
        classes = np.array(sorted(self.by_label))
        for _ in range(self.batches_per_epoch):
            batch = []
            for c in rng.choice(classes, size=self.p, replace=False):
                idx = self.by_label[int(c)]
                replace = len(idx) < self.k
                batch.extend(rng.choice(idx, size=self.k, replace=replace).tolist())
            yield batch
