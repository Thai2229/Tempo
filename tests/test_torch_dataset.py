from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from tempo.data import KeystrokeData, Normalizer, make_synthetic_dataset  # noqa: E402
from tempo.data.torch_dataset import KeystrokeDataset, PKBatchSampler  # noqa: E402


@pytest.fixture
def data():
    return make_synthetic_dataset(n_users=8, samples_per_user=12, seed=0)


def test_dataset_items(data):
    ds = KeystrokeDataset(data, Normalizer.fit(data))
    x, length, label = ds[0]
    assert x.shape == (data.max_len, 3) and x.dtype == torch.float32
    assert int(length) == 11
    assert len(ds) == len(data)
    assert ds.labels.unique().numel() == 8


def test_dataset_label_map_shared_across_splits(data):
    label_map = {u: i for i, u in enumerate(data.users)}
    sub = data.for_users(["s003"])
    ds = KeystrokeDataset(sub, Normalizer.fit(data), label_map=label_map)
    assert set(ds.labels.tolist()) == {3}
    with pytest.raises(ValueError):
        KeystrokeDataset(data, Normalizer.fit(data), label_map={"s000": 0})


def test_pk_sampler_batch_composition(data):
    ds = KeystrokeDataset(data, Normalizer.fit(data))
    sampler = PKBatchSampler(ds.labels, p=4, k=3, batches_per_epoch=5, seed=0)
    batches = list(sampler)
    assert len(batches) == 5
    for b in batches:
        counts = Counter(ds.labels[b].tolist())
        assert len(counts) == 4 and set(counts.values()) == {3}
    # a new epoch gives different batches
    assert list(sampler) != batches


def test_pk_sampler_small_users_with_replacement():
    labels = [0, 0, 1, 1, 2, 2, 2, 2]
    batch = next(iter(PKBatchSampler(labels, p=3, k=4)))
    assert len(batch) == 12


def test_pk_sampler_validation():
    with pytest.raises(ValueError):
        PKBatchSampler([0, 0, 1, 1], p=3, k=2)


def test_dataloader_with_sampler_variable_lengths():
    a = KeystrokeData.from_arrays([np.full((4, 3), 0.1)] * 3, ["u1"] * 3)
    b = KeystrokeData.from_arrays([np.full((6, 3), 0.2)] * 3, ["u2"] * 3)
    data = KeystrokeData.concat([a, b])
    ds = KeystrokeDataset(data, Normalizer.fit(data))
    loader = DataLoader(ds, batch_sampler=PKBatchSampler(ds.labels, p=2, k=2))
    x, lengths, labels = next(iter(loader))
    assert x.shape == (4, 6, 3)
    assert sorted(lengths.tolist()) == [4, 4, 6, 6]
