import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from keyauth.data import make_synthetic_dataset, split_users  # noqa: E402
from keyauth.train import (  # noqa: E402
    TrainConfig,
    compute_loss,
    embed,
    load_checkpoint,
    main,
    save_checkpoint,
    train,
)

FAST = TrainConfig(epochs=3, batches_per_epoch=15, p=4, k=4, n_enroll=5, patience=10)


@pytest.fixture(scope="module")
def trained():
    data = make_synthetic_dataset(n_users=16, samples_per_user=30, seed=0)
    train_data, val_data, test_data = split_users(data, val_frac=0.25, test_frac=0.25, seed=0)
    model, normalizer, history = train(train_data, val_data, FAST, log=lambda s: None)
    return model, normalizer, history, test_data


def test_train_history(trained):
    _, _, history, _ = trained
    assert [h["epoch"] for h in history] == [1, 2, 3]
    for h in history:
        assert 0.0 <= h["eer"] <= 1.0
        assert np.isfinite(h["loss"])


def test_embed_shape_and_norm(trained):
    model, normalizer, _, test_data = trained
    z = embed(model, normalizer, test_data, batch_size=7)  # odd batch size exercises chunking
    assert z.shape == (len(test_data), model.config.embed_dim)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)


def test_checkpoint_roundtrip(trained, tmp_path):
    model, normalizer, _, test_data = trained
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, normalizer, threshold=0.42, metrics={"test": {"eer": 0.1}})
    model2, normalizer2, meta = load_checkpoint(path)
    assert meta["threshold"] == pytest.approx(0.42)
    assert meta["metrics"]["test"]["eer"] == 0.1
    np.testing.assert_allclose(embed(model2, normalizer2, test_data), embed(model, normalizer, test_data), atol=1e-6)


def test_load_rejects_other_files(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_cli_on_synthetic(tmp_path):
    out = tmp_path / "ckpt" / "model.pt"
    metrics = main(["--data", "synthetic", "--epochs", "2", "--batches-per-epoch", "5", "--out", str(out)])
    assert out.exists()
    report = json.loads(out.with_suffix(".json").read_text())
    assert len(report["history"]) == 2
    assert 0.0 <= metrics["test"]["eer"] <= 1.0
    _, _, meta = load_checkpoint(out)
    assert meta["threshold"] == pytest.approx(metrics["val"]["eer_threshold"])


def _small_split():
    data = make_synthetic_dataset(n_users=16, samples_per_user=30, seed=1)
    return split_users(data, val_frac=0.25, test_frac=0.25, seed=1)


def _snapshot_hook(states: list):
    def hook(row, model):
        row["hooked"] = True
        states.append({k: v.clone() for k, v in model.state_dict().items()})

    return hook


def _states_equal(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_no_patience_trains_every_epoch_and_keeps_the_last():
    train_data, val_data, _ = _small_split()
    states: list = []
    cfg = TrainConfig(epochs=3, batches_per_epoch=5, p=4, k=4, n_enroll=5, patience=None)
    model, _, history = train(train_data, val_data, cfg, log=lambda s: None, epoch_hook=_snapshot_hook(states))
    assert len(history) == 3 and all(h["hooked"] for h in history)
    assert _states_equal(model.state_dict(), states[-1])


def test_patience_stops_and_restores_the_best_val_epoch():
    train_data, val_data, _ = _small_split()
    states: list = []
    cfg = TrainConfig(epochs=8, batches_per_epoch=5, p=4, k=4, n_enroll=5, patience=1)
    model, _, history = train(train_data, val_data, cfg, log=lambda s: None, epoch_hook=_snapshot_hook(states))
    eers = [h["eer"] for h in history]
    assert len(history) < 8 or eers == sorted(eers, reverse=True)
    assert _states_equal(model.state_dict(), states[int(np.argmin(eers))])


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError):
        compute_loss(torch.eye(4), torch.tensor([0, 0, 1, 1]), TrainConfig(loss="nope"))
