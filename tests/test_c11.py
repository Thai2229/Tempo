"""C11 code paths on synthetic data: final-Enter dropping, key identity, --init fine-tuning,
checkpoint meta, the embedder's input handling, phrase-independent splits, and their failure paths."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from keyauth.auth import EngineConfig, InvalidTimings, KeyAuthEngine, TemplateStore  # noqa: E402
from keyauth.auth.embedder import TorchEmbedder  # noqa: E402
from keyauth.data import KeystrokeData, make_synthetic_dataset, split_phrase_independent, split_users  # noqa: E402
from keyauth.data.torch_dataset import KeystrokeDataset  # noqa: E402
from keyauth.model import EncoderConfig, RhythmEncoder  # noqa: E402
from keyauth.train import (  # noqa: E402
    TrainConfig,
    aalto_splits,
    embed,
    load_checkpoint,
    main,
    save_checkpoint,
    train,
)

FAST = TrainConfig(epochs=1, batches_per_epoch=3, p=4, k=4, n_enroll=5)
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_models.py"


def free_text(n_users: int = 40, per_user: int = 20, n_sentences: int = 40, seed: int = 0) -> KeystrokeData:
    """Synthetic stand-in for Aalto: each user types `per_user` distinct sentence ids once, with key codes
    and a correction flag on every third attempt."""
    data = make_synthetic_dataset(n_users=n_users, samples_per_user=per_user, seed=seed)
    user_index = np.repeat(np.arange(n_users), per_user)
    attempt = np.tile(np.arange(per_user), n_users)
    rng = np.random.default_rng(seed)
    codes = rng.integers(65, 91, size=data.values.shape[:2]).astype(np.int64)
    codes[np.arange(data.max_len)[None, :] >= data.lengths[:, None]] = 0
    return dataclasses.replace(
        data,
        sentence_ids=((user_index * 7 + attempt) % n_sentences).astype(np.int64),
        key_codes=codes,
        has_correction=attempt % 3 == 0,
    )


# ---- dropping the final Enter ---------------------------------------------------------


def test_drop_final_key_shortens_and_zeroes_the_new_last_keystroke():
    data = free_text(n_users=2, per_user=3)
    original = data.values.copy()
    out = data.drop_final_key()
    n = int(data.lengths[0])
    assert np.all(out.lengths == data.lengths - 1)
    np.testing.assert_array_equal(out.values[0, : n - 2], data.values[0, : n - 2])
    assert out.values[0, n - 2, 0] == data.values[0, n - 2, 0]  # hold kept
    assert np.all(out.values[0, n - 2, 1:] == 0.0) and np.all(out.values[0, n - 1] == 0.0)
    assert out.key_codes[0, n - 1] == 0 and out.key_codes[0, n - 2] == data.key_codes[0, n - 2]
    np.testing.assert_array_equal(data.values, original)  # the input is not modified
    assert data.lengths[0] == n


def test_drop_final_key_rejects_too_short_attempts():
    two_keys = KeystrokeData.from_arrays([np.array([[0.1, 0.2, 0.1], [0.1, 0.0, 0.0]])], ["u"])
    with pytest.raises(ValueError):
        two_keys.drop_final_key()
    with pytest.raises(ValueError):
        two_keys.subset(np.zeros(1, dtype=bool)).drop_final_key()


def test_npz_cache_roundtrip(tmp_path):
    data = free_text(n_users=3, per_user=4)
    path = tmp_path / "cache" / "data.npz"
    data.save_npz(path)
    back = KeystrokeData.load_npz(path)
    np.testing.assert_array_equal(back.values, data.values)
    np.testing.assert_array_equal(back.key_codes, data.key_codes)
    np.testing.assert_array_equal(back.has_correction, data.has_correction)
    assert back.user_ids.tolist() == data.user_ids.tolist()
    plain = make_synthetic_dataset(n_users=2, samples_per_user=3)
    plain.save_npz(tmp_path / "plain.npz")
    assert KeystrokeData.load_npz(tmp_path / "plain.npz").key_codes is None


# ---- phrase-independent splits ------------------------------------------------------


def test_split_phrase_independent_is_user_and_sentence_disjoint():
    data = free_text()
    train_data, val_data, test_data = split_phrase_independent(data, holdout_frac=0.5, min_eval_attempts=6, seed=0)
    assert not set(train_data.users) & set(val_data.users)
    assert not set(train_data.users) & set(test_data.users)
    assert not set(val_data.users) & set(test_data.users)
    seen = set(train_data.sentence_ids.tolist())
    assert not seen & set(val_data.sentence_ids.tolist())
    assert not seen & set(test_data.sentence_ids.tolist())
    for part in (val_data, test_data):
        assert min(int(np.sum(part.user_ids == u)) for u in part.users) >= 6


def test_split_phrase_independent_fails_without_enough_eval_attempts():
    with pytest.raises(ValueError):
        split_phrase_independent(free_text(), holdout_frac=0.5, min_eval_attempts=100)
    with pytest.raises(ValueError):
        split_phrase_independent(make_synthetic_dataset(n_users=10, samples_per_user=5))  # no sentence ids


def test_aalto_splits_drop_corrections_only_changes_training():
    data = free_text()
    keep = aalto_splits(data, "keep", n_enroll=5)
    drop = aalto_splits(data, "drop", n_enroll=5)
    assert not drop[0].has_correction.any() and keep[0].has_correction.any()
    assert drop[1].user_ids.tolist() == keep[1].user_ids.tolist()
    assert drop[2].user_ids.tolist() == keep[2].user_ids.tolist()
    with pytest.raises(ValueError):
        aalto_splits(data, "maybe")
    with pytest.raises(ValueError):
        aalto_splits(dataclasses.replace(data, has_correction=None))


# ---- key identity ---------------------------------------------------------------------


def test_key_identity_model_needs_keys_and_plain_model_refuses_them():
    data = free_text(n_users=2, per_user=2)
    x = torch.zeros(4, data.max_len, 3)
    lengths = torch.from_numpy(data.lengths)
    keys = torch.from_numpy(data.key_codes)
    with_keys = RhythmEncoder(EncoderConfig(key_vocab=256)).eval()
    plain = RhythmEncoder().eval()
    assert with_keys(x, lengths, keys).shape == (4, 64)
    with pytest.raises(ValueError):
        with_keys(x, lengths)
    with pytest.raises(ValueError):
        plain(x, lengths, keys)
    with pytest.raises(ValueError):
        with_keys(x, lengths, keys[:, :3])


def test_key_identity_changes_the_embedding():
    data = free_text(n_users=2, per_user=2)
    model = RhythmEncoder(EncoderConfig(key_vocab=256)).eval()
    x = torch.randn(4, data.max_len, 3)
    lengths = torch.from_numpy(data.lengths)
    keys = torch.from_numpy(data.key_codes)
    with torch.no_grad():
        a = model(x, lengths, keys)
        b = model(x, lengths, (keys + 1) * (keys > 0))
    assert not torch.allclose(a, b)


def test_training_a_key_model_needs_key_codes():
    data = free_text()
    train_data, val_data, _ = aalto_splits(data, "keep", n_enroll=5)
    model, normalizer, _ = train(train_data, val_data, FAST, EncoderConfig(key_vocab=256), log=lambda s: None)
    assert embed(model, normalizer, val_data).shape == (len(val_data), 64)
    assert KeystrokeDataset(train_data, normalizer).keys is not None
    no_keys = dataclasses.replace(train_data, key_codes=None)
    with pytest.raises(ValueError):
        train(no_keys, val_data, FAST, EncoderConfig(key_vocab=256), log=lambda s: None)
    with pytest.raises(ValueError):
        embed(model, normalizer, dataclasses.replace(val_data, key_codes=None))


# ---- --init fine-tuning -----------------------------------------------------------------


@pytest.fixture(scope="module")
def base_model():
    train_data, val_data, test_data = split_users(make_synthetic_dataset(n_users=16, samples_per_user=30, seed=0), 0.25, 0.25)
    model, normalizer, _ = train(train_data, val_data, FAST, log=lambda s: None)
    return model, normalizer, train_data, val_data, test_data


def test_init_fine_tunes_a_copy(base_model):
    model, normalizer, train_data, val_data, _ = base_model
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tuned, tuned_norm, history = train(train_data, val_data, FAST, init=model, normalizer=normalizer, log=lambda s: None)
    assert len(history) == 1 and tuned is not model
    assert tuned_norm is normalizer
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # init left untouched
    assert not all(torch.equal(before[k], v) for k, v in tuned.state_dict().items())


def test_zero_epochs_keeps_the_init_weights_and_needs_an_init(base_model):
    model, normalizer, train_data, val_data, _ = base_model
    cfg = dataclasses.replace(FAST, epochs=0)
    same, _, history = train(train_data, val_data, cfg, init=model, normalizer=normalizer, log=lambda s: None)
    assert history == []
    np.testing.assert_allclose(embed(same, normalizer, val_data), embed(model, normalizer, val_data), atol=1e-6)
    with pytest.raises(ValueError):
        train(train_data, val_data, cfg, log=lambda s: None)
    with pytest.raises(ValueError):
        train(train_data, val_data, dataclasses.replace(FAST, epochs=-1), init=model, log=lambda s: None)


# ---- checkpoint meta and the embedder ---------------------------------------------------


def _no_enter_checkpoint(base_model, tmp_path) -> Path:
    model, normalizer, *_ = base_model
    path = tmp_path / "no_enter.pt"
    save_checkpoint(path, model, normalizer, threshold=2.0, metrics={}, final_enter=False)
    return path


def test_checkpoint_records_final_enter(base_model, tmp_path):
    model, normalizer, *_ = base_model
    save_checkpoint(tmp_path / "with.pt", model, normalizer, threshold=1.0, metrics={})
    assert load_checkpoint(tmp_path / "with.pt")[2]["final_enter"] is True
    assert load_checkpoint(_no_enter_checkpoint(base_model, tmp_path))[2]["final_enter"] is False
    old = torch.load(tmp_path / "with.pt", weights_only=True)
    del old["final_enter"]  # written before C11: trained on CMU with Return
    torch.save(old, tmp_path / "old.pt")
    assert load_checkpoint(tmp_path / "old.pt")[2]["final_enter"] is True


def test_embedder_drops_the_final_enter_when_the_model_expects_none(base_model, tmp_path):
    model, normalizer, _, _, test_data = base_model
    embedder, _ = TorchEmbedder.from_checkpoint(_no_enter_checkpoint(base_model, tmp_path))
    assert embedder.final_enter is False
    t = test_data.values[0, : test_data.lengths[0]].copy()
    expected = embed(model, normalizer, KeystrokeData.from_arrays([t], ["u"]).drop_final_key())[0]
    np.testing.assert_allclose(embedder(t), expected, atol=1e-6)
    changed = t.copy()
    changed[-1, 0] += 0.3  # the Enter's own hold
    changed[-2, 1:] += 0.4  # and the latency into Enter: both are dropped
    np.testing.assert_allclose(embedder(changed), expected, atol=1e-6)
    with_enter, _ = TorchEmbedder.from_checkpoint(tmp_path / "no_enter.pt")
    with_enter.final_enter = True
    assert not np.allclose(with_enter(changed), expected, atol=1e-4)


def test_embedder_rejects_attempts_too_short_without_enter(base_model, tmp_path):
    embedder, _ = TorchEmbedder.from_checkpoint(_no_enter_checkpoint(base_model, tmp_path))
    with pytest.raises(InvalidTimings):
        embedder(np.array([[0.1, 0.2, 0.1], [0.1, 0.0, 0.0]], dtype=np.float32))


def test_embedder_refuses_a_key_identity_model(tmp_path):
    model = RhythmEncoder(EncoderConfig(key_vocab=256))
    data = free_text(n_users=3, per_user=3)
    from keyauth.data import Normalizer

    save_checkpoint(tmp_path / "keys.pt", model, Normalizer.fit(data), threshold=1.0, metrics={}, final_enter=False)
    with pytest.raises(ValueError):
        TorchEmbedder.from_checkpoint(tmp_path / "keys.pt")


def test_engine_with_a_no_enter_model_enrolls_verifies_and_length_checks(base_model, tmp_path):
    _, _, _, _, test_data = base_model
    embedder, _ = TorchEmbedder.from_checkpoint(_no_enter_checkpoint(base_model, tmp_path))
    engine = KeyAuthEngine(embedder, TemplateStore(), 2.0, EngineConfig(enroll_samples=5))
    rows = np.flatnonzero(test_data.user_ids == test_data.users[0])
    samples = [test_data.values[i, : test_data.lengths[i]].copy() for i in rows[:7]]
    results = [engine.enroll("app", "alice", s) for s in samples[:5]]
    assert results[-1].status == "complete"
    assert engine.store.get_user("app", "alice").n_keys == len(samples[0])  # the full attempt, Enter included
    assert engine.verify("app", "alice", samples[6]).accepted
    assert engine.verify("app", "alice", samples[6][:-1]).reason == "length_mismatch"


# ---- CLI ------------------------------------------------------------------------------


def test_cli_aalto_then_init_fine_tune(tmp_path):
    cache = tmp_path / "aalto.npz"
    free_text().save_npz(cache)
    aalto_out = tmp_path / "aalto.pt"
    small = ["--epochs", "1", "--batches-per-epoch", "2", "--p", "4", "--k", "4"]
    metrics = main(["--data", "aalto", "--aalto-cache", str(cache), "--out", str(aalto_out), *small])
    assert metrics["train_config"]["n_enroll"] == 5
    _, _, meta = load_checkpoint(aalto_out)
    assert meta["final_enter"] is False and meta["per_user_threshold"]["scale"] > 0
    assert json.loads(aalto_out.with_suffix(".json").read_text())["final_enter"] is False

    tuned_out = tmp_path / "tuned.pt"
    main(["--data", "synthetic", "--init", str(aalto_out), "--normalizer", "init", "--out", str(tuned_out), *small])
    tuned, tuned_norm, tuned_meta = load_checkpoint(tuned_out)
    _, aalto_norm, _ = load_checkpoint(aalto_out)
    assert tuned_meta["final_enter"] is False  # inherited, so the synthetic data's last key was dropped too
    np.testing.assert_allclose(tuned_norm.mean, aalto_norm.mean)
    assert tuned_meta["per_user_threshold"] is not None

    calibrated = tmp_path / "calibrated.pt"
    main(["--data", "synthetic", "--init", str(aalto_out), "--epochs", "0", "--out", str(calibrated)])
    same, _, _ = load_checkpoint(calibrated)
    original, _, _ = load_checkpoint(aalto_out)
    assert all(torch.equal(a, b) for a, b in zip(same.state_dict().values(), original.state_dict().values()))


def test_cli_rejects_keys_outside_aalto_and_aalto_on_an_enter_model(tmp_path):
    with pytest.raises(ValueError):
        main(["--data", "synthetic", "--keys", "--out", str(tmp_path / "x.pt")])
    with_enter = tmp_path / "with_enter.pt"
    main(["--data", "synthetic", "--epochs", "1", "--batches-per-epoch", "2", "--out", str(with_enter)])
    cache = tmp_path / "aalto.npz"
    free_text().save_npz(cache)
    with pytest.raises(ValueError):
        main(["--data", "aalto", "--aalto-cache", str(cache), "--init", str(with_enter), "--out", str(tmp_path / "y.pt")])


# ---- compare_models c11 helpers ---------------------------------------------------------


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("compare_models_c11", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aalto_metrics_splits_clean_and_corrected(compare):
    data = free_text(n_users=6, per_user=10)
    z = np.random.default_rng(0).normal(size=(len(data), 8))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    out = compare.aalto_metrics(z, data, n_enroll=5)
    assert out["all"]["n_genuine"] == out["clean"]["n_genuine"] + out["corrected"]["n_genuine"] == 6 * 5
    assert out["all"]["n_impostor"] == out["clean"]["n_impostor"] + out["corrected"]["n_impostor"] == 6 * 5 * 5
    assert out["all"]["eer"] == pytest.approx(out["summary"]["eer"])
    fixed = compare.aalto_metrics(z, data, n_enroll=5, threshold=0.5)
    assert fixed["summary"]["threshold"] == 0.5 and 0.0 <= fixed["clean"]["far"] <= 1.0


def test_cmu_normalizer_modes(compare):
    data = make_synthetic_dataset(n_users=3, samples_per_user=5)
    from keyauth.data import Normalizer

    given = Normalizer(mean=np.zeros(3), std=np.ones(3))
    assert compare.cmu_normalizer("aalto", given, data) is given
    assert not np.allclose(compare.cmu_normalizer("cmu", given, data).mean, 0.0)
    with pytest.raises(ValueError):
        compare.cmu_normalizer("other", given, data)


# ---- C11b: fine-tuning with the final Enter kept from a model trained without it -----------------


def test_cli_keep_enter_from_a_no_enter_checkpoint_works_through_the_embedder(base_model, tmp_path):
    no_enter = _no_enter_checkpoint(base_model, tmp_path)
    small = ["--epochs", "1", "--batches-per-epoch", "2", "--p", "4", "--k", "4"]
    out = tmp_path / "kept.pt"
    main(["--data", "synthetic", "--init", str(no_enter), "--keep-enter", "--out", str(out), *small])
    model, normalizer, meta = load_checkpoint(out)
    assert meta["final_enter"] is True and meta["per_user_threshold"] is not None
    assert json.loads(out.with_suffix(".json").read_text())["final_enter"] is True

    embedder, threshold = TorchEmbedder.from_checkpoint(out)
    assert embedder.final_enter is True
    _, _, _, _, test_data = base_model
    t = test_data.values[0, : test_data.lengths[0]].copy()
    expected = embed(model, normalizer, KeystrokeData.from_arrays([t], ["u"]))[0]  # the full attempt, Enter included
    np.testing.assert_allclose(embedder(t), expected, atol=1e-6)
    changed = t.copy()
    changed[-1, 0] += 0.3  # the Enter's hold now matters
    assert not np.allclose(embedder(changed), expected, atol=1e-4)

    engine = KeyAuthEngine(embedder, TemplateStore(), threshold, EngineConfig(enroll_samples=5))
    rows = np.flatnonzero(test_data.user_ids == test_data.users[0])
    samples = [test_data.values[i, : test_data.lengths[i]].copy() for i in rows[:7]]
    assert [engine.enroll("app", "alice", s) for s in samples[:5]][-1].status == "complete"
    assert engine.verify("app", "alice", samples[6]).reason != "length_mismatch"
    assert engine.verify("app", "alice", samples[6][:-1]).reason == "length_mismatch"


def test_cli_keep_enter_needs_init_and_is_refused_for_aalto(base_model, tmp_path):
    with pytest.raises(ValueError):
        main(["--data", "synthetic", "--keep-enter", "--epochs", "1", "--out", str(tmp_path / "x.pt")])
    cache = tmp_path / "aalto.npz"
    free_text().save_npz(cache)
    no_enter = _no_enter_checkpoint(base_model, tmp_path)
    with pytest.raises(ValueError):  # Aalto has no Enter to keep
        main(["--data", "aalto", "--aalto-cache", str(cache), "--init", str(no_enter), "--keep-enter",
              "--out", str(tmp_path / "y.pt")])
    assert not (tmp_path / "x.pt").exists() and not (tmp_path / "y.pt").exists()


def test_b2_job_keep_enter_writes_a_final_enter_checkpoint(compare, base_model, tmp_path, monkeypatch):
    no_enter = _no_enter_checkpoint(base_model, tmp_path)
    data = make_synthetic_dataset(n_users=16, samples_per_user=30, seed=1)

    def fake_cmu(final_enter: bool, keys: bool = False) -> KeystrokeData:
        if not final_enter:
            return data.drop_final_key()
        return data

    monkeypatch.setattr(compare, "C11_DIR", tmp_path / "c11")
    monkeypatch.setattr(compare, "cmu_data", fake_cmu)
    monkeypatch.setattr(compare, "aalto_ckpt", lambda variant, seed: no_enter)
    job = {"variant": "v", "normalizer": "aalto", "lr": 1e-3, "epochs": 1, "split": 0, "replicate": 0}
    kept = compare.b2_job({**job, "final_enter": True})
    dropped = compare.b2_job(job)
    assert kept["final_enter"] is True and dropped["final_enter"] is False
    kept_path = compare.b2_ckpt("v", 1e-3, 1, 0, 0, True, "aalto")
    dropped_path = compare.b2_ckpt("v", 1e-3, 1, 0, 0)
    assert kept_path.parent.name == "b2e" and "normaalto" in kept_path.name and dropped_path.parent.name == "b2"
    assert load_checkpoint(kept_path)[2]["final_enter"] is True
    assert load_checkpoint(dropped_path)[2]["final_enter"] is False
    # the test-scoring path reads final_enter from the checkpoint and feeds the matching input
    scored = compare.cmu_ckpt_job({"checkpoint": str(kept_path), "variant": "b2e", "split": 0, "replicate": 0})
    assert scored["val"]["eer"] == pytest.approx(kept["val"]["eer"]) and 0.0 <= scored["test"]["eer"] <= 1.0
    assert TorchEmbedder.from_checkpoint(kept_path)[0].final_enter is True
