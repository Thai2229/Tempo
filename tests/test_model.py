import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from keyauth.data import Normalizer, make_synthetic_dataset  # noqa: E402
from keyauth.data.torch_dataset import KeystrokeDataset, PKBatchSampler  # noqa: E402
from keyauth.model import (  # noqa: E402
    EncoderConfig,
    RhythmEncoder,
    batch_hard_triplet_loss,
    pairwise_distances,
    supervised_contrastive_loss,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


# ---- encoder ------------------------------------------------------------


def test_output_shape_and_unit_norm():
    model = RhythmEncoder()
    x = torch.randn(5, 11, 3)
    lengths = torch.tensor([11, 11, 5, 2, 8])
    z = model(x, lengths)
    assert z.shape == (5, 64)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(5))


def test_padding_does_not_change_embedding():
    model = RhythmEncoder().eval()
    x = torch.randn(1, 6, 3)
    z1 = model(x, torch.tensor([6]))
    # garbage in the padded steps must be ignored
    x_pad = torch.cat([x, 100 * torch.randn(1, 4, 3)], dim=1)
    z2 = model(x_pad, torch.tensor([6]))
    torch.testing.assert_close(z1, z2, atol=1e-5, rtol=1e-5)
    # and so must the other attempts in the batch
    batch = torch.cat([x_pad, torch.randn(1, 10, 3)], dim=0)
    z3 = model(batch, torch.tensor([6, 10]))[:1]
    torch.testing.assert_close(z1, z3, atol=1e-5, rtol=1e-5)


def test_rejects_bad_input():
    model = RhythmEncoder()
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 3), torch.tensor([5, 1]))
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 3), torch.tensor([5, 6]))
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 4), torch.tensor([5, 5]))


def test_gradients_reach_every_parameter():
    model = RhythmEncoder()
    z = model(torch.randn(4, 7, 3), torch.tensor([7, 6, 3, 2]))
    z.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_config_roundtrip_and_custom_sizes():
    cfg = EncoderConfig(hidden_size=16, num_layers=2, embed_dim=8)
    assert EncoderConfig.from_dict(cfg.to_dict()) == cfg
    z = RhythmEncoder(cfg)(torch.randn(3, 5, 3), torch.tensor([5, 4, 3]))
    assert z.shape == (3, 8)


# ---- loss ---------------------------------------------------------------


def test_pairwise_distances():
    emb = torch.tensor([[0.0, 0.0], [3.0, 4.0]], requires_grad=True)
    d = pairwise_distances(emb)
    torch.testing.assert_close(d, torch.tensor([[0.0, 5.0], [5.0, 0.0]]))
    d.sum().backward()
    assert torch.isfinite(emb.grad).all()


def test_loss_zero_when_users_separated():
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    stats = batch_hard_triplet_loss(emb, torch.tensor([0, 0, 1, 1]), margin=0.2)
    assert stats.loss.item() == pytest.approx(0.0)
    assert stats.active_frac == 0.0


def test_loss_equals_margin_when_collapsed():
    emb = torch.tensor([[1.0, 0.0]] * 4)
    stats = batch_hard_triplet_loss(emb, torch.tensor([0, 0, 1, 1]), margin=0.2)
    assert stats.loss.item() == pytest.approx(0.2)


def test_loss_picks_hardest_pairs():
    # 1-D points; per-anchor losses worked out by hand: 0.7, 0.7, 2.2, 0.7
    emb = torch.tensor([[0.0], [1.0], [0.5], [3.0]])
    stats = batch_hard_triplet_loss(emb, torch.tensor([0, 0, 1, 1]), margin=0.2)
    assert stats.loss.item() == pytest.approx(1.075, abs=1e-5)


def test_anchor_without_positive_is_skipped():
    emb = torch.tensor([[0.0], [1.0], [5.0]])
    stats = batch_hard_triplet_loss(emb, torch.tensor([0, 0, 1]), margin=0.2)
    assert stats.loss.item() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        batch_hard_triplet_loss(emb, torch.tensor([0, 1, 2]))


# ---- end to end: the model can learn typists ----------------------------


def test_training_reduces_loss_on_synthetic_typists():
    data = make_synthetic_dataset(n_users=8, samples_per_user=16, seed=0)
    ds = KeystrokeDataset(data, Normalizer.fit(data))
    model = RhythmEncoder(EncoderConfig(conv_channels=32, hidden_size=32, embed_dim=32))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def full_loss():
        model.eval()
        with torch.no_grad():
            return batch_hard_triplet_loss(model(ds.x, ds.lengths), ds.labels).loss.item()

    before = full_loss()
    sampler = PKBatchSampler(ds.labels, p=4, k=4, batches_per_epoch=150, seed=0)
    model.train()
    for batch in sampler:
        stats = batch_hard_triplet_loss(model(ds.x[batch], ds.lengths[batch]), ds.labels[batch])
        opt.zero_grad()
        stats.loss.backward()
        opt.step()
    after = full_loss()
    assert after < before


# ---- supervised contrastive loss -----------------------------------------


def test_supcon_at_collapse_equals_log_of_other_attempts():
    emb = torch.zeros(8, 4)
    emb[:, 0] = 1.0  # every attempt is the same point
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    assert supervised_contrastive_loss(emb, labels).loss.item() == pytest.approx(math.log(7), abs=1e-5)


def test_supcon_prefers_separated_users():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert supervised_contrastive_loss(separated, labels).loss < supervised_contrastive_loss(mixed, labels).loss


def test_supcon_gradient_pushes_a_collapsed_batch_apart():
    emb = F.normalize(torch.eye(4)[0].repeat(8, 1) + 1e-3 * torch.randn(8, 4, generator=torch.Generator().manual_seed(0)), dim=1)
    emb.requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    supervised_contrastive_loss(emb, labels).loss.backward()
    assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_supcon_rejects_bad_temperature(temperature):
    with pytest.raises(ValueError):
        supervised_contrastive_loss(torch.eye(4), torch.tensor([0, 0, 1, 1]), temperature)


def test_supcon_rejects_length_mismatch():
    with pytest.raises(ValueError):
        supervised_contrastive_loss(torch.eye(4), torch.tensor([0, 0, 1]))


@pytest.mark.parametrize("labels", [[0, 1, 2, 3], [0, 0, 0, 0]])
def test_supcon_rejects_batches_without_pairs(labels):
    with pytest.raises(ValueError):
        supervised_contrastive_loss(torch.eye(4), torch.tensor(labels))
