"""Metric-learning loss: batch-hard triplet loss (Hermans et al., 2017).

For every anchor in a P x K batch:
    hardest positive = the same user's attempt that is FARTHEST away
    hardest negative = another user's attempt that is CLOSEST
    loss = relu(d(anchor, pos) - d(anchor, neg) + margin)

Pushing the worst cases apart is what sets the accept/reject threshold later,
so this trains directly for verification rather than classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def pairwise_distances(emb: torch.Tensor) -> torch.Tensor:
    """(B, D) -> (B, B) Euclidean distances. Numerically safe at 0."""
    sq = (emb * emb).sum(dim=1)
    d2 = sq[:, None] - 2.0 * emb @ emb.T + sq[None, :]
    d2 = d2.clamp_min(0.0)
    # sqrt has an infinite gradient at 0; add eps only where needed
    zero = d2 == 0
    d = torch.sqrt(d2 + zero.to(d2.dtype) * 1e-12)
    return d * (~zero).to(d.dtype)


@dataclass
class TripletStats:
    loss: torch.Tensor
    active_frac: float  # anchors still violating the margin
    mean_pos: float  # mean hardest-positive distance
    mean_neg: float  # mean hardest-negative distance


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> TripletStats:
    if emb.shape[0] != labels.shape[0]:
        raise ValueError("emb and labels length mismatch")
    dist = pairwise_distances(emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    # anchors need at least one positive and one negative
    usable = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not usable.any():
        raise ValueError("batch needs >=2 users with >=2 attempts each")

    hardest_pos = (dist * pos_mask.to(dist.dtype)).max(dim=1).values
    hardest_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    per_anchor = torch.relu(hardest_pos - hardest_neg + margin)[usable]
    return TripletStats(
        loss=per_anchor.mean(),
        active_frac=float((per_anchor > 0).float().mean()),
        mean_pos=float(hardest_pos[usable].mean()),
        mean_neg=float(hardest_neg[usable].mean()),
    )


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> TripletStats:
    """Supervised contrastive loss (Khosla et al., 2020) on unit embeddings.

    Each anchor must rank all of its positives above every other attempt in
    the batch. Unlike a hinge on distances, this cannot be satisfied by
    mapping everyone to one point: its value there is log(batch size - 1).
    """
    if emb.shape[0] != labels.shape[0] or temperature <= 0.0:
        raise ValueError(f"need matching emb and labels lengths and temperature > 0, got temperature {temperature}")
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos_mask = same & ~eye
    usable = pos_mask.any(dim=1) & (~same).any(dim=1)
    if not usable.any():
        raise ValueError("batch needs >=2 users with >=2 attempts each")

    logits = (emb @ emb.T / temperature).masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob.masked_fill(~pos_mask, 0.0)).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)

    dist = pairwise_distances(emb.detach())
    return TripletStats(
        loss=per_anchor[usable].mean(),
        active_frac=1.0,
        mean_pos=float((dist * pos_mask.to(dist.dtype)).max(dim=1).values[usable].mean()),
        mean_neg=float(dist.masked_fill(same, float("inf")).min(dim=1).values[usable].mean()),
    )
