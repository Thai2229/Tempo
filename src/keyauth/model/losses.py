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
