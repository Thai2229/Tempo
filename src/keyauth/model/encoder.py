"""Typing-rhythm encoder: (attempt timings) -> L2-normalized embedding.

Architecture, for a batch x of shape (B, L, 3) with per-attempt lengths:

    [hold, dd, ud] + has_next flag          (B, L, 4)
      -> Conv1d(k=3) + GELU                 local digraph/trigraph patterns
      -> bidirectional GRU (packed)         whole-phrase rhythm, any length
      -> masked mean || masked max pooling  fixed-size summary
      -> Linear -> L2 normalize             embedding on the unit sphere

Distances between embeddings measure "same typist or not". Because the
network learns a general typing-style space, new users can enroll without
retraining.

Padding never affects the output: padded steps are zeroed after the conv,
skipped by the packed GRU, and excluded from pooling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from keyauth.config import N_CHANNELS


@dataclass
class EncoderConfig:
    in_channels: int = N_CHANNELS
    conv_channels: int = 64
    hidden_size: int = 64
    num_layers: int = 1
    embed_dim: int = 64
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        return cls(**d)


class RhythmEncoder(nn.Module):
    def __init__(self, config: EncoderConfig | None = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()
        self.config = config
        c = config

        # +1 input channel: "has_next" flag, 0 on the last keystroke where dd/ud are undefined
        self.conv = nn.Conv1d(c.in_channels + 1, c.conv_channels, kernel_size=3, padding=1)
        gru_dropout = 0.0
        if c.num_layers > 1:
            gru_dropout = c.dropout
        self.gru = nn.GRU(
            input_size=c.conv_channels,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(c.dropout)
        self.head = nn.Linear(4 * c.hidden_size, c.embed_dim)  # (mean || max) x 2 directions

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 3) normalized timings, lengths: (B,) -> (B, embed_dim) unit vectors."""
        if x.ndim != 3 or x.shape[2] != self.config.in_channels:
            raise ValueError(f"expected x of shape (B, L, {self.config.in_channels}), got {tuple(x.shape)}")
        if lengths.min() < 2 or lengths.max() > x.shape[1]:
            raise ValueError("lengths must be in [2, L]")

        batch, max_len, _ = x.shape
        pos = torch.arange(max_len, device=x.device)[None, :]
        lengths_d = lengths.to(x.device)
        valid = (pos < lengths_d[:, None]).unsqueeze(-1).to(x.dtype)  # (B, L, 1)
        has_next = (pos < lengths_d[:, None] - 1).unsqueeze(-1).to(x.dtype)

        h = torch.cat([x * valid, has_next], dim=-1)  # (B, L, 4)
        h = F.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2)  # (B, L, C)
        h = h * valid

        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)  # (B, L, 2H)

        mean = (out * valid).sum(dim=1) / lengths_d[:, None].to(out.dtype)
        masked = out.masked_fill(valid == 0, float("-inf"))
        mx = masked.max(dim=1).values

        z = self.head(self.dropout(torch.cat([mean, mx], dim=-1)))
        return F.normalize(z, dim=-1)
