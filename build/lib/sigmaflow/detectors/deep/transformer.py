"""Transformer encoder reconstruction detector."""

from __future__ import annotations

import math

import torch
from torch import nn

from ._base import DeepReconstructionDetector

__all__ = ["TransformerDetector"]


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (batch, seq, d_model)
        return x + self.pe[: x.shape[1]]


class _TransformerAE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, window: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = _PositionalEncoding(d_model, window)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # (batch, window)
        h = self.pos(self.proj(x.unsqueeze(-1)))
        return self.head(self.encoder(h)).squeeze(-1)


class TransformerDetector(DeepReconstructionDetector):
    """Transformer encoder over signal windows (``sigmaflow[deep]``).

    Self-attention captures long-range dependencies inside each window,
    where LSTMs struggle with vanishing gradients. The heaviest of the
    deep detectors — reach for it on long, structurally complex signals.
    """

    name = "transformer"

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        window_size: int = 64,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        threshold_percentile: float = 99.0,
        noise_std: float = 0.05,
        random_state: int | None = 0,
    ):
        super().__init__(window_size=window_size, epochs=epochs,
                         learning_rate=learning_rate, batch_size=batch_size,
                         threshold_percentile=threshold_percentile,
                         noise_std=noise_std, random_state=random_state)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)

    def _build_model(self):
        torch.manual_seed(self.random_state or 0)
        return _TransformerAE(self.d_model, self.n_heads, self.n_layers,
                              self.window_size)
