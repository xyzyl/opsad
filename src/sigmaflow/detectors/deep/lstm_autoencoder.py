"""LSTM autoencoder detector."""

from __future__ import annotations

import torch
from torch import nn

from ._base import DeepReconstructionDetector

__all__ = ["LSTMAutoencoderDetector"]


class _LSTMAE(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):  # (batch, window)
        seq = x.unsqueeze(-1)
        _, (h, _) = self.encoder(seq)
        latent = h[-1]  # (batch, hidden)
        repeated = latent.unsqueeze(1).repeat(1, seq.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.head(decoded).squeeze(-1)


class LSTMAutoencoderDetector(DeepReconstructionDetector):
    """LSTM encoder -> latent -> LSTM decoder (``sigmaflow[deep]``).

    Captures temporal ordering: an anomaly that is a specific *sequence*
    of changes (e.g. a disruption precursor's growth pattern) rather
    than an unusual value. Slower to train than the feedforward
    autoencoder; prefer it when order matters.
    """

    name = "lstm_autoencoder"

    def __init__(
        self,
        lstm_hidden_size: int = 64,
        num_layers: int = 1,
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
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.num_layers = int(num_layers)

    def _build_model(self):
        torch.manual_seed(self.random_state or 0)
        return _LSTMAE(self.lstm_hidden_size, self.num_layers)
