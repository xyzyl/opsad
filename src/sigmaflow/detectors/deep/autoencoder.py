"""Feedforward and Conv1D autoencoder detectors."""

from __future__ import annotations

from torch import nn

from ._base import DeepReconstructionDetector

__all__ = ["AutoencoderDetector"]


class _Conv1dAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 1, 5, padding=2),
        )

    def forward(self, x):  # (batch, window)
        return self.net(x.unsqueeze(1)).squeeze(1)


class AutoencoderDetector(DeepReconstructionDetector):
    """Autoencoder reconstruction detector (``pip install sigmaflow[deep]``).

    Trains a denoising autoencoder to reconstruct normal signal windows;
    the anomaly score is the reconstruction error. Architectures:

    - ``fc``: window -> 128 -> 64 -> latent -> 64 -> 128 -> window.
    - ``conv1d``: stacked 1-D convolutions — better at local temporal
      patterns.

    Use when normal behavior has complex nonlinear structure that
    statistical detectors can't capture. Fit on normal data.
    """

    name = "autoencoder"

    def __init__(
        self,
        architecture: str = "fc",
        latent_dim: int = 16,
        window_size: int = 64,
        epochs: int = 15,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        threshold_percentile: float = 99.0,
        noise_std: float = 0.05,
        random_state: int | None = 0,
    ):
        if architecture not in ("fc", "conv1d"):
            raise ValueError("architecture must be 'fc' or 'conv1d'")
        super().__init__(window_size=window_size, epochs=epochs,
                         learning_rate=learning_rate, batch_size=batch_size,
                         threshold_percentile=threshold_percentile,
                         noise_std=noise_std, random_state=random_state)
        self.architecture = architecture
        self.latent_dim = int(latent_dim)

    def _build_model(self):
        if self.architecture == "conv1d":
            return _Conv1dAE()
        w, z = self.window_size, self.latent_dim
        return nn.Sequential(
            nn.Linear(w, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, z), nn.ReLU(),
            nn.Linear(z, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, w),
        )
