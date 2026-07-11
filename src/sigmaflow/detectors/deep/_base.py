"""Shared machinery for reconstruction-based deep detectors.

All three deep detectors follow the same recipe: learn to reconstruct
normal signal windows (with light denoising so the model can't learn the
identity map), then score each timestep by the reconstruction error of
the window centered on it. Anomalies reconstruct poorly because the
model never saw their shapes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame

__all__ = ["DeepReconstructionDetector"]


class DeepReconstructionDetector(BaseDetector):
    """Base class: windowing, normalization, training loop, scoring."""

    def __init__(
        self,
        window_size: int = 64,
        epochs: int = 15,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        threshold_percentile: float = 99.0,
        noise_std: float = 0.05,
        random_state: int | None = 0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.threshold_percentile = float(threshold_percentile)
        self.noise_std = float(noise_std)
        self.random_state = random_state
        self._models: dict[str, nn.Module] = {}
        self._norms: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------ subclass
    def _build_model(self) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------ helpers
    def _normalize(self, x: np.ndarray, channel: str, fit: bool) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0)
        if fit:
            mean, std = float(np.mean(x)), float(np.std(x)) or 1.0
            self._norms[channel] = (mean, std)
        mean, std = self._norms.get(channel, (0.0, 1.0))
        return (x - mean) / std

    def _train_windows(self, x: np.ndarray) -> torch.Tensor:
        w = self.window_size
        if len(x) < w:
            x = np.pad(x, (0, w - len(x)), mode="edge")
        stride = max(1, w // 4)
        wins = np.lib.stride_tricks.sliding_window_view(x, w)[::stride]
        return torch.from_numpy(np.ascontiguousarray(wins, dtype=np.float32))

    def _score_windows(self, x: np.ndarray) -> torch.Tensor:
        w = self.window_size
        half = w // 2
        padded = np.pad(x, (half, w - half - 1), mode="edge")
        wins = np.lib.stride_tricks.sliding_window_view(padded, w)
        return torch.from_numpy(np.ascontiguousarray(wins, dtype=np.float32))

    # ------------------------------------------------------------ fit/score
    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        xn = self._normalize(x, channel, fit=True)
        windows = self._train_windows(xn)
        model = self._build_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        n = len(windows)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch = windows[perm[start : start + self.batch_size]]
                noisy = batch + self.noise_std * torch.randn_like(batch)
                optimizer.zero_grad()
                loss = loss_fn(model(noisy), batch)
                loss.backward()
                optimizer.step()
        model.eval()
        self._models[channel] = model

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        if channel not in self._models:
            self._fit_channel(x, channel, sf)
        xn = self._normalize(x, channel, fit=False)
        windows = self._score_windows(xn)
        model = self._models[channel]
        errors = []
        with torch.no_grad():
            for start in range(0, len(windows), 512):
                batch = windows[start : start + 512]
                recon = model(batch)
                errors.append(((recon - batch) ** 2).mean(dim=1).numpy())
        return np.concatenate(errors)[: len(x)]

    def _auto_threshold(self, scores: np.ndarray) -> float:
        return float(np.percentile(scores, self.threshold_percentile))
