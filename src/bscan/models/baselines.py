"""Same-protocol baselines.

LCNN: a light convolutional network with max-feature-map (MFM) activations, following the
LCNN design used in anti-spoofing (Lavrentyeva et al., Interspeech 2019; the LFCC-LCNN baseline
family of ASVspoof 2019/2021).  It is adapted here to the same [3, F, T] cepstral input as BSCAN
(static, delta, delta-delta) and to global mean+std temporal pooling; it is therefore an
"LCNN-style" baseline, not a byte-exact re-implementation of any challenge system.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class MFM(nn.Module):
    """Max-feature-map: split channels (dim 1) in two halves and keep the element-wise max."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


def _conv(cin: int, cout2: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout2, k, padding=k // 2), MFM())


class LCNN(nn.Module):
    def __init__(self, feature: str = "lfcc", in_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.feature = feature
        self.body = nn.Sequential(
            _conv(in_channels, 64, 5), nn.MaxPool2d(2),
            _conv(32, 64, 1), nn.BatchNorm2d(32), _conv(32, 96, 3), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            _conv(48, 96, 1), nn.BatchNorm2d(48), _conv(48, 128, 3), nn.MaxPool2d(2),
            _conv(64, 128, 1), nn.BatchNorm2d(64), _conv(64, 64, 3), nn.BatchNorm2d(32),
            _conv(32, 64, 1), nn.BatchNorm2d(32), _conv(32, 64, 3), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(64, 160), MFMLinear(), nn.BatchNorm1d(80), nn.Linear(80, 1)
        )

    def forward(self, feats: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = self.body(feats[self.feature]).mean(dim=2)       # [B, 32, T'] (average over frequency)
        x = torch.cat([x.mean(dim=-1), x.std(dim=-1)], dim=1)  # [B, 64]
        return self.head(x).squeeze(1)


class MFMLinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)
