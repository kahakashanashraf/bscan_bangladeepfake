"""BSCAN and its controlled variants.

The default configuration reproduces the architecture of the original notebook
(code/banglafkpart.ipynb, cell 53): per branch two SE-residual blocks (3->32->hidden),
max-pool, frequency average-pool, selective temporal attention pooling; branch embeddings
are concatenated and classified by Linear(.,128)-ReLU-Dropout-Linear(128,1).
With branches=("mel", "lfcc"), hidden_dim=64 it has 162,819 parameters.

Switches used by the ablation study (all variants are trained from scratch):
    branches          any non-empty subset/sequence of feature names, e.g. ("mel",), ("lfcc",),
                      ("mfcc",), ("mel", "lfcc"), ("mel", "mfcc")
    use_se            squeeze-and-excitation channel attention inside each residual block
    use_temporal_attn attention pooling over time (otherwise temporal mean pooling)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Squeeze-and-excitation channel attention (reduction 16, at least 4 hidden units)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class SEResBlock(nn.Module):
    """Residual block: conv3x3-BN-ReLU-conv3x3-BN-[SE] + (1x1 projection) shortcut, ReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEModule(out_channels) if use_se else nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + identity)


class TemporalAttentionPool(nn.Module):
    """Softmax attention over time frames: w = softmax(Conv1d-ReLU-Conv1d(x)), f = sum_t w_t x_t."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1), nn.ReLU(inplace=True), nn.Conv1d(dim, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        w = torch.softmax(self.attn(x), dim=-1)  # [B, 1, T]
        f = (x * w).sum(dim=-1)  # [B, C]
        return (f, w) if return_weights else f


class ResidualBranch(nn.Module):
    """One feature branch: [B, 3, F, T] -> [B, hidden_dim]."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, use_se: bool = True,
                 use_temporal_attn: bool = True):
        super().__init__()
        self.layer1 = SEResBlock(in_channels, 32, use_se=use_se)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.layer2 = SEResBlock(32, hidden_dim, use_se=use_se)
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.attn = TemporalAttentionPool(hidden_dim) if use_temporal_attn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x)).squeeze(2)  # [B, C, T']
        return self.attn(x) if self.attn is not None else x.mean(dim=-1)


class BSCAN(nn.Module):
    """Multi-branch spectro-cepstral residual network with late (concatenation) fusion."""

    def __init__(self, branches: Sequence[str] = ("mel", "lfcc"), hidden_dim: int = 64,
                 use_se: bool = True, use_temporal_attn: bool = True, dropout: float = 0.3,
                 in_channels: int = 3):
        super().__init__()
        if len(branches) == 0:
            raise ValueError("at least one branch is required")
        self.branch_names = tuple(branches)
        self.branches = nn.ModuleDict({
            name: ResidualBranch(in_channels, hidden_dim, use_se, use_temporal_attn) for name in self.branch_names
        })
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(self.branch_names), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def embed(self, feats: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.branches[n](feats[n]) for n in self.branch_names], dim=1)

    def forward(self, feats: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """feats maps branch name -> [B, 3, F, T]; returns logits [B] (positive = spoof)."""
        return self.classifier(self.embed(feats)).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
