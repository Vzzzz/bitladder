"""DS-CNN for keyword spotting (Hello Edge, Zhang et al. 2017) from quant-aware layers.

Input: (B, 1, frames, n_mels) log-mel. DSCNN(channels=64, blocks=4) is roughly the
"DS-CNN S" size class (~20-25k parameters).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bitladder.quant.layers import QuantConv2d, QuantLinear  # noqa: E402


class DSBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = QuantConv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.pw = QuantConv2d(ch, ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class DSCNN(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4, num_classes: int = 12):
        super().__init__()
        self.stem = QuantConv2d(1, channels, (10, 4), stride=(2, 2), padding=(5, 1), bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[DSBlock(channels) for _ in range(blocks)])
        self.fc = QuantLinear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.stem(x)))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
