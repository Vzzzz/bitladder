"""Tiny MNIST models built from quant-aware layers.

TinyCNN(width=8) has about 6k parameters: roughly 1.2 KB at 1.58 bits per weight,
24 KB in fp32. Width scales the channel count; use it to hit size budgets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant.layers import QuantConv2d, QuantLinear


class TinyCNN(nn.Module):
    def __init__(self, width: int = 8, num_classes: int = 10, bn: bool = True):
        super().__init__()
        c = width
        bias = not bn
        self.conv1 = QuantConv2d(1, c, 3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(c) if bn else nn.Identity()
        self.conv2 = QuantConv2d(c, 2 * c, 3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(2 * c) if bn else nn.Identity()
        self.conv3 = QuantConv2d(2 * c, 4 * c, 3, padding=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(4 * c) if bn else nn.Identity()
        self.fc = QuantLinear(4 * c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # 14x14
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)   # 7x7
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 32, num_classes: int = 10):
        super().__init__()
        self.fc1 = QuantLinear(28 * 28, hidden)
        self.fc2 = QuantLinear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


def build_mnist_model(kind: str = "cnn", width: int = 8) -> nn.Module:
    if kind == "cnn":
        return TinyCNN(width=width)
    if kind == "mlp":
        return TinyMLP(hidden=width)
    raise ValueError(f"unknown model kind {kind!r}")
