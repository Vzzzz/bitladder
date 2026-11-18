"""HOPE-style channel capacity from BatchNorm statistics and outgoing weight norms."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..quant.fold import find_conv_bn_pairs
from ..quant.layers import quant_layers


@torch.no_grad()
def hope_capacity(model: nn.Module, example: torch.Tensor) -> List[Dict[str, object]]:
    """Per-channel E[ReLU(y)^2] * ||outgoing_slice||^2 for sequential Conv -> BN -> ReLU layers."""
    modules = dict(model.named_modules())
    pairs = dict(find_conv_bn_pairs(model, example))
    layers = list(quant_layers(model))
    out = []
    for idx, (name, layer) in enumerate(layers[:-1]):
        bn_name = pairs.get(name)
        if bn_name is None or not isinstance(modules[bn_name], nn.modules.batchnorm._BatchNorm):
            continue
        bn = modules[bn_name]
        beta = bn.bias.detach().float()
        gamma = bn.weight.detach().float()
        sigma = gamma.abs().clamp_min(1e-12)
        z = beta / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
        second_moment = (beta.square() + gamma.square()) * cdf + beta * sigma * pdf

        next_layer = layers[idx + 1][1]
        w = next_layer.masked_weight().detach().float()
        depthwise = (isinstance(next_layer, nn.modules.conv._ConvNd)
                     and next_layer.groups == next_layer.in_channels and next_layer.groups > 1)
        norms = torch.stack([(w[c] if depthwise else w[:, c]).norm() for c in range(layer.weight.shape[0])])
        capacity = second_moment.cpu().numpy() * np.square(norms.cpu().numpy())
        out.append({"name": name, "capacity": capacity.tolist(), "mean": float(capacity.mean()),
                    "median": float(np.median(capacity)), "min": float(capacity.min()), "max": float(capacity.max())})
    return out


def _ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties and no scipy dependency."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def hope_spearman(rep: Dict[str, object]) -> List[Dict[str, object]]:
    """Compare mean layer capacity with each channel-pruning sensitivity column."""
    if "hope" not in rep or "sensitivity" not in rep:
        return []
    sens = rep["sensitivity"]
    means = {r["name"]: r["mean"] for r in rep["hope"]}
    indices = [i for i, name in enumerate(sens["layers"]) if name in means]
    out = []
    for j, probe in enumerate(sens["probes"]):
        if not probe.startswith("prune"):
            continue
        names = [sens["layers"][i] for i in indices]
        capacity = [means[name] for name in names]
        drop = [sens["drop"][i][j] for i in indices]
        out.append({"probe": probe, "spearman": spearman(capacity, drop),
                    "layers": [{"name": name, "mean_capacity": cap, "drop": dr}
                               for name, cap, dr in zip(names, capacity, drop)]})
    return out
