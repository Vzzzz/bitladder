"""Fixed unstructured magnitude pruning for quant-aware conv / linear layers."""

from __future__ import annotations

import fnmatch
from typing import Dict, List, Union

import torch
import torch.nn as nn

from .quant.layers import quant_layers


def magnitude_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep the largest weights; exactly round(sparsity * numel) entries are masked out."""
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    mask = torch.ones(weight.numel(), dtype=torch.bool, device=weight.device)
    k = int(round(sparsity * weight.numel()))
    if k:
        idx = torch.argsort(weight.detach().abs().flatten())[:k]
        mask[idx] = False
    return mask.reshape_as(weight)


def apply_magnitude_pruning(model: nn.Module, sparsity: float,
                            layers: Union[str, List[str]] = "blocks.*.pw") -> List[Dict[str, object]]:
    """Set a fixed per-weight mask on matching QuantConv / QuantLinear layers."""
    patterns = [p.strip() for p in layers.split(",") if p.strip()] if isinstance(layers, str) else list(layers)
    if sparsity == 0.0:
        return []
    report = []
    for name, layer in quant_layers(model):
        if not any(fnmatch.fnmatchcase(name, p) for p in patterns):
            continue
        mask = magnitude_mask(layer.weight, sparsity)
        layer.set_weight_mask(mask)
        report.append({"name": name, "requested_sparsity": sparsity,
                       "sparsity": 1.0 - float(mask.float().mean()), "nonzero_params": int(mask.sum())})
    if not report:
        raise ValueError(f"no quant layers match pruning patterns {patterns}")
    return report


def pruning_report(model: nn.Module) -> List[Dict[str, object]]:
    """Actual masks in a model, independent of the requested pruning fraction."""
    out = []
    for name, layer in quant_layers(model):
        mask = getattr(layer, "weight_mask", None)
        if mask is not None:
            out.append({"name": name, "sparsity": 1.0 - float(mask.float().mean()),
                        "nonzero_params": int(mask.sum())})
    return out


def load_state_with_masks(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = True) -> nn.Module:
    """Restore checkpoints with mask buffers without requiring masks in old checkpoints."""
    for name, layer in quant_layers(model):
        key = f"{name}.weight_mask"
        if key in state:
            layer.set_weight_mask(state[key])
    model.load_state_dict(state, strict=strict)
    return model
