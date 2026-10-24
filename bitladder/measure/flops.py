"""MACs and BOPs via forward hooks.

BOPs (bit operations) = MACs * weight_bits * activation_bits, the usual proxy for
mixed-precision cost (e.g. HAWQ, Mixed-precision DNNs). Layers without an activation
quantizer count 32 activation bits; fp16 storage counts 16 weight bits.

Covered: nn.Conv1d / nn.Conv2d (and the Quant* subclasses), nn.Linear, nn.GRU
(3 gates * (in*hidden + hidden*hidden) per step, per direction, per layer).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _macs_conv(m: nn.modules.conv._ConvNd, out: torch.Tensor) -> int:
    per_output = (m.in_channels // m.groups) * math.prod(m.kernel_size)
    return int(out.numel() // out.shape[0] * per_output)


def _macs_linear(m: nn.Linear, out: torch.Tensor) -> int:
    return int(m.in_features * out.numel() // out.shape[0])


def _macs_gru(m: nn.GRU, inp: Tuple[torch.Tensor, ...], out) -> int:
    x = inp[0]
    steps = x.shape[1] if m.batch_first else x.shape[0]
    dirs = 2 if m.bidirectional else 1
    total = 0
    for layer in range(m.num_layers):
        in_size = m.input_size if layer == 0 else m.hidden_size * dirs
        total += dirs * 3 * (in_size * m.hidden_size + m.hidden_size * m.hidden_size)
    return int(total * steps)


def count_macs(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, int]:
    """Per-layer MACs for a single sample. ``input_shape`` excludes the batch dim."""
    macs: Dict[str, int] = {}
    handles = []

    def add(name, fn):
        return lambda mod, inp, out: macs.__setitem__(name, fn(mod, inp, out))

    for name, m in model.named_modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            handles.append(m.register_forward_hook(add(name, lambda mod, inp, out: _macs_conv(mod, out))))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(add(name, lambda mod, inp, out: _macs_linear(mod, out))))
        elif isinstance(m, nn.GRU):
            handles.append(m.register_forward_hook(add(name, _macs_gru)))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)
    for h in handles:
        h.remove()
    return macs


def count_bops(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    macs = count_macs(model, input_shape)
    modules = dict(model.named_modules())
    bops: Dict[str, float] = {}
    for name, n_macs in macs.items():
        m = modules[name]
        wbits = getattr(m, "wbits", 32.0)
        abits = getattr(m, "abits", 32.0)
        bops[name] = n_macs * wbits * abits
    return bops
