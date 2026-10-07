"""MACs and BOPs via forward hooks.

BOPs (bit operations) = MACs * weight_bits * activation_bits, the usual proxy for
mixed-precision cost (e.g. HAWQ, Mixed-precision DNNs). Layers without an activation
quantizer count 32 activation bits; fp16 storage counts 16 weight bits.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _macs_conv(m: nn.Conv2d, out: torch.Tensor) -> int:
    kh, kw = m.kernel_size
    per_output = (m.in_channels // m.groups) * kh * kw
    return int(out.numel() // out.shape[0] * per_output)


def _macs_linear(m: nn.Linear, out: torch.Tensor) -> int:
    return int(m.in_features * out.numel() // out.shape[0])


def count_macs(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, int]:
    """Per-layer MACs for a single sample. ``input_shape`` excludes the batch dim."""
    macs: Dict[str, int] = {}
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(lambda mod, inp, out, n=name: macs.__setitem__(n, _macs_conv(mod, out))))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lambda mod, inp, out, n=name: macs.__setitem__(n, _macs_linear(mod, out))))
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
