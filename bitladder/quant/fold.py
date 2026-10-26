"""Fold BatchNorm into the preceding quant conv / linear layer (for export and for
"fold-then-quantize" experiments).

y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
  = conv'(x) + b'   with   w' = w * gamma / sqrt(var + eps) (per output channel),
                            b' = beta + (b - mean) * gamma / sqrt(var + eps)

Works for any model where a ``nn.BatchNorm1d/2d`` is applied directly to the output of a
``QuantConv2d`` / ``QuantConv1d`` / ``QuantLinear`` and both are attributes of the same
parent module with the conv registered before the norm (DS-CNN, TinyCNN, ExpandBlock all
qualify). Pairs are detected by tracing a forward pass with hooks, so unusual wiring is
handled as long as the norm's input is exactly the conv's output.

The quantizers stay attached: after folding they see the folded weights, which is the
"fold before QAT" variant. Call on a model copy; returns the same object.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .layers import QUANT_LAYER_TYPES

_BN = (nn.BatchNorm1d, nn.BatchNorm2d)


@torch.no_grad()
def find_conv_bn_pairs(model: nn.Module, example: torch.Tensor) -> List[Tuple[str, str]]:
    """(conv_name, bn_name) pairs where the BN input is exactly the conv output."""
    outputs: Dict[int, str] = {}
    pairs: List[Tuple[str, str]] = []
    handles = []

    def conv_hook(name):
        def fn(mod, inp, out):
            outputs[id(out)] = name
        return fn

    def bn_hook(name):
        def fn(mod, inp, out):
            if id(inp[0]) in outputs:
                pairs.append((outputs[id(inp[0])], name))
        return fn

    for name, m in model.named_modules():
        if isinstance(m, QUANT_LAYER_TYPES):
            handles.append(m.register_forward_hook(conv_hook(name)))
        elif isinstance(m, _BN):
            handles.append(m.register_forward_hook(bn_hook(name)))
    was_training = model.training
    model.eval()
    model(example)
    model.train(was_training)
    for h in handles:
        h.remove()
    return pairs


@torch.no_grad()
def fold_batchnorm(model: nn.Module, example: torch.Tensor, inplace: bool = False) -> nn.Module:
    """Fold every conv/linear + BN pair found by tracing ``example``; BN becomes Identity."""
    if not inplace:
        model = copy.deepcopy(model)
    modules = dict(model.named_modules())
    for conv_name, bn_name in find_conv_bn_pairs(model, example):
        conv, bn = modules[conv_name], modules[bn_name]
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shape = [-1] + [1] * (conv.weight.dim() - 1)
        conv.weight.mul_(g.view(shape))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        new_bias = bn.bias + (b - bn.running_mean) * g
        if conv.bias is None:
            conv.bias = nn.Parameter(new_bias)
        else:
            conv.bias.copy_(new_bias)
        if conv.wq is not None:
            conv.wq.init_from(conv.weight)  # learnable quantizers re-initialise on the folded weights
        parent_name, _, attr = bn_name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, attr, nn.Identity())
    return model
