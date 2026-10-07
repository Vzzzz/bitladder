"""Quantization-aware layers and a helper to switch a whole model to a scheme.

Models are built from ``QuantConv2d`` / ``QuantLinear`` from the start. Without
quantizers attached they behave exactly like ``nn.Conv2d`` / ``nn.Linear``.
``apply_quantization`` attaches weight and activation quantizers according to a
``QuantConfig``; it is idempotent and can be called again with a different config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import ActQuantizer, WeightQuantizer, make_weight_quantizer


class _QuantMixin:
    wq: Optional[WeightQuantizer]
    aq: Optional[ActQuantizer]
    storage_bits: float  # bits per weight when exported (32 / 16 / quantizer.bits)

    def _init_quant(self) -> None:
        self.wq = None
        self.aq = None
        self.storage_bits = 32.0

    def set_quantizers(self, wq: Optional[WeightQuantizer], aq: Optional[ActQuantizer],
                       storage_bits: Optional[float] = None) -> None:
        self.wq = wq
        self.aq = aq
        if wq is not None:
            wq.init_from(self.weight)
            self.storage_bits = wq.bits
        else:
            self.storage_bits = 32.0 if storage_bits is None else storage_bits

    def quantized_weight(self) -> torch.Tensor:
        return self.weight if self.wq is None else self.wq(self.weight)

    @property
    def wbits(self) -> float:
        return self.storage_bits

    @property
    def abits(self) -> float:
        return 32.0 if self.aq is None else self.aq.bits


class QuantConv2d(nn.Conv2d, _QuantMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.aq is not None:
            x = self.aq(x)
        return self._conv_forward(x, self.quantized_weight(), self.bias)


class QuantLinear(nn.Linear, _QuantMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.aq is not None:
            x = self.aq(x)
        return F.linear(x, self.quantized_weight(), self.bias)


def quant_layers(model: nn.Module) -> Iterator[tuple[str, _QuantMixin]]:
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            yield name, m


@dataclass
class QuantConfig:
    """What to do with every QuantConv2d / QuantLinear in a model.

    scheme           fp32 | fp16 | uniform | ternary | binary
    wbits            bits for the uniform scheme (ignored otherwise)
    abits            activation bits (PACT) for the *inputs* of all quantized layers
                     except the first one; None keeps activations in float
    per_channel      per-output-channel scales for weights
    lsq              learnable step size for the uniform scheme
    first_last_bits  precision of the first and last layer: an int -> uniform k-bit,
                     None -> same scheme as the rest, 32 -> keep float
    """

    scheme: str = "fp32"
    wbits: int = 8
    abits: Optional[int] = None
    per_channel: bool = True
    lsq: bool = False
    first_last_bits: Optional[int] = 8

    @property
    def label(self) -> str:
        if self.scheme in ("fp32", "fp16"):
            return self.scheme
        w = f"{self.scheme}{self.wbits}" if self.scheme == "uniform" else self.scheme
        a = f"-a{self.abits}" if self.abits else ""
        fl = "" if self.first_last_bits is None else f"-fl{self.first_last_bits}"
        return f"{w}{a}{fl}{'-lsq' if self.lsq else ''}"


def apply_quantization(model: nn.Module, cfg: QuantConfig) -> nn.Module:
    """Attach quantizers in place according to ``cfg`` and return the model."""
    layers = list(quant_layers(model))
    if not layers:
        raise ValueError("model has no QuantConv2d / QuantLinear layers")
    n = len(layers)
    for idx, (_, layer) in enumerate(layers):
        is_edge = idx == 0 or idx == n - 1
        if cfg.scheme in ("fp32", "fp16"):
            layer.set_quantizers(None, None, storage_bits=32.0 if cfg.scheme == "fp32" else 16.0)
            continue
        if is_edge and cfg.first_last_bits is not None:
            if cfg.first_last_bits >= 32:
                wq = None
            else:
                wq = make_weight_quantizer("uniform", cfg.first_last_bits, cfg.per_channel, cfg.lsq)
        else:
            wq = make_weight_quantizer(cfg.scheme, cfg.wbits, cfg.per_channel, cfg.lsq)
        # Never quantize the raw network input; it is normalised float data.
        aq = ActQuantizer(cfg.abits) if (cfg.abits is not None and idx > 0) else None
        layer.set_quantizers(wq, aq)
    return model
