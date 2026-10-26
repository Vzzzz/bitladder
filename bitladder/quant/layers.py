"""Quantization-aware layers and a helper to switch a whole model to a scheme.

Models are built from ``QuantConv2d`` / ``QuantLinear`` from the start. Without
quantizers attached they behave exactly like ``nn.Conv2d`` / ``nn.Linear``.
``apply_quantization`` attaches weight and activation quantizers according to a
``QuantConfig``; it is idempotent and can be called again with a different config.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

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


class QuantConv1d(nn.Conv1d, _QuantMixin):
    """For TDNN / ECAPA-style speaker models and streaming enhancement front-ends."""

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


class QuantGRU(nn.GRU):
    """nn.GRU whose input-to-hidden and hidden-to-hidden weights are fake-quantized.

    The recurrence itself runs through the stock fused kernel (``torch._VF.gru``, the same
    call ``nn.GRU.forward`` makes), so training speed is unchanged. One weight quantizer
    per weight tensor (``wq[k]`` matches ``self._flat_weights_names[k]``); biases stay
    float. ``aq`` quantizes the *input* sequence only: the hidden state between steps is
    not quantized, which matches int8 deployments that keep the state in higher precision.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wq = nn.ModuleDict()
        self.aq = None
        self.storage_bits = 32.0

    def set_quantizers(self, make_wq: Optional[Callable[[], Optional[WeightQuantizer]]],
                       aq: Optional[ActQuantizer], storage_bits: Optional[float] = None) -> None:
        self.wq = nn.ModuleDict()
        self.aq = aq
        bits = None
        for name in self._flat_weights_names:
            if not name.startswith("weight"):
                continue
            q = make_wq() if make_wq is not None else None
            if q is not None:
                q.init_from(getattr(self, name))
                self.wq[name] = q
                bits = q.bits
        self.storage_bits = bits if bits is not None else (32.0 if storage_bits is None else storage_bits)

    def quantized_weights(self) -> list:
        out = []
        for name in self._flat_weights_names:
            w = getattr(self, name)
            out.append(self.wq[name](w) if name in self.wq else w)
        return out

    @property
    def wbits(self) -> float:
        return self.storage_bits

    @property
    def abits(self) -> float:
        return 32.0 if self.aq is None else self.aq.bits

    def weight_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name in self._flat_weights_names:
            if name.startswith("weight"):
                yield name, getattr(self, name)

    def forward(self, x, hx=None):  # type: ignore[override]
        if self.aq is not None:
            x = self.aq(x)
        if not self.wq:
            return super().forward(x, hx)
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed:
            raise NotImplementedError("QuantGRU does not support PackedSequence")
        batch_first = self.batch_first
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(1)
            batch_first = False
        if hx is None:
            n = x.size(0) if batch_first else x.size(1)
            hx = x.new_zeros(self.num_layers * (2 if self.bidirectional else 1), n, self.hidden_size)
        elif unbatched:
            hx = hx.unsqueeze(1)
        out, h = torch._VF.gru(x, hx, self.quantized_weights(), self.bias, self.num_layers,
                               float(self.dropout), self.training, self.bidirectional, batch_first)
        if unbatched:
            out, h = out.squeeze(1), h.squeeze(1)
        return out, h


QUANT_LAYER_TYPES = (QuantConv2d, QuantConv1d, QuantLinear)


def quant_layers(model: nn.Module) -> Iterator[tuple[str, _QuantMixin]]:
    """Conv / linear quant layers (the ones with a single ``weight``)."""
    for name, m in model.named_modules():
        if isinstance(m, QUANT_LAYER_TYPES):
            yield name, m


def quant_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Everything apply_quantization touches: conv / linear layers and QuantGRU."""
    for name, m in model.named_modules():
        if isinstance(m, QUANT_LAYER_TYPES + (QuantGRU,)):
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
    overrides: Optional[Dict[str, Union[int, str]]] = None
    """Per-layer precision by module name or fnmatch pattern (``blocks.*.pw``), checked in
    order; the first match wins and takes precedence over the edge rule. Values: an int
    (uniform k-bit, 32 = float, 16 = fp16 storage) or a scheme name (ternary, twn, ttq,
    binary, fp32, fp16)."""

    @property
    def label(self) -> str:
        mix = "-mix" if self.overrides else ""
        if self.scheme in ("fp32", "fp16"):
            return self.scheme + mix
        w = f"{self.scheme}{self.wbits}" if self.scheme == "uniform" else self.scheme
        a = f"-a{self.abits}" if self.abits else ""
        fl = "" if self.first_last_bits is None else f"-fl{self.first_last_bits}"
        return f"{w}{a}{fl}{'-lsq' if self.lsq else ''}{mix}"


def parse_overrides(spec: str) -> Dict[str, Union[int, str]]:
    """``"blocks.*.pw=2,stem=8,fc=ternary"`` -> ordered dict of overrides."""
    out: Dict[str, Union[int, str]] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, val = item.partition("=")
        val = val.strip()
        out[key.strip()] = int(val) if val.isdigit() else val
    return out


def _override_for(name: str, overrides: Optional[Dict[str, Union[int, str]]]) -> Optional[Union[int, str]]:
    if not overrides:
        return None
    for pattern, val in overrides.items():
        if name == pattern or fnmatch.fnmatchcase(name, pattern):
            return val
    return None


def _factory(scheme: str, bits: int, cfg: "QuantConfig") -> Tuple[Callable[[], Optional[WeightQuantizer]], float]:
    """Weight-quantizer factory and storage bits for a (scheme, bits) pair."""
    if scheme in ("fp32", "fp16"):
        return (lambda: None), (32.0 if scheme == "fp32" else 16.0)
    if scheme == "uniform" and bits >= 32:
        return (lambda: None), 32.0
    if scheme == "uniform" and bits == 16:
        return (lambda: None), 16.0
    return (lambda: make_weight_quantizer(scheme, bits, cfg.per_channel, cfg.lsq)), -1.0


def apply_quantization(model: nn.Module, cfg: QuantConfig) -> nn.Module:
    """Attach quantizers in place according to ``cfg`` and return the model.

    Layer order = ``named_modules`` order; the first and last quantizable modules are the
    "edge" layers for ``first_last_bits``. QuantGRU gets one quantizer per weight tensor.
    """
    layers = list(quant_modules(model))
    if not layers:
        raise ValueError("model has no QuantConv2d / QuantConv1d / QuantLinear / QuantGRU layers")
    n = len(layers)
    for idx, (name, layer) in enumerate(layers):
        is_edge = idx == 0 or idx == n - 1
        is_gru = isinstance(layer, QuantGRU)
        ov = _override_for(name, cfg.overrides)
        if ov is not None:
            scheme, bits = ("uniform", int(ov)) if isinstance(ov, int) else (str(ov), cfg.wbits)
        elif cfg.scheme in ("fp32", "fp16"):
            scheme, bits = cfg.scheme, 32
        elif is_edge and cfg.first_last_bits is not None:
            scheme, bits = "uniform", cfg.first_last_bits
        else:
            scheme, bits = cfg.scheme, cfg.wbits
        make_wq, storage = _factory(scheme, bits, cfg)
        float_layer = scheme in ("fp32", "fp16") and ov is None and cfg.scheme in ("fp32", "fp16")
        # Never quantize the raw network input; it is normalised float data.
        aq = ActQuantizer(cfg.abits) if (cfg.abits is not None and idx > 0 and not float_layer) else None
        if is_gru:
            layer.set_quantizers(make_wq if storage < 0 else None, aq,
                                 storage_bits=None if storage < 0 else storage)
        else:
            wq = make_wq()
            layer.set_quantizers(wq, aq, storage_bits=None if wq is not None else storage)
    return model
