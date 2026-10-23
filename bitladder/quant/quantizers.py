"""Fake-quantizers with straight-through estimators (STE).

All weight quantizers map a float tensor to a float tensor of the same shape whose
values lie on a quantization grid. The grid is ``q * scale`` where ``q`` is an integer
code. ``integer_repr`` exposes ``(q, scale)`` for packing and export.

Schemes
-------
uniform  symmetric k-bit grid  q in [-(2^(k-1)-1), 2^(k-1)-1], scale = absmax / qmax
         (or a learned step size, LSQ, Esser et al. 2019). Note that k=2 gives three
         levels, i.e. ternary with absmax scaling.
ternary  BitNet b1.58 (Ma et al. 2024): q in {-1, 0, 1}, scale = mean(|w|).
         Storage cost log2(3) = 1.58 bits per weight.
binary   XNOR-Net (Rastegari et al. 2016): q = sign(w), scale = mean(|w|). 1 bit.

Activations use PACT (Choi et al. 2018): unsigned k-bit grid on [0, alpha] with a
learnable clipping value alpha.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TERNARY_BITS = math.log2(3)
_EPS = 1e-8


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass, identity gradient in the backward pass."""
    return (x.round() - x).detach() + x


def grad_scale(x: torch.Tensor, g: float) -> torch.Tensor:
    """Identity in forward, multiplies the gradient by ``g`` (LSQ trick)."""
    return (x - x * g).detach() + x * g


def _per_channel_view(stat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return stat.view(-1, *([1] * (w.dim() - 1)))


def _absmax(w: torch.Tensor, per_channel: bool) -> torch.Tensor:
    if per_channel:
        return _per_channel_view(w.abs().flatten(1).amax(dim=1), w)
    return w.abs().max()


def _absmean(w: torch.Tensor, per_channel: bool) -> torch.Tensor:
    if per_channel:
        return _per_channel_view(w.abs().flatten(1).mean(dim=1), w)
    return w.abs().mean()


class WeightQuantizer(nn.Module):
    """Base class. Subclasses set ``bits`` (float) and implement ``integer_repr``."""

    bits: float = 32.0

    def __init__(self, per_channel: bool = True):
        super().__init__()
        self.per_channel = per_channel

    def init_from(self, w: torch.Tensor) -> None:  # noqa: D401
        """Hook for quantizers with learnable state that depends on weight shape."""

    def integer_repr(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return integer codes ``q`` (same shape as ``w``) and ``scale``."""
        raise NotImplementedError

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"bits={self.bits:.2f}, per_channel={self.per_channel}"


class UniformWeightQuantizer(WeightQuantizer):
    """Symmetric k-bit quantizer, optionally with a learned step size (LSQ)."""

    def __init__(self, bits: int, per_channel: bool = True, learnable: bool = False):
        super().__init__(per_channel)
        if bits < 2 or bits > 16:
            raise ValueError("uniform quantizer supports 2..16 bits")
        self.bits = float(bits)
        self.qmax = 2 ** (bits - 1) - 1
        self.learnable = learnable
        self.scale = None  # nn.Parameter after init_from when learnable

    def init_from(self, w: torch.Tensor) -> None:
        if not self.learnable:
            return
        with torch.no_grad():
            # LSQ initialisation: 2 * mean|w| / sqrt(qmax)
            init = 2 * _absmean(w, self.per_channel) / math.sqrt(self.qmax)
        self.scale = nn.Parameter(init.clone().clamp_min(_EPS))
        self._numel_per_scale = w.numel() // init.numel()

    def _scale(self, w: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            if self.scale is None:
                raise RuntimeError("call init_from(weight) before using a learnable quantizer")
            g = 1.0 / math.sqrt(self._numel_per_scale * self.qmax)
            return grad_scale(self.scale.abs() + _EPS, g)
        return (_absmax(w, self.per_channel) / self.qmax).detach().clamp_min(_EPS)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        s = self._scale(w)
        x = torch.clamp(w / s, -self.qmax, self.qmax)
        if self.learnable:
            # LSQ: gradient flows through the clamp (zero for clipped weights) and into s.
            return round_ste(x) * s
        # Fixed absmax scale: nothing is clipped except by fp rounding at the boundary,
        # so use a pure identity STE and keep every weight's gradient.
        return (x.round() * s).detach() + w - w.detach()

    @torch.no_grad()
    def integer_repr(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._scale(w)
        q = torch.clamp(torch.round(w / s), -self.qmax, self.qmax).to(torch.int64)
        return q, s


class TernaryQuantizer(WeightQuantizer):
    """BitNet b1.58: weights in {-1, 0, 1} * mean(|w|)."""

    bits = TERNARY_BITS

    def _scale(self, w: torch.Tensor) -> torch.Tensor:
        return _absmean(w, self.per_channel).detach().clamp_min(_EPS)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # BitNet b1.58: w + (RoundClip(w / s) * s - w).detach(), i.e. identity STE for all
        # weights including the clipped ones.
        s = self._scale(w)
        q = torch.clamp(torch.round(w / s), -1, 1)
        return (q * s).detach() + w - w.detach()

    @torch.no_grad()
    def integer_repr(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._scale(w)
        q = torch.clamp(torch.round(w / s), -1, 1).to(torch.int64)
        return q, s


class BinaryQuantizer(WeightQuantizer):
    """XNOR-Net style: sign(w) * mean(|w|), identity STE."""

    bits = 1.0

    def _scale(self, w: torch.Tensor) -> torch.Tensor:
        return _absmean(w, self.per_channel).detach().clamp_min(_EPS)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        s = self._scale(w)
        q = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return (q * s).detach() + w - w.detach()

    @torch.no_grad()
    def integer_repr(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._scale(w)
        q = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w)).to(torch.int64)
        return q, s


def make_weight_quantizer(scheme: str, bits: int = 8, per_channel: bool = True,
                          learnable: bool = False) -> WeightQuantizer | None:
    """Factory. ``scheme`` in {fp32, fp16, uniform, ternary, binary}.

    fp32/fp16 return ``None`` (no fake quantization; fp16 only affects size accounting).
    """
    scheme = scheme.lower()
    if scheme in ("fp32", "fp16", "none"):
        return None
    if scheme == "uniform":
        return UniformWeightQuantizer(bits, per_channel=per_channel, learnable=learnable)
    if scheme == "ternary":
        return TernaryQuantizer(per_channel=per_channel)
    if scheme == "binary":
        return BinaryQuantizer(per_channel=per_channel)
    raise ValueError(f"unknown scheme {scheme!r}")


class ActQuantizer(nn.Module):
    """PACT activation quantizer: clamp to [0, alpha] (or [-alpha, alpha]) and round."""

    def __init__(self, bits: int = 8, alpha_init: float = 6.0, signed: bool = False):
        super().__init__()
        if bits < 1 or bits > 16:
            raise ValueError("activation quantizer supports 1..16 bits")
        self.bits = float(bits)
        self.signed = signed
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.levels = (2 ** (bits - 1) - 1) if signed else (2 ** bits - 1)
        if self.levels < 1:
            raise ValueError("signed activation quantizer needs at least 2 bits")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.abs() + _EPS
        if self.signed:
            x_c = torch.maximum(torch.minimum(x, alpha), -alpha)
        else:
            x_c = torch.minimum(F.relu(x), alpha)
        s = alpha / self.levels
        return round_ste(x_c / s) * s

    def extra_repr(self) -> str:
        return f"bits={self.bits:.0f}, signed={self.signed}"
