"""Honest model size: bytes after bit-packing, not parameter counts.

Accounting rules (see docs/decision-log.md, ADR-005):

* quantized weights: integer codes packed at ``bits`` per weight; ternary codes are
  packed five per byte (3^5 = 243 <= 256), binary eight per byte;
* float layers: 4 bytes (fp32) or 2 bytes (fp16) per weight;
* scales: fp16, one per output channel (per-channel) or one per tensor;
* biases: fp16 per output channel;
* BatchNorm: folded into a per-channel fp16 scale and fp16 bias, 4 bytes per channel.
  (Folding BN *into quantized weights* changes the weights; we count it separately
  and leave the "fold before QAT" question open.)
* ``zlib_bytes``: the whole payload compressed with zlib, a cheap lower bound for what
  entropy coding could buy.
"""

from __future__ import annotations

import math
import zlib
from typing import Dict, List

import numpy as np
import torch.nn as nn

from ..quant.layers import QuantConv2d, QuantLinear
from ..quant.quantizers import TERNARY_BITS


def pack_uint(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integers < 2**bits into bytes, LSB first."""
    values = np.asarray(values, dtype=np.uint64).ravel()
    if values.size == 0:
        return b""
    if values.max() >= (1 << bits):
        raise ValueError("value does not fit into the requested bit width")
    shifts = np.arange(bits, dtype=np.uint64)
    bit_matrix = ((values[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bit_matrix.ravel(), bitorder="little").tobytes()


def unpack_uint(data: bytes, bits: int, count: int) -> np.ndarray:
    raw = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[: count * bits]
    bit_matrix = raw.reshape(count, bits).astype(np.uint64)
    shifts = np.arange(bits, dtype=np.uint64)
    return (bit_matrix << shifts).sum(axis=1)


def pack_ternary(codes: np.ndarray) -> bytes:
    """Pack values in {-1, 0, 1} five per byte (base-3)."""
    codes = np.asarray(codes).ravel().astype(np.int64) + 1
    if codes.size == 0:
        return b""
    if codes.min() < 0 or codes.max() > 2:
        raise ValueError("ternary codes must be in {-1, 0, 1}")
    pad = (-codes.size) % 5
    codes = np.concatenate([codes, np.zeros(pad, dtype=np.int64)]).reshape(-1, 5)
    weights = 3 ** np.arange(5, dtype=np.int64)
    return (codes * weights).sum(axis=1).astype(np.uint8).tobytes()


def unpack_ternary(data: bytes, count: int) -> np.ndarray:
    bytes_arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    digits = np.stack([(bytes_arr // 3**i) % 3 for i in range(5)], axis=1).ravel()
    return digits[:count] - 1


def _layer_payload(layer) -> Dict[str, object]:
    """Return packed bytes for one QuantConv2d / QuantLinear."""
    w = layer.weight.detach()
    n = w.numel()
    out_ch = w.shape[0]
    bits = layer.wbits
    if layer.wq is None:
        dtype = np.float32 if bits == 32 else np.float16
        weight_bytes = w.cpu().numpy().astype(dtype).tobytes()
        scale_bytes = b""
    else:
        q, scale = layer.wq.integer_repr(w)
        q = q.cpu().numpy()
        if math.isclose(bits, TERNARY_BITS):
            weight_bytes = pack_ternary(q)
        elif bits == 1:
            weight_bytes = pack_uint((q > 0).astype(np.uint64), 1)
        else:
            k = int(round(bits))
            qmax = 2 ** (k - 1) - 1
            weight_bytes = pack_uint((q + qmax).astype(np.uint64), k)
        scale_bytes = scale.detach().cpu().numpy().astype(np.float16).tobytes()
    bias_bytes = b"" if layer.bias is None else layer.bias.detach().cpu().numpy().astype(np.float16).tobytes()
    return {
        "params": n,
        "out_channels": out_ch,
        "wbits": bits,
        "abits": layer.abits,
        "weight_bytes": len(weight_bytes),
        "scale_bytes": len(scale_bytes),
        "bias_bytes": len(bias_bytes),
        "payload": weight_bytes + scale_bytes + bias_bytes,
    }


def model_size_report(model: nn.Module) -> Dict[str, object]:
    """Per-layer and total byte counts for a model built from quant layers."""
    layers: List[Dict[str, object]] = []
    payload = b""
    bn_bytes = 0
    bn_params = 0
    other_params = 0
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            info = _layer_payload(m)
            payload += info.pop("payload")
            info["name"] = name
            info["total_bytes"] = info["weight_bytes"] + info["scale_bytes"] + info["bias_bytes"]
            layers.append(info)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            bn_bytes += 4 * m.num_features
            bn_params += sum(p.numel() for p in m.parameters())
            payload += np.zeros(2 * m.num_features, dtype=np.float16).tobytes()  # placeholder
        elif len(list(m.children())) == 0:
            other_params += sum(p.numel() for p in m.parameters(recurse=False))
    quant_params = sum(int(rec["params"]) for rec in layers)
    total_bytes = sum(int(rec["total_bytes"]) for rec in layers) + bn_bytes
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "layers": layers,
        "quant_params": quant_params,
        "bn_bytes": bn_bytes,
        "bn_params": bn_params,
        "other_params": other_params,
        "total_params": total_params,
        "fp32_bytes": total_params * 4,
        "total_bytes": total_bytes,
        "zlib_bytes": len(zlib.compress(payload, 9)),
        "mean_wbits": (sum(float(rec["wbits"]) * int(rec["params"]) for rec in layers) / max(quant_params, 1)),
    }
