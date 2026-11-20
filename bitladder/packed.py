"""Self-describing packed weight artifact: BLP1 + JSON manifest + payload + CRC32."""

from __future__ import annotations

import json
import math
import struct
import zlib
from typing import Dict, Tuple

import numpy as np
import torch.nn as nn

from .measure.size import pack_uint, pack_weight, unpack_ternary, unpack_uint
from .quant.layers import QuantGRU, quant_layers
from .quant.quantizers import TERNARY_BITS

MAGIC = b"BLP1"
COMPONENTS = ("weight", "scale", "bias", "mask")


def _components(layer) -> Dict[str, bytes]:
    mask = getattr(layer, "weight_mask", None)
    weight, scale = pack_weight(layer.weight.detach(), layer.wq, layer.wbits, mask)
    bias = b"" if layer.bias is None else layer.bias.detach().cpu().numpy().astype(np.float16).tobytes()
    bitmap = b"" if mask is None else pack_uint(mask.cpu().numpy().astype(np.uint64), 1)
    return dict(zip(COMPONENTS, (weight, scale, bias, bitmap)))


def pack_model(model: nn.Module) -> Tuple[bytes, Dict[str, object]]:
    """Pack conv / linear weights after BN folding and return the blob plus its manifest report."""
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("fold BatchNorm before packing")
    if any(isinstance(m, QuantGRU) for m in model.modules()):
        raise ValueError("QuantGRU is not supported by the packed exporter yet")
    layers = []
    payload = b""
    for name, layer in quant_layers(model):
        components = _components(layer)
        mask = getattr(layer, "weight_mask", None)
        nonzero_params = layer.weight.numel() if mask is None else int(mask.sum())
        entry = {"name": name, "shape": list(layer.weight.shape), "wbits": float(layer.wbits),
                 "abits": float(layer.abits), "params": layer.weight.numel(), "nonzero_params": nonzero_params}
        for key in COMPONENTS:
            entry[f"{key}_offset"] = len(payload)
            entry[f"{key}_bytes"] = len(components[key])
            payload += components[key]
        layers.append(entry)
    manifest = {"version": 1, "payload_bytes": len(payload), "layers": layers}
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = manifest_bytes + payload
    blob = MAGIC + struct.pack("<I", len(manifest_bytes)) + body + struct.pack("<I", zlib.crc32(body))
    report = {"format": "BLP1", "payload_bytes": len(payload), "manifest_bytes": len(manifest_bytes),
              "total_bytes": len(blob), "layers": layers}
    return blob, report


def read_packed(blob: bytes) -> Dict[str, object]:
    """Validate and decode a packed blob; raw components remain available for byte-level verification."""
    if len(blob) < 16 or blob[:4] != MAGIC:
        raise ValueError("not a BLP1 artifact")
    manifest_len = struct.unpack_from("<I", blob, 4)[0]
    manifest_start = 8
    manifest_end = manifest_start + manifest_len
    if manifest_end + 4 > len(blob):
        raise ValueError("truncated manifest")
    manifest = json.loads(blob[manifest_start:manifest_end].decode("utf-8"))
    payload_end = len(blob) - 4
    crc_expected = struct.unpack_from("<I", blob, payload_end)[0]
    if zlib.crc32(blob[manifest_start:payload_end]) != crc_expected:
        raise ValueError("CRC32 mismatch")
    payload = blob[manifest_end:payload_end]
    if len(payload) != manifest["payload_bytes"]:
        raise ValueError("payload length mismatch")
    layers = []
    for entry in manifest["layers"]:
        raw = {}
        for key in COMPONENTS:
            start, n = entry[f"{key}_offset"], entry[f"{key}_bytes"]
            if start + n > len(payload):
                raise ValueError(f"component outside payload: {entry['name']}:{key}")
            raw[key] = payload[start:start + n]
        bits = float(entry["wbits"])
        if bits == 32:
            weight = np.frombuffer(raw["weight"], dtype=np.float32)
        elif bits == 16:
            weight = np.frombuffer(raw["weight"], dtype=np.float16)
        elif math.isclose(bits, TERNARY_BITS):
            weight = unpack_ternary(raw["weight"], entry["nonzero_params"])
        else:
            weight = unpack_uint(raw["weight"], int(round(bits)), entry["nonzero_params"])
        mask = (unpack_uint(raw["mask"], 1, entry["params"]).astype(bool)
                if raw["mask"] else np.array([]))
        layers.append({"meta": entry, "raw": raw, "weight": weight,
                       "scale": np.frombuffer(raw["scale"], dtype=np.float16),
                       "bias": np.frombuffer(raw["bias"], dtype=np.float16), "mask": mask})
    return {"manifest": manifest, "layers": layers}


def verify_packed_model(model: nn.Module, blob: bytes) -> bool:
    """Verify every packed component byte-for-byte against the source model."""
    packed = read_packed(blob)
    expected = dict(quant_layers(model))
    if [r["meta"]["name"] for r in packed["layers"]] != list(expected):
        raise ValueError("layer list mismatch")
    for record in packed["layers"]:
        components = _components(expected[record["meta"]["name"]])
        for key in COMPONENTS:
            if record["raw"][key] != components[key]:
                raise ValueError(f"component mismatch: {record['meta']['name']}:{key}")
    return True
