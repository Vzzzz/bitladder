"""Rebuild a model from a run.py result JSON (+ .pt checkpoint next to it)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audio.kws.model import DSCNN  # noqa: E402
from bitladder.quant import QuantConfig, apply_quantization  # noqa: E402


def load_run(json_path: str, freeze: bool = False) -> Tuple[DSCNN, dict]:
    """Model with the quantizers it was trained with, in eval mode, plus the result dict."""
    meta = json.loads(Path(json_path).read_text())
    model = DSCNN(channels=meta["channels"], blocks=meta["blocks"], num_classes=len(meta["classes"]),
                  time_stride=meta.get("time_stride", 2))
    apply_quantization(model, QuantConfig(**meta["config"]))
    model.load_state_dict(torch.load(Path(json_path).with_suffix(".pt"), map_location="cpu"))
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model, meta
