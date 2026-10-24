import sys
from pathlib import Path

import torch

from bitladder.measure import count_bops, count_macs
from bitladder.models import TinyCNN, TinyMLP
from bitladder.quant import QuantConfig, apply_quantization

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio.kws.features import LogMel  # noqa: E402
from audio.kws.model import DSCNN  # noqa: E402


def test_tiny_models_shapes_and_macs():
    for model in (TinyCNN(width=8), TinyMLP(hidden=32)):
        out = model(torch.randn(3, 1, 28, 28))
        assert out.shape == (3, 10)
        macs = count_macs(model, (1, 28, 28))
        assert sum(macs.values()) > 0


def test_bops_follow_bits():
    model = TinyCNN(width=8)
    fp = sum(count_bops(apply_quantization(model, QuantConfig("fp32")), (1, 28, 28)).values())
    cfg = QuantConfig("ternary", abits=8, first_last_bits=None)
    tern = sum(count_bops(apply_quantization(model, cfg), (1, 28, 28)).values())
    assert tern < fp / 20


def test_logmel_and_dscnn_shapes():
    fe = LogMel()
    wav = torch.randn(2, 16000)
    feats = fe(wav)
    assert feats.shape == (2, fe.num_frames(16000), 40)
    assert fe.num_frames(16000) == 49
    model = DSCNN(channels=64, blocks=4)
    out = model(feats.unsqueeze(1))
    assert out.shape == (2, 12)
    n_params = sum(p.numel() for p in model.parameters())
    assert 15_000 < n_params < 40_000
