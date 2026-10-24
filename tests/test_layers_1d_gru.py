import math

import torch
import torch.nn as nn

from bitladder.measure import count_bops, count_macs, model_size_report
from bitladder.quant import (
    TERNARY_BITS,
    QuantConfig,
    QuantConv1d,
    QuantGRU,
    QuantLinear,
    apply_quantization,
    quant_modules,
)


class TinyTDNN(nn.Module):
    """Conv1d front-end -> GRU -> linear embedding, (B, C, T) in, (B, D) out."""

    def __init__(self, c_in=20, c=16, hidden=24, emb=8):
        super().__init__()
        self.conv = QuantConv1d(c_in, c, 5, padding=2)
        self.gru = QuantGRU(c, hidden, num_layers=2, batch_first=True)
        self.fc = QuantLinear(hidden, emb)

    def forward(self, x):
        h = torch.relu(self.conv(x)).transpose(1, 2)  # (B, T, C)
        out, _ = self.gru(h)
        return self.fc(out[:, -1])


def test_conv1d_and_gru_match_torch_without_quantizers():
    torch.manual_seed(0)
    m = TinyTDNN()
    x = torch.randn(3, 20, 30)
    ref_conv = nn.Conv1d(20, 16, 5, padding=2)
    ref_conv.load_state_dict(m.conv.state_dict())
    assert torch.allclose(m.conv(x), ref_conv(x))
    ref_gru = nn.GRU(16, 24, num_layers=2, batch_first=True)
    ref_gru.load_state_dict(m.gru.state_dict())
    h = torch.relu(m.conv(x)).transpose(1, 2)
    assert torch.allclose(m.gru(h)[0], ref_gru(h)[0])


def test_apply_quantization_covers_gru_and_runs():
    torch.manual_seed(0)
    m = TinyTDNN()
    names = [n for n, _ in quant_modules(m)]
    assert names == ["conv", "gru", "fc"]
    apply_quantization(m, QuantConfig(scheme="ternary", abits=8, first_last_bits=8))
    assert m.conv.wbits == 8 and m.fc.wbits == 8 and math.isclose(m.gru.wbits, TERNARY_BITS)
    assert set(m.gru.wq.keys()) == {"weight_ih_l0", "weight_hh_l0", "weight_ih_l1", "weight_hh_l1"}
    assert m.gru.aq is not None and m.conv.aq is None
    x = torch.randn(3, 20, 30, requires_grad=True)
    y = m(x)
    assert y.shape == (3, 8)
    y.sum().backward()
    assert all(p.grad is not None for n, p in m.gru.named_parameters() if n.startswith("weight"))
    # ternary fake-quant really happened: the effective weights have <= 3 distinct magnitudes per row
    w_q = m.gru.quantized_weights()[0]
    assert all(len(torch.unique(row.abs())) <= 2 for row in w_q)  # {0, scale}


def test_gru_unbatched_input():
    m = TinyTDNN()
    apply_quantization(m, QuantConfig(scheme="uniform", wbits=4))
    out, h = m.gru(torch.randn(7, 16))
    assert out.shape == (7, 24) and h.shape == (2, 24)


def test_size_and_macs_include_gru():
    m = TinyTDNN()
    fp = model_size_report(apply_quantization(m, QuantConfig("fp32")))
    layers = {rec["name"]: rec for rec in fp["layers"]}
    assert set(layers) == {"conv", "gru", "fc"}
    gru_params = sum(p.numel() for n, p in m.gru.named_parameters() if n.startswith("weight"))
    assert layers["gru"]["params"] == gru_params
    assert layers["gru"]["weight_bytes"] == 4 * gru_params
    tern = model_size_report(apply_quantization(m, QuantConfig("ternary", first_last_bits=None)))
    assert tern["total_bytes"] * 8 < fp["total_bytes"]
    macs = count_macs(m, (20, 30))
    assert macs["conv"] == 16 * 20 * 5 * 30
    assert macs["gru"] == 30 * 3 * ((16 * 24 + 24 * 24) + (24 * 24 + 24 * 24))
    bops = count_bops(m, (20, 30))
    assert bops["gru"] < macs["gru"] * 32 * 32
