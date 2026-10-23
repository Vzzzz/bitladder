import math

import numpy as np
import pytest
import torch

from bitladder.quant import (
    TERNARY_BITS,
    ActQuantizer,
    BinaryQuantizer,
    TernaryQuantizer,
    UniformWeightQuantizer,
    QuantConfig,
    apply_quantization,
    quant_layers,
)
from bitladder.measure import model_size_report, pack_ternary, pack_uint, unpack_ternary, unpack_uint
from bitladder.models import TinyCNN


def test_uniform_8bit_is_close_to_identity():
    torch.manual_seed(0)
    w = torch.randn(16, 8, 3, 3)
    q = UniformWeightQuantizer(8)
    err = (q(w) - w).abs().max().item()
    assert err <= (w.abs().flatten(1).amax(1).max().item() / 127) / 2 + 1e-6


def test_uniform_codes_in_range_and_grad_flows():
    torch.manual_seed(0)
    w = torch.randn(4, 10, requires_grad=True)
    q = UniformWeightQuantizer(4)
    codes, scale = q.integer_repr(w)
    assert codes.min() >= -7 and codes.max() <= 7
    assert scale.shape == (4, 1)
    q(w).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert torch.allclose(w.grad, torch.ones_like(w.grad))  # identity STE, incl. channel absmax
    w2 = torch.tensor([[1.0, -7.0, 3.5, 7.0]], requires_grad=True)  # exact boundary values
    q(w2).sum().backward()
    assert torch.allclose(w2.grad, torch.ones_like(w2.grad))


def test_lsq_scale_is_learnable():
    w = torch.randn(4, 10, requires_grad=True)
    q = UniformWeightQuantizer(4, learnable=True)
    q.init_from(w)
    assert q.scale.requires_grad and q.scale.shape == (4, 1)
    (q(w) ** 2).sum().backward()
    assert q.scale.grad is not None and torch.isfinite(q.scale.grad).all()


def test_ternary_values_and_bits():
    w = torch.randn(6, 5) * 0.1
    q = TernaryQuantizer()
    codes, scale = q.integer_repr(w)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert math.isclose(q.bits, TERNARY_BITS)
    out = q(w)
    assert torch.allclose(out, codes.float() * scale)
    assert (codes == 0).any(), "absmean scaling should zero out small weights"


def test_binary_values():
    w = torch.randn(3, 7)
    q = BinaryQuantizer()
    codes, scale = q.integer_repr(w)
    assert set(codes.unique().tolist()) <= {-1, 1}
    assert torch.allclose(q(w).abs(), scale.expand_as(w))


def test_pact_clips_and_learns_alpha():
    aq = ActQuantizer(bits=4, alpha_init=2.0)
    x = torch.tensor([-1.0, 0.5, 1.0, 5.0], requires_grad=True)
    y = aq(x)
    assert y[0].item() == 0.0 and y[-1].item() == pytest.approx(2.0, abs=1e-6)
    assert len(set(y.tolist())) <= 16
    y.sum().backward()
    assert aq.alpha.grad is not None and aq.alpha.grad.item() != 0.0


def test_pack_uint_roundtrip():
    rng = np.random.default_rng(0)
    for bits in (1, 2, 3, 4, 8):
        vals = rng.integers(0, 2**bits, size=37)
        data = pack_uint(vals, bits)
        assert len(data) == math.ceil(37 * bits / 8)
        assert np.array_equal(unpack_uint(data, bits, 37), vals)


def test_pack_ternary_roundtrip():
    rng = np.random.default_rng(1)
    codes = rng.integers(-1, 2, size=101)
    data = pack_ternary(codes)
    assert len(data) == math.ceil(101 / 5)
    assert np.array_equal(unpack_ternary(data, 101), codes)


@pytest.mark.parametrize("scheme,expected_ratio", [("fp16", 1.9), ("uniform", 3.5), ("ternary", 10.0), ("binary", 14.0)])
def test_size_report_shrinks(scheme, expected_ratio):
    model = TinyCNN(width=8)
    fp32 = model_size_report(apply_quantization(model, QuantConfig(scheme="fp32")))
    cfg = QuantConfig(scheme=scheme, wbits=8, first_last_bits=None)
    rep = model_size_report(apply_quantization(model, cfg))
    assert rep["total_bytes"] * expected_ratio <= fp32["total_bytes"]
    assert rep["zlib_bytes"] > 0


def test_first_last_override_and_act_quant_placement():
    model = TinyCNN(width=8)
    apply_quantization(model, QuantConfig(scheme="ternary", abits=8, first_last_bits=8))
    layers = [m for _, m in quant_layers(model)]
    assert layers[0].wbits == 8 and layers[-1].wbits == 8
    assert all(math.isclose(m.wbits, TERNARY_BITS) for m in layers[1:-1])
    assert layers[0].aq is None and all(m.aq is not None for m in layers[1:])
    out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)
