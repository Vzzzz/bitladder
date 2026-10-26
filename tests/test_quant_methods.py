"""Overrides / mixed precision, TWN / TTQ ternary modes, BN folding, logit distillation."""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bitladder.measure import analysis_report, model_size_report
from bitladder.measure.mixed import greedy_profile, profile_bytes
from bitladder.models import TinyCNN
from bitladder.quant import (
    TERNARY_BITS,
    QuantConfig,
    TernaryQuantizer,
    apply_quantization,
    fold_batchnorm,
    parse_overrides,
    quant_layers,
)
from bitladder.train import fit, kd_loss


def test_overrides_by_name_and_pattern():
    model = TinyCNN(width=8)
    ov = parse_overrides("conv2=2, conv*=4, fc=ternary")
    assert ov == {"conv2": 2, "conv*": 4, "fc": "ternary"}
    apply_quantization(model, QuantConfig(scheme="uniform", wbits=8, abits=8, overrides=ov))
    bits = {n: m.wbits for n, m in quant_layers(model)}
    assert bits["conv2"] == 2 and bits["conv1"] == 4 and bits["conv3"] == 4  # first match wins
    assert math.isclose(bits["fc"], TERNARY_BITS)
    assert "-mix" in QuantConfig(scheme="uniform", overrides=ov).label
    # float override inside a quantized model
    apply_quantization(model, QuantConfig(scheme="ternary", first_last_bits=None, overrides={"conv1": 32}))
    assert model.conv1.wq is None and model.conv1.wbits == 32 and math.isclose(model.conv2.wbits, TERNARY_BITS)


def test_twn_and_ttq_modes():
    torch.manual_seed(0)
    w = torch.randn(6, 20, requires_grad=True)
    twn = TernaryQuantizer(mode="twn")
    q, s = twn.integer_repr(w)
    assert set(q.unique().tolist()) <= {-1, 0, 1} and s.shape == (6, 1)
    # TWN zeroes more than absmean (threshold 0.7 vs 0.5 of mean|w|)
    q0, _ = TernaryQuantizer(mode="absmean").integer_repr(w)
    assert (q == 0).float().mean() > (q0 == 0).float().mean()
    out = twn(w)
    assert torch.allclose(out[q != 0].abs(), s.expand_as(w)[q != 0])
    out.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))  # identity STE

    ttq = TernaryQuantizer(mode="ttq")
    ttq.init_from(w.detach())
    assert ttq.scale_pos.shape == (6, 1) and ttq.scale_neg.requires_grad
    w.grad = None
    (ttq(w) ** 2).sum().backward()
    assert ttq.scale_pos.grad is not None and ttq.scale_neg.grad is not None
    assert torch.isfinite(w.grad).all()
    q2, s2 = ttq.integer_repr(w)
    assert s2.numel() == 12  # pos + neg scales exported


def test_ttq_in_model_and_size():
    model = TinyCNN(width=8)
    apply_quantization(model, QuantConfig(scheme="ttq", abits=8, first_last_bits=8))
    assert model.conv2.wq.mode == "ttq" and model.conv2.wq.scale_pos is not None
    out = model(torch.randn(2, 1, 28, 28))
    out.sum().backward()
    rep = model_size_report(model)
    conv2 = next(r for r in rep["layers"] if r["name"] == "conv2")
    assert conv2["scale_bytes"] == 2 * 16 * 2  # fp16, two scales per channel


def test_greedy_mixed_profile():
    torch.manual_seed(0)
    model = TinyCNN(width=8).eval()
    x = torch.randn(64, 1, 28, 28)
    loader = DataLoader(TensorDataset(x, torch.randint(0, 10, (64,))), batch_size=32)

    def eval_fn(m):
        with torch.no_grad():
            return float(torch.cat([m(b).softmax(1).max(1).values for b, _ in loader]).mean())

    rep = analysis_report(model, eval_fn, None, [("bits", 8), ("bits", 4), ("bits", 2), ("ternary", 0)])
    prof_tight = greedy_profile(rep, budget=0.0)
    assert all(b == 8 for b in prof_tight.values())  # no drop allowed -> everything stays at the top
    prof_loose = greedy_profile(rep, budget=10.0)
    assert all(b == "ternary" for b in prof_loose.values())  # unlimited -> everything at the bottom
    assert profile_bytes(rep, prof_loose) < profile_bytes(rep, prof_tight)
    prof_fixed = greedy_profile(rep, budget=10.0, fixed={"conv1": 8})
    assert prof_fixed["conv1"] == 8
    cfg = QuantConfig(scheme="uniform", wbits=8, overrides=prof_loose)
    apply_quantization(model, cfg)
    assert all(math.isclose(m.wbits, TERNARY_BITS) for _, m in quant_layers(model))


def test_fold_batchnorm_matches_and_keeps_quantizers():
    torch.manual_seed(0)
    model = TinyCNN(width=8)
    # give BN non-trivial statistics
    with torch.no_grad():
        for bn in (model.bn1, model.bn2, model.bn3):
            bn.running_mean.normal_()
            bn.running_var.uniform_(0.5, 2.0)
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.normal_()
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    ref = model(x)
    folded = fold_batchnorm(model, x)
    assert isinstance(folded.bn1, nn.Identity) and folded.conv1.bias is not None
    assert torch.allclose(folded(x), ref, atol=1e-4)
    assert isinstance(model.bn1, nn.BatchNorm2d)  # original untouched
    # fold then quantize vs quantize then fold: both run, quantizers stay attached
    q = apply_quantization(TinyCNN(width=8).eval(), QuantConfig(scheme="uniform", wbits=4))
    fq = fold_batchnorm(q, x)
    assert fq.conv2.wq is not None and fq.conv2.wbits == 4
    assert fq(x).shape == (4, 10)


def test_kd_loss_and_fit_with_teacher():
    torch.manual_seed(0)
    s = torch.randn(8, 10)
    assert kd_loss(s, s, torch.randint(0, 10, (8,)), alpha=1.0, T=2.0).item() < 1e-5
    teacher = TinyCNN(width=8).eval()
    student = TinyCNN(width=4)
    x = torch.randn(64, 1, 28, 28)
    loader = DataLoader(TensorDataset(x, torch.randint(0, 10, (64,))), batch_size=32)
    hist = fit(student, loader, loader, epochs=2, device=torch.device("cpu"), teacher=teacher,
               log=lambda s: None, progress=False)
    assert len(hist) == 2 and all(math.isfinite(h["train_loss"]) for h in hist)
