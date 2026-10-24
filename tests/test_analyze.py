import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from bitladder.measure import (
    activation_stats,
    analysis_report,
    apply_probe,
    sensitivity,
    to_markdown,
    weight_stats,
)
from bitladder.models import TinyCNN
from bitladder.quant import QuantConfig, apply_quantization, quant_layers


def _model_and_data():
    torch.manual_seed(0)
    model = TinyCNN(width=8).eval()
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    return model, DataLoader(TensorDataset(x, y), batch_size=32)


def test_weight_stats_fields():
    model, _ = _model_and_data()
    stats = weight_stats(model)
    names = [n for n, _ in quant_layers(model)]
    assert [s.name for s in stats] == names
    s = stats[0]
    assert s.kind == "conv" and s.params == 8 * 1 * 3 * 3 and s.out_channels == 8 and s.wbits == 32
    assert 0.0 <= s.near_zero_frac <= 1.0 and 0.0 <= s.ternary_zero_frac <= 1.0
    assert s.chan_norm_min_over_median <= 1.0 + 1e-6 and s.chan_norm_cv >= 0.0
    apply_quantization(model, QuantConfig(scheme="ternary", first_last_bits=None))
    assert all(abs(t.wbits - 1.58) < 0.01 for t in weight_stats(model))


def test_activation_stats_ranges():
    model, loader = _model_and_data()
    acts = activation_stats(model, loader, max_batches=2)
    assert len(acts) == len(list(quant_layers(model)))
    for a in acts:
        assert 0.0 <= a.input_zero_frac <= 1.0 and 0.0 <= a.dead_out_channels_frac <= 1.0
        assert a.input_abs_mean >= 0.0 and a.output_abs_mean >= 0.0
    # conv2 sees post-ReLU, max-pooled input: some exact zeros are expected
    assert acts[1].input_zero_frac > 0.0


def test_probes_modify_only_target_layer():
    model, _ = _model_and_data()
    layers = dict(quant_layers(model))
    m = copy.deepcopy(model)
    apply_probe(dict(quant_layers(m))["conv2"], ("prune", 0.5))
    w = dict(quant_layers(m))["conv2"].weight
    assert (w.flatten(1).norm(dim=1) == 0).sum().item() == 8  # half of 16 channels
    assert torch.equal(dict(quant_layers(m))["conv1"].weight, layers["conv1"].weight)
    m2 = copy.deepcopy(model)
    apply_probe(dict(quant_layers(m2))["fc"], ("sparsity", 0.9))
    assert (dict(quant_layers(m2))["fc"].weight == 0).float().mean().item() >= 0.89
    m3 = copy.deepcopy(model)
    apply_probe(dict(quant_layers(m3))["conv3"], ("bits", 4))
    assert dict(quant_layers(m3))["conv3"].wbits == 4 and model.conv3.wq is None
    m4 = copy.deepcopy(model)
    apply_probe(dict(quant_layers(m4))["conv3"], ("ternary", 0))
    assert abs(dict(quant_layers(m4))["conv3"].wbits - 1.58) < 0.01


def test_sensitivity_and_report():
    model, loader = _model_and_data()

    def eval_fn(m):  # deterministic proxy metric: mean max-softmax on fixed data
        with torch.no_grad():
            return float(torch.cat([m(x).softmax(1).max(1).values for x, _ in loader]).mean())

    probes = [("bits", 8), ("bits", 2), ("prune", 0.5), ("sparsity", 0.9)]
    sens = sensitivity(model, eval_fn, probes)
    assert sens["probes"] == ["w8", "w2", "prune0.5", "sparsity0.9"]
    assert len(sens["layers"]) == 4 and all(len(r) == 4 for r in sens["drop"])
    assert abs(sens["drop"][0][0]) < 0.05  # 8-bit weights change almost nothing
    rep = analysis_report(model, eval_fn, loader, probes, max_batches=1)
    md = to_markdown(rep)
    assert "| conv1 |" in md and "sparsity0.9" in md and "Baseline metric" in md
