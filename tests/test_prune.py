import math

import torch

from bitladder.measure import model_size_report
from bitladder.models import TinyCNN
from bitladder.prune import apply_magnitude_pruning, load_state_with_masks, pruning_report
from bitladder.quant import QuantConfig, apply_quantization


def test_magnitude_pruning_masks_exact_fraction_and_gradients():
    torch.manual_seed(0)
    model = TinyCNN(width=8, bn=False)
    report = apply_magnitude_pruning(model, 0.5, "conv2")
    mask = model.conv2.weight_mask
    assert report[0]["name"] == "conv2" and report[0]["nonzero_params"] == mask.sum()
    assert torch.equal(model.conv2.masked_weight()[~mask], torch.zeros_like(model.conv2.weight)[~mask])
    model(torch.randn(2, 1, 28, 28)).sum().backward()
    assert model.conv2.weight.grad[~mask].eq(0).all()
    assert pruning_report(model)[0]["sparsity"] == 0.5


def test_pruned_checkpoint_restores_mask_and_outputs():
    torch.manual_seed(0)
    model = TinyCNN(width=8, bn=False).eval()
    apply_magnitude_pruning(model, 0.7, "conv2,fc")
    restored = TinyCNN(width=8, bn=False).eval()
    load_state_with_masks(restored, model.state_dict())
    x = torch.randn(2, 1, 28, 28)
    assert torch.equal(restored.conv2.weight_mask, model.conv2.weight_mask)
    assert torch.equal(restored.fc.weight_mask, model.fc.weight_mask)
    assert torch.allclose(restored(x), model(x))


def test_sparse_size_packs_nonzero_values_and_bitmap():
    torch.manual_seed(0)
    model = TinyCNN(width=8, bn=False)
    apply_quantization(model, QuantConfig(scheme="uniform", wbits=4, first_last_bits=None))
    dense = model_size_report(model)
    apply_magnitude_pruning(model, 0.5, "conv2")
    sparse = model_size_report(model)
    layer = next(r for r in sparse["layers"] if r["name"] == "conv2")
    n = layer["params"]
    assert layer["nonzero_params"] == n // 2
    assert layer["weight_bytes"] == math.ceil((n // 2) * 4 / 8)
    assert layer["mask_bytes"] == math.ceil(n / 8)
    assert layer["total_bytes"] == (layer["weight_bytes"] + layer["scale_bytes"]
                                    + layer["bias_bytes"] + layer["mask_bytes"])
    assert sparse["nonzero_params"] == dense["nonzero_params"] - n // 2
    assert sparse["total_bytes"] < dense["total_bytes"]
