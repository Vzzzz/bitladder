import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bitladder.models import TinyCNN
from bitladder.quant import QuantConfig, apply_quantization

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio.kws.export_onnx import bake_quantizers, export_onnx, prepare_for_export  # noqa: E402

pytest.importorskip("onnxruntime")


def test_bake_puts_weights_on_the_grid():
    torch.manual_seed(0)
    model = TinyCNN(width=8)
    apply_quantization(model, QuantConfig(scheme="uniform", wbits=4, first_last_bits=None))
    before = model.conv2.weight.detach().clone()
    n = bake_quantizers(model)
    assert n == 4
    w = model.conv2.weight.detach()
    assert not torch.allclose(w, before)
    # every output channel has at most 15 distinct values (4-bit symmetric grid)
    assert all(len(torch.unique(ch)) <= 15 for ch in w)
    # baking is idempotent: fake-quant of a grid value is the grid value
    assert torch.allclose(model.conv2.wq(w), w, atol=1e-6)


def test_export_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    model = TinyCNN(width=8)
    with torch.no_grad():  # non-trivial BN statistics
        for bn in (model.bn1, model.bn2, model.bn3):
            bn.running_mean.normal_()
            bn.running_var.uniform_(0.5, 2.0)
    apply_quantization(model, QuantConfig(scheme="uniform", wbits=4, abits=8))
    model.eval()
    x = torch.randn(3, 1, 28, 28)
    prepared, n = prepare_for_export(model, x[:1])
    assert n == 4 and prepared.conv1.wq is None and prepared.conv1.aq is None
    info = export_onnx(model, x[:1], str(tmp_path / "m.onnx"))
    assert info["baked"] == 4 and info["max_abs_diff"] < 1e-4
    import onnxruntime as ort

    sess = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = np.concatenate([sess.run(None, {name: x[i:i + 1].numpy()})[0] for i in range(3)])  # static batch of 1
    with torch.no_grad():
        ref = prepared(x).numpy()
    assert np.abs(out - ref).max() < 1e-4
