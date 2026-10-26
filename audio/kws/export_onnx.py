"""Export a trained KWS model to float ONNX for X-CUBE-AI (deployment path A in
docs/stm32_deploy.md): bake the fake-quantized weights, fold BatchNorm, drop the
quantizer modules, export, and check the ONNX against PyTorch with ONNX Runtime.

    python audio/kws/export_onnx.py --run results/kws12_fp32_s0.json --out export/kws12.onnx
    python audio/kws/export_onnx.py --run results/base8_u4a8_s0.json --out export/base8_u4.onnx --no-bake

"Bake" replaces every quantized weight by its fake-quantized value (e.g. the 4-bit grid),
so an int8 PTQ by the deployment tool starts from the QAT solution; activations go out
in float and are quantized by the tool from the calibration set (export_calib.py).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bitladder.quant import QuantGRU, fold_batchnorm, quant_layers, quant_modules  # noqa: E402


@torch.no_grad()
def bake_quantizers(model: nn.Module) -> int:
    """weight := fake-quantized weight for every layer with a weight quantizer; returns count."""
    n = 0
    for _, m in quant_layers(model):
        if m.wq is not None:
            m.weight.copy_(m.wq(m.weight))
            n += 1
    for _, m in model.named_modules():
        if isinstance(m, QuantGRU) and m.wq:
            for name, w in m.weight_tensors():
                if name in m.wq:
                    w.copy_(m.wq[name](w))
                    n += 1
    return n


def strip_quantizers(model: nn.Module) -> None:
    for _, m in quant_modules(model):
        if isinstance(m, QuantGRU):
            m.set_quantizers(None, None)
        else:
            m.set_quantizers(None, None)


def prepare_for_export(model: nn.Module, example: torch.Tensor, bake: bool = True) -> Tuple[nn.Module, int]:
    """Copy of the model: baked (optional), BN folded, quantizers removed, eval mode."""
    m = copy.deepcopy(model).eval()
    n = bake_quantizers(m) if bake else 0
    m = fold_batchnorm(m, example, inplace=True)
    strip_quantizers(m)
    return m.eval(), n


def export_onnx(model: nn.Module, example: torch.Tensor, path: str, bake: bool = True,
                input_name: str = "features", output_name: str = "logits") -> dict:
    """Export and verify. Returns {"baked": n, "max_abs_diff": float, "path": str}."""
    m, n = prepare_for_export(model, example, bake)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, (example,), path, input_names=[input_name], output_names=[output_name],
                      dynamo=True, external_data=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = m(example).numpy()
    out = sess.run(None, {sess.get_inputs()[0].name: example.numpy()})[0]
    return {"baked": n, "max_abs_diff": float(np.abs(out - ref).max()), "path": path}


def main() -> None:
    from audio.kws.checkpoint import load_run

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None, help="default export/<run name>.onnx")
    p.add_argument("--no-bake", action="store_true", help="export latent float weights instead of the QAT grid")
    p.add_argument("--calib", default=None, help="calib .npz from export_calib.py: also report ONNX vs PyTorch on it")
    a = p.parse_args()

    model, meta = load_run(a.run)
    frames, mels = 49, 40
    example = torch.zeros(1, 1, frames, mels)
    out = a.out or str(ROOT / "export" / f"{meta['name']}.onnx")
    info = export_onnx(model, example, out, bake=not a.no_bake)
    print(f"wrote {info['path']}: baked {info['baked']} quantized weight tensor(s), "
          f"onnx vs torch max |diff| on zeros {info['max_abs_diff']:.2e}")
    if a.calib:
        import onnxruntime as ort

        d = np.load(a.calib)
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        o = np.concatenate([sess.run(None, {name: d["x"][i:i + 1]})[0] for i in range(len(d["x"]))])  # static batch 1
        acc_onnx = (o.argmax(1) == d["y"]).mean()
        acc_ref = (d["logits"].argmax(1) == d["y"]).mean()
        print(f"calib set: onnx acc {acc_onnx:.4f} vs pytorch {acc_ref:.4f}, "
              f"max |logit diff| {np.abs(o - d['logits']).max():.2e}")


if __name__ == "__main__":
    main()
