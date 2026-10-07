"""One MNIST experiment: build, quantize, train, measure, save JSON.

Examples
--------
python sandbox_mnist/run.py --scheme fp32 --epochs 2
python sandbox_mnist/run.py --scheme uniform --wbits 4 --abits 8 --epochs 5
python sandbox_mnist/run.py --scheme ternary --abits 8 --first-last-bits 8 --epochs 5
python sandbox_mnist/run.py --scheme binary --no-first-last --width 16 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bitladder.data import get_mnist_loaders  # noqa: E402
from bitladder.measure import count_bops, count_macs, model_size_report  # noqa: E402
from bitladder.models import build_mnist_model  # noqa: E402
from bitladder.quant import QuantConfig, apply_quantization  # noqa: E402
from bitladder.train import evaluate, fit, pick_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default="fp32", choices=["fp32", "fp16", "uniform", "ternary", "binary"])
    p.add_argument("--wbits", type=int, default=8, help="bits for the uniform scheme")
    p.add_argument("--abits", type=int, default=None, help="PACT activation bits; omit for float activations")
    p.add_argument("--first-last-bits", type=int, default=8, help="uniform bits for first/last layer (32 = float)")
    p.add_argument("--no-first-last", action="store_true", help="quantize first/last layer like the rest")
    p.add_argument("--per-tensor", action="store_true", help="per-tensor scales instead of per-channel")
    p.add_argument("--lsq", action="store_true", help="learned step size for the uniform scheme")
    p.add_argument("--model", default="cnn", choices=["cnn", "mlp"])
    p.add_argument("--width", type=int, default=8, help="channel width (cnn) or hidden size (mlp)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subset", type=int, default=None, help="limit training samples (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--name", default=None, help="result file stem; defaults to a config label")
    p.add_argument("--quiet", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    cfg = QuantConfig(
        scheme=args.scheme,
        wbits=args.wbits,
        abits=args.abits,
        per_channel=not args.per_tensor,
        lsq=args.lsq,
        first_last_bits=None if args.no_first_last else args.first_last_bits,
    )
    model = build_mnist_model(args.model, args.width)
    apply_quantization(model, cfg)
    log = (lambda s: None) if args.quiet else print
    log(f"[{cfg.label}] model={args.model} width={args.width} device={device}")

    train_loader, test_loader = get_mnist_loaders(args.data, args.batch_size, subset=args.subset, seed=args.seed)
    t0 = time.time()
    history = fit(model, train_loader, test_loader, args.epochs, lr=args.lr, device=device, log=log)
    train_time = time.time() - t0

    model.eval()
    final = evaluate(model, test_loader, device)
    size = model_size_report(model)
    macs = count_macs(model, (1, 28, 28))
    bops = count_bops(model, (1, 28, 28))

    result = {
        "label": cfg.label,
        "config": vars(cfg),
        "model": args.model,
        "width": args.width,
        "epochs": args.epochs,
        "seed": args.seed,
        "test_acc": final["acc"],
        "test_loss": final["loss"],
        "train_time_s": train_time,
        "total_params": size["total_params"],
        "total_bytes": size["total_bytes"],
        "zlib_bytes": size["zlib_bytes"],
        "fp32_bytes": size["fp32_bytes"],
        "mean_wbits": size["mean_wbits"],
        "macs": sum(macs.values()),
        "bops": sum(bops.values()),
        "layers": size["layers"],
        "history": history,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"mnist_{args.model}{args.width}_{cfg.label}_s{args.seed}"
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    log(
        f"RESULT {name}: acc={final['acc']:.4f} params={size['total_params']} "
        f"bytes={size['total_bytes']} zlib={size['zlib_bytes']} "
        f"MACs={sum(macs.values()):,} BOPs={sum(bops.values()):,.0f}"
    )
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
