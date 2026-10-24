"""One KWS experiment on Speech Commands v2: build, quantize, train, measure, save JSON.

Requires the feature cache (python audio/kws/cache.py ...). Examples:

python audio/kws/run.py --epochs 30                              # fp32 DS-CNN, 12 classes
python audio/kws/run.py --scheme ternary --abits 8 --epochs 30
python audio/kws/run.py --keywords yes,no,up,down,left,right,on,off --exclude stop,go --name base8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audio.kws.data import KEYWORDS10, KWSCache, KWSDataset, KWSOfficialTest  # noqa: E402
from audio.kws.model import DSCNN  # noqa: E402
from bitladder.cli import add_quant_args, quant_config  # noqa: E402
from bitladder.measure import count_bops, count_macs, model_size_report  # noqa: E402
from bitladder.quant import apply_quantization  # noqa: E402
from bitladder.train import evaluate, fit, pick_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_quant_args(p)
    p.add_argument("--cache", default=str(ROOT / "data" / "kws_cache"))
    p.add_argument("--keywords", default=",".join(KEYWORDS10), help="comma-separated positive classes")
    p.add_argument("--exclude", default="", help="comma-separated words removed from the pool entirely")
    p.add_argument("--unknown-ratio", type=float, default=0.1)
    p.add_argument("--silence-ratio", type=float, default=0.1)
    p.add_argument("--cv-unknown", type=int, default=0,
                   help="Common Voice speech windows added to _unknown_ in train (needs cv_train.npy)")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--init", default=None, help="state_dict to start from (e.g. fp32 checkpoint before QAT)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--name", default=None, help="result file stem; defaults to a config label")
    p.add_argument("--quiet", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    cfg = quant_config(args)
    keywords = [w for w in args.keywords.split(",") if w]
    exclude = [w for w in args.exclude.split(",") if w]
    log = (lambda s: None) if args.quiet else print

    cache = KWSCache(args.cache)
    ds_kw = dict(keywords=keywords, exclude=exclude, unknown_ratio=args.unknown_ratio,
                 silence_ratio=args.silence_ratio, seed=args.seed, cv_unknown=args.cv_unknown)
    train_ds = KWSDataset(cache, "train", augment=not args.no_augment, **ds_kw)
    val_ds = KWSDataset(cache, "validation", **ds_kw)
    test_ds = KWSDataset(cache, "test", **ds_kw)
    log(f"classes={train_ds.classes}")
    log(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} counts={train_ds.label_counts()}")
    def mk(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size if shuffle else 512, shuffle=shuffle,
                          num_workers=args.workers, drop_last=shuffle, pin_memory=True)

    train_loader, val_loader, test_loader = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    model = DSCNN(channels=args.channels, blocks=args.blocks, num_classes=len(train_ds.classes))
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"), strict=False)
    apply_quantization(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[{cfg.label}] DSCNN(c={args.channels}, blocks={args.blocks}) params={n_params} device={device}")

    t0 = time.time()
    history = fit(model, train_loader, val_loader, args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                  device=device, log=log)
    train_time = time.time() - t0

    model.eval()
    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    official = None
    if cache.test_index is not None and set(keywords) == set(KEYWORDS10):
        official = evaluate(model, mk(KWSOfficialTest(cache, keywords), False), device)
    size = model_size_report(model)
    shape = (1, cache.frames, cache.n_mels)
    macs, bops = count_macs(model, shape), count_bops(model, shape)

    name = args.name or f"kws_dscnn{args.channels}x{args.blocks}_{cfg.label}_s{args.seed}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{name}.pt")
    result = {
        "label": cfg.label, "config": vars(cfg), "name": name,
        "keywords": keywords, "exclude": exclude, "classes": train_ds.classes,
        "channels": args.channels, "blocks": args.blocks, "epochs": args.epochs, "seed": args.seed,
        "val_acc": val["acc"], "test_acc": test["acc"],
        "official_test_acc": None if official is None else official["acc"],
        "train_time_s": train_time, "total_params": size["total_params"],
        "total_bytes": size["total_bytes"], "zlib_bytes": size["zlib_bytes"],
        "fp32_bytes": size["fp32_bytes"], "mean_wbits": size["mean_wbits"],
        "macs": sum(macs.values()), "bops": sum(bops.values()),
        "layers": size["layers"], "history": history,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    off = "n/a" if official is None else f"{official['acc']:.4f}"
    log(f"RESULT {name}: val={val['acc']:.4f} test={test['acc']:.4f} official={off} "
        f"params={size['total_params']} bytes={size['total_bytes']} zlib={size['zlib_bytes']} "
        f"MACs={sum(macs.values()):,} train={train_time:.0f}s")
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
