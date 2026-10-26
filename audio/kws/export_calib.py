"""Calibration and validation data for X-CUBE-AI (or any PTQ tool): normalised log-mel
windows from the feature cache, as the model sees them, plus the reference outputs.

    python audio/kws/export_calib.py --run results/kws12_fp32_s0.json --n 500 --out export/calib

Writes <out>.npz with
    x        (N, 1, 49, 40) float32   model inputs (normalised features)
    y        (N,)           int64     labels in the run's class order
    logits   (N, 12)        float32   outputs of the PyTorch model (for "validate on target")
    classes, mean, std                metadata
and <out>_x.npy / <out>_logits.npy for tools that want plain arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audio.kws.checkpoint import load_run  # noqa: E402
from audio.kws.data import SILENCE, UNKNOWN, KWSCache, KWSDataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="result JSON of the model to be deployed")
    p.add_argument("--cache", default=str(ROOT / "data" / "kws_cache"))
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "export" / "calib"))
    a = p.parse_args()

    model, meta = load_run(a.run)
    cache = KWSCache(a.cache)
    keywords = [c for c in meta["classes"] if c not in (SILENCE, UNKNOWN)]
    ds = KWSDataset(cache, a.split, keywords=keywords, exclude=meta.get("exclude", []), seed=meta["seed"])
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(ds))[: a.n]
    xs, ys = zip(*(ds[int(i)] for i in idx))
    x = torch.stack(xs)
    y = torch.tensor(ys)
    with torch.no_grad():
        logits = torch.cat([model(b) for b in DataLoader(x, batch_size=256)])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), x=x.numpy().astype(np.float32), y=y.numpy(),
             logits=logits.numpy().astype(np.float32), classes=np.array(ds.classes),
             mean=np.float32(cache.mean), std=np.float32(cache.std))
    np.save(out.parent / (out.name + "_x.npy"), x.numpy().astype(np.float32))
    np.save(out.parent / (out.name + "_logits.npy"), logits.numpy().astype(np.float32))
    acc = (logits.argmax(1) == y).float().mean().item()
    print(f"wrote {out.with_suffix('.npz')}: x{tuple(x.shape)} from {a.split}, model acc on them {acc:.4f}, "
          f"mean={cache.mean:.4f} std={cache.std:.4f}")


if __name__ == "__main__":
    main()
