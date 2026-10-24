"""Layer analysis of a trained KWS model: weight statistics, activation statistics and
one-shot sensitivity to lower bits / channel pruning / unstructured sparsity.

python audio/kws/analyze.py --run results/kws12_fp32_s0.json
python audio/kws/analyze.py --run results/kws12_fp32_s0.json --probes bits:8,4,2 ternary prune:0.25,0.5 sparsity:0.5,0.8
python audio/kws/analyze.py --run results/base8_unk1_fp32_s0.json --split test --batches 20 --no-heatmap

Writes results/<name>_analysis.json, results/<name>_analysis.md and results/<name>_sensitivity.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audio.kws.checkpoint import load_run  # noqa: E402
from audio.kws.data import SILENCE, UNKNOWN, KWSCache, KWSDataset  # noqa: E402
from bitladder.measure import analysis_report, save_heatmap, to_markdown  # noqa: E402
from bitladder.train import cap_workers, evaluate, pick_device  # noqa: E402


def parse_probes(specs: List[str]) -> List[Tuple[str, float]]:
    probes = []
    for spec in specs:
        kind, _, vals = spec.partition(":")
        if kind == "ternary":
            probes.append(("ternary", 0.0))
            continue
        for v in vals.split(","):
            if v:
                probes.append((kind, float(v)))
    return probes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="result JSON of a run.py experiment")
    p.add_argument("--cache", default=str(ROOT / "data" / "kws_cache"))
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--probes", nargs="*", default=["bits:8,4,2", "ternary", "prune:0.25,0.5", "sparsity:0.5,0.8"])
    p.add_argument("--batches", type=int, default=10, help="batches for activation statistics")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-heatmap", action="store_true")
    p.add_argument("--out", default=None, help="output stem; defaults to results/<run name>_analysis")
    args = p.parse_args()

    device = pick_device(args.device)
    model, meta = load_run(args.run)
    model.to(device)
    cache = KWSCache(args.cache)
    keywords = [c for c in meta["classes"] if c not in (SILENCE, UNKNOWN)]
    ds = KWSDataset(cache, args.split, keywords=keywords, exclude=meta.get("exclude", []), seed=meta["seed"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cap_workers(args.workers))
    print(f"{meta['name']}: {len(ds)} {args.split} samples, classes={ds.classes}")

    def eval_fn(m):
        m.to(device)
        return evaluate(m, loader, device)["acc"]

    rep = analysis_report(model, eval_fn, loader, parse_probes(args.probes), device, args.batches, log=print)
    rep["run"] = meta["name"]
    rep["split"] = args.split
    stem = Path(args.out) if args.out else Path(args.run).parent / f"{meta['name']}_analysis"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    md = f"# Layer analysis: {meta['name']} ({args.split})\n\n" + to_markdown(rep)
    stem.with_suffix(".md").write_text(md)
    print(md)
    if not args.no_heatmap and "sensitivity" in rep:
        png = stem.parent / f"{meta['name']}_sensitivity.png"
        save_heatmap(rep["sensitivity"], str(png), title=f"{meta['name']}: accuracy drop, % ({args.split})")
        print(f"heatmap: {png}")
    print(f"wrote {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
