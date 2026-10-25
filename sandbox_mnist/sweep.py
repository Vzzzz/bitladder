"""Run the bit ladder on MNIST over several model widths and collect a CSV + Pareto table.

Ladder: fp32, fp16, uniform 8/4/2, ternary (1.58), binary (1), each weight-only and
with 8-bit PACT activations. First/last layer stay at 8 bits unless --no-first-last.
Widths scale the channel count of TinyCNN (or the hidden size of TinyMLP), so the sweep
answers "smallest model in bytes at a given accuracy".

python sandbox_mnist/sweep.py --epochs 10 --widths 4,6,8,12,16
python sandbox_mnist/sweep.py --epochs 10 --widths 8 --act-bits 0 8 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import build_parser, run  # noqa: E402

LADDER = [
    ("fp32", 32),
    ("fp16", 16),
    ("uniform", 8),
    ("uniform", 4),
    ("uniform", 2),
    ("ternary", 2),
    ("binary", 1),
]

FIELDS = ["label", "width", "scheme", "wbits", "abits", "test_acc", "total_params", "total_bytes",
          "zlib_bytes", "fp32_bytes", "mean_wbits", "macs", "bops", "train_time_s"]


def pareto(rows: List[Dict]) -> List[Dict]:
    """Rows not dominated by a smaller-or-equal model with higher-or-equal accuracy."""
    out = []
    for r in rows:
        dominated = any(o is not r and o["total_bytes"] <= r["total_bytes"] and o["test_acc"] >= r["test_acc"]
                        and (o["total_bytes"] < r["total_bytes"] or o["test_acc"] > r["test_acc"]) for o in rows)
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: r["total_bytes"])


def to_markdown(rows: List[Dict], front: List[Dict]) -> str:
    lines = ["| label | width | wbits | abits | acc | params | bytes | zlib | MACs | BOPs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['width']} | {r['wbits']} | {r['abits'] or 'fp'} | "
                     f"{100*r['test_acc']:.2f} % | {r['total_params']} | {r['total_bytes']} | {r['zlib_bytes']} | "
                     f"{r['macs']:,} | {r['bops']:,.0f} |")
    lines += ["", "Pareto front (bytes vs accuracy):", "", "| bytes | acc | label | width |", "|---|---|---|---|"]
    for r in front:
        lines.append(f"| {r['total_bytes']} | {100*r['test_acc']:.2f} % | {r['label']} | {r['width']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = build_parser()
    parser.add_argument("--widths", default="8", help="comma-separated widths to sweep")
    parser.add_argument("--act-bits", type=int, nargs="*", default=[0, 8],
                        help="activation settings to sweep; 0 = float activations")
    parser.add_argument("--csv", default=str(ROOT / "results" / "ladder.csv"))
    args = parser.parse_args()
    act_settings = [None if a == 0 else a for a in args.act_bits]
    widths = [int(w) for w in args.widths.split(",") if w]

    rows = []
    for width in widths:
        for scheme, wbits in LADDER:
            for abits in act_settings:
                if scheme in ("fp32", "fp16") and abits is not None:
                    continue
                a = argparse.Namespace(**vars(args))
                a.scheme, a.wbits, a.abits, a.width, a.name = scheme, wbits, abits, width, None
                res = run(a)
                rows.append({"label": res["label"], "width": width, "scheme": scheme, "wbits": wbits, "abits": abits,
                             **{k: res[k] for k in FIELDS if k in res}})
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    front = pareto(rows)
    md = to_markdown(rows, front)
    csv_path.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {csv_path} ({len(rows)} rows) and {csv_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
