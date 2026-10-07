"""Run the bit ladder on MNIST and collect a CSV.

Ladder: fp32, fp16, uniform 8/4/2, ternary (1.58), binary (1), each weight-only and
with 8-bit PACT activations. First/last layer stay at 8 bits unless --no-first-last.

python sandbox_mnist/sweep.py --epochs 5 --width 8
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

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

FIELDS = ["label", "scheme", "wbits", "abits", "test_acc", "total_params", "total_bytes",
          "zlib_bytes", "fp32_bytes", "mean_wbits", "macs", "bops", "train_time_s"]


def main() -> None:
    parser = build_parser()
    parser.add_argument("--act-bits", type=int, nargs="*", default=[None, 8],
                        help="activation settings to sweep; use 0 for float")
    parser.add_argument("--csv", default=str(ROOT / "results" / "ladder.csv"))
    args = parser.parse_args()
    act_settings = [None if a in (None, 0) else a for a in args.act_bits]

    rows = []
    for scheme, wbits in LADDER:
        for abits in act_settings:
            if scheme in ("fp32", "fp16") and abits is not None:
                continue
            a = argparse_copy(args)
            a.scheme, a.wbits, a.abits, a.name = scheme, wbits, abits, None
            res = run(a)
            rows.append({
                "label": res["label"], "scheme": scheme, "wbits": wbits, "abits": abits,
                **{k: res[k] for k in FIELDS if k in res},
            })
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print(f"wrote {args.csv} ({len(rows)} rows)")


def argparse_copy(ns):
    import argparse
    return argparse.Namespace(**vars(ns))


if __name__ == "__main__":
    main()
