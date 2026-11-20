"""Export a trained KWS model as a verified BLP1 packed binary and JSON report.

python audio/kws/export_packed.py --run results/kws12_edge-stem4fc2_prune50_inner-u4a8_s0.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audio.kws.checkpoint import load_run  # noqa: E402
from bitladder.measure import model_size_report  # noqa: E402
from bitladder.packed import pack_model, verify_packed_model  # noqa: E402
from bitladder.quant import fold_batchnorm  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None, help="default results/<run name>_packed.blpk")
    p.add_argument("--onnx", default=None, help="optional existing ONNX file whose byte size is added to the report")
    args = p.parse_args()

    model, meta = load_run(args.run)
    model.eval()
    example = torch.zeros(1, 1, 49, 40)
    original = copy.deepcopy(model)
    folded = fold_batchnorm(model, example, inplace=True).eval()
    with torch.no_grad():
        fold_diff = float((original(example) - folded(example)).abs().max())

    blob, report = pack_model(folded)
    verify_packed_model(folded, blob)
    out = Path(args.out) if args.out else Path(args.run).parent / f"{meta['name']}_packed.blpk"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    report.update({"run": meta["name"], "path": str(out), "verified": True,
                   "fold_max_abs_diff": fold_diff,
                   "source_model_bytes": model_size_report(original)["total_bytes"],
                   "folded_model_bytes": model_size_report(folded)["total_bytes"]})
    if args.onnx:
        report["onnx_bytes"] = Path(args.onnx).stat().st_size
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"EXPORT {out}: payload={report['payload_bytes']} manifest={report['manifest_bytes']} "
          f"total={report['total_bytes']} fold_diff={fold_diff:.3g} verified={report['verified']}")


if __name__ == "__main__":
    main()
