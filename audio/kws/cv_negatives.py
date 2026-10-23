"""Hard negatives for threshold calibration from Common Voice (speech without keywords).

P-001 (Yandex) calibrates per-keyword thresholds at a fixed FAR on the Common Voice test
split. Negatives drawn from the other Speech Commands words are too easy: the frozen core
fires on 56–74 % of clips of unseen words (see docs/results/kws_expansion_2026-10-23.md).

This script decodes the ``test.tsv`` clips of a Common Voice language folder (mp3 via
soundfile / libsndfile >= 1.1), resamples to 16 kHz, cuts non-overlapping 1 s windows,
drops near-silent windows, peak-normalises, and stores log-mel features next to the
Speech Commands cache:

    <cache>/cv_test.npy         (N, frames, n_mels) float16
    <cache>/cv_test_index.json  clip file + offset per row, parameters

python audio/kws/cv_negatives.py --cv data/cv/cv-corpus-27.0-2026-09-11/ru --max-windows 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audio.kws.features import LogMel  # noqa: E402

SAMPLE_RATE = 16000
WINDOW = SAMPLE_RATE  # 1 s, same as Speech Commands clips


def load_clip(path: Path) -> np.ndarray:
    """Decode to mono float32 at 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return x


def windows(x: np.ndarray, min_rms: float, peak: float) -> List[Tuple[int, np.ndarray]]:
    out = []
    for start in range(0, len(x) - WINDOW + 1, WINDOW):
        w = x[start:start + WINDOW]
        if np.sqrt(np.mean(w ** 2)) < min_rms:
            continue  # leading / trailing silence of the recording
        w = w * (peak / max(np.abs(w).max(), 1e-6))
        out.append((start, w.astype(np.float32)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", required=True, help="Common Voice language folder with test.tsv and clips/")
    ap.add_argument("--split", default="test", help="tsv name without extension")
    ap.add_argument("--cache", default="data/kws_cache")
    ap.add_argument("--max-windows", type=int, default=20000)
    ap.add_argument("--min-rms", type=float, default=0.01)
    ap.add_argument("--peak", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cv = Path(a.cv)
    with open(cv / f"{a.split}.tsv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(rows)
    print(f"{len(rows)} clips in {a.split}.tsv")

    fe = LogMel().to(a.device)
    feats, index = [], []
    t0 = time.time()
    n_clips = 0
    for row in rows:
        if len(index) >= a.max_windows:
            break
        path = cv / "clips" / row["path"]
        try:
            x = load_clip(path)
        except Exception as e:  # noqa: BLE001
            print(f"skip {path.name}: {e}")
            continue
        n_clips += 1
        for start, w in windows(x, a.min_rms, a.peak):
            feats.append(w)
            index.append({"file": row["path"], "offset": int(start)})
        if n_clips % 500 == 0:
            print(f"  {n_clips} clips, {len(index)} windows, {time.time() - t0:.0f}s")
    feats = feats[: a.max_windows]
    index = index[: a.max_windows]

    out = np.empty((len(feats), fe.num_frames(WINDOW), fe.fb.shape[0]), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(feats), 512):
            batch = torch.from_numpy(np.stack(feats[i:i + 512])).to(a.device)
            out[i:i + len(batch)] = fe(batch).to(torch.float16).cpu().numpy()
    cache = Path(a.cache)
    np.save(cache / f"cv_{a.split}.npy", out)
    (cache / f"cv_{a.split}_index.json").write_text(json.dumps({
        "source": str(cv), "split": a.split, "clips": n_clips, "windows": index,
        "min_rms": a.min_rms, "peak": a.peak, "seed": a.seed}))
    print(f"wrote {cache / f'cv_{a.split}.npy'} {out.shape} from {n_clips} clips in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
