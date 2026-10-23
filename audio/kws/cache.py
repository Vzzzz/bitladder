"""Precompute log-mel features for Speech Commands v2 once; training reads a memmap.

Reading 105k small WAVs through Python every epoch is the bottleneck on this dataset,
so we run the LogMel front-end once and store float16 features:

    <out>/feats.npy        (N, frames, n_mels) float16, one row per WAV in index.json
    <out>/index.json       files, word per file, split per file (0 train / 1 val / 2 test),
                           normalisation mean/std over the train split
    <out>/silence.npy      (S, frames, n_mels) float16 random crops of _background_noise_
    <out>/test_feats.npy   official test set (speech_commands_test_set_v0.02), optional
    <out>/test_index.json  class folder per row

python audio/kws/cache.py --root data/speech_commands_v2 --test-root data/speech_commands_test_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audio.kws.features import LogMel  # noqa: E402

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE
NOISE_DIR = "_background_noise_"


def read_wav(path: Path) -> np.ndarray:
    """16-bit PCM mono WAV -> float32 in [-1, 1]. Standard library only."""
    import wave

    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def fit_length(x: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


@torch.no_grad()
def featurize(paths: List[Path], fe: LogMel, device: torch.device, batch: int = 512,
              log=print) -> np.ndarray:
    out = None
    t0 = time.time()
    for i in range(0, len(paths), batch):
        wavs = np.stack([fit_length(read_wav(p)) for p in paths[i:i + batch]])
        f = fe(torch.from_numpy(wavs).to(device)).to(torch.float16).cpu().numpy()
        if out is None:
            out = np.empty((len(paths), *f.shape[1:]), dtype=np.float16)
        out[i:i + len(f)] = f
        if (i // batch) % 20 == 0:
            log(f"  {i + len(f)}/{len(paths)} clips, {time.time() - t0:.0f}s")
    return out


def build_cache(root: str, out: str, test_root: Optional[str] = None, n_silence: int = 4000,
                seed: int = 0, device: str = "auto", log=print) -> None:
    root_p, out_p = Path(root), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       "cpu" if device == "auto" else device)
    fe = LogMel().to(dev)

    val = set((root_p / "validation_list.txt").read_text().split())
    test = set((root_p / "testing_list.txt").read_text().split())
    paths = sorted(p for p in root_p.glob("*/*.wav") if p.parent.name != NOISE_DIR)
    rel = [f"{p.parent.name}/{p.name}" for p in paths]
    split = [2 if r in test else 1 if r in val else 0 for r in rel]
    log(f"{len(paths)} clips: train {split.count(0)}, val {split.count(1)}, test {split.count(2)}")

    feats = featurize(paths, fe, dev, log=log)
    np.save(out_p / "feats.npy", feats)

    # Silence: random 1 s crops of background noise at low gain (TF recipe), split 80/10/10.
    rng = np.random.default_rng(seed)
    noise = [read_wav(p) for p in sorted((root_p / NOISE_DIR).glob("*.wav"))]
    crops = []
    for _ in range(n_silence):
        src = noise[rng.integers(len(noise))]
        start = rng.integers(0, max(len(src) - CLIP_SAMPLES, 1))
        crops.append(fit_length(src[start:start + CLIP_SAMPLES]) * rng.uniform(0.0, 0.3))
    with torch.no_grad():
        sil = fe(torch.from_numpy(np.stack(crops)).to(dev)).to(torch.float16).cpu().numpy()
    np.save(out_p / "silence.npy", sil)

    train_rows = feats[np.array(split) == 0].astype(np.float32)
    index = {
        "files": rel,
        "words": [r.split("/")[0] for r in rel],
        "split": split,
        "frames": int(feats.shape[1]),
        "n_mels": int(feats.shape[2]),
        "mean": float(train_rows.mean()),
        "std": float(train_rows.std()),
        "n_silence": int(n_silence),
    }
    (out_p / "index.json").write_text(json.dumps(index))
    log(f"wrote {out_p / 'feats.npy'} {feats.shape}, silence {sil.shape}, "
        f"mean={index['mean']:.3f} std={index['std']:.3f}")

    if test_root:
        tp = Path(test_root)
        tpaths = sorted(p for p in tp.glob("*/*.wav"))
        tfeats = featurize(tpaths, fe, dev, log=log)
        np.save(out_p / "test_feats.npy", tfeats)
        (out_p / "test_index.json").write_text(json.dumps({
            "files": [f"{p.parent.name}/{p.name}" for p in tpaths],
            "words": [p.parent.name for p in tpaths],
        }))
        log(f"wrote official test set {tfeats.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/speech_commands_v2")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--out", default="data/kws_cache")
    ap.add_argument("--n-silence", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    build_cache(a.root, a.out, a.test_root, a.n_silence, a.seed, a.device)


if __name__ == "__main__":
    main()
