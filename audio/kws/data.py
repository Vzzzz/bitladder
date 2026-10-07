"""Google Speech Commands v2 with the official 12-class protocol.

Not yet exercised on real data. Expects the extracted archive at ``root``:
root/<word>/<file>.wav, root/_background_noise_/*.wav, root/validation_list.txt,
root/testing_list.txt.
"""

from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE, UNKNOWN = "_silence_", "_unknown_"
CLASSES = KEYWORDS + [SILENCE, UNKNOWN]
SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE  # 1 second


def read_wav(path: Path) -> np.ndarray:
    """16-bit PCM mono WAV -> float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _fit_length(x: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


class SpeechCommands(Dataset):
    """split in {train, validation, test}. Returns (waveform[16000], label_index)."""

    def __init__(self, root: str, split: str = "train", unknown_ratio: float = 0.1,
                 silence_ratio: float = 0.1, seed: int = 0):
        self.root = Path(root)
        self.split = split
        rng = random.Random(seed)
        val = set((self.root / "validation_list.txt").read_text().split())
        test = set((self.root / "testing_list.txt").read_text().split())

        items: List[Tuple[Path, int]] = []
        unknown: List[Path] = []
        for wav in sorted(self.root.glob("*/*.wav")):
            word = wav.parent.name
            if word == "_background_noise_":
                continue
            rel = f"{word}/{wav.name}"
            in_split = (rel in test) if split == "test" else (rel in val) if split == "validation" else (rel not in val and rel not in test)
            if not in_split:
                continue
            if word in KEYWORDS:
                items.append((wav, CLASSES.index(word)))
            else:
                unknown.append(wav)
        n_kw = len(items)
        rng.shuffle(unknown)
        n_unknown = int(n_kw * unknown_ratio) if split == "train" else len(unknown) if split == "test" else int(n_kw * unknown_ratio)
        items += [(p, CLASSES.index(UNKNOWN)) for p in unknown[:n_unknown]]
        self.items = items
        self.n_silence = int(n_kw * silence_ratio)
        self.noise = [read_wav(p) for p in sorted((self.root / "_background_noise_").glob("*.wav"))]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items) + self.n_silence

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < len(self.items):
            path, label = self.items[idx]
            x = _fit_length(read_wav(path))
        else:
            label = CLASSES.index(SILENCE)
            x = np.zeros(CLIP_SAMPLES, dtype=np.float32)
            if self.noise:
                src = self.noise[self.rng.integers(len(self.noise))]
                start = self.rng.integers(0, max(len(src) - CLIP_SAMPLES, 1))
                x = _fit_length(src[start:start + CLIP_SAMPLES]) * self.rng.uniform(0.0, 0.3)
        return torch.from_numpy(np.ascontiguousarray(x)), label
