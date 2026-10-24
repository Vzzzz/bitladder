"""Speech Commands v2 datasets on top of the feature cache (see cache.py).

Protocol follows the TensorFlow / Hello Edge recipe: ``keywords`` are the positive
classes, every other word is ``_unknown_`` (subsampled to ``unknown_ratio`` of the
keyword count), and ``_silence_`` clips are low-gain background noise. Words listed in
``exclude`` are removed from the pool entirely (used to hold out "new" words for the
modular-expansion experiment, P-001).

Splits come from the official validation_list / testing_list. The official test set
archive (speech_commands_test_set_v0.02) is exposed separately by ``KWSOfficialTest``;
it already contains balanced ``_silence_`` and ``_unknown_`` folders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

KEYWORDS10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE, UNKNOWN = "_silence_", "_unknown_"
SPLITS = {"train": 0, "validation": 1, "test": 2}


def class_names(keywords: Sequence[str]) -> List[str]:
    return list(keywords) + [SILENCE, UNKNOWN]


class KWSCache:
    """Memory-mapped feature cache produced by cache.py."""

    def __init__(self, path: str = "data/kws_cache"):
        self.path = Path(path)
        self.index = json.loads((self.path / "index.json").read_text())
        self.feats = np.load(self.path / "feats.npy", mmap_mode="r")
        self.silence = np.load(self.path / "silence.npy", mmap_mode="r")
        self.words = np.array(self.index["words"])
        self.split = np.array(self.index["split"])
        self.mean, self.std = self.index["mean"], self.index["std"]
        self.frames, self.n_mels = self.index["frames"], self.index["n_mels"]
        tp = self.path / "test_index.json"
        self.test_index = json.loads(tp.read_text()) if tp.exists() else None
        self.test_feats = np.load(self.path / "test_feats.npy", mmap_mode="r") if tp.exists() else None
        cv = self.path / "cv_test.npy"
        self.cv_feats = np.load(cv, mmap_mode="r") if cv.exists() else None
        cvt = self.path / "cv_train.npy"
        self.cv_train = np.load(cvt, mmap_mode="r") if cvt.exists() else None

    @property
    def all_words(self) -> List[str]:
        return sorted(set(self.words.tolist()))

    def normalize(self, x: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(np.asarray(x, dtype=np.float32)) - self.mean) / self.std


def spec_augment(x: torch.Tensor, rng: np.random.Generator, max_t: int = 10, max_f: int = 6) -> torch.Tensor:
    """One time mask and one frequency mask, in place. x: (frames, n_mels)."""
    T, F = x.shape
    t = int(rng.integers(0, max_t + 1))
    t0 = int(rng.integers(0, max(T - t, 1)))
    x[t0:t0 + t, :] = 0.0
    f = int(rng.integers(0, max_f + 1))
    f0 = int(rng.integers(0, max(F - f, 1)))
    x[:, f0:f0 + f] = 0.0
    return x


class KWSDataset(Dataset):
    """Returns (features (1, frames, n_mels) float32 normalised, label)."""

    def __init__(self, cache: KWSCache, split: str = "train", keywords: Sequence[str] = KEYWORDS10,
                 exclude: Iterable[str] = (), unknown_ratio: float = 0.1, silence_ratio: float = 0.1,
                 seed: int = 0, augment: bool = False, cv_unknown: int = 0):
        """``cv_unknown``: number of Common Voice speech windows (cv_train.npy) added to the
        ``_unknown_`` class of the train split; the validation split gets a tenth of that
        from a disjoint slice. The test split never uses them."""
        self.cache = cache
        self.classes = class_names(keywords)
        self.augment = augment
        self.rng = np.random.default_rng(seed + SPLITS[split])
        kw = set(keywords)
        ex = set(exclude)
        if kw & ex:
            raise ValueError(f"keywords and exclude overlap: {kw & ex}")
        in_split = cache.split == SPLITS[split]
        is_kw = np.isin(cache.words, list(kw))
        is_unk = ~is_kw & ~np.isin(cache.words, list(ex))
        kw_idx = np.flatnonzero(in_split & is_kw)
        unk_idx = np.flatnonzero(in_split & is_unk)
        n_kw = len(kw_idx)
        rng = np.random.default_rng(seed)
        unk_idx = rng.permutation(unk_idx)[: int(n_kw * unknown_ratio)]
        # Silence rows are split 80/10/10 so that the splits never share a crop.
        S = cache.silence.shape[0]
        bounds = {0: (0, int(0.8 * S)), 1: (int(0.8 * S), int(0.9 * S)), 2: (int(0.9 * S), S)}
        lo, hi = bounds[SPLITS[split]]
        sil = rng.permutation(np.arange(lo, hi))[: int(n_kw * silence_ratio)]
        word_to_label = {w: i for i, w in enumerate(keywords)}
        self.items: List[Tuple[str, int, int]] = (
            [("feats", int(i), word_to_label[cache.words[i]]) for i in kw_idx]
            + [("feats", int(i), self.classes.index(UNKNOWN)) for i in unk_idx]
            + [("silence", int(i), self.classes.index(SILENCE)) for i in sil]
        )
        if cv_unknown and split != "test":
            if cache.cv_train is None:
                raise FileNotFoundError("cache has no cv_train.npy; run cv_negatives.py --split train")
            n_cv = cache.cv_train.shape[0]
            n_val = n_cv // 10
            lo, hi = (n_val, n_cv) if split == "train" else (0, n_val)
            n_take = cv_unknown if split == "train" else max(cv_unknown // 10, 1)
            cv_idx = rng.permutation(np.arange(lo, hi))[:n_take]
            self.items += [("cv", int(i), self.classes.index(UNKNOWN)) for i in cv_idx]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        src, i, label = self.items[idx]
        arr = (self.cache.feats[i] if src == "feats" else
               self.cache.silence[i] if src == "silence" else self.cache.cv_train[i])
        x = self.cache.normalize(arr)
        if self.augment:
            x = spec_augment(x, self.rng)
        return x.unsqueeze(0), label

    def label_counts(self) -> dict:
        counts = np.bincount([it[2] for it in self.items], minlength=len(self.classes))
        return dict(zip(self.classes, counts.tolist()))


class CVNegatives(Dataset):
    """1 s windows of Common Voice speech (see cv_negatives.py); label -1 = no keyword."""

    def __init__(self, cache: KWSCache):
        if cache.cv_feats is None:
            raise FileNotFoundError("cache has no cv_test.npy; run audio/kws/cv_negatives.py")
        self.cache = cache

    def __len__(self) -> int:
        return self.cache.cv_feats.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.cache.normalize(self.cache.cv_feats[idx]).unsqueeze(0), -1


class KWSOfficialTest(Dataset):
    """Official 12-class test set. Words outside ``keywords`` are labelled ``_unknown_``."""

    def __init__(self, cache: KWSCache, keywords: Sequence[str] = KEYWORDS10):
        if cache.test_index is None:
            raise FileNotFoundError("cache has no official test set; rerun cache.py with --test-root")
        self.cache = cache
        self.classes = class_names(keywords)
        lookup = {w: i for i, w in enumerate(self.classes)}
        unk = lookup[UNKNOWN]
        self.labels = [lookup.get(w, unk) for w in cache.test_index["words"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.cache.normalize(self.cache.test_feats[idx]).unsqueeze(0), self.labels[idx]
