"""Feature cache + dataset protocol on a synthetic mini Speech Commands tree."""

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio.kws.cache import build_cache  # noqa: E402
from audio.kws.data import SILENCE, UNKNOWN, KWSCache, KWSDataset, KWSOfficialTest  # noqa: E402

WORDS = ["yes", "no", "stop", "go", "cat", "dog"]


def _write_wav(path: Path, n: int, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = (rng.standard_normal(n) * 3000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.tobytes())


@pytest.fixture(scope="module")
def mini_cache(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("sc")
    val, test = [], []
    for w in WORDS:
        for i in range(20):
            rel = f"{w}/spk{i:02d}_nohash_0.wav"
            _write_wav(root / rel, 16000 if i % 3 else 12000, rng)  # some short clips
            (val if i % 10 == 8 else test if i % 10 == 9 else []).append(rel)
    (root / "validation_list.txt").write_text("\n".join(val) + "\n")
    (root / "testing_list.txt").write_text("\n".join(test) + "\n")
    _write_wav(root / "_background_noise_" / "white.wav", 16000 * 5, rng)
    troot = tmp_path_factory.mktemp("sc_test")
    for w in ["yes", "no", "stop", "go", SILENCE, UNKNOWN]:
        for i in range(3):
            _write_wav(troot / w / f"t{i}.wav", 16000, rng)
    out = tmp_path_factory.mktemp("cache")
    build_cache(str(root), str(out), test_root=str(troot), n_silence=50, device="cpu", log=lambda s: None)
    return KWSCache(str(out))


def test_cache_shapes(mini_cache):
    c = mini_cache
    assert c.feats.shape == (len(WORDS) * 20, 49, 40)
    assert c.feats.dtype == np.float16
    assert c.silence.shape == (50, 49, 40)
    assert (c.split == 0).sum() == len(WORDS) * 16 and (c.split == 1).sum() == len(WORDS) * 2
    assert c.test_feats.shape == (18, 49, 40)


def test_dataset_protocol_and_exclude(mini_cache):
    kw = ["yes", "no"]
    ds = KWSDataset(mini_cache, "train", keywords=kw, exclude=["stop", "go"], unknown_ratio=0.5, silence_ratio=0.25)
    counts = ds.label_counts()
    assert counts["yes"] == 16 and counts["no"] == 16
    assert counts[UNKNOWN] == 16 and counts[SILENCE] == 8
    unk_words = {mini_cache.words[i] for src, i, lab in ds.items if src == "feats" and lab == ds.classes.index(UNKNOWN)}
    assert unk_words <= {"cat", "dog"}, "excluded words must not leak into unknown"
    x, y = ds[0]
    assert x.shape == (1, 49, 40) and x.dtype == np.float32 or str(x.dtype) == "torch.float32"
    assert abs(float(x.mean())) < 5  # normalised


def test_splits_disjoint(mini_cache):
    sets = []
    for split in ("train", "validation", "test"):
        ds = KWSDataset(mini_cache, split, keywords=["yes", "no"], unknown_ratio=1.0, silence_ratio=1.0)
        sets.append({(src, i) for src, i, _ in ds.items})
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_official_test_maps_unlisted_words_to_unknown(mini_cache):
    ds = KWSOfficialTest(mini_cache, keywords=["yes", "no"])
    assert len(ds) == 18
    labels = np.array(ds.labels)
    unk = ds.classes.index(UNKNOWN)
    assert (labels == unk).sum() == 3 * 3  # stop, go, _unknown_ folders
    assert (labels == ds.classes.index(SILENCE)).sum() == 3


def test_keywords_overlap_exclude_raises(mini_cache):
    with pytest.raises(ValueError):
        KWSDataset(mini_cache, "train", keywords=["yes"], exclude=["yes"])
