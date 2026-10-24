"""Modular network expansion for new keywords (P-001, Yandex, arXiv 2607.19918) on DS-CNN.

The base DS-CNN (trained by run.py on the *base* keywords) is frozen entirely, BatchNorm
statistics included. A small branch taps the output of every DS block, reduces it to
``k`` channels, concatenates with the previous branch block, mixes with a 3x3 conv, and
ends in its own head over the *new* keywords plus background. Only the branch trains.

Decision rule (core-first): the base head is consulted first; the new head only matters
when the base says background (silence / unknown). Metric as in the paper: per-word
threshold calibrated to FAR 1 % on negatives, macro FRR over the new words.

Modes
  expand   the proposed branch (default)
  separate parameter-matched independent small DS-CNN on the new words ("Ensemble" in the paper)
  head     only a new linear head on the frozen pooled base features ("Head-Only")

Quantization: the base comes with whatever scheme it was trained with (its JSON next to
the checkpoint is used to rebuild it); --scheme/--wbits/... apply to the branch only.

python audio/kws/run.py --keywords yes,no,up,down,left,right,on,off --exclude stop,go --name base8
python audio/kws/expand.py --base results/base8.json --new stop,go
python audio/kws/expand.py --base results/base8.json --new stop,go --scheme uniform --wbits 4 --no-first-last
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audio.kws.checkpoint import load_run  # noqa: E402
from audio.kws.data import SILENCE, UNKNOWN, CVNegatives, KWSCache, KWSDataset  # noqa: E402
from audio.kws.model import DSCNN  # noqa: E402
from bitladder.cli import add_quant_args, quant_config  # noqa: E402
from bitladder.measure import count_macs, model_size_report  # noqa: E402
from bitladder.quant import QuantConv2d, QuantLinear, apply_quantization  # noqa: E402
from bitladder.train import pick_device  # noqa: E402


# ----------------------------------------------------------------------------- models
def load_base(json_path: str) -> Tuple[DSCNN, dict]:
    """Rebuild a run.py model (with its quantizers) from the result JSON + checkpoint, frozen."""
    return load_run(json_path, freeze=True)


def base_forward(base: DSCNN, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Base logits plus the output of every DS block (the taps)."""
    x = F.relu(base.bn(base.stem(x)))
    taps = []
    for blk in base.blocks:
        x = blk(x)
        taps.append(x)
    logits = base.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
    return logits, taps


class ExpandBlock(nn.Module):
    """reduce (1x1, c_base -> k) | concat previous | mix (3x3, or depthwise 3x3 + 1x1) | BN | hard-swish.

    Dense mix: c_base*k + 9*k*(k+k_prev) params; depthwise-separable: c_base*k + 9*(k+k_prev) + (k+k_prev)*k.
    """

    def __init__(self, c_base: int, k: int, k_prev: int, separable: bool = False):
        super().__init__()
        self.reduce = QuantConv2d(c_base, k, 1, bias=False)
        cin = k + k_prev
        if separable:
            self.mix = nn.Sequential(QuantConv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                                     QuantConv2d(cin, k, 1, bias=False))
        else:
            self.mix = QuantConv2d(cin, k, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(k)

    def forward(self, tap: torch.Tensor, prev: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.reduce(tap)
        if prev is not None:
            h = torch.cat([h, prev], dim=1)
        return F.hardswish(self.bn(self.mix(h)))


class ExpansionBranch(nn.Module):
    """Taps every DS block of the frozen base; see ExpandBlock for the parameter count."""

    def __init__(self, c_base: int, blocks: int, k: int, n_new: int, separable: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ExpandBlock(c_base, k, 0 if i == 0 else k, separable) for i in range(blocks)])
        self.head = QuantLinear(k, n_new + 1)  # new words + background

    def forward(self, taps: List[torch.Tensor]) -> torch.Tensor:
        h = None
        for blk, tap in zip(self.blocks, taps):
            h = blk(tap, h)
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))


class HeadOnly(nn.Module):
    def __init__(self, c_base: int, n_new: int):
        super().__init__()
        self.head = QuantLinear(c_base, n_new + 1)

    def forward(self, taps: List[torch.Tensor]) -> torch.Tensor:
        return self.head(F.adaptive_avg_pool2d(taps[-1], 1).flatten(1))


class Separate(nn.Module):
    """Independent small DS-CNN on the raw features; ignores the taps."""

    def __init__(self, channels: int, blocks: int, n_new: int):
        super().__init__()
        self.net = DSCNN(channels=channels, blocks=blocks, num_classes=n_new + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def branch_logits(branch: nn.Module, taps: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return branch(x) if isinstance(branch, Separate) else branch(taps)


# ----------------------------------------------------------------------------- protocol
@torch.no_grad()
def collect_core(base: DSCNN, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities of the frozen core and labels."""
    base.eval()
    probs, labels = [], []
    for x, y in loader:
        logits, _ = base_forward(base, x.to(device))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def thresholds_at_far(probs: np.ndarray, labels: np.ndarray, word_ids: List[int], far: float) -> np.ndarray:
    """Per word: the score threshold that gives ``far`` false accepts on all non-target samples."""
    thr = np.ones(probs.shape[1], dtype=np.float64)
    for w in word_ids:
        neg = probs[labels != w, w]
        thr[w] = float(np.quantile(neg, 1.0 - far)) if len(neg) else 1.0
    return thr


def frr_at_thresholds(probs: np.ndarray, labels: np.ndarray, word_ids: List[int], thr: np.ndarray) -> List[float]:
    out = []
    for w in word_ids:
        pos = probs[labels == w, w]
        out.append(float((pos < thr[w]).mean()) if len(pos) else float("nan"))
    return out


def far_at_thresholds(probs: np.ndarray, labels: np.ndarray, word_ids: List[int], thr: np.ndarray) -> List[float]:
    out = []
    for w in word_ids:
        neg = probs[labels != w, w]
        out.append(float((neg >= thr[w]).mean()) if len(neg) else float("nan"))
    return out


@torch.no_grad()
def collect_scores(base: DSCNN, branch: nn.Module, loader: DataLoader, device: torch.device,
                   core_word_ids: List[int], core_thr: np.ndarray, background_ids: List[int]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample: new-word probabilities (ungated), labels, whether the core fired (hard
    core-first rule), and the core's background probability (soft gate).

    Hard rule as in the paper: the core "outputs a keyword" only when the probability of
    one of its words exceeds that word's own threshold (calibrated at the target FAR);
    otherwise it outputs background and the new head is consulted. Soft gate: new-word
    score = p_branch(word) * p_core(background), no core thresholds needed.
    """
    base.eval()
    branch.eval()
    thr_t = torch.tensor(core_thr, dtype=torch.float32, device=device)
    ids = torch.tensor(core_word_ids, device=device)
    bg_ids = torch.tensor(background_ids, device=device)
    probs, labels, fired, bg = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        core_logits, taps = base_forward(base, x)
        core_p = F.softmax(core_logits, dim=1)
        fires = (core_p[:, ids] > thr_t[ids]).any(dim=1)
        p = F.softmax(branch_logits(branch, taps, x), dim=1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
        fired.append(fires.cpu().numpy())
        bg.append(core_p[:, bg_ids].sum(dim=1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(fired), np.concatenate(bg)


Scores = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (branch probs, core fired, core background prob)


def gated(scores: Scores, gate: str) -> np.ndarray:
    """New-word scores under a gating rule: none | hard (core-first) | soft (multiply)."""
    p, fired, bg = scores
    s = p.copy()
    if gate == "hard":
        s[fired] = 0.0
    elif gate == "soft":
        s = s * bg[:, None]
    elif gate != "none":
        raise ValueError(gate)
    return s


def frr_at_far(scores: Scores, labels: np.ndarray, n_new: int, far: float, gate: str = "none",
               neg: Optional[Scores] = None) -> Dict[str, object]:
    """Per new word: threshold at ``far``, then FRR on the positives; plus macro FRR.

    Thresholds come from the non-target samples of the set itself, or, with ``neg``,
    from an external negative set (Common Voice), under the same gating rule.
    """
    s_all = gated(scores, gate)
    ids = list(range(n_new))
    if neg is None:
        thr = thresholds_at_far(s_all, labels, ids, far)
    else:
        s_neg = gated(neg, gate)
        thr = thresholds_at_far(s_neg, np.full(len(s_neg), -1), ids, far)
    frr = frr_at_thresholds(s_all, labels, ids, thr)
    far_inset = far_at_thresholds(s_all, labels, ids, thr)
    return {"frr": frr, "thr": thr[:n_new].tolist(), "macro_frr": float(np.nanmean(frr)),
            "far_gsc": far_inset}


# ----------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_quant_args(p)
    p.add_argument("--base", required=True, help="result JSON of the base run (checkpoint .pt next to it)")
    p.add_argument("--new", required=True, help="comma-separated new keywords")
    p.add_argument("--mode", default="expand", choices=["expand", "separate", "head"])
    p.add_argument("--k", type=int, default=8, help="branch width (channels per expand block)")
    p.add_argument("--separable", action="store_true", help="depthwise-separable mix conv in the branch")
    p.add_argument("--sep-channels", type=int, default=40, help="channels of the separate baseline (40x2 ~ 6k params)")
    p.add_argument("--sep-blocks", type=int, default=2)
    p.add_argument("--cache", default=str(ROOT / "data" / "kws_cache"))
    p.add_argument("--unknown-ratio", type=float, default=1.0,
                   help="background clips per new-word clip (paper: new commands plus background)")
    p.add_argument("--silence-ratio", type=float, default=0.1)
    p.add_argument("--cv-unknown", type=int, default=0,
                   help="Common Voice windows added to the branch background (train)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--far", type=float, default=0.01)
    p.add_argument("--neg", default="gsc", choices=["gsc", "cv"],
                   help="negatives for threshold calibration: other GSC words + silence (gsc) "
                        "or Common Voice speech windows (cv, needs cv_test.npy in the cache)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--name", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    log = (lambda s: None) if args.quiet else print
    cfg = quant_config(args)

    base, meta = load_base(args.base)
    base.to(device)
    base_words = [c for c in meta["classes"] if c not in (SILENCE, UNKNOWN)]
    new_words = [w for w in args.new.split(",") if w]
    if set(new_words) & set(base_words):
        raise ValueError("new words overlap the base keywords")
    n_new = len(new_words)

    cache = KWSCache(args.cache)
    # Branch data: new words positive; everything else (base words included) is background.
    ds_kw = dict(keywords=new_words, unknown_ratio=args.unknown_ratio,
                 silence_ratio=args.silence_ratio, seed=args.seed, cv_unknown=args.cv_unknown)
    train_ds = KWSDataset(cache, "train", augment=True, **ds_kw)
    val_ds = KWSDataset(cache, "validation", **ds_kw)
    test_ds = KWSDataset(cache, "test", **ds_kw)
    log(f"base={base_words} new={new_words} train={len(train_ds)} counts={train_ds.label_counts()}")

    def mk(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size if shuffle else 512, shuffle=shuffle,
                          num_workers=args.workers, drop_last=shuffle, pin_memory=True)

    train_loader, val_loader, test_loader = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    # Core thresholds: calibrated once on the base's own view of the validation split
    # (its keywords vs. everything else except the new words), FAR = args.far per word.
    core_ids = [meta["classes"].index(w) for w in base_words]
    background_ids = [meta["classes"].index(SILENCE), meta["classes"].index(UNKNOWN)]
    base_view = dict(keywords=base_words, exclude=new_words, seed=args.seed)
    cv_loader = mk(CVNegatives(cache), False) if args.neg == "cv" else None
    if cv_loader is None:
        core_cal_p, core_cal_y = collect_core(base, mk(KWSDataset(cache, "validation", **base_view), False), device)
    else:
        core_cal_p, _ = collect_core(base, cv_loader, device)
        core_cal_y = np.full(len(core_cal_p), -1)
    core_thr = thresholds_at_far(core_cal_p, core_cal_y, core_ids, args.far)
    core_test_p, core_test_y = collect_core(base, mk(KWSDataset(cache, "test", **base_view), False), device)
    core_frr = frr_at_thresholds(core_test_p, core_test_y, core_ids, core_thr)
    core_far_gsc = far_at_thresholds(core_test_p, core_test_y, core_ids, core_thr)
    log(f"core thresholds from {args.neg}: FRR@FAR{args.far} on its own words (test) macro={np.nanmean(core_frr):.4f} "
        + " ".join(f"{w}={f:.3f}" for w, f in zip(base_words, core_frr))
        + f"; FAR on GSC test negatives at these thresholds macro={np.nanmean(core_far_gsc):.4f}")

    if args.mode == "expand":
        branch: nn.Module = ExpansionBranch(meta["channels"], meta["blocks"], args.k, n_new, args.separable)
    elif args.mode == "head":
        branch = HeadOnly(meta["channels"], n_new)
    else:
        branch = Separate(args.sep_channels, args.sep_blocks, n_new)
    apply_quantization(branch, cfg)
    branch.to(device)
    n_branch = sum(p.numel() for p in branch.parameters())
    log(f"[{args.mode}/{cfg.label}] branch params={n_branch}")

    # Branch labels: new word i -> i, silence/unknown -> n_new (single background class).
    def to_branch_label(y: torch.Tensor) -> torch.Tensor:
        return torch.where(y < n_new, y, torch.full_like(y, n_new))

    opt = torch.optim.Adam(branch.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.15)
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        branch.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), to_branch_label(y.to(device))
            with torch.no_grad():
                _, taps = base_forward(base, x)
            loss = F.cross_entropy(branch_logits(branch, taps, x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            n += len(y)
        def scores_of(loader):
            p, y, fired, bg = collect_scores(base, branch, loader, device, core_ids, core_thr, background_ids)
            return (p, fired, bg), y

        neg = scores_of(cv_loader)[0] if cv_loader is not None else None
        sc, labels = scores_of(val_loader)
        val = {g: frr_at_far(sc, labels, n_new, args.far, g, neg) for g in ("hard", "soft", "none")}
        rec = {"epoch": epoch, "train_loss": running / max(n, 1),
               "val_macro_frr": val["hard"]["macro_frr"], "val_macro_frr_soft": val["soft"]["macro_frr"],
               "val_macro_frr_ungated": val["none"]["macro_frr"]}
        history.append(rec)
        log(f"epoch={epoch} loss={rec['train_loss']:.4f} val_macro_frr@far{args.far}: "
            f"hard={rec['val_macro_frr']:.4f} soft={rec['val_macro_frr_soft']:.4f} "
            f"none={rec['val_macro_frr_ungated']:.4f}")
    train_time = time.time() - t0

    sc, labels = scores_of(test_loader)
    test = frr_at_far(sc, labels, n_new, args.far, "hard", neg)
    test_s = frr_at_far(sc, labels, n_new, args.far, "soft", neg)
    test_u = frr_at_far(sc, labels, n_new, args.far, "none", neg)
    # How often the frozen core fires (above its thresholds) on new-word clips: the branch
    # is never consulted there, so this is a floor for the hard-gated FRR.
    fired = sc[1]
    new_mask = labels < n_new
    core_fires_on_new = float(fired[new_mask].mean()) if new_mask.any() else float("nan")

    size = model_size_report(branch)
    macs = count_macs(branch, (1, cache.frames, cache.n_mels)) if args.mode == "separate" else None
    name = args.name or (f"expand_{args.mode}_{'-'.join(new_words)}_k{args.k}{'ds' if args.separable else ''}"
                         f"_{cfg.label}_s{args.seed}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(branch.state_dict(), out_dir / f"{name}.pt")
    result = {
        "name": name, "mode": args.mode, "separable": args.separable, "base": args.base, "base_label": meta["label"],
        "base_words": base_words, "new_words": new_words, "k": args.k, "config": vars(cfg),
        "branch_params": n_branch, "branch_bytes": size["total_bytes"], "branch_zlib_bytes": size["zlib_bytes"],
        "branch_macs": None if macs is None else sum(macs.values()),
        "far": args.far,
        "test_frr": dict(zip(new_words, test["frr"])), "test_thr": dict(zip(new_words, test["thr"])),
        "test_macro_frr": test["macro_frr"],
        "test_frr_ungated": dict(zip(new_words, test_u["frr"])), "test_macro_frr_ungated": test_u["macro_frr"],
        "test_far_gsc_ungated": dict(zip(new_words, test_u["far_gsc"])),
        "test_frr_soft": dict(zip(new_words, test_s["frr"])), "test_macro_frr_soft": test_s["macro_frr"],
        "neg": args.neg, "cv_unknown": args.cv_unknown,
        "core_fires_on_new": core_fires_on_new,
        "core_frr": dict(zip(base_words, core_frr)), "core_macro_frr": float(np.nanmean(core_frr)),
        "core_thr": dict(zip(base_words, core_thr[core_ids].tolist())),
        "epochs": args.epochs, "seed": args.seed, "train_time_s": train_time, "history": history,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    per_word = " ".join(f"{w}={f:.4f}" for w, f in zip(new_words, test["frr"]))
    log(f"RESULT {name}: macro_FRR@FAR{args.far} hard={test['macro_frr']:.4f} [{per_word}] "
        f"soft={test_s['macro_frr']:.4f} none={test_u['macro_frr']:.4f} core_fires_on_new={core_fires_on_new:.3f} "
        f"branch_params={n_branch} branch_bytes={size['total_bytes']} train={train_time:.0f}s")
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
