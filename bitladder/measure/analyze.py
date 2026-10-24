"""Per-layer analysis for pruning and mixed precision: static weight statistics,
activation statistics on data, and one-shot sensitivity probes.

Everything works on models built from ``QuantConv2d`` / ``QuantLinear``.

* ``weight_stats(model)``: per layer, without data. Parameter count, output channels,
  abs-mean / abs-max / std of weights, share of near-zero weights, share of weights the
  ternary rule would zero (|w| < mean|w| / 2 per channel), spread of channel L2 norms
  (min / median and coefficient of variation): a wide spread means cheap channel pruning.
* ``activation_stats(model, loader)``: per layer, on a few batches. Zero fraction of the
  layer input (post-ReLU sparsity), share of output channels that are almost never
  active (dead channels), mean |activation| per layer.
* ``sensitivity(model, eval_fn, probes)``: for each layer in turn, apply one probe to a
  copy of the model and re-evaluate, no retraining. Probes:
    ("bits", k)      weights of that layer quantized to k bits (uniform, per-channel)
    ("ternary", 0)   weights of that layer ternarized (BitNet absmean)
    ("prune", f)     the f lowest-L2-norm output channels of that layer zeroed
    ("sparsity", f)  the f smallest-|w| weights of that layer zeroed
  Result: metric per (layer, probe) plus the baseline; ``drop`` = baseline - metric.
* ``to_markdown(...)`` and ``save_heatmap(...)`` for docs/results.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..quant.layers import quant_layers
from ..quant.quantizers import TernaryQuantizer, UniformWeightQuantizer

Probe = Tuple[str, float]


# ----------------------------------------------------------------------------- static
@dataclass
class LayerStats:
    name: str
    kind: str              # conv | dwconv | linear
    params: int
    out_channels: int
    in_per_output: int     # fan-in of one output unit
    wbits: float
    abs_mean: float
    abs_max: float
    std: float
    near_zero_frac: float  # |w| < 1e-2 * abs_max
    ternary_zero_frac: float
    chan_norm_min_over_median: float
    chan_norm_cv: float    # std / mean of channel L2 norms


def _kind(m: nn.Module) -> str:
    if isinstance(m, nn.Conv2d):
        return "dwconv" if m.groups == m.in_channels and m.groups > 1 else "conv"
    return "linear"


@torch.no_grad()
def weight_stats(model: nn.Module) -> List[LayerStats]:
    out = []
    for name, m in quant_layers(model):
        w = m.weight.detach().float()
        flat = w.flatten(1)
        absw = w.abs()
        absmean_ch = absw.flatten(1).mean(1, keepdim=True)
        tern_zero = (absw.flatten(1) < 0.5 * absmean_ch).float().mean().item()
        norms = flat.norm(dim=1)
        out.append(LayerStats(
            name=name, kind=_kind(m), params=w.numel(), out_channels=w.shape[0],
            in_per_output=flat.shape[1], wbits=float(m.wbits),
            abs_mean=absw.mean().item(), abs_max=absw.max().item(), std=w.std().item(),
            near_zero_frac=(absw < 1e-2 * absw.max()).float().mean().item(),
            ternary_zero_frac=tern_zero,
            chan_norm_min_over_median=(norms.min() / norms.median().clamp_min(1e-12)).item(),
            chan_norm_cv=(norms.std() / norms.mean().clamp_min(1e-12)).item(),
        ))
    return out


# ----------------------------------------------------------------------------- activations
@dataclass
class ActStats:
    name: str
    input_zero_frac: float      # share of exactly-zero inputs (post-ReLU sparsity)
    input_abs_mean: float
    output_abs_mean: float
    dead_out_channels_frac: float  # output channels with mean |y| < 1e-3 * layer mean


@torch.no_grad()
def activation_stats(model: nn.Module, loader: Iterable, device: Optional[torch.device] = None,
                     max_batches: int = 10) -> List[ActStats]:
    device = device or next(model.parameters()).device
    acc: Dict[str, Dict[str, object]] = {}
    handles = []

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach()
            y = out.detach()
            ch_abs = y.abs().transpose(0, 1).flatten(1).mean(1)  # per output channel
            a = acc.setdefault(name, {"zero": 0.0, "in_abs": 0.0, "out_abs": 0.0, "n": 0, "ch": None})
            a["zero"] += (x == 0).float().mean().item()
            a["in_abs"] += x.abs().mean().item()
            a["out_abs"] += y.abs().mean().item()
            a["n"] += 1
            a["ch"] = ch_abs.cpu() if a["ch"] is None else a["ch"] + ch_abs.cpu()
        return fn

    for name, m in quant_layers(model):
        handles.append(m.register_forward_hook(hook(name)))
    was_training = model.training
    model.eval()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        model(x.to(device))
    model.train(was_training)
    for h in handles:
        h.remove()
    out = []
    for name, a in acc.items():
        n = max(a["n"], 1)
        ch = a["ch"] / n
        dead = (ch < 1e-3 * ch.mean().clamp_min(1e-12)).float().mean().item()
        out.append(ActStats(name, a["zero"] / n, a["in_abs"] / n, a["out_abs"] / n, dead))
    return out


# ----------------------------------------------------------------------------- probes
@torch.no_grad()
def apply_probe(layer: nn.Module, probe: Probe) -> None:
    """Modify one QuantConv2d / QuantLinear in place according to the probe."""
    kind, val = probe
    if kind == "bits":
        layer.set_quantizers(UniformWeightQuantizer(int(val), per_channel=True), layer.aq)
    elif kind == "ternary":
        layer.set_quantizers(TernaryQuantizer(per_channel=True), layer.aq)
    elif kind == "prune":
        w = layer.weight
        norms = w.flatten(1).norm(dim=1)
        k = int(round(val * w.shape[0]))
        if k > 0:
            idx = torch.argsort(norms)[:k]
            w[idx] = 0.0
            if layer.bias is not None:
                layer.bias[idx] = 0.0
    elif kind == "sparsity":
        w = layer.weight
        k = int(round(val * w.numel()))
        if k > 0:
            thr = w.abs().flatten().kthvalue(k).values
            w[w.abs() <= thr] = 0.0
    else:
        raise ValueError(f"unknown probe {kind!r}")


def probe_label(probe: Probe) -> str:
    kind, val = probe
    if kind == "bits":
        return f"w{int(val)}"
    if kind == "ternary":
        return "w1.58"
    return f"{kind}{val:g}"


def sensitivity(model: nn.Module, eval_fn: Callable[[nn.Module], float], probes: Sequence[Probe],
                layers: Optional[Sequence[str]] = None, log: Callable[[str], None] = lambda s: None
                ) -> Dict[str, object]:
    """One-shot sensitivity: metric after applying each probe to each layer alone.

    ``eval_fn(model) -> float`` (higher is better, e.g. accuracy). Returns
    {"baseline": m0, "probes": [labels], "layers": [names], "metric": [[m]], "drop": [[m0-m]]}.
    """
    names = [n for n, _ in quant_layers(model)] if layers is None else list(layers)
    baseline = float(eval_fn(model))
    log(f"baseline={baseline:.4f}")
    metric = []
    for name in names:
        row = []
        for probe in probes:
            m = copy.deepcopy(model)
            apply_probe(dict(quant_layers(m))[name], probe)
            val = float(eval_fn(m))
            row.append(val)
            log(f"  {name:32s} {probe_label(probe):10s} {val:.4f} (drop {baseline - val:+.4f})")
        metric.append(row)
    drop = [[baseline - v for v in row] for row in metric]
    return {"baseline": baseline, "probes": [probe_label(p) for p in probes], "layers": names,
            "metric": metric, "drop": drop}


# ----------------------------------------------------------------------------- reporting
def analysis_report(model: nn.Module, eval_fn: Optional[Callable[[nn.Module], float]] = None,
                    loader: Optional[Iterable] = None, probes: Sequence[Probe] = (),
                    device: Optional[torch.device] = None, max_batches: int = 10,
                    log: Callable[[str], None] = lambda s: None) -> Dict[str, object]:
    rep: Dict[str, object] = {"weights": [asdict(s) for s in weight_stats(model)]}
    if loader is not None:
        rep["activations"] = [asdict(s) for s in activation_stats(model, loader, device, max_batches)]
    if eval_fn is not None and probes:
        rep["sensitivity"] = sensitivity(model, eval_fn, probes, log=log)
    return rep


def to_markdown(rep: Dict[str, object]) -> str:
    lines = ["| layer | kind | params | out ch | bits | abs mean | near-zero | ternary zeros "
             "| ch-norm min/med | ch-norm CV |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in rep["weights"]:
        lines.append(f"| {s['name']} | {s['kind']} | {s['params']} | {s['out_channels']} | {s['wbits']:g} | "
                     f"{s['abs_mean']:.4f} | {100*s['near_zero_frac']:.1f} % | {100*s['ternary_zero_frac']:.1f} % | "
                     f"{s['chan_norm_min_over_median']:.2f} | {s['chan_norm_cv']:.2f} |")
    if "activations" in rep:
        lines += ["", "| layer | input zeros | input abs mean | output abs mean | dead out channels |",
                  "|---|---|---|---|---|"]
        for a in rep["activations"]:
            lines.append(f"| {a['name']} | {100*a['input_zero_frac']:.1f} % | {a['input_abs_mean']:.3f} | "
                         f"{a['output_abs_mean']:.3f} | {100*a['dead_out_channels_frac']:.1f} % |")
    if "sensitivity" in rep:
        s = rep["sensitivity"]
        lines += ["", f"Baseline metric {s['baseline']:.4f}; "
                  "cells are the drop when only that layer gets the probe.", "",
                  "| layer | " + " | ".join(s["probes"]) + " |", "|---|" + "---|" * len(s["probes"])]
        for name, row in zip(s["layers"], s["drop"]):
            lines.append(f"| {name} | " + " | ".join(f"{100*d:+.1f}" for d in row) + " |")
    return "\n".join(lines) + "\n"


def save_heatmap(sens: Dict[str, object], path: str, title: str = "") -> None:
    """Heatmap of metric drop (layers x probes); needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drop = np.array(sens["drop"]) * 100
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(sens["probes"]), 0.8 + 0.35 * len(sens["layers"])))
    im = ax.imshow(drop, cmap="Reds", aspect="auto", vmin=0, vmax=max(drop.max(), 1e-6))
    ax.set_xticks(range(len(sens["probes"])))
    ax.set_xticklabels(sens["probes"], rotation=45, ha="right")
    ax.set_yticks(range(len(sens["layers"])))
    ax.set_yticklabels(sens["layers"], fontsize=7)
    for i in range(drop.shape[0]):
        for j in range(drop.shape[1]):
            ax.text(j, i, f"{drop[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if drop[i, j] > 0.6 * drop.max() else "black")
    ax.set_title(title or f"metric drop, % (baseline {100*sens['baseline']:.2f})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
