"""Minimal training loop shared by the sandboxes."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1)}


def fit(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
        lr: float = 2e-3, weight_decay: float = 0.0, device: Optional[torch.device] = None,
        log: Callable[[str], None] = print, eval_every: int = 1) -> List[Dict[str, float]]:
    """Adam + cosine schedule. Returns per-epoch history with test accuracy."""
    device = device or pick_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.15)
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * y.numel()
            n += y.numel()
        rec = {"epoch": epoch, "train_loss": running / max(n, 1), "time": time.time() - t0}
        if epoch % eval_every == 0 or epoch == epochs:
            rec.update({f"test_{k}": v for k, v in evaluate(model, test_loader, device).items()})
        history.append(rec)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    return history
