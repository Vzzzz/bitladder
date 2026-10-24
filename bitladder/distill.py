"""Embedding distillation: train a small (quantized) student to reproduce a teacher's
embeddings on unlabeled data. No speaker / class labels needed, which is what makes it
usable for voice-ID models with Common Voice as the only data.

    teacher_fn(x) -> embeddings (B, D), any callable: a torch model, an ONNX Runtime
                     session wrapper, or precomputed embeddings looked up by index.
    student(x)    -> (B, D')  with D' == D unless ``proj`` maps it.

Loss = (1 - cosine) + ``mse_weight`` * MSE on L2-normalised embeddings (cosine is what
verification uses; the MSE term keeps norms sane early in training).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_distill_loss(student: torch.Tensor, teacher: torch.Tensor, mse_weight: float = 0.1) -> torch.Tensor:
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher.detach(), dim=-1)
    cos = (s * t).sum(-1)
    return (1.0 - cos).mean() + mse_weight * F.mse_loss(s, t)


@torch.no_grad()
def embedding_agreement(student: nn.Module, teacher_fn: Callable, loader: Iterable,
                        device: torch.device, max_batches: Optional[int] = None) -> Dict[str, float]:
    """Mean cosine between student and teacher embeddings, and the share above 0.9."""
    student.eval()
    cos: List[torch.Tensor] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        s = F.normalize(student(x), dim=-1)
        t = F.normalize(torch.as_tensor(teacher_fn(x)).to(device), dim=-1)
        cos.append((s * t).sum(-1))
    c = torch.cat(cos)
    return {"cos_mean": c.mean().item(), "cos_min": c.min().item(), "frac_above_0.9": (c > 0.9).float().mean().item()}


def distill_fit(student: nn.Module, teacher_fn: Callable, train_loader: Iterable, epochs: int,
                lr: float = 1e-3, weight_decay: float = 0.0, device: Optional[torch.device] = None,
                val_loader: Optional[Iterable] = None, mse_weight: float = 0.1,
                log: Callable[[str], None] = print) -> List[Dict[str, float]]:
    """Adam + one-cycle schedule, like ``bitladder.train.fit`` but with the distillation loss.

    ``teacher_fn`` is called on the same input batch (on ``device``) and may run outside
    autograd (ONNX Runtime); its output is converted with ``torch.as_tensor``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(train_loader)  # type: ignore[arg-type]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.15)
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        student.train()
        t0 = time.time()
        running, n = 0.0, 0
        for batch in train_loader:
            x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            with torch.no_grad():
                t = torch.as_tensor(teacher_fn(x)).to(device)
            loss = embedding_distill_loss(student(x), t, mse_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * x.shape[0]
            n += x.shape[0]
        rec = {"epoch": epoch, "train_loss": running / max(n, 1), "time": time.time() - t0}
        if val_loader is not None:
            rec.update({f"val_{k}": v for k, v in embedding_agreement(student, teacher_fn, val_loader, device).items()})
        history.append(rec)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    return history
