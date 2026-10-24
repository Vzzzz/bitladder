import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bitladder.distill import distill_fit, embedding_agreement, embedding_distill_loss
from bitladder.quant import QuantConfig, QuantLinear, apply_quantization


def test_loss_zero_for_identical_embeddings():
    e = torch.randn(5, 8)
    assert embedding_distill_loss(e, e).item() < 1e-6
    assert embedding_distill_loss(e, -e).item() > 1.5


def test_student_learns_teacher_embeddings():
    torch.manual_seed(0)
    teacher = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8)).eval()
    student = nn.Sequential(QuantLinear(16, 32), nn.Tanh(), QuantLinear(32, 8))
    apply_quantization(student, QuantConfig(scheme="uniform", wbits=4, first_last_bits=8))
    x = torch.randn(512, 16)
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)
    dev = torch.device("cpu")
    before = embedding_agreement(student, teacher, loader, dev)["cos_mean"]
    hist = distill_fit(student, teacher, loader, epochs=30, lr=1e-2, device=dev, val_loader=loader, log=lambda s: None)
    after = embedding_agreement(student, teacher, loader, dev)["cos_mean"]
    assert after > before and after > 0.9
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert "val_cos_mean" in hist[-1]
