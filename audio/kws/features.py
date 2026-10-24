"""Log-mel front-end without torchaudio, so the same math can be ported to CMSIS-DSP."""

from __future__ import annotations

import torch
import torch.nn as nn


def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
    """Triangular HTK-style filters, shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2, n_bins)
    mel_pts = torch.linspace(hz_to_mel(torch.tensor(fmin)), hz_to_mel(torch.tensor(fmax)), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = torch.zeros(n_mels, n_bins)
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / (ctr - lo)
        down = (hi - fft_freqs) / (hi - ctr)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class LogMel(nn.Module):
    """(B, samples) -> (B, frames, n_mels) log-mel energies."""

    def __init__(self, sample_rate: int = 16000, win_ms: float = 30.0, hop_ms: float = 20.0,
                 n_fft: int = 512, n_mels: int = 40, fmin: float = 20.0, fmax: float = 4000.0,
                 eps: float = 1e-6):
        super().__init__()
        self.win_length = int(round(sample_rate * win_ms / 1000))
        self.hop_length = int(round(sample_rate * hop_ms / 1000))
        self.n_fft = n_fft
        self.eps = eps
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer("fb", mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wav, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window, center=False, return_complex=True)
        power = spec.abs() ** 2                         # (B, bins, frames)
        mel = torch.matmul(self.fb, power)              # (B, n_mels, frames)
        return torch.log(mel + self.eps).transpose(1, 2)

    def num_frames(self, n_samples: int) -> int:
        # torch.stft with center=False frames by n_fft, not win_length (the window is zero-padded).
        return 1 + (n_samples - self.n_fft) // self.hop_length if n_samples >= self.n_fft else 0
