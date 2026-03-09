"""Spectrogram visualization for Streamlit."""

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def waveform_to_spec(audio: np.ndarray, n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    """Convert waveform to log-magnitude spectrogram [F, T]."""
    try:
        import torch
        t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        spec = torch.stft(t, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        mag = spec.abs().squeeze(0).numpy()
        return np.log(mag + 1e-8)
    except Exception:
        n_frames = max(1, (len(audio) - n_fft) // hop_length + 1)
        spec = np.zeros((n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft]
            if len(frame) >= n_fft // 2:
                f = np.fft.rfft(np.pad(frame, (0, n_fft - len(frame))))
                spec[:, i] = np.log(np.abs(f) + 1e-8)
        return spec


def plot_spectrogram(
    spec: np.ndarray,
    title: str = "Spectrogram",
    sr: int = 16000,
    hop_length: int = 128,
) -> bytes:
    """Render spectrogram to PNG bytes. Accepts waveform (1D) or spec [F,T]."""
    fig, ax = plt.subplots(figsize=(8, 3))
    if spec.ndim == 1:
        spec = waveform_to_spec(spec)
    if spec.ndim == 3:
        spec = spec.squeeze(0)
    if spec.ndim >= 2:
        img = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
        )
        plt.colorbar(img, ax=ax)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency bin")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return buf.read()
