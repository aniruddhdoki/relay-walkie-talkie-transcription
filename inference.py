"""Inference pipeline: load checkpoint, denoise audio."""

from pathlib import Path
from typing import Union

import numpy as np
import torch

from models.denoiser import Denoiser


def load_model(checkpoint_path: Union[str, Path], device: str = "cpu") -> Denoiser:
    """Load denoiser from checkpoint."""
    model = Denoiser(base_channels=32, n_levels=4)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model.to(device)


class DenoisePipeline:
    """Waveform -> spectrogram -> model -> spectrogram -> waveform."""

    def __init__(
        self,
        model: Denoiser,
        n_fft: int = 512,
        hop_length: int = 128,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device

    def _stft(self, audio: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(audio).float().to(self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return torch.stft(
            t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            return_complex=True,
        )

    def _to_spec(self, audio: np.ndarray) -> torch.Tensor:
        spec = self._stft(audio)
        mag = spec.abs()
        return torch.log(mag + 1e-8)

    def _to_waveform(self, log_mag: torch.Tensor, phase: torch.Tensor) -> np.ndarray:
        mag = torch.exp(log_mag).clamp(min=1e-8)
        complex_spec = mag * torch.exp(1j * phase.to(log_mag.device))
        wav = torch.istft(
            complex_spec.unsqueeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
        )
        return wav.squeeze().cpu().numpy()

    def _get_phase(self, audio: np.ndarray) -> torch.Tensor:
        spec = self._stft(audio)
        return torch.angle(spec)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Denoise waveform. Returns denoised waveform."""
        phase = self._get_phase(audio)
        noisy_spec = self._to_spec(audio).unsqueeze(0)
        with torch.no_grad():
            clean_spec = self.model(noisy_spec)
        return self._to_waveform(clean_spec.squeeze(0), phase.squeeze(0))
