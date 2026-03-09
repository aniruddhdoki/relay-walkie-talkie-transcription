"""PyTorch Dataset for VOiCES devkit with train_index.csv / test_index.csv."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_audio(path: str, root: str) -> np.ndarray:
    """Load audio from path relative to root. Returns (samples, sr)."""
    full = os.path.join(root, path) if not os.path.isabs(path) else path
    if sf is not None:
        data, sr = sf.read(full)
        return data.astype(np.float32) / 32768.0, sr
    # Fallback: scipy
    from scipy.io import wavfile

    sr, data = wavfile.read(full)
    return data.astype(np.float32) / 32768.0, sr


class VoicesDataset(Dataset):
    """Dataset for VOiCES devkit. Returns (noisy_spec, clean_spec) or (noisy, clean) waveforms."""

    def __init__(
        self,
        index_csv: str,
        root: str,
        use_noisy_as_input: bool = True,
        use_spec: bool = True,
        n_fft: int = 512,
        hop_length: int = 128,
        max_len_sec: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.df = pd.read_csv(index_csv)
        self.use_noisy_as_input = use_noisy_as_input
        self.use_spec = use_spec
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = int(max_len_sec * sample_rate)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def _to_spec(self, audio: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio.transforms as T

        t = torch.from_numpy(audio).float().unsqueeze(0)
        spec = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(t)
        mag = spec.abs()
        log_mag = torch.log(mag + 1e-8)
        return log_mag

    def _load_pair(self, idx):
        row = self.df.iloc[idx]
        source_path = row.get("source", row.get("filename", ""))
        distant_path = row.get("filename", "")

        clean, sr_clean = load_audio(str(source_path), str(self.root))
        noisy, sr_noisy = load_audio(str(distant_path), str(self.root))

        if sr_clean != self.sample_rate or sr_noisy != self.sample_rate:
            from scipy import signal as scipy_signal

            if len(clean) > 0 and sr_clean != self.sample_rate:
                num = int(len(clean) * self.sample_rate / sr_clean)
                clean = scipy_signal.resample(clean, num).astype(np.float32)
            if len(noisy) > 0 and sr_noisy != self.sample_rate:
                num = int(len(noisy) * self.sample_rate / sr_noisy)
                noisy = scipy_signal.resample(noisy, num).astype(np.float32)

        L = min(len(clean), len(noisy), self.max_len)
        clean = clean[:L]
        noisy = noisy[:L]
        if len(clean) < self.max_len:
            clean = np.pad(clean, (0, self.max_len - len(clean)))
            noisy = np.pad(noisy, (0, self.max_len - len(noisy)))
        return noisy, clean

    def __getitem__(self, idx):
        noisy, clean = self._load_pair(idx)

        if self.use_spec:
            clean_spec = self._to_spec(clean).squeeze(0).numpy()
            noisy_spec = self._to_spec(noisy).squeeze(0).numpy()
            return (
                torch.from_numpy(noisy_spec).float().unsqueeze(0),
                torch.from_numpy(clean_spec).float().unsqueeze(0),
            )
        return (
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float(),
        )

    def get_waveforms(self, idx: int):
        """Return raw (noisy, clean) numpy arrays for eval metrics (SI-SDR, etc.)."""
        return self._load_pair(idx)
