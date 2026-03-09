"""Cellular-style synthetic noising pipeline. All transforms are length-preserving."""

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low: float = 300.0,
    high: float = 3400.0,
) -> np.ndarray:
    """Apply bandpass filter (AMR-NB telephony bandwidth). Length-preserving."""
    nyq = sample_rate / 2
    low_norm = low / nyq
    high_norm = min(high / nyq, 0.99)
    b, a = signal.butter(4, [low_norm, high_norm], btype="band")
    return signal.filtfilt(b, a, audio)


def add_environmental_noise(
    audio: np.ndarray,
    snr_db: float = 10.0,
    noise_type: str = "white",
) -> np.ndarray:
    """Add environmental noise at given SNR. Length-preserving (additive)."""
    if noise_type == "white":
        noise = np.random.randn(len(audio)).astype(np.float32)
    elif noise_type == "pink":
        white = np.random.randn(len(audio))
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        noise = signal.lfilter(b, a, white).astype(np.float32)
    else:
        noise = np.random.randn(len(audio)).astype(np.float32)

    sig_power = np.mean(audio**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    scale = np.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))
    return audio + (noise / scale).astype(audio.dtype)


def resample_roundtrip(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 8000,
) -> np.ndarray:
    """Resample to target_sr and back. Simulates narrowband. Length-preserving (round-trip)."""
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    back = signal.resample(resampled, len(audio))
    return back.astype(audio.dtype)


def apply_noising_pipeline(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    bandpass: bool = True,
    add_noise: bool = True,
    resample_nb: bool = False,
    snr_db: float = 12.0,
) -> np.ndarray:
    """Apply a subset of cellular-style degradations. All length-preserving."""
    out = audio.astype(np.float32) / (np.max(np.abs(audio)) + 1e-8)

    if bandpass:
        out = bandpass_filter(out, sample_rate, 300, 3400)

    if resample_nb and sample_rate != 8000:
        out = resample_roundtrip(out, sample_rate, 8000)

    if add_noise:
        out = add_environmental_noise(out, snr_db=snr_db)

    return (out * 32767).astype(np.int16)
