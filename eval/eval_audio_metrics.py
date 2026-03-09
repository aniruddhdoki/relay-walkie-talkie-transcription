"""Audio quality metrics: SI-SDR, PESQ, MSE for denoiser evaluation."""

from typing import Optional

import numpy as np


def si_sdr(reference: np.ndarray, estimated: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio. Higher = better."""
    ref = reference.flatten().astype(np.float64)
    est = estimated.flatten().astype(np.float64)
    if len(ref) != len(est):
        m = min(len(ref), len(est))
        ref, est = ref[:m], est[:m]
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    e_target = alpha * ref
    e_res = est - e_target
    sig_power = np.mean(e_target**2) + eps
    res_power = np.mean(e_res**2) + eps
    return 10 * np.log10(sig_power / res_power + eps)


def mse_audio(reference: np.ndarray, estimated: np.ndarray) -> float:
    """MSE between waveforms."""
    m = min(len(reference.flatten()), len(estimated.flatten()))
    return float(np.mean((reference.flatten()[:m] - estimated.flatten()[:m]) ** 2))


def compute_pesq(reference: np.ndarray, degraded: np.ndarray, sr: int = 16000) -> Optional[float]:
    """PESQ if pesq available. Narrow-band for 8k, wide for 16k."""
    try:
        from pesq import pesq as pesq_fn

        ref = reference.flatten().astype(np.float64)
        deg = degraded.flatten().astype(np.float64)
        m = min(len(ref), len(deg))
        ref, deg = ref[:m], deg[:m]
        mode = "nb" if sr <= 8000 else "wb"
        return float(pesq_fn(sr, ref, deg, mode))
    except Exception:
        return None
