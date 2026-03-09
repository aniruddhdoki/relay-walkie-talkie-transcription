"""Audio playback utilities for Streamlit."""

import io
from typing import Optional

import numpy as np


def waveform_to_bytes(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> bytes:
    """Convert float waveform to WAV bytes for Streamlit audio."""
    import soundfile as sf

    wav = (waveform * 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()
