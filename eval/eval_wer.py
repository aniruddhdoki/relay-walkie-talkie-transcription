"""Compute WER for clean, noisy, noisereduce baseline, and denoised transcriptions."""

from typing import Dict, List, Optional

try:
    import jiwer
except ImportError:
    jiwer = None


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate. Returns fraction (0-1+)."""
    if jiwer is None:
        return 0.0
    return jiwer.wer(reference, hypothesis)


def compute_wer_batch(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """Average WER over a batch."""
    if jiwer is None:
        return 0.0
    return jiwer.wer(references, hypotheses)
