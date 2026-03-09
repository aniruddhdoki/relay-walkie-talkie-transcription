"""Streamlit demo app for Relay Denoiser."""

import json
import os
import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np

from inference import DenoisePipeline, load_model
from data.synthetic_noise import apply_noising_pipeline
from demo.spectrogram_viz import plot_spectrogram
from demo.audio_player import waveform_to_bytes
try:
    from demo.stt_client import transcribe_bytes
except ImportError:
    from stt_client import transcribe_bytes

# Config
CHECKPOINT = ROOT / "checkpoints" / "best.pt"
EVAL_RESULTS = ROOT / "eval_results.json"
SAMPLE_RATE = 16000


@st.cache_resource
def load_denoiser():
    """Load model once and cache."""
    if not CHECKPOINT.exists():
        st.warning("Checkpoint not found. Using placeholder (untrained model). Run Colab notebook to train.")
        from models.denoiser import Denoiser
        model = Denoiser(base_channels=32, n_levels=4)
        return DenoisePipeline(model, n_fft=512, hop_length=128, device="cpu")
    try:
        model = load_model(CHECKPOINT, device="cpu")
        return DenoisePipeline(model, n_fft=512, hop_length=128, device="cpu")
    except Exception as e:
        st.warning(f"Could not load checkpoint: {e}. Using untrained model.")
        from models.denoiser import Denoiser
        model = Denoiser(base_channels=32, n_levels=4)
        return DenoisePipeline(model, n_fft=512, hop_length=128, device="cpu")


def load_eval_results():
    """Load eval results JSON."""
    if EVAL_RESULTS.exists():
        with open(EVAL_RESULTS) as f:
            return json.load(f)
    return {
        "mse": None,
        "wer": {"clean": None, "noisy": None, "noisereduce": None, "denoised": None},
        "si_sdr": {"noisy": None, "noisereduce": None, "denoised": None},
        "placeholder": True,
    }


def audio_to_float(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def main():
    st.set_page_config(page_title="Relay Denoiser Demo", layout="wide")
    st.title("Relay Walkie-Talkie Denoiser")

    # Stats panel
    results = load_eval_results()
    with st.expander("Model & Eval Stats", expanded=True):
        if results.get("placeholder"):
            st.info("Placeholder results. Run the Colab notebook to get real metrics.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MSE", results.get("mse", "—"))
        with col2:
            wer = results.get("wer", {})
            st.write("**WER (test set)**")
            st.write(f"Clean: {wer.get('clean', '—')} | Noisy: {wer.get('noisy', '—')}")
            st.write(f"Noisereduce: {wer.get('noisereduce', '—')} | Denoised: {wer.get('denoised', '—')}")
        with col3:
            si = results.get("si_sdr", {})
            st.write("**SI-SDR**")
            st.write(f"Noisy: {si.get('noisy', '—')} | NR: {si.get('noisereduce', '—')} | Denoised: {si.get('denoised', '—')}")

    # Upload / record
    st.subheader("Input Audio")
    audio_bytes = st.file_uploader("Upload clean audio (WAV)", type=["wav"])
    if not audio_bytes:
        st.info("Upload a WAV file to start.")
        return

    import soundfile as sf
    from io import BytesIO
    clean_audio, sr = sf.read(BytesIO(audio_bytes.read()))
    clean_audio = audio_to_float(clean_audio)
    if sr != SAMPLE_RATE:
        from scipy import signal as scipy_signal
        num = int(len(clean_audio) * SAMPLE_RATE / sr)
        clean_audio = scipy_signal.resample(clean_audio, num).astype(np.float32)

    # Apply synthetic noising
    clean_int16 = (clean_audio * 32767).astype(np.int16)
    noisy_audio = apply_noising_pipeline(clean_int16, SAMPLE_RATE)
    noisy_float = noisy_audio.astype(np.float32) / 32768.0

    # Denoise
    pipeline = load_denoiser()
    denoised = pipeline(noisy_float)
    denoised = denoised.astype(np.float32)

    # Spectrograms
    st.subheader("Spectrograms")
    col1, col2, col3 = st.columns(3)
    with col1:
        img_clean = plot_spectrogram(clean_audio, title="Clean")
        st.image(img_clean)
        st.audio(waveform_to_bytes(clean_audio), sample_rate=SAMPLE_RATE)
    with col2:
        img_noisy = plot_spectrogram(noisy_float, title="Noisy")
        st.image(img_noisy)
        st.audio(waveform_to_bytes(noisy_float), sample_rate=SAMPLE_RATE)
    with col3:
        img_denoised = plot_spectrogram(denoised, title="Denoised")
        st.image(img_denoised)
        st.audio(waveform_to_bytes(denoised), sample_rate=SAMPLE_RATE)

    # Transcription (optional)
    if os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""):
        st.subheader("Transcription (Whisper API)")
        api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        t_clean = transcribe_bytes(waveform_to_bytes(clean_audio), "clean.wav", api_key)
        t_noisy = transcribe_bytes(waveform_to_bytes(noisy_float), "noisy.wav", api_key)
        t_denoised = transcribe_bytes(waveform_to_bytes(denoised), "denoised.wav", api_key)
        st.write("**Clean:**", t_clean or "(empty)")
        st.write("**Noisy:**", t_noisy or "(empty)")
        st.write("**Denoised:**", t_denoised or "(empty)")
    else:
        st.info("Set OPENAI_API_KEY for transcription comparison.")


if __name__ == "__main__":
    main()
