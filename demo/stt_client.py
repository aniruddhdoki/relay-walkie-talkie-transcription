"""Whisper API client for transcription."""

import os
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def transcribe_audio(
    audio_path: str,
    api_key: Optional[str] = None,
) -> str:
    """Transcribe audio file using Whisper API. Returns transcript text."""
    if OpenAI is None:
        return ""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return ""
    client = OpenAI(api_key=key)
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    return resp.text.strip()


def transcribe_bytes(audio_bytes: bytes, filename: str = "audio.wav", api_key: Optional[str] = None) -> str:
    """Transcribe audio from bytes using Whisper API."""
    if OpenAI is None:
        return ""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return ""
    client = OpenAI(api_key=key)
    from io import BytesIO

    f = BytesIO(audio_bytes)
    f.name = filename
    resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    return resp.text.strip()
