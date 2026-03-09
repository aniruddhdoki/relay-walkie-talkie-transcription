"""Generate demo sample audio (clean + noisy) for testing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


def main():
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Simple tone + noise as placeholder
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noisy = clean + 0.1 * np.random.randn(len(t)).astype(np.float32)

    out_dir = ROOT / "samples"
    out_dir.mkdir(exist_ok=True)

    if sf:
        sf.write(out_dir / "clean.wav", clean, sr)
        sf.write(out_dir / "noisy.wav", noisy, sr)
        print(f"Saved to {out_dir}")
    else:
        from scipy.io import wavfile
        wavfile.write(out_dir / "clean.wav", sr, (clean * 32767).astype(np.int16))
        wavfile.write(out_dir / "noisy.wav", sr, (noisy * 32767).astype(np.int16))
        print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
