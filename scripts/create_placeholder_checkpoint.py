"""Create placeholder checkpoint for demo before Colab training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from models.denoiser import Denoiser

def main():
    model = Denoiser(base_channels=32, n_levels=4)
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "best.pt"
    torch.save({"state_dict": model.state_dict(), "epoch": 0, "mse": float("nan")}, path)
    print(f"Saved placeholder checkpoint to {path}")


if __name__ == "__main__":
    main()
