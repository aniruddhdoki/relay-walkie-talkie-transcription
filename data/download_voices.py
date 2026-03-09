"""Download VOiCES devkit from AWS S3."""

import os
import subprocess
import sys
from pathlib import Path


def download_voices_devkit(target_dir: str = ".") -> str:
    """Download VOiCES_devkit.tar.gz from S3. Returns path to extracted directory."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "VOiCES_devkit.tar.gz"

    if not archive.exists():
        print("Downloading VOiCES_devkit.tar.gz from S3...")
        subprocess.run(
            [
                "aws", "s3", "cp",
                "s3://lab41openaudiocorpus/VOiCES_devkit.tar.gz",
                str(archive),
            ],
            check=True,
        )
    else:
        print(f"Found existing {archive}")

    extract_dir = target / "VOiCES_devkit"
    if not extract_dir.exists():
        print("Extracting...")
        import tarfile

        with tarfile.open(archive) as tf:
            tf.extractall(target)
    else:
        print(f"Found existing {extract_dir}")

    return str(extract_dir)


if __name__ == "__main__":
    out = download_voices_devkit(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(f"VOiCES devkit at: {out}")
