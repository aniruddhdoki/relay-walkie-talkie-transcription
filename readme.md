# Relay Walkie-Talkie Denoiser

A small U-Net denoiser for cellular-degraded audio. Trained on VOiCES, with Streamlit demo, spectrograms, and Whisper API transcription comparison.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

## Structure

- **`notebooks/denoiser_train_eval.ipynb`** — Train and evaluate in Google Colab. Produces `best.pt` and `eval_results.json`.
- **`demo/app.py`** — Streamlit app: upload audio, apply synthetic noising, denoise, view spectrograms and transcription.
- **`checkpoints/best.pt`** — Model checkpoint. Use placeholder until you run the Colab notebook.
- **`eval_results.json`** — Eval metrics (WER, SI-SDR, etc.). Replace with Colab output.

## Placeholder Checkpoint

Until you run training in Colab:

```bash
pip install torch
python scripts/create_placeholder_checkpoint.py
```

This saves a randomly initialized model so the demo runs. Replace `checkpoints/best.pt` with your trained checkpoint from Colab.

## Training (Google Colab)

1. Open `notebooks/denoiser_train_eval.ipynb` in Colab.
2. Set GPU runtime.
3. Update `REPO_URL` and `VOICES_ROOT` paths.
4. Download VOiCES devkit (or mount Drive with pre-downloaded data).
5. Run all cells. Download `best.pt` and `eval_results.json` when done.

## Deployment (Hugging Face Spaces)

1. Create a new Space with Streamlit SDK.
2. Set the main app file to `app.py` (root) or `demo/app.py`.
3. Commit `checkpoints/best.pt` and `eval_results.json` to the Space.
4. Add secret `OPENAI_API_KEY` for Whisper transcription.
5. Space will run at `https://huggingface.co/spaces/<username>/<space-name>`.

## Scripts

- `scripts/create_placeholder_checkpoint.py` — Create placeholder `best.pt` (requires `torch`)
- `scripts/demo_sample_audio.py` — Generate sample clean/noisy WAV files for testing

## Data

VOiCES devkit: `aws s3 cp s3://lab41openaudiocorpus/VOiCES_devkit.tar.gz .`  
Splits: `references/train_index.csv`, `references/test_index.csv`.
