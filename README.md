[nemuai-server-README.md](https://github.com/user-attachments/files/30779798/nemuai-server-README.md)
# NemuAI Server

FastAPI backend that runs the NemuAI audio-screening model behind a single HTTP endpoint, so the Nemuru app doesn't need to embed the model itself.

> NemuAI is an experimental screening prototype. It does not diagnose sleep apnea, calculate clinical AHI, or replace polysomnography or evaluation by a qualified medical professional.

## What it does

Reproduces the NemuAI Colab inference pipeline as a server:

1. Accepts an uploaded audio file (wav/mp3/m4a/aac/flac/ogg)
2. Converts to mono, resamples to 16 kHz
3. Splits into overlapping 30-second windows (5-second step)
4. Runs a reliability gate to reject noisy/silent/speech-contaminated windows
5. Extracts YAMNet embeddings from usable windows
6. Scales features and classifies with a logistic regression model
7. Smooths probabilities across neighboring windows, applies the validation-selected 0.40 threshold
8. Merges nearby positive windows into events
9. Returns the same JSON report shape as the Colab notebook

The model loads once at server startup instead of per-request.

## Endpoint

```
POST /analyze
Content-Type: multipart/form-data
Body: audio file

Response: JSON report (schema_version "nemuai-final-colab-1")
```

## Setup

```
pip install -r requirements.txt
```

Drop the trained model bundle into `model_bundle/` before starting the server. Required files:

- `nemuai_final_model.joblib`
- `nemuai_final_scaler.joblib`
- `nemuai_final_threshold.json`
- `nemuai_final_config.json`
- `nemuai_final_selection.json`
- `README.txt`

Run locally:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deployment

A `Dockerfile` is included, structured for Render. Render's free tier (512 MB RAM, 0.1 CPU, 750 hrs/month) works for a demo but spins down after 15 min idle (30–60s cold start on the next request), and TensorFlow/YAMNet's memory footprint should be watched against the 512 MB ceiling.

## Status

Built, not yet verified end-to-end against a reference clip/report. Next: confirm output matches the Colab notebook's output for the same input audio, then deploy and point the Nemuru app's Record/Upload flow at the live endpoint.
