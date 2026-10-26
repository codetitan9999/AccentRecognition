# Accent Recognition

An English speech accent classifier with a browser recorder, upload flow, prediction API, and reproducible training pipeline. The original Colab exploration remains in [`Accent_Detection.ipynb`](Accent_Detection.ipynb); the application is implemented in `accent_recognition/` and `static/`.

**A model is not bundled.** The repository has no licensed audio dataset or trained weights. Train on labeled recordings before the API can return predictions. No accent classifier can be guaranteed perfect: performance depends on label quality, speaker coverage, recording conditions, and how well deployment speech matches training speech. The API returns an uncertain result when confidence is below the configured threshold.

## Quick start

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
accent-train --manifest data/manifest.csv --output models/accent.joblib
uvicorn accent_recognition.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The microphone requires localhost or HTTPS. The server loads the model on startup; restart after retraining. Set `ACCENT_MODEL_PATH` to use a different artifact. `GET /api/health` reports whether it loaded. `POST /api/predict` accepts multipart form field `audio` and returns the best label, confidence, alternatives, audio durations, and an uncertainty flag. OpenAPI documentation is at `/docs`.

The browser turns microphone recordings into 16 kHz mono WAV. Uploads accept WAV, FLAC, OGG, and MP3 when the installed libsndfile decoder supports them. Files must be at most 12 MB and contain 2–30 seconds of audio with at least 2 seconds of detected speech. The server processes audio in memory without saving uploads.

## Training data

The CSV manifest has one row per clip, with paths relative to the manifest (absolute paths also work):

```csv
path,accent,speaker_id
clips/us_001.wav,American English,speaker-001
clips/us_002.wav,American English,speaker-002
clips/uk_001.wav,British English,speaker-003
```

Use **accent annotations from a licensed dataset or a documented human labeling process**. A speaker must have one consistent label. The trainer needs at least two labels and three distinct speakers per label, and requires a split where every label appears on both sides. It skips invalid clips and records them in the metrics file. For trustworthy measurements, gather many more speakers and use an independent external test set drawn from the intended audience.

The original notebook references the [Speech Accent Archive](https://accent.gmu.edu/). If you have downloaded its Kaggle layout yourself, generate a manifest with:

```bash
accent-prepare-saa --metadata data/speakers_all.csv \
  --recordings data/recordings/recordings --output data/manifest.csv
```

**Caution:** `native_language` is a proxy, not a verified regional accent label. Models trained on this manifest predict groups named for first languages and should be described that way. The Kaggle copy is listed under CC BY-NC-SA 4.0; check its terms before training, distribution, or commercial use. Dataset files and weights are ignored by Git.

## Model and evaluation

Audio is normalized to mono 16 kHz, screened for duration and speech energy, and represented by MFCC, delta MFCC, spectral centroid, and zero-crossing summaries. A scaled, class-weighted RBF SVM produces probabilities. Training uses a seeded **speaker-disjoint** 80/20 split. `models/accent.metrics.json` records class counts, split sizes, accuracy, balanced accuracy, macro F1, classification report, confusion matrix, and coverage at the confidence threshold. Review per-class recall and errors before deploying. Probabilities are estimates, not proof of geographic origin.

To change the abstention threshold:

```bash
accent-train --manifest data/manifest.csv --threshold 0.65
```

The held-out set is used only for reporting; it does not tune the model. For production use, tune architecture and threshold with a separate validation set, then evaluate once on a locked, speaker-disjoint test set. Test on different microphones and read material as well as spontaneous speech. The Archive has every speaker read the same passage, so its evaluation alone cannot establish performance on arbitrary speech.

Model artifacts use `joblib` serialization, which can execute code while loading. Load only artifacts created in a trusted training environment. Deploy with a pinned artifact, dependency lock or image digest, HTTPS, ingress rate limits, resource limits, and observability appropriate to your hosting environment. The `/api/health` endpoint should be used for readiness. Avoid storing user audio unless users explicitly consent and a retention policy is in place.

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
```

The Docker image runs the API but does not include a model. Mount a trained `models/accent.joblib` file or set `ACCENT_MODEL_PATH` to a trusted artifact path:

```bash
docker build -t accent-recognition .
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" accent-recognition
```

## Known limits

- Accent categories and errors reflect the chosen dataset; an untrained group cannot be recognized.
- Native language, nationality, ethnicity, and accent are different concepts. Do not use the output as a reliable identity or demographic determination.
- Calibrated probabilities from a small dataset may still be unreliable; the confidence threshold is a simple abstention rule.
- Background noise, very short samples, illness, and code switching can reduce reliability.
