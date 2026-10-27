# Accent Recognition

An English speech accent classifier with a browser recorder, upload flow, prediction API, and reproducible training pipeline. The original Colab exploration remains in [`Accent_Detection.ipynb`](Accent_Detection.ipynb); the application is implemented in `accent_recognition/` and `static/`.

**A trained model is bundled** in `models/accent.joblib`. It predicts five first-language groups from English speech: English, Spanish, Arabic, Mandarin, and Dutch. These are broad speech-background groups, not regional accent identities. See [MODEL_CARD.md](MODEL_CARD.md) for measured performance and limits. No accent classifier can be guaranteed perfect.

## Quick start

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
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

The bundled model was trained on the [Kaggle Speech Accent Archive v1](https://www.kaggle.com/datasets/rtatman/speech-accent-archive). The archive contains two identical copies of each recording; the preparation command extracts one copy and makes a speaker-level manifest:

```bash
mkdir -p data
curl -L -o data/speech-accent-archive.zip \
  'https://www.kaggle.com/api/v1/datasets/download/rtatman/speech-accent-archive?datasetVersionNumber=1'
accent-prepare-kaggle-zip --archive data/speech-accent-archive.zip --output-dir data
accent-train --manifest data/manifest.csv --output models/accent.joblib --seed 2026 --threshold 0.5
```

The original notebook also references the v2 metadata CSV. If you have that Kaggle layout, generate a manifest with:

```bash
accent-prepare-saa --metadata data/speakers_all.csv \
  --recordings data/recordings/recordings --output data/manifest.csv
```

**Caution:** The archive filenames and v2 `native_language` field provide first-language proxies, not verified regional accent labels. The Kaggle copy is listed under CC BY-NC-SA 4.0. The bundled weights should be treated as subject to the same noncommercial and attribution restrictions; check the source terms before reuse. Audio files are excluded from Git.

## Model and evaluation

Audio is normalized to mono 16 kHz, screened for duration and speech energy, and represented by MFCC, delta MFCC, spectral centroid, and zero-crossing summaries. A scaled linear discriminant analysis model with covariance shrinkage produces probabilities. Training uses a seeded **speaker-disjoint** 80/20 split. `models/accent.metrics.json` records class counts, split sizes, accuracy, balanced accuracy, macro F1, classification report, confusion matrix, and coverage at the confidence threshold. Review per-class recall and errors before deploying. Probabilities are estimates, not proof of geographic origin.

To change the abstention threshold:

```bash
accent-train --manifest data/manifest.csv --threshold 0.65
```

The training script uses its held-out set only for reporting. The bundled model's exploratory class and classifier selection used overlapping archive recordings, so its reported evaluation is not a fully untouched test. For production use, tune architecture and threshold with a separate validation set, then evaluate once on a locked, speaker-disjoint test set. Test on different microphones and read material as well as spontaneous speech. The Archive has every speaker read the same passage, so its evaluation alone cannot establish performance on arbitrary speech.

Model artifacts use `joblib` serialization, which can execute code while loading. Load only artifacts created in a trusted training environment. Deploy with a pinned artifact, dependency lock or image digest, HTTPS, ingress rate limits, resource limits, and observability appropriate to your hosting environment. The `/api/health` endpoint should be used for readiness. Avoid storing user audio unless users explicitly consent and a retention policy is in place.

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
```

The Docker image includes the bundled model. To use a retrained model, mount your own trusted `models` directory or set `ACCENT_MODEL_PATH`:

```bash
docker build -t accent-recognition .
docker run --rm -p 8000:8000 accent-recognition
```

## Known limits

- Accent categories and errors reflect the chosen dataset; an untrained group cannot be recognized.
- Native language, nationality, ethnicity, and accent are different concepts. Do not use the output as a reliable identity or demographic determination.
- Probabilities from this small dataset may be unreliable; the confidence threshold is a simple abstention rule.
- Background noise, very short samples, illness, and code switching can reduce reliability.
