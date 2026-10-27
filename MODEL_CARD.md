# Bundled accent model

## Purpose and scope

`models/accent.joblib` is a trained **English speech background classifier**. Its three outputs, `english`, `spanish`, and `arabic`, come from the speakers' first-language group in the Speech Accent Archive filenames. They are proxies for broad accent influence, not verified regional accents, nationality, ethnicity, or identity. Speech from another background is outside the model's training scope; a probability threshold of 0.5 allows an uncertain response but does not reliably detect every unknown group.

## Training data and terms

- Source: [Speech Accent Archive on Kaggle, version 1](https://www.kaggle.com/datasets/rtatman/speech-accent-archive), which mirrors recordings from [George Mason University's Speech Accent Archive](https://accent.gmu.edu/).
- Archive SHA-256: `6c84582145e030a25a7f180bbcfcbd76fb3babbb00895b4c7326dfc443399d4a`.
- 843 recordings, one per speaker: 579 English, 162 Spanish, and 102 Arabic first-language speakers. No archive recording is committed to this repository.
- Kaggle lists the dataset license as **CC BY-NC-SA 4.0**. Treat the bundled model as subject to attribution and noncommercial/share-alike restrictions unless the rights holder grants broader permission. Do not use it commercially without a licensing review.
- Each speaker reads the same English passage. Validation on that passage does not establish accuracy on spontaneous speech or other microphones.

## Method

Audio is decoded, cropped to the first 30 seconds for training, resampled to 16 kHz mono, and screened for at least 2 seconds of sound. MFCC, delta MFCC, spectral centroid, and zero-crossing summaries produce a 210-dimensional feature vector. A standard scaler and class-weighted logistic regression classifier are fitted on the training speakers. Split seed: 42. Training code: `accent_recognition/train.py`; archive preparation: `accent_recognition/prepare_archive.py`.

The three-group architecture was chosen using a separate validation split. The bundled artifact was then fitted on the outer training split and evaluated once on 169 held-out speakers. No test speaker appears in training. The artifact contains only the estimator, labels, threshold, and metrics; it does not include source audio.

## Held-out results

| Measure | Value |
| --- | ---: |
| Training speakers | 674 |
| Test speakers | 169 |
| Accuracy | 0.6568 |
| Balanced accuracy | 0.5882 |
| Macro F1 | 0.5769 |
| Coverage at 0.5 threshold | 0.9645 |
| Accuracy among accepted predictions | 0.6626 |

| First-language group | Precision | Recall | F1 | Test speakers |
| --- | ---: | ---: | ---: | ---: |
| Arabic | 0.483 | 0.538 | 0.509 | 26 |
| English | 0.849 | 0.752 | 0.798 | 105 |
| Spanish | 0.383 | 0.474 | 0.424 | 38 |

Full metrics and confusion matrix: [`models/accent.metrics.json`](models/accent.metrics.json). These numbers are a baseline, not a claim of production accuracy. The Spanish and Arabic recall means many speakers in those groups are misclassified. A held-out evaluation across different prompts, recording devices, and speakers is required before any consequential use.

## Runtime and reproducibility

Artifact SHA-256: `39149229a0724abf4514657f3bd677679996ec1ba97d60504f773048f69d0f6c`. Trained with Python 3.12, NumPy 2.5.4, SciPy 1.18.1, librosa 0.11.0, soundfile 0.14.0, scikit-learn 1.9.1, and joblib 1.6.0. `joblib` artifacts must be loaded only from trusted sources. The app processes uploads in memory and does not retain them.
