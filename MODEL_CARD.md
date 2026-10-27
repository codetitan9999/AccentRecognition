# Bundled accent model

## Purpose and scope

`models/accent.joblib` classifies English speech into five speaker first-language groups: `arabic`, `dutch`, `english`, `mandarin`, and `spanish`. These groups are proxies for broad accent influence, not verified regional accents, nationality, ethnicity, or identity. Speech from another background is outside the training scope. A probability threshold of 0.5 allows an uncertain response but does not reliably detect every unknown group.

## Training data and terms

- Source: [Speech Accent Archive on Kaggle, version 1](https://www.kaggle.com/datasets/rtatman/speech-accent-archive), which mirrors recordings from [George Mason University's Speech Accent Archive](https://accent.gmu.edu/).
- Archive SHA-256: `6c84582145e030a25a7f180bbcfcbd76fb3babbb00895b4c7326dfc443399d4a`.
- 955 recordings, one per speaker: 579 English, 162 Spanish, 102 Arabic, 65 Mandarin, and 47 Dutch first-language speakers. No archive recording is committed to this repository.
- Kaggle lists the dataset license as **CC BY-NC-SA 4.0**. Treat the bundled model as subject to attribution and noncommercial/share-alike restrictions unless the rights holder grants broader permission. Do not use it commercially without a licensing review.
- Each speaker reads the same English passage. Evaluation on that passage does not establish accuracy on spontaneous speech or other microphones.

## Method

Audio is decoded, cropped to the first 30 seconds for training, resampled to 16 kHz mono, and screened for at least 2 seconds of sound. MFCC, delta MFCC, spectral centroid, and zero-crossing summaries produce a 210-dimensional feature vector. A standard scaler and linear discriminant analysis classifier with shrinkage 0.1 are fitted on the training speakers. Split seed: 2026. Training code: `accent_recognition/train.py`; archive preparation: `accent_recognition/prepare_archive.py`.

The five-group label set and classifier were selected using exploratory speaker-disjoint cross-validation on this archive. The bundled artifact was fitted on 764 training speakers and evaluated on 191 held-out speakers. No speaker appears in both final fit and evaluation splits. However, the exploratory comparisons used overlapping recordings, so this evaluation is not a fully untouched test and may be optimistic. The artifact contains the estimator, labels, threshold, and metrics; it does not include source audio.

## Held-out results

| Measure | Value |
| --- | ---: |
| Training speakers | 764 |
| Test speakers | 191 |
| Accuracy | 0.6021 |
| Balanced accuracy | 0.3896 |
| Macro F1 | 0.3963 |
| Coverage at 0.5 threshold | 0.8796 |
| Accuracy among accepted predictions | 0.6369 |

| First-language group | Precision | Recall | F1 | Test speakers |
| --- | ---: | ---: | ---: | ---: |
| Arabic | 0.318 | 0.368 | 0.341 | 19 |
| Dutch | 0.375 | 0.250 | 0.300 | 12 |
| English | 0.710 | 0.823 | 0.762 | 113 |
| Mandarin | 0.250 | 0.250 | 0.250 | 8 |
| Spanish | 0.455 | 0.256 | 0.328 | 39 |

Full metrics and confusion matrix: [`models/accent.metrics.json`](models/accent.metrics.json). This is a weak baseline for the minority groups; Dutch and Mandarin each had only a quarter of held-out speakers classified correctly. The broader label set also reduces aggregate accuracy relative to the previous three-group model, so results across those models are not directly comparable. An independent evaluation with different prompts, devices, and speakers is required before consequential use.

## Runtime and reproducibility

Artifact SHA-256: `ac3a40cae553546929abc0edfa67ffbe0f071ff6c0a1d1189924fd63487dcec1`. Trained with Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, librosa 0.11.0, soundfile 0.14.0, scikit-learn 1.9.1, and joblib 1.6.0. `joblib` artifacts must be loaded only from trusted sources. The app processes uploads in memory and does not retain them.
