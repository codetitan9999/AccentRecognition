"""Versioned model artifact and inference contract."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .audio import decode_audio, extract_features

ARTIFACT_VERSION = 1


@dataclass
class AccentModel:
    estimator: object
    labels: list[str]
    threshold: float
    metrics: dict
    version: int = ARTIFACT_VERSION

    @classmethod
    def load(cls, path: Path) -> "AccentModel":
        # joblib uses pickle. Load only artifacts created by a trusted training run.
        artifact = joblib.load(path)
        if not isinstance(artifact, cls) or artifact.version != ARTIFACT_VERSION:
            raise ValueError("Unsupported model artifact")
        return artifact

    def predict(self, audio: bytes) -> dict:
        sample = decode_audio(audio)
        features = extract_features(sample)
        probabilities = self.estimator.predict_proba(features[None, :])[0]
        ranked = np.argsort(probabilities)[::-1]
        best = int(ranked[0])
        score = float(probabilities[best])
        return {
            "accent": self.labels[best] if score >= self.threshold else None,
            "confidence": round(score, 4),
            "uncertain": score < self.threshold,
            "alternatives": [
                {"accent": self.labels[int(i)], "confidence": round(float(probabilities[i]), 4)}
                for i in ranked[:3]
            ],
            "duration_seconds": round(sample.duration_seconds, 2),
            "speech_seconds": round(sample.speech_seconds, 2),
            "model_version": self.version,
        }
