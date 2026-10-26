"""Contract tests using synthetic audio; these do not measure accent accuracy."""

import csv
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from accent_recognition.api import create_app
from accent_recognition.audio import AudioError, decode_audio, extract_features
from accent_recognition.model import AccentModel
from accent_recognition.train import train


def wav_bytes(frequency: float, seconds: float = 3.0) -> bytes:
    rate = 16_000
    t = np.arange(int(rate * seconds)) / rate
    signal = (0.22 * np.sin(2 * np.pi * frequency * t) *
              (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, signal, rate, format="WAV")
    return buffer.getvalue()


def test_audio_validation_and_stable_features():
    sample = decode_audio(wav_bytes(180))
    assert sample.duration_seconds == 3
    assert sample.speech_seconds >= 2
    features = extract_features(sample)
    assert features.ndim == 1 and np.isfinite(features).all()
    assert np.array_equal(features, extract_features(sample))
    with pytest.raises(AudioError):
        decode_audio(wav_bytes(180, 1))
    with pytest.raises(AudioError):
        decode_audio(b"not audio")


def test_training_and_api_round_trip(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "accent", "speaker_id"])
        writer.writeheader()
        for label, base in (("group-a", 160), ("group-b", 300)):
            for speaker in range(5):
                path = tmp_path / f"{label}-{speaker}.wav"
                path.write_bytes(wav_bytes(base + speaker * 8))
                writer.writerow({"path": path.name, "accent": label,
                                 "speaker_id": f"{label}-{speaker}"})
    artifact = tmp_path / "accent.joblib"
    metrics = train(manifest, artifact)
    assert artifact.is_file() and artifact.with_suffix(".metrics.json").is_file()
    assert metrics["train_speakers"] + metrics["test_speakers"] == 10
    assert set(metrics["labels"]) == {"group-a", "group-b"}
    assert 0 <= metrics["macro_f1"] <= 1
    model = AccentModel.load(artifact)
    assert model.labels == ["group-a", "group-b"]

    client = TestClient(create_app(artifact))
    assert client.get("/api/health").json()["model_loaded"] is True
    response = client.post("/api/predict", files={"audio": ("voice.wav", wav_bytes(165), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["accent"] in {None, "group-a", "group-b"}
    assert len(body["alternatives"]) == 2
    assert client.post("/api/predict", files={"audio": ("bad.wav", b"bad", "audio/wav")}).status_code == 422
    assert client.post("/api/predict", files={"audio": ("bad.txt", b"bad", "text/plain")}).status_code == 415


def test_untrained_server_reports_unavailable(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.joblib"))
    assert client.get("/api/health").json()["status"] == "model_unavailable"
    assert client.post("/api/predict", files={"audio": ("voice.wav", wav_bytes(180), "audio/wav")}).status_code == 503
