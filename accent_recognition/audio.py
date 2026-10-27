"""Shared, deterministic audio preprocessing for training and serving."""

from dataclasses import dataclass
from io import BytesIO
from math import gcd
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
MIN_SECONDS = 2.0
MAX_SECONDS = 30.0
MAX_BYTES = 12 * 1024 * 1024
N_MFCC = 20


class AudioError(ValueError):
    """Audio cannot be used for accent inference."""


@dataclass(frozen=True)
class AudioSample:
    samples: np.ndarray
    duration_seconds: float
    speech_seconds: float


def decode_audio(source: bytes | Path, *, truncate_long: bool = False) -> AudioSample:
    """Decode audio and enforce duration and speech-quality limits.

    Training may crop archive recordings longer than the serving limit.
    """
    if isinstance(source, bytes) and len(source) > MAX_BYTES:
        raise AudioError("Audio file is larger than 12 MB.")
    stream = BytesIO(source) if isinstance(source, bytes) else source
    try:
        with sf.SoundFile(stream) as file:
            duration = len(file) / file.samplerate
            if duration < MIN_SECONDS or (duration > MAX_SECONDS and not truncate_long):
                raise AudioError("Record between 2 and 30 seconds of speech.")
            if file.channels > 2 or file.samplerate < 8_000 or file.samplerate > 96_000:
                raise AudioError("Unsupported channel count or sample rate.")
            frames = min(len(file), int(MAX_SECONDS * file.samplerate))
            samples = file.read(frames=frames, dtype="float32", always_2d=True).mean(axis=1)
            duration = len(samples) / file.samplerate
            rate = file.samplerate
    except (RuntimeError, sf.LibsndfileError, ValueError) as exc:
        if isinstance(exc, AudioError):
            raise
        raise AudioError("Could not decode audio. Use WAV, FLAC, or OGG.") from exc
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.003:
        raise AudioError("No usable speech was detected. Try a clearer recording.")
    factor = gcd(rate, SAMPLE_RATE)
    samples = resample_poly(samples, SAMPLE_RATE // factor, rate // factor).astype(np.float32)
    # A conservative energy gate rejects silence without deleting accent cues.
    frame = int(0.03 * SAMPLE_RATE)
    padded = np.pad(samples, (0, (-len(samples)) % frame))
    rms = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1))
    speech_seconds = float(np.count_nonzero(rms > max(0.008, float(rms.max()) * 0.06)) * 0.03)
    if speech_seconds < MIN_SECONDS:
        raise AudioError("At least 2 seconds of clear speech are required.")
    return AudioSample(samples, duration, speech_seconds)


def extract_features(sample: AudioSample) -> np.ndarray:
    """Fixed-size MFCC and spectral summaries for variable-length recordings."""
    y = sample.samples
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=400, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    spectral = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=400, hop_length=160)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=160)
    bands = np.concatenate([mfcc, delta, spectral, zcr], axis=0)
    result = np.concatenate([bands.mean(axis=1), bands.std(axis=1),
                             np.percentile(bands, [10, 50, 90], axis=1).ravel()])
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
