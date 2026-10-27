"""Speaker-disjoint training and evaluation from a labeled audio manifest."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audio import AudioError, decode_audio, extract_features
from .model import AccentModel


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"path", "accent", "speaker_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs path, accent, and speaker_id columns")
        rows = [row for row in reader if all(row.get(key, "").strip() for key in required)]
    if not rows:
        raise ValueError("Manifest has no complete rows")
    return rows


def train(manifest: Path, output: Path, seed: int = 42, threshold: float = 0.5) -> dict:
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    rows = read_manifest(manifest)
    speakers: dict[str, str] = {}
    features, labels, groups = [], [], []
    skipped = []
    for row in rows:
        label, speaker = row["accent"].strip(), row["speaker_id"].strip()
        if speaker in speakers and speakers[speaker] != label:
            raise ValueError(f"Speaker {speaker} has conflicting labels")
        speakers[speaker] = label
        audio_path = (manifest.parent / row["path"]).resolve()
        try:
            feature = extract_features(decode_audio(audio_path, truncate_long=True))
        except (AudioError, OSError) as exc:
            skipped.append({"path": row["path"], "reason": str(exc)})
            continue
        features.append(feature)
        labels.append(label)
        groups.append(speaker)
    if not features:
        raise ValueError("No usable audio in manifest")
    per_class = Counter(labels)
    speaker_classes: dict[str, set[str]] = defaultdict(set)
    for label, speaker in zip(labels, groups):
        speaker_classes[label].add(speaker)
    if len(per_class) < 2 or any(len(ids) < 3 for ids in speaker_classes.values()):
        raise ValueError("Need at least two classes and three distinct speakers per class")

    X, y, group_array = np.stack(features), np.array(labels), np.array(groups)
    all_labels = sorted(per_class)
    split = None
    # GroupShuffleSplit prevents a speaker's clips appearing in both train and test.
    for train_idx, test_idx in GroupShuffleSplit(n_splits=100, test_size=0.2,
                                                  random_state=seed).split(X, y, group_array):
        if (set(y[train_idx]) == set(all_labels) and set(y[test_idx]) == set(all_labels)
                and min(Counter(y[train_idx]).values()) >= 2):
            split = train_idx, test_idx
            break
    if split is None:
        raise ValueError("Could not create a speaker-disjoint split containing every class")
    train_idx, test_idx = split
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
    )
    estimator.fit(X[train_idx], y[train_idx])
    predicted = estimator.predict(X[test_idx])
    probabilities = estimator.predict_proba(X[test_idx])
    confidence = probabilities.max(axis=1)
    accepted = confidence >= threshold
    metrics = {
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "train_speakers": len(set(group_array[train_idx])),
        "test_speakers": len(set(group_array[test_idx])),
        "class_counts": dict(per_class),
        "accuracy": round(float(accuracy_score(y[test_idx], predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y[test_idx], predicted)), 4),
        "macro_f1": round(float(f1_score(y[test_idx], predicted, average="macro")), 4),
        "coverage_at_threshold": round(float(accepted.mean()), 4),
        "accuracy_at_threshold": round(float(accuracy_score(y[test_idx][accepted], predicted[accepted])), 4)
            if accepted.any() else None,
        "confusion_matrix": confusion_matrix(y[test_idx], predicted, labels=all_labels).tolist(),
        "classification_report": classification_report(y[test_idx], predicted,
                                                         labels=all_labels, output_dict=True,
                                                         zero_division=0),
        "labels": all_labels,
        "skipped": skipped,
        "seed": seed,
        "threshold": threshold,
        "model": "standard_scaler_logistic_regression_balanced",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = AccentModel(estimator, list(estimator.classes_), threshold, metrics)
    temporary = output.with_suffix(output.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/accent.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.seed, args.threshold), indent=2))


if __name__ == "__main__":
    main()
