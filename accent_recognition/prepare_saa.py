"""Create a manifest from the Speech Accent Archive Kaggle layout."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def prepare(csv_path: Path, recordings: Path, output: Path, min_speakers: int = 20) -> int:
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    candidates = []
    for row in rows:
        label = (row.get("native_language") or "").strip().lower()
        filename = (row.get("filename") or "").strip()
        if not label or not filename or str(row.get("file_missing?", "")).lower() == "true":
            continue
        path = recordings / (filename if filename.endswith(".mp3") else filename + ".mp3")
        if path.is_file():
            candidates.append((path, label, str(row.get("speakerid") or filename)))
    counts = Counter(label for _, label, _ in candidates)
    chosen = [(path, label, speaker) for path, label, speaker in candidates
              if counts[label] >= min_speakers]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "accent", "speaker_id"])
        writer.writeheader()
        for path, label, speaker in chosen:
            writer.writerow({"path": str(path.resolve()), "accent": label, "speaker_id": speaker})
    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--min-speakers", type=int, default=20)
    args = parser.parse_args()
    count = prepare(args.metadata, args.recordings, args.output, args.min_speakers)
    print(f"Wrote {count} rows to {args.output}")
    print("Labels are native-language proxies, not verified accent annotations.")


if __name__ == "__main__":
    main()
