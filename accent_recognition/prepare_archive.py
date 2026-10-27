"""Extract one copy of labeled Speech Accent Archive recordings from Kaggle v1."""

import argparse
import csv
import re
from pathlib import Path
from zipfile import ZipFile

DEFAULT_LABELS = (
    "english", "spanish", "arabic", "mandarin", "dutch",
)
FILENAME = re.compile(r"recordings/([a-z_-]+?)([0-9]+)\.mp3$")


def prepare(archive: Path, output_dir: Path, labels: tuple[str, ...] = DEFAULT_LABELS) -> int:
    clips = output_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    count = 0
    with ZipFile(archive) as source, manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "accent", "speaker_id"])
        writer.writeheader()
        for name in source.namelist():
            match = FILENAME.fullmatch(name)
            if not match or match.group(1) not in labels:
                continue
            label, number = match.groups()
            destination = clips / f"{label}{number}.mp3"
            if not destination.is_file():
                with source.open(name) as input_file, destination.open("wb") as output_file:
                    while block := input_file.read(1024 * 1024):
                        output_file.write(block)
            writer.writerow({"path": f"clips/{destination.name}", "accent": label,
                             "speaker_id": f"{label}-{number}"})
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    args = parser.parse_args()
    print(f"Prepared {prepare(args.archive, args.output_dir, tuple(args.labels))} recordings")


if __name__ == "__main__":
    main()
