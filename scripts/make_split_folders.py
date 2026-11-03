import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SOURCE_DIR = BASE_DIR / "data" / "quality_dataset_600"
OUTPUT_DIR = BASE_DIR / "data" / "quality_split"

SEED = 24500007
SPLITS = ["train", "validation", "test"]
CLASSES = ["bad", "good"]


def main():
    global SOURCE_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Copy the existing metadata split into class folders.")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    SOURCE_DIR = args.source.resolve()
    OUTPUT_DIR = args.output.resolve()
    metadata_path = SOURCE_DIR / "metadata.jsonl"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    records = []

    with metadata_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 1800:
        raise ValueError(
            f"Expected 1800 records, found {len(records)}."
        )

    source_splits = defaultdict(set)
    image_ids = set()
    destination_paths = set()
    prepared = []

    for record in records:
        split = record["split"]
        quality = record["quality_label"]

        if split not in SPLITS or quality not in CLASSES:
            raise ValueError(
                f"Invalid split or class: {record['image_id']}"
            )

        if record.get("seed") != SEED:
            raise ValueError(
                f"Wrong seed: {record['image_id']}"
            )

        if record["image_id"] in image_ids:
            raise ValueError(
                f"Duplicate image ID: {record['image_id']}"
            )

        image_ids.add(record["image_id"])

        relative_source = Path(
            record["image_path"].replace("\\", "/")
        )
        source_path = SOURCE_DIR / relative_source

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {source_path}"
            )

        with Image.open(source_path) as image:
            image.load()

            if image.size != (640, 640):
                raise ValueError(
                    f"Wrong image size: {source_path}"
                )

        relative_destination = (
            Path(split) / quality / source_path.name
        )

        if relative_destination in destination_paths:
            raise ValueError(
                f"Duplicate destination: {relative_destination}"
            )

        destination_paths.add(relative_destination)

        source_key = (
            record["class_name"],
            record["source_id"],
        )
        source_splits[source_key].add(split)

        prepared.append((
            record,
            source_path,
            relative_destination,
        ))

    if any(len(values) > 1 for values in source_splits.values()):
        raise ValueError(
            "Data leakage: a source appears in multiple splits."
        )

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT_DIR}"
        )

    for split in SPLITS:
        for quality in CLASSES:
            (OUTPUT_DIR / split / quality).mkdir(
                parents=True,
                exist_ok=True,
            )

    output_records = []

    for record, source_path, relative_destination in prepared:
        shutil.copy2(
            source_path,
            OUTPUT_DIR / relative_destination,
        )

        output_record = record.copy()
        output_record["image_path"] = (
            relative_destination.as_posix()
        )
        output_records.append(output_record)

    with (OUTPUT_DIR / "metadata.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in output_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    counts = Counter(
        (record["split"], record["quality_label"])
        for record in output_records
    )

    print("Split folders created successfully.")
    print(f"Seed: {SEED}")
    print(f"Total images: {len(output_records)}")
    print("Source leakage: 0")
    print()

    for split in SPLITS:
        good = counts[(split, "good")]
        bad = counts[(split, "bad")]

        print(
            f"{split}: {good + bad} images "
            f"(good={good}, bad={bad})"
        )

    print()
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
