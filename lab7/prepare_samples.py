import argparse
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
METADATA_SHA256 = "db7b8e2ffd31d4991e3d6ca57b061ce517b837a98c91312b352a40e5491048de"
BOUNDARY = "validation/bad/bad_inn_glare_282_v3.jpg"


def select_rows(records):
    boundary = next((row for row in records if row["image_path"].replace("\\", "/") == BOUNDARY), None)
    if boundary is None or boundary["split"] != "validation" or boundary["quality_label"] != "bad":
        raise ValueError("The validation boundary example is missing.")
    selected = []
    for label in ["good", "bad"]:
        rows = sorted(
            (row for row in records if row["split"] == "validation" and row["quality_label"] == label),
            key=lambda row: row["image_path"],
        )
        if len(rows) < 10:
            raise ValueError("At least ten validation images per class are required.")
        chosen = rows[:10]
        if label == "bad" and not any(row["image_path"] == boundary["image_path"] for row in chosen):
            chosen[-1] = boundary
        selected.extend(chosen)
    return selected


def prepare(data_dir, output_dir):
    raw = (data_dir / "metadata.jsonl").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != METADATA_SHA256:
        raise ValueError("Metadata differs from the Lab 6 split. Use the original quality_split directory.")
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    selected = select_rows(records)
    rows = []
    for index, row in enumerate(selected):
        source = data_dir / row["image_path"].replace("\\", "/")
        with Image.open(source) as image:
            if image.size != (640, 640) or image.format not in {"JPEG", "PNG"}:
                raise ValueError(f"Unexpected source image: {source.name}")
            image.verify()
        relative = row["image_path"].replace("\\", "/")
        rows.append({
            "file": f"{index + 1:02d}_{source.name}",
            "source_image": relative,
            "source_split": "validation",
            "true_quality": row["quality_label"],
            "expected_status": "accepted" if row["quality_label"] == "good" else "rejected",
            "boundary": relative == BOUNDARY,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    if output_dir.exists():
        raise FileExistsError("The samples directory already exists. Existing files were not changed.")
    output_dir.mkdir(parents=True)
    for source_row, target_row in zip(selected, rows):
        source = data_dir / source_row["image_path"].replace("\\", "/")
        shutil.copy2(source, output_dir / target_row["file"])
    (output_dir / "non_image.jpg").write_text("This is a text file renamed to JPG.", encoding="utf-8")
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (30, 120, 200)).save(buffer, format="PNG")
    (output_dir / "corrupted.png").write_bytes(buffer.getvalue()[:45])
    rows.extend([
        {"file": "non_image.jpg", "true_quality": None, "expected_status": "error", "boundary": False},
        {"file": "corrupted.png", "true_quality": None, "expected_status": "error", "boundary": False},
    ])
    manifest = {
        "purpose": "Lab 7 integration and latency examples; not a new independent accuracy benchmark",
        "source_split": "validation",
        "metadata_sha256": digest,
        "seed": 24500007,
        "selection": "First ten sorted validation images per quality class; include the recorded 5B boundary case in the ten bad examples",
        "expected_counts": {"accepted": 10, "rejected": 10, "error": 2},
        "files": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Prepared 22 samples: 10 good, 10 bad, 1 renamed text file, 1 corrupted PNG.")
    print("All 20 real images were copied from validation. Original files were not modified.")
    print(f"Samples: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed integration samples from the existing Lab 6 validation split.")
    parser.add_argument("--data", type=Path, default=BASE_DIR.parent / "lab4" / "data" / "quality_split")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "samples")
    args = parser.parse_args()
    try:
        prepare(args.data.resolve(), args.output.resolve())
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
