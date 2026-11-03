import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import ultralytics
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torchvision import transforms as T
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SEED = 24500007
IMAGE_SIZE = 640
CLASS_NAMES = ["bad", "good"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_transform(size=IMAGE_SIZE):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])


class QualityTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.torch_transforms = image_transform(self.args.imgsz)
        return dataset


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def limit_threads(trainer=None):
    torch.set_num_threads(4)


def verify_data(data_dir, baseline_path):
    metadata_path = data_dir / "metadata.jsonl"
    raw = metadata_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if digest != baseline["metadata_sha256"]:
        raise ValueError("Metadata differs from the MobileNetV2 experiment.")
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    counts = Counter()
    paths = set()
    groups = {}
    for row in records:
        split, label = row["split"], row["quality_label"]
        if split not in {"train", "validation", "test"} or label not in CLASS_NAMES:
            raise ValueError("Invalid split or class in metadata.")
        if int(row["seed"]) != SEED:
            raise ValueError("Unexpected seed in metadata.")
        relative = Path(row["image_path"].replace("\\", "/"))
        if relative.parts[:2] != (split, label) or ".." in relative.parts:
            raise ValueError(f"Invalid split image path: {relative}")
        path = data_dir / relative
        if not path.is_file() or path in paths:
            raise ValueError(f"Missing or duplicate image: {path}")
        with Image.open(path) as image:
            if image.size != (640, 640):
                raise ValueError(f"Unexpected image size: {path}")
            image.verify()
        key = (row["class_name"], row["source_id"])
        if key in groups and groups[key] != split:
            raise ValueError("Source leakage detected.")
        groups[key] = split
        paths.add(path)
        counts[(split, label)] += 1
    for split in ("train", "validation", "test"):
        actual = {p for p in (data_dir / split).rglob("*") if p.suffix.lower() in EXTENSIONS}
        expected = {p for p in paths if p.relative_to(data_dir).parts[0] == split}
        if actual != expected:
            raise ValueError(f"Folder contents differ from metadata: {split}")
        total = sum(counts[(split, label)] for label in CLASS_NAMES)
        if total != baseline["split_counts"][split] or any(counts[(split, label)] == 0 for label in CLASS_NAMES):
            raise ValueError(f"Split counts differ from baseline: {split}")
        print(f"{split}: {total} images (bad={counts[(split, 'bad')]}, good={counts[(split, 'good')]})")
    print(f"Seed: {SEED}; source leakage: 0; metadata matches MobileNetV2")
    return records, digest


def prepare_yolo_folders(data_dir, records, digest):
    adapter = data_dir.parent / "quality_split_yolo"
    marker = adapter / "metadata_sha256.txt"
    if adapter.exists() and (not marker.is_file() or marker.read_text().strip() != digest):
        raise FileExistsError(f"Unrecognized existing YOLO folder: {adapter}")
    adapter.mkdir(exist_ok=True)
    marker.write_text(digest, encoding="utf-8")
    expected = set()
    for row in records:
        split = "val" if row["split"] == "validation" else row["split"]
        source = data_dir / row["image_path"].replace("\\", "/")
        target = adapter / split / row["quality_label"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise ValueError(f"YOLO adapter image differs from source: {target}")
        expected.add(target)
    actual = {p for p in adapter.rglob("*") if p.suffix.lower() in EXTENSIONS}
    if actual != expected:
        raise ValueError("Unexpected images in YOLO adapter.")
    return adapter


def probabilities(core, batch):
    output = core(batch)
    output = output[0] if isinstance(output, (tuple, list)) else output
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError("Expected probabilities for two classes.")
    return output


def evaluate(best_path, data_dir, records, result_dir):
    wrapper = YOLO(str(best_path))
    names = wrapper.names
    actual_names = [names[i] for i in range(len(names))]
    if actual_names != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {actual_names}")
    core = wrapper.model.cpu().float().eval()
    limit_threads()
    transform = image_transform()
    test_rows = sorted((r for r in records if r["split"] == "test"), key=lambda r: r["image_path"])
    actual, predicted, probability_good = [], [], []
    errors = result_dir / "errors"
    errors.mkdir(exist_ok=True)
    with torch.inference_mode():
        for start in range(0, len(test_rows), 32):
            rows = test_rows[start:start + 32]
            tensors = []
            for row in rows:
                with Image.open(data_dir / row["image_path"].replace("\\", "/")) as image:
                    tensors.append(transform(image.convert("RGB")))
            values = probabilities(core, torch.stack(tensors)).cpu().numpy()
            probability_good.extend(values[:, 1].tolist())
            predicted.extend(values.argmax(axis=1).tolist())
            actual.extend(CLASS_NAMES.index(row["quality_label"]) for row in rows)
            print(f"Test evaluation: {min(start + 32, len(test_rows))}/{len(test_rows)}")
        images = []
        for row in test_rows[:100]:
            with Image.open(data_dir / row["image_path"].replace("\\", "/")) as image:
                images.append(image.convert("RGB").copy())
        durations = []
        for i in range(120):
            image = images[i % len(images)]
            started = time.perf_counter()
            batch = transform(image).unsqueeze(0)
            decision = int(probabilities(core, batch).argmax(dim=1).item())
            elapsed = time.perf_counter() - started
            if i >= 20:
                durations.append(elapsed * 1000)
    report = classification_report(actual, predicted, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, cmap="Blues")
    fig.tight_layout()
    fig.savefig(result_dir / "confusion_matrix_test.png", dpi=160)
    plt.close(fig)
    with (result_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "true_label", "predicted_label", "probability_good", "defect_type"])
        for row, truth, prediction, probability in zip(test_rows, actual, predicted, probability_good):
            writer.writerow([row["image_path"], CLASS_NAMES[truth], CLASS_NAMES[prediction], probability, row["defect_type"]])
            if truth != prediction:
                source = data_dir / row["image_path"].replace("\\", "/")
                shutil.copy2(source, errors / f"true_{CLASS_NAMES[truth]}_pred_{CLASS_NAMES[prediction]}_{source.name}")
    save_json(result_dir / "classification_report.json", report)
    return report, cm.tolist(), float(np.mean(durations)), sum(a != p for a, p in zip(actual, predicted)), sum(p.numel() for p in core.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolov8n-cls", "yolov8s-cls"], default="yolov8n-cls")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "quality_split")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASE_DIR / "results" / "mobilenetv2_frozen" / "results.json")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    data_dir = args.data.resolve()
    records, digest = verify_data(data_dir, args.baseline.resolve())
    if args.check_data:
        print("Data check completed. Training was not started.")
        return
    adapter = prepare_yolo_folders(data_dir, records, digest)
    experiment = "week7_" + args.model.replace("-", "_")
    result_dir = args.output.resolve() if args.output else BASE_DIR / "results" / experiment
    if result_dir.exists():
        raise FileExistsError(f"Results already exist; they will not be overwritten: {result_dir}")
    result_dir.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_interop_threads(1)
    limit_threads()
    torch.use_deterministic_algorithms(True, warn_only=True)
    result = {
        "experiment": experiment, "week": 7, "task": 3, "model": args.model,
        "regime": "full_finetuning", "input_size": [640, 640], "seed": SEED,
        "device": "CPU", "epochs_planned": 50, "epochs_run": 0,
        "train_time_s": 0, "latency_ms_per_image": None, "model_size_mb": None,
        "params": None, "test_samples": sum(r["split"] == "test" for r in records),
        "metrics": {}, "confusion_matrix": [], "history": {}, "status": "running",
        "metadata_sha256": digest, "class_indices": {"bad": 0, "good": 1},
        "command": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "hyperparameters": {"batch_size": 32, "learning_rate": 0.001, "optimizer": "Adam", "patience": 5, "preprocessing": "RGB; full-image bilinear resize to 640x640; pixels in [0,1]", "additional_augmentation": "none; existing dataset variants retained", "checkpoint_selection": "validation top1 accuracy (Ultralytics fitness)"},
        "latency_protocol": {"warmup": 20, "timed_predictions": 100, "batch_size": 1, "includes": "resize, preprocessing, inference, decision", "excludes": "image decoding, disk IO, model loading, network"},
        "environment": {"python": sys.version, "ultralytics": ultralytics.__version__, "pytorch": torch.__version__, "torchvision": torchvision.__version__, "platform": platform.platform(), "processor": platform.processor(), "cpu_threads": 4, "inter_op_threads": 1, "precision": "float32"},
    }
    save_json(result_dir / "results.json", result)
    started = None
    try:
        model = YOLO(args.model + ".pt")
        model.add_callback("on_train_start", limit_threads)
        started = time.perf_counter()
        model.train(trainer=QualityTrainer, data=str(adapter), epochs=50, patience=5,
                    imgsz=640, batch=32, seed=SEED, device="cpu", workers=0,
                    optimizer="Adam", lr0=0.001, deterministic=True, amp=False,
                    cache=False, plots=True, save=True, project=str(result_dir),
                    name="training", exist_ok=False, auto_augment=None, erasing=0.0,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, flipud=0.0,
                    scale=0.0, dropout=0.0)
        result["train_time_s"] = time.perf_counter() - started
        run_dir = Path(model.trainer.save_dir)
        with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(file)]
        result["epochs_run"] = len(rows)
        result["history"] = {key: [row[key] for row in rows] for key in rows[0]}
        best_path = result_dir / "best.pt"
        shutil.copy2(run_dir / "weights" / "best.pt", best_path)
        result["model_size_mb"] = best_path.stat().st_size / 1_000_000
        result["model_file"] = "best.pt"
        result["size_note"] = "File size of Ultralytics best.pt; evaluated after loading as float32."
        report, cm, latency, mistakes, params = evaluate(best_path, data_dir, records, result_dir)
        result.update({"metrics": {"accuracy": report["accuracy"], "precision_macro": report["macro avg"]["precision"], "recall_macro": report["macro avg"]["recall"], "f1_macro": report["macro avg"]["f1-score"], "per_class": {name: report[name] for name in CLASS_NAMES}}, "confusion_matrix": cm, "latency_ms_per_image": latency, "misclassified_samples": mistakes, "params": params, "status": "completed"})
        save_json(result_dir / "results.json", result)
        print("Training and evaluation completed.")
        print(f"Model: {args.model}")
        print(f"Test accuracy: {report['accuracy']:.4f}")
        print(f"Test macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Bad recall: {report['bad']['recall']:.4f}")
        print(f"Latency: {latency:.2f} ms/image")
        print(f"Model size: {result['model_size_mb']:.2f} MB")
        print(f"Results saved to: {result_dir}")
    except Exception as error:
        result["status"] = "failed"
        result["note"] = f"{type(error).__name__}: {error}"
        if started is not None and not result["train_time_s"]:
            result["train_time_s"] = time.perf_counter() - started
        save_json(result_dir / "results.json", result)
        (result_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
