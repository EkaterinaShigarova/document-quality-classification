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
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SEED = 24500007
IMAGE_SIZE = 224
CLASS_NAMES = ["bad", "good"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_transform(size=IMAGE_SIZE):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class QualityDataset(Dataset):
    def __init__(self, data_dir, records, split):
        self.data_dir = data_dir
        self.rows = sorted((r for r in records if r["split"] == split), key=lambda r: r["image_path"])
        self.transform = image_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_dir / row["image_path"].replace("\\", "/")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, float(CLASS_NAMES.index(row["quality_label"]))


def build_model(pretrained):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 1))
    return model.cpu().float()


def probabilities(core, batch):
    probability_good = torch.sigmoid(core(batch).reshape(-1))
    return torch.stack([1 - probability_good, probability_good], dim=1)


def run_epoch(model, loader, criterion, optimizer=None):
    model.eval()
    if optimizer is not None:
        model.classifier.train()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            labels = labels.float()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            correct += int(((logits.detach() >= 0) == labels.bool()).sum())
            count += labels.numel()
    return total_loss / count, correct / count


def save_curves(history, result_dir):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[key], label="Train")
        ax.plot(epochs, history["val_" + key], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.capitalize())
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "curves.png", dpi=160)
    plt.close(fig)



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


def evaluate(best_path, data_dir, records, result_dir):
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    core = build_model(pretrained=False)
    core.load_state_dict(checkpoint["state_dict"])
    core.eval()
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
            predicted.extend((values[:, 1] >= 0.5).astype(int).tolist())
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
            decision = int(probabilities(core, batch)[0, 1].item() >= 0.5)
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
    parser = argparse.ArgumentParser(description="Train a frozen EfficientNet-B0 quality classifier.")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "quality_split")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results" / "efficientnetb0_frozen")
    parser.add_argument("--baseline", type=Path, default=BASE_DIR / "results" / "mobilenetv2_frozen" / "results.json")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    data_dir = args.data.resolve()
    records, digest = verify_data(data_dir, args.baseline.resolve())
    if args.check_data:
        print("Data check completed. Training was not started.")
        return
    result_dir = args.output.resolve()
    if result_dir.exists():
        raise FileExistsError(f"Results already exist; they will not be overwritten: {result_dir}")
    result_dir.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    limit_threads()
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {
        "experiment": "efficientnetb0_frozen", "week": 8, "task": 4,
        "model": "EfficientNet-B0", "regime": "frozen_base",
        "input_size": [224, 224], "seed": SEED, "device": "CPU",
        "epochs_planned": 50, "epochs_run": 0, "best_epoch": None,
        "train_time_s": 0, "latency_ms_per_image": None, "model_size_mb": None,
        "params": None, "test_samples": sum(r["split"] == "test" for r in records),
        "metrics": {}, "confusion_matrix": [], "history": {}, "status": "running",
        "metadata_sha256": digest, "class_indices": {"bad": 0, "good": 1},
        "split_counts": dict(Counter(r["split"] for r in records)),
        "command": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "hyperparameters": {
            "batch_size": 32, "learning_rate": 0.001, "patience": 5,
            "optimizer": "Adam", "loss": "BCEWithLogitsLoss", "threshold": 0.5,
            "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
            "head": "Dropout(0.2), Linear(1280, 1); sigmoid at inference",
            "preprocessing": "RGB; full-image bilinear resize to 224x224; pixels to [0,1]; ImageNet mean/std normalization",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "checkpoint_selection": "minimum validation loss",
            "additional_augmentation": "none; existing dataset variants retained",
            "base_mode": "eval; base parameters and BatchNorm statistics frozen",
        },
        "latency_protocol": {
            "warmup": 20, "timed_predictions": 100, "batch_size": 1,
            "includes": "resize, preprocessing, inference, decision",
            "excludes": "image decoding, disk IO, model loading, network",
        },
        "environment": {
            "python": sys.version, "pytorch": torch.__version__, "torchvision": torchvision.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_threads": 4, "inter_op_threads": 1, "precision": "float32",
        },
    }
    save_json(result_dir / "results.json", result)
    started = None
    try:
        generator = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(QualityDataset(data_dir, records, "train"), batch_size=32, shuffle=True, num_workers=0, generator=generator)
        validation_loader = DataLoader(QualityDataset(data_dir, records, "validation"), batch_size=32, shuffle=False, num_workers=0)
        model = build_model(pretrained=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
        result["params"] = sum(p.numel() for p in model.parameters())
        result["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: EfficientNet-B0; total parameters: {result['params']:,}; trainable: {result['trainable_params']:,}")
        history = {key: [] for key in ["loss", "accuracy", "val_loss", "val_accuracy"]}
        best_loss = float("inf")
        patience_count = 0
        best_path = result_dir / "best.pt"
        started = time.perf_counter()
        with (result_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", *history.keys(), "epoch_time_s"])
            for epoch in range(1, 51):
                epoch_start = time.perf_counter()
                train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
                validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion)
                if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                    raise ValueError("Non-finite training or validation loss.")
                for key, value in zip(history, [train_loss, train_accuracy, validation_loss, validation_accuracy]):
                    history[key].append(float(value))
                elapsed = time.perf_counter() - epoch_start
                writer.writerow([epoch, *[history[key][-1] for key in history], elapsed])
                file.flush()
                print(f"Epoch {epoch}/50 - {elapsed:.1f}s - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {validation_loss:.4f} - val_accuracy: {validation_accuracy:.4f}", flush=True)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    patience_count = 0
                    result["best_epoch"] = epoch
                    torch.save({"state_dict": model.state_dict(), "model": "efficientnet_b0", "head": "Dropout(0.2), Linear(1280,1)", "class_names": CLASS_NAMES, "seed": SEED, "epoch": epoch, "val_loss": validation_loss, "input_size": [224, 224]}, best_path)
                else:
                    patience_count += 1
                result.update({"epochs_run": epoch, "history": history, "train_time_s": time.perf_counter() - started})
                save_json(result_dir / "results.json", result)
                if patience_count >= 5:
                    print(f"Early stopping. Best epoch: {result['best_epoch']}")
                    break
        result["train_time_s"] = time.perf_counter() - started
        save_curves(history, result_dir)
        report, cm, latency, mistakes, params = evaluate(best_path, data_dir, records, result_dir)
        result.update({
            "metrics": {"accuracy": report["accuracy"], "precision_macro": report["macro avg"]["precision"], "recall_macro": report["macro avg"]["recall"], "f1_macro": report["macro avg"]["f1-score"], "per_class": {name: report[name] for name in CLASS_NAMES}},
            "confusion_matrix": cm, "latency_ms_per_image": latency,
            "misclassified_samples": mistakes, "params": params,
            "model_size_mb": best_path.stat().st_size / 1_000_000,
            "model_file": "best.pt", "size_note": "Float32 state_dict with architecture metadata; no optimizer state. Reconstruct using build_model(pretrained=False).",
            "status": "completed",
        })
        save_json(result_dir / "results.json", result)
        print("Training and evaluation completed.")
        print(f"Test accuracy: {report['accuracy']:.4f}")
        print(f"Test macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Bad recall: {report['bad']['recall']:.4f}")
        print(f"Latency: {latency:.2f} ms/image")
        print(f"Model size: {result['model_size_mb']:.2f} MB")
        print(f"Misclassified test images: {mistakes}")
        print(f"Results saved to: {result_dir}")
    except Exception as error:
        result["status"] = "failed"
        result["note"] = f"{type(error).__name__}: {error}"
        save_json(result_dir / "results.json", result)
        (result_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
