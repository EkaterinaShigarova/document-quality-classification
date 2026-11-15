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
import warnings
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
LEARNING_RATE = 1e-5
FINETUNE_LAYERS = 30
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
        self.rows = sorted(
            (r for r in records if r["split"] == split),
            key=lambda r: r["image_path"],
        )
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


def unfreeze_top30(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = [
        ("features." + name, module)
        for name, module in model.features.named_modules()
        if not list(module.children()) and list(module.parameters(recurse=False))
    ]
    if len(layers) < FINETUNE_LAYERS:
        raise ValueError("The base has fewer than 30 parameterized leaf modules.")
    selected = layers[-FINETUNE_LAYERS:]
    for name, module in selected:
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return [name for name, module in selected]


def load_frozen_checkpoint(checkpoint_path, frozen):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_fields = [
        ("model", "efficientnet_b0"),
        ("class_names", CLASS_NAMES),
        ("seed", SEED),
        ("input_size", [224, 224]),
        ("epoch", frozen["best_epoch"]),
    ]
    for field, expected in expected_fields:
        if checkpoint.get(field) != expected:
            raise ValueError(f"The frozen checkpoint has an unexpected {field}.")
    expected_val_loss = frozen["history"]["val_loss"][frozen["best_epoch"] - 1]
    if not np.isclose(checkpoint["val_loss"], expected_val_loss):
        raise ValueError("The checkpoint validation loss differs from frozen results.")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != frozen["params"]:
        raise ValueError("The parameter count differs from the frozen model.")
    return model, checkpoint


def load_frozen_results(path):
    frozen = json.loads(path.read_text(encoding="utf-8-sig"))
    if frozen.get("status") != "completed" or frozen.get("regime") != "frozen_base":
        raise ValueError("A completed frozen-base results.json is required.")
    if frozen.get("model") != "EfficientNet-B0" or frozen.get("seed") != SEED:
        raise ValueError("The frozen model name or seed does not match.")
    if (
        frozen.get("input_size") != [224, 224]
        or frozen.get("class_indices") != {"bad": 0, "good": 1}
    ):
        raise ValueError("The frozen input size or class order does not match.")
    expected = {
        "batch_size": 32,
        "patience": 5,
        "threshold": 0.5,
        "optimizer": "Adam",
    }
    if frozen.get("epochs_planned") != 50 or any(
        frozen["hyperparameters"].get(k) != v for k, v in expected.items()
    ):
        raise ValueError(
            "The frozen protocol differs from the required fine-tuning protocol."
        )
    return frozen


def logged_gap(result):
    index = result["best_epoch"] - 1
    history = result["history"]
    return {
        "train_accuracy": history["accuracy"][index],
        "val_accuracy": history["val_accuracy"][index],
        "accuracy_gap": (
            history["accuracy"][index] - history["val_accuracy"][index]
        ),
        "train_loss": history["loss"][index],
        "val_loss": history["val_loss"][index],
    }


def write_delta_memo(frozen, tuned, report_path):
    if frozen["metadata_sha256"] != tuned["metadata_sha256"]:
        raise ValueError("Cannot compare experiments with different metadata.")

    lines = [
        "# Fine-Tuning Delta",
        "",
        "Student: Ekaterina Shigarova, IIKb-24. Task 5A. Seed: 24500007.",
        "",
        (
            "Values come from the frozen and fine-tuned results.json files. "
            "Quality metrics use the same test split."
        ),
        "",
        "| Metric | Frozen | Fine-tuned | Delta |",
        "|---|---:|---:|---:|",
    ]

    comparisons = [
        (
            "Accuracy",
            frozen["metrics"]["accuracy"],
            tuned["metrics"]["accuracy"],
            6,
        ),
        (
            "Precision macro",
            frozen["metrics"]["precision_macro"],
            tuned["metrics"]["precision_macro"],
            6,
        ),
        (
            "Recall macro",
            frozen["metrics"]["recall_macro"],
            tuned["metrics"]["recall_macro"],
            6,
        ),
        (
            "F1 macro",
            frozen["metrics"]["f1_macro"],
            tuned["metrics"]["f1_macro"],
            6,
        ),
        (
            "Bad recall",
            frozen["metrics"]["per_class"]["bad"]["recall"],
            tuned["metrics"]["per_class"]["bad"]["recall"],
            6,
        ),
        (
            "False accepts",
            frozen["confusion_matrix"][0][1],
            tuned["confusion_matrix"][0][1],
            0,
        ),
        (
            "False rejects",
            frozen["confusion_matrix"][1][0],
            tuned["confusion_matrix"][1][0],
            0,
        ),
        (
            "Latency (ms/image)",
            frozen["latency_ms_per_image"],
            tuned["latency_ms_per_image"],
            2,
        ),
        (
            "Size (MB)",
            frozen["model_size_mb"],
            tuned["model_size_mb"],
            2,
        ),
        (
            "Parameters",
            frozen["params"],
            tuned["params"],
            0,
        ),
        (
            "Trainable parameters",
            frozen["trainable_params"],
            tuned["trainable_params"],
            0,
        ),
        (
            "Training time (min)",
            frozen["train_time_s"] / 60,
            tuned["train_time_s"] / 60,
            2,
        ),
        (
            "Epochs run",
            frozen["epochs_run"],
            tuned["epochs_run"],
            0,
        ),
    ]

    for name, old, new, digits in comparisons:
        lines.append(
            f"| {name} | {old:.{digits}f} | {new:.{digits}f} "
            f"| {new - old:+.{digits}f} |"
        )

    fg = logged_gap(frozen)
    tg = logged_gap(tuned)

    lines += [
        "",
        "## Train and validation at the selected checkpoint",
        "",
        (
            "| Run | Epoch | Train accuracy | Validation accuracy "
            "| Train minus validation | Train loss | Validation loss |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, row, gap in [
        ("Frozen", frozen, fg),
        ("Fine-tuned", tuned, tg),
    ]:
        lines.append(
            f"| {name} | {row['best_epoch']} "
            f"| {gap['train_accuracy']:.6f} "
            f"| {gap['val_accuracy']:.6f} "
            f"| {gap['accuracy_gap']:+.6f} "
            f"| {gap['train_loss']:.6f} "
            f"| {gap['val_loss']:.6f} |"
        )

    qualified = []

    for name, row in [("frozen", frozen), ("fine-tuned", tuned)]:
        metrics = row["metrics"]
        if (
            metrics["accuracy"] >= 0.92
            and metrics["f1_macro"] >= 0.90
            and metrics["per_class"]["bad"]["recall"] >= 0.90
            and row["latency_ms_per_image"] < 100
            and row["model_size_mb"] <= 50
        ):
            qualified.append((name, row))

    if qualified:
        best_f1 = max(row["metrics"]["f1_macro"] for name, row in qualified)
        candidates = [
            (name, row)
            for name, row in qualified
            if best_f1 - row["metrics"]["f1_macro"] <= 0.01
        ]
        selected, selected_row = min(
            candidates,
            key=lambda item: (
                item[1]["latency_ms_per_image"],
                item[1]["model_size_mb"],
            ),
        )
        verdict = (
            "The original acceptance and F1-within-0.01 latency tie-break "
            f"selects the {selected} version in this benchmark."
        )
    else:
        verdict = "Neither version passes all original acceptance thresholds."

    f1_delta = tuned["metrics"]["f1_macro"] - frozen["metrics"]["f1_macro"]
    accuracy_delta = 100 * (
        tuned["metrics"]["accuracy"] - frozen["metrics"]["accuracy"]
    )

    if f1_delta > 0:
        changed = "increased"
    elif f1_delta < 0:
        changed = "decreased"
    else:
        changed = "did not change"

    after_best = tuned["history"]["val_loss"][tuned["best_epoch"]:]

    if tg["accuracy_gap"] > 0:
        gap_note = (
            "The selected checkpoint has higher train than validation "
            "accuracy, which is a warning to inspect for overfitting."
        )
    else:
        gap_note = (
            "The selected checkpoint does not have higher train "
            "than validation accuracy."
        )

    if after_best:
        trend_note = (
            "Validation loss rose after the selected epoch before early stopping."
        )
    else:
        trend_note = "No later epochs are available after the selected checkpoint."

    lr_ratio = frozen["hyperparameters"]["learning_rate"] / LEARNING_RATE

    lines += [
        "",
        "## Five-line interpretation",
        "",
        (
            f"1. Test macro F1 {changed} by {f1_delta:+.6f}, "
            f"and accuracy changed by {accuracy_delta:+.2f} percentage points."
        ),
        (
            "2. False accepts changed from "
            f"{frozen['confusion_matrix'][0][1]} "
            f"to {tuned['confusion_matrix'][0][1]}, "
            "and false rejects changed from "
            f"{frozen['confusion_matrix'][1][0]} "
            f"to {tuned['confusion_matrix'][1][0]}."
        ),
        (
            "3. The train-minus-validation accuracy gap changed from "
            f"{fg['accuracy_gap']:+.6f} to {tg['accuracy_gap']:+.6f}; "
            f"{gap_note[0].lower() + gap_note[1:]} {trend_note} "
            "The gap alone does not prove overfitting."
        ),
        (
            "4. The architecture and total parameter count stayed the same, "
            "so inference cost and float32 checkpoint size should remain "
            "similar; the measured deltas above include timing variation "
            "and minor file-metadata overhead."
        ),
        (
            f"5. {verdict} This is a provisional CPU prototype choice: "
            "the gap, glare failures and single-run timing uncertainty "
            "still require review, and real-photo performance remains untested."
        ),
        "",
        "## Protocol",
        "",
        (
            "I loaded the best frozen checkpoint, then unfroze the last 30 "
            "parameterized leaf modules of the feature extractor plus the "
            "existing classification head. Containers and layers without "
            "parameters are excluded from the count; results.json lists the "
            "exact names. Earlier parameters stay frozen. BatchNorm running "
            "statistics and base stochastic-depth mode remain fixed as in "
            "the frozen run; gradients still update the selected weights "
            "and BatchNorm affine parameters. The classifier dropout "
            "remains active in training."
        ),
        "",
        (
            f"Adam learning rate is 1e-5, which is {lr_ratio:g} times lower "
            "than this frozen run's head learning rate. The handout's factor "
            "of 10 is inconsistent with its 1e-3 to 1e-5 values. Smaller "
            "updates reduce the risk of damaging useful pretrained features."
        ),
        "",
        (
            "The seed, split, 224 x 224 preprocessing, batch size 32, "
            "50-epoch maximum and patience 5 are unchanged. Minimum validation "
            "loss selects the checkpoint; test data is evaluated after training. "
            "Gaps use the training and validation logs at their respective "
            "best epochs, with classifier dropout active only for training."
        ),
        "",
        (
            f"Frozen results SHA256: {tuned['parent_results_sha256']}. "
            f"Frozen checkpoint SHA256: {tuned['parent_checkpoint_sha256']}."
        ),
        "",
        (
            "Implementation reference: "
            "https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"
        ),
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines))


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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def limit_threads(trainer=None):
    torch.set_num_threads(4)


def verify_data(data_dir, baseline_path):
    metadata_path = data_dir / "metadata.jsonl"
    raw = metadata_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    if digest != baseline["metadata_sha256"]:
        raise ValueError("Metadata differs from the frozen EfficientNet-B0 experiment.")

    records = [
        json.loads(line)
        for line in raw.decode("utf-8-sig").splitlines()
        if line.strip()
    ]

    counts = Counter()
    paths = set()
    groups = {}

    for row in records:
        split = row["split"]
        label = row["quality_label"]

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
        actual = {
            p
            for p in (data_dir / split).rglob("*")
            if p.suffix.lower() in EXTENSIONS
        }
        expected = {
            p
            for p in paths
            if p.relative_to(data_dir).parts[0] == split
        }

        if actual != expected:
            raise ValueError(f"Folder contents differ from metadata: {split}")

        total = sum(counts[(split, label)] for label in CLASS_NAMES)

        if (
            total != baseline["split_counts"][split]
            or any(counts[(split, label)] == 0 for label in CLASS_NAMES)
        ):
            raise ValueError(f"Split counts differ from baseline: {split}")

        print(
            f"{split}: {total} images "
            f"(bad={counts[(split, 'bad')]}, good={counts[(split, 'good')]})"
        )

    print(
        f"Seed: {SEED}; source leakage: 0; "
        "metadata matches frozen EfficientNet-B0"
    )
    return records, digest


def evaluate(best_path, data_dir, records, result_dir):
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    core = build_model(pretrained=False)
    core.load_state_dict(checkpoint["state_dict"])
    core.eval()
    limit_threads()

    transform = image_transform()
    test_rows = sorted(
        (r for r in records if r["split"] == "test"),
        key=lambda r: r["image_path"],
    )

    actual = []
    predicted = []
    probability_good = []

    errors = result_dir / "errors"
    errors.mkdir(exist_ok=True)

    with torch.inference_mode():
        for start in range(0, len(test_rows), 32):
            rows = test_rows[start:start + 32]
            tensors = []

            for row in rows:
                image_path = data_dir / row["image_path"].replace("\\", "/")
                with Image.open(image_path) as image:
                    tensors.append(transform(image.convert("RGB")))

            values = probabilities(core, torch.stack(tensors)).cpu().numpy()
            probability_good.extend(values[:, 1].tolist())
            predicted.extend((values[:, 1] >= 0.5).astype(int).tolist())
            actual.extend(
                CLASS_NAMES.index(row["quality_label"]) for row in rows
            )

            print(
                f"Test evaluation: "
                f"{min(start + 32, len(test_rows))}/{len(test_rows)}"
            )

        images = []

        for row in test_rows[:100]:
            image_path = data_dir / row["image_path"].replace("\\", "/")
            with Image.open(image_path) as image:
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

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(actual, predicted, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        cm,
        display_labels=CLASS_NAMES,
    ).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    fig.tight_layout()
    fig.savefig(result_dir / "confusion_matrix_test.png", dpi=160)
    plt.close(fig)

    with (result_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "true_label",
            "predicted_label",
            "probability_good",
            "defect_type",
        ])

        for row, truth, prediction, probability in zip(
            test_rows,
            actual,
            predicted,
            probability_good,
        ):
            writer.writerow([
                row["image_path"],
                CLASS_NAMES[truth],
                CLASS_NAMES[prediction],
                probability,
                row["defect_type"],
            ])

            if truth != prediction:
                source = data_dir / row["image_path"].replace("\\", "/")
                target_name = (
                    f"true_{CLASS_NAMES[truth]}_"
                    f"pred_{CLASS_NAMES[prediction]}_{source.name}"
                )
                shutil.copy2(source, errors / target_name)

    save_json(result_dir / "classification_report.json", report)

    return (
        report,
        cm.tolist(),
        float(np.mean(durations)),
        sum(a != p for a, p in zip(actual, predicted)),
        sum(p.numel() for p in core.parameters()),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the last 30 parameterized base layers "
            "of a saved EfficientNet-B0."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE_DIR / "data" / "quality_split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results" / "efficientnetb0_finetune_top30",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASE_DIR / "results" / "efficientnetb0_frozen" / "results.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE_DIR / "results" / "efficientnetb0_frozen" / "best.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=BASE_DIR / "reports" / "bonus_finetune_delta.md",
    )
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()

    data_dir = args.data.resolve()
    frozen_path = args.baseline.resolve()
    checkpoint_path = args.checkpoint.resolve()
    frozen = load_frozen_results(frozen_path)

    environment = {
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cpu_threads": 4,
        "inter_op_threads": 1,
    }

    for key, value in environment.items():
        if value != frozen["environment"].get(key):
            raise ValueError(
                f"Environment {key} differs from the frozen run; "
                "use the original virtual environment."
            )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint not found: {checkpoint_path}")

    records, digest = verify_data(data_dir, frozen_path)
    report_path = args.report.resolve()

    if not args.check_data and report_path.exists():
        raise FileExistsError(f"The delta report already exists: {report_path}")

    result_dir = args.output.resolve()

    if result_dir.exists() and not args.check_data:
        raise FileExistsError(
            f"Results already exist; they will not be overwritten: {result_dir}"
        )

    if not args.check_data:
        result_dir.mkdir(parents=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    limit_threads()
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    result = {
        "experiment": "efficientnetb0_finetune_top30",
        "week": 8,
        "task": "5A",
        "model": "EfficientNet-B0",
        "regime": "finetune_top30",
        "input_size": [224, 224],
        "seed": SEED,
        "device": "CPU",
        "epochs_planned": 50,
        "epochs_run": 0,
        "best_epoch": None,
        "train_time_s": 0,
        "latency_ms_per_image": None,
        "model_size_mb": None,
        "params": None,
        "test_samples": sum(r["split"] == "test" for r in records),
        "metrics": {},
        "confusion_matrix": [],
        "history": {},
        "status": "running",
        "metadata_sha256": digest,
        "class_indices": {"bad": 0, "good": 1},
        "split_counts": dict(Counter(r["split"] for r in records)),
        "command": " ".join([
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]),
        "hyperparameters": {
            "batch_size": 32,
            "learning_rate": LEARNING_RATE,
            "patience": 5,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "threshold": 0.5,
            "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
            "head": "Dropout(0.2), Linear(1280, 1); sigmoid at inference",
            "preprocessing": (
                "RGB; full-image bilinear resize to 224x224; pixels to [0,1]; "
                "ImageNet mean/std normalization"
            ),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "checkpoint_selection": "minimum validation loss",
            "additional_augmentation": "none; existing dataset variants retained",
            "base_mode": (
                "eval; last 30 parameterized leaf base modules plus classifier "
                "trainable; BatchNorm running statistics fixed"
            ),
            "layer_count_convention": (
                "last 30 leaf modules owning parameters in features; "
                "classifier trained separately"
            ),
            "head_learning_rate": frozen["hyperparameters"]["learning_rate"],
            "learning_rate_reduction_factor": (
                frozen["hyperparameters"]["learning_rate"] / LEARNING_RATE
            ),
        },
        "latency_protocol": {
            "warmup": 20,
            "timed_predictions": 100,
            "batch_size": 1,
            "includes": "resize, preprocessing, inference, decision",
            "excludes": "image decoding, disk IO, model loading, network",
        },
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_threads": 4,
            "inter_op_threads": 1,
            "precision": "float32",
        },
    }

    result.update({
        "parent_experiment": frozen["experiment"],
        "parent_results_sha256": hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest(),
        "parent_checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "parent_checkpoint": str(checkpoint_path),
        "frozen_best_epoch": frozen["best_epoch"],
        "delta_report": str(report_path),
    })

    if not args.check_data:
        save_json(result_dir / "results.json", result)

    started = None

    try:
        generator = torch.Generator().manual_seed(SEED)

        train_loader = DataLoader(
            QualityDataset(data_dir, records, "train"),
            batch_size=32,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        validation_loader = DataLoader(
            QualityDataset(data_dir, records, "validation"),
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        model, checkpoint = load_frozen_checkpoint(checkpoint_path, frozen)
        selected = unfreeze_top30(model)
        result["unfrozen_base_layer_names"] = selected
        result["unfrozen_base_layer_count"] = len(selected)

        if args.check_data:
            print(f"Frozen checkpoint loaded: epoch {checkpoint['epoch']}")
            print(
                f"Unfrozen base layers: {len(selected)}; "
                "classification head also trainable"
            )
            print(
                "Trainable parameters: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            )
            print(
                "Data and checkpoint check completed. Training was not started."
            )
            return

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE,
        )

        result["params"] = sum(p.numel() for p in model.parameters())
        result["trainable_params"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(
            f"Model: EfficientNet-B0; total parameters: {result['params']:,}; "
            f"trainable: {result['trainable_params']:,}"
        )
        print(f"Starting from frozen checkpoint epoch {checkpoint['epoch']}")
        print(
            f"Unfrozen base layers: {len(selected)}; "
            f"Adam learning rate: {LEARNING_RATE}"
        )
        print(
            "Base running statistics fixed; "
            "selected weights and classifier receive gradients."
        )

        save_json(result_dir / "results.json", result)

        history = {
            key: []
            for key in ["loss", "accuracy", "val_loss", "val_accuracy"]
        }
        best_loss = float("inf")
        patience_count = 0
        best_path = result_dir / "best.pt"
        started = time.perf_counter()

        with (result_dir / "training_log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", *history.keys(), "epoch_time_s"])

            for epoch in range(1, 51):
                epoch_start = time.perf_counter()

                train_loss, train_accuracy = run_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                )
                validation_loss, validation_accuracy = run_epoch(
                    model,
                    validation_loader,
                    criterion,
                )

                if (
                    not np.isfinite(train_loss)
                    or not np.isfinite(validation_loss)
                ):
                    raise ValueError("Non-finite training or validation loss.")

                for key, value in zip(
                    history,
                    [
                        train_loss,
                        train_accuracy,
                        validation_loss,
                        validation_accuracy,
                    ],
                ):
                    history[key].append(float(value))

                elapsed = time.perf_counter() - epoch_start

                writer.writerow([
                    epoch,
                    *[history[key][-1] for key in history],
                    elapsed,
                ])
                file.flush()

                print(
                    f"Epoch {epoch}/50 - {elapsed:.1f}s "
                    f"- loss: {train_loss:.4f} "
                    f"- accuracy: {train_accuracy:.4f} "
                    f"- val_loss: {validation_loss:.4f} "
                    f"- val_accuracy: {validation_accuracy:.4f}",
                    flush=True,
                )

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    patience_count = 0
                    result["best_epoch"] = epoch

                    torch.save({
                        "state_dict": model.state_dict(),
                        "model": "efficientnet_b0",
                        "head": "Dropout(0.2), Linear(1280,1)",
                        "class_names": CLASS_NAMES,
                        "seed": SEED,
                        "epoch": epoch,
                        "val_loss": validation_loss,
                        "input_size": [224, 224],
                    }, best_path)
                else:
                    patience_count += 1

                result.update({
                    "epochs_run": epoch,
                    "history": history,
                    "train_time_s": time.perf_counter() - started,
                })
                save_json(result_dir / "results.json", result)

                if patience_count >= 5:
                    print(
                        f"Early stopping. Best epoch: {result['best_epoch']}"
                    )
                    break

        result["train_time_s"] = time.perf_counter() - started
        save_curves(history, result_dir)

        report, cm, latency, mistakes, params = evaluate(
            best_path,
            data_dir,
            records,
            result_dir,
        )

        result.update({
            "metrics": {
                "accuracy": report["accuracy"],
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
                "per_class": {
                    name: report[name] for name in CLASS_NAMES
                },
            },
            "confusion_matrix": cm,
            "latency_ms_per_image": latency,
            "misclassified_samples": mistakes,
            "params": params,
            "model_size_mb": best_path.stat().st_size / 1_000_000,
            "model_file": "best.pt",
            "size_note": (
                "Float32 state_dict with architecture metadata; "
                "no optimizer state. Reconstruct using "
                "build_model(pretrained=False)."
            ),
            "status": "completed",
        })

        result["train_val_at_best_epoch"] = logged_gap(result)
        result["best_val_loss"] = best_loss
        save_json(result_dir / "results.json", result)

        try:
            write_delta_memo(frozen, result, report_path)
        except Exception as report_error:
            result["report_status"] = "failed"
            result["report_note"] = (
                f"{type(report_error).__name__}: {report_error}"
            )
            save_json(result_dir / "results.json", result)
            warnings.warn(
                "Training results are saved, but the delta memo failed: "
                f"{report_error}"
            )
        else:
            result["report_status"] = "completed"
            save_json(result_dir / "results.json", result)
            print(f"Delta memo saved to: {report_path}")

        print("Training and evaluation completed.")
        print(f"Test accuracy: {report['accuracy']:.4f}")
        print(f"Test macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Bad recall: {report['bad']['recall']:.4f}")
        print(f"Latency: {latency:.2f} ms/image")
        print(f"Model size: {result['model_size_mb']:.2f} MB")
        print(f"Misclassified test images: {mistakes}")
        print(f"Results saved to: {result_dir}")

    except (Exception, KeyboardInterrupt) as error:
        if args.check_data:
            raise

        result["train_time_s"] = (
            time.perf_counter() - started if started is not None else 0
        )

        if result.get("history"):
            save_curves(result["history"], result_dir)

        result["status"] = "failed"
        result["note"] = f"{type(error).__name__}: {error}"
        save_json(result_dir / "results.json", result)

        (result_dir / "error.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()