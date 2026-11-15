import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['OMP_NUM_THREADS'] = '4'

os.environ['MKL_NUM_THREADS'] = '4'

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

matplotlib.use('Agg')

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

BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'scripts' else SCRIPT_DIR

SEED = 24500007

IMAGE_SIZE = 224

CLASS_NAMES = ['bad', 'good']

LEARNING_RATE = 1e-05

FINETUNE_LAYERS = 30

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def image_transform(size=IMAGE_SIZE):
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class QualityDataset(Dataset):

    def __init__(self, data_dir, records, split):
        self.data_dir = data_dir
        self.rows = sorted((r for r in records if r['split'] == split), key=lambda r: r['image_path'])
        self.transform = image_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_dir / row['image_path'].replace('\\', '/')
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return (tensor, float(CLASS_NAMES.index(row['quality_label'])))

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
    layers = [('features.' + name, module) for name, module in model.features.named_modules() if not list(module.children()) and list(module.parameters(recurse=False))]
    if len(layers) < FINETUNE_LAYERS:
        raise ValueError('The base has fewer than 30 parameterized leaf modules.')
    selected = layers[-FINETUNE_LAYERS:]
    for name, module in selected:
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return [name for name, module in selected]

def load_frozen_checkpoint(checkpoint_path, frozen):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    expected_fields = [('model', 'efficientnet_b0'), ('class_names', CLASS_NAMES), ('seed', SEED), ('input_size', [224, 224]), ('epoch', frozen['best_epoch'])]
    for field, expected in expected_fields:
        if checkpoint.get(field) != expected:
            raise ValueError(f'The frozen checkpoint has an unexpected {field}.')
    expected_val_loss = frozen['history']['val_loss'][frozen['best_epoch'] - 1]
    if not np.isclose(checkpoint['val_loss'], expected_val_loss):
        raise ValueError('The checkpoint validation loss differs from frozen results.')
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if sum((p.numel() for p in model.parameters())) != frozen['params']:
        raise ValueError('The parameter count differs from the frozen model.')
    return (model, checkpoint)

def load_frozen_results(path):
    frozen = json.loads(path.read_text(encoding='utf-8-sig'))
    if frozen.get('status') != 'completed' or frozen.get('regime') != 'frozen_base':
        raise ValueError('A completed frozen-base results.json is required.')
    if frozen.get('model') != 'EfficientNet-B0' or frozen.get('seed') != SEED:
        raise ValueError('The frozen model name or seed does not match.')
    if frozen.get('input_size') != [224, 224] or frozen.get('class_indices') != {'bad': 0, 'good': 1}:
        raise ValueError('The frozen input size or class order does not match.')
    expected = {'batch_size': 32, 'patience': 5, 'threshold': 0.5, 'optimizer': 'Adam'}
    if frozen.get('epochs_planned') != 50 or any((frozen['hyperparameters'].get(k) != v for k, v in expected.items())):
        raise ValueError('The frozen protocol differs from the required fine-tuning protocol.')
    return frozen

def logged_gap(result):
    index = result['best_epoch'] - 1
    history = result['history']
    return {'train_accuracy': history['accuracy'][index], 'val_accuracy': history['val_accuracy'][index], 'accuracy_gap': history['accuracy'][index] - history['val_accuracy'][index], 'train_loss': history['loss'][index], 'val_loss': history['val_loss'][index]}

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
    return (total_loss / count, correct / count)

def save_curves(history, result_dir):
    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ['loss', 'accuracy']):
        ax.plot(epochs, history[key], label='Train')
        ax.plot(epochs, history['val_' + key], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key.capitalize())
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / 'curves.png', dpi=160)
    plt.close(fig)

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def limit_threads(trainer=None):
    torch.set_num_threads(4)

def verify_data(data_dir, baseline_path):
    metadata_path = data_dir / 'metadata.jsonl'
    raw = metadata_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if digest != baseline['metadata_sha256']:
        raise ValueError('Metadata differs from the frozen EfficientNet-B0 experiment.')
    records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    counts = Counter()
    paths = set()
    groups = {}
    for row in records:
        split = row['split']
        label = row['quality_label']
        if split not in {'train', 'validation', 'test'} or label not in CLASS_NAMES:
            raise ValueError('Invalid split or class in metadata.')
        if int(row['seed']) != SEED:
            raise ValueError('Unexpected seed in metadata.')
        relative = Path(row['image_path'].replace('\\', '/'))
        if relative.parts[:2] != (split, label) or '..' in relative.parts:
            raise ValueError(f'Invalid split image path: {relative}')
        path = data_dir / relative
        if not path.is_file() or path in paths:
            raise ValueError(f'Missing or duplicate image: {path}')
        with Image.open(path) as image:
            if image.size != (640, 640):
                raise ValueError(f'Unexpected image size: {path}')
            image.verify()
        key = (row['class_name'], row['source_id'])
        if key in groups and groups[key] != split:
            raise ValueError('Source leakage detected.')
        groups[key] = split
        paths.add(path)
        counts[split, label] += 1
    for split in ('train', 'validation', 'test'):
        actual = {p for p in (data_dir / split).rglob('*') if p.suffix.lower() in EXTENSIONS}
        expected = {p for p in paths if p.relative_to(data_dir).parts[0] == split}
        if actual != expected:
            raise ValueError(f'Folder contents differ from metadata: {split}')
        total = sum((counts[split, label] for label in CLASS_NAMES))
        if total != baseline['split_counts'][split] or any((counts[split, label] == 0 for label in CLASS_NAMES)):
            raise ValueError(f'Split counts differ from baseline: {split}')
        print(f"{split}: {total} images (bad={counts[split, 'bad']}, good={counts[split, 'good']})")
    print(f'Seed: {SEED}; source leakage: 0; metadata matches frozen EfficientNet-B0')
    return (records, digest)

def evaluate(best_path, data_dir, records, result_dir):
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
    core = build_model(pretrained=False)
    core.load_state_dict(checkpoint['state_dict'])
    core.eval()
    limit_threads()
    transform = image_transform()
    test_rows = sorted((r for r in records if r['split'] == 'test'), key=lambda r: r['image_path'])
    actual = []
    predicted = []
    probability_good = []
    errors = result_dir / 'errors'
    errors.mkdir(exist_ok=True)
    with torch.inference_mode():
        for start in range(0, len(test_rows), 32):
            rows = test_rows[start:start + 32]
            tensors = []
            for row in rows:
                image_path = data_dir / row['image_path'].replace('\\', '/')
                with Image.open(image_path) as image:
                    tensors.append(transform(image.convert('RGB')))
            values = probabilities(core, torch.stack(tensors)).cpu().numpy()
            probability_good.extend(values[:, 1].tolist())
            predicted.extend((values[:, 1] >= 0.5).astype(int).tolist())
            actual.extend((CLASS_NAMES.index(row['quality_label']) for row in rows))
            print(f'Test evaluation: {min(start + 32, len(test_rows))}/{len(test_rows)}')
        images = []
        for row in test_rows[:100]:
            image_path = data_dir / row['image_path'].replace('\\', '/')
            with Image.open(image_path) as image:
                images.append(image.convert('RGB').copy())
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
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, cmap='Blues')
    fig.tight_layout()
    fig.savefig(result_dir / 'confusion_matrix_test.png', dpi=160)
    plt.close(fig)
    with (result_dir / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'true_label', 'predicted_label', 'probability_good', 'defect_type'])
        for row, truth, prediction, probability in zip(test_rows, actual, predicted, probability_good):
            writer.writerow([row['image_path'], CLASS_NAMES[truth], CLASS_NAMES[prediction], probability, row['defect_type']])
            if truth != prediction:
                source = data_dir / row['image_path'].replace('\\', '/')
                target_name = f'true_{CLASS_NAMES[truth]}_pred_{CLASS_NAMES[prediction]}_{source.name}'
                shutil.copy2(source, errors / target_name)
    save_json(result_dir / 'classification_report.json', report)
    return (report, cm.tolist(), float(np.mean(durations)), sum((a != p for a, p in zip(actual, predicted))), sum((p.numel() for p in core.parameters())))

from PIL import ImageDraw, ImageFilter

AUGMENTATION = {
    "name": "strong_white_ellipse_glare", "split": "train",
    "eligible_quality_label": "bad", "eligible_defect_type": "glare",
    "probability": 0.5, "ellipses": [1, 2], "alpha_8bit": [150, 230],
    "width_fraction": [0.20, 0.50], "height_fraction": [0.08, 0.22],
    "center_x_fraction": [0.15, 0.85], "center_y_fraction": [0.20, 0.75],
    "edge_blur_radius_fraction": 0.012, "color_rgb": [255, 255, 255],
    "applied_before_resize": True, "labels_changed": False,
    "rng": "local Random(SHA256(seed:epoch:image_path)); independent of model and shuffle RNG",
    "on_disk_images_modified": False, "validation_augmentation": "none", "test_augmentation": "none",
}


def add_white_glare(image, rng):
    width, height = image.size
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(rng.randint(*AUGMENTATION["ellipses"])):
        ew = rng.uniform(*AUGMENTATION["width_fraction"]) * width
        eh = rng.uniform(*AUGMENTATION["height_fraction"]) * height
        cx = rng.uniform(*AUGMENTATION["center_x_fraction"]) * width
        cy = rng.uniform(*AUGMENTATION["center_y_fraction"]) * height
        draw.ellipse((cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2),
                     fill=rng.randint(*AUGMENTATION["alpha_8bit"]))
    mask = mask.filter(ImageFilter.GaussianBlur(width * AUGMENTATION["edge_blur_radius_fraction"]))
    return Image.composite(Image.new("RGB", image.size, (255, 255, 255)), image.convert("RGB"), mask)


def augment_training_image(image, row, epoch):
    if row["split"] != "train" or row["quality_label"] != "bad" or row["defect_type"] != "glare":
        return image
    key = f"{SEED}:{epoch}:{row['image_path']}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    if rng.random() >= AUGMENTATION["probability"]:
        return image
    return add_white_glare(image, rng)


class GlareDataset(QualityDataset):
    def __init__(self, data_dir, records, split):
        super().__init__(data_dir, records, split)
        self.epoch = 0

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_dir / row["image_path"].replace("\\", "/")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image = augment_training_image(image, row, self.epoch)
        return self.transform(image), float(CLASS_NAMES.index(row["quality_label"]))


def evaluation_manifests(data_dir, records):
    result = {}
    for split in ("validation", "test"):
        digest = hashlib.sha256()
        for row in sorted((r for r in records if r["split"] == split), key=lambda r: r["image_path"]):
            relative = row["image_path"].replace("\\", "/")
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256((data_dir / relative).read_bytes()).digest())
        result[split] = digest.hexdigest()
    return result


def read_test_predictions(path, records):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    expected = {r["image_path"].replace("\\", "/"): r for r in records if r["split"] == "test"}
    actual = [r["image"].replace("\\", "/") for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("Saved predictions do not cover exactly the original test split.")
    for row, name in zip(rows, actual):
        metadata = expected[name]
        probability = float(row["probability_good"])
        if (not np.isfinite(probability) or not 0 <= probability <= 1
                or row["true_label"] != metadata["quality_label"]
                or row["defect_type"] != metadata["defect_type"]
                or row["predicted_label"] != ("good" if probability >= 0.5 else "bad")):
            raise ValueError(f"Invalid saved prediction or non-default threshold: {name}")
        row["image"] = name
    return sorted(rows, key=lambda r: r["image"])


def report_from_predictions(rows):
    return classification_report([r["true_label"] for r in rows], [r["predicted_label"] for r in rows],
                                 labels=CLASS_NAMES, output_dict=True, zero_division=0)


def verify_reference(reference, frozen, frozen_path, checkpoint_path, rows):
    required = {"status": "completed", "model": "EfficientNet-B0", "regime": "finetune_top30", "seed": SEED,
                "input_size": [224, 224], "class_indices": {"bad": 0, "good": 1}, "epochs_planned": 50}
    for key, expected in required.items():
        if reference.get(key) != expected:
            raise ValueError(f"The Task 5A comparison experiment has an unexpected {key}.")
    for key in ["metadata_sha256", "split_counts", "environment"]:
        if reference[key] != frozen[key]:
            raise ValueError(f"The reference and frozen starting run differ in {key}.")
    protocol = {"learning_rate": 1e-5, "batch_size": 32, "patience": 5, "threshold": 0.5, "optimizer": "Adam"}
    if any(reference["hyperparameters"].get(k) != v for k, v in protocol.items()):
        raise ValueError("Task 5A used a different training protocol.")
    if reference["hyperparameters"].get("additional_augmentation") != "none; existing dataset variants retained":
        raise ValueError("The reference must have no additional online augmentation.")
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != reference["parent_results_sha256"]:
        raise ValueError("The frozen starting results differ from the actual parent of Task 5A.")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != reference["parent_checkpoint_sha256"]:
        raise ValueError("Use the same frozen starting checkpoint as Task 5A.")
    report = report_from_predictions(rows)
    actual = confusion_matrix([r["true_label"] for r in rows], [r["predicted_label"] for r in rows], labels=CLASS_NAMES).tolist()
    if actual != reference["confusion_matrix"]:
        raise ValueError("Reference test predictions and confusion matrix differ.")
    for key, value in [("accuracy", report["accuracy"]), ("f1_macro", report["macro avg"]["f1-score"]),
                       ("precision_macro", report["macro avg"]["precision"]), ("recall_macro", report["macro avg"]["recall"])]:
        if not np.isclose(value, reference["metrics"][key], rtol=0, atol=1e-10):
            raise ValueError("Reference prediction metrics differ from Task 5A results.")
    for label in CLASS_NAMES:
        for key in ["precision", "recall", "f1-score", "support"]:
            if not np.isclose(report[label][key], reference["metrics"]["per_class"][label][key], rtol=0, atol=1e-10):
                raise ValueError("Reference per-class metrics do not match.")
    return report


def defect_metrics(rows):
    result = {}
    defects = sorted({r["defect_type"] for r in rows if r["true_label"] == "bad"})
    for defect in defects:
        subset = [r for r in rows if r["true_label"] == "good" or r["defect_type"] == defect]
        report = report_from_predictions(subset)
        result[defect] = {"bad_support": int(report["bad"]["support"]), "good_support": int(report["good"]["support"]),
                          "precision_bad": report["bad"]["precision"], "recall_bad": report["bad"]["recall"],
                          "f1_bad": report["bad"]["f1-score"], "f1_good_in_subset": report["good"]["f1-score"],
                          "false_accepts": sum(r["true_label"] == "bad" and r["predicted_label"] == "good" for r in subset),
                          "false_rejects": sum(r["true_label"] == "good" and r["predicted_label"] == "bad" for r in subset)}
    return result


def write_augmentation_memo(reference, result, before_rows, after_rows, report_path, result_dir):
    if [r["image"] for r in before_rows] != [r["image"] for r in after_rows]:
        raise ValueError("Before/after predictions use different test images.")
    before, after = defect_metrics(before_rows), defect_metrics(after_rows)
    if "glare" not in before:
        raise ValueError("No glare subgroup is present in the comparison.")
    result["per_defect_binary_comparison"] = {"before": before, "after": after}
    old, new = reference["metrics"], result["metrics"]
    table = []
    for label in CLASS_NAMES:
        a, b = old["per_class"][label], new["per_class"][label]
        table.append({"class": label, "support": int(a["support"]), "precision_before": a["precision"],
                      "precision_after": b["precision"], "recall_before": a["recall"], "recall_after": b["recall"],
                      "f1_before": a["f1-score"], "f1_after": b["f1-score"], "f1_delta": b["f1-score"] - a["f1-score"]})
    with (result_dir / "per_class_delta.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    defect_table = [{"defect": d, "bad_support": before[d]["bad_support"], "good_support": before[d]["good_support"],
                     "f1_bad_before": before[d]["f1_bad"], "f1_bad_after": after[d]["f1_bad"],
                     "f1_delta": after[d]["f1_bad"] - before[d]["f1_bad"], "recall_bad_before": before[d]["recall_bad"],
                     "recall_bad_after": after[d]["recall_bad"], "false_accepts_before": before[d]["false_accepts"],
                     "false_accepts_after": after[d]["false_accepts"]} for d in before]
    with (result_dir / "per_defect_delta.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(defect_table[0]))
        writer.writeheader(); writer.writerows(defect_table)
    lines = ["# Error-Driven Glare Augmentation", "", "Student: Ekaterina Shigarova, IIKb-24. Task 5C. Seed: 24500007.", "",
             "## Evidence and protocol", "",
             "Task 4 identified glare as the dominant defect: all five false-accept events across its four models involved glare. "
             "The Task 5A reference has one false accept, also glare. These are measured results from this project; "
             "the handout's capstone F1 = 0.92 is not a result of this binary classifier.", "",
             "I repeated Task 5A from the exact same frozen checkpoint, with a fresh Adam optimizer and the same seed, "
             "last 30 parameterized base layers plus head, learning rate 1e-5, batch 32, maximum 50 epochs, patience 5, "
             "224 x 224 preprocessing and minimum-validation-loss checkpoint selection. The base remains in eval mode "
             "with fixed BatchNorm running statistics; classifier dropout is active in training. "
             "Both runs use the original P(good) >= 0.5 decision. The Task 5B threshold is not applied, so it does not confound this comparison.", "",
             "Only existing train/bad/glare images receive extra white ellipses, with probability 0.5 per image per epoch. "
             "Each overlay has 1-2 ellipses with alpha 150-230/255 and softly blurred edges. Positions and sizes vary as recorded in results.json. "
             "A local deterministic RNG depends on seed, epoch and image path, preserving model and shuffle RNG state. "
             "Good labels and other defect subtypes remain unchanged. Augmentation occurs in memory before resizing; "
             "no source images are overwritten, no extra validation/test images are generated, and their byte manifests are checked before and after training.", "",
             "## Per-class comparison on the unchanged test split", "",
             "| Class | Support | Precision before | Precision after | Recall before | Recall after | F1 before | F1 after | F1 delta |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['class']} | {row['support']} | {row['precision_before']:.6f} | {row['precision_after']:.6f} | "
                     f"{row['recall_before']:.6f} | {row['recall_after']:.6f} | {row['f1_before']:.6f} | {row['f1_after']:.6f} | {row['f1_delta']:+.6f} |")
    lines += ["", "## Defect-specific binary comparisons", "",
              "Glare, blur, skew, occlusion and exposure are metadata subgroups of bad, not separate model outputs. "
              "Each row below evaluates bad-class F1 on that defect's bad images plus ALL good test images. "
              "The shared good images supply false positives for precision. These overlapping subsets are not independent and must not be summed. "
              "Good means suitable in this dataset. This is not multiclass defect-identification F1.", "",
              "| Defect vs good | Bad support | Good support | Bad F1 before | Bad F1 after | F1 delta | Bad recall before | Bad recall after | Missed bad before | Missed bad after |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in defect_table:
        lines.append(f"| {row['defect']} | {row['bad_support']} | {row['good_support']} | {row['f1_bad_before']:.6f} | "
                     f"{row['f1_bad_after']:.6f} | {row['f1_delta']:+.6f} | {row['recall_bad_before']:.6f} | "
                     f"{row['recall_bad_after']:.6f} | {row['false_accepts_before']} | {row['false_accepts_after']} |")
    other_worse = [f"{d} ({after[d]['f1_bad'] - before[d]['f1_bad']:+.6f})" for d in before if d != "glare" and after[d]["f1_bad"] < before[d]["f1_bad"] - 1e-12]
    glare_delta = after["glare"]["f1_bad"] - before["glare"]["f1_bad"]
    verdict = "improved glare-subset F1" if glare_delta > 1e-12 else "reduced glare-subset F1" if glare_delta < -1e-12 else "did not change glare-subset F1"
    useful = glare_delta > 1e-12 and not other_worse and new["per_class"]["good"]["f1-score"] >= old["per_class"]["good"]["f1-score"] - 1e-12
    lines += ["", "## Five-line conclusion", "",
              f"1. Extra train-only augmentation {verdict}: {before['glare']['f1_bad']:.6f} to {after['glare']['f1_bad']:.6f} "
              f"({glare_delta:+.6f}); missed glare images changed from {before['glare']['false_accepts']} to {after['glare']['false_accepts']}.",
              f"2. Good-class F1 changed from {old['per_class']['good']['f1-score']:.6f} to {new['per_class']['good']['f1-score']:.6f}; "
              f"unnecessary good-image rejections changed from {reference['confusion_matrix'][1][0]} to {result['confusion_matrix'][1][0]}.",
              "3. Other defect-subset F1 decreases: " + (", ".join(other_worse) if other_worse else "none observed") + ".",
              f"4. Overall test macro F1 changed from {old['f1_macro']:.6f} to {new['f1_macro']:.6f}; accuracy changed from "
              f"{old['accuracy']:.6f} to {new['accuracy']:.6f}. Training took {result['train_time_s'] / 60:.2f} minutes over {result['epochs_run']} epochs.",
              "5. " + ("The augmented model is a candidate for further evaluation." if useful else "This run does not establish an overall benefit without a trade-off; retain the previous prototype pending further evaluation.") +
              " Validation and test remain unaugmented because changing evaluation images would change the benchmark. "
              "The augmentation choice was informed by earlier test errors, so this reused test set is diagnostic, not a fresh independent holdout. Real-photo evaluation is still needed.", "",
              "Task 4 evidence: https://github.com/EkaterinaShigarova/document-quality-classification/blob/main/reports/error_analysis.md", "",
              "Overlay API reference: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html", ""]
    with report_path.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description='Task 5C: repeat Task 5A with additional train-only glare augmentation.')
    parser.add_argument('--data', type=Path, default=BASE_DIR / 'data' / 'quality_split')
    parser.add_argument('--output', type=Path, default=BASE_DIR / 'results' / 'efficientnetb0_glare_aug')
    parser.add_argument('--baseline', type=Path, default=BASE_DIR / 'results' / 'efficientnetb0_frozen' / 'results.json')
    parser.add_argument('--checkpoint', type=Path, default=BASE_DIR / 'results' / 'efficientnetb0_frozen' / 'best.pt')
    parser.add_argument('--report', type=Path, default=BASE_DIR / 'reports' / 'bonus_augmentation.md')
    parser.add_argument('--reference', type=Path, default=BASE_DIR / 'results' / 'efficientnetb0_finetune_top30' / 'results.json')
    parser.add_argument('--reference-predictions', type=Path, default=BASE_DIR / 'results' / 'efficientnetb0_finetune_top30' / 'test_predictions.csv')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    data_dir = args.data.resolve()
    frozen_path = args.baseline.resolve()
    checkpoint_path = args.checkpoint.resolve()
    frozen = load_frozen_results(frozen_path)
    environment = {'pytorch': torch.__version__, 'torchvision': torchvision.__version__, 'cpu_threads': 4, 'inter_op_threads': 1}
    for key, value in environment.items():
        if value != frozen['environment'].get(key):
            raise ValueError(f'Environment {key} differs from the frozen run; use the original virtual environment.')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Frozen checkpoint not found: {checkpoint_path}')
    records, digest = verify_data(data_dir, frozen_path)
    reference_path = args.reference.resolve()
    reference = json.loads(reference_path.read_text(encoding='utf-8-sig'))
    reference_rows = read_test_predictions(args.reference_predictions, records)
    verify_reference(reference, frozen, frozen_path, checkpoint_path, reference_rows)
    evaluation_before = evaluation_manifests(data_dir, records)
    eligible = sum(r['split'] == 'train' and r['quality_label'] == 'bad' and r['defect_type'] == 'glare' for r in records)
    if not eligible:
        raise ValueError('No training glare images are available.')
    print(f'Task 5C: {eligible} training glare images eligible; augmentation probability = 0.5; alpha = 150-230')
    print('Comparison: repeat Task 5A from identical frozen weights at threshold 0.5; Task 5B threshold is not used.')
    report_path = args.report.resolve()
    if not args.check_data and report_path.exists():
        raise FileExistsError(f'The augmentation report already exists: {report_path}')
    result_dir = args.output.resolve()
    if result_dir.exists() and (not args.check_data):
        raise FileExistsError(f'Results already exist; they will not be overwritten: {result_dir}')
    if not args.check_data:
        result_dir.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    limit_threads()
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {'experiment': 'efficientnetb0_glare_aug', 'week': 8, 'task': '5C', 'model': 'EfficientNet-B0', 'regime': 'finetune_top30', 'input_size': [224, 224], 'seed': SEED, 'device': 'CPU', 'epochs_planned': 50, 'epochs_run': 0, 'best_epoch': None, 'train_time_s': 0, 'latency_ms_per_image': None, 'model_size_mb': None, 'params': None, 'test_samples': sum((r['split'] == 'test' for r in records)), 'metrics': {}, 'confusion_matrix': [], 'history': {}, 'status': 'running', 'metadata_sha256': digest, 'class_indices': {'bad': 0, 'good': 1}, 'split_counts': dict(Counter((r['split'] for r in records))), 'command': ' '.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]), 'hyperparameters': {'batch_size': 32, 'learning_rate': LEARNING_RATE, 'patience': 5, 'optimizer': 'Adam', 'loss': 'BCEWithLogitsLoss', 'threshold': 0.5, 'pretrained_weights': 'EfficientNet_B0_Weights.IMAGENET1K_V1', 'head': 'Dropout(0.2), Linear(1280, 1); sigmoid at inference', 'preprocessing': 'RGB; full-image bilinear resize to 224x224; pixels to [0,1]; ImageNet mean/std normalization', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'checkpoint_selection': 'minimum validation loss', 'additional_augmentation': AUGMENTATION, 'base_mode': 'eval; last 30 parameterized leaf base modules plus classifier trainable; BatchNorm running statistics fixed', 'layer_count_convention': 'last 30 leaf modules owning parameters in features; classifier trained separately', 'head_learning_rate': frozen['hyperparameters']['learning_rate'], 'learning_rate_reduction_factor': frozen['hyperparameters']['learning_rate'] / LEARNING_RATE}, 'latency_protocol': {'warmup': 20, 'timed_predictions': 100, 'batch_size': 1, 'includes': 'resize, preprocessing, inference, decision', 'excludes': 'image decoding, disk IO, model loading, network'}, 'environment': {'python': sys.version, 'pytorch': torch.__version__, 'torchvision': torchvision.__version__, 'platform': platform.platform(), 'processor': platform.processor(), 'cpu_threads': 4, 'inter_op_threads': 1, 'precision': 'float32'}}
    result.update({'parent_experiment': frozen['experiment'], 'parent_results_sha256': hashlib.sha256(frozen_path.read_bytes()).hexdigest(), 'parent_checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), 'parent_checkpoint': str(checkpoint_path), 'frozen_best_epoch': frozen['best_epoch'], 'delta_report': str(report_path)})
    result.update({'augmentation': AUGMENTATION, 'reference_experiment': reference['experiment'], 'reference_results_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(), 'reference_predictions_sha256': hashlib.sha256(args.reference_predictions.read_bytes()).hexdigest(), 'reference_metrics': reference['metrics'], 'reference_confusion_matrix': reference['confusion_matrix'], 'training_initialization': 'same frozen checkpoint as Task 5A, not continuation of its fine-tuned checkpoint', 'evaluation_manifests_before': evaluation_before, 'train_glare_eligible': eligible})
    if not args.check_data:
        save_json(result_dir / 'results.json', result)
    started = None
    try:
        generator = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(GlareDataset(data_dir, records, 'train'), batch_size=32, shuffle=True, num_workers=0, generator=generator)
        validation_loader = DataLoader(QualityDataset(data_dir, records, 'validation'), batch_size=32, shuffle=False, num_workers=0)
        model, checkpoint = load_frozen_checkpoint(checkpoint_path, frozen)
        selected = unfreeze_top30(model)
        if selected != reference['unfrozen_base_layer_names']:
            raise ValueError('The trainable base layers differ from Task 5A.')
        result['unfrozen_base_layer_names'] = selected
        result['unfrozen_base_layer_count'] = len(selected)
        if args.check_data:
            print(f"Frozen checkpoint loaded: epoch {checkpoint['epoch']}")
            print(f'Unfrozen base layers: {len(selected)}; classification head also trainable')
            print(f'Trainable parameters: {sum((p.numel() for p in model.parameters() if p.requires_grad)):,}')
            print('Data and checkpoint check completed. Training was not started.')
            return
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
        result['params'] = sum((p.numel() for p in model.parameters()))
        result['trainable_params'] = sum((p.numel() for p in model.parameters() if p.requires_grad))
        print(f"Model: EfficientNet-B0; total parameters: {result['params']:,}; trainable: {result['trainable_params']:,}")
        print(f"Starting from frozen checkpoint epoch {checkpoint['epoch']}")
        print(f'Unfrozen base layers: {len(selected)}; Adam learning rate: {LEARNING_RATE}')
        print('Base running statistics fixed; selected weights and classifier receive gradients.')
        save_json(result_dir / 'results.json', result)
        history = {key: [] for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']}
        best_loss = float('inf')
        patience_count = 0
        best_path = result_dir / 'best.pt'
        started = time.perf_counter()
        with (result_dir / 'training_log.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', *history.keys(), 'epoch_time_s'])
            for epoch in range(1, 51):
                train_loader.dataset.epoch = epoch
                epoch_start = time.perf_counter()
                train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
                validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion)
                if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                    raise ValueError('Non-finite training or validation loss.')
                for key, value in zip(history, [train_loss, train_accuracy, validation_loss, validation_accuracy]):
                    history[key].append(float(value))
                elapsed = time.perf_counter() - epoch_start
                writer.writerow([epoch, *[history[key][-1] for key in history], elapsed])
                file.flush()
                print(f'Epoch {epoch}/50 - {elapsed:.1f}s - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {validation_loss:.4f} - val_accuracy: {validation_accuracy:.4f}', flush=True)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    patience_count = 0
                    result['best_epoch'] = epoch
                    torch.save({'state_dict': model.state_dict(), 'model': 'efficientnet_b0', 'head': 'Dropout(0.2), Linear(1280,1)', 'class_names': CLASS_NAMES, 'seed': SEED, 'epoch': epoch, 'val_loss': validation_loss, 'input_size': [224, 224]}, best_path)
                else:
                    patience_count += 1
                result.update({'epochs_run': epoch, 'history': history, 'train_time_s': time.perf_counter() - started})
                save_json(result_dir / 'results.json', result)
                if patience_count >= 5:
                    print(f"Early stopping. Best epoch: {result['best_epoch']}")
                    break
        result['train_time_s'] = time.perf_counter() - started
        save_curves(history, result_dir)
        result['evaluation_manifests_after'] = evaluation_manifests(data_dir, records)
        if result['evaluation_manifests_after'] != evaluation_before:
            raise ValueError('Validation or test image bytes changed during training.')
        report, cm, latency, mistakes, params = evaluate(best_path, data_dir, records, result_dir)
        result.update({'metrics': {'accuracy': report['accuracy'], 'precision_macro': report['macro avg']['precision'], 'recall_macro': report['macro avg']['recall'], 'f1_macro': report['macro avg']['f1-score'], 'per_class': {name: report[name] for name in CLASS_NAMES}}, 'confusion_matrix': cm, 'latency_ms_per_image': latency, 'misclassified_samples': mistakes, 'params': params, 'model_size_mb': best_path.stat().st_size / 1000000, 'model_file': 'best.pt', 'size_note': 'Float32 state_dict with architecture metadata; no optimizer state. Reconstruct using build_model(pretrained=False).', 'status': 'completed'})
        result['train_val_at_best_epoch'] = logged_gap(result)
        result['best_val_loss'] = best_loss
        save_json(result_dir / 'results.json', result)
        try:
            after_rows = read_test_predictions(result_dir / 'test_predictions.csv', records)
            write_augmentation_memo(reference, result, reference_rows, after_rows, report_path, result_dir)
        except Exception as report_error:
            result['report_status'] = 'failed'
            result['report_note'] = f'{type(report_error).__name__}: {report_error}'
            save_json(result_dir / 'results.json', result)
            warnings.warn(f'Training results are saved, but the augmentation memo failed: {report_error}')
        else:
            result['report_status'] = 'completed'
            save_json(result_dir / 'results.json', result)
            print(f'Augmentation memo saved to: {report_path}')
        print('Training and evaluation completed.')
        print(f"Test accuracy: {report['accuracy']:.4f}")
        print(f"Test macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Bad recall: {report['bad']['recall']:.4f}")
        print(f'Latency: {latency:.2f} ms/image')
        print(f"Model size: {result['model_size_mb']:.2f} MB")
        print(f'Misclassified test images: {mistakes}')
        print(f'Results saved to: {result_dir}')
    except (Exception, KeyboardInterrupt) as error:
        if args.check_data:
            raise
        result['train_time_s'] = time.perf_counter() - started if started is not None else 0
        if result.get('history'):
            save_curves(result['history'], result_dir)
        result['status'] = 'failed'
        result['note'] = f'{type(error).__name__}: {error}'
        save_json(result_dir / 'results.json', result)
        (result_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == "__main__":
    main()
