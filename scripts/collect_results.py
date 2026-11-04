import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
EXPERIMENTS = [
    ("mobilenetv2_frozen", "MobileNetV2"),
    ("week7_yolov8n_cls", "YOLOv8n-cls"),
    ("week7_yolov8s_cls", "YOLOv8s-cls"),
    ("efficientnetb0_frozen", "EfficientNet-B0"),
]


def passes_screening(row):
    metrics = row["metrics"]
    return (
        metrics["accuracy"] >= 0.92
        and metrics["f1_macro"] >= 0.90
        and metrics["per_class"]["bad"]["recall"] >= 0.90
        and row["latency_ms_per_image"] < 100
        and row["model_size_mb"] <= 50
    )


def load_results(folder):
    rows = []
    for experiment, label in EXPERIMENTS:
        path = folder / experiment / "results.json"
        row = json.loads(path.read_text(encoding="utf-8-sig"))
        if row["experiment"] != experiment or row.get("status", "completed") != "completed":
            raise ValueError(f"Unexpected or unfinished experiment: {path}")
        if row["seed"] != 24500007 or row["test_samples"] != 276:
            raise ValueError(f"Unexpected seed or test count: {path}")
        matrix = np.asarray(row["confusion_matrix"], dtype=int)
        if matrix.shape != (2, 2) or np.any(matrix < 0) or matrix.sum() != 276:
            raise ValueError(f"Invalid confusion matrix: {path}")
        if not np.isclose(matrix.trace() / matrix.sum(), row["metrics"]["accuracy"]):
            raise ValueError(f"Accuracy disagrees with confusion matrix: {path}")
        row["display_name"] = label
        rows.append(row)
    if len({row["metadata_sha256"] for row in rows}) != 1:
        raise ValueError("The metadata hashes differ across experiments.")
    return rows


def write_comparison(rows, output):
    lines = [
        "# Full Model Comparison", "",
        "Student: Ekaterina Shigarova, IIKb-24. Task 4. Seed: 24500007.", "",
        "This table is generated from the four experiment results.json files. Precision, recall and F1 are macro averages. All runs use the same 276 test images and metadata hash.", "",
        "| Model | Accuracy | Precision | Recall | F1 | Latency (ms/image) | Size (MB) | Parameters | Training (min) | Epochs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['display_name']} | {m['accuracy'] * 100:.2f}% | {m['precision_macro']:.6f} | "
            f"{m['recall_macro']:.6f} | {m['f1_macro']:.6f} | {row['latency_ms_per_image']:.2f} | "
            f"{row['model_size_mb']:.2f} | {row['params']:,} | {row['train_time_s'] / 60:.2f} | {row['epochs_run']} |"
        )
    lines += [
        "", "## Quality-gate errors and acceptance criteria", "",
        "False accept means bad predicted good. False reject means good predicted bad.", "",
        "| Model | Bad recall | False accepts | False rejects | Passes all thresholds |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        matrix = row["confusion_matrix"]
        lines.append(
            f"| {row['display_name']} | {row['metrics']['per_class']['bad']['recall'] * 100:.2f}% | "
            f"{matrix[0][1]} | {matrix[1][0]} | {'Yes' if passes_screening(row) else 'No'} |"
        )
    qualified = [row for row in rows if passes_screening(row)]
    if qualified:
        highest_f1 = max(row["metrics"]["f1_macro"] for row in qualified)
        candidates = [row for row in qualified if highest_f1 - row["metrics"]["f1_macro"] <= 0.01]
        selected = min(candidates, key=lambda row: (row["latency_ms_per_image"], row["model_size_mb"]))
        lines += ["", "## Selection using the preregistered rule", "",
                  f"{selected['display_name']} is selected for the initial CPU quality gate. It passes all screening thresholds. Among passing models whose macro F1 is within 0.01 of the highest value, it has the lowest measured latency. This is a selection for the synthetic-data prototype, pending real-photo testing."]
    else:
        lines += ["", "No model passes all screening thresholds."]
    lines += [
        "", "## Protocol and limits", "",
        "The actual split is 1,308 train, 216 validation and 276 test images. It comes from source-level sampling with 70/15/15 probabilities. Related variants stay together. Models use seed 24500007, maximum 50 epochs, patience 5 and batch size 32.", "",
        "MobileNetV2 and EfficientNet-B0 use frozen bases and 224 x 224 inputs. Both YOLO variants use full fine-tuning and 640 x 640 inputs. Architecture, input size, head design, framework and training regime therefore all contribute to the measured differences.", "",
        "CPU latency uses batch size 1, 20 warm-up predictions and 100 timed predictions. It includes resizing, preprocessing, inference and the decision. Decoding, disk access, model loading and network operations are excluded. The recorded CPU is the same, with four intra-op threads and one inter-op thread. These are single timing measurements and do not establish a stable ranking across different system loads or machines.", "",
        "Saved sizes describe inference.keras for MobileNetV2, optimizer-stripped best.pt for YOLO and a float32 state_dict checkpoint for EfficientNet-B0. Formats and stored precision differ. File size is not runtime memory usage. MB means 1,000,000 bytes.", "",
        "EfficientNet-B0 passes the speed budget but accepts two bad images, compared with one for each YOLO variant. Its speed advantage over nano is approximately 6.63 ms per image. Error cost remains relevant even when the screening thresholds pass. No real-photo evaluation is available.", "",
        "## Figures", "",
        "![Test accuracy and CPU latency](accuracy_vs_latency.png)", "",
        "![Test confusion matrices](confusion_matrices.png)", "",
    ]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(rows, output):
    colors = ["#555555", "#0072B2", "#D55E00", "#009E73"]
    offsets = [(-8, 12), (12, -20), (0, 12), (10, 12)]
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for row, color, offset in zip(rows, colors, offsets):
        x = row["latency_ms_per_image"]
        y = row["metrics"]["accuracy"] * 100
        ax.scatter(x, y, s=90, color=color, zorder=3)
        ax.annotate(row["display_name"], (x, y), xytext=offset, textcoords="offset points", ha="right" if x > 120 else "left", fontsize=10)
    ax.axvline(100, color="#777777", linestyle="--", label="CPU budget: <100 ms")
    ax.set(xlim=(0, 155), ylim=(98.9, 100), xlabel="Mean CPU latency (ms/image)", ylabel="Test accuracy (%)", title="Accuracy and latency on the same 276 synthetic test images")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.text(0.5, 0.015, "Accuracy axis is zoomed. A 0.36 percentage-point difference equals one test image.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "accuracy_vs_latency.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, row in zip(axes, rows):
        cm = np.asarray(row["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=138)
        ax.set_xticks([0, 1], ["bad", "good"])
        ax.set_yticks([0, 1], ["bad", "good"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(row["display_name"], fontsize=12)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, str(cm[a, b]), ha="center", va="center", color="white" if cm[a, b] > 69 else "#111111", fontsize=16)
    fig.suptitle("Final test confusion matrices: identical class order and colour scale", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build the Task 4 comparison from saved results only.")
    parser.add_argument("--results", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "reports")
    args = parser.parse_args()
    rows = load_results(args.results.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_comparison(rows, output)
    make_plots(rows, output)
    print("Comparison generated from four results.json files. No training was started.")
    print(f"Reports saved to: {output}")


if __name__ == "__main__":
    main()
