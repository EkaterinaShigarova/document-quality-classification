import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path, PureWindowsPath

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SEED = 24500007
THRESHOLDS = [i / 100 for i in range(10, 91, 5)]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_validation(data_dir, source):
    metadata = data_dir / "metadata.jsonl"
    if file_hash(metadata) != source["metadata_sha256"]:
        raise ValueError("Metadata differs from Task 5A. Use the original data and split.")
    records = [json.loads(line) for line in metadata.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    counts, groups, seen = Counter(), {}, set()
    for row in records:
        split, label = row["split"], row["quality_label"]
        if split not in {"train", "validation", "test"} or label not in {"bad", "good"}:
            raise ValueError("Invalid split or quality label in metadata.")
        if int(row["seed"]) != SEED:
            raise ValueError("The metadata seed does not match Task 5A.")
        relative = row["image_path"].replace("\\", "/")
        path = Path(relative)
        if (path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts
                or path.parts[:2] != (split, label) or relative in seen):
            raise ValueError(f"Invalid or duplicate image path: {relative}")
        seen.add(relative)
        key = (row["class_name"], str(row["source_id"]))
        if key in groups and groups[key] != split:
            raise ValueError("A source document occurs in multiple splits.")
        groups[key] = split
        counts[split] += 1
    if dict(counts) != source["split_counts"]:
        raise ValueError("Split sizes differ from Task 5A.")
    rows = sorted((row for row in records if row["split"] == "validation"), key=lambda row: row["image_path"])
    if set(row["quality_label"] for row in rows) != {"bad", "good"}:
        raise ValueError("Validation must contain both classes.")
    for row in rows:
        if not (data_dir / row["image_path"].replace("\\", "/")).is_file():
            raise FileNotFoundError(row["image_path"])
    return rows


def load_model(checkpoint_path, source):
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import efficientnet_b0

    for key, actual in [("pytorch", torch.__version__), ("torchvision", torchvision.__version__)]:
        if actual != source["environment"][key]:
            raise ValueError(f"{key} differs from Task 5A. Use the original .venv_quality environment.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {"model": "efficientnet_b0", "class_names": ["bad", "good"], "seed": SEED,
                "input_size": [224, 224], "epoch": source["best_epoch"]}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key} differs from Task 5A results.")
    expected_loss = source["history"]["val_loss"][source["best_epoch"] - 1]
    if not math.isclose(float(checkpoint["val_loss"]), expected_loss, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Checkpoint validation loss differs from Task 5A.")
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, 1))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != source["params"]:
        raise ValueError("Model parameter count differs from Task 5A.")
    model.requires_grad_(False)
    return model.cpu().float().eval()


def collect_probabilities(model, rows, data_dir):
    import torch
    from PIL import Image
    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 32):
            batch_rows = rows[offset:offset + 32]
            images = []
            for row in batch_rows:
                if row["split"] != "validation":
                    raise ValueError("Only validation images may enter threshold selection.")
                path = data_dir / row["image_path"].replace("\\", "/")
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            probabilities_good = torch.sigmoid(model(torch.stack(images)).reshape(-1)).tolist()
            for row, probability_good in zip(batch_rows, probabilities_good):
                if not math.isfinite(probability_good) or not 0 <= probability_good <= 1:
                    raise ValueError("Invalid model probability.")
                predictions.append({"image_path": row["image_path"], "true_label": row["quality_label"],
                                    "probability_good": probability_good, "probability_bad": 1.0 - probability_good})
            print(f"Validation predictions: {len(predictions)}/{len(rows)}", flush=True)
    return predictions


def score_threshold(predictions, threshold, false_accept_cost, false_reject_cost):
    if not predictions:
        raise ValueError("No validation predictions.")
    tp = tn = fa = fr = 0
    for row in predictions:
        actual_bad = row["true_label"] == "bad"
        predicted_bad = row["probability_bad"] > threshold
        if actual_bad and predicted_bad:
            tp += 1
        elif actual_bad:
            fa += 1
        elif predicted_bad:
            fr += 1
        else:
            tn += 1
    precision = tp / (tp + fr) if tp + fr else 0.0
    recall = tp / (tp + fa) if tp + fa else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cost = false_accept_cost * fa + false_reject_cost * fr
    return {"threshold_bad": threshold, "precision_bad": precision, "recall_bad": recall, "f1_bad": f1,
            "true_bad_rejected": tp, "true_good_accepted": tn, "false_accepts": fa, "false_rejects": fr,
            "accuracy": (tp + tn) / len(predictions), "weighted_cost": cost,
            "weighted_cost_per_image": cost / len(predictions)}


def select_threshold(sweep):
    return min(sweep, key=lambda row: (row["weighted_cost"], -row["f1_bad"],
                                      round(abs(row["threshold_bad"] - 0.5), 8),
                                      -row["recall_bad"], row["threshold_bad"]))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(predictions, default, chosen, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    labels = [int(row["true_label"] == "bad") for row in predictions]
    scores = [row["probability_bad"] for row in predictions]
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(recall, precision, where="post", label="Validation PR curve (bad)")
    ax.scatter(default["recall_bad"], default["precision_bad"], marker="o", s=150,
               facecolors="none", edgecolors="black", linewidths=2, zorder=4, label="Default: t_bad = 0.50")
    ax.scatter(chosen["recall_bad"], chosen["precision_bad"], marker="x", s=85,
               color="red", linewidths=2, zorder=5, label=f"Chosen: t_bad = {chosen['threshold_bad']:.2f}")
    ax.set(xlabel="Recall (bad)", ylabel="Precision (bad)", title="Threshold selection on validation only",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_memo(path, result):
    default, chosen = result["validation_at_default"], result["validation_at_chosen"]
    fa_cost, fr_cost = result["cost_assumption"]["false_accept"], result["cost_assumption"]["false_reject"]
    choice = ("The default threshold remains the best choice under this rule; I do not force a change without validation evidence."
              if chosen["threshold_bad"] == 0.5 else "I recommend the selected threshold for a prototype under the stated cost assumption.")
    counts = result["validation_class_counts"]
    lines = ["# Threshold and Cost Analysis", "", "Student: Ekaterina Shigarova, IIKb-24. Task 5B. Seed: 24500007.", "",
             "I used the saved fine-tuned EfficientNet-B0 checkpoint from Task 5A. No training was performed. "
             f"Threshold selection used only {result['validation_samples']} validation images "
             f"({counts['bad']} bad, {counts['good']} good). Test images and test predictions were not read.", "",
             "The model sigmoid output is P(good); P(bad) = 1 - P(good). I reject an image as bad when P(bad) > t_bad; "
             "otherwise it is good. The strict comparison preserves the original default decision, including ties at 0.5. "
             "All 17 thresholds from 0.10 to 0.90 in steps of 0.05 were evaluated for bad-class precision, recall and F1.", "",
             "A false accept passes a poor-quality document to OCR or later processing, risking incorrect extracted data and later rework. "
             "A false reject asks for another photo or manual review of a usable image. "
             f"I assume costs of {fa_cost:g} units per false accept and {fr_cost:g} per false reject. "
             "These are hypothetical relative costs, not measured money or operator time. "
             f"The rule minimizes {fa_cost:g} * false_accepts + {fr_cost:g} * false_rejects. "
             "Ties prefer higher bad-class F1, then proximity to 0.5, higher bad recall, and finally the lower threshold. "
             "The rule and costs are set before inspecting the sweep.", "",
             "| Validation measure | Default t_bad = 0.50 | Chosen t_bad = " + f"{chosen['threshold_bad']:.2f} |",
             "|---|---:|---:|",]
    for label, key in [("Bad precision", "precision_bad"), ("Bad recall", "recall_bad"), ("Bad F1", "f1_bad"),
                       ("False accepts", "false_accepts"), ("False rejects", "false_rejects"), ("Weighted cost", "weighted_cost")]:
        digits = 0 if key in {"false_accepts", "false_rejects"} else 4
        lines.append(f"| {label} | {default[key]:.{digits}f} | {chosen[key]:.{digits}f} |")
    n = result["validation_samples"]
    lines += ["", f"Expected counts per {n} images with a similar class mix are the validation counts above. "
              f"Rescaled to 1,000 images, false accepts change from {default['false_accepts'] * 1000 / n:.1f} to "
              f"{chosen['false_accepts'] * 1000 / n:.1f}, and false rejects from {default['false_rejects'] * 1000 / n:.1f} to "
              f"{chosen['false_rejects'] * 1000 / n:.1f}. These are sample-based estimates, not deployment guarantees. "
              + choice + " This threshold has no independent test evaluation in Task 5B. "
              "Synthetic validation performance and the assumed costs may not transfer to real document photos.", "",
              "![Validation precision-recall curve](../results/efficientnetb0_threshold/precision_recall_curve.png)", "",
              "Sweep table: ../results/efficientnetb0_threshold/threshold_sweep.csv", "",
              "Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(predictions, source, output, report, fa_cost, fr_cost, provenance):
    if any(not math.isfinite(c) or c <= 0 for c in [fa_cost, fr_cost]):
        raise ValueError("Error costs must be finite positive numbers.")
    if any(row["true_label"] not in {"bad", "good"} or not math.isfinite(row["probability_bad"])
           or not 0 <= row["probability_bad"] <= 1 for row in predictions):
        raise ValueError("Invalid validation predictions.")
    sweep = [score_threshold(predictions, t, fa_cost, fr_cost) for t in THRESHOLDS]
    chosen = select_threshold(sweep)
    default = next(row for row in sweep if row["threshold_bad"] == 0.5)
    if output.exists() or report.exists():
        raise FileExistsError("Threshold results or report already exist. They will not be overwritten.")
    result = {"experiment": "efficientnetb0_threshold", "task": "5B", "model": source["model"],
              "regime": source["regime"], "parent_experiment": source["experiment"], "seed": source["seed"],
              "status": "running", "metrics_split": "validation", "validation_samples": len(predictions),
              "validation_class_counts": dict(Counter(row["true_label"] for row in predictions)),
              "chosen_threshold": chosen["threshold_bad"], "threshold_probability": "P(bad) = 1 - sigmoid(logit)",
              "decision_rule": "bad if P(bad) > chosen_threshold, otherwise good", "default_threshold": 0.5,
              "sweep_thresholds": THRESHOLDS, "cost_assumption": {"false_accept": fa_cost, "false_reject": fr_cost,
              "units": "hypothetical relative cost units, not measured operational costs"},
              "selection_rule": "minimum weighted cost; ties: higher bad F1, nearest 0.5, higher bad recall, lower threshold",
              "validation_at_default": default, "validation_at_chosen": chosen, "test_evaluated": False,
              "training_performed": False, "original_task5a_results_modified": False, **provenance}
    output.mkdir(parents=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    save_json(output / "results.json", result)
    for row in predictions:
        row["predicted_default"] = "bad" if row["probability_bad"] > 0.5 else "good"
        row["predicted_chosen"] = "bad" if row["probability_bad"] > chosen["threshold_bad"] else "good"
    save_csv(output / "validation_predictions.csv", predictions)
    save_csv(output / "threshold_sweep.csv", sweep)
    plot_curve(predictions, default, chosen, output / "precision_recall_curve.png")
    write_memo(report, result)
    result["status"] = "completed"
    save_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Task 5B: validation-only threshold and cost analysis. No training.")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "quality_split")
    parser.add_argument("--source-results", type=Path, default=BASE_DIR / "results" / "efficientnetb0_finetune_top30" / "results.json")
    parser.add_argument("--checkpoint", type=Path, default=BASE_DIR / "results" / "efficientnetb0_finetune_top30" / "best.pt")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results" / "efficientnetb0_threshold")
    parser.add_argument("--report", type=Path, default=BASE_DIR / "reports" / "bonus_threshold_memo.md")
    parser.add_argument("--false-accept-cost", type=float, default=5.0)
    parser.add_argument("--false-reject-cost", type=float, default=1.0)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(c) or c <= 0 for c in [args.false_accept_cost, args.false_reject_cost]):
        raise ValueError("Error costs must be finite positive numbers.")
    source = read_json(args.source_results)
    expected = {"status": "completed", "regime": "finetune_top30", "model": "EfficientNet-B0", "seed": SEED,
                "class_indices": {"bad": 0, "good": 1}, "input_size": [224, 224]}
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"Unexpected {key} in source results. Use the completed Task 5A experiment.")
    if not args.check_data and (args.output.exists() or args.report.exists()):
        raise FileExistsError("Threshold results or report already exist. They will not be overwritten.")
    rows = read_validation(args.data, source)
    model = load_model(args.checkpoint, source)
    print(f"Checkpoint loaded: epoch {source['best_epoch']}; validation images: {len(rows)}")
    print(f"Cost assumption set before prediction: false accept = {args.false_accept_cost:g}, false reject = {args.false_reject_cost:g}")
    if args.check_data:
        print("Validation metadata and checkpoint check completed. No predictions or training were started.")
        return
    predictions = collect_probabilities(model, rows, args.data)
    result = run_analysis(predictions, source, args.output, args.report, args.false_accept_cost, args.false_reject_cost,
                          {"metadata_sha256": source["metadata_sha256"], "source_results_sha256": file_hash(args.source_results),
                           "checkpoint_sha256": file_hash(args.checkpoint), "checkpoint_epoch": source["best_epoch"],
                           "preprocessing": source["hyperparameters"]["preprocessing"], "environment": source["environment"]})
    print(f"Chosen P(bad) threshold: {result['chosen_threshold']:.2f}")
    for name, key in [("Default 0.50", "validation_at_default"), ("Chosen", "validation_at_chosen")]:
        row = result[key]
        print(f"{name}: bad F1 = {row['f1_bad']:.4f}; false accepts = {row['false_accepts']}; "
              f"false rejects = {row['false_rejects']}; weighted cost = {row['weighted_cost']:g}")
    print("Threshold analysis completed. No test images were read and no model was trained.")
    print(f"Results: {args.output.resolve()}")
    print(f"Memo: {args.report.resolve()}")


if __name__ == "__main__":
    main()
