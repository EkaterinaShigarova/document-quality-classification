import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path, PureWindowsPath

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SEED = 24500007
CLASS_NAMES = ["bad", "good"]
OPSET = 18


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dependencies():
    modules = {}
    for name in ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "onnxscript"]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError as error:
            raise SystemExit(f"Cannot import {name}: {error}\nInstall ONNX dependencies in .venv_quality: python -m pip install onnx onnxruntime onnxscript") from error
    return modules


def load_test_rows(data_dir, source):
    metadata = data_dir / "metadata.jsonl"
    if file_hash(metadata) != source["metadata_sha256"]:
        raise ValueError("Metadata differs from Task 5A. Use the same data and split.")
    records = [json.loads(line) for line in metadata.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    counts, seen, groups = Counter(), set(), {}
    for row in records:
        split, label = row["split"], row["quality_label"]
        if split not in {"train", "validation", "test"} or label not in CLASS_NAMES or int(row["seed"]) != SEED:
            raise ValueError("Invalid split, label or seed in metadata.")
        relative = row["image_path"].replace("\\", "/")
        path = Path(relative)
        if (path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts
                or path.parts[:2] != (split, label) or relative in seen):
            raise ValueError(f"Invalid or duplicate image path: {relative}")
        seen.add(relative)
        key = (row["class_name"], str(row["source_id"]))
        if key in groups and groups[key] != split:
            raise ValueError("Source leakage detected.")
        groups[key] = split
        counts[split] += 1
    if dict(counts) != source["split_counts"]:
        raise ValueError("Split counts differ from Task 5A.")
    rows = sorted((r for r in records if r["split"] == "test"), key=lambda r: r["image_path"])
    if len(rows) != source["test_samples"] or set(r["quality_label"] for r in rows) != set(CLASS_NAMES):
        raise ValueError("Unexpected test split.")
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
    expected = {"model": "efficientnet_b0", "class_names": CLASS_NAMES, "seed": SEED,
                "input_size": [224, 224], "epoch": source["best_epoch"]}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key} differs from Task 5A.")
    expected_loss = source["history"]["val_loss"][source["best_epoch"] - 1]
    if not math.isclose(float(checkpoint["val_loss"]), expected_loss, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Checkpoint validation loss differs from Task 5A.")
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, 1))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != source["params"]:
        raise ValueError("Model parameter count differs from Task 5A.")
    return model.cpu().float().requires_grad_(False).eval()


def preprocessing():
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def export_model(core, output_path):
    import torch
    import onnx

    class ProbabilityModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.core = model

        def forward(self, image):
            return torch.sigmoid(self.core(image).reshape(-1))

    wrapper = ProbabilityModel(core).eval()
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(wrapper, (example,), str(output_path), dynamo=True,
                      input_names=["image"], output_names=["probability_good"],
                      opset_version=OPSET, external_data=False)
    onnx.checker.check_model(str(output_path), full_check=True)
    graph = onnx.load(str(output_path))
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("The export unexpectedly uses external weight files.")
    return wrapper


def create_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 224, 224] or inputs[0].type != "tensor(float)":
        raise ValueError("Unexpected ONNX input contract.")
    if len(outputs) != 1 or outputs[0].name != "probability_good":
        raise ValueError("Unexpected ONNX output contract.")
    return session


def quality_metrics(rows, prediction_key):
    cm = [[0, 0], [0, 0]]
    for row in rows:
        cm[CLASS_NAMES.index(row["true_label"])][CLASS_NAMES.index(row[prediction_key])] += 1
    count = sum(map(sum, cm))
    if not count:
        raise ValueError("No evaluation predictions.")
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp, support = cm[i][i], sum(cm[i])
        predicted = cm[0][i] + cm[1][i]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
    return {"accuracy": (cm[0][0] + cm[1][1]) / count,
            "precision_macro": statistics.mean(v["precision"] for v in per_class.values()),
            "recall_macro": statistics.mean(v["recall"] for v in per_class.values()),
            "f1_macro": statistics.mean(v["f1-score"] for v in per_class.values()),
            "per_class": per_class, "confusion_matrix": cm}


def evaluate_pair(wrapper, session, data_dir, rows, transform):
    import torch
    from PIL import Image

    predictions = []
    input_name = session.get_inputs()[0].name
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            if row["split"] != "test":
                raise ValueError("Deployment evaluation must use the original test split.")
            with Image.open(data_dir / row["image_path"].replace("\\", "/")) as image:
                batch = transform(image.convert("RGB")).unsqueeze(0)
            before = float(wrapper(batch).reshape(-1)[0].item())
            after = float(session.run(None, {input_name: batch.numpy()})[0].reshape(-1)[0])
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in [before, after]):
                raise ValueError("An engine returned an invalid probability.")
            predictions.append({"image": row["image_path"], "true_label": row["quality_label"],
                                "defect_type": row["defect_type"], "probability_good_pytorch": before,
                                "probability_good_onnx": after, "prediction_pytorch": "good" if before >= 0.5 else "bad",
                                "prediction_onnx": "good" if after >= 0.5 else "bad",
                                "probability_abs_difference": abs(before - after)})
            if index % 32 == 0 or index == len(rows):
                print(f"Paired test evaluation: {index}/{len(rows)}", flush=True)
    return predictions


def benchmark_pair(wrapper, session, data_dir, rows, transform, clock=time.perf_counter_ns):
    import torch
    from PIL import Image

    images = []
    for row in rows[:100]:
        with Image.open(data_dir / row["image_path"].replace("\\", "/")) as opened:
            images.append(opened.convert("RGB").copy())
    if not images:
        raise ValueError("No images are available for the latency benchmark.")
    input_name = session.get_inputs()[0].name
    times = []
    with torch.inference_mode():
        for index in range(120):
            image = images[index % len(images)]
            order = ["pytorch", "onnx"] if index % 2 == 0 else ["onnx", "pytorch"]
            for engine in order:
                start = clock()
                batch = transform(image).unsqueeze(0)
                if engine == "pytorch":
                    probability = float(wrapper(batch).reshape(-1)[0].item())
                else:
                    probability = float(session.run(None, {input_name: batch.numpy()})[0].reshape(-1)[0])
                decision = int(probability >= 0.5)
                elapsed_ms = (clock() - start) / 1_000_000
                if index >= 20:
                    times.append({"iteration": index - 19, "engine": engine,
                                  "image": rows[index % len(images)]["image_path"], "latency_ms": elapsed_ms,
                                  "decision_good": decision, "measured_first": engine == order[0]})
    return times


def summarize_times(times, engine):
    import numpy as np
    values = [row["latency_ms"] for row in times if row["engine"] == engine]
    if len(values) != 100 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Each engine requires exactly 100 finite timed measurements.")
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)), "timed_predictions": 100, "warmup": 20}


def benchmark_rows(benchmark):
    rows = []
    for key in ["pytorch", "onnx"]:
        item = benchmark[key]
        rows.append({"format": key, "size_mb": item["size_mb"], "mean_ms": item["latency"]["mean_ms"],
                     "median_ms": item["latency"]["median_ms"], "p95_ms": item["latency"]["p95_ms"],
                     "accuracy": item["metrics"]["accuracy"], "f1_macro": item["metrics"]["f1_macro"],
                     "bad_recall": item["metrics"]["per_class"]["bad"]["recall"]})
    return rows


def write_memo(path, result):
    b = result["deployment_benchmark"]
    pt, onnx = b["pytorch"], b["onnx"]
    latency = onnx["latency"]["mean_ms"]
    accuracy_drop = 100 * b["accuracy_drop"]
    unchanged = b["decision_disagreements"] == 0
    quality_pass = (onnx["metrics"]["accuracy"] >= .92 and onnx["metrics"]["f1_macro"] >= .90
                    and onnx["metrics"]["per_class"]["bad"]["recall"] >= .90)
    verdict = ("A local CPU prototype on hardware similar to this laptop is a candidate: measured quality passes the original limits, "
               "and this measured stage fits within 52 ms. A complete end-to-end measurement is still needed."
               if quality_pass and latency < 52 else
               "The current local-CPU result does not meet both the measured quality criteria and the 52 ms stage reference. "
               "Use a server or stronger-hardware prototype and measure it before claiming the target is met.")
    lines = ["# Deployment Export and CPU Benchmark", "", "Student: Ekaterina Shigarova, IIKb-24. Task 5D. Seed: 24500007.", "",
             "I exported the fine-tuned EfficientNet-B0 from Task 5A to one FP32 ONNX file (opset 18, fixed input 1 x 3 x 224 x 224). "
             "The 5C augmentation did not improve the reported test decisions, so the previous prototype was retained. "
             "No model was trained. Both runtimes output P(good), with the same threshold 0.5. "
             "The separate validation-selected threshold from 5B is not applied, so this experiment isolates export/runtime effects.", "",
             "## Measured results on the same test split", "",
             "| Format | File size (MB) | Mean (ms/image) | Median (ms/image) | P95 (ms/image) | Accuracy | Macro F1 | Bad recall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in benchmark_rows(b):
        lines.append(f"| {row['format']} | {row['size_mb']:.2f} | {row['mean_ms']:.2f} | {row['median_ms']:.2f} | "
                     f"{row['p95_ms']:.2f} | {row['accuracy']:.6f} | {row['f1_macro']:.6f} | {row['bad_recall']:.6f} |")
    lines += ["", f"Accuracy drop (PyTorch minus ONNX): {accuracy_drop:+.4f} percentage points. "
              f"Changed decisions: {b['decision_disagreements']}/{b['test_samples']}. "
              f"Maximum absolute probability difference: {b['max_probability_abs_difference']:.9g}. "
              + ("All test decisions agree." if unchanged else "Some test decisions differ and require review before deployment."), "",
              "## Timing protocol", "",
              "Both engines were remeasured in this session on the same CPU, using batch 1, four intra-op threads, "
              "20 warm-up calls and 100 timed calls per engine. Engine order alternates to reduce ordering bias. "
              "Timing includes the same full-image resize, tensor conversion, ImageNet normalization, inference and threshold decision. "
              "Decoded RGB images are preloaded; file reading, decoding, model loading, network transfer and downstream OCR are excluded. "
              "ONNX Runtime uses CPUExecutionProvider, sequential execution, graph optimizations and disabled idle thread spinning. "
              "Raw timings are saved. Values are from one session, not a confidence interval. File sizes exclude runtime libraries and are not RAM measurements.", "",
              "## Server versus edge recommendation", "", verdict + " For an actual low-power edge device, measure its CPU, memory use and energy first; "
              "this laptop benchmark does not establish mobile/embedded performance. A server may be preferable for centrally managed or higher-throughput use, "
              "but no server was benchmarked here.", "",
              f"Compared with the handout's 52 ms quality-stage reference, measured ONNX time is {latency:.2f} ms/image "
              f"({52 - latency:+.2f} ms remaining before the excluded work). This is not proof of a 52 ms complete pipeline. "
              "The handout's YOLOv8 2.1 ms / 2.8 MB numbers are external capstone references, not measurements on this hardware or this EfficientNet model. "
              "Different hardware, input sizes and timing scopes prevent a direct speed ranking.", "",
              "## Quantization risks and limitations", "",
              "Quantization was not applied: the export remains FP32. INT8 rounding and clipping can alter logits and move borderline photos across the threshold, "
              "potentially increasing false accepts or false rejects. Any future quantization should use representative training data for calibration, "
              "validation data for operating-point decisions, and a separate final evaluation. "
              "The test split here has been reused for laboratory comparisons, and the photos are synthetic; a fresh real-photo evaluation is needed before operational deployment.", "",
              "References: https://docs.pytorch.org/docs/2.15/onnx.html ; "
              "https://onnxruntime.ai/docs/performance/tune-performance/threading.html", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Task 5D: FP32 ONNX export and paired CPU deployment benchmark.")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "quality_split")
    parser.add_argument("--source-results", type=Path, default=BASE_DIR / "results" / "efficientnetb0_finetune_top30" / "results.json")
    parser.add_argument("--checkpoint", type=Path, default=BASE_DIR / "results" / "efficientnetb0_finetune_top30" / "best.pt")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results" / "efficientnetb0_onnx")
    parser.add_argument("--report", type=Path, default=BASE_DIR / "reports" / "bonus_deployment_memo.md")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    modules = dependencies()
    source = json.loads(args.source_results.read_text(encoding="utf-8-sig"))
    expected = {"status": "completed", "model": "EfficientNet-B0", "regime": "finetune_top30", "seed": SEED,
                "input_size": [224, 224], "class_indices": {"bad": 0, "good": 1}, "experiment": "efficientnetb0_finetune_top30"}
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"Unexpected {key} in source results. Use the completed Task 5A model.")
    if source["hyperparameters"]["threshold"] != 0.5:
        raise ValueError("Use the original 0.5 source evaluation to isolate export effects.")
    if not args.check_data and (args.output.exists() or args.report.exists()):
        raise FileExistsError("Deployment results or report already exist. They will not be overwritten.")
    rows = load_test_rows(args.data, source)
    core = load_model(args.checkpoint, source)
    print(f"Task 5D source loaded: epoch {source['best_epoch']}; test images: {len(rows)}")
    print("ONNX dependencies: " + ", ".join(f"{name}={modules[name].__version__}" for name in ["onnx", "onnxruntime", "onnxscript"]))
    if args.check_data:
        print("Data, dependencies and checkpoint check completed. Export and benchmark were not started.")
        return
    args.output.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = {"experiment": "efficientnetb0_onnx", "task": "5D", "model": source["model"], "regime": source["regime"],
              "artifact_format": "ONNX FP32", "parent_experiment": source["experiment"], "seed": SEED,
              "input_size": [224, 224], "class_indices": source["class_indices"], "params": source["params"],
              "test_samples": len(rows), "metadata_sha256": source["metadata_sha256"], "split_counts": source["split_counts"],
              "source_results_sha256": file_hash(args.source_results), "source_checkpoint_sha256": file_hash(args.checkpoint),
              "checkpoint_epoch": source["best_epoch"], "training_performed": False, "quantization": "none",
              "threshold": 0.5, "threshold_probability": "P(good)", "status": "running",
              "command": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
              "environment": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                              **{name: modules[name].__version__ for name in modules}, "cpu_threads": 4, "inter_op_threads": 1},
              "export": {"opset": OPSET, "dynamo": True, "external_data": False, "precision": "float32",
                         "input": "image: float32[1,3,224,224]", "output": "probability_good: float32[1]",
                         "preprocessing": source["hyperparameters"]["preprocessing"]}}
    save_json(args.output / "results.json", result)
    try:
        onnx_path = args.output / "model.onnx"
        print("Exporting EfficientNet-B0 to ONNX...", flush=True)
        wrapper = export_model(core, onnx_path)
        session = create_session(onnx_path)
        transform = preprocessing()
        predictions = evaluate_pair(wrapper, session, args.data, rows, transform)
        before = quality_metrics(predictions, "prediction_pytorch")
        after = quality_metrics(predictions, "prediction_onnx")
        save_csv(args.output / "test_predictions_export.csv", predictions)
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            if not math.isclose(before[key], source["metrics"][key], rel_tol=0, abs_tol=1e-10):
                raise ValueError("Remeasured PyTorch quality differs from Task 5A. Inspect saved predictions before comparing export.")
        if before["confusion_matrix"] != source["confusion_matrix"]:
            raise ValueError("Remeasured PyTorch confusion matrix differs from Task 5A.")
        print("Benchmarking: 20 warm-up + 100 timed calls per engine...", flush=True)
        times = benchmark_pair(wrapper, session, args.data, rows, transform)
        save_csv(args.output / "latency_measurements.csv", times)
        benchmark = {"test_samples": len(rows), "threshold": 0.5, "metrics_split": "test",
                     "pytorch": {"size_mb": args.checkpoint.stat().st_size / 1_000_000, "metrics": before,
                                 "latency": summarize_times(times, "pytorch")},
                     "onnx": {"size_mb": onnx_path.stat().st_size / 1_000_000, "metrics": after,
                              "latency": summarize_times(times, "onnx"), "providers": session.get_providers()},
                     "accuracy_drop": before["accuracy"] - after["accuracy"],
                     "f1_macro_drop": before["f1_macro"] - after["f1_macro"],
                     "decision_disagreements": sum(r["prediction_pytorch"] != r["prediction_onnx"] for r in predictions),
                     "max_probability_abs_difference": max(r["probability_abs_difference"] for r in predictions),
                     "protocol": {"warmup": 20, "timed_predictions": 100, "batch_size": 1, "engine_order": "alternating",
                                  "includes": "RGB image resize, tensor conversion, normalization, inference, decision",
                                  "excludes": "disk IO, image decoding, model loading, network, downstream OCR",
                                  "scope": "quality classifier stage; not the full end-to-end document pipeline"}}
        result.update({"deployment_benchmark": benchmark, "metrics": {k: v for k, v in after.items() if k != "confusion_matrix"},
                       "confusion_matrix": after["confusion_matrix"], "latency_ms_per_image": benchmark["onnx"]["latency"]["mean_ms"],
                       "model_size_mb": benchmark["onnx"]["size_mb"], "model_file": "model.onnx"})
        save_csv(args.output / "benchmark.csv", benchmark_rows(benchmark))
        write_memo(args.report, result)
        result["status"] = "completed"
        save_json(args.output / "results.json", result)
        print("Export and deployment benchmark completed.")
        print(f"PyTorch: {benchmark['pytorch']['size_mb']:.2f} MB; {benchmark['pytorch']['latency']['mean_ms']:.2f} ms/image; accuracy {before['accuracy']:.4f}")
        print(f"ONNX: {benchmark['onnx']['size_mb']:.2f} MB; {benchmark['onnx']['latency']['mean_ms']:.2f} ms/image; accuracy {after['accuracy']:.4f}")
        print(f"Accuracy drop: {100 * benchmark['accuracy_drop']:+.4f} percentage points; changed decisions: {benchmark['decision_disagreements']}")
        print(f"Results: {args.output.resolve()}")
        print(f"Memo: {args.report.resolve()}")
    except (Exception, KeyboardInterrupt) as error:
        result["status"] = "failed"
        result["note"] = f"{type(error).__name__}: {error}"
        save_json(args.output / "results.json", result)
        (args.output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
