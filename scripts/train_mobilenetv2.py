import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import hashlib
import argparse
import json
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DATA_DIR = BASE_DIR / "data" / "quality_split"
RESULT_DIR = BASE_DIR / "results" / "mobilenetv2_frozen"

SEED = 24500007
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 0.001
CPU_THREADS = 4
CLASS_NAMES = ["bad", "good"]


def check_data():
    metadata_path = DATA_DIR / "metadata.jsonl"

    records = [
        json.loads(line)
        for line in metadata_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    groups = defaultdict(set)
    listed = set()

    for record in records:
        if record["seed"] != SEED:
            raise ValueError(
                "Dataset seed does not match the experiment seed."
            )

        if (
            record["split"] not in ["train", "validation", "test"]
            or record["quality_label"] not in CLASS_NAMES
        ):
            raise ValueError("Invalid split or class in metadata.")

        relative = Path(
            record["image_path"].replace("\\", "/")
        )

        if relative.parts[:2] != (
            record["split"],
            record["quality_label"],
        ):
            raise ValueError(f"Wrong split path: {relative}")

        if relative.as_posix() in listed:
            raise ValueError(
                f"Duplicate image path: {relative}"
            )

        listed.add(relative.as_posix())

        if not (DATA_DIR / relative).is_file():
            raise FileNotFoundError(DATA_DIR / relative)

        key = (
            record["class_name"],
            record["source_id"],
        )
        groups[key].add(record["split"])

    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Source leakage detected.")

    extensions = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".ppm", ".tif", ".tiff",
    }

    actual = {
        path.relative_to(DATA_DIR).as_posix()
        for split in ["train", "validation", "test"]
        for label in CLASS_NAMES
        for path in (DATA_DIR / split / label).rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    }

    if actual != listed:
        raise ValueError(
            "Image folders and metadata do not match."
        )

    counts = Counter(
        record["split"] for record in records
    )

    print("Dataset counts:", dict(counts))
    print("Class mapping: bad=0, good=1")

    metadata_hash = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()

    return dict(counts), metadata_hash


def make_generator(split, shuffle):
    generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    return generator.flow_from_directory(
        str(DATA_DIR / split),
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        classes=CLASS_NAMES,
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        interpolation="bilinear",
    )


def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_plots(history, matrix):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4),
    )

    for axis, metric in zip(axes, ["loss", "accuracy"]):
        axis.plot(
            epochs,
            history[metric],
            label="Train",
        )
        axis.plot(
            epochs,
            history["val_" + metric],
            label="Validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.capitalize())
        axis.legend()
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / "curves.png",
        dpi=160,
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 4))
    plot = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(plot, ax=axis)

    axis.set_xticks([0, 1], CLASS_NAMES)
    axis.set_yticks([0, 1], CLASS_NAMES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(
        "MobileNetV2 frozen: test confusion matrix"
    )

    for row in range(2):
        for col in range(2):
            color = (
                "white"
                if matrix[row, col] > matrix.max() / 2
                else "black"
            )
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)


def measure_latency(model, paths):
    images = []

    for path in paths[:100]:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    def predict_one(image):
        resized = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(
            resized,
            dtype=np.float32,
        )
        batch = preprocess_input(array)[None, ...]
        probability = float(
            model(batch, training=False).numpy()[0, 0]
        )
        return int(probability >= 0.5)

    for index in range(20):
        predict_one(images[index % len(images)])

    timings = []

    for index in range(100):
        start = time.perf_counter()
        predict_one(images[index % len(images)])
        timings.append(
            (time.perf_counter() - start) * 1000
        )

    return float(np.mean(timings))


def main():
    global DATA_DIR, RESULT_DIR
    parser = argparse.ArgumentParser(description="Train the frozen MobileNetV2 baseline.")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=RESULT_DIR)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    DATA_DIR = args.data.resolve()
    RESULT_DIR = args.output.resolve()
    if args.check_data:
        check_data()
        print("Data check completed. Training was not started.")
        return
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(
        CPU_THREADS
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()

    counts, metadata_hash = check_data()

    train = make_generator("train", True)
    validation = make_generator("validation", False)
    test = make_generator("test", False)

    for split, generator in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if (
            generator.samples != counts.get(split, 0)
            or set(generator.classes) != {0, 1}
        ):
            raise ValueError(
                f"Invalid image counts or missing class in {split}."
            )

    if RESULT_DIR.exists() and any(RESULT_DIR.iterdir()):
        raise FileExistsError(
            f"Result folder is not empty: {RESULT_DIR}. "
            "Preserve the previous run before starting another."
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    errors_dir = RESULT_DIR / "errors"
    errors_dir.mkdir(exist_ok=True)

    model = build_model()
    model.summary()

    checkpoint = RESULT_DIR / "best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            str(RESULT_DIR / "training_log.csv")
        ),
    ]

    start = time.perf_counter()

    training = model.fit(
        train,
        validation_data=validation,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    train_time = time.perf_counter() - start

    history = {
        key: [float(value) for value in values]
        for key, values in training.history.items()
    }

    model = tf.keras.models.load_model(
        str(checkpoint),
        compile=False,
    )

    inference_path = RESULT_DIR / "inference.keras"
    model.save(str(inference_path))

    test.reset()

    probabilities = model.predict(
        test,
        verbose=1,
    ).reshape(-1)

    true_labels = np.asarray(
        test.classes,
        dtype=int,
    )
    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    save_plots(history, matrix)

    with (RESULT_DIR / "test_predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "true_label",
            "predicted_label",
            "probability_good",
        ])

        for filename, path, true, predicted, probability in zip(
            test.filenames,
            test.filepaths,
            true_labels,
            predicted_labels,
            probabilities,
        ):
            writer.writerow([
                filename,
                CLASS_NAMES[true],
                CLASS_NAMES[predicted],
                float(probability),
            ])

            if true != predicted:
                error_name = (
                    f"true_{CLASS_NAMES[true]}_"
                    f"pred_{CLASS_NAMES[predicted]}_"
                    f"{Path(path).name}"
                )
                shutil.copy2(
                    path,
                    errors_dir / error_name,
                )

    latency = measure_latency(
        model,
        test.filepaths,
    )

    results = {
        "experiment": "mobilenetv2_frozen",
        "week": 6,
        "task": 2,
        "model": "MobileNetV2",
        "regime": "frozen_base",
        "input_size": [224, 224],
        "seed": SEED,
        "device": "CPU",
        "epochs_planned": EPOCHS,
        "epochs_run": len(history["loss"]),
        "best_epoch": int(
            np.argmin(history["val_loss"]) + 1
        ),
        "train_time_s": train_time,
        "latency_ms_per_image": latency,
        "model_size_mb": (
            inference_path.stat().st_size / 1_000_000
        ),
        "checkpoint_size_mb": (
            checkpoint.stat().st_size / 1_000_000
        ),
        "model_file": inference_path.name,
        "params": int(model.count_params()),
        "test_samples": int(test.samples),
        "metrics": {
            "accuracy": float(report["accuracy"]),
            "f1_macro": float(
                report["macro avg"]["f1-score"]
            ),
            "precision_macro": float(
                report["macro avg"]["precision"]
            ),
            "recall_macro": float(
                report["macro avg"]["recall"]
            ),
            "per_class": {
                name: report[name]
                for name in CLASS_NAMES
            },
        },
        "class_indices": test.class_indices,
        "confusion_matrix": matrix.tolist(),
        "history": history,
        "split_counts": counts,
        "metadata_sha256": metadata_hash,
        "misclassified_samples": int(
            np.sum(true_labels != predicted_labels)
        ),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "patience": PATIENCE,
            "threshold": 0.5,
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "preprocessing": (
                "mobilenet_v2.preprocess_input: pixels to [-1, 1]"
            ),
        },
        "latency_protocol": {
            "warmup": 20,
            "timed_predictions": 100,
            "batch_size": 1,
            "includes": (
                "resize, preprocessing, inference, decision"
            ),
            "excludes": (
                "image decoding, disk IO, model loading, network"
            ),
        },
        "environment": {
            "python": sys.version,
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_threads": CPU_THREADS,
            "inter_op_threads": 1,
            "precision": "float32",
        },
    }

    (RESULT_DIR / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    (RESULT_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nTraining and evaluation completed.")
    print(
        f"Test accuracy: {results['metrics']['accuracy']:.4f}"
    )
    print(
        f"Test macro F1: {results['metrics']['f1_macro']:.4f}"
    )
    print(f"Bad recall: {report['bad']['recall']:.4f}")
    print(f"Latency: {latency:.2f} ms/image")
    print(
        f"Model size: {results['model_size_mb']:.2f} MB"
    )
    print(
        f"Misclassified test images: "
        f"{results['misclassified_samples']}"
    )
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
