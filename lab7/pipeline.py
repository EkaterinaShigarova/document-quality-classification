import argparse
import io
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent


class DocumentPipeline:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path or BASE_DIR / "config.yaml").resolve()
        self.session = None
        self.startup_error = "config_error"
        self.model_version = "unavailable"
        self.doctype_version = "deferred_next_semester"
        self.threshold = None
        try:
            self._load_config()
        except Exception:
            return
        self.startup_error = "model_unavailable"
        try:
            self._load_model()
        except Exception:
            return
        self.startup_error = None

    def _load_config(self):
        with self.config_path.open(encoding="utf-8-sig") as file:
            config = yaml.safe_load(file)
        quality = config["quality"]
        self.model_version = str(quality["model_version"])
        value = quality["chosen_threshold"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Threshold must be a number.")
        self.threshold = float(value)
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between zero and one.")
        if quality["threshold_probability"] != "bad":
            raise ValueError("The selected threshold applies to P(bad).")
        if quality["input_size"] != [224, 224]:
            raise ValueError("This model requires a 224 x 224 image.")
        if quality["mean"] != [0.485, 0.456, 0.406] or quality["std"] != [0.229, 0.224, 0.225]:
            raise ValueError("Preprocessing must match the exported model.")
        self.mean = np.array(quality["mean"], dtype=np.float32)[:, None, None]
        self.std = np.array(quality["std"], dtype=np.float32)[:, None, None]
        relative = str(quality["model_path"])
        path = Path(relative)
        if path.is_absolute() or PureWindowsPath(relative).is_absolute() or PureWindowsPath(relative).drive:
            raise ValueError("Use a relative model path.")
        self.model_path = (self.config_path.parent / path).resolve()
        runtime = config["runtime"]
        self.cpu_threads = runtime["cpu_threads"]
        self.inter_op_threads = runtime["inter_op_threads"]
        upload = config["upload"]
        self.max_bytes = upload["max_bytes"]
        self.max_pixels = upload["max_pixels"]
        for value in [self.cpu_threads, self.inter_op_threads, self.max_bytes, self.max_pixels]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Thread and upload limits must be positive integers.")
        self.allowed_formats = set(upload["allowed_formats"])
        if not self.allowed_formats or not self.allowed_formats.issubset({"JPEG", "PNG"}):
            raise ValueError("Supported image formats are JPEG and PNG.")
        if config["document_classifier"]["enabled"] is not False:
            raise ValueError("Document classification is deferred by the instructor.")

    def _load_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError("The ONNX model is missing.")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.cpu_threads
        options.inter_op_num_threads = self.inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 224, 224] or inputs[0].type != "tensor(float)":
            raise ValueError("Unexpected model input contract.")
        if len(outputs) != 1 or outputs[0].name != "probability_good" or outputs[0].shape != [1]:
            raise ValueError("Unexpected model output contract.")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.session = session

    def _result(self, filename, started, status="error", reason=None, quality=None, confidence=None):
        return {
            "file": str(filename).replace("\\", "/").rsplit("/", 1)[-1],
            "status": status,
            "quality": quality,
            "quality_confidence": confidence,
            "document_type": None,
            "document_type_confidence": None,
            "reason": reason,
            "model_versions": {"quality": self.model_version, "doctype": self.doctype_version},
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _probability(self, batch):
        values = self.session.run([self.output_name], {self.input_name: batch})
        output = np.asarray(values[0])
        if output.size != 1:
            raise ValueError("The model must return one probability.")
        probability = float(output.reshape(-1)[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The model returned an invalid probability.")
        return probability

    def _process(self, data, filename, started):
        if self.startup_error:
            return self._result(filename, started, reason=self.startup_error)
        if not isinstance(data, (bytes, bytearray, memoryview)) or not len(data):
            return self._result(filename, started, reason="unsupported_file")
        if len(data) > self.max_bytes:
            return self._result(filename, started, reason="image_too_large")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in self.allowed_formats:
                        return self._result(filename, started, reason="unsupported_file")
                    if image.width * image.height > self.max_pixels:
                        return self._result(filename, started, reason="image_too_large")
                    image.load()
                    rgb = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    pixels = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
                    tensor = pixels.transpose(2, 0, 1)
                    batch = np.ascontiguousarray(((tensor - self.mean) / self.std)[None], dtype=np.float32)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            return self._result(filename, started, reason="image_too_large")
        except UnidentifiedImageError:
            return self._result(filename, started, reason="unsupported_file")
        except Exception:
            return self._result(filename, started, reason="corrupted_image")
        try:
            probability_good = self._probability(batch)
            probability_bad = 1.0 - probability_good
            rejected = probability_bad > self.threshold
            return self._result(
                filename, started,
                status="rejected" if rejected else "accepted",
                reason="poor_quality" if rejected else None,
                quality="bad" if rejected else "good",
                confidence=float(probability_bad if rejected else probability_good),
            )
        except Exception:
            return self._result(filename, started, reason="inference_error")

    def run_bytes(self, data, filename="upload.jpg"):
        started = time.perf_counter()
        return self._process(data, filename, started)

    def run(self, image_path):
        started = time.perf_counter()
        try:
            path = Path(image_path)
        except Exception:
            return self._result("unknown", started, reason="unsupported_file")
        if self.startup_error:
            return self._result(path.name, started, reason=self.startup_error)
        try:
            with path.open("rb") as file:
                data = file.read(self.max_bytes + 1)
        except FileNotFoundError:
            return self._result(path.name, started, reason="missing_file")
        except Exception:
            return self._result(path.name, started, reason="unreadable_file")
        return self._process(data, path.name, started)

    def check_model(self):
        started = time.perf_counter()
        if self.startup_error:
            return self._result("", started, reason=self.startup_error)
        try:
            self._probability(np.zeros((1, 3, 224, 224), dtype=np.float32))
        except Exception:
            return self._result("", started, reason="inference_error")
        return {
            "status": "ready",
            "model_version": self.model_version,
            "provider": "CPUExecutionProvider",
            "input_shape": [1, 3, 224, 224],
            "output_probability": "P(good)",
            "chosen_threshold": self.threshold,
            "decision_rule": "reject if 1 - P(good) > chosen_threshold",
            "document_classifier": "deferred_next_semester",
        }


def main():
    parser = argparse.ArgumentParser(description="Document photo quality inference on CPU.")
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.yaml")
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    if not args.check_model and args.image is None:
        parser.error("Provide an image path or use --check-model.")
    pipeline = DocumentPipeline(args.config)
    result = pipeline.check_model() if args.check_model else pipeline.run(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(1 if result["status"] == "error" else 0)


if __name__ == "__main__":
    main()
