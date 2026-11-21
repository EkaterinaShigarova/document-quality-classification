import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from pipeline import DocumentPipeline


@pytest.fixture(scope="module")
def manifest():
    path = BASE_DIR / "samples" / "manifest.json"
    assert path.is_file(), "Run prepare_samples.py before pytest."
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["files"]) == 22
    return data


@pytest.fixture(scope="module")
def pipe():
    instance = DocumentPipeline(BASE_DIR / "config.yaml")
    assert instance.startup_error is None, instance.check_model()
    assert instance.threshold == 0.35, "Use the operating point selected in Task 5B."
    return instance


def sample_path(row):
    return BASE_DIR / "samples" / row["file"]


def check_schema(result):
    assert set(result) == {
        "file", "status", "quality", "quality_confidence", "document_type",
        "document_type_confidence", "reason", "model_versions", "latency_ms", "timestamp",
    }
    assert isinstance(result["file"], str)
    assert result["status"] in {"accepted", "rejected", "error"}
    assert result["document_type"] is None
    assert result["document_type_confidence"] is None
    assert set(result["model_versions"]) == {"quality", "doctype"}
    assert all(isinstance(value, str) for value in result["model_versions"].values())
    assert result["model_versions"]["doctype"] == "deferred_next_semester"
    assert isinstance(result["latency_ms"], (int, float)) and result["latency_ms"] >= 0
    assert datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00")).tzinfo is not None
    if result["status"] == "error":
        assert result["quality"] is None and result["quality_confidence"] is None
        assert isinstance(result["reason"], str)
    else:
        assert result["quality"] in {"good", "bad"}
        assert isinstance(result["quality_confidence"], float)
        assert 0 <= result["quality_confidence"] <= 1
    json.dumps(result, allow_nan=False)


def test_json_schema(pipe, manifest):
    for status in ["accepted", "rejected", "error"]:
        row = next(row for row in manifest["files"] if row["expected_status"] == status)
        result = pipe.run(sample_path(row))
        check_schema(result)
        assert result["status"] == status


def test_good_photo_is_accepted(pipe, manifest):
    row = next(row for row in manifest["files"] if row["true_quality"] == "good")
    result = pipe.run(sample_path(row))
    assert (result["status"], result["quality"], result["reason"]) == ("accepted", "good", None)
    assert result["document_type"] is None


def test_glare_photo_is_rejected(pipe, manifest):
    row = next(row for row in manifest["files"] if row.get("boundary"))
    result = pipe.run(sample_path(row))
    assert (result["status"], result["quality"], result["reason"]) == ("rejected", "bad", "poor_quality")
    assert result["document_type"] is None


def test_corrupted_image_returns_error(pipe):
    result = pipe.run(BASE_DIR / "samples" / "corrupted.png")
    check_schema(result)
    assert result["status"] == "error"
    assert "Traceback" not in json.dumps(result)


def test_renamed_text_returns_error(pipe):
    result = pipe.run(BASE_DIR / "samples" / "non_image.jpg")
    check_schema(result)
    assert (result["status"], result["reason"]) == ("error", "unsupported_file")


def test_cpu_latency_on_twenty_images(pipe, manifest):
    rows = [row for row in manifest["files"] if row["true_quality"] is not None]
    assert len(rows) == 20
    for row in rows:
        assert pipe.run(sample_path(row))["status"] != "error"
    measurements = []
    for row in rows:
        started = time.perf_counter()
        result = pipe.run(sample_path(row))
        measurements.append((time.perf_counter() - started) * 1000)
        assert result["status"] != "error"
    mean_ms = float(np.mean(measurements))
    report = {
        "status": "passed" if mean_ms < 500 else "failed",
        "mean_ms": mean_ms,
        "median_ms": float(np.median(measurements)),
        "p95_ms": float(np.percentile(measurements, 95)),
        "budget_ms": 500,
        "warmup_images": 20,
        "timed_images": 20,
        "batch_size": 1,
        "provider": "CPUExecutionProvider",
        "includes": "file reading, image decoding, resize, normalization, inference and JSON-ready result creation",
        "excludes": "startup model loading, UI/network transfer and document classification deferred by the instructor",
        "files": [row["file"] for row in rows],
        "measurements_ms": measurements,
    }
    (BASE_DIR / "latency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CPU latency: {mean_ms:.2f} ms/image over 20 samples; limit < 500 ms.")
    assert mean_ms < 500, f"Mean latency {mean_ms:.2f} ms exceeds the CPU budget."


def test_threshold_changes_boundary_decision(pipe, manifest, tmp_path):
    row = next(row for row in manifest["files"] if row.get("boundary"))
    first = pipe.run(sample_path(row))
    config = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8-sig"))
    config["quality"]["chosen_threshold"] = 0.5
    config["quality"]["model_path"] = os.path.relpath(pipe.model_path, tmp_path)
    path = tmp_path / "changed_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    changed = DocumentPipeline(path)
    assert changed.startup_error is None, changed.check_model()
    second = changed.run(sample_path(row))
    assert first["status"] == "rejected"
    assert second["status"] == "accepted"


def test_batch_integration_saves_predictions(pipe, manifest):
    results = []
    for row in manifest["files"]:
        result = pipe.run(sample_path(row))
        check_schema(result)
        results.append(result)
    counts = dict(Counter(result["status"] for result in results))
    payload = {
        "scope": "quality-only pipeline per instructor clarification",
        "sample_origin": "fixed validation examples plus two invalid files",
        "chosen_threshold": pipe.threshold,
        "threshold_probability": "P(bad)",
        "counts": counts,
        "expected_counts": manifest["expected_counts"],
        "predictions": results,
    }
    destination = BASE_DIR / "predictions.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Batch counts: {counts}; saved predictions.json.")
    assert counts == manifest["expected_counts"]
    for row, result in zip(manifest["files"], results):
        assert result["status"] == row["expected_status"], row["file"]


def test_missing_model_returns_error(manifest, tmp_path):
    config = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8-sig"))
    config["quality"]["model_path"] = "models/missing.onnx"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    instance = DocumentPipeline(path)
    result = instance.run(sample_path(manifest["files"][0]))
    check_schema(result)
    assert (result["status"], result["reason"]) == ("error", "model_unavailable")


def test_requests_reuse_loaded_model(pipe, manifest, monkeypatch):
    import onnxruntime as ort

    original = pipe.session
    def fail_reload(*args, **kwargs):
        raise AssertionError("A request tried to reload the model.")
    monkeypatch.setattr(ort, "InferenceSession", fail_reload)
    row = next(row for row in manifest["files"] if row["true_quality"] == "good")
    for _ in range(3):
        assert pipe.run(sample_path(row))["status"] == "accepted"
        assert pipe.session is original
