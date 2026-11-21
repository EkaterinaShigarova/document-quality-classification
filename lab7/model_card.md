# Model Card — Deployed Document Quality Gate

**Author:** Ekaterina Shigarova, IIKb-24  
**Updated:** 21 November 2026  
**Website:** https://document-quality-classification.onrender.com

## Model and intended use

EfficientNet-B0 with a binary classification head, partially fine-tuned from the frozen Lab 6 model. The selected checkpoint is epoch **28**; the final 30 parameterized leaf modules of the feature extractor and the classification head were trainable. There are **4,008,829 total parameters** and **2,639,937 trainable parameters** in that training phase.

The deployed FP32 ONNX model is `models/model.onnx`, **16,702,506 bytes** (16.70 decimal MB). Version: `efficientnetb0_finetune_top30_onnx_fp32_epoch28`. Inference uses ONNX Runtime on CPU. It estimates good/bad image quality before document processing; it does not verify identity, document authenticity or OCR correctness.

The instructor deferred INN/SNILS/PASSPORT classification to the next semester. `document_type` and its confidence remain null; doctype version is `deferred_next_semester`.

## Data and preprocessing

Training/evaluation used **1,800 synthetic 640 × 640 document images** covering INN, SNILS and passport documents. The seed is **24500007**. Source groups `(class_name, source_id)` were kept across only one split.

| Split | Bad | Good | Total |
| --- | ---: | ---: | ---: |
| Train | 654 | 654 | 1,308 |
| Validation | 108 | 108 | 216 |
| Test | 138 | 138 | 276 |

Input is decoded into RGB, resized to 224 × 224 with PIL bilinear interpolation, scaled by 1/255 and normalized using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Input tensor: float32 `[1, 3, 224, 224]`. The output is `P(good)`.

## Threshold and metric provenance

The deployed rule is **reject if `1 - P(good) > 0.35`**. Task 5B selected 0.35 on validation only, with assumed relative costs of 5 for a false accept and 1 for a false reject. At 0.35, validation false accepts/rejects were 0/0; at 0.50, they were 1/0. These costs are an experiment assumption, not measured financial losses.

The following **historical test results use threshold 0.50**, from Tasks 5A/5D:

| Metric on 276 synthetic test images | Result |
| --- | ---: |
| Accuracy | 0.996377 |
| Macro precision | 0.996403 |
| Macro recall | 0.996377 |
| Macro F1 | 0.996377 |
| Bad recall | 0.992754 |
| False accepts / false rejects | 1 / 0 |

Confusion matrix: `[[137, 1], [0, 138]]`, rows=true, columns=predicted, label order bad/good. ONNX and PyTorch decisions agreed on all 276 images at that threshold. See [fine-tuning results](../results/efficientnetb0_finetune_top30/results.json), [export report](../reports/bonus_deployment_memo.md) and [threshold report](../reports/bonus_threshold_memo.md).

**A full test-set evaluation at the deployed 0.35 threshold has not been performed.** Do not label the historical 99.64% test accuracy as a verified result for that operating point. Lab 7's 20 validation-derived fixtures, all INN examples, produced 10 accepted/10 rejected plus two errors for invalid files. That checks integration, not generalization accuracy.

## Latency and deployment evidence

| Measurement | Result and scope |
| --- | --- |
| Lab 7 laptop CPU | Mean 19.11 ms; median 18.82 ms; P95 21.84 ms |
| Protocol | 20 warm-up images, 20 timed images, batch 1, AMD Ryzen 7 5800U, Windows/Python 3.11.9 |
| Included | File reading, image decoding, preprocessing, inference and JSON-ready result creation |
| Excluded | Startup model load, network/UI transfer and deferred document classification |
| Required local mean budget | <500 ms; passed |
| Public Render smoke checks | Accepted, rejected and corrupted-image responses observed |

The local test suite passed **10 tests in 2.34 seconds**. Evidence: [pytest output](pytest_output.txt), [latency measurements](latency_report.json) and [batch predictions](predictions.json). Individual hosted requests of 176.5 ms and 357.4 ms do not establish hosted mean latency. Free-instance cold starts can add delay before the app processes a request.

## Limitations, ethics and next evaluation

All quality metrics come from synthetic data. The assignment cites a **2.8–5.8% synthetic-to-real gap** in the capstone reference; that range is external context, not a measured gap, correction factor or guarantee for this model. No representative real-photo holdout has been evaluated. Repeated laboratory test comparisons and error inspection informed the augmentation experiment, limiting independence of the reported test evidence.

Confidence is uncalibrated. The UI rounds percentages; 100.0% is not certainty. The model does not independently validate that an arbitrary uploaded image is a document, explain the specific defect, or establish identity/authenticity. Strong glare and unfamiliar real-camera conditions may cause errors. False accepts can pass unreadable images downstream, while false rejects create unnecessary retakes.

The app processes uploads in memory without intentionally saving photo files. Use the supplied synthetic samples for public demos; do not publish real personal documents without permission. Before operational use, evaluate a consented real-photo holdout with source separation and per-document/per-defect reporting, calibrate confidence, and reassess the threshold against the actual cost of both error types. Add human review for ambiguous cases.
