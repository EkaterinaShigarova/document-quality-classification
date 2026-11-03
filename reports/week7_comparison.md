# YOLOv8 Classification: Comparison with MobileNetV2

**Student:** Ekaterina Shigarova  
**Group:** AIb-24  
**Task:** 3 — YOLOv8 Classification  
**Seed:** 24500007

## 1. Experiment setup

I trained the ImageNet-pretrained YOLOv8n-cls and YOLOv8s-cls classifiers on the same synthetic document-quality dataset as my MobileNetV2 baseline.

The dataset contained 1,800 images: 1,308 for training, 216 for validation and 276 for testing. All versions of the same source document stayed in one split. The actual proportions were approximately 72.67% / 12.00% / 15.33%, produced by source-level sampling with 70/15/15 probabilities. I did not split the data again.

The metadata SHA-256 matched across all three experiments:

`db7b8e2ffd31d4991e3d6ca57b061ce517b837a98c91312b352a40e5491048de`

Both YOLO variants used these settings:

| Setting | Value |
|---|---|
| Pretrained weights | ImageNet |
| Training regime | Full fine-tuning |
| Input | RGB, 640 × 640 |
| Pixel scaling | [0, 1] |
| Optimizer | Adam |
| Nominal initial learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping patience | 5 |
| Seed | 24500007 |
| Checkpoint selection | Validation top-1 accuracy |

The actual learning-rate schedule, including warm-up, is recorded in the JSON history and Ultralytics artifacts. Additional random cropping, colour changes and erasing were disabled to preserve the quality labels. The existing generated image variants remained in the dataset.

The YOLO adapter used the folder name `val` instead of `validation`, without changing which images belonged to each split. The test set was used only for final evaluation.

All experiments ran on the same AMD Ryzen 7 5800U CPU, with four intra-op threads and one inter-op thread. YOLO used Python 3.11.9, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu and Ultralytics 8.4.201. MobileNetV2 used TensorFlow 2.21.0 and Keras 3.15.1.

## 2. Test results

All values below come from the three `results.json` files. Precision, recall and F1 are macro averages.

| Model | Input | Accuracy | Precision | Recall | F1 | Bad recall | Latency (ms/image) | File size (MB) | Parameters | Training (min) | Epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 224 × 224 | 99.64% | 0.9964 | 0.9964 | 0.9964 | 99.28% | 135.49 | 10.28 | 2,422,081 | 26.75 | 37 |
| YOLOv8n-cls | 640 × 640 | 99.28% | 0.9928 | 0.9928 | 0.9928 | 99.28% | 31.85 | 2.96 | 1,440,850 | 32.87 | 8 |
| YOLOv8s-cls | 640 × 640 | 99.64% | 0.9964 | 0.9964 | 0.9964 | 99.28% | 89.41 | 10.26 | 5,083,298 | 98.82 | 10 |

## 3. Comparison with MobileNetV2

YOLOv8n-cls achieved 99.28% test accuracy, which was 0.36 percentage points below MobileNetV2. Its latency was 103.64 ms lower, making it approximately 4.25 times as fast in these implementations. Its saved file was also 7.31 MB smaller.

YOLOv8s-cls matched MobileNetV2 in test accuracy and macro F1. Its latency was 46.08 ms lower, while its saved file was about 0.02 MB smaller. Compared with nano, small correctly classified one additional test image, but took 57.56 ms longer per image and had a 7.29 MB larger file.

Both YOLO variants passed my predefined accuracy, macro F1, bad-recall, latency and size thresholds. MobileNetV2 passed the quality and size thresholds but exceeded the latency limit of less than 100 ms per image.

The models differ in architecture, framework, input size and training regime. MobileNetV2 used a frozen feature extractor and a head consisting of global average pooling, Dense(128), Dropout and a sigmoid output. YOLO adapted its pretrained backbone and two-class classification head together.

YOLO small had approximately 5.08 million parameters, compared with 1.44 million for nano, which is consistent with its higher computation cost. Both YOLO variants processed 640 × 640 inputs instead of 224 × 224, but still had lower measured latency than the baseline.

These experiments do not isolate architecture as the only cause of the speed difference, because the frameworks and execution paths also differ. A one-image difference on 276 synthetic test images is too small to establish a general accuracy advantage.

## 4. Training curves

YOLO nano stopped after 8 epochs. According to the training log, its best checkpoint was from epoch 3. Training loss decreased rapidly, while validation loss showed some fluctuations.

YOLO small stopped after 10 epochs, with its best checkpoint from epoch 5. Training loss decreased from 0.36395 to 0.01160. Validation loss rose to 0.26508 at epoch 2, then decreased, reaching 0.01509 at epoch 8.

Small first reached a validation accuracy of 0.99537 at epoch 5 and did not exceed it later. The checkpoint was selected by validation accuracy, rather than minimum validation loss. This explains why the best checkpoint was not from epoch 8.

The curves show rapid learning and some fluctuations, without a sustained increase in validation loss. However, these results alone do not prove good performance on real photographs.

Top-5 accuracy was always 100% because the task had only two classes. Therefore, top-5 accuracy is not useful for judging model quality here.

## 5. Test errors

The confusion matrices use true classes as rows and predicted classes as columns, in the order `bad`, `good`.

| Model | Bad predicted bad | Bad predicted good | Good predicted bad | Good predicted good |
|---|---:|---:|---:|---:|
| MobileNetV2 | 137 | 1 | 0 | 138 |
| YOLOv8n-cls | 137 | 1 | 1 | 137 |
| YOLOv8s-cls | 137 | 1 | 0 | 138 |

MobileNetV2 and YOLO small each made one error: a bad image was accepted as good. Nano made two errors: one bad image was accepted and one good image was rejected.

Accepting a bad photograph can cause problems in later OCR processing. Rejecting a good photograph makes the user take another photo unnecessarily. All three models had the same bad-image recall of 99.28%.

Identical confusion matrices do not prove that MobileNetV2 and YOLO small failed on the same image. The YOLO defect types still need to be checked using their error images or `test_predictions.csv`. The hardest defect cannot be identified from these aggregate matrices alone.

## 6. Latency and model size

Latency was measured with batch size 1, after 20 warm-up predictions and over 100 timed predictions on decoded test images.

Timing included resizing, preprocessing, inference and the final class decision. It excluded image decoding, disk access, model loading and network operations. The models ran in float32.

File size was measured in decimal MB from `inference.keras` for MobileNetV2 and optimizer-stripped `best.pt` for YOLO. The file formats and stored weight representations differ, so saved file size should not be interpreted as float32 runtime memory usage.

Parameter counts describe the unfused models used for the custom test evaluation. The fused summaries printed by Ultralytics can show slightly smaller counts.

## 7. Current conclusion

Among the two YOLO variants, I currently prefer nano under my preregistered selection rule. Both variants pass all thresholds, their macro F1 differs by less than 0.01, and nano has lower latency and a smaller saved file.

Small remains a useful alternative if avoiding the additional false rejection is more important. I will make the final cross-family decision after testing EfficientNet-B0.

These results apply to the synthetic test dataset. Performance on real mobile photographs has not yet been tested.

## 8. Commands used

I ran these commands in PowerShell from the local project root, with the training script stored in `lab4`.

```powershell
.\.venv_quality\Scripts\python.exe ".\lab4\train_yolo_quality.py" --model yolov8n-cls
.\.venv_quality\Scripts\python.exe ".\lab4\train_yolo_quality.py" --model yolov8s-cls
```

## 9. Experiment artifacts

Each YOLO experiment folder contains:

- `best.pt`
- `results.json`
- `classification_report.json`
- `confusion_matrix_test.png`
- `test_predictions.csv`
- `errors/`
- `training/`

The experiment folders are:

- `results/week7_yolov8n_cls/`
- `results/week7_yolov8s_cls/`

The `training/` folders retain the Ultralytics training artifacts, including `results.csv`, `results.png`, `args.yaml` and validation plots. The separately generated `confusion_matrix_test.png` files contain the final test results.
