# Model Selection Memo

**Student:** Ekaterina Shigarova  
**Group:** AIb-24  
**Task:** 1 — Quality Classification Model  
**Date:** 2 November 2026  
**Seed:** 24500007

## 1. Purpose and pipeline context

The goal is to choose a model that checks whether a photograph of a Russian personal document is suitable for further processing. The two quality classes are `good` and `bad`. The model should detect problems such as blur, glare, strong skew, occlusion and poor exposure.

In the Telegram bot pipeline, the quality check is the first processing gate after image upload. It comes before document type classification, perspective correction, OCR and JSON extraction. A rejected photograph should lead to a request to take another photo. A suitable photograph can continue through the pipeline.

Speed matters because the user is waiting for feedback. Correct rejection also matters: accepting a bad photo can cause OCR errors, while rejecting a good photo makes the user repeat an unnecessary step. This memo records my model choices and acceptance criteria before training.

## 2. Dataset and project constraints

My dataset contains 1,800 synthetic images: 900 good and 900 bad. It includes INN, SNILS and passport images, with 300 good and 300 bad images for each document type. These images were created from 300 source documents (100 per type), with three good and three bad versions of each source. They are augmented versions, not 1,800 independent source documents. Each bad-image defect type (blur, glare, skew, occlusion and exposure) has 180 images. All 1,800 metadata records use seed 24500007.

The generator assigns each source document to train, validation or test using probabilities of 70%, 15% and 15%. The actual metadata counts are 1,308 training images (654 good and 654 bad), 216 validation images (108 good and 108 bad), and 276 test images (138 good and 138 bad). This is approximately 72.67% / 12.00% / 15.33%, rather than an exact 70/15/15 count. A metadata check found no source document appearing in more than one split and no repeated image IDs or image paths. All models will use this same recorded split and the same test images. Image file availability and loading will be checked before training.

| Constraint | Effect on my model choice |
|---|---|
| The quality check is the first pipeline gate | Prediction must be fast enough to provide feedback without a long delay. |
| The dataset is small and synthetic | I will use ImageNet-pretrained weights. A large model trained from scratch could memorise the training images. |
| CPU server deployment first | I will measure CPU latency on the same machine for every model. GPU benchmarks from papers cannot replace this measurement. |
| Possible mobile deployment later | A compact model is preferable because storage and memory are limited. Export can be investigated after the main comparison. |

## 3. Candidate model comparison

The parameter counts below describe the standard ImageNet models. Replacing their original classifiers with a two-class head changes the counts, so I will also report the actual parameters of my trained models. Published ImageNet accuracy is not the expected accuracy on my document dataset.

| Model | Family and main feature | Reference input | Input for this work | Approximate parameters | Strength for this project | Main project risk |
|---|---|---|---|---|---|---|
| MobileNetV2 | CNN with depthwise convolutions, inverted residual blocks and linear bottlenecks [1] | 224 × 224 | 224 × 224 | 3.5 million [4] | Lightweight baseline with a clear Keras implementation | Resizing to 224 × 224 may remove small defects; frozen features may not fully represent document quality. |
| YOLOv8n-cls | Compact CNN for whole-image classification, nano variant [3] | 224 × 224 | 640 × 640 | 2.7 million [3] | Small candidate for a fast quality gate | The larger course input increases computation; speed at 224 × 224 does not prove speed at 640 × 640. |
| YOLOv8s-cls | CNN classifier, larger small variant of the same family [3] | 224 × 224 | 640 × 640 | 6.4 million [3] | Tests whether extra capacity improves the nano result | More parameters may increase latency and overfitting without improving quality detection. |
| EfficientNet-B0 | CNN using mobile inverted bottleneck blocks; the EfficientNet family balances depth, width and resolution [2] | 224 × 224 | 224 × 224 | 5.3 million [6] | Additional architecture with a moderate parameter count | Wrong input normalization can damage transfer learning; useful features may still need fine-tuning. |

YOLOv8-cls assigns a label to the whole photograph. Object detection models and bounding-box annotations are not needed for this task. The 640 × 640 YOLO input follows the course requirements, although the published classification reference uses 224 × 224 [3, 7].

I considered ResNet50, but did not include it in the training shortlist. Keras lists its standard model size as about 98 MB, compared with 14 MB for MobileNetV2 and 29 MB for EfficientNet-B0 [4]. These are reference file sizes, not measurements of my future checkpoints. ResNet50 is a less convenient starting choice for my 50 MB size limit.

## 4. Transfer learning and training regimes

Transfer learning starts with a model already trained on a large dataset. Its base has learned image features, including edges, textures and shapes. I replace the original ImageNet classifier with a head for document quality [5].

**Frozen base:** the pretrained base weights stay unchanged. Only the new classification head learns from my images. It learns how to combine existing features to distinguish `good` from `bad`. The base does not learn new document-specific filters in this regime. This reduces the number of trainable parameters and the risk of overfitting.

**Full fine-tuning:** after training the head, the base is unfrozen and its trainable weights can also adapt to the document images. I would use a lower learning rate, such as 0.00001, to avoid large changes to useful pretrained weights. This may improve accuracy, but takes more computation and can overfit [5]. Training only the last few base layers is an intermediate option, rather than a fully frozen base.

## 5. Selected models and experiment plan

I selected MobileNetV2, YOLOv8-cls and EfficientNet-B0. MobileNetV2 provides the required lightweight baseline. YOLOv8n-cls and YOLOv8s-cls allow me to compare two sizes within one family. EfficientNet-B0 adds a different architecture while remaining a reasonable candidate for the size limit. This gives three architecture families and four main training runs. I will choose the final model from measured results, rather than assuming that one architecture will win.

| Setting | Planned value |
|---|---|
| Seed | 24500007 for all experiments |
| Maximum epochs | 50 |
| Early stopping patience | 5 |
| Batch size | 32; any necessary reduction will be documented |
| Data | Same source-based split for every model |
| MobileNetV2 | Frozen ImageNet base; global average pooling, Dense(128, ReLU), Dropout(0.3), Dense(1, sigmoid) |
| MobileNetV2 optimization | Adam, learning rate 0.001, binary cross-entropy |
| YOLOv8 | Pretrained nano and small classifiers, each adapted to two quality classes |
| EfficientNet-B0 | Pretrained torchvision base with a new binary head; frozen-base run first |

MobileNetV2 will use its TensorFlow-style preprocessing. The torchvision EfficientNet-B0 will use pixels scaled to [0, 1], followed by ImageNet mean/std normalization [6]. YOLO will use the preprocessing required by its classifier. These implementations need different numerical preprocessing even when they receive the same photographs.

## 6. Acceptance criteria

I set the following thresholds before running the classification experiments. A model must meet all of them to pass my initial synthetic-data screening.

| Measure | Acceptance threshold | Reason |
|---|---|---|
| Test accuracy | At least 0.92 | Leaves a margin for lower performance on real photographs. |
| Test macro F1 | At least 0.90 | Requires useful performance across both quality classes. |
| Test recall for `bad` | At least 0.90 | Limits the proportion of unsuitable photos that pass the gate. |
| Mean CPU latency | Less than 100 ms per image | Keeps the quality model within the first-gate processing budget. |
| Saved model size | At most 50 MB | Supports a compact server model and possible later mobile use. |

The course materials report a synthetic-to-real accuracy gap of 2.8–5.8% [7]. I therefore use approximately six percentage points as a conservative planning allowance: 92% synthetic accuracy could become about 86% real accuracy under that assumption. This is an illustration, not a measured result or a guarantee. Passing the synthetic criteria does not establish readiness for real deployment; that needs a separate real-photo evaluation.

## 7. Evaluation and final selection

The validation set will guide early stopping and checkpoint selection. The test set will only be used for final evaluation, not for training, choosing hyperparameters or tuning thresholds. For the binary baseline, I will use the required threshold of 0.5.

For each run, I will save accuracy, precision, recall and F1 for each class and their macro averages, together with a confusion matrix. I will also save training curves, actual epochs, training time, parameter count and misclassified images. Results will be stored in `results.json` using the course schema.

For latency, I will load the model once, use batch size 1, perform 20 warm-up predictions, then average 100 timed predictions on test images. If there are fewer than 100 test images, I will cycle through them for timing only. The timed interval will include model-specific preprocessing, inference and the quality decision, but exclude disk loading, model loading and Telegram communication. Hardware, software versions and CPU thread settings will be recorded and kept consistent.

Model size will be measured from the saved inference-ready file, excluding optimizer state. I will record its format and numerical precision, with MB defined as 1,000,000 bytes. Full training checkpoints will also be retained.

I will inspect mistakes by defect type, especially glare. Some generated defects may leave text readable, so label quality is also a possible cause of errors. Among models meeting all criteria, I will prefer the one with the highest macro F1. If macro F1 differs by no more than 0.01, I will prefer lower latency, then smaller size. Small differences on this dataset will be treated cautiously. If none passes, I will report the failed criteria without changing the original thresholds after seeing the results.

## 8. References

[1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR, pp. 4510–4520. https://arxiv.org/abs/1801.04381

[2] Tan, M., and Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML, PMLR 97, pp. 6105–6114. https://proceedings.mlr.press/v97/tan19a.html

[3] Ultralytics. *YOLOv8 documentation*, classification model comparison. https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md

[4] Keras. *Keras Applications*, model comparison table. https://keras.io/api/applications/

[5] Chollet, F. *Transfer learning and fine-tuning*. Keras guide. https://keras.io/guides/transfer_learning/

[6] PyTorch. *torchvision.models.efficientnet_b0*, pretrained weights and preprocessing. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html

[7] Course materials: *Task 1 — Model Selection Memo*; *Requirements for Task 2, 3 and 4*; *Lecture 4 — Quality Classification Model: Training and Model Comparison*. Provided for the course.

Online references accessed on 2 November 2026.
