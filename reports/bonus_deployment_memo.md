# Deployment Export and CPU Benchmark



I exported the fine-tuned EfficientNet-B0 from Task 5A to one FP32 ONNX file (opset 18, fixed input 1 x 3 x 224 x 224). The 5C augmentation did not improve the reported test decisions, so the previous prototype was retained. No model was trained. Both runtimes output P(good), with the same threshold 0.5. The separate validation-selected threshold from 5B is not applied, so this experiment isolates export/runtime effects.

## Measured results on the same test split

| Format | File size (MB) | Mean (ms/image) | Median (ms/image) | P95 (ms/image) | Accuracy | Macro F1 | Bad recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| pytorch | 16.31 | 26.93 | 26.90 | 29.70 | 0.996377 | 0.996377 | 0.992754 |
| onnx | 16.70 | 12.95 | 12.93 | 14.33 | 0.996377 | 0.996377 | 0.992754 |

Accuracy drop (PyTorch minus ONNX): +0.0000 percentage points. Changed decisions: 0/276. Maximum absolute probability difference: 4.82797623e-06. All test decisions agree.

## Timing protocol

Both engines were remeasured in this session on the same CPU, using batch 1, four intra-op threads, 20 warm-up calls and 100 timed calls per engine. Engine order alternates to reduce ordering bias. Timing includes the same full-image resize, tensor conversion, ImageNet normalization, inference and threshold decision. Decoded RGB images are preloaded; file reading, decoding, model loading, network transfer and downstream OCR are excluded. ONNX Runtime uses CPUExecutionProvider, sequential execution, graph optimizations and disabled idle thread spinning. Raw timings are saved. Values are from one session, not a confidence interval. File sizes exclude runtime libraries and are not RAM measurements.

## Server versus edge recommendation

A local CPU prototype on hardware similar to this laptop is a candidate: measured quality passes the original limits, and this measured stage fits within 52 ms. A complete end-to-end measurement is still needed. For an actual low-power edge device, measure its CPU, memory use and energy first; this laptop benchmark does not establish mobile/embedded performance. A server may be preferable for centrally managed or higher-throughput use, but no server was benchmarked here.

Compared with the handout's 52 ms quality-stage reference, measured ONNX time is 12.95 ms/image (+39.05 ms remaining before the excluded work). This is not proof of a 52 ms complete pipeline. The handout's YOLOv8 2.1 ms / 2.8 MB numbers are external capstone references, not measurements on this hardware or this EfficientNet model. Different hardware, input sizes and timing scopes prevent a direct speed ranking.

## Quantization risks and limitations

Quantization was not applied: the export remains FP32. INT8 rounding and clipping can alter logits and move borderline photos across the threshold, potentially increasing false accepts or false rejects. Any future quantization should use representative training data for calibration, validation data for operating-point decisions, and a separate final evaluation. The test split here has been reused for laboratory comparisons, and the photos are synthetic; a fresh real-photo evaluation is needed before operational deployment.

References: https://docs.pytorch.org/docs/2.15/onnx.html ; https://onnxruntime.ai/docs/performance/tune-performance/threading.html
