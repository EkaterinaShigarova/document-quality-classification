


Document Quality Classification
Ekaterina Shigarova, IIKb-24. Machine Learning Lab 6.

This project classifies synthetic document photographs as bad or good before further document processing. It compares MobileNetV2, YOLOv8n-cls, YOLOv8s-cls and EfficientNet-B0, followed by fine-tuning, threshold selection, glare augmentation and ONNX export.

Repository contents
scripts: data preparation, training, evaluation and export code.
results: saved checkpoints, metrics, predictions and plots.
reports: experiment reports and conclusions.
Data access: required before running
The complete dataset is not included in this repository, and no public dataset download link is currently provided. Request the original quality_split dataset, including its unchanged metadata.jsonl, from the repository author before attempting reproduction. A repository download alone is not sufficient to run these experiments.

Place that directory at data/quality_split under the repository root. It must contain metadata.jsonl and the six image directories below:

Directory	bad	good	Total
data/quality_split/train	654	654	1308
data/quality_split/validation	108	108	216
data/quality_split/test	138	138	276
Each split contains bad and good subdirectories. Image paths in the metadata are relative to data/quality_split, for example test/bad/<original-image-name>.jpg.

The dataset contains 1800 synthetic 640 x 640 images of INN, SNILS and passport documents. The seed is 24500007. Source documents are grouped by (class_name, source_id) so that versions of the same source do not cross splits. Use the recorded split; do not randomly split the image files again.

Metadata is one JSON object per line. Fields used by the experiments include image_path, split, quality_label, class_name, source_id, seed and defect_type. Keep all original fields and original file bytes.

Expected SHA-256 of metadata.jsonl:

db7b8e2ffd31d4991e3d6ca57b061ce517b837a98c91312b352a40e5491048de
The scripts check metadata consistency against the saved experiment results. A different dataset or regenerated metadata is not an exact reproduction. In PowerShell, check the supplied file with:

Get-FileHash ".\data\quality_split\metadata.jsonl" -Algorithm SHA256
Environment and installation
The recorded runs used Windows, Python 3.11.9 and an AMD Ryzen 7 5800U CPU. GPU access is not required.

Clone this repository, or use GitHub Code > Download ZIP and extract it. Open a PowerShell terminal in the directory containing this README, scripts, results and reports.

If Git is installed:

git clone https://github.com/EkaterinaShigarova/document-quality-classification.git
cd document-quality-classification
Create a dedicated environment. Run each command separately:

py -3.11 -m venv .venv_quality
.\.venv_quality\Scripts\python.exe -m pip install --upgrade pip
.\.venv_quality\Scripts\python.exe -m pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv_quality\Scripts\python.exe -m pip install numpy==2.4.6 Pillow matplotlib scikit-learn
.\.venv_quality\Scripts\python.exe -m pip install onnx==1.23.2 onnxruntime==1.32.0 onnxscript==0.7.2
These commands cover the EfficientNet experiments and ONNX reproduction below. The saved source runs used PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu; fine-tuning and export intentionally reject other versions. The commands pin recorded core versions, but are not a complete environment lock: Pillow, matplotlib, scikit-learn and transitive packages are not fully pinned.

MobileNetV2 additionally requires TensorFlow; its recorded baseline used tensorflow==2.21.0. YOLO additionally requires Ultralytics; the recorded runs used ultralytics==8.4.201. Those packages are not needed for the ONNX experiment below.

Reproduce one experiment: Task 5D ONNX export
This is a complete export/evaluation experiment using the saved Task 5A checkpoint. It does not train the model again.

Required repository files:

Task 5A checkpoint
Task 5A source results
Export script
The checkpoint is from epoch 28. Its SHA-256 is:

4e0cf19e42f570cdbdf018e90ad1d9ee06136133fa330693ae927cbfeea51746
After installing dependencies and obtaining the original dataset, first run:

.\.venv_quality\Scripts\python.exe ".\scripts\export_efficientnetb0_onnx.py" --check-data
A successful check reports epoch 28, 276 test images and that export and benchmarking were not started. This check alone is not a reproduced experiment.

Then run the actual experiment:

.\.venv_quality\Scripts\python.exe ".\scripts\export_efficientnetb0_onnx.py" --output ".\results\reproduction_onnx" --report ".\reports\reproduction_deployment_memo.md"
These custom output paths preserve the original submitted results and report. They must not already exist. For a further run, choose a new output directory and report filename.

The script exports one FP32 ONNX file, evaluates PyTorch and ONNX on the same 276 test images, and benchmarks both engines with 20 warm-up and 100 timed calls per engine. It uses batch 1, four intra-op threads, one inter-op thread and alternating engine order.

The new output directory contains model.onnx, results.json, benchmark.csv, test_predictions_export.csv and latency_measurements.csv. The deployment memo is written to the custom report path above.

Recorded reference results
Format	Accuracy	Macro F1	Bad recall	Size (MB)	Mean CPU time (ms/image)
PyTorch, remeasured in Task 5D	0.996377	0.996377	0.992754	16.31	26.93
ONNX FP32	0.996377	0.996377	0.992754	16.70	12.95
The recorded accuracy drop was zero, with 0/276 changed decisions. Both confusion matrices were [[137, 1], [0, 138]], with rows = true labels and columns = predicted labels in the order bad, good.

Latency includes resize, tensor conversion, normalization, inference and thresholding. It excludes disk reading, image decoding, model loading, networking and downstream OCR. These are laptop CPU measurements of one stage, not a complete document-processing pipeline. Timing will vary with hardware, background load and dependency versions.

Both runtimes output P(good); good is selected when P(good) >= 0.5. The validation-selected P(bad) threshold of 0.35 from Task 5B is not used in the export comparison.

Reproduce training: frozen EfficientNet-B0
This run starts from ImageNet-pretrained weights and trains the binary head. It requires the complete original dataset, the dependencies above and the included MobileNetV2 baseline results for split verification. Initial ImageNet weight download requires internet access.

Check the data first:

.\.venv_quality\Scripts\python.exe ".\scripts\train_efficientnetb0.py" --check-data
Run training and test evaluation into a new directory:

.\.venv_quality\Scripts\python.exe ".\scripts\train_efficientnetb0.py" --output ".\results\reproduction_efficientnetb0_frozen"
Settings: seed 24500007, CPU, 224 x 224 RGB input, ImageNet normalization, batch 32, Adam learning rate 0.001, maximum 50 epochs, patience 5 and minimum validation loss checkpoint selection. Only the classification head is trainable. Test decisions use P(good) >= 0.5.

The original frozen run completed 50 epochs, selected epoch 50 and achieved test accuracy 0.992754, macro F1 0.992753 and confusion matrix [[136, 2], [0, 138]]. Its recorded training time was approximately 61 minutes on the original CPU.

The new directory contains the best checkpoint, training log, curves, classification report, test predictions, confusion matrix and misclassified images. Matching the seed and split improves reproducibility but does not guarantee identical floating-point results or runtime on a different setup.

Other experiments and reports
Task	Code	Report
1: model selection	—	
Model selection memo
2: MobileNetV2 baseline	
Training script
Baseline
3: YOLO nano/small	
Training script
YOLO comparison
4: additional architecture and analysis	
EfficientNet script
Comparison, decision memo, error analysis
5A: partial fine-tuning	
Fine-tuning script
Fine-tuning delta
5B: validation-only threshold selection	
Threshold script
Threshold memo
5C: extra training glare augmentation	
Augmentation script
Augmentation memo
5D: ONNX export and CPU benchmark	
Export script
Deployment memo
Limitations and responsible use
All reported quality results use synthetic document images. The test split has been reused for laboratory comparisons, and error inspection informed the glare augmentation experiment. These results do not establish performance on an untouched real-photo holdout.

The classifier checks image quality, not document authenticity, identity or OCR correctness. Do not use it as an identity-verification decision. Evaluate real photographs, ambiguous labels, false accepts and false rejects before operational use. Real personal documents require permission and appropriate privacy protection; do not publish them in this repository.





