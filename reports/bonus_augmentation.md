# Error-Driven Glare Augmentation



## Evidence and protocol

Task 4 identified glare as the dominant defect: all five false-accept events across its four models involved glare. The Task 5A reference has one false accept, also glare. These are measured results from this project; the handout's capstone F1 = 0.92 is not a result of this binary classifier.

I repeated Task 5A from the exact same frozen checkpoint, with a fresh Adam optimizer and the same seed, last 30 parameterized base layers plus head, learning rate 1e-5, batch 32, maximum 50 epochs, patience 5, 224 x 224 preprocessing and minimum-validation-loss checkpoint selection. The base remains in eval mode with fixed BatchNorm running statistics; classifier dropout is active in training. Both runs use the original P(good) >= 0.5 decision. The Task 5B threshold is not applied, so it does not confound this comparison.

Only existing train/bad/glare images receive extra white ellipses, with probability 0.5 per image per epoch. Each overlay has 1-2 ellipses with alpha 150-230/255 and softly blurred edges. Positions and sizes vary as recorded in results.json. A local deterministic RNG depends on seed, epoch and image path, preserving model and shuffle RNG state. Good labels and other defect subtypes remain unchanged. Augmentation occurs in memory before resizing; no source images are overwritten, no extra validation/test images are generated, and their byte manifests are checked before and after training.

## Per-class comparison on the unchanged test split

| Class | Support | Precision before | Precision after | Recall before | Recall after | F1 before | F1 after | F1 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bad | 138 | 1.000000 | 1.000000 | 0.992754 | 0.992754 | 0.996364 | 0.996364 | +0.000000 |
| good | 138 | 0.992806 | 0.992806 | 1.000000 | 1.000000 | 0.996390 | 0.996390 | +0.000000 |

## Defect-specific binary comparisons

Glare, blur, skew, occlusion and exposure are metadata subgroups of bad, not separate model outputs. Each row below evaluates bad-class F1 on that defect's bad images plus ALL good test images. The shared good images supply false positives for precision. These overlapping subsets are not independent and must not be summed. Good means suitable in this dataset. This is not multiclass defect-identification F1.

| Defect vs good | Bad support | Good support | Bad F1 before | Bad F1 after | F1 delta | Bad recall before | Bad recall after | Missed bad before | Missed bad after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blur | 29 | 138 | 1.000000 | 1.000000 | +0.000000 | 1.000000 | 1.000000 | 0 | 0 |
| exposure | 29 | 138 | 1.000000 | 1.000000 | +0.000000 | 1.000000 | 1.000000 | 0 | 0 |
| glare | 28 | 138 | 0.981818 | 0.981818 | +0.000000 | 0.964286 | 0.964286 | 1 | 1 |
| occlusion | 26 | 138 | 1.000000 | 1.000000 | +0.000000 | 1.000000 | 1.000000 | 0 | 0 |
| skew | 26 | 138 | 1.000000 | 1.000000 | +0.000000 | 1.000000 | 1.000000 | 0 | 0 |

## Five-line conclusion

1. Extra train-only augmentation did not change glare-subset F1: 0.981818 to 0.981818 (+0.000000); missed glare images changed from 1 to 1.
2. Good-class F1 changed from 0.996390 to 0.996390; unnecessary good-image rejections changed from 0 to 0.
3. Other defect-subset F1 decreases: none observed.
4. Overall test macro F1 changed from 0.996377 to 0.996377; accuracy changed from 0.996377 to 0.996377. Training took 40.97 minutes over 32 epochs.
5. This run does not establish an overall benefit without a trade-off; retain the previous prototype pending further evaluation. Validation and test remain unaugmented because changing evaluation images would change the benchmark. The augmentation choice was informed by earlier test errors, so this reused test set is diagnostic, not a fresh independent holdout. Real-photo evaluation is still needed.

Task 4 evidence: https://github.com/EkaterinaShigarova/document-quality-classification/blob/main/reports/error_analysis.md

Overlay API reference: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
