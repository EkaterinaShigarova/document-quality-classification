# Fine-Tuning Delta


Values come from the frozen and fine-tuned results.json files. Quality metrics use the same test split.

| Metric | Frozen | Fine-tuned | Delta |
|---|---:|---:|---:|
| Accuracy | 0.992754 | 0.996377 | +0.003623 |
| Precision macro | 0.992857 | 0.996403 | +0.003546 |
| Recall macro | 0.992754 | 0.996377 | +0.003623 |
| F1 macro | 0.992753 | 0.996377 | +0.003624 |
| Bad recall | 0.985507 | 0.992754 | +0.007246 |
| False accepts | 2 | 1 | -1 |
| False rejects | 0 | 0 | +0 |
| Latency (ms/image) | 25.22 | 23.47 | -1.75 |
| Size (MB) | 16.31 | 16.31 | +0.00 |
| Parameters | 4008829 | 4008829 | +0 |
| Trainable parameters | 1281 | 2639937 | +2638656 |
| Training time (min) | 60.97 | 41.74 | -19.24 |
| Epochs run | 50 | 33 | -17 |

## Train and validation at the selected checkpoint

| Run | Epoch | Train accuracy | Validation accuracy | Train minus validation | Train loss | Validation loss |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 50 | 0.981651 | 0.981481 | +0.000170 | 0.079825 | 0.074662 |
| Fine-tuned | 28 | 0.996177 | 0.995370 | +0.000807 | 0.011615 | 0.010417 |

## Five-line interpretation

1. Test macro F1 increased by +0.003624, and accuracy changed by +0.36 percentage points.
2. False accepts changed from 2 to 1, and false rejects changed from 0 to 0.
3. At the selected checkpoints, the train-minus-validation accuracy gap was 0.017 percentage points for frozen and 0.081 percentage points for fine-tuned. Training and validation losses generally decreased together, without a large sustained divergence. Validation loss fluctuated above its epoch-28 minimum for five more epochs; early stopping ended training at epoch 33 and restored epoch 28.
4. The architecture, total parameter count and recorded float32 checkpoint size stayed the same. Measured latency decreased by 1.75 ms/image, but a single timing run cannot establish a reliable speed advantage. Total fine-tuning time was shorter because early stopping ran 33 rather than 50 epochs; this does not imply cheaper training per epoch.
5. Both versions pass the original acceptance limits. Under the original F1-within-0.01 latency tie-break, the recorded results select the fine-tuned version for the CPU prototype. The accuracy improvement is only one test image out of 276. The remaining error is bad_inn_glare_182_v2.jpg, incorrectly accepted as good; bad_inn_glare_242_v2.jpg is now classified correctly. Real-photo performance remains untested.

## Protocol

I loaded the best frozen checkpoint, then unfroze the last 30 parameterized leaf modules of the feature extractor plus the existing classification head. Containers and layers without parameters are excluded from the count; results.json lists the exact names. Earlier parameters stay frozen. BatchNorm running statistics and base stochastic-depth mode remain fixed as in the frozen run; gradients still update the selected weights and BatchNorm affine parameters. The classifier dropout remains active in training.

Adam learning rate is 1e-5, which is 100 times lower than this frozen run's head learning rate. The handout's factor of 10 is inconsistent with its 1e-3 to 1e-5 values. Smaller updates reduce the risk of damaging useful pretrained features.

The seed, split, 224 x 224 preprocessing, batch size 32, 50-epoch maximum and patience 5 are unchanged. Minimum validation loss selects the checkpoint; test data is evaluated after training. Gaps use the training and validation logs at their respective best epochs, with classifier dropout active only for training.

Frozen results SHA256: df5bf32b5830d064d2d17d2bca3c1fb61bebdedba3e68fa9c239a09cb468a7a9. Frozen checkpoint SHA256: 08127a6a064d870075f5bd9ee132759d8ae8377b3adae45ff89ee58cc8220a2c.

Implementation reference: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
